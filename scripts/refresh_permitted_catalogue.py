#!/usr/bin/env python3
"""WPC-HARVEST-IMPL (USFED-WPC-1112): refresh config/permitted_catalogue.json, the vendored catalogue snapshot of the
ORIGINAL_HTML route (config/permitted_originals.json). Plan: work/USFED-ROUTE-1111/PLAN.md A2.2 and section 4, renamed to
the generic catalogue by ALLDOMAINS-1111 WP-G. THE ONLY WRITER of the snapshot: vera/original_html.py only reads it.

Two separate phases (robots-first, owner rule 23: a separate, earlier step per host):

  1. python3 -B -I scripts/refresh_permitted_catalogue.py --robots-only --row ID --robots-out PATH
       GET https://<host>/robots.txt only (through vera.public_fetch.robots_verdict); prints the verdict for the row's
       discovery URL and for every path stem/prefix, and the Crawl-delay; writes a small JSON evidence file to PATH
       (which must resolve under the project's scratch/ directory); exits.
  2. python3 -B -I scripts/refresh_permitted_catalogue.py --harvest --row ID --robots-evidence PATH [--write]
                                                           [--save-pages DIR]
       Refuses unless the evidence file exists, is for this row and host, is under 60 minutes old and says allowed. Then
       fetches the row's discovery page only (index_url, or sitemap_url with a child-sitemap cap), then one GET per
       catalogue page to record its fetched <title>/h1 and to learn the per-host boilerplate line-hash set (plan step 6c).
       Every request goes through vera.public_fetch's guarded GET: SSRF guard, pinned transport, robots verdict re-evaluated
       for every URL and every redirect hop, byte caps, text/html only. Default is a dry run (counts only, never page
       text); --write merges this host's result into the snapshot.

Stop rules (no retry, no URL variant): robots disallow/unreadable -> stop the host and record it; 401/403 -> the canonical
URL goes to `denied` and the host stops; 429, 5xx, timeout, network error -> the host stops and the cause is recorded.
A URL already in the snapshot's `denied` list is never requested. Pacing: robots.txt is read on its own first, then every
request and every redirect hop waits max(row crawl_delay_s, robots Crawl-delay) since the previous response (hooked in the
per-hop gate). DIVERGENCE: the row's `rate_portal` (vera.rate_limit) is NOT wired in; the crawl delay is the only pacing, and
this script is a manual, single-host, single-threaded run, not a runtime route. The network is touched only by a --robots-only or
--harvest run; imports, tests and --help do not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vera import html_main, original_html as oh, public_fetch as pf  # noqa: E402

SCRIPT_VERSION = "refresh_permitted_catalogue/1 (WPC-HARVEST-IMPL, USFED-WPC-1112)"
EVIDENCE_MAX_AGE = timedelta(minutes=60)
# UNCALIBRATED PRIORS from PLAN.md section 4 (Harvest bounds); a config block "harvest" in permitted_originals.json overrides.
DEFAULT_CAPS = {"max_child_sitemaps": 10, "max_urls_per_host": 300, "max_title_fetches_per_run": 400}
HTML_TYPES = ("text/html",)
SITEMAP_TYPES = ("application/xml", "text/xml")
INDEX_TYPES = ("text/html", "application/xhtml+xml")
_CTRL = re.compile(r"[\x00-\x1f\x7f]")
DOC = ("Vendored catalogue snapshot for the ORIGINAL_HTML route (config/permitted_originals.json), written ONLY by "
       "scripts/refresh_permitted_catalogue.py (never at runtime, never by hand). entries: harvested page URL, host, fetched "
       "page title, slug terms. denied: URLs that answered 401/403, never requested again. boilerplate: per host, sha256 of "
       "normalised text lines on at least max(floor, half) of the host's harvested pages (plan step 6c; headings never "
       "included). robots_status: per host robots read result, Crawl-delay and harvest outcome. sha256 covers entries, "
       "denied and boilerplate. Recorded divergence from feedback_no_hardcoded_data: work/USFED-ROUTE-1111/PLAN.md pass 2 "
       "ruling; later DB home: table permitted_catalogue (owner-gated DDL).")


class Stop(Exception):
    """The host's harvest stops. `kind` is a short fixed code; the message never carries page text."""

    def __init__(self, kind: str, code: str = ""):
        super().__init__(kind)
        self.kind, self.code = kind, code


class Drop(Exception):
    """This one URL is dropped (no entry); the harvest of the host continues."""


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def scratch_root() -> Path:
    """The project's scratch/ directory: the nearest ancestor of this checkout named `scratch` (a worktree lives under
    <root>/scratch/<task>/wt), else <checkout>/scratch."""
    for p in [ROOT, *ROOT.parents]:
        if p.name == "scratch":
            return p
    return ROOT / "scratch"


def under_scratch(path: str) -> Path:
    """Absolute path under scratch/, no symlink anywhere on it; raises SystemExit otherwise."""
    raw = Path(os.path.abspath(path))
    real = Path(os.path.realpath(path))
    root = scratch_root().resolve()
    if raw != real or (real != root and root not in real.parents):
        raise SystemExit(f"FAIL: {path!r} must be a real (no symlink) path under {root}")
    return real


def caps(cfg: dict) -> dict:
    out = dict(DEFAULT_CAPS)
    h = cfg.get("harvest")
    if isinstance(h, dict):
        for k in DEFAULT_CAPS:
            v = h.get(k)
            if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
                out[k] = v
    return out


def slug_terms(row: dict, canon: str) -> list:
    """Terms of the URL slug: the segment after a matching path stem (DOE names pages stem+slug in one segment), else the
    last path segment. Lower-case [a-z0-9] tokens of 1..40 characters, de-duplicated, at most 30."""
    path = urlsplit(canon).path
    seg = path.rsplit("/", 1)[-1]
    for s in row.get("path_stems") or ():
        if path.startswith(s) and len(path) > len(s):
            seg = path[len(s):]
            break
    out: list = []
    for t in re.split(r"[^a-z0-9]+", seg.lower()):
        if t and len(t) <= 40 and t not in out:
            out.append(t)
    return out[:30]


def page_title(raw: bytes, xcfg: dict) -> str:
    """The fetched page <title>, else its first h1; whitespace collapsed, control characters removed, at most 200
    characters. "" when neither exists."""
    doc = html_main.parse(raw, xcfg.get("max_depth", 200), xcfg.get("max_nodes", 400000))
    t = html_main._title(doc)
    if not t:
        for n in html_main.walk(doc):
            if n.tag == "h1":
                t = html_main._text(n)
                break
    t = re.sub(r"\s+", " ", _CTRL.sub(" ", t or "")).strip()
    return t[:200].strip()


class Harvester:
    """One host's run. All network access is `self.get`, which is public_fetch._get with robots_verdict as the robots check."""

    def __init__(self, cfg, row, *, resolve=None, transport=None, clock=time.monotonic, sleep=time.sleep,
                 now=lambda: datetime.now(timezone.utc), denied=(), save_dir=None):
        self.cfg, self.row, self.resolve, self.clock, self.sleep, self.now = cfg, row, resolve, clock, sleep, now
        self.host = row["host"]
        self.lim = cfg.get("limits") or {}
        self.xcfg = cfg.get("extract") or {}
        self.pcfg = dict(pf.load_config())
        self.cache: dict = {}
        self.base_tr = transport or pf.default_transport
        self.last_status = None
        self.last_t = None
        self.requests = 0
        self.denied_known = set(denied)
        self.denied_new: list = []
        self.save_dir = save_dir
        self.delay_cfg = float(row.get("crawl_delay_s") or 0)
        self.last_drop = ""
        disc = row.get("discovery") or {}
        self.disc_paths = {urlsplit(c).path for c in (oh.normalise_url(disc.get("index_url")),
                                                      oh.normalise_url(disc.get("sitemap_url"))) if c}

    # ---- transport wrapper: remembers the status code of the last response (the guarded GET hides it)
    def _tr(self, *a):
        r = self.base_tr(*a)
        self.last_status = getattr(r, "status_code", None)
        self.last_t = self.clock()  # every response (redirect hops and robots.txt included) counts for pacing
        return r

    def robots_read(self) -> str:
        v = self.cache.get((self.host, 443))
        if v is None:
            return "not_read"
        if v is True:
            return "absent_404_410"
        if v == pf._UNREADABLE:
            return "unreadable"
        return "parsed_200"

    def crawl_delay(self) -> float:
        return max(self.delay_cfg, pf.crawl_delay_for(self.cache, self.host, 443, self.pcfg))

    def wait(self) -> None:
        if self.last_t is None:
            return
        need = self.last_t + self.crawl_delay() - self.clock()
        if need > 0:
            self.sleep(need)

    def gate(self, kind: str):
        row, host = self.row, self.host

        def g(u):
            c = oh.normalise_url(u)
            if c is None:
                return "hop_url"
            sp = urlsplit(c)
            if sp.hostname != host:
                return "hop_host"
            if sp.path == "/robots.txt":
                return None  # robots.txt is exempt from the crawl delay (as in public_fetch)
            ok = sp.path in self.disc_paths if kind == "disc" else oh.path_ok(row, sp.path)
            if not ok:
                return "hop_path"
            self.wait()  # pacing before EVERY hop, redirect hops included (the guarded GET calls this before each request)
            return None
        return g

    def get(self, url: str, kind: str, ctypes: tuple, cap: int):
        """(body, final_url). Raises pf.FetchRefused, Stop or Drop. One attempt only."""
        canon = oh.normalise_url(url)
        if canon is None or urlsplit(canon).hostname != self.host:
            raise Drop()
        if canon in self.denied_known:
            raise Drop()  # a denied URL is never requested
        self.last_status = None
        self.last_drop = ""
        gate = self.gate(kind)
        end = self.clock() + float(self.lim.get("fetch_timeout_s", 20))
        try:
            body, final, _st, _ct = pf._get(
                canon, self.pcfg, self.resolve, self._tr, self.clock, end, ctypes,
                lambda u, h, p: pf.robots_verdict(u, h, p, self.pcfg, self.resolve, self._tr, self.clock, end,
                                                  self.cache, gate),
                cap, cap, hop_gate=gate)
        except pf.FetchRefused as e:
            raise self.classify(e, canon) from None
        finally:
            self.last_t = self.clock()
            self.requests += 1
        return body, final

    def classify(self, e, canon: str):
        reason, st = e.reason, self.last_status
        if reason == "access_denied" or (reason == "status" and st in (401, 403)):
            code = str(st) if st in (401, 403) else "access_denied"
            self.denied_known.add(canon)
            self.denied_new.append({"url": canon, "code": code, "seen_at": iso(self.now())})
            return Stop("denied", code)
        if reason == "robots_unreadable":
            return Stop("robots_unreadable")
        if reason == "robots":
            self.last_drop = "robots"
            return Drop()  # a rule that disallows this URL: not requested, no entry (counted by the caller)
        if reason == "status":
            if st in (404, 410):
                return Drop()
            if st == 429 or (isinstance(st, int) and st >= 500):
                return Stop("status", str(st))
            return Drop()
        if reason in ("timeout", "unreachable", "throttled", "busy", "dns", "read_error", "error", "rate_limited",
                      "crawl_delay"):
            return Stop(reason)
        return Drop()  # format_unsupported, too_large, redirects, hop refusals ...: this URL only

    def robots_first(self) -> None:
        """Read robots.txt on its own (before any other request, so the first discovery GET already waits the Crawl-delay);
        anything but an allow verdict for the discovery URL stops the host."""
        disc = self.row.get("discovery") or {}
        durl = oh.normalise_url(disc.get("index_url") or disc.get("sitemap_url"))
        if durl is None:
            raise Stop("no_discovery")
        end = self.clock() + float(self.lim.get("fetch_timeout_s", 20))
        v = pf.robots_verdict(durl, self.host, 443, self.pcfg, self.resolve, self._tr, self.clock, end, self.cache,
                              self.gate("disc"))
        self.last_t = self.clock()
        if v == "unreadable":
            raise Stop("robots_unreadable")
        if v != "allow":
            raise Stop("robots_disallowed")

    # ---- discovery
    def discover(self) -> list:
        disc = self.row.get("discovery") or {}
        cap = int(self.lim.get("max_sitemap_bytes", 5000000))
        urls: list = []
        try:
            if disc.get("index_url"):
                body, final = self.get(disc["index_url"], "disc", INDEX_TYPES, int(self.lim.get("max_html_bytes", 2000000)))
                urls = oh.parse_index_links(body, final, self.row, self.cfg)
            elif disc.get("sitemap_url"):
                urls = self.sitemaps(disc["sitemap_url"], cap)
            else:
                raise Stop("no_discovery")
        except Drop:
            raise Stop("robots_disallowed" if self.last_drop == "robots" else "discovery_unusable") from None
        except (html_main.HtmlRefused, oh.SitemapRefused) as e:
            raise Stop("discovery_refused", getattr(e, "args", [""])[0] if isinstance(e, oh.SitemapRefused) else "html") from None
        return urls

    def sitemaps(self, root_url: str, cap: int) -> list:
        max_child = caps(self.cfg)["max_child_sitemaps"]
        queue, seen, urls, fetched_children = [root_url], set(), [], 0
        while queue:
            u = queue.pop(0)
            c = oh.normalise_url(u)
            if c is None or c in seen:
                continue
            seen.add(c)
            if u != root_url:
                if fetched_children >= max_child:
                    break
                fetched_children += 1
            try:
                body, _final = self.get(c, "disc", SITEMAP_TYPES, cap)
                got = oh.parse_sitemap(body, self.row, self.cfg)
            except Drop:
                if u == root_url:
                    raise
                continue  # one unusable child sitemap
            except oh.SitemapRefused:
                if u == root_url:
                    raise
                continue
            for x in got["urls"]:
                if x not in urls:
                    urls.append(x)
            for ch in got["children"]:
                self.disc_paths.add(urlsplit(ch).path)  # only child sitemaps named by a parsed sitemap may be requested
            queue.extend(got["children"])
        return urls

    # ---- pages
    def run(self) -> dict:
        res = {"host": self.host, "row": self.row["id"], "entries": [], "boilerplate": [], "stopped": None,
               "complete": False, "counts": {}, "robots": {}}
        cnt = res["counts"] = {"discovered": 0, "fetched": 0, "kept": 0, "dropped": 0, "robots_skipped": 0,
                               "denied": 0}
        hashes_by_page: list = []
        try:
            self.robots_first()
            found = [u for u in self.discover() if u not in self.denied_known]
            cap = min(caps(self.cfg)["max_urls_per_host"], caps(self.cfg)["max_title_fetches_per_run"])
            cnt["discovered"] = len(found)
            for u in found[:cap]:
                try:
                    body, final = self.get(u, "page", HTML_TYPES, int(self.lim.get("max_html_bytes", 2000000)))
                    cnt["fetched"] += 1
                    if oh.normalise_url(final) != u:
                        raise Drop()  # a redirect: the catalogue holds only URLs that answer themselves
                    title = page_title(body, self.xcfg)
                    if not title:
                        raise Drop()
                    hs = html_main.block_hashes(body, self.xcfg, self.host)
                except Drop:
                    cnt["robots_skipped" if self.last_drop == "robots" else "dropped"] += 1
                    continue
                except html_main.HtmlRefused:
                    cnt["dropped"] += 1
                    continue
                if self.save_dir is not None:
                    (self.save_dir / (hashlib.sha256(u.encode()).hexdigest()[:24] + ".html")).write_bytes(body)
                res["entries"].append({"url": u, "host": self.host, "title": title, "slug_terms": slug_terms(self.row, u)})
                hashes_by_page.append(hs)
            res["complete"] = True
        except Stop as s:
            res["stopped"] = {"kind": s.kind, "code": s.code}
        cnt["kept"] = len(res["entries"])
        cnt["denied"] = len(self.denied_new)
        n = len(hashes_by_page)
        floor = int(self.xcfg.get("boilerplate_min_pages_floor", 2))
        need = max(floor, n / 2)
        tally: dict = {}
        for hs in hashes_by_page:
            for h in hs:
                tally[h] = tally.get(h, 0) + 1
        res["boilerplate"] = sorted(h for h, c in tally.items() if c >= need)
        res["denied"] = list(self.denied_new)
        res["robots"] = {"read": self.robots_read(), "crawl_delay_s": self.crawl_delay()}
        res["requests"] = self.requests
        return res


# ------------------------------------------------------------------------------------------------ merge and write
def snapshot_path(cfg: dict) -> Path:
    rel = str((cfg.get("snapshot") or {}).get("path") or "")
    if not rel:
        raise SystemExit("FAIL: config has no snapshot.path")
    p = Path(rel) if Path(rel).is_absolute() else oh._ROOT / rel
    cdir = (oh._ROOT / "config").resolve()
    if cdir not in p.resolve().parents:
        raise SystemExit("FAIL: snapshot path must sit inside <root>/config")
    return p


def load_old(p: Path) -> dict:
    empty = {"version": 1, "harvested_at": "", "script_version": "none", "robots_status": {}, "entries": [], "denied": [],
             "boilerplate": {}}
    if not p.exists():
        return empty
    try:
        old = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SystemExit("FAIL: existing snapshot is unreadable; refusing to overwrite") from None
    if not isinstance(old, dict) or old.get("version") != 1 or not isinstance(old.get("entries"), list) \
            or not isinstance(old.get("denied"), list) or not isinstance(old.get("boilerplate"), dict) \
            or not isinstance(old.get("robots_status"), dict):
        raise SystemExit("FAIL: existing snapshot has an unexpected shape; refusing to overwrite")
    return old


def merge(old: dict, res: dict, now: datetime) -> dict:
    """Replace only this host's entries, boilerplate and robots_status. A host that stopped keeps its earlier entries
    (minus any URL now denied) unless the stop was a robots failure, which removes them (fail closed)."""
    host, stamp = res["host"], iso(now)
    denied = [d for d in old.get("denied") or () if isinstance(d, dict)]
    seen = {d["url"] for d in denied}
    for d in res["denied"]:
        if d["url"] not in seen:
            denied.append(d)
            seen.add(d["url"])
    entries = [e for e in old.get("entries") or () if e.get("host") != host]
    boil = {h: v for h, v in (old.get("boilerplate") or {}).items() if h != host}
    rs = dict(old.get("robots_status") or {})
    prev = rs.get(host) if isinstance(rs.get(host), dict) else {}
    st = res["stopped"]
    rec = {"robots": res["robots"]["read"], "crawl_delay_s": res["robots"]["crawl_delay_s"], "checked_at": stamp,
           "counts": res["counts"]}
    if st is None and res["complete"]:
        entries += res["entries"]
        boil[host] = res["boilerplate"]
        rec.update({"harvest": "complete", "harvested_at": stamp})
    else:
        rec.update({"harvest": f"stopped:{st['kind']}" + (f":{st['code']}" if st and st["code"] else "")})
        if st and st["kind"] in ("robots_unreadable", "robots_disallowed"):
            pass  # fail closed: this host's earlier entries are not carried over
        else:
            entries += [e for e in old.get("entries") or () if e.get("host") == host and e["url"] not in seen]
            if host in (old.get("boilerplate") or {}):
                boil[host] = old["boilerplate"][host]
            if prev.get("harvested_at"):
                rec["harvested_at"] = prev["harvested_at"]
    rs[host] = rec
    hosts = {e["host"] for e in entries}
    stamps = [(rs.get(h) or {}).get("harvested_at") or old.get("harvested_at") or stamp for h in hosts]
    body = {"_doc": DOC, "version": 1, "harvested_at": min(stamps) if stamps else stamp, "script_version": SCRIPT_VERSION,
            "robots_status": rs, "entries": sorted(entries, key=lambda e: e["url"]), "denied": denied, "boilerplate": boil}
    body["sha256"] = oh.snapshot_digest(body)
    return body


def write_snapshot(p: Path, body: dict) -> None:
    tmp = p.with_name("." + p.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    json.loads(p.read_text(encoding="utf-8"))  # fail loudly on a malformed write


# ------------------------------------------------------------------------------------------------ phases
def find_row(cfg: dict, rid: str) -> dict:
    for r in oh.rows(cfg):
        if r["id"] == rid:
            return r
    raise SystemExit(f"FAIL: no valid enabled row {rid!r} in config/permitted_originals.json")


def robots_phase(cfg, row, out_path, *, resolve=None, transport=None, clock=time.monotonic, now=None) -> int:
    now = now or (lambda: datetime.now(timezone.utc))
    h = Harvester(cfg, row, resolve=resolve, transport=transport, clock=clock, now=now)
    disc = row.get("discovery") or {}
    durl = oh.normalise_url(disc.get("index_url") or disc.get("sitemap_url"))
    if durl is None:
        raise SystemExit("FAIL: the row's discovery URL does not normalise")
    end = clock() + float(h.lim.get("fetch_timeout_s", 20))
    paths = list(row.get("path_prefixes") or []) + list(row.get("path_stems") or [])
    verdicts = {}
    targets = [("discovery", durl)] + [(p, f"https://{row['host']}{p}") for p in paths]
    gate = h.gate("disc")  # a cross-host robots.txt redirect is refused (fails closed)
    for label, u in targets:
        verdicts[label] = pf.robots_verdict(u, row["host"], 443, h.pcfg, resolve, h._tr, clock, end, h.cache, gate)
        if verdicts[label] == "unreadable":  # robots.txt is never requested a second time
            for lb, _u in targets:
                verdicts.setdefault(lb, "not_checked")
            break
    read = "unreadable" if "unreadable" in verdicts.values() else h.robots_read()
    delay = h.crawl_delay()
    allowed = read in ("parsed_200", "absent_404_410") and verdicts["discovery"] == "allow"
    ev = {"version": 1, "row": row["id"], "host": row["host"], "checked_at": iso(now()), "read": read,
          "crawl_delay_s": delay, "verdicts": verdicts, "allowed": allowed}
    print(f"robots {row['host']}: read={read} crawl_delay={delay:g}s allowed={allowed}")
    for label, v in verdicts.items():
        print(f"  {label}: {v}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ev, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0 if allowed else 2


def check_evidence(path: str, row: dict, now: datetime) -> None:
    p = Path(path)
    try:
        ev = json.loads(p.read_text(encoding="utf-8"))
        at = datetime.fromisoformat(str(ev["checked_at"]))
    except (OSError, ValueError, KeyError, TypeError):
        raise SystemExit("FAIL: robots evidence missing or unreadable; run --robots-only first") from None
    if not isinstance(ev, dict) or ev.get("host") != row["host"] or ev.get("row") != row["id"]:
        raise SystemExit("FAIL: robots evidence is for another row or host")
    if at.tzinfo is None or at > now + timedelta(minutes=1) or now - at > EVIDENCE_MAX_AGE:
        raise SystemExit("FAIL: robots evidence is stale (older than 60 minutes) or dated in the future")
    if ev.get("allowed") is not True:
        raise SystemExit("FAIL: robots evidence does not say allowed")


def harvest_phase(cfg, row, evidence, *, write, save_pages, resolve=None, transport=None, clock=time.monotonic,
                  sleep=time.sleep, now=None) -> int:
    now = now or (lambda: datetime.now(timezone.utc))
    check_evidence(evidence, row, now())
    sp = snapshot_path(cfg)
    old = load_old(sp)
    save_dir = None
    if save_pages:
        save_dir = under_scratch(save_pages)
        save_dir.mkdir(parents=True, exist_ok=True)
    h = Harvester(cfg, row, resolve=resolve, transport=transport, clock=clock, sleep=sleep, now=now,
                  denied={d.get("url") for d in old.get("denied") or () if isinstance(d, dict)}, save_dir=save_dir)
    res = h.run()
    if res["stopped"] and res["stopped"]["kind"] == "robots_unreadable":
        res["robots"]["read"] = "unreadable"
    c = res["counts"]
    st = res["stopped"]
    print(f"harvest {row['id']} ({row['host']}): robots={res['robots']['read']} crawl_delay={res['robots']['crawl_delay_s']:g}s "
          f"requests={res['requests']} discovered={c['discovered']} fetched={c['fetched']} kept={c['kept']} "
          f"dropped={c['dropped']} denied={c['denied']} boilerplate_lines={len(res['boilerplate'])} "
          f"stopped={'no' if st is None else st['kind'] + (':' + st['code'] if st['code'] else '')}")
    body = merge(old, res, now())
    if write:
        if not oh._snapshot_valid(body, cfg):
            raise SystemExit("FAIL: the merged snapshot does not validate (oh._snapshot_valid); nothing written")
        write_snapshot(sp, body)
        print(f"wrote {sp.relative_to(oh._ROOT)} (entries {len(body['entries'])}, denied {len(body['denied'])})")
    else:
        print("dry run: nothing written (use --write)")
    return 0 if st is None else 2


def main(argv=None, **inject) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--row", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--robots-only", action="store_true")
    g.add_argument("--harvest", action="store_true")
    ap.add_argument("--robots-out")
    ap.add_argument("--robots-evidence")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--save-pages")
    a = ap.parse_args(argv)
    cfg = oh.load_config()
    row = find_row(cfg, a.row)
    if a.robots_only:
        if not a.robots_out or a.harvest or a.write or a.save_pages:
            raise SystemExit("FAIL: --robots-only takes --row and --robots-out only")
        return robots_phase(cfg, row, under_scratch(a.robots_out), **inject)
    if not a.robots_evidence or a.robots_out:
        raise SystemExit("FAIL: --harvest needs --robots-evidence (from an earlier --robots-only run)")
    return harvest_phase(cfg, row, a.robots_evidence, write=a.write, save_pages=a.save_pages, **inject)


if __name__ == "__main__":
    raise SystemExit(main())
