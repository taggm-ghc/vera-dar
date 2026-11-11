"""CHUNK-VALUE-1110 WP1: pure chunk-value and known-bad-document rules (no I/O, no DB, no network, no env).

One shared module for the ingest filter (later, fulldoc_db.build_rows) and the known-bad audit. Every threshold, heading
list and word list is read from the config dict (config/chunk_value.json); the module holds only rule-id names.

Chunk level: evaluate_chunks(text, sections, cfg, chunking) -> keep/drop per live 400-character chunk with reason codes
and the dropped character ranges (offsets plus sha256 of the slice, never text). There is deliberately NO symbol or
alpha-ratio rule at chunk level (equations are evidence).
Document level: document_flags(doc, cfg, words, chunk_result) and find_duplicates(docs, cfg)."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import unicodedata
from urllib.parse import unquote, urlparse, urlsplit

# rule ids (the only named constants)
C1, C2, C3, C4, C5, C6 = ("C1_section_kind", "C2_heading", "C3_boilerplate", "C4_reference_list", "C5_low_info",
                          "C6_near_dup")
D1, D2, D3, D4 = "D1_arxiv", "D2_incomprehensible", "D3_incomplete", "D4_duplicate"
D1P = "D1_portal_sourced"  # REPROV-W3-1110: stored row comes from a portal host or record (port of fulltext_loader.portal_verdict)
CAP = "cap_exceeded"
MASK_CHAR = " "


def config_sha256(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ chunker
def chunk_spans(text: str, chunking: dict) -> list:
    """Same algorithm as DarCorpus._chunk_spans at c3aabb1; chunking = {chunk_chars, chunk_overlap, min_chunk_chars}."""
    size, ov, mn = int(chunking["chunk_chars"]), int(chunking["chunk_overlap"]), int(chunking["min_chunk_chars"])
    n, spans, pos = len(text), [], 0
    rx = re.compile(r"\S")
    while True:
        m = rx.search(text, pos)
        if not m:
            break
        start = m.start()
        end = n if start + size >= n else None
        if end is None:
            end = start + size
            ws = max((i for i in range(start + 1, end + 1) if text[i].isspace()), default=None)
            if ws is not None:
                end = ws
        reached = end >= n
        while end > start + 1 and text[end - 1].isspace():
            end -= 1
        spans.append((start, end))
        if reached:
            break
        nxt = max(end - ov, start + 1)
        while nxt < n and not text[nxt - 1].isspace():
            nxt += 1
        if nxt >= end:
            nxt = end
        pos = nxt
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < mn:
        last = spans.pop()
        spans[-1] = (spans[-1][0], last[1])
    return [(s, e) for s, e in spans if text[s:e].strip()]


# ------------------------------------------------------------------------------------------------ helpers
def _norm_heading(title, ccfg: dict) -> str:
    t = re.sub(ccfg["section_number_regex"], "", str(title or ""), count=1)
    t = re.sub(r"\s+", " ", t.casefold()).strip()
    return re.sub(r"[\W_]+$", "", t).strip()


def _valid_sections(sections, n: int) -> list:
    out = []
    for s in sections or ():
        try:
            if isinstance(s, dict) and 0 <= int(s["start"]) < int(s["end"]) <= n:
                out.append({"kind": s.get("kind"), "title": s.get("title") or "", "level": int(s.get("level") or 0),
                            "start": int(s["start"]), "end": int(s["end"])})
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out, key=lambda x: (x["start"], x["level"]))


def _merge(rs, gap: int = 0) -> list:
    out: list = []
    for a, b in sorted((int(a), int(b)) for a, b in rs):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _inside(ranges: list, a: int, b: int, gap: int) -> bool:
    return any(x <= a and b <= y + gap for x, y in ranges)


def _content_tokens(text: str, ccfg: dict, stop: set) -> list:
    rx = re.compile(ccfg["token_regex"])
    mn = int(ccfg["min_token_chars"])
    return [w for w in rx.findall(text.casefold()) if w not in stop and len(w) >= mn]


def _heading_ranges(secs: list, ccfg: dict) -> list:
    """C2 ranges: a section whose normalised heading equals a listed heading (exact), plus nested deeper sections of the
    same kind. back_only_headings apply to kind 'back' only."""
    h = ccfg["C2_heading"]
    kinds = set(h.get("heading_kinds") or ())
    any_h = {_norm_heading(x, ccfg) for x in h.get("drop_headings") or ()}
    back_h = {_norm_heading(x, ccfg) for x in h.get("back_only_headings") or ()}
    out = []
    for i, m in enumerate(secs):
        if m["kind"] not in kinds:
            continue
        nh = _norm_heading(m["title"], ccfg)
        if not nh or not (nh in any_h or (m["kind"] == "back" and nh in back_h)):
            continue
        end = m["end"]
        for s in secs[i + 1:]:
            if s["kind"] == m["kind"] and s["level"] > m["level"]:
                end = max(end, s["end"])
            else:
                break
        out.append((m["start"], end))
    return out


def _boilerplate_flags(text: str, ccfg: dict) -> list:
    """Per-line spans [(start, end, flagged)] by line features: URL share, a normalised line repeated >= N times."""
    b = ccfg["C3_boilerplate"]
    urlrx = re.compile(b["url_regex"])
    lines, pos = [], 0
    for ln in text.split("\n"):
        lines.append((pos, pos + len(ln), ln))
        pos += len(ln) + 1
    counts: dict = {}
    for _, _, ln in lines:
        k = re.sub(r"\s+", " ", ln.strip().casefold())
        if len(k) >= int(b["repeat_min_line_chars"]):
            counts[k] = counts.get(k, 0) + 1
    out = []
    for s, e, ln in lines:
        st = ln.strip()
        if not st:
            continue
        urlchars = sum(len(m.group(0)) for m in urlrx.finditer(st))
        k = re.sub(r"\s+", " ", st.casefold())
        flagged = (urlchars / len(st) >= float(b["url_share_max"])) or counts.get(k, 0) >= int(b["repeat_min_count"])
        out.append((s, e, flagged))
    return out


def _chunk_boiler(a: int, b: int, lines: list, share: float) -> bool:
    inl = [f for s, e, f in lines if s < b and e > a]
    return bool(inl) and sum(inl) / len(inl) >= share


# ------------------------------------------------------------------------------------------------ chunk rules
def evaluate_chunks(text: str, sections, cfg: dict, chunking: dict, fmt: str | None = None) -> dict:
    """Keep/drop decision per chunk. Returns:
      chunks: [{start, end, keep, rule}]  (rule None when kept; CAP rule is not stored per chunk, see `capped`)
      dropped: merged [{start, end, rule, sha256}] character ranges to mask (chunk spans minus kept-chunk coverage)
      kept: complement of `dropped` in [0, len(text)) (tiles with dropped exactly)
      by_rule: {rule: {chunks, chars}} ; substantive_chunks (kept by the rules before the cap)
      capped: True when the rules would drop more than max_evidence_drop_share of evidence-range characters; then the
              document is stored whole (dropped = [])."""
    cc = cfg["chunk"]
    n = len(text)
    gap = int(cc["section_gap_chars"])
    stop = set(cc["function_words"])
    secs = _valid_sections(sections, n)
    spans = chunk_spans(text, chunking)
    c1r = _merge([(s["start"], s["end"]) for s in secs if s["kind"] in set(cc["C1_section_kind"]["drop_section_kinds"])], gap) \
        if cc["C1_section_kind"].get("enabled") else []
    c2r = _merge(_heading_ranges(secs, cc), gap) if cc["C2_heading"].get("enabled") else []
    b3 = cc["C3_boilerplate"]
    lines = _boilerplate_flags(text, cc) if b3.get("enabled") and fmt in set(b3.get("formats") or ()) else None
    ref_rx = re.compile(cc["C4_reference_list"]["line_regex"], re.M)
    shn = int(cc["C6_near_dup"]["shingle_words"])
    seen: list = []  # shingle sets of kept chunks
    inv: dict = {}  # shingle -> [index into seen]
    reasons = []
    for a, b in spans:
        t = text[a:b]
        why = None
        if c1r and _inside(c1r, a, b, gap):
            why = C1
        elif c2r and _inside(c2r, a, b, gap):
            why = C2
        elif cc["C4_reference_list"].get("enabled") and len(ref_rx.findall(t)) >= int(cc["C4_reference_list"]["min_lines"]):
            why = C4
        elif lines is not None and _chunk_boiler(a, b, lines, float(b3["chunk_flagged_line_share"])):
            why = C3
        else:
            toks = _content_tokens(t, cc, stop)
            nsp = [ch for ch in t if not ch.isspace()]
            symdig = (sum(1 for ch in nsp if not ch.isalpha()) / len(nsp)) if nsp else 0.0
            exempt = symdig >= float(cc["C5_low_info"]["exempt_symbol_digit_share_min"])  # math/table chunk: C5 and C6 skip it
            if (cc["C5_low_info"].get("enabled") and not exempt
                    and len(set(toks)) < int(cc["C5_low_info"]["min_distinct_content_tokens"])):
                why = C5
            elif cc["C6_near_dup"].get("enabled") and not exempt:
                sh = {" ".join(toks[i:i + shn]) for i in range(max(len(toks) - shn + 1, 1))} if toks else set()
                if sh:
                    cand = {j for g in sh for j in inv.get(g, ())}
                    jm = float(cc["C6_near_dup"]["jaccard_min"])
                    if any(len(sh & seen[j]) / len(sh | seen[j]) >= jm for j in cand):
                        why = C6
                    else:
                        seen.append(sh)
                        for g in sh:
                            inv.setdefault(g, []).append(len(seen) - 1)
        reasons.append(why)
    rec = {"version": cfg.get("version"), "config_sha256": config_sha256(cfg), "chars": n,
           "chunks_before": len(spans), "capped": False}
    rec["substantive_chunks"] = sum(1 for w in reasons if w is None)
    rec["rule_chunks_pre_cap"] = {w: reasons.count(w) for w in set(reasons) if w}
    cover = bytearray(n)
    for (a, b), w in zip(spans, reasons):
        if w is None:
            cover[a:b] = b"\x01" * (b - a)
    dropped = []
    for (a, b), w in zip(spans, reasons):
        if w is None:
            continue
        i = a
        while i < b:
            if cover[i]:
                i += 1
                continue
            j = i
            while j < b and not cover[j]:
                j += 1
            if dropped and dropped[-1]["rule"] == w and dropped[-1]["end"] >= i:
                dropped[-1]["end"] = max(dropped[-1]["end"], j)
            else:
                dropped.append({"start": i, "end": j, "rule": w})
            i = j
    dropped.sort(key=lambda d: d["start"])
    # safety cap: counted rules inside evidence-kind sections vs evidence-range characters
    ev = _merge([(s["start"], s["end"]) for s in secs if s["kind"] in set(cc["evidence_section_kinds"])], gap)
    ev_chars = sum(y - x for x, y in ev)
    counted = set(cc["cap_counted_rules"])
    lost = 0
    for d in dropped:
        if d["rule"] in counted:
            for x, y in ev:
                lost += max(0, min(d["end"], y) - max(d["start"], x))
    rec["evidence_chars"], rec["evidence_chars_dropped"] = ev_chars, lost
    if ev_chars and lost / ev_chars > float(cc["max_evidence_drop_share"]):
        rec["capped"] = True
        dropped = []
        reasons_final = [None] * len(spans)
    else:
        reasons_final = reasons
    for d in dropped:
        d["sha256"] = _sha(text[d["start"]:d["end"]])
    kept, pos = [], 0
    for d in dropped:
        if d["start"] > pos:
            kept.append([pos, d["start"]])
        pos = d["end"]
    if pos < n:
        kept.append([pos, n])
    by_rule: dict = {}
    for (a, b), w in zip(spans, reasons_final):
        if w:
            r = by_rule.setdefault(w, {"chunks": 0, "chars": 0})
            r["chunks"] += 1
    for d in dropped:
        by_rule.setdefault(d["rule"], {"chunks": 0, "chars": 0})["chars"] += d["end"] - d["start"]
    rec.update({"chunks": [{"start": a, "end": b, "keep": w is None, "rule": w} for (a, b), w in zip(spans, reasons_final)],
                "dropped": dropped, "kept": kept, "by_rule": by_rule,
                "chunks_after": sum(1 for w in reasons_final if w is None),
                "chars_dropped": sum(d["end"] - d["start"] for d in dropped)})
    return rec


def mask_text(text: str, dropped) -> str:
    """The text with each dropped range replaced by MASK_CHAR (same length, offsets unchanged)."""
    out = list(text)
    for d in dropped:
        for i in range(d["start"], d["end"]):
            out[i] = MASK_CHAR
    return "".join(out)


# ------------------------------------------------------------------------------------------------ document rules
def _hosts_hit(urls, hosts: set) -> bool:
    for u in urls:
        try:
            h = (urlparse(str(u)).hostname or "").lower()
        except ValueError:
            continue
        if h in hosts:
            return True
    return False


def d1_arxiv(doc: dict, cfg: dict) -> list:
    c = cfg["document"][D1]
    if not c.get("enabled"):
        return []
    hosts = {h.lower() for h in c["hosts"]}
    urls = list(doc.get("urls") or ()) + [u for u in (doc.get("fulltext_url"), doc.get("orig_url")) if u]
    idents = [str(x).lower() for x in list(doc.get("identifiers") or ()) + [doc.get("doi")] + urls if x]
    repo = str(doc.get("repository") or "").lower()
    if repo and any(x.lower() in repo for x in c.get("repository_substrings", ())):
        return [D1 + ":repository"]
    if _hosts_hit(urls, hosts):
        return [D1 + ":host"]
    if any(sub.lower() in i for sub in c["identifier_substrings"] for i in idents):
        return [D1 + ":identifier"]
    if str(doc.get("resolver") or "").lower() in {r.lower() for r in c["resolvers"]}:
        return [D1 + ":resolver"]
    return []


# ------------------------------------------------------------------------------------------------ D1 portal rule
# REPROV-W3-1110 (review R1): a self-contained port of vera.fulltext_loader.portal_verdict and vera.host_role
# (role_for_url, role_for_record) as at batch6-rc-1110 d3fe0fe, so the audit kit (which runs this file standalone, pinned by
# sha256) selects exactly the rows the serve-time quarantine (CP5) refuses. The host-role table, the preprint DOI prefixes, the
# record flags and types live in cfg["document"]["D1_portal_sourced"]["policy"] (a copy of config/host_roles.json at d3fe0fe,
# re-copied after PVFIX-A-1110 75b0c6a, which narrowed accepted_versions; source_sha256 names the file copied).
# tests/test_chunk_value_portal_parity_1110.py asserts verdict and rule id equal the original row for row. Pure; never raises.
_PR_ROLES = ("ORIGINAL", "REPOSITORY_OF_RECORD", "PORTAL", "RESOLVER", "EXCLUDED")
_PR_RANK = {"EXCLUDED": 0, "PORTAL": 1, "RESOLVER": 2, "REPOSITORY_OF_RECORD": 3, "ORIGINAL": 4}
_PR_STRICT = {"EXCLUDED": 0, "PORTAL": 1, "UNKNOWN": 1.5, "RESOLVER": 2, "REPOSITORY_OF_RECORD": 3, "ORIGINAL": 4}
_PR_DOI_HOSTS = ("doi.org", "dx.doi.org")
_PR_NUMERIC_LABEL = re.compile(r"^(?:0x[0-9a-f]*|[0-9]+)$", re.I)
_PR_SEP = re.compile(r"%(2f|5c|25)", re.I)
_PR_CACHE: dict = {}  # id(policy dict) -> (policy, _PortalReg)


class _PortalReg:
    def __init__(self, cfg):
        self.error = None
        self.hosts, self.patterns, self.doi_rules, self.flags = [], [], [], []
        self.type_portal: set = set()
        self.rv: dict = {}
        try:
            self._build(cfg)
        except Exception as e:  # noqa: BLE001 - any defect means fail closed
            self.error = f"{type(e).__name__}: {e}"

    def _build(self, cfg):
        if not isinstance(cfg, dict):
            raise ValueError("top level must be an object")
        seen = set()
        for i, row in enumerate(cfg.get("hosts") or []):
            m = str(row.get("match", "")).strip().lower().strip(".")
            if not m or "/" in m or " " in m:
                raise ValueError(f"hosts[{i}]: bad match")
            if row.get("role") not in _PR_ROLES or not row.get("kind"):
                raise ValueError(f"hosts[{i}] {m}: bad role or kind")
            pr = row.get("path_regex")
            rx = re.compile(pr, re.I) if pr else None
            if (m, pr) in seen:
                raise ValueError("duplicate match")
            seen.add((m, pr))
            self.hosts.append({**row, "match": m, "_rx": rx, "_spec": (m.count(".") + 1) + (1000 if rx else 0),
                               "_id": f"host:{m}" + (f"#{pr}" if pr else "")})
        for i, row in enumerate(cfg.get("host_patterns") or []):
            if row.get("role") not in _PR_ROLES or not row.get("id"):
                raise ValueError(f"host_patterns[{i}]: bad role or id")
            self.patterns.append({**row, "_rx": re.compile(row["regex"], re.I)})
        for row in cfg.get("doi_prefix_roles") or []:
            if row.get("role") not in _PR_ROLES:
                raise ValueError("doi_prefix_roles: bad role")
            self.doi_rules.append((re.compile(row["regex"], re.I), row["role"], str(row.get("id", "doi"))))
        for row in cfg.get("source_flags") or []:
            if row.get("role") not in _PR_ROLES:
                raise ValueError("source_flags: bad role")
            self.flags.append({**row, "value": str(row["value"]).upper()})
        self.type_portal = {str(x).lower() for x in cfg.get("record_type_portal") or []}
        rv = cfg.get("repository_version") or {}
        self.rv = {k: {str(x).lower() for x in rv.get(k, [])} for k in ("portal_versions", "accepted_versions", "non_article_types")}
        if not self.hosts:
            raise ValueError("no host rows")


def _pr_resolve(path: str, orig: str) -> str:
    out: list = []
    for seg in path.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if out:
                out.pop()
            continue
        out.append(seg)
    return "/" + "/".join(out) + ("/" if orig.endswith("/") and out else "")


def _pr_norm_path(raw: str) -> str:
    path = raw
    for _ in range(3):
        d = unquote(path)
        if d == path:
            break
        path = d
    return _pr_resolve(path.replace("\\", "/"), path)


def _pr_literal_path(raw: str, decode: bool) -> str:
    s = re.sub(r"%2e", ".", raw, flags=re.I)
    if decode:
        s = _PR_SEP.sub(lambda m: "\x00" + m.group(1), s)
        s = unquote(s).replace("\x00", "%")
    return _pr_resolve(s, s)


def _pr_split(url):
    """(host, [path readings], problem). host None when unusable."""
    if not isinstance(url, str) or not url.strip() or len(url) > 4096 or re.search(r"[\x00-\x20\x7f\\]", url.strip()):
        return None, [], "malformed_url"
    u = url.strip()
    if "://" not in u and not u.startswith("//"):
        u = "//" + u
    try:
        p = urlsplit(u)
        if p.scheme and p.scheme.lower() not in ("http", "https"):
            return None, [], "malformed_url"
        host = p.hostname
        p.port  # noqa: B018 - validates the port
    except Exception:  # noqa: BLE001
        return None, [], "malformed_url"
    if not host:
        return None, [], "malformed_url"
    host = re.sub("[。．｡]", ".", host.lower()).rstrip(".")
    if not host or "." not in host or ".." in host or "%" in host:
        return None, [], "malformed_url"
    try:
        host = host.encode("idna").decode("ascii").lower().rstrip(".")
    except Exception:  # noqa: BLE001
        return None, [], "malformed_url"
    if not host or "." not in host or ".." in host:
        return None, [], "malformed_url"
    try:
        ipaddress.ip_address(host)
        return None, [], "ip_literal"
    except ValueError:
        pass
    if _PR_NUMERIC_LABEL.match(host.rsplit(".", 1)[-1]) or all(_PR_NUMERIC_LABEL.match(x) for x in host.split(".")):
        return None, [], "ip_literal"
    raw = p.path or "/"
    paths: list = []
    for q in (_pr_norm_path(raw), _pr_literal_path(raw, False), _pr_literal_path(raw, True)):
        if p.query and q == "/":
            q = "/?" + p.query
        if q not in paths:
            paths.append(q)
    return host, paths, ""


def _pr_match_row(host, path, reg):
    best = None
    for row in reg.hosts:
        if not (host == row["match"] or host.endswith("." + row["match"])):
            continue
        if row["_rx"] is not None and not row["_rx"].search(path):
            continue
        key = (-row["_spec"], -int(row.get("priority", 0)), _PR_RANK[row["role"]])
        if best is None or key < best[0]:
            best = (key, row)
    for row in reg.patterns:
        if row["_rx"].search(host):
            key = (-1, 0, _PR_RANK[row["role"]])
            if best is None or key < best[0]:
                best = (key, row)
    return best[1] if best else None


def _pr_doi_of(text) -> str:
    t = str(text or "").strip()
    t = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", t, flags=re.I)
    return unquote(t).strip()


def _pr_doi_role(doi, reg):
    d = _pr_doi_of(doi)
    if d:
        for rx, role, rid in reg.doi_rules:
            if rx.search(d):
                return (role, f"doi:{rid}")
    return None


def _pr_qualifies(reg, meta) -> bool:
    m = meta if isinstance(meta, dict) else {}
    ver = str(m.get("version") or "").strip().lower()
    typ = str(m.get("type") or m.get("resource_type") or m.get("source_type") or "").strip().lower()
    if typ in reg.type_portal or ver in reg.rv["portal_versions"] and ver:
        return False
    if ver in reg.rv["accepted_versions"]:
        return True
    return not ver and typ in reg.rv["non_article_types"]


def _pr_role_for_path(host, path, reg, meta):
    row = _pr_match_row(host, path, reg)
    if row is None:
        return ("UNKNOWN", "no_row")
    if host in _PR_DOI_HOSTS or (row["role"] == "RESOLVER" and row.get("match") in _PR_DOI_HOSTS):
        hit = _pr_doi_role(path.lstrip("/"), reg)
        if hit:
            return hit
    rid = row.get("_id") or f"pattern:{row.get('id')}"
    if row.get("version_gated") and not _pr_qualifies(reg, meta):
        return ("PORTAL", f"version_unqualified:{rid}")
    return (row["role"], rid)


def _pr_role_for_url(url, reg, meta=None):
    try:
        if reg.error:
            return ("PORTAL", "config_invalid")
        host, paths, problem = _pr_split(url)
        if host is None:
            return ("UNKNOWN", problem)
        best = None
        for path in paths:
            hr = _pr_role_for_path(host, path, reg, meta)
            if best is None or _PR_STRICT.get(hr[0], 1.5) < _PR_STRICT.get(best[0], 1.5):
                best = hr
        return best
    except Exception:  # noqa: BLE001
        return ("PORTAL", "internal_error")


def _pr_role_for_record(meta, reg):
    try:
        if reg.error:
            return ("PORTAL", "config_invalid")
        if not isinstance(meta, dict):
            return ("UNKNOWN", "no_record")
        hit = _pr_doi_role(meta.get("doi") or "", reg)
        if hit:
            return hit
        oa = meta.get("oa") if isinstance(meta.get("oa"), dict) else {}
        for f in reg.flags:
            if f["value"] in (str(meta.get(f["field"]) or "").strip().upper(), str(oa.get(f["field"]) or "").strip().upper()):
                return (f["role"], f"flag:{f.get('id', f['field'])}")
        ptypes = meta.get("pub_types") or meta.get("pubTypeList") or []
        ptypes = ptypes.get("pubType", []) if isinstance(ptypes, dict) else ptypes
        ptypes = [ptypes] if isinstance(ptypes, str) else list(ptypes or [])
        typ = str(meta.get("source_type") or meta.get("type") or "").strip().lower()
        if typ in reg.type_portal or any(str(p).strip().lower() in reg.type_portal for p in ptypes) or meta.get("is_preprint") is True:
            return ("PORTAL", "record_type:preprint")
        for k in ("url", "landing_url", "fulltext_url", "full_text_url"):
            if meta.get(k):
                return _pr_role_for_url(meta[k], reg, meta)
        return ("UNKNOWN", "no_url")
    except Exception:  # noqa: BLE001
        return ("PORTAL", "internal_error")


def _portal_policy(cfg):
    return ((cfg or {}).get("document") or {}).get(D1P) or {}


def _portal_reg(cfg) -> _PortalReg:
    pol = _portal_policy(cfg).get("policy")
    hit = _PR_CACHE.get(id(pol))
    if hit is None or hit[0] is not pol:
        hit = (pol, _PortalReg(pol))
        _PR_CACHE.clear()
        _PR_CACHE[id(pol)] = hit
    return hit[1]


_PR_EU_REPO_PREFIX = "info:eu-repo/semantics/"


def _pr_norm_version(v) -> str:
    """PVFIX-A-1110 port of fulltext_loader._norm_version (incl. S1, a7a4df8): strip only the OpenAIRE/eu-repo prefix (case-insensitive);
    any other string is left as is, so an unknown version ('v1.0') stays undecidable (fail closed). None / '' mean 'no version' ("");
    a non-string, or a bare eu-repo prefix with nothing after it, is undecidable (never 'no version')."""
    if v is None or v == "":
        return ""
    if not isinstance(v, str):
        return "undecidable"
    v = v.strip()
    if v.lower().startswith(_PR_EU_REPO_PREFIX):
        v = v[len(_PR_EU_REPO_PREFIX):].strip()
        if not v:
            return "undecidable"
    return v


def portal_verdict(item, prov=None, canonical_url=None, cfg=None):
    """The host_role rule id when this stored row (item + provenance + canonical URL) is a PORTAL / EXCLUDED record, else
    None; same decision and rule id as vera.fulltext_loader.portal_verdict at d3fe0fe plus PVFIX-A-1110 (75b0c6a: item version/type
    passed to the URL role checks; accepted_versions = published/VoR only). Never raises (an error counts as a
    portal: "internal_error"; an invalid policy: "config_invalid")."""
    try:
        reg = _portal_reg(cfg or CONFIG or _default_cfg())
        item = item if isinstance(item, dict) else {}
        prov = prov if isinstance(prov, dict) else {}
        acq = prov.get("acquisition") if isinstance(prov.get("acquisition"), dict) else {}
        if isinstance(acq.get("arxiv"), dict) and acq["arxiv"].get("id"):
            urls_extra = [f"https://arxiv.org/abs/{acq['arxiv']['id']}"]
        else:
            urls_extra = []
        rec = {"doi": item.get("doi") or prov.get("doi") or "", "source": item.get("source") or "",
               "source_type": item.get("source_type") or "", "pub_types": item.get("pub_types") or [],
               "is_preprint": item.get("is_preprint")}
        urls = [item.get("url"), item.get("landing_url"), item.get("fulltext_url"), prov.get("url"), prov.get("fulltext_url"),
                (item.get("identifier") or {}).get("url") if isinstance(item.get("identifier"), dict) else None,
                canonical_url, acq.get("abs_url")]
        urls += urls_extra  # acquisition["url_chain"] is deliberately NOT checked (wire-A F1)
        # PVFIX-A-1110 port: the item's own version / type is meta for the URL role checks only (relaxes a version-gated row; rec unchanged)
        vmeta = {"version": _pr_norm_version(item.get("version") if item.get("version") not in (None, "") else prov.get("version")),
                 "type": item.get("type") or prov.get("type") or "",
                 "resource_type": item.get("resource_type") or prov.get("resource_type") or "",
                 "source_type": item.get("source_type") or ""}
        hr = _pr_role_for_record(rec, reg)
        if hr[0] in ("PORTAL", "EXCLUDED"):
            return hr[1]
        for u in urls:
            if isinstance(u, str) and u.lower().startswith(("http://", "https://")):
                hr = _pr_role_for_url(u, reg, vmeta)
                if hr[0] in ("PORTAL", "EXCLUDED"):
                    return hr[1]
        return None
    except Exception:  # noqa: BLE001
        return "internal_error"


def portal_rule_ids(cfg: dict) -> set:
    """Every rule id portal_verdict can return for this policy (for sanitising audit output by membership, never by pattern)."""
    reg = _portal_reg(cfg)
    ids = {"internal_error", "config_invalid", "record_type:preprint"}
    ids |= {r["_id"] for r in reg.hosts} | {f"version_unqualified:{r['_id']}" for r in reg.hosts}
    ids |= {f"pattern:{r.get('id')}" for r in reg.patterns} | {f"version_unqualified:pattern:{r.get('id')}" for r in reg.patterns}
    ids |= {f"doi:{rid}" for _, _, rid in reg.doi_rules}
    ids |= {f"flag:{f.get('id', f['field'])}" for f in reg.flags}
    return ids


def normalise_extraction(text: str) -> str:
    """Repair common extraction artefacts before the dictionary and non-alnum metrics: soft hyphens removed, ligatures
    and compatibility forms folded (NFKC), line-break hyphens joined ("compre-\\nhension" -> "comprehension")."""
    t = unicodedata.normalize("NFKC", text.replace("\u00ad", ""))
    return re.sub(r"(?<=[^\W\d_])-[ \t]*\n[ \t]*(?=[^\W\d_])", "", t)


def d2_metrics(text: str, cfg: dict, words=None) -> dict:
    """Garbled-extraction metrics at document level (pure numbers; thresholds are applied in d2_incomprehensible)."""
    c = cfg["document"][D2]
    n = max(len(text), 1)
    raw = text
    text = normalise_extraction(text)  # dictionary, language and non-alnum metrics see repaired text
    m: dict = {}
    nonws = [ch for ch in text if not ch.isspace()]
    m["non_alnum_share"] = (sum(1 for ch in nonws if not ch.isalnum()) / len(nonws)) if nonws else 1.0
    m["bad_chars_per_1000"] = 1000 * len(re.findall(c["bad_char_regex"], raw)) / n
    m["cid_per_1000"] = 1000 * len(re.findall(c["cid_regex"], raw)) / n
    m["hyphen_break_per_1000"] = 1000 * len(re.findall(c["hyphen_break_regex"], raw)) / n
    lines = [ln for ln in raw.split("\n") if ln.strip()]
    m["n_lines"] = len(lines)
    m["mean_line_chars"] = (sum(len(ln) for ln in lines) / len(lines)) if lines else 0.0
    cnt: dict = {}
    for ln in lines:
        k = re.sub(r"\s+", " ", ln.strip().casefold())
        if len(k) >= int(c["repeat_line_min_chars"]):
            cnt[k] = cnt.get(k, 0) + 1
    rep = sum(v for v in cnt.values() if v >= int(c["repeat_line_min_count"]))
    m["repeat_line_share"] = rep / len(lines) if lines else 0.0
    toks = re.findall(c["token_regex"], text)
    m["n_alpha_tokens"] = len(toks)
    allw = re.findall(r"[^\W\d_]+", text.casefold())
    mk = {w.casefold() for w in c["english_marker_words"]}
    m["english_marker_share"] = (sum(1 for w in allw if w in mk) / len(allw)) if allw else 0.0
    m["is_english"] = int(m["english_marker_share"] >= float(c["english_marker_share_min"]))
    m["math_heavy"] = int(m["non_alnum_share"] >= float(c["math_heavy_non_alnum_share_min"]))
    if words is not None and m["is_english"] and len(toks) >= int(c["min_tokens_for_dictionary"]):
        m["dictionary_word_share"] = sum(1 for t in toks if t.lower() in words) / len(toks)
    else:
        m["dictionary_word_share"] = None  # not evaluated (no word list, or too few tokens)
    return m


def d2_incomprehensible(text: str, cfg: dict, words=None) -> tuple:
    c = cfg["document"][D2]
    if not c.get("enabled"):
        return [], {}
    m = d2_metrics(text, cfg, words)
    hits = []
    if m["dictionary_word_share"] is not None and m["dictionary_word_share"] < float(c["dictionary_word_share_min"]):
        hits.append("dictionary_word_share")
    if m["non_alnum_share"] > float(c["non_alnum_share_max"]):
        hits.append("non_alnum_share")
    for k, lim in (("bad_chars_per_1000", "bad_chars_per_1000_max"), ("cid_per_1000", "cid_per_1000_max"),
                   ("hyphen_break_per_1000", "hyphen_break_per_1000_max"), ("repeat_line_share", "repeat_line_share_max")):
        if m[k] > float(c[lim]):
            hits.append(k)
    if m["n_lines"] >= int(c["min_lines_for_line_metrics"]) and m["mean_line_chars"] < float(c["mean_line_chars_min"]):
        hits.append("mean_line_chars")
    return [f"{D2}:{h}" for h in hits], m


def d2_severity(metrics: dict, hits=None) -> str:
    """'purge' only for detected-English, not math-heavy text; any other D2 hit is 'review' (never purged). A
    hyphen-break hit alone (readable text with PDF line-break hyphens) is review too."""
    if hits is not None and all(str(h).split(":")[-1] == "hyphen_break_per_1000" for h in hits):
        return "review"
    if hits is not None and hits and all(str(h).split(":")[-1] == "dictionary_word_share" for h in hits):
        return "review"  # a dictionary-only hit (no other signal) is weak: dense domain vocabulary is not garble
    return "purge" if metrics.get("is_english") and not metrics.get("math_heavy") else "review"


def d3_code_severity(code: str, chunk_result: dict | None, cfg: dict) -> str:
    """missing_title alone is review (the body is intact); too_few follows d3_severity's C5 rule; the rest purge."""
    if code.endswith(":missing_title"):
        return "review"
    if code.endswith("too_few_substantive_chunks"):
        rc = (chunk_result or {}).get("rule_chunks_pre_cap") or {}
        dom = max(rc, key=rc.get) if rc else None
        return "review" if dom == cfg["document"][D3]["review_when_dominant_drop_rule"] else "purge"
    return "purge"


def d3_severity(code_list: list, chunk_result: dict | None, cfg: dict) -> str:
    return "purge" if any(d3_code_severity(x, chunk_result, cfg) == "purge" for x in code_list) else "review"



def d3_incomplete(doc: dict, cfg: dict, chunk_result: dict | None = None) -> list:
    c = cfg["document"][D3]
    if not c.get("enabled"):
        return []
    out = []
    text = doc.get("text") or ""
    prov = doc.get("provenance") or {}
    if doc.get("rebuild_ok") is False:
        out.append(D3 + ":rebuild_failed")
    for f in c["required_provenance_fields"]:
        if not prov.get(f):
            out.append(f"{D3}:missing_{f}")
    if c.get("check_sha") and prov.get("text_sha256") and text and _sha(text) != prov["text_sha256"]:
        out.append(D3 + ":sha_mismatch")
    if doc.get("has_body") is False:
        out.append(D3 + ":no_body")
    bc = doc.get("body_chars")
    if bc is not None and int(bc) < int(c["body_chars_min"]):
        out.append(D3 + ":body_chars_below_floor")
    if chunk_result is not None:
        sub, before = chunk_result["substantive_chunks"], chunk_result["chunks_before"]
        if sub < int(c["doc_min_chunks"]) or (before >= int(c["share_check_min_chunks"])
                                              and sub / before < float(c["doc_min_substantive_share"])):
            out.append(D3 + ":too_few_substantive_chunks")
    if c.get("truncation_check") and text and bc is not None and len(text) < int(bc):
        has_back = any(s.get("kind") in set(c["back_section_kinds"]) for s in doc.get("sections") or () if isinstance(s, dict))
        if not has_back and text.rstrip()[-1:] not in c["sentence_end_chars"]:
            out.append(D3 + ":truncated")
    return out


def document_flags(doc: dict, cfg: dict, words=None, chunk_result: dict | None = None) -> dict:
    """Known-bad predicates for one document (D1-D3; D4 needs the whole set: find_duplicates).
    doc: {text, sections, urls, identifiers, doi, resolver, provenance{licence_id, canonical_url, title, text_sha256},
          body_chars, has_body, rebuild_ok}. Returns {known_bad, reasons, metrics}."""
    reasons = d1_arxiv(doc, cfg)
    r2, metrics = d2_incomprehensible(doc.get("text") or "", cfg, words)
    r3 = d3_incomplete(doc, cfg, chunk_result)
    reasons += r2 + r3
    sev = {}
    for r in reasons:
        sev[r] = "purge" if r.startswith(D1) else (d2_severity(metrics, [r]) if r.startswith(D2) else d3_code_severity(r, chunk_result, cfg))
    return {"known_bad": bool(reasons), "reasons": reasons, "severity": sev,
            "purge": any(v == "purge" for v in sev.values()), "metrics": metrics}


def _sketch(text: str, dcfg: dict) -> tuple:
    toks = re.findall(dcfg["token_regex"], text.casefold())
    n = int(dcfg["shingle_words"])
    hs = {int.from_bytes(hashlib.blake2b(" ".join(toks[i:i + n]).encode("utf-8"), digest_size=8).digest(), "big")
          for i in range(max(len(toks) - n + 1, 1 if toks else 0))}
    return sorted(hs)[:int(dcfg["minhash_k"])], len(hs)


def _jaccard_est(a: list, b: list, k: int) -> float:
    sa, sb = set(a), set(b)
    union = sorted(sa | sb)[:k]
    if not union:
        return 0.0
    return sum(1 for h in union if h in sa and h in sb) / len(union)


def find_duplicates(docs: list, cfg: dict) -> dict:
    """D4. docs: [{id, text_sha256, doi, pmcid, text, acquired_at, licence_ok}] in any order. Duplicates by text hash,
    DOI/PMCID, or bottom-k MinHash Jaccard >= jaccard_min. Per group the keeper is the earliest acquired document with a
    passing licence (licence_ok not False), ties by input order; the rest map to {code, of, basis}."""
    c = cfg["document"][D4]
    if not c.get("enabled"):
        return {}
    ids = [d["id"] for d in docs]
    parent = {i: i for i in ids}
    basis: dict = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b, why):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            basis[frozenset((a, b))] = why

    for key, why in (("text_sha256", "text_sha256"), ("doi", "doi"), ("pmcid", "pmcid")):
        seen: dict = {}
        for d in docs:
            v = str(d.get(key) or "").strip().lower()
            if v:
                if v in seen:
                    union(seen[v], d["id"], why)
                else:
                    seen[v] = d["id"]
    k, jm = int(c["minhash_k"]), float(c["jaccard_min"])
    min_sh = int(c.get("minhash_min_shingles", 0))  # sparse/short texts: only hash, DOI or PMCID matches apply
    sk = {d["id"]: _sketch(d.get("text") or "", c) for d in docs if d.get("text")}
    keys = list(sk)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            na, nb = sk[a][1], sk[b][1]
            if not na or not nb or min(na, nb) < min_sh or min(na, nb) / max(na, nb) < jm or find(a) == find(b):
                continue
            if _jaccard_est(sk[a][0], sk[b][0], k) >= jm:
                union(a, b, "minhash")
    order = {d["id"]: n for n, d in enumerate(docs)}
    byid = {d["id"]: d for d in docs}
    groups: dict = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    out = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        keeper = min(members, key=lambda i: (byid[i].get("licence_ok") is False, str(byid[i].get("acquired_at") or "9999"), order[i]))
        for i in members:
            if i != keeper:
                why = basis.get(frozenset((keeper, i))) or next((v for kk, v in basis.items() if i in kk), "minhash")
                out[i] = {"code": D4, "of": keeper, "basis": why}
    return out


# ------------------------------------------------------------------------------------------------ audit entry point
CONFIG = None  # callers (the audit) may set these two after loading the module; otherwise audit_document uses its args
WORDS = None   # optional lower-case word set for the D2 dictionary metric (skipped when None)


def _default_cfg() -> dict:
    """The only file read in this module, and only when no config was injected: config/chunk_value.json next to vera/."""
    from pathlib import Path
    return json.loads((Path(__file__).resolve().parent.parent / "config" / "chunk_value.json").read_text(encoding="utf-8"))


def audit_document(meta: dict, chunks, cfg: dict | None = None, words=None) -> dict:
    """Read-only audit entry. meta: {source_id, document_id, host, licence_id, doi, title_present, chars, text_sha256, optional portal: {item, prov, canonical_url}};
    chunks: [(span_id, start_index, end_index, text)] (stored partitions, contiguous). Sections are not available to the
    audit, so C1 and C2 (section rules) do not fire here; C3-C6 run on the rebuilt text with fmt=None.
    Returns {"d_flags": [{"rule": "D1|D2|D3", "metrics": {name: number}}], "dropped": {span_id: "C4"...}}: rule ids and
    numbers only, never text. A stored partition counts as dropped when every non-space character in it lies in a
    dropped range (rule = the rule covering most of it). D4 needs the whole set (find_duplicates) and is not returned."""
    cfg = cfg or CONFIG or _default_cfg()
    words = words if words is not None else WORDS
    parts = sorted(chunks, key=lambda c: int(c[1]))
    text = "".join(c[3] for c in parts)
    # offsets come from the text actually received, never from the declared indices (stored spans may disagree)
    mismatch = sum(1 for c in parts if int(c[2]) - int(c[1]) != len(c[3]))
    gaps = sum(1 for i in range(1, len(parts)) if int(parts[i][1]) != int(parts[i - 1][2]))
    base = int(parts[0][1]) if parts else 0
    cr = evaluate_chunks(text, None, cfg, cfg["audit"]["chunking"], fmt=None)
    d_flags = []
    doc = {"text": text, "urls": ["https://%s/" % meta["host"]] if meta.get("host") else [], "doi": meta.get("doi"),
           "fulltext_url": meta.get("fulltext_url"), "orig_url": meta.get("orig_url"), "repository": meta.get("repository"),
           "identifiers": [meta.get("identifier")] if meta.get("identifier") else [],
           "provenance": {"licence_id": meta.get("licence_id"), "canonical_url": "x", "title": "x" if meta.get("title_present") else None,
                          "text_sha256": meta.get("text_sha256")},
           "rebuild_ok": mismatch == 0 and gaps == 0 and base == 0}
    if d1_arxiv(doc, cfg):
        d_flags.append({"rule": "D1", "severity": "purge", "metrics": {"arxiv": 1}})
    if "portal" in meta:  # REPROV-W3-1110: stored-row inputs {item, prov, canonical_url}; absent for callers that do not supply them
        pin = meta["portal"] if isinstance(meta["portal"], dict) else {}
        prule = portal_verdict(pin.get("item"), pin.get("prov"), pin.get("canonical_url"), cfg)
        if prule in ("internal_error", "config_invalid"):  # an undecidable row is never purged
            d_flags.append({"rule": "RULES_ERROR", "severity": "unverified", "metrics": {"error_class_len": 0}})
        elif prule:
            prior = next((f for f in d_flags if f["rule"] == "D1"), None)  # one D1 flag per document: arXiv and portal merge
            if prior is not None:
                prior["metrics"]["portal"] = 1
                prior["portal_rule"] = prule
            else:
                d_flags.append({"rule": "D1", "severity": "purge", "metrics": {"portal": 1}, "portal_rule": prule})
    hits, m = d2_incomprehensible(text, cfg, words)
    if hits:
        d_flags.append({"rule": "D2", "severity": d2_severity(m, hits),
                        "metrics": {k: float(m[k]) for k in [h.split(":", 1)[1] for h in hits] + ["is_english", "math_heavy"]}})
    r3 = d3_incomplete(doc, cfg, cr)
    if r3:
        d_flags.append({"rule": "D3", "severity": d3_severity(r3, cr, cfg), "metrics": {"substantive_chunks": cr["substantive_chunks"], "n_hits": len(r3),
                                                  "sha_mismatch": int(any(x.endswith("sha_mismatch") for x in r3)),
                                                  "rebuild_failed": int(any(x.endswith("rebuild_failed") for x in r3)),
                                                  "length_mismatch_spans": mismatch, "index_gaps": gaps}})
    dropped, lo = {}, 0
    for sid, _s, _e, t in parts:
        hi = lo + len(t)
        cover: dict = {}
        for d in cr["dropped"]:
            ov = min(d["end"], hi) - max(d["start"], lo)
            if ov > 0:
                cover[d["rule"]] = cover.get(d["rule"], 0) + ov
        if cover and all(any(d["start"] <= i < d["end"] for d in cr["dropped"])
                         for i in range(lo, hi) if not text[i].isspace()):
            dropped[sid] = max(cover, key=cover.get)
        lo = hi
    return {"d_flags": d_flags, "dropped": dropped}
