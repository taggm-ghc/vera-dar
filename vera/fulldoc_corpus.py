"""D-055 Option C, step 1 (ACQ): acquire + verify a bundle of full documents and retrieve verbatim spans from them.

Stdlib + existing vera modules. No network, no DB; the only environment read is the one variable NAMED by
config/fulldoc_answer.json (bundle_dir_env), whose value is a directory path (never a secret).

A document is ACQUIRED only when: the bundle's manifest and sums files match their pins (when pinned); the document's
.txt and .meta.json are listed in texts.SHA256SUMS and match; and vera.fulltext_loader.prepare() passes every filter
(format, converter record, full-text size/body checks, licence gate with licence_decision == "allow", text_file
traversal refusal). Anything else becomes an UNUSABLE entry {document_id, code, reason} and is never indexed or
retrieved. Retrieval reuses vera.dar_corpus.DarCorpus (chunker + BM25) over the verified texts only; every returned
span satisfies text == self.text(document_id)[start:end] (checked in code).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path

from vera import domain_registry, fulltext_loader
from vera.dar_corpus import DarCorpus

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "fulldoc_answer.json"
ROUTING_PATH = ROOT / "config" / "domain_routing.json"

STATUS_OK, STATUS_UNUSABLE = "ok", "corpus_unusable"
# own reason codes (loader codes come from Refused.code)
R_BUNDLE, R_PIN, R_SUMS, R_DUP, R_ERR = "R-BUNDLE", "R-PIN", "R-SUMS", "R-DUP", "R-ERROR"
R_SOURCE = "R-SOURCE"  # both corpus sources configured (bundle dir AND db run id): refused, never silently preferred
DB_RUN_ID_ENV_DEFAULT = "VERA_FULLDOC_DB_RUN_ID"  # env var NAME (cfg["db_run_id_env"] overrides)


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise RuntimeError(f"fulldoc config at {path} is not an object")
    return cfg


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _parse_sums(raw: bytes) -> dict | None:
    """sha256sum format -> {basename under texts/: hex}. A name listed twice with different hashes poisons the file."""
    out: dict = {}
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return None
    for ln in lines:
        if not ln.strip():
            continue
        m = re.match(r"^([0-9a-f]{64}) [ *](.+)$", ln)
        if not m:
            return None
        name = m.group(2).strip()
        for pre in ("./", "texts/"):
            if name.startswith(pre):
                name = name[len(pre):]
        if name in out and out[name] != m.group(1):
            return None
        out[name] = m.group(1)
    return out


def _unusable(did, code, reason) -> dict:
    return {"document_id": did, "code": code, "reason": str(reason)}


class FullDocCorpus:
    def __init__(self, status: str, acquired: list, unusable: list, texts: dict, dar: DarCorpus | None,
                 key_to_doc: dict, cfg: dict, docs: list | None = None):
        self.status = status
        self.acquired = acquired
        self.unusable = unusable
        self._texts = texts
        self._dar = dar
        self._key_to_doc = key_to_doc
        self._cfg = cfg
        self._docs_for_dar = list(docs or [])
        self._ev_dar = None  # evidence-only index (C1), built lazily on the first evidence_only retrieve
        self._ev_lock = threading.Lock()
        self._chunk_counts: dict = {}
        self.source_kind: str | None = None  # set by get_corpus from the branch actually taken ("db" | "bundle")
        # FRESH-AMEND-1023: document_id -> {"corpus_added_at", "corpus_stamps": [per stored chunk stamp dicts]} (the
        # corpus write times, vera.fulldoc_db.corpus_stamps); kept OFF `acquired` so the bundle/DB parity is unchanged.
        self.stamps: dict = {}
        if dar is not None:
            for c in dar._chunks:
                did = key_to_doc.get(c["source_key"])
                self._chunk_counts[did] = self._chunk_counts.get(did, 0) + 1

    def text(self, document_id: str) -> str:
        """The verified text of an acquired document (KeyError when not acquired)."""
        return self._texts[document_id]

    def _in_evidence(self, document_id, start: int, end: int) -> bool:
        """True when [start, end) lies wholly inside one evidence range of the acquired document (the admission rule)."""
        for a in self.acquired:
            if a.get("document_id") == document_id:
                return any(r[0] <= start and end <= r[1] for r in a.get("evidence_ranges") or ())
        return False

    def _evidence_dar(self) -> DarCorpus:
        """A second index holding ONLY evidence chunks (wholly inside evidence_ranges: not title, abstract, back matter or
        C1-excluded text), so evidence_only retrieval filters BEFORE top-k. Built from the same documents with the same
        chunker; df / length statistics are recomputed over the kept chunks."""
        with self._ev_lock:
            if self._ev_dar is None:
                if self._dar is not None:
                    # ACQ-REBUILD-PERF-1021: a fresh index over the same documents, order and config reaches exactly the
                    # main index's state, so reuse it (shallow copy: shared docs, registry and identifier, read-only at
                    # retrieve) instead of re-running domain identification; only the chunk list and BM25 stats differ.
                    ev = copy.copy(self._dar)
                else:
                    ev = DarCorpus(_routing_cfg())
                    for d in self._docs_for_dar:
                        ev.ingest_document(d)
                kept = []
                for c in ev._chunks:
                    did = self._key_to_doc.get(c["source_key"])
                    if did is not None and self._in_evidence(did, int(c["start"]), int(c["end"])):
                        kept.append(c)
                ev._chunks = kept
                ev._by_span = {c["span_id"]: c for c in kept}
                ev._df = Counter()
                for c in kept:
                    ev._df.update(c["tf"].keys())
                ev._total_len = sum(c["len"] for c in kept)
                self._ev_dar = ev
            return self._ev_dar

    def retrieve(self, question: str, k: int | None = None, evidence_only: bool = False) -> list:
        """Top-k candidate spans. evidence_only=True drops non-evidence chunks (abstract, title, back matter, C1-excluded
        sections) BEFORE the top-k cut; the default keeps today's behaviour. Each span carries `in_evidence`."""
        if self.status != STATUS_OK or self._dar is None or not self._texts:
            return []
        rcfg = self._cfg.get("retrieval") or {}
        k = int(k if k is not None else rcfg.get("candidate_k", 12))
        mpd = rcfg.get("max_chunks_per_doc")  # WED-P16-RET: absent = the routing config's value (unchanged behaviour)
        dar = self._evidence_dar() if evidence_only else self._dar
        out = []
        for c in dar.retrieve(str(question or ""), k, max_per_doc=None if mpd is None else int(mpd))["candidates"]:
            did = self._key_to_doc.get(c["source_key"])
            if did is None:
                continue
            s, e = int(c["start"]), int(c["end"])
            span = self._texts[did][s:e]
            assert span == c["text"], "span text differs from the verified document slice"
            out.append({"span_id": f"{did}:{s}-{e}", "document_id": did, "start": s, "end": e, "text": span,
                        "score": round(self._dar.relevance(question, span), 6),
                        "in_evidence": self._in_evidence(did, s, e)})
        return out

    def evidence_chunk_texts(self) -> list:
        """Texts of the body evidence chunks (wholly inside evidence_ranges): the universe for IDF weighting."""
        if self._dar is None:
            return []
        return [c["text"] for c in self._dar._chunks
                if self._key_to_doc.get(c["source_key"]) is not None
                and self._in_evidence(self._key_to_doc[c["source_key"]], int(c["start"]), int(c["end"]))]

    def inspection(self) -> dict:
        """Every chunk of every acquired document is scored by retrieve(), so chunks_inspected = its chunk count."""
        return {d["document_id"]: {"chunks_inspected": self._chunk_counts.get(d["document_id"], 0)}
                for d in self.acquired}


def _manifest_items(raw: bytes):
    try:
        m = json.loads(raw.decode("utf-8"))
        items = m.get("sources") if isinstance(m, dict) else None
        return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else None
    except (ValueError, UnicodeDecodeError, AttributeError):
        return None


def _bundle_unusable(cfg, items, code, reason) -> FullDocCorpus:
    un = [_unusable(i.get("document_id"), code, reason) for i in (items or [])] or [_unusable(None, code, reason)]
    return FullDocCorpus(STATUS_UNUSABLE, [], un, {}, None, {}, cfg)


def _routing_cfg(path: Path = ROUTING_PATH) -> dict:
    shipped = domain_registry.load_config(path)  # read-only; deep copy, then force routing on in the COPY
    if not shipped:
        raise RuntimeError("routing config unreadable")
    out = copy.deepcopy(shipped)
    out["enabled"] = True
    return out


def _norm_heading(title) -> str:
    """C1 heading rule: strip a leading section number, casefold, collapse whitespace, drop trailing punctuation."""
    t = _SECNUM.sub("", str(title or ""), count=1)
    t = re.sub(r"\s+", " ", t.casefold()).strip()
    return re.sub(r"[\W_]+$", "", t).strip()


_SECNUM = re.compile(r"^\s*[\dIVX]+(\.\d+)*[.)]?\s+")


def _merge(rs, gap: int = 0) -> list:
    out: list = []
    for a, b in sorted((int(a), int(b)) for a, b in rs):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


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


def _excluded_ranges(sections, cfg: dict, text: str | None) -> list:
    """C1 exclusions inside evidence-kind sections, each {start, end, reason: "excluded_section", rule, match}.
    (1) admission.excluded_section_headings: a section whose normalised heading equals a listed heading, plus its nested
        subsections (following sections of the same kind with a deeper level, up to the next of equal or higher level).
    (2) admission.excluded_paragraph_labels: a paragraph that STARTS with a listed label (anchored, never a substring)
        is removed from the label to the end of its innermost enclosing section.
    Absent or empty lists => [] (today's behaviour)."""
    adm = (cfg or {}).get("admission") or {}
    kinds = set(adm.get("evidence_section_kinds") or ["body"])
    heads = {_norm_heading(h) for h in adm.get("excluded_section_headings") or () if _norm_heading(h)}
    labels = [str(x) for x in adm.get("excluded_paragraph_labels") or () if str(x).strip()]
    n = len(text) if isinstance(text, str) else max([int(s.get("end", 0)) for s in sections or () if isinstance(s, dict)] or [0])
    secs = _valid_sections(sections, n)
    out = []
    if heads:
        for i, m in enumerate(secs):
            if m["kind"] not in kinds or _norm_heading(m["title"]) not in heads:
                continue
            end = m["end"]
            for s in secs[i + 1:]:
                if s["kind"] == m["kind"] and s["level"] > m["level"]:
                    end = max(end, s["end"])
                else:
                    break
            out.append({"start": m["start"], "end": end, "reason": "excluded_section", "rule": "heading",
                        "match": _norm_heading(m["title"])})
    if labels and isinstance(text, str):
        pats = [(lb, re.compile(r"[ \t]*" + r"\s+".join(re.escape(w) for w in lb.split()), re.I)) for lb in labels if lb.split()]
        body = [s for s in secs if s["kind"] in kinds]
        for m in body:
            pos = m["start"]
            while pos < m["end"]:
                nl = text.find("\n", pos, m["end"])
                line_end = m["end"] if nl < 0 else nl
                for lb, pat in pats:
                    hit = pat.match(text, pos, line_end)
                    if hit:
                        inner = min((s for s in body if s["start"] <= pos < s["end"]), key=lambda s: s["end"] - s["start"])
                        rec = {"start": pos, "end": inner["end"], "reason": "excluded_section", "rule": "label",
                               "match": lb}
                        if rec not in out:
                            out.append(rec)
                        break
                pos = line_end + 1
    return out


def _evidence_ranges(sections, cfg: dict, n: int, text: str | None = None, excluded: list | None = None) -> list:
    """Merged [start, end] ranges of the sections whose kind may be evidence (admission.evidence_section_kinds, default
    body only: never title, abstract or back matter), minus the C1 excluded ranges (computed from `text` when not given).
    Gaps of <= admission.section_gap_chars (the converter's blank line between sections) are bridged. No section record
    => [] (nothing admissible: fail closed)."""
    adm = (cfg or {}).get("admission") or {}
    kinds = set(adm.get("evidence_section_kinds") or ["body"])
    gap = int(adm.get("section_gap_chars", 2))
    rs = [(int(s["start"]), int(s["end"])) for s in sections or ()
          if isinstance(s, dict) and s.get("kind") in kinds and 0 <= int(s["start"]) < int(s["end"]) <= n]
    out = _merge(rs, gap)
    if excluded is None:
        excluded = _excluded_ranges(sections, cfg, text)
    for x in _merge([(e["start"], e["end"]) for e in excluded]):
        nxt = []
        for a, b in out:
            if x[1] <= a or b <= x[0]:
                nxt.append([a, b])
                continue
            if a < x[0]:
                nxt.append([a, x[0]])
            if x[1] < b:
                nxt.append([x[1], b])
        out = nxt
    return out


def load_bundle(bundle_dir, cfg: dict, ingest_cfg: dict | None = None) -> FullDocCorpus:
    """Never raises for bad input: any problem yields corpus_unusable / UNUSABLE entries."""
    try:
        return _load(Path(bundle_dir), cfg, ingest_cfg)
    except Exception as e:  # noqa: BLE001  fail closed
        return _bundle_unusable(cfg, None, R_ERR, f"bundle could not be loaded ({type(e).__name__})")


def _load(base: Path, cfg: dict, ingest_cfg: dict | None) -> FullDocCorpus:
    icfg = ingest_cfg if ingest_cfg is not None else fulltext_loader.load_config()
    base = base.resolve()
    mpath, spath = base / cfg["manifest_name"], base / cfg["sums_name"]
    if not (mpath.is_file() and spath.is_file()):
        return _bundle_unusable(cfg, None, R_BUNDLE, "manifest or sums file missing")
    mraw, sraw = mpath.read_bytes(), spath.read_bytes()
    items = _manifest_items(mraw)
    pins = cfg.get("pins") or {}
    if pins.get("manifest_sha256") and _sha_bytes(mraw) != pins["manifest_sha256"]:
        return _bundle_unusable(cfg, items, R_PIN, "manifest sha256 differs from the pin")
    if pins.get("sums_sha256") and _sha_bytes(sraw) != pins["sums_sha256"]:
        return _bundle_unusable(cfg, items, R_PIN, "sums sha256 differs from the pin")
    if items is None:
        return _bundle_unusable(cfg, None, R_BUNDLE, "manifest unreadable or has no sources list")
    sums = _parse_sums(sraw)
    if sums is None:
        return _bundle_unusable(cfg, items, R_SUMS, "sums file malformed or contradictory")

    acquired, unusable, texts, key_to_doc, docs = [], [], {}, {}, []
    seen_ids, seen_keys, seen_hashes = set(), set(), set()
    for item in items:
        did = item.get("document_id")
        try:
            if not isinstance(did, str) or did in seen_ids:
                raise fulltext_loader.Refused(R_DUP, f"document_id {did!r} missing or duplicated")
            seen_ids.add(did)
            tf = item.get("text_file")
            if isinstance(tf, str):
                tp = (base / tf).resolve()
                # S6 (AE-1020): the sums check is mandatory. A symlink, or a path resolving outside <bundle>/texts,
                # is refused here instead of skipping the check (fail closed).
                if (base / tf).is_symlink() or tp.with_suffix(".meta.json").is_symlink():
                    raise fulltext_loader.Refused(R_SUMS, "text_file or its meta file is a symlink")
                if tp.parent != base / "texts":
                    raise fulltext_loader.Refused(R_SUMS, "text_file is not directly under the bundle texts/ directory")
                for p in (tp, tp.with_suffix(".meta.json")):  # sums must list and match both files
                    want = sums.get(p.name)
                    if want is None:
                        raise fulltext_loader.Refused(R_SUMS, f"{p.name} is not listed in the sums file")
                    if not p.is_file() or _sha_bytes(p.read_bytes()) != want:
                        raise fulltext_loader.Refused(R_SUMS, f"{p.name} does not match the sums file")
                want_text = sums[tp.name]
            else:
                want_text = None
            rec = fulltext_loader.prepare(item, base, icfg)
            text = rec["text"]
            if want_text is not None and _sha_bytes(text.encode("utf-8")) != want_text:
                raise fulltext_loader.Refused(R_SUMS, "verified text differs from the sums file")
            if rec["provenance"].get("licence_decision") != "allow":
                raise fulltext_loader.Refused(R_ERR, "licence_decision is not allow")
            key = rec["canonical_url"]
            if key in seen_keys:
                raise fulltext_loader.Refused(R_DUP, f"canonical source {key} already acquired")
            if rec["content_hash"] in seen_hashes:  # D-057 parity: identity is the content, not the URL
                raise fulltext_loader.Refused(R_DUP, "same full text already acquired under another canonical URL")
        except fulltext_loader.Refused as e:
            unusable.append(_unusable(did, e.code, e))
            continue
        except Exception as e:  # noqa: BLE001
            unusable.append(_unusable(did, R_ERR, f"document could not be verified ({type(e).__name__})"))
            continue
        seen_keys.add(key)
        seen_hashes.add(rec["content_hash"])
        p = rec["provenance"]
        lic = (p.get("licence") or {}).get("content") or {}
        texts[did] = text
        ex_ranges = _excluded_ranges(p.get("sections"), cfg, text)
        key_to_doc[key] = did
        docs.append(fulltext_loader.doc_for_dar(rec, None, None, icfg))
        acquired.append({"document_id": did, "canonical_source_id": key, "title": rec["title"], "url": rec["url"],
                         "doi": p["doi"], "published": p["published"], "authors": p["authors"],
                         "licence": lic.get("id"), "licence_url": lic.get("url"),
                         "licence_decision": p["licence_decision"], "attribution": p.get("attribution"),
                         "fulltext_format": p.get("fulltext_format"),
                         "text_sha256": _sha_bytes(text.encode("utf-8")), "chars": len(text),
                         "evidence_ranges": _evidence_ranges(p.get("sections"), cfg, len(text), text, ex_ranges),
                         "excluded_ranges": ex_ranges})
    dar = None
    if docs:
        dar = DarCorpus(_routing_cfg())
        for d in docs:
            res = dar.ingest_document(d)
            if res.get("status") != "indexed":  # cannot be retrieved => not usable
                did = key_to_doc.pop(d["source_key"])
                texts.pop(did, None)
                acquired[:] = [a for a in acquired if a["document_id"] != did]
                unusable.append(_unusable(did, R_ERR, f"retrieval index refused the document ({res.get('reason')})"))
    return FullDocCorpus(STATUS_OK, acquired, unusable, texts, dar, key_to_doc, cfg,
                         [d for d in docs if d["source_key"] in key_to_doc])


CACHE_MAX_ENTRIES = 8  # S7 (AE-1020): bound on cached corpora; config key "cache_max_entries" (int >= 1) overrides


class _BoundedCache(dict):
    """dict with a size bound (oldest-inserted evicted first) and a lock around mutation. Hits do not reorder."""

    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES):
        super().__init__()
        self.max_entries = max_entries
        self.lock = threading.RLock()

    def __setitem__(self, key, value):
        with self.lock:
            super().__setitem__(key, value)
            while len(self) > max(1, int(self.max_entries)):
                del self[next(iter(self))]

    def clear(self):
        with self.lock:
            super().clear()


_CACHE: dict = _BoundedCache()
_LOAD_LOCK = threading.Lock()  # serialises bundle loads so concurrent first calls do not double-load


def _apply_cache_bound(cfg: dict) -> None:
    n = cfg.get("cache_max_entries")
    _CACHE.max_entries = n if isinstance(n, int) and not isinstance(n, bool) and n >= 1 else CACHE_MAX_ENTRIES


def reset_for_tests() -> None:
    _CACHE.clear()
    reset_acquired_for_tests()
    from vera import fulldoc_db
    fulldoc_db.reset_for_tests()


def _db_run_env(cfg: dict) -> str:
    return str(cfg.get("db_run_id_env") or DB_RUN_ID_ENV_DEFAULT)


def _label_bundle(c):
    if isinstance(c, FullDocCorpus):
        c.source_kind = "bundle"
    return c


def flag_enabled(cfg: dict | None) -> bool:
    """The fulldoc.enabled check of get_corpus alone (strict allowlist, absent = off). No I/O, never raises."""
    fd = cfg.get("fulldoc") if isinstance(cfg, dict) else None
    if not isinstance(fd, dict) or "enabled" not in fd:
        return False
    en = fd["enabled"]
    return en is True or str(en).strip().lower() in ("true", "1", "yes", "on")


def get_corpus(cfg: dict | None = None, env=os.environ) -> FullDocCorpus | None:
    """None when the path is disabled (fulldoc.enabled is false, or the env var is unset/empty): /ask then behaves exactly
    as before. An absent `fulldoc` / `enabled` key counts as DISABLED (fail closed, ON-1020-FIX-1); only an explicit
    true-like value enables it."""
    cfg = cfg if cfg is not None else load_config()
    fd = cfg.get("fulldoc")
    if not isinstance(fd, dict) or "enabled" not in fd:
        return None
    en = fd["enabled"]
    if en is not True and str(en).strip().lower() not in ("true", "1", "yes", "on"):  # strict allowlist: fail closed
        return None
    raw = (env.get(cfg["bundle_dir_env"]) or "").strip()
    run_raw = (env.get(_db_run_env(cfg)) or "").strip()
    if raw and run_raw:  # one source only: never pick silently
        return _bundle_unusable(cfg, None, R_SOURCE, "both the bundle and the database corpus source are configured")
    if run_raw:
        from vera import fulldoc_db
        connect = fulldoc_db.default_connect(cfg, env)
        if connect is None:
            return _bundle_unusable(cfg, None, fulldoc_db.R_DBCONN, "database corpus source has no read-only connection")
        _apply_cache_bound(cfg)
        c = fulldoc_db.cached_load(connect, run_raw, cfg, _CACHE)
        c.source_kind = "db"
        return c
    if not raw:
        return None
    base = Path(raw).resolve()
    try:
        sig = (str(base), _sha_bytes((base / cfg["manifest_name"]).read_bytes()),
               _sha_bytes((base / cfg["sums_name"]).read_bytes()))
    except OSError:
        return _label_bundle(load_bundle(base, cfg))  # unusable; not cached so a later fix is seen
    _apply_cache_bound(cfg)
    hit = _CACHE.get(sig)  # lock-free hit: never waits on another bundle's slow load
    if hit is not None:
        return _label_bundle(hit)
    with _LOAD_LOCK:
        hit = _CACHE.get(sig)
        if hit is None:
            hit = _CACHE[sig] = load_bundle(base, cfg)
        return _label_bundle(hit)


def readiness(cfg: dict | None = None, env=os.environ) -> dict:
    """C-3/S8 (AE-1020): codes and counts only (no path, no titles). Never raises; any exception => status "error".
    status: disabled (flag off) | not_configured (flag on, env unset) | ok | unusable | error.
    `source` names the corpus backend so a later DB-backed source can report without a shape change."""
    out = {"enabled": False, "bundle_configured": False, "status": "error", "n_acquired": 0, "n_unusable": 0,
           "pins_ok": None, "source": "bundle"}
    try:
        cfg = cfg if cfg is not None else load_config()
        fd = cfg.get("fulldoc")
        en = fd.get("enabled") if isinstance(fd, dict) else None
        out["enabled"] = en is True or str(en).strip().lower() in ("true", "1", "yes", "on")
        out["bundle_configured"] = bool((env.get(cfg["bundle_dir_env"]) or "").strip())
        db_on = bool((env.get(_db_run_env(cfg)) or "").strip())  # a flag only: never the run id value
        if db_on:  # the db keys appear only when the db source is configured (bundle shape unchanged)
            out.update(db_configured=True, codes=[], source="db" if not out["bundle_configured"] else "bundle")
        if not out["enabled"]:
            out["status"] = "disabled"
            return out
        if out["bundle_configured"] and db_on:
            out.update(source="conflict", status="unusable", codes=[R_SOURCE])
            return out
        if not (out["bundle_configured"] or db_on):
            out["status"] = "not_configured"
            return out
        c = get_corpus(cfg, env)
        if c is None:
            out["status"] = "error"
            return out
        out["n_acquired"], out["n_unusable"] = len(c.acquired), len(c.unusable)
        if db_on:
            out["codes"] = sorted({str(u.get("code")) for u in c.unusable if u.get("code")})
        from vera import fulldoc_db  # local import: fulldoc_db imports this module
        out["pins_ok"] = not any(u.get("code") in (R_PIN, fulldoc_db.R_DBDOCPIN) for u in c.unusable) and c.status == STATUS_OK
        out["status"] = "ok" if c.status == STATUS_OK and c.acquired else "unusable"
        return out
    except Exception:  # noqa: BLE001  fail closed, never leak the message
        out.update(status="error", n_acquired=0, n_unusable=0, pins_ok=None)
        if "codes" in out:
            out["codes"] = []
        return out


# ------------------------------------------------------------------------------------------------ WED-ACQ2 acquired corpus
# The automatically acquired documents (vera/auto_acquire.py) are served as a SEPARATE FullDocCorpus (plan s11 V6):
# ask_service admits over the base corpus first and consults this one only when the base cannot answer. Refresh is a
# cheap read-only probe (max source_id, count) at most every serve.refresh_ttl_s; a rebuild runs OFF the request path
# (a daemon thread, atomic swap), so a request never waits for one (V7). Tests call refresh_acquired(sync=True).
_ACQ: dict = {"sig": None, "corpus": None, "checked": None, "building": False, "probing": False}
_ACQ_LOCK = threading.Lock()


def _acq_setup(cfg: dict, env):
    """(acq_cfg, connect, base_run_id) when acquired documents may be served, else None. Never raises."""
    try:
        from vera import acq_config, fulldoc_db
        acfg = acq_config.load_config()
        if not acq_config.enabled(acfg, env):  # flag, kill switch AND both DB env vars (AC-12)
            return None
        run_raw = (env.get(_db_run_env(cfg)) or "").strip()
        if not run_raw or (env.get(cfg["bundle_dir_env"]) or "").strip():
            return None  # acquired documents are DB rows: served only next to the DB-backed base corpus
        rid = fulldoc_db.parse_run_id(run_raw)
        ro_cfg = {**cfg, "db_url_env": acfg.get("db_url_env_ro") or cfg.get("db_url_env")}
        connect = fulldoc_db.default_connect(ro_cfg, env)
        if rid is None or connect is None:
            return None
        return acfg, connect, rid
    except Exception:  # noqa: BLE001
        return None


def _rebuild_acquired(cfg, setup, sig) -> None:
    from vera import fulldoc_db
    acfg, connect, rid = setup
    try:
        c = fulldoc_db.load_acquired_db(connect, rid, cfg, acfg)
        if c.status == STATUS_OK and c.acquired and (cfg.get("retrieval") or {}).get("evidence_only"):
            c._evidence_dar()  # pre-warm the evidence-only index here, never on the request path (measured, plan V7)
        with _ACQ_LOCK:
            _ACQ["corpus"] = c if (c.status == STATUS_OK and c.acquired) else None
            _ACQ["sig"] = sig
    finally:
        with _ACQ_LOCK:
            _ACQ["building"] = False


def refresh_acquired(cfg: dict | None = None, env=os.environ, sync: bool = False, now=time.monotonic) -> None:
    """Probe the acquired-row signature and rebuild when it changed. Never raises; sync=True rebuilds in this thread
    (the acquisition worker after a commit, and tests)."""
    try:
        cfg = cfg if cfg is not None else load_config()
        setup = _acq_setup(cfg, env)
        if setup is None:
            with _ACQ_LOCK:
                _ACQ.update(sig=None, corpus=None, checked=now())
            return
        from vera import fulldoc_db
        acfg, connect, rid = setup
        sig = fulldoc_db.acquired_signature(connect, rid, acfg["marker"])
        with _ACQ_LOCK:
            _ACQ["checked"] = now()
            if sig == _ACQ["sig"] or _ACQ["building"]:
                return
            _ACQ["building"] = True
        if sync:
            _rebuild_acquired(cfg, setup, sig)
        else:
            threading.Thread(target=_rebuild_acquired, args=(cfg, setup, sig), name="vera-autoacq-serve",
                             daemon=True).start()
    except Exception as e:  # noqa: BLE001  (class only; the base corpus is unaffected)
        import logging
        logging.getLogger("vera").warning("acquired corpus refresh failed (%s)", type(e).__name__)
        with _ACQ_LOCK:
            _ACQ["checked"] = now()


def get_acquired_corpus(cfg: dict | None = None, env=os.environ, now=time.monotonic) -> FullDocCorpus | None:
    """The current acquired corpus, or None (feature off, nothing acquired yet, or first build still running)."""
    try:
        cfg = cfg if cfg is not None else load_config()
        if _acq_setup(cfg, env) is None:
            return None
        from vera import acq_config
        ttl = float(((acq_config.load_config() or {}).get("serve") or {}).get("refresh_ttl_s", 60))
        with _ACQ_LOCK:
            due = (_ACQ["checked"] is None or now() - _ACQ["checked"] >= ttl) and not _ACQ["probing"]
            if due:
                _ACQ["probing"], _ACQ["checked"] = True, now()  # one probe at a time; failures back off for ttl
        if due:  # the probe AND any rebuild run in a background thread: the request never waits on the DB (V7, D1)
            threading.Thread(target=_probe_and_rebuild, args=(cfg, env, now), name="vera-autoacq-serve",
                             daemon=True).start()
        with _ACQ_LOCK:
            return _ACQ["corpus"]
    except Exception:  # noqa: BLE001
        return None


def _probe_and_rebuild(cfg, env, now) -> None:
    try:
        refresh_acquired(cfg, env, sync=True, now=now)
    finally:
        with _ACQ_LOCK:
            _ACQ["probing"] = False


def invalidate_acquired() -> None:
    """Force the next get_acquired_corpus() to probe (called by the acquisition worker after a commit)."""
    with _ACQ_LOCK:
        _ACQ["checked"] = None


def reset_acquired_for_tests() -> None:
    with _ACQ_LOCK:
        _ACQ.update(sig=None, corpus=None, checked=None, building=False, probing=False)
