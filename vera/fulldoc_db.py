"""AE-1020/DBC-1: DB-backed source for the full-document corpus (offline-built; nothing here opens a connection by itself).

Owner D-4: the full texts stay local (not in git); their chunks go into the project's PostgreSQL chunk table
(vera_vjay.evidence_spans). Storage model (vet R1): each document is ONE `sources` row (stored_text_kind
"fulldoc_chunks", content = '') plus a NON-overlapping, CONTIGUOUS partition of its text in `evidence_spans`
(start_index/end_index = offsets into the text). Rebuild = concatenation, so no gap list and no chunk-config coupling of
the stored rows. (R4) content_hash is sha256 of the empty content, which keeps the invariant content_hash == sha256(content);
the sha256 of the full text lives in provenance.text_sha256.

Reader (R3): rebuilds each text, verifies its sha256 (provenance.text_sha256 == the conversion record's text_sha256 ==
sha of the concatenation) and the bundle pins recorded with the run, then RE-APPLIES the fulltext_ingest filters through the
same pure helpers prepare() uses (fulltext_loader.check_item_declared / prepare_from_parts), then builds the same
FullDocCorpus the bundle path builds. The rebuilt text is checked as the configured full-document kind, never under the
row's own marker kind (that marker is listed in discovery_contract non_document_text_kinds to keep the empty row out of the
content-based legacy paths). Fail closed: any mismatch => the document (or the whole run, for pins) is UNUSABLE.

DB access: injected `connect` factory (a DB-API connection; the runtime default is built from the env var NAMED by config,
default VERA_DB_URL_RO, read-only account). Only two static SELECT statements are issued. Env var NAMES only; values are
never logged or returned."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time

from vera import fulltext_loader
from vera.corpus_admission import SCHEMA
from vera.dar_corpus import DarCorpus
from vera.dar_store import FULLDOC_CHUNKS_KIND
from vera.fulldoc_corpus import (STATUS_OK, STATUS_UNUSABLE, FullDocCorpus, _evidence_ranges, _excluded_ranges,
                                 _routing_cfg, _unusable, R_DUP, R_ERR, R_PIN)

DEFAULT_RUN_ID_ENV = "VERA_FULLDOC_DB_RUN_ID"
DEFAULT_URL_ENV = "VERA_DB_URL_RO"
PART_TARGET_CHARS = 4000
PART_EVIDENCE_TYPE = "fulldoc_partition"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

R_DBROW, R_DBHASH, R_DBEMPTY, R_DBRUN, R_DBCHUNK, R_DBCONN, R_DBPARTIAL = (
    "R-DBROW", "R-DBHASH", "R-DBEMPTY", "R-DBRUN", "R-DBCHUNK", "R-DBCONN", "R-DBPARTIAL")
R_DBDOCPIN = "R-DBDOCPIN"  # per-document identity pin (text sha256, licence id, licence decision) missing or mismatched


class _DocPinFail(Exception):
    """A per-document pin mismatch: the WHOLE run is unusable (fail closed), not just the one document."""

SQL_SOURCES = (f"SELECT s.source_id, cs.canonical_url, s.content, s.content_hash, s.provenance, s.created_at "
               f"FROM {SCHEMA}.sources s JOIN {SCHEMA}.canonical_sources cs ON cs.canonical_source_id = s.canonical_source_id "
               f"JOIN {SCHEMA}.candidates d ON d.candidate_id = s.candidate_id "
               f"WHERE d.run_id = %(run_id)s AND s.provenance->>'stored_text_kind' = %(kind)s ORDER BY s.source_id")
SQL_SPANS = (f"SELECT e.source_id, e.start_index, e.end_index, e.text, e.span_id, e.created_at FROM {SCHEMA}.evidence_spans e "
             f"WHERE e.source_id = ANY(%(ids)s) ORDER BY e.source_id, e.start_index")


def _at(r, i) -> dict:
    """FRESH-AMEND-1023: the row's DB write time (created_at, set by the database at INSERT) when the SELECT carries it."""
    v = r[i] if len(r) > i else None
    return {} if v is None else {"created_at": v.isoformat() if hasattr(v, "isoformat") else str(v)}


def _with_stamps(stamps: dict, corpus):
    corpus.stamps = {k: v for k, v in stamps.items() if k in corpus._texts}
    return corpus


def corpus_stamps(row: dict, spans: list, prov: dict) -> list:
    """FRESH-AMEND-1023: per stored chunk {span_id, start, end, corpus_added_at, basis}: the chunk's own write time
    (evidence_spans.created_at, basis 'chunk'), else the document's (sources.created_at, else the writer's app-clock
    provenance.retrieved_at, basis 'document'). Shared with the served corpus: only a missing/invalid stamp is ever
    revised (to the request's now(), vera.corpus_stamp); a valid one never changes."""
    doc_at = row.get("created_at") or (prov or {}).get("retrieved_at")
    out = []
    for sp in spans or ():
        own = sp.get("created_at")
        sid = sp.get("span_id")
        out.append({"span_id": sid if sid is not None else None, "start": sp.get("start_index"), "end": sp.get("end_index"),
                    "corpus_added_at": own if own not in (None, "") else doc_at,
                    "basis": "chunk" if own not in (None, "") else "document"})
    return out


# ------------------------------------------------------------------------------------------------ writer-side (pure)
def chunking_params(routing_cfg: dict | None = None) -> dict:
    """The live DarCorpus chunk settings (recorded at ingest, compared at read: CAL parity)."""
    c = (routing_cfg or _routing_cfg()).get("corpus") or {}
    return {k: int(c[k]) for k in ("chunk_chars", "chunk_overlap", "min_chunk_chars")}


def partition(text: str, target: int = PART_TARGET_CHARS) -> list:
    """Contiguous, non-overlapping [(start, end)] covering text exactly; cuts after whitespace when one is near the target."""
    n, out, pos = len(text), [], 0
    while pos < n:
        end = pos + target
        if end >= n:
            end = n
        else:
            cut = next((i for i in range(end, max(pos + 1, end - 200), -1) if text[i - 1].isspace()), None)
            end = cut if cut else end
        out.append((pos, end))
        pos = end
    return out


def build_rows(rec: dict, item: dict, meta: dict, seq: int, pins: dict, chunking: dict,
               target: int = PART_TARGET_CHARS, n_documents: int | None = None) -> dict:
    """The row plan for one prepared document: {"source": {...}, "spans": [...]}. No SQL, no ids (the DB assigns them)."""
    text = rec["text"]
    if "\x00" in text:
        raise ValueError("text contains NUL (cannot be stored in PostgreSQL TEXT)")
    parts = partition(text, target)
    prov = dict(rec["provenance"])
    prov.update({"stored_text_kind": FULLDOC_CHUNKS_KIND, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                 "chars": len(text), "partition": {"kind": "contiguous", "n_parts": len(parts), "target_chars": target},
                 "bundle": {"manifest_sha256": pins.get("manifest_sha256"), "sums_sha256": pins.get("sums_sha256")},
                 "chunking": dict(chunking), "seq": int(seq), "n_documents": n_documents, "item": {k: v for k, v in item.items() if k != "domain_hint"},
                 "meta": meta})
    return {"document_id": rec["document_id"],
            "source": {"canonical_url": rec["canonical_url"], "content": "", "content_hash": EMPTY_SHA256,
                       "provenance": prov},
            "spans": [{"start_index": s, "end_index": e, "text": text[s:e], "evidence_type": PART_EVIDENCE_TYPE,
                       "relevance_score": None} for s, e in parts]}


# ------------------------------------------------------------------------------------------------ reader (pure)
def _prov(row) -> dict:
    p = row.get("provenance")
    if isinstance(p, (str, bytes)):
        p = json.loads(p)
    if not isinstance(p, dict):
        raise fulltext_loader.Refused(R_DBROW, "provenance is not an object")
    return p


def _rebuild(src: dict, spans: list, prov: dict) -> str:
    """Concatenate the partition; every structural property is checked, nothing is repaired."""
    if src.get("content") != "" or src.get("content_hash") != EMPTY_SHA256:
        raise fulltext_loader.Refused(R_DBROW, "source row is not an empty-content chunk-stored row")
    if prov.get("stored_text_kind") != FULLDOC_CHUNKS_KIND:
        raise fulltext_loader.Refused(R_DBROW, "source row is not the chunk-stored kind")
    if not spans:
        raise fulltext_loader.Refused(R_DBROW, "no partition rows")
    pos, parts = 0, []
    for sp in sorted(spans, key=lambda x: int(x["start_index"])):
        s, e, t = int(sp["start_index"]), int(sp["end_index"]), sp["text"]
        if s != pos or e <= s or not isinstance(t, str) or len(t) != e - s:
            raise fulltext_loader.Refused(R_DBROW, "partition is not contiguous / lengths differ")
        parts.append(t)
        pos = e
    n = (prov.get("partition") or {}).get("n_parts")
    if prov.get("chars") != pos or n != len(parts):
        raise fulltext_loader.Refused(R_DBROW, "partition size differs from the recorded size")
    text = "".join(parts)
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if h != prov.get("text_sha256") or h != (prov.get("meta") or {}).get("text_sha256"):
        raise fulltext_loader.Refused(R_DBHASH, "rebuilt text sha256 differs from the recorded sha256")
    return text


_PORTAL_REFUSED = [0]  # PORTAL-RULE-1110 CP5: stored rows refused at load since process start (a counter, no host or text)


def portal_refused_count() -> int:
    return _PORTAL_REFUSED[0]


def _unusable_corpus(cfg, items, code, reason) -> FullDocCorpus:
    un = [_unusable(i, code, reason) for i in items] or [_unusable(None, code, reason)]
    return FullDocCorpus(STATUS_UNUSABLE, [], un, {}, None, {}, cfg)


def load_from_rows(sources: list, spans: list, cfg: dict, ingest_cfg: dict | None = None,
                   extra_keys: frozenset | set | None = None, build_index: bool = True) -> FullDocCorpus:
    """sources: dicts {source_id, canonical_url, content, content_hash, provenance}; spans: dicts {source_id, start_index,
    end_index, text}. Never raises for bad input: any problem yields corpus_unusable / UNUSABLE entries."""
    try:
        return _load_rows(list(sources or ()), list(spans or ()), cfg, ingest_cfg, extra_keys, build_index)
    except Exception as e:  # noqa: BLE001  fail closed
        return _unusable_corpus(cfg, [], R_ERR, f"rows could not be loaded ({type(e).__name__})")


def _load_rows(sources, spans, cfg, ingest_cfg, extra_keys=None, build_index=True) -> FullDocCorpus:
    icfg = ingest_cfg if ingest_cfg is not None else fulltext_loader.load_config()
    ek = _EXTRA_KEYS if extra_keys is None else frozenset(extra_keys)  # WED-ACQ2: wider ONLY on the acquired path
    if not sources:
        return _unusable_corpus(cfg, [], R_DBEMPTY, "no chunk-stored documents for the configured run")
    by_src: dict = {}
    for sp in spans:
        by_src.setdefault(sp.get("source_id"), []).append(sp)
    live_chunking = chunking_params()
    recs = []
    for row in sources:
        try:
            recs.append((row, _prov(row)))
        except Exception:  # noqa: BLE001
            return _unusable_corpus(cfg, [], R_DBROW, "a source row has unreadable provenance")
    recs.sort(key=lambda rp: (int(rp[1].get("seq") or 0), int(rp[0].get("source_id") or 0)))
    ids = [(p.get("document_id") if isinstance(p, dict) else None) for _, p in recs]
    # pins recorded with the run: one set across all rows, equal to the configured pins when those are set
    recorded = {json.dumps(p.get("bundle"), sort_keys=True) for _, p in recs}
    pins = cfg.get("pins") or {}
    if len(recorded) != 1:
        return _unusable_corpus(cfg, ids, R_PIN, "rows record different bundle pins")
    b = recs[0][1].get("bundle") or {}
    if not (b.get("manifest_sha256") and b.get("sums_sha256")):
        return _unusable_corpus(cfg, ids, R_PIN, "rows record no bundle pins")
    for k in ("manifest_sha256", "sums_sha256"):
        if pins.get(k) and b.get(k) != pins[k]:
            return _unusable_corpus(cfg, ids, R_PIN, f"recorded {k} differs from the configured pin")
    # E3: with the bundle pinned, every document's identity is pinned too; the recorded sha256s alone are self-consistent
    # under a coordinated tamper (rows + provenance rewritten together). Fully unpinned configs (tests only) skip this.
    doc_pins = None
    if pins.get("manifest_sha256") or pins.get("sums_sha256"):
        doc_pins = pins.get("documents")
        if not isinstance(doc_pins, dict) or not doc_pins or set(doc_pins) != set(ids):
            return _unusable_corpus(cfg, ids, R_DBDOCPIN, "per-document pins are missing or differ from the run's documents")

    nds = {p.get("n_documents") for _, p in recs}
    if len(nds) != 1 or not isinstance(next(iter(nds)), int) or next(iter(nds)) != len(recs):
        return _unusable_corpus(cfg, ids, R_DBPARTIAL, "the run does not hold all of its recorded documents")
    max_bytes = int(icfg["max_text_bytes"])

    acquired, unusable, texts, key_to_doc, docs = [], [], {}, {}, []
    stamps: dict = {}
    seen_ids, seen_keys, seen_hashes = set(), set(), set()
    for row, prov in recs:
        did = prov.get("document_id")
        try:
            if not isinstance(did, str) or did in seen_ids:
                raise fulltext_loader.Refused(R_DUP, f"document_id {did!r} missing or duplicated")
            seen_ids.add(did)
            _prule = fulltext_loader.portal_verdict(prov.get("item"), prov, row.get("canonical_url"))
            if _prule:  # PORTAL-RULE-1110 CP5: a stored portal row is never served as evidence (no DB write; counted + logged)
                _PORTAL_REFUSED[0] += 1
                logging.getLogger("vera").warning("fulldoc load: stored row refused R-ACQ-PORTAL rule=%s", _prule[:60])
                unusable.append(_unusable(did, fulltext_loader.R_PORTAL, "stored row comes from a portal host or record"))
                continue
            if prov.get("chunking") != live_chunking:
                raise fulltext_loader.Refused(R_DBCHUNK, "chunking settings at ingest differ from the live settings")
            text = _rebuild(row, by_src.get(row.get("source_id"), []), prov)
            if len(text.encode("utf-8")) > max_bytes:  # same limit as fulltext_loader.load_text
                raise fulltext_loader.Refused(icfg["reason_codes"]["too_large"], "rebuilt text exceeds max_text_bytes")
            item, meta = prov.get("item"), prov.get("meta")
            if not isinstance(item, dict) or not isinstance(meta, dict):
                raise fulltext_loader.Refused(R_DBROW, "recorded item / conversion record missing")
            # R3: the ingest filters, re-applied to the rebuilt text as the configured full-document kind
            # ARX-SECREV-1023 R1: only the acquired path, only rows the arXiv route wrote (arxiv_fulltext.stored_evidence)
            ev = None
            route = None
            if "acquisition" in ek:
                from vera import arxiv_fulltext
                ev = arxiv_fulltext.stored_evidence(prov)
                # ALLDOMAINS-WPG-IMPL-1111: only an acquired row whose acquisition record names the route (written by it)
                route = ((prov.get("acquisition") or {}).get("resolver") if isinstance(prov.get("acquisition"), dict)
                         else None)
            rec = fulltext_loader.prepare_from_parts(item, text, meta, icfg, now=prov.get("retrieved_at"),
                                                     arxiv_evidence=ev, route=route)
            rec["document_id"] = did
            # R3NORM-FIX-1028 D2: a row stored under a legacy spelling of the same source is accepted (no rewrite)
            from vera.url_key import legacy_keys
            if row.get("canonical_url") not in [rec["canonical_url"], *legacy_keys({"doi": item.get("doi"), "url": item.get("landing_url")})]:
                raise fulltext_loader.Refused(R_DBROW, "canonical URL differs from the recorded item")
            if rec["provenance"].get("licence_decision") != "allow":
                raise fulltext_loader.Refused(R_ERR, "licence_decision is not allow")
            if doc_pins is not None:
                dp = doc_pins.get(did)
                got = {"text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                       "licence_id": ((rec["provenance"].get("licence") or {}).get("content") or {}).get("id"),
                       "licence_decision": rec["provenance"].get("licence_decision")}
                if not isinstance(dp, dict) or any(got[k] is None or got[k] != dp.get(k) for k in got):
                    raise _DocPinFail(f"document {did!r} differs from its configured pin")
            base = {k: v for k, v in prov.items() if k not in ek}
            derived = {k: v for k, v in rec["provenance"].items() if k != "stored_text_kind"}
            # FIXR-7DA-1028: rows written before PROVFIX-1028 lack its label-only fields; accept the exact record or
            # the pre-PROVFIX view of it (all-or-nothing; licence id/url/decision and hashes are compared either way)
            if base != derived and base != fulltext_loader.legacy_label_view(derived, meta):
                raise fulltext_loader.Refused(R_DBROW, "stored provenance differs from the re-derived record")
            key = rec["canonical_url"]
            if key in seen_keys:
                raise fulltext_loader.Refused(R_DUP, f"canonical source {key} already acquired")
            if rec["content_hash"] in seen_hashes:
                raise fulltext_loader.Refused(R_DUP, "same full text already acquired under another canonical URL")
        except _DocPinFail as e:
            return _unusable_corpus(cfg, ids, R_DBDOCPIN, str(e))
        except fulltext_loader.InjectionRefused as e:  # S3: drop ONLY this document, pinned or not; fixed-code log
            fulltext_loader.log_injection_drop(e.code)
            unusable.append(_unusable(did, e.code, e))
            continue
        except fulltext_loader.Refused as e:
            if doc_pins is not None:  # pinned run: any per-document failure makes the whole run unusable
                return _unusable_corpus(cfg, ids, R_DBDOCPIN, f"pinned document {did!r} failed verification ({e.code})")
            unusable.append(_unusable(did, e.code, e))
            continue
        except Exception as e:  # noqa: BLE001
            if doc_pins is not None:
                return _unusable_corpus(cfg, ids, R_DBDOCPIN,
                                        f"pinned document {did!r} could not be verified ({type(e).__name__})")
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
                         "text_sha256": rec["content_hash"], "chars": len(text),
                         "evidence_ranges": _evidence_ranges(p.get("sections"), cfg, len(text), text, ex_ranges),
                         "excluded_ranges": ex_ranges})
        # FRESH-AMEND-1023: corpus write time (doc level) + per stored chunk stamps (shared; valid stamps immutable)
        stamps[did] = {"corpus_added_at": row.get("created_at") or prov.get("retrieved_at"),
                       "corpus_stamps": corpus_stamps(row, by_src.get(row.get("source_id"), []), prov)}
    dar = None
    if docs and not build_index:
        # ACQ-REBUILD-PERF-1021 (acquired path only): verification without the throwaway per-run index. The caller
        # (_AcqBuilder.add) indexes the kept documents once and applies the same index-refusal rule there.
        return _with_stamps(stamps, FullDocCorpus(STATUS_OK, acquired, unusable, texts, None, key_to_doc, cfg, docs))
    if docs:
        dar = DarCorpus(_routing_cfg())
        for d in docs:
            res = dar.ingest_document(d)
            if res.get("status") != "indexed":
                if doc_pins is not None:  # pinned run: a retrieval-index refusal makes the whole run unusable
                    return _unusable_corpus(cfg, ids, R_DBDOCPIN,
                                            f"pinned document {key_to_doc.get(d['source_key'])!r} refused by the retrieval index ({res.get('reason')})")
                did = key_to_doc.pop(d["source_key"])
                texts.pop(did, None)
                acquired[:] = [a for a in acquired if a["document_id"] != did]
                unusable.append(_unusable(did, R_ERR, f"retrieval index refused the document ({res.get('reason')})"))
    return _with_stamps(stamps, FullDocCorpus(STATUS_OK, acquired, unusable, texts, dar, key_to_doc, cfg,
                         [d for d in docs if d["source_key"] in key_to_doc]))


_EXTRA_KEYS = {"stored_text_kind", "text_sha256", "chars", "partition", "bundle", "chunking", "seq", "n_documents", "item", "meta"}


# ------------------------------------------------------------------------------------------------ reader (SQL, injected)
def parse_run_id(raw) -> int | None:
    s = str(raw or "").strip()
    return int(s) if re.fullmatch(r"[0-9]{1,18}", s) else None


# FRESH-AMEND-1023: the write-time columns (sources.created_at, evidence_spans.span_id/created_at; migrations 001:132,
# 003:17) are appended at the END of each SELECT. If a live schema lacks them, the read falls back ONCE to the previous
# column list (no write-time stamp -> corpus_added_at from provenance.retrieved_at, else unknown) instead of failing.
# ONLY on SQLSTATE 42703 (undefined column); every other error is re-raised (FRESH-AMEND-1023 review D3).
def _legacy(sql: str) -> str:
    return (sql.replace(", s.created_at ", " ").replace(", e.span_id, e.created_at ", " "))


UNDEFINED_COLUMN = "42703"  # PostgreSQL SQLSTATE undefined_column (the only fallback case; vera_vjay_ro has SELECT on all)


def _sqlstate(exc) -> str | None:
    for e in (exc, getattr(exc, "orig", None)):
        for k in ("pgcode", "sqlstate"):
            v = getattr(e, k, None)
            if isinstance(v, str) and v:
                return v
    return None


def _select_compat(conn, cur, sql: str, params: dict) -> list:
    try:
        return _select(cur, sql, params)
    except Exception as exc:  # noqa: BLE001
        legacy = _legacy(sql)
        if legacy == sql or _sqlstate(exc) != UNDEFINED_COLUMN:
            raise  # only a missing column falls back; timeouts, locks, privilege errors etc. are NOT masked
        logging.getLogger("vera").warning("write-time columns unavailable (%s); legacy column list used",
                                          type(exc).__name__)
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return _select(cur, legacy, params)


def _select(cur, sql: str, params: dict) -> list:
    if not sql.lstrip().upper().startswith("SELECT"):
        raise RuntimeError("only SELECT is allowed")
    cur.execute(sql, params)
    return list(cur.fetchall())


def fetch_rows(connect, run_id: int) -> tuple:
    """One read-only round trip pair. `connect()` returns a DB-API connection (closed here)."""
    conn = connect()
    try:
        cur = conn.cursor()
        try:
            srows = _select_compat(conn, cur, SQL_SOURCES, {"run_id": int(run_id), "kind": FULLDOC_CHUNKS_KIND})
            sources = [{"source_id": r[0], "canonical_url": r[1], "content": r[2], "content_hash": r[3],
                        "provenance": r[4], **_at(r, 5)} for r in srows]
            ids = [s["source_id"] for s in sources]
            prow = _select_compat(conn, cur, SQL_SPANS, {"ids": ids}) if ids else []
            spans = [{"source_id": r[0], "start_index": r[1], "end_index": r[2], "text": r[3],
                      **({"span_id": r[4]} if len(r) > 4 else {}), **_at(r, 5)} for r in prow]
        finally:
            try:
                cur.close()
            except Exception:  # noqa: BLE001
                pass
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
    return sources, spans


def load_db(connect, run_id, cfg: dict, ingest_cfg: dict | None = None) -> FullDocCorpus:
    """Never raises. A bad run id or any DB failure => corpus_unusable (class name only, never the URL or message)."""
    rid = parse_run_id(run_id)
    if rid is None:
        return _unusable_corpus(cfg, [], R_DBRUN, "run id is not a positive integer")
    try:
        sources, spans = _fetch_with_deadline(connect, rid, deadline_s(cfg, "db_load_deadline_s", 30.0))
    except Exception as e:  # noqa: BLE001
        return _unusable_corpus(cfg, [], R_DBCONN, f"rows could not be read ({type(e).__name__})")
    return load_from_rows(sources, spans, cfg, ingest_cfg)


class LoadTimeout(Exception):
    """DIAG-1021: the DB read did not finish within its total deadline (connect, DNS, TLS and queries together)."""


def deadline_s(cfg: dict, key: str, default: float) -> float:
    """A positive number of seconds from config/fulldoc_answer.json `db_bounds` (missing/invalid -> default)."""
    v = ((cfg or {}).get("db_bounds") or {}).get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else float(default)


def _fetch_with_deadline(connect, rid: int, limit_s: float) -> tuple:
    """DIAG-1021: run fetch_rows in a daemon worker and give up after limit_s, so a hung connect/DNS/TLS/query never
    blocks the calling thread past the deadline. The abandoned worker ends on its own (statement_timeout, keepalives)."""
    box: dict = {}

    def work():
        try:
            box["ok"] = fetch_rows(connect, rid)
        except BaseException as e:  # noqa: BLE001
            box["err"] = e
    t = threading.Thread(target=work, name="fulldoc-db-load", daemon=True)
    t.start()
    t.join(limit_s)
    if t.is_alive():
        raise LoadTimeout()
    if "err" in box:
        raise box["err"]
    return box["ok"]


def default_connect(cfg: dict, env):
    """Factory from the env var NAMED by config (default VERA_DB_URL_RO, the read-only account) or None when unset.
    Lazy sqlalchemy import; not exercised by the offline tests (they inject a fake)."""
    url = (env.get(str(cfg.get("db_url_env") or DEFAULT_URL_ENV)) or "").strip()
    if not url:
        return None
    ms = int(cfg.get("db_statement_timeout_ms") or 15000)

    def connect():
        from sqlalchemy import create_engine
        eng = create_engine(url, connect_args={"connect_timeout": 5, "options": f"-c statement_timeout={ms}",
                                               "keepalives": 1, "keepalives_idle": 10, "keepalives_interval": 5, "keepalives_count": 3},  # DIAG-1021: TCP keepalives
                            hide_parameters=True)
        raw = eng.raw_connection()
        orig = raw.close

        def _close():
            try:
                orig()
            finally:
                eng.dispose()
        raw.close = _close  # noqa: B010
        return raw
    return connect


# short failure cache so a down DB is not hit on every /ask
_FAIL: dict = {}
_LOCK = threading.Lock()
FAIL_TTL_S = 30.0


_INFLIGHT: dict = {}  # WED-P161: key -> {"done": Event, "result": corpus | None}; single-flight per key


def cached_load(connect, run_id, cfg, cache: dict, now=time.monotonic) -> FullDocCorpus:
    """Single-flight (WED-P161): concurrent first callers for one key share ONE load_db; followers wait on the leader's
    Event (no lock held while waiting) and take its result. Cache and the short failure cache are unchanged."""
    key = ("db", str(run_id).strip(), json.dumps(cfg.get("pins") or {}, sort_keys=True))
    while True:
        with _LOCK:
            hit = cache.get(key)
            if hit is not None:
                return hit
            f = _FAIL.get(key)
            if f and now() < f[0]:
                return f[1]
            flight = _INFLIGHT.get(key)
            leader = flight is None
            if leader:
                flight = _INFLIGHT[key] = {"done": threading.Event(), "result": None}
        if not leader:
            if not flight["done"].wait(deadline_s(cfg, "single_flight_wait_s", 60.0)):  # DIAG-1021: bounded wait
                return _unusable_corpus(cfg, [], R_DBCONN, "rows could not be read (LoadTimeout)")  # not cached
            if flight["result"] is not None:
                return flight["result"]
            continue  # the leader died without a result: retry (as leader or follower of a new flight)
        try:
            c = load_db(connect, run_id, cfg)
            with _LOCK:
                if c.status == STATUS_OK and c.acquired:
                    cache[key] = c
                else:
                    _FAIL[key] = (now() + FAIL_TTL_S, c)
            flight["result"] = c
            return c
        finally:
            with _LOCK:
                if _INFLIGHT.get(key) is flight:
                    del _INFLIGHT[key]
            flight["done"].set()


def reset_for_tests() -> None:
    with _LOCK:
        _FAIL.clear()


# ------------------------------------------------------------------------------------------------ WED-ACQ2 acquired documents
# Automatically acquired documents (vera/auto_acquire.py) are stored one document per run, marked in provenance
# (`acquisition.marker`), and served as a SEPARATE corpus next to the pinned base run (plan s11 V6): the base run's
# validation and pins are untouched. Each acquired run is verified alone by the same _load_rows with a pins-free cfg,
# the acquisition label overlay and a wider extra-keys set; any failure makes that ONE document unusable.
ACQ_EXTRA_KEYS = frozenset(_EXTRA_KEYS | {"acquisition", "isbn", "isbn10"})
R_ACQINJECT = "R-ACQ-INJECT"
# OD-1109-NOCAP (owner 2026-11-09): no cap on stored documents or chunks served to internal search. The former serve
# caps (bounds.served_acquired_max / served_acquired_max_bytes / served_scan_max) and their drop code R-ACQ-SERVECAP
# are removed; the ~30-document soft ceiling applies only to acquisition (bounds.max_docs_per_prompt, stop ceiling31).

SQL_ACQ_FILTER = (f"FROM {SCHEMA}.sources s JOIN {SCHEMA}.canonical_sources cs ON cs.canonical_source_id = s.canonical_source_id "
                  f"JOIN {SCHEMA}.candidates d ON d.candidate_id = s.candidate_id "
                  f"WHERE s.provenance->>'stored_text_kind' = %(kind)s "
                  f"AND s.provenance->'acquisition'->>'marker' = %(marker)s AND d.run_id <> %(base)s")
SQL_ACQ_SOURCES = (f"SELECT d.run_id, s.source_id, cs.canonical_url, s.content, s.content_hash, s.provenance, s.created_at "
                   f"{SQL_ACQ_FILTER} ORDER BY s.source_id DESC")
SQL_ACQ_MAX = f"SELECT max(s.source_id), count(*) {SQL_ACQ_FILTER}"
# DAR-INLINE-1021-R2: the same acquired-row filter, restricted to given canonical URLs (inline reuse of stored documents)
SQL_ACQ_SOURCES_FOR = (f"SELECT d.run_id, s.source_id, cs.canonical_url, s.content, s.content_hash, s.provenance, s.created_at "
                       f"{SQL_ACQ_FILTER} AND cs.canonical_url = ANY(%(cus)s) ORDER BY s.source_id DESC LIMIT %(lim)s")
# ACQ S4-0 L9: light index (no content) to order what is served per micro-topic, and a by-id fetch of the rows.
# OD-1109-NOCAP: no LIMIT (every stored acquired row is indexed and served).
SQL_ACQ_INDEX = (f"SELECT s.source_id, d.run_id, s.provenance {SQL_ACQ_FILTER} "
                 f"ORDER BY s.source_id DESC")
SQL_ACQ_SOURCES_BY_ID = (f"SELECT d.run_id, s.source_id, cs.canonical_url, s.content, s.content_hash, s.provenance "
                         f"{SQL_ACQ_FILTER} AND s.source_id = ANY(%(sids)s) ORDER BY s.source_id DESC")
# D1: acquired rows store content '' (text lives in evidence_spans), so the read-time key needs the first span's text.
SQL_ACQ_FIRST_SPANS = (f"SELECT DISTINCT ON (e.source_id) e.source_id, left(e.text, %(win)s) FROM {SCHEMA}.evidence_spans e "
                       f"WHERE e.source_id = ANY(%(ids)s) ORDER BY e.source_id, e.start_index")


def _first_span_texts(spans) -> dict:
    """source_id -> text of the document's first span (lowest start_index)."""
    first: dict = {}
    for sp in spans or ():
        sid = sp.get("source_id")
        if sid not in first or int(sp.get("start_index") or 0) < first[sid][0]:
            first[sid] = (int(sp.get("start_index") or 0), sp.get("text"))
    return {k: v[1] for k, v in first.items()}


def select_served(index_rows, per_topic_max, cfg: dict | None = None) -> list:
    """Pure. index_rows: dicts with source_id and provenance (newest-first not required). Returns source_ids:
    round-robin across micro-topic keys, topics ordered by newest member first, newest first within a topic, each
    topic at most per_topic_max, a row with no key is its own topic. Provenance only.
    per_topic_max None (FLOOR6-1028, owner 2026-10-28: 6 is a floor, not a ceiling): no per-topic ceiling; the
    round-robin is only an ORDER (every topic gets depth d before any gets d+1) and every row is returned.
    OD-1109-NOCAP (owner 2026-11-09): there is no total cap; the former total_max argument is removed."""
    from vera import micro_topic

    per = None if per_topic_max is None else int(per_topic_max)
    if per is not None and per <= 0:
        return []
    topics: dict = {}
    for r in sorted(index_rows, key=lambda r: -int(r.get("source_id") or 0)):
        sid = r.get("source_id")
        k = micro_topic.key_id(micro_topic.key_for_provenance(micro_topic._row_prov(r), r.get("content") or None, cfg or {}))
        topics.setdefault(k if k else f"\0unkeyed:{sid}", []).append(sid)  # dict order = newest member first
    queues = [v if per is None else v[:per] for v in topics.values()]
    out: list = []
    depth = 0
    while any(depth < len(q) for q in queues):
        for q in queues:
            if depth < len(q):
                out.append(q[depth])
        depth += 1
    return out


def _pos_int(v) -> int | None:
    return int(v) if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None


def _per_topic(acq_cfg) -> int | None:
    return _pos_int(((acq_cfg or {}).get("bounds") or {}).get("served_per_topic_max"))


def _topic_balanced(acq_cfg) -> bool:
    """FLOOR6-1028: served_topic_balanced true = topic-balanced ORDER with no per-topic ceiling (when
    served_per_topic_max is null). false/absent with a null ceiling = the legacy newest-N order."""
    return ((acq_cfg or {}).get("bounds") or {}).get("served_topic_balanced") is True


def _per_topic_mode(acq_cfg):
    """(use per-topic selection?, ceiling or None)."""
    ptm = _per_topic(acq_cfg)
    return (ptm is not None or _topic_balanced(acq_cfg)), ptm


def acquired_ids_per_topic(conn, cur, base_run_id, marker, per_topic_max, cfg, allow_ids=None) -> list:
    """allow_ids (H1B-PROMOTE-1107, pins mode on): only rows whose provenance document_id is pinned are served.
    A pre-selection only: the reader still verifies each row and compares it with its pin. OD-1109-NOCAP: every
    stored acquired row is indexed (no scan or total cap)."""
    rows = _select(cur, SQL_ACQ_INDEX, {"kind": FULLDOC_CHUNKS_KIND, "marker": str(marker),
                                                      "base": int(base_run_id)})
    idx = [{"source_id": r[0], "run_id": r[1], "provenance": r[2]} for r in rows]
    if allow_ids is not None:
        from vera import acquired_pins
        idx = [i for i in idx if acquired_pins.row_document_id(i) in allow_ids]
    if idx:  # D1: first-span text per document (read-only, parameterized SELECT) so keys come from title + text
        from vera import micro_topic
        win = int((cfg or {}).get("text_window_chars") or micro_topic.DEFAULT_WINDOW)
        try:
            ft = {r[0]: r[1] for r in _select(cur, SQL_ACQ_FIRST_SPANS,
                                                      {"ids": [int(i["source_id"]) for i in idx], "win": win})}
        except Exception:  # noqa: BLE001  fail soft to title-only keys
            ft = {}
            try:  # a failed SELECT aborts a non-autocommit transaction; clear it so later SQL works
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
        for i in idx:
            i["content"] = ft.get(i["source_id"]) or ""
    return select_served(idx, per_topic_max, cfg)


def acquired_signature(connect, base_run_id, marker: str) -> tuple:
    """(max source_id, count) of marked acquired rows: the cheap refresh probe (one static SELECT, read-only)."""
    conn = connect()
    try:
        cur = conn.cursor()
        try:
            r = _select(cur, SQL_ACQ_MAX, {"kind": FULLDOC_CHUNKS_KIND, "marker": str(marker), "base": int(base_run_id)})
        finally:
            try:
                cur.close()
            except Exception:  # noqa: BLE001
                pass
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
    row = r[0] if r else (None, 0)
    return (row[0], int(row[1] or 0))


def fetch_acquired_rows(connect, base_run_id, marker: str, limit: int | None = None, canonical_urls=None,
                        per_topic_max=None, micro_cfg=None, topic_balanced=False, allow_ids=None) -> tuple:
    """Read-only. canonical_urls (DAR-INLINE-1021-R2): only the acquired rows of those canonical URLs (inline reuse);
    `limit` bounds that by-URL lookup only. Without canonical_urls every stored acquired row is returned
    (OD-1109-NOCAP: no serve cap): per_topic_max (L9) / topic_balanced (FLOOR6-1028) choose the per-micro-topic
    order, otherwise newest first."""
    if canonical_urls is not None and not canonical_urls:
        return [], []
    conn = connect()
    try:
        cur = conn.cursor()
        try:
            params = {"kind": FULLDOC_CHUNKS_KIND, "marker": str(marker), "base": int(base_run_id)}
            if canonical_urls is not None:
                params["cus"] = [str(u) for u in canonical_urls]
                params["lim"] = int(limit if limit is not None else len(params["cus"]))
            if canonical_urls is None and (per_topic_max or topic_balanced):
                sids = acquired_ids_per_topic(conn, cur, base_run_id, marker, per_topic_max, micro_cfg or {},
                                              allow_ids=allow_ids)
                params["sids"] = [int(x) for x in sids]
                srows = _select_compat(conn, cur, SQL_ACQ_SOURCES_BY_ID, params) if sids else []
            else:
                srows = _select_compat(conn, cur, SQL_ACQ_SOURCES if canonical_urls is None else SQL_ACQ_SOURCES_FOR,
                                       params)
            sources = [{"run_id": r[0], "source_id": r[1], "canonical_url": r[2], "content": r[3], "content_hash": r[4],
                        "provenance": r[5], **_at(r, 6)} for r in srows]
            ids = [x["source_id"] for x in sources]
            prow = _select_compat(conn, cur, SQL_SPANS, {"ids": ids}) if ids else []
            spans = [{"source_id": r[0], "start_index": r[1], "end_index": r[2], "text": r[3],
                      **({"span_id": r[4]} if len(r) > 4 else {}), **_at(r, 5)} for r in prow]
        finally:
            try:
                cur.close()
            except Exception:  # noqa: BLE001
                pass
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
    return sources, spans


def load_acquired_rows(sources: list, spans: list, cfg: dict, acq_cfg: dict, ingest_cfg: dict | None = None) -> FullDocCorpus:
    """One FullDocCorpus of the acquired documents. Never raises. Rows are processed newest first (source_id DESC),
    in the per-micro-topic order when configured. A document is served only if its run verifies alone.
    OD-1109-NOCAP: no document-count or byte cap; every row that passes the per-document checks is served."""
    try:
        b = _AcqBuilder(cfg, acq_cfg, ingest_cfg)
        b.add(list(sources or ()), list(spans or ()), order="legacy")
        return b.finish()
    except Exception as e:  # noqa: BLE001  fail closed
        return _unusable_corpus(cfg, [], R_ERR, f"acquired rows could not be loaded ({type(e).__name__})")


def _acquired_open_cfg(cfg: dict) -> dict:
    """The pins-free config for acquired runs (they are not part of the bundle). CTL-PIN exception site (formerly in
    _load_acquired; moved unchanged by NOCAP-1109)."""
    return {k: v for k, v in cfg.items() if k != "pins"}


class _AcqBuilder:
    """Builds the acquired FullDocCorpus from batches of stored rows (NOCAP-1109). Each batch is verified (one run at a
    time, the same checks as before) and its accepted documents are indexed at once, so a streamed build holds only
    one batch of raw rows. Duplicates are refused against everything accepted so far. With `base`, the builder
    continues an existing corpus (append): its state is copied and its index forked, never rebuilt."""

    def __init__(self, cfg: dict, acq_cfg: dict, ingest_cfg: dict | None = None, base: FullDocCorpus | None = None):
        from vera import acq_config
        from vera import acquired_pins as ap
        self.cfg, self.acq_cfg = cfg, acq_cfg
        self.ingest_cfg = ingest_cfg if ingest_cfg is not None else acq_config.ingest_cfg(fulltext_loader.load_config(),
                                                                                          acq_cfg)
        self.open_cfg = _acquired_open_cfg(cfg)
        # H1B-PROMOTE-1107: the base pins do not cover acquired runs; the manifest config/acquired_pins.json does
        self.pm = ap.load()
        self.pmode = self.pm["mode"]
        st = getattr(base, "_acq_state", None) if base is not None else None
        if st is not None:
            self.acquired, self.unusable = list(base.acquired), list(base.unusable)
            self.texts, self.key_to_doc, self.docs = dict(base._texts), dict(base._key_to_doc), list(base._docs_for_dar)
            self.seen_keys, self.seen_hashes = set(st["seen_keys"]), set(st["seen_hashes"])
            self.stamps = dict(base.stamps)
            self.pin_candidates = dict(getattr(base, "pin_candidates", None) or {})
            self.report = dict(st["report"])
            self.dar = base._dar.fork() if base._dar is not None else None
        else:
            self.acquired, self.unusable, self.texts, self.key_to_doc, self.docs = [], [], {}, {}, []
            self.seen_keys, self.seen_hashes = set(), set()
            self.stamps, self.pin_candidates = {}, {}
            self.report = {"mode": self.pmode, "pinned": 0, "unpinned": 0, "mismatch": 0, "ephemeral": 0,
                           "manifest_error": self.pm["error"]}
            self.dar = None
        self.order = None

    def discard(self) -> None:
        if self.dar is not None and getattr(self.dar, "_fork_base", None) is not None:
            self.dar.discard()

    def add(self, sources: list, spans: list, order: str) -> None:
        """order "legacy": newest first (source_id DESC), per-micro-topic order when configured (the in-memory and
        test callers). order "oldest": source_id ascending (the stored-corpus path), so appending newer rows later
        gives exactly the corpus a full build would."""
        from vera import acq_config
        from vera import acquired_pins as ap
        self.order = order
        acq_cfg, pm, pmode = self.acq_cfg, self.pm, self.pmode
        ek = ap.EPHEMERAL_KEY
        by_src: dict = {}
        for sp in spans:
            by_src.setdefault(sp.get("source_id"), []).append(sp)
        sign = 1 if order == "oldest" else -1
        ordered = sorted(sources, key=lambda r: sign * int(r.get("source_id") or 0))
        rest: list = []
        if pmode == "on":  # pinned stored rows and in-memory inline rows are served; the rest only meet quarantine
            pids = ap.pinned_ids(pm)
            keep = [r for r in ordered if r.get(ek) is True or ap.row_document_id(r) in pids]
            rest = [r for r in ordered if not (r.get(ek) is True or ap.row_document_id(r) in pids)]
            ordered = keep
        use_pt, ptm = _per_topic_mode(acq_cfg)
        if use_pt and (order == "legacy" or ptm is not None):  # L9 / FLOOR6 order; a numeric ceiling is rollback only
            first = _first_span_texts(spans)  # D1: real rows have content '' (text is in the spans)
            pick = select_served([{"source_id": r.get("source_id"), "provenance": r.get("provenance"),
                                   "content": r.get("content") or first.get(r.get("source_id"))} for r in ordered], ptm,
                                 acq_cfg.get("micro_topic") or {})
            rank = {sid: i for i, sid in enumerate(pick)}
            ordered = [r for r in ordered if r.get("source_id") in rank]
            if order == "legacy":
                ordered.sort(key=lambda r: rank[r.get("source_id")])
        runs: dict = {}
        for row in ordered + rest:  # H1B review: an in-memory row never shares a run group with a stored row
            runs.setdefault((row.get(ek) is True, row.get("run_id")), []).append(row)
        batch_docs = []
        for (is_eph, rid), rows in runs.items():
            rspans = [sp for r in rows for sp in by_src.get(r.get("source_id"), [])]
            bare = [{k: v for k, v in r.items() if k not in ("run_id", ek)} for r in rows]
            c = load_from_rows(bare, rspans, self.open_cfg, self.ingest_cfg, extra_keys=ACQ_EXTRA_KEYS, build_index=False)
            self.unusable.extend(c.unusable)
            self.stamps.update(getattr(c, "stamps", None) or {})
            provs, eph = {}, set()
            for r in rows:
                try:
                    p = _prov(r)
                except Exception:  # noqa: BLE001
                    continue
                if (p.get("acquisition") or {}).get("marker") != acq_cfg.get("marker"):
                    continue
                provs[p.get("document_id")] = p
                if is_eph:
                    eph.add(p.get("document_id"))
            for a in c.acquired:
                did = a["document_id"]
                text = c._texts[did]
                key = a["canonical_source_id"]
                if did not in provs:
                    self.unusable.append(_unusable(did, R_DBROW, "acquired row carries no matching acquisition marker"))
                    continue
                try:  # WED-P17-VER v2: every JATS licence element, re-decided at read (filters apply to all ingest)
                    from vera import auto_acquire
                    lic = ((provs[did].get("licence") or {}).get("content") or {}).get("id")
                    elements = (provs[did].get("meta") or {}).get("jats_licences")
                    if not isinstance(elements, list):
                        raise fulltext_loader.Refused(R_DBROW, "acquired row has no licence-element record")
                    if (provs[did].get("acquisition") or {}).get("resolver") == "original_html":
                        # ALLDOMAINS-WPG-IMPL-1111: a US-federal page has no JATS licence element; the declared licence
                        # must be exactly the owner-approved federal one and the element list empty (anything else holds)
                        if elements or lic != "LicenseRef-us-government-work":
                            raise fulltext_loader.Refused(self.ingest_cfg["reason_codes"]["licence_hold"],
                                                          "original_html row with an unexpected licence record: held")
                    else:
                        auto_acquire.licence_elements_decision(elements, lic, self.ingest_cfg)
                except fulltext_loader.Refused as e:
                    self.unusable.append(_unusable(did, e.code, e))
                    continue
                if acq_config.injection_refused(text, acq_cfg):  # re-applied at read (filters apply to all ingest)
                    self.unusable.append(_unusable(did, R_ACQINJECT, "instruction-like text found by the injection scan"))
                    continue
                if did in eph:  # fetched within this request, verified in memory, never read back from the DB
                    self.report["ephemeral"] += 1
                else:
                    got = ap.pin_entry(text, provs[did], key, a.get("licence"), a.get("licence_decision"))
                    self.pin_candidates[did] = {**got, "acquired_on": str(
                        (provs[did].get("acquisition") or {}).get("acquired_at") or "")[:10] or None}
                    if pmode != "off":
                        if pm["error"] is not None:
                            if pmode == "on":
                                self.unusable.append(_unusable(did, ap.R_PINCFG, "acquired pins manifest unusable"))
                                continue
                        else:
                            st = ap.status(pm, did, got)
                            self.report[st] += 1
                            if st == "mismatch":
                                self.unusable.append(_unusable(did, ap.R_PINMISMATCH, "stored document differs from its pin"))
                                continue
                            if st == "unpinned" and pmode == "on":
                                self.unusable.append(_unusable(did, ap.R_QUARANTINE, "stored document not promoted (no pin)"))
                                continue
                if key in self.seen_keys or a["text_sha256"] in self.seen_hashes or did in self.texts:
                    self.unusable.append(_unusable(did, R_DUP, "acquired document duplicated"))
                    continue
                self.seen_keys.add(key)
                self.seen_hashes.add(a["text_sha256"])
                self.texts[did] = text
                self.key_to_doc[key] = did
                p = provs[did]
                if p.get("isbn"):
                    a = {**a, "isbn": p["isbn"]}
                self.acquired.append(a)
                for d in c._docs_for_dar:
                    if d["source_key"] == key:
                        batch_docs.append({**d, **({"isbn": p["isbn"]} if p.get("isbn") else {})})
        if not batch_docs:
            return
        if self.dar is None:
            self.dar = DarCorpus(_routing_cfg())
        for d in batch_docs:  # the index-refusal rule (ACQ-REBUILD-PERF-1021): refused => UNUSABLE
            res = self.dar.ingest_document(d)
            if res.get("status") != "indexed":
                did = self.key_to_doc.pop(d["source_key"], None)
                self.texts.pop(did, None)
                self.acquired[:] = [a for a in self.acquired if a["document_id"] != did]
                self.unusable.append(_unusable(did, R_ERR, f"retrieval index refused the document ({res.get('reason')})"))
                continue
            self.docs.append(d)

    def finish(self) -> FullDocCorpus:
        cfg = self.cfg
        if not self.acquired:
            out = FullDocCorpus(STATUS_UNUSABLE, [], self.unusable or [_unusable(None, R_DBEMPTY, "no acquired documents")],
                                {}, None, {}, cfg)
        else:
            out = _with_stamps(self.stamps, FullDocCorpus(STATUS_OK, self.acquired, self.unusable, self.texts, self.dar,
                                                          self.key_to_doc, cfg, self.docs))
        out.source_kind = "db"
        out._acq_state = {"seen_keys": self.seen_keys, "seen_hashes": self.seen_hashes, "report": self.report,
                          "order": self.order}
        return _pin_attrs(out, self.pin_candidates, dict(self.report))


def _pin_attrs(out, pin_candidates: dict, report: dict):
    """H1B-PROMOTE-1107: hash-only pin candidates (every stored document that passed the non-pin checks) and, unless
    mode is off, the pinned / unpinned / mismatch / ephemeral counts."""
    out.pin_candidates = pin_candidates
    if report.get("mode") != "off":
        out.pin_report = report
    return out


# NOCAP-1109: streamed reads of the stored acquired rows. The id list is read first (no content); rows and spans are then
# read and verified ACQ_BATCH documents at a time, so raw rows never pile up. SQL_ACQ_INDEX_AFTER lists only rows newer
# than a source_id (the append refresh) together with the count of ALL marked rows, in ONE statement (one snapshot,
# F2): the append test then cannot be confused by a row committed between the refresh probe and the listing.
ACQ_BATCH = 25
SQL_ACQ_INDEX_AFTER = (f"SELECT m.source_id, m.run_id, m.provenance, m.n_all FROM (SELECT s.source_id, d.run_id, "
                       f"s.provenance, count(*) OVER () AS n_all {SQL_ACQ_FILTER}) m WHERE m.source_id > %(after)s "
                       f"ORDER BY m.source_id DESC")


def _acq_ids(conn, cur, rid: int, acq_cfg: dict, after=None) -> tuple[list, tuple | None, int]:
    """(source ids to serve in ascending order, (max source_id, count) of ALL marked rows in the same snapshot as the
    listing or None when unknown, number of marked rows listed before the pins pre-selection). A numeric served_per_topic_max (rollback only) selects per micro-topic over the
    whole set. With `after`, only newer rows are listed; the count still covers all marked rows."""
    from vera import acquired_pins
    marker = acq_cfg["marker"]
    use_pt, ptm = _per_topic_mode(acq_cfg)
    pm = acquired_pins.load()
    allow = acquired_pins.pinned_ids(pm) if pm["mode"] == "on" else None  # H1B-PROMOTE-1107: only pinned rows
    if after is None and use_pt and ptm is not None:
        ids = acquired_ids_per_topic(conn, cur, rid, marker, ptm, acq_cfg.get("micro_topic") or {}, allow_ids=allow)
        return sorted(int(x) for x in ids), None, -1
    params = {"kind": FULLDOC_CHUNKS_KIND, "marker": str(marker), "base": int(rid)}
    if after is not None:
        params["after"] = int(after)
    rows = _select(cur, SQL_ACQ_INDEX if after is None else SQL_ACQ_INDEX_AFTER, params)
    idx = [{"source_id": r[0], "run_id": r[1], "provenance": r[2]} for r in rows]
    if after is None:
        snap = (max((int(i["source_id"]) for i in idx), default=None), len(idx))
    else:
        snap = (max(int(i["source_id"]) for i in idx), int(rows[0][3])) if rows else None
    n_listed = len(idx)
    if allow is not None:
        idx = [i for i in idx if acquired_pins.row_document_id(i) in allow]
    return sorted(int(i["source_id"]) for i in idx), snap, n_listed


def _acq_batches(conn, cur, rid: int, acq_cfg: dict, sids: list):
    params = {"kind": FULLDOC_CHUNKS_KIND, "marker": str(acq_cfg["marker"]), "base": int(rid)}
    for i in range(0, len(sids), ACQ_BATCH):
        part = sids[i:i + ACQ_BATCH]
        srows = _select_compat(conn, cur, SQL_ACQ_SOURCES_BY_ID, {**params, "sids": part})
        sources = [{"run_id": r[0], "source_id": r[1], "canonical_url": r[2], "content": r[3], "content_hash": r[4],
                    "provenance": r[5], **_at(r, 6)} for r in srows]
        ids = [x["source_id"] for x in sources]
        prow = _select_compat(conn, cur, SQL_SPANS, {"ids": ids}) if ids else []
        spans = [{"source_id": r[0], "start_index": r[1], "end_index": r[2], "text": r[3],
                  **({"span_id": r[4]} if len(r) > 4 else {}), **_at(r, 5)} for r in prow]
        yield sources, spans


def _close(conn, cur) -> None:
    for x in (cur, conn):
        try:
            x.close()
        except Exception:  # noqa: BLE001
            pass


def _log_pins(c) -> None:
    rep = getattr(c, "pin_report", None)
    if rep:  # counts only (no ids, titles or URLs)
        logging.getLogger("vera").info("acquired pins: mode=%s pinned=%d unpinned=%d mismatch=%d manifest_error=%s",
                                       rep["mode"], rep["pinned"], rep["unpinned"], rep["mismatch"],
                                       "yes" if rep["manifest_error"] else "no")


def load_acquired_db(connect, base_run_id, cfg: dict, acq_cfg: dict) -> FullDocCorpus:
    """Never raises: any DB failure => an unusable (empty) acquired corpus; the base corpus is unaffected.
    OD-1109-NOCAP: every stored acquired row is read (no document, byte or scan cap), streamed in batches, oldest
    first, so a later append (extend_acquired_db) reaches exactly the corpus a full build would."""
    rid = parse_run_id(base_run_id)
    if rid is None:
        return _unusable_corpus(cfg, [], R_DBRUN, "run id is not a positive integer")
    try:
        conn = connect()
        cur = conn.cursor()
    except Exception as e:  # noqa: BLE001
        return _unusable_corpus(cfg, [], R_DBCONN, f"acquired rows could not be read ({type(e).__name__})")
    try:
        b = _AcqBuilder(cfg, acq_cfg)
        b.order = "oldest"
        sids, snap, _ = _acq_ids(conn, cur, rid, acq_cfg)
        for sources, spans in _acq_batches(conn, cur, rid, acq_cfg, sids):
            b.add(sources, spans, order="oldest")
    except Exception as e:  # noqa: BLE001
        return _unusable_corpus(cfg, [], R_DBCONN, f"acquired rows could not be read ({type(e).__name__})")
    finally:
        _close(conn, cur)
    try:
        c = b.finish()
    except Exception as e:  # noqa: BLE001  fail closed
        return _unusable_corpus(cfg, [], R_ERR, f"acquired rows could not be loaded ({type(e).__name__})")
    if snap is not None:
        c._acq_sig = snap  # the (max source_id, count) this corpus was built from (F2)
    _log_pins(c)
    return c


def extend_acquired_db(connect, base_run_id, cfg: dict, acq_cfg: dict, corpus, old_sig, new_sig):
    """NOCAP-1109 append refresh. When the change since `old_sig` (the signature `corpus` was built at) is a pure append
    (in ONE snapshot: the count of all marked rows equals old_count plus the rows newer than the old maximum
    source_id), verify and index ONLY those newer rows on a fork of `corpus` (the serving corpus is never modified) and
    return the new corpus, carrying in `_acq_sig` the signature of that snapshot; otherwise None (the caller then does
    a full rebuild). `new_sig` (the refresh probe) only triggers the attempt. Never raises; a failure discards the fork
    and returns None. A caller that abandons the returned corpus before serving it must call its _dar.discard()."""
    rid = parse_run_id(base_run_id)
    st = getattr(corpus, "_acq_state", None)
    if rid is None or st is None or st.get("order") != "oldest" or corpus.status != STATUS_OK or corpus._dar is None:
        return None
    try:
        old_max, old_n = old_sig
        new_max, new_n = new_sig
        if old_max is None or new_max is None or new_max <= old_max or new_n <= old_n:
            return None
    except Exception:  # noqa: BLE001
        return None
    use_pt, ptm = _per_topic_mode(acq_cfg)
    if ptm is not None:  # a numeric per-topic ceiling selects over the whole set: not appendable
        return None
    b = None
    try:
        conn = connect()
        cur = conn.cursor()
    except Exception:  # noqa: BLE001
        return None
    try:
        sids, snap, n_listed = _acq_ids(conn, cur, rid, acq_cfg, after=old_max)
        if snap is None or n_listed <= 0 or snap[1] != old_n + n_listed:
            return None  # rows were also removed or changed (same snapshot): not a pure append
        b = _AcqBuilder(cfg, acq_cfg, base=corpus)
        b.order = "oldest"
        for sources, spans in _acq_batches(conn, cur, rid, acq_cfg, sids):
            b.add(sources, spans, order="oldest")
        c = b.finish()
    except Exception:  # noqa: BLE001
        if b is not None:
            b.discard()
        return None
    finally:
        _close(conn, cur)
    if c.status != STATUS_OK:
        b.discard()
        return None
    c._acq_sig = snap
    _log_pins(c)
    return c
