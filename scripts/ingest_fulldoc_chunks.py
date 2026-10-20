#!/usr/bin/env python
"""AE-1020/DBC-1: plan (and, only with explicit owner authorization, write) the chunk-stored DAR-9 full documents.

Owner D-4: full texts stay local; their chunks go into the project's PostgreSQL chunk table (evidence_spans).
For each document of the LOCAL bundle this builds (vera.fulldoc_db.build_rows): one `sources` row (stored_text_kind
"fulldoc_chunks", content '' , content_hash = sha256 of the empty content so content_hash == sha256(content) holds; the
sha256 of the full text lives in provenance.text_sha256 (vet R4)) and a NON-overlapping contiguous partition of the text as
`evidence_spans` rows (evidence_type fulldoc_partition; rebuild = concatenation, vet R1).

DRY-RUN IS THE DEFAULT: no engine, no DB, no env read except through --db-url-env checks on --write. It prints counts only
(never text) and writes a plan file (ids, offsets, hashes; no text) under R/work/fd-1020/ingest-plan/.

--write is refused unless BOTH `--i-have-owner-authorization <token>` is given AND the env var named by --db-url-env
(default VERA_DB_URL_WO, a NAME only) is set. It additionally requires --new-run, refuses under a test runner, and installs
connect_guard("production"). It was NOT exercised in AE-1020/DBC-1 and needs the owner's same-turn authorization naming the
account and the row caps. A token alone never authorizes anything: the owner's decision does.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vera import fulldoc_corpus, fulldoc_db  # noqa: E402
from vera import fulltext_loader as fl  # noqa: E402

PLAN_DIR = ROOT.parent / "work" / "fd-1020" / "ingest-plan"
QUESTION = "AE-1020 fulldoc chunk-stored DAR-9 corpus (partition rows; no question was asked)"
EXIT_REFUSED, EXIT_ERROR = 4, 3


def build_plan(bundle_dir, cfg: dict | None = None, icfg: dict | None = None, target: int = fulldoc_db.PART_TARGET_CHARS):
    """(rows, report). rows: [{"document_id", "source", "spans"}] with text; report: no text. Raises RuntimeError unless the
    bundle loads ok with EVERY manifest document acquired (same pins, filters and admission as the bundle reader)."""
    cfg = cfg if cfg is not None else fulldoc_corpus.load_config()
    icfg = icfg if icfg is not None else fl.load_config()
    base = Path(bundle_dir).resolve()
    corpus = fulldoc_corpus.load_bundle(base, cfg, icfg)
    mraw = (base / cfg["manifest_name"]).read_bytes()
    sraw = (base / cfg["sums_name"]).read_bytes()
    items = fulldoc_corpus._manifest_items(mraw) or []
    if corpus.status != fulldoc_corpus.STATUS_OK or len(corpus.acquired) != len(items) or corpus.unusable or not items:
        codes = sorted({u.get("code") for u in corpus.unusable})
        raise RuntimeError(f"bundle is not fully acquired (status {corpus.status}, "
                           f"{len(corpus.acquired)}/{len(items)} acquired, codes {codes})")
    pins = {"manifest_sha256": hashlib.sha256(mraw).hexdigest(), "sums_sha256": hashlib.sha256(sraw).hexdigest()}
    chunking = fulldoc_db.chunking_params()
    rows, docs = [], []
    for seq, item in enumerate(items):
        text, meta = fl.load_text(item, base, icfg)
        rec = fl.prepare_from_parts(item, text, meta, icfg)
        rec["document_id"] = item["document_id"]
        r = fulldoc_db.build_rows(rec, item, meta, seq, pins, chunking, target, len(items))
        rebuilt = "".join(s["text"] for s in r["spans"])  # offline self-check: partition rebuilds byte-exactly
        if rebuilt != text:
            raise RuntimeError(f"{item['document_id']}: partition does not rebuild the text")
        rows.append(r)
        docs.append({"document_id": item["document_id"], "canonical_url": r["source"]["canonical_url"],
                     "chars": len(text), "text_sha256": r["source"]["provenance"]["text_sha256"],
                     "n_spans": len(r["spans"]),
                     "spans": [[s["start_index"], s["end_index"],
                                hashlib.sha256(s["text"].encode("utf-8")).hexdigest()] for s in r["spans"]]})
    report = {"kind": fulldoc_db.FULLDOC_CHUNKS_KIND, "pins": pins, "chunking": chunking, "target_chars": target,
              "n_documents": len(rows), "n_sources": len(rows), "n_spans": sum(d["n_spans"] for d in docs),
              "content_hash_of_empty_content": fulldoc_db.EMPTY_SHA256, "documents": docs}
    return rows, report


def write_plan_file(report: dict, plan_dir: Path) -> Path:
    plan_dir.mkdir(parents=True, exist_ok=True)
    out = plan_dir / "ingest-plan.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return out


MAX_SOURCES, MAX_SPANS = 9, 2500


def check_caps(rows: list, expect_spans: int | None) -> None:
    """Row caps, enforced BEFORE any engine or INSERT: sources <= 9, spans <= min(2500, 1.05 x the reviewed dry-run count)."""
    n_sp = sum(len(r["spans"]) for r in rows)
    cap = min(MAX_SPANS, int((expect_spans or 0) * 1.05))
    if len(rows) > MAX_SOURCES:
        raise fl.IngestError(f"{len(rows)} sources exceed the cap {MAX_SOURCES}")
    if not expect_spans or n_sp > cap:
        raise fl.IngestError(f"{n_sp} spans exceed the cap {cap}")


def write_rows(engine_wo, engine_ro, rows: list, icfg: dict, expect_spans: int | None = None) -> dict:
    """INSERT-only, one transaction per document, stops at the first error. NEW RUN only; a canonical URL or content hash that
    already exists is a stop-and-escalate condition (vet R5). Never called by the dry-run and never run in DBC-1."""
    from sqlalchemy import text as T
    from vera.corpus_admission import SCHEMA

    def S(sql):
        return T(sql.replace("{S}", SCHEMA))
    check_caps(rows, expect_spans)
    cc = icfg["candidate"]
    lookup = fl.RoLookup(engine_ro)
    for r in rows:  # preflight (ro SELECTs) before any write
        src = r["source"]
        if lookup.canonical(src["canonical_url"]) is not None:
            raise fl.IngestError(f"{r['document_id']}: canonical URL already exists (stop and escalate: version shadowing)")
        if lookup.hash_any_canonical(src["provenance"]["text_sha256"]) is not None:
            raise fl.IngestError(f"{r['document_id']}: same text already stored (stop and escalate)")
    rid, done = None, {"sources": 0, "spans": 0}
    for r in rows:
        src, prov = r["source"], r["source"]["provenance"]
        with engine_wo.begin() as c:
            if rid is None:
                qid = c.execute(S("INSERT INTO {S}.questions (research_question) VALUES (:q) RETURNING question_id"),
                                {"q": QUESTION}).scalar()
                rid = c.execute(S("INSERT INTO {S}.runs (question_id, question) VALUES (:q, :t) RETURNING run_id"),
                                {"q": qid, "t": QUESTION}).scalar()
            cid = c.execute(S("INSERT INTO {S}.candidates (run_id, source_url, title, snippet, gate_a_decision, "
                              "gate_a_rationale, fetch_status) VALUES (:r,:u,:t,NULL,:d,:ra,:fs) RETURNING candidate_id"),
                            {"r": rid, "u": prov["url"], "t": prov["title"], "d": cc["gate_a_decision"],
                             "ra": "AE-1020 chunk-stored full document", "fs": cc["fetch_status"]}).scalar()
            csid = c.execute(S("INSERT INTO {S}.canonical_sources (canonical_url) VALUES (:u) RETURNING canonical_source_id"),
                             {"u": src["canonical_url"]}).scalar()
            sid = c.execute(S("INSERT INTO {S}.sources (canonical_source_id, candidate_id, version, content, content_hash, "
                              "provenance) VALUES (:cs,:cid,1,'',:h,CAST(:p AS jsonb)) RETURNING source_id"),
                            {"cs": csid, "cid": cid, "h": src["content_hash"], "p": json.dumps(prov)}).scalar()
            for sp in r["spans"]:
                c.execute(S("INSERT INTO {S}.evidence_spans (source_id, text, start_index, end_index, relevance_score, "
                            "evidence_type) VALUES (:s,:t,:a,:b,NULL,:k)"),
                          {"s": sid, "t": sp["text"], "a": sp["start_index"], "b": sp["end_index"],
                           "k": sp["evidence_type"]})
        done["sources"] += 1
        done["spans"] += len(r["spans"])
    done["run_id"] = rid
    return done


def main(argv=None, env=os.environ, engine_factory=None, cfg=None, guard=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    ap.add_argument("--bundle", required=True, help="local bundle directory (manifest, sums, texts/)")
    ap.add_argument("--plan-dir", default=str(PLAN_DIR))
    ap.add_argument("--target-chars", type=int, default=fulldoc_db.PART_TARGET_CHARS)
    ap.add_argument("--write", action="store_true", help="INSERT the plan (refused without authorization + DB env)")
    ap.add_argument("--new-run", action="store_true")
    ap.add_argument("--i-have-owner-authorization", default=None, metavar="TOKEN")
    ap.add_argument("--expect-spans", type=int, default=None, help="span count of the reviewed dry-run (required by --write)")
    ap.add_argument("--db-url-env", default="VERA_DB_URL_WO", help="NAME of the write-account env var")
    ap.add_argument("--ro-url-env", default="VERA_DB_URL_RO", help="NAME of the read-only env var (preflight)")
    a = ap.parse_args(argv)
    if a.write:
        why = None
        if not (a.i_have_owner_authorization or "").strip():
            why = "missing --i-have-owner-authorization <token>"
        elif not (env.get(a.db_url_env) or "").strip() or not (env.get(a.ro_url_env) or "").strip():
            why = "the write and read-only DB URL env vars (names above) are not both set"
        elif not a.new_run:
            why = "--write requires --new-run"
        elif not a.expect_spans or a.expect_spans < 1:
            why = "--write requires --expect-spans <dry-run span count>"
        if why:
            print(f"REFUSED --write: {why}. Nothing was read or written.")
            return EXIT_REFUSED
        (guard or _ops_guard)()  # write mode ALWAYS runs the ops guard (refuses under a test runner; production tag)
    try:
        rows, report = build_plan(a.bundle, cfg, target=a.target_chars)
    except Exception as e:  # noqa: BLE001  never echo texts or paths
        print(f"ERROR {type(e).__name__}: {str(e)[:200] if isinstance(e, RuntimeError) else 'details withheld'}")
        return EXIT_ERROR
    out = write_plan_file(report, Path(a.plan_dir))
    print(f"plan: documents={report['n_documents']} sources={report['n_sources']} spans={report['n_spans']} "
          f"max_span_chars={a.target_chars + 200} plan_file={out.name} mode={'WRITE' if a.write else 'dry-run'}")
    if not a.write:
        return 0
    try:
        check_caps(rows, a.expect_spans)
    except fl.IngestError as e:
        print(f"REFUSED --write: {e}")
        return EXIT_REFUSED
    from sqlalchemy import create_engine  # lazy; only reached with authorization + env present
    factory = engine_factory or create_engine
    eng_wo, eng_ro = factory(env[a.db_url_env]), factory(env[a.ro_url_env])
    try:
        done = write_rows(eng_wo, eng_ro, rows, fl.load_config(), a.expect_spans)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR {type(e).__name__} (details withheld: may contain connection data)")
        return EXIT_ERROR
    finally:
        for e_ in (eng_wo, eng_ro):
            try:
                e_.dispose()
            except Exception:  # noqa: BLE001
                pass
    print(f"written: sources={done['sources']} spans={done['spans']} run_id={done['run_id']}")
    return 0


def _ops_guard() -> None:
    if os.environ.get("VERA_TEST_RUNNER") or os.environ.get("PYTEST_CURRENT_TEST"):
        sys.stderr.write("refused: ops script, not for test runs\n")
        raise SystemExit(3)
    from safety import connect_guard
    connect_guard.install("production")


if __name__ == "__main__":
    sys.exit(main())
