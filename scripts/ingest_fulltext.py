"""DARINT full-text ingest CLI (G1 + G2 + G5). No network: fetching is a separate, owner-run step (OWNER-RUNBOOK).

Ops class: ops-write (`ingest --write` inserts through the write-only account; running it is owner-gated). Run as a
script it installs connect_guard (app context) and refuses under a test runner.

  convert  MANIFEST --raw-dir DIR      raw/<document_id>.xml|.html -> <manifest dir>/<text_file> + .meta.json
  ingest   MANIFEST                    DRY RUN (default): every filter + conversion record + in-memory DAR; no DB
  ingest   MANIFEST --write (--run-id N | --new-run) [--sources-only] [--max-docs N]
                                       INSERT-only write through the write-only account; the read-only account is
                                       used for lookups, preflight and the DAR snapshot

Database URLs come ONLY from the env var NAMES in config/fulltext_ingest.json (db_url_env_wo / db_url_env_ro); their
values are never printed, logged or written. Output lines: `OK <id> ...`, `KNOWN <id> ...`, `REFUSED <id> <code>:
<reason>`, `ERROR ...`. The write run stops at the first refusal or error. A JSON report (counts and ids only, never
text) is written with --report.

Exit codes: 0 all OK; 2 at least one item refused; 3 write/persist error; 4 preflight refused; 64 usage.
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

from vera import fulltext_loader as fl  # noqa: E402
from vera import jats_text  # noqa: E402

def _items(manifest: Path) -> list[dict]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    items = data.get("sources")
    if not isinstance(items, list):
        raise SystemExit("manifest has no `sources` list")
    return items


def _bound(args, cfg) -> int:
    n = int(args.max_docs if args.max_docs is not None else cfg["max_docs_default"])
    if n < 1 or n > int(cfg["max_docs_ceiling"]):
        raise SystemExit(f"--max-docs must be 1..{cfg['max_docs_ceiling']}")
    return n


def cmd_convert(args, cfg) -> int:
    manifest = Path(args.manifest).resolve()
    base, raw_dir = manifest.parent, Path(args.raw_dir).resolve()
    rc = 0
    for item in _items(manifest)[: _bound(args, cfg)]:
        did, fmt = item.get("document_id"), item.get("fulltext_format")
        src = raw_dir / f"{did}{cfg['format_extensions'].get(fmt, '.unsupported')}"
        try:
            raw = src.read_bytes()
            conv = jats_text.convert(raw, fmt, cfg["converter"])
        except (OSError, jats_text.ConversionError) as e:
            print(f"REFUSED {did} {cfg['reason_codes']['conversion']}: {e}")
            rc = 2
            continue
        except Exception as e:  # noqa: BLE001  one hostile file never aborts the batch
            print(f"REFUSED {did} {cfg['reason_codes']['conversion']}: converter failed ({type(e).__name__})")
            rc = 2
            continue
        out = (base / item["text_file"]).resolve()
        if base not in out.parents:
            print(f"REFUSED {did} {cfg['reason_codes']['text_missing']}: text_file outside the manifest directory")
            rc = 2
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        data = conv["text"].encode("utf-8")
        out.write_bytes(data)
        meta = {k: v for k, v in conv.items() if k != "text"}
        meta.update({"document_id": did, "raw_sha256": hashlib.sha256(raw).hexdigest(),
                     "text_sha256": hashlib.sha256(data).hexdigest(), "raw_file": src.name})
        out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"OK {did} chars={len(conv['text'])} body={conv['body_chars']} abstract={conv['abstract_chars']} "
              f"sections={len(conv['sections'])} citations={len(conv['citations'])} refs={len(conv['references'])} "
              f"text_sha256={meta['text_sha256']}")
    return rc


def _dar(cfg):
    from vera.dar_corpus import DarCorpus
    return DarCorpus(fl.ingest_routing_config(cfg=cfg))


def _non_doc_kinds() -> list[str]:
    from vera import answer_contract
    return list((answer_contract.load_config().get("acquisition") or {}).get("non_document_text_kinds") or [])


def cmd_ingest(args, cfg, env=os.environ, engine_factory=None) -> int:
    manifest = Path(args.manifest).resolve()
    items = _items(manifest)[: _bound(args, cfg)]
    report = {"mode": "write" if args.write else "dry-run", "items": [], "resumed": 0}
    rc = 0
    if not args.write:
        dar = _dar(cfg)
        for item in items:
            did = item.get("document_id")
            try:
                rec = fl.prepare(item, manifest.parent, cfg)
            except fl.Refused as e:
                print(f"REFUSED {did} {e.code}: {e}")
                report["items"].append({"id": did, "outcome": "refused", "code": e.code, "reason": str(e)})
                rc = 2
                continue
            plan = dar.ingest_document(fl.doc_for_dar(rec, None, None, cfg))
            s = fl.dar_summary(plan)
            print(f"OK {did} chars={len(rec['text'])} chunks={s['chunks']} domains={s['domains']} "
                  f"new_domains={s['new_domains']} dar={s['status']}")
            report["items"].append({"id": did, "outcome": "ok", "dar": s})
        report["dar_stats"] = dar.stats()
        _write_report(args, report)
        return rc

    if not args.run_id and not args.new_run:
        print("ERROR --write needs --run-id N (an existing run) or --new-run")
        return 64
    url_wo, url_ro = (env.get(cfg["db_url_env_wo"]) or "").strip(), (env.get(cfg["db_url_env_ro"]) or "").strip()
    if not url_wo or not url_ro:
        print(f"ERROR --write needs both {cfg['db_url_env_wo']} and {cfg['db_url_env_ro']} set (values not shown)")
        return 4
    if engine_factory is None:
        from sqlalchemy import create_engine as engine_factory
    eng_wo, eng_ro = engine_factory(url_wo, pool_pre_ping=True), engine_factory(url_ro, pool_pre_ping=True)
    del url_wo, url_ro
    from vera.dar_store import DarReader, persist
    lookup = fl.RoLookup(eng_ro)
    store = fl.FulltextStore(eng_wo, cfg["question_text"], lookup, cfg, run_id=args.run_id)
    if args.run_id and not store.run_exists(args.run_id):
        print(f"ERROR run {args.run_id} not found")
        return 4
    need = list(cfg["write_tables"]) + ([] if args.sources_only else list(cfg["dar"]["write_tables"]))
    if cfg.get("licence_observations_enabled") is True:
        need.append(cfg["licence_observation_table"])
    dar, ctx, reader = None, None, None
    if not args.sources_only:
        missing = lookup.tables_present(cfg["dar"]["required_tables"])
        if missing:
            print(f"ERROR DAR tables absent ({', '.join(missing)}): migration 007 not applied; refusing --write "
                  "(use --sources-only to store sources now; DAR persist resumes on a later run)")
            return 4
    no_priv = store.missing_insert_privileges(need)
    if no_priv:
        print(f"ERROR the write account lacks INSERT on {', '.join(no_priv)} (grants missing, e.g. 007 grants); "
              "refusing --write before any row is written")
        return 4
    if not args.sources_only:
        reader = DarReader(eng_ro)
        dar = _dar(cfg)
        ctx, pending = fl.prime_from_store(dar, lookup, reader, _non_doc_kinds())
        try:
            done = fl.resume_pending(dar, eng_wo, reader, pending, ctx, persist)
        except fl.IngestError as e:
            print(f"ERROR {e}")
            report["error"] = str(e)
            _write_report(args, report)
            return 3
        report["resumed"] = len(done)
        if done:
            print(f"RESUMED {len(done)} stored full-text document(s) without spans")
    for item in items:
        did = item.get("document_id")
        try:
            rec = fl.prepare(item, manifest.parent, cfg)
            step = None
            if dar is not None:
                step = fl.dar_step(dar, reader, ctx, persist,
                                   lambda ids, rec=rec: fl.doc_for_dar(rec, ids["source_id"], ids["canonical_source_id"],
                                                                       cfg), did)
            outcome, ids = store.admit_fulltext(rec, in_tx=step)
            entry = {"id": did, "outcome": outcome, "source_id": ids.get("source_id"),
                     "canonical_source_id": ids.get("canonical_source_id"), "version": ids.get("version"),
                     "licence_observations": ids.get("licence_observations")}
            if ids.get("decision"):  # D-057: duplicate content under another canonical URL, skipped (ids only)
                entry["decision"] = ids["decision"]
                entry["duplicate_of_canonical_source_id"] = ids.get("duplicate_of_canonical_source_id")
            if outcome == "admitted" and dar is not None:
                got = ids["in_tx"]
                ctx.update(got["ctx"])  # merged only after the commit
                res = got["persist"]
                entry["dar"] = {**fl.dar_summary(got["plan"]),
                                **{k: v for k, v in res.items() if k.endswith("_written")}}
            elif outcome == "admitted":
                entry["dar"] = "pending (--sources-only)"
            report["items"].append(entry)
            tag = "OK" if outcome == "admitted" else "KNOWN"
            print(f"{tag} {did} source_id={entry['source_id']} canonical_source_id={entry['canonical_source_id']} "
                  f"version={entry['version']} dar={json.dumps(entry.get('dar'), sort_keys=True)}")
        except fl.Refused as e:
            print(f"REFUSED {did} {e.code}: {e}")
            report["items"].append({"id": did, "outcome": "refused", "code": e.code, "reason": str(e)})
            rc = 2
            break
        except Exception as e:  # noqa: BLE001  stop on first failure, verbosely; never echo URLs
            # IngestError text is ours; a driver error may carry a host name, so only its type is shown.
            print(f"ERROR {did} {type(e).__name__}" + (f": {e}" if isinstance(e, fl.IngestError) else ""))
            report["items"].append({"id": did, "outcome": "error", "error": type(e).__name__})
            rc = 3
            break
    report["run_id"] = store._run_id
    print(f"RUN run_id={store._run_id}")
    _write_report(args, report)
    return rc


def _write_report(args, report):
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("manifest")
    c.add_argument("--raw-dir", required=True)
    c.add_argument("--max-docs", type=int)
    i = sub.add_parser("ingest")
    i.add_argument("manifest")
    i.add_argument("--write", action="store_true")
    g = i.add_mutually_exclusive_group()
    g.add_argument("--run-id", type=int)
    g.add_argument("--new-run", action="store_true")
    i.add_argument("--sources-only", action="store_true")
    i.add_argument("--max-docs", type=int)
    i.add_argument("--report")
    i.add_argument("--config", default=str(fl.CONFIG_PATH))
    c.add_argument("--config", default=str(fl.CONFIG_PATH))
    return ap


def main(argv=None, env=os.environ, engine_factory=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = fl.load_config(Path(args.config))
    try:
        if args.cmd == "convert":
            return cmd_convert(args, cfg)
        return cmd_ingest(args, cfg, env=env, engine_factory=engine_factory)
    except fl.IngestError as e:
        print(f"ERROR {e}")
        return 3
    except Exception as e:  # noqa: BLE001  never echo driver text (it may name a host)
        print(f"ERROR {type(e).__name__} (details withheld: may contain connection data)")
        return 3


def _ops_guard() -> None:
    """ENV-E (E7/C8): refuse under a test runner; classify-and-tag every connection as an APP context (never disposable)."""
    if os.environ.get("VERA_TEST_RUNNER") or os.environ.get("PYTEST_CURRENT_TEST"):
        sys.stderr.write("refused: ops script, not for test runs\n")
        raise SystemExit(3)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from safety import connect_guard
    connect_guard.install("production")


if __name__ == "__main__":
    _ops_guard()
    sys.exit(main())
