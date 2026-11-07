"""TAXO-INDUCE-1107: owner-run taxonomy induction (vera/taxonomy_induction.py). Wrapper: work/TAXO-INDUCE-1107/scripts/run_induction.sh.

    python -I -B scripts/taxonomy_induce.py --out FILE [--write] [--no-llm]

Dry run by default: reads only (READ ONLY transaction through the rw URL in VERA_DB_URL_RW; the value is never printed), clusters
chunks, names clusters with one free Groq model (GROQ_API_KEY from the environment; --no-llm skips naming), and writes the review
manifest to --out. --write ALSO inserts the named proposals as runtime DRAFT registry rows (app-path INSERT; nothing becomes active:
the owner approves with scripts/taxonomy_approve.py). Prints counts, proposed names and keyphrases; never titles, text, ids or URLs.
Exit 0 OK, 2 usage, 3 REFUSED, 4 CANNOT_VERIFY."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vera import taxonomy_induction as ti  # noqa: E402


def _heartbeat(stop, state, every=10.0):
    t0 = time.monotonic()
    while not stop.wait(every):
        print(f"[{int(time.monotonic() - t0):>4}s] working: {state[0]}", file=sys.stderr, flush=True)


def main(argv=None, *, engine=None, cfg_path=None, mt_cfg=None, entries_llm=None, client_for=None, out=print) -> int:
    ap = argparse.ArgumentParser(prog="taxonomy_induce.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        return 2
    if a.write and a.no_llm:
        out("INDUCE: REFUSED (--write needs named proposals; drop --no-llm)")
        return 3
    try:
        cfg = ti.load_config(cfg_path)
        if mt_cfg is None:
            from vera import acq_config
            mt_cfg = (acq_config.load_config() or {}).get("micro_topic") or {}
    except Exception as exc:  # noqa: BLE001
        out(f"INDUCE: REFUSED (configuration: {type(exc).__name__})")
        return 3
    state, stop = ["starting"], threading.Event()
    hb = threading.Thread(target=_heartbeat, args=(stop, state), daemon=True)
    hb.start()

    def say(m):
        state[0] = m
        print(f"[{time.strftime('%H:%M:%S')}] {m}", file=sys.stderr, flush=True)
    try:
        if engine is None:
            from vera import db
            engine = db.get_role_engine("rw")
        from vera import topic_registry
        say("reading the registry")
        rows = topic_registry.read_all_rows(engine, mt_cfg)
        man = ti.induce(engine, rows, cfg, mt_cfg, use_llm=not a.no_llm, entries_llm=entries_llm, client_for=client_for, progress=say)
        if a.write:
            Path(a.out).write_text(json.dumps(man, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")  # before any insert
            say(f"inserting {len(man['proposals'])} draft rows")
            man["write"] = [{"key": k, "status": s} for k, s in topic_registry.insert_induced_drafts(engine, man["proposals"])]
    except Exception as exc:  # noqa: BLE001 - class only
        out(f"INDUCE: CANNOT_VERIFY ({type(exc).__name__})")
        return 4
    finally:
        stop.set()
    Path(a.out).write_text(json.dumps(man, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    b = man["basis"]
    out(f"documents {b['documents']}, chunks {b['chunks']}, active registry entries {b['active_registry_entries']}")
    out(f"not matched now: {100 * man['unmatched_share_now']:.1f}%; projected ({man['projection_basis']}): "
        f"{100 * man['unmatched_share_projected']:.1f}%")
    n = man["naming"]
    out(f"naming: model {n['model']}, calls {n['calls']}, tokens {n['tokens']}, rate-limited {n['rate_limited']}, "
        f"clusters skipped {len(n['skipped'])}")
    for p in man["proposals"]:
        where = f"under {p['parent']}" if p["parent"] else f"domain {p['field']!r}{' (new)' if p['new_domain'] else ''}"
        out(f"  {p['level']:8} {p['key']} {p['label']!r} {where}; docs {p['n_docs']} (unmatched now {p['n_unmatched_docs']}), "
            f"cohesion {p['cohesion']}; {', '.join(p['keyphrases'][:8])}")
    out(f"rejected clusters: {len(man['rejected'])} " + json.dumps(sorted({r['reason'] for r in man['rejected']})))
    if man.get("round_note"):
        out(man["round_note"])
    if a.write:
        out("write: " + json.dumps({s: sum(1 for w in man["write"] if w["status"] == s) for s in {w["status"] for w in man["write"]}}))
    out(f"INDUCE: OK ({len(man['proposals'])} proposals; manifest {a.out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
