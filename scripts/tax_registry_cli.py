"""TAXO-EXTENSIBLE-1108: owner CLI for the extensible taxonomy registry (vera/tax_registry.py). INERT until migration 022 is applied
and the owner turns on a flag in config/tax_registry.json.

    python -I -B scripts/tax_registry_cli.py show --mode active|proposed
    python -I -B scripts/tax_registry_cli.py drift
    python -I -B scripts/tax_registry_cli.py sync-anchors [--apply]
    python -I -B scripts/tax_registry_cli.py propose --manifest FILE [--apply]
    python -I -B scripts/tax_registry_cli.py approve --manifest FILE [--approve k1,domain:x] [--reject k2] --actor owner:NAME [--apply]

Default is a dry run (READ ONLY transactions). The engine is the app rw engine (vera.db.get_role_engine), created only when not
injected; no host or secret is printed. Exit 0 OK, 2 usage, 3 REFUSED (flag off, 022 absent, problems), 4 CANNOT_VERIFY."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vera import tax_registry as tr  # noqa: E402

_ITEMS = re.compile(r"(domain:)?[a-z0-9][a-z0-9-]{1,63}(,(domain:)?[a-z0-9][a-z0-9-]{1,63})*")


def _items(v):
    if v is None:
        return set()
    if not _ITEMS.fullmatch(v):
        raise ValueError("bad item list")
    return set(v.split(","))


def main(argv=None, *, engine=None, out=print) -> int:
    ap = argparse.ArgumentParser(prog="tax_registry_cli.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("--mode", choices=("active", "proposed"), required=True)
    sub.add_parser("drift")
    s = sub.add_parser("sync-anchors")
    s.add_argument("--apply", action="store_true")
    s = sub.add_parser("propose")
    s.add_argument("--manifest", required=True)
    s.add_argument("--apply", action="store_true")
    s = sub.add_parser("approve")
    s.add_argument("--manifest", required=True)
    s.add_argument("--approve")
    s.add_argument("--reject")
    s.add_argument("--actor", required=True)
    s.add_argument("--apply", action="store_true")
    try:
        a = ap.parse_args(argv)
        approve, reject = (_items(getattr(a, "approve", None)), _items(getattr(a, "reject", None)))
    except (SystemExit, ValueError):
        return 2
    try:
        cfg = tr.load_config()
        if engine is None:
            from vera import db
            engine = db.get_role_engine("rw")
        if a.cmd == "show":
            present, rows = tr.read_registry_rows(engine, a.mode)
            if not present:
                out("REFUSED: migration 022 is not applied")
                return 3
            data = tr.registry_data_from_rows(rows, a.mode)
            out(f"version {data['anchor_registry_version']}; entries {len(data['entries'])}; skipped {len(data['skipped'])}")
            for e in data["entries"]:
                out(f"{e['kind']} {e['id']} state={e['state']} proposed={e['includes_proposed']} parent={e['parent_topic_key']} "
                    f"field={e['field']}")
            for k in data["skipped"]:
                out(f"skipped {k['kind']} {k['id']} ({k['reason']})")
            return 0
        if a.cmd == "drift":
            from vera import acq_config, micro_topic
            topreg = micro_topic.registry_active_data((acq_config.load_config() or {}).get("micro_topic") or {}, engine=engine)
            tax = tr.load_registry_data(cfg, engine, "proposed")
            if tax is None:
                out("REFUSED: migration 022 is not applied")
                return 3
            out(json.dumps(tr.drift_report(topreg, tax), sort_keys=True))
            return 0
        if a.cmd == "sync-anchors":
            from vera import acq_config, micro_topic
            topreg = micro_topic.registry_active_data((acq_config.load_config() or {}).get("micro_topic") or {}, engine=engine)
            rep = tr.sync_anchors(engine, topreg, cfg, apply=a.apply)
            out(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in rep.items()}, sort_keys=True))
            return 0
        man = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
        if a.cmd == "propose":
            present, _rows = tr.read_registry_rows(engine, "active")
            if not present:
                out("REFUSED: migration 022 is not applied")
                return 3
            plan = tr.plan_proposals(man, tr.read_existing(engine))
            rep = tr.write_proposals(engine, plan, cfg, apply=a.apply and not plan["problems"])
            out(json.dumps({"applied": rep["applied"], "counts": rep["counts"], "problems": rep["problems"]}, sort_keys=True))
            if plan["problems"]:
                out("REFUSED: plan has problems; nothing written")
                return 3
            return 0
        rep = tr.approve_batch(engine, man, approve, reject, a.actor, cfg, apply=a.apply)
        out(json.dumps(rep, sort_keys=True))
        bad = rep["problems"] or any(v.startswith("refused:") for v in rep["outcomes"].values())
        return 3 if bad else 0
    except tr.TaxRefused as exc:
        out(f"REFUSED: {exc}")
        return 3
    except Exception as exc:  # noqa: BLE001 - class name only, never driver text
        out(f"CANNOT_VERIFY: {type(exc).__name__}: {str(exc) if isinstance(exc, tr.TaxRegistryError) else ''}".rstrip(": "))
        return 4


if __name__ == "__main__":
    sys.exit(main())
