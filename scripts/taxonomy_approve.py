"""TAXO-INDUCE-1107: owner batch approval of induced registry drafts. Wrapper: work/TAXO-INDUCE-1107/scripts/approve_batch.sh.

    python -I -B scripts/taxonomy_approve.py --manifest FILE [--approve all|k1,k2] [--reject k1,k2] [--write]

Lists by default (READ ONLY). With --write: approve = draft -> active, reject = draft -> suspended, through the rw app path (the only
state changes the 011 trigger allows an app role). Only manifest keys whose stored draft content matches the manifest hash change.
Exit 0 OK, 2 usage, 3 REFUSED, 4 CANNOT_VERIFY."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vera import taxonomy_induction as ti  # noqa: E402

_KEYS = re.compile(r"[a-z0-9][a-z0-9-]{1,63}(,[a-z0-9][a-z0-9-]{1,63})*")


def _keys(v, allow_all):
    if v is None:
        return set()
    if v == "all" and allow_all:
        return {"*"}
    if not _KEYS.fullmatch(v):
        raise ValueError("bad key list")
    return set(v.split(","))


def main(argv=None, *, engine=None, out=print) -> int:
    ap = argparse.ArgumentParser(prog="taxonomy_approve.py")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--approve")
    ap.add_argument("--reject")
    ap.add_argument("--write", action="store_true")
    try:
        a = ap.parse_args(argv)
        approve, reject = _keys(a.approve, True), _keys(a.reject, False)
    except (SystemExit, ValueError):
        return 2
    try:
        man = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
        keys = {p["key"] for p in man["proposals"]}
        ti.batch_hash(man["proposals"])
    except Exception as exc:  # noqa: BLE001
        out(f"APPROVE: REFUSED (manifest unreadable: {type(exc).__name__})")
        return 3
    if man.get("batch_sha256") != ti.batch_hash(man["proposals"]):
        out("APPROVE: REFUSED (manifest batch hash does not match its proposals: the manifest was edited)")
        return 3
    if "*" in approve:
        approve = keys - reject
    if approve & reject or not (approve | reject) <= keys:
        out("APPROVE: REFUSED (a key is both approved and rejected, or is not in the manifest)")
        return 3
    try:
        if engine is None:
            from vera import db
            engine = db.get_role_engine("rw")
        from vera import topic_registry
        stored = topic_registry.latest_rows(engine, [p["key"] for p in man["proposals"]])
        status = {p["key"]: ti.row_status(stored.get(p["key"]), p) for p in man["proposals"]}
        for p in man["proposals"]:
            act = "approve" if p["key"] in approve else "reject" if p["key"] in reject else "-"
            out(f"  {status.get(p['key'])!s:18} {act:7} {p['level']:8} {p['key']} {p['label']!r} "
                f"{'under ' + p['parent'] if p['parent'] else 'domain ' + repr(p['field'])}")
        if not a.write:
            out(f"APPROVE: OK (listed {len(keys)}; planned approve {len(approve)}, reject {len(reject)}; add --write to apply)")
            return 0
        res = topic_registry.decide_drafts(engine, ti.decisions(man, approve, reject))  # topics before subtopics
    except Exception as exc:  # noqa: BLE001 - class only
        out(f"APPROVE: CANNOT_VERIFY ({type(exc).__name__})")
        return 4
    for k, s in res:
        out(f"  {k}: {s}")
    done = sum(1 for _k, s in res if s in ("activated", "suspended"))
    out(f"APPROVE: OK ({done} of {len(res)} changed; the Corpus Description page picks active topics up within its cache time)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
