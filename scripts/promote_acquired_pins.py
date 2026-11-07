"""H1B-PROMOTE-1107: propose a promotion batch for config/acquired_pins.json (agent-run, offline: no DB, no network).

    <venv>/bin/python -I -B scripts/promote_acquired_pins.py --export FILE --export-sha256 HEX --batch-id ID
                                                           [--manifest PATH] [--dry-run] [--report PATH]

Input is ONLY an owner-run, read-only, hash-only export (work/H1B-PROMOTE-1107/scripts/run_pin_export.sh); the agent
never touches the DB. --export-sha256 is the sha256 the owner's export printed on screen, RELAYED BY THE OWNER (never read
from the agent-writable .sha256 sidecar): the file must match it, so the batch is bound to what the owner produced. The result is a change to config/acquired_pins.json that goes through the
normal rule-7.a release with review; that reviewed commit, not any DB row, is the trust root.

Refused (exit 3, nothing written): a hash mismatch on the export file, a malformed export, a batch id already used,
or any exported document whose id is already pinned with DIFFERENT values or whose text is already pinned under another
id (TAMPER-SUSPECT: the stored row changed after promotion; investigate, never re-pin silently). Only documents the
acquired-path reader verified in the export (verdict "verified", licence decision "allow") are added. Exit 0 OK (also
when there is nothing new), 2 usage.
Output names documents only by their hash-form ids (acq-<16 hex>); no titles, URLs or text exist in the input.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vera import acquired_pins as ap  # noqa: E402

_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class Refused(Exception):
    pass


def _entry(d: dict, batch: str) -> dict:
    return {**{k: d[k] for k in ap.PIN_FIELDS}, "batch": batch}


def propose(manifest: dict, export: dict, export_sha256: str, batch_id: str, today: str) -> tuple[dict, dict]:
    """Pure. (new manifest, report). Raises Refused."""
    if export.get("kind") != "vera-acquired-pin-export" or export.get("schema_version") != 1:
        raise Refused("export is not a schema_version 1 vera-acquired-pin-export")
    if not (isinstance(export.get("repo_head"), str) and _HEX40.fullmatch(export["repo_head"])):
        raise Refused("export repo_head is not a commit sha")
    if not isinstance(export.get("documents"), list):
        raise Refused("export documents is not a list")
    if export.get("account") != "vera_eval_ro":
        raise Refused("export was not made with the read-only account vera_eval_ro")
    if not isinstance(export.get("rows_read"), int) or not isinstance(export.get("row_limit"), int) \
            or export["rows_read"] >= export["row_limit"]:
        raise Refused("export may be truncated (rows_read reached row_limit) or lacks the counts")
    err = ap.validate(manifest)
    if err:
        raise Refused(f"current manifest invalid: {err}")
    if any(b["id"] == batch_id for b in manifest["batches"]):
        raise Refused(f"batch id {batch_id} already used")
    if any(b["export_sha256"] == export_sha256 for b in manifest["batches"]):
        raise Refused("this export was already promoted")
    pinned = manifest["documents"]
    by_text = {e["text_sha256"]: did for did, e in pinned.items()}
    add, same, skipped, suspect = {}, [], {}, []
    for d in export["documents"]:
        did = d.get("document_id") if isinstance(d, dict) else None
        if not isinstance(did, str) or not re.fullmatch(r"acq-[0-9a-f]{16}", did):
            raise Refused("export has a malformed document id")
        if d.get("verdict") != "verified" or d.get("licence_decision") != "allow":
            skipped[did] = "not verified by the reader or licence not allow"
            continue
        e = _entry(d, batch_id)
        if did in pinned:
            if all(pinned[did].get(k) == e[k] for k in ap.PIN_FIELDS):
                same.append(did)
            else:
                suspect.append(did)
            continue
        if e["text_sha256"] in by_text or any(x["text_sha256"] == e["text_sha256"] for x in add.values()):
            suspect.append(did)
            continue
        if did in add:
            raise Refused("export lists a document twice")
        add[did] = e
    if suspect:
        raise Refused("TAMPER-SUSPECT: exported documents differ from their pins or duplicate a pinned text: "
                      + ", ".join(sorted(suspect)))
    new = json.loads(json.dumps(manifest))
    report = {"batch": batch_id, "added": sorted(add), "already_pinned": sorted(same), "skipped": skipped,
              "repo_head": export["repo_head"], "exported_at": export.get("exported_at"),
              "licence_ids": {}}
    for e in add.values():
        report["licence_ids"][e["licence_id"]] = report["licence_ids"].get(e["licence_id"], 0) + 1
    if add:
        new["batches"].append({"id": batch_id, "export_sha256": export_sha256, "repo_head": export["repo_head"],
                               "exported_at": export.get("exported_at"), "proposed_on": today, "added": len(add)})
        new["documents"] = dict(sorted({**pinned, **add}.items()))
        err = ap.validate(new)
        if err:
            raise Refused(f"proposed manifest invalid: {err}")
    return new, report


def main(argv=None) -> int:
    ap_ = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap_.add_argument("--export", required=True)
    ap_.add_argument("--export-sha256", required=True)
    ap_.add_argument("--batch-id", required=True)
    ap_.add_argument("--manifest", default=str(ap.CONFIG_PATH))
    ap_.add_argument("--dry-run", action="store_true")
    ap_.add_argument("--report")
    try:
        a = ap_.parse_args(argv)
    except SystemExit:
        return 2
    want = a.export_sha256.strip().lower()
    if not _HEX64.fullmatch(want):
        print("promote: --export-sha256 is not a sha256", file=sys.stderr)
        return 2
    try:
        raw = Path(a.export).read_bytes()
        if hashlib.sha256(raw).hexdigest() != want:
            raise Refused("export file sha256 differs from the sha256 the owner's export printed")
        export = json.loads(raw.decode("utf-8"))
        mpath = Path(a.manifest)
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        new, rep = propose(manifest, export, want, a.batch_id,
                           datetime.datetime.now(datetime.timezone.utc).date().isoformat())
    except Refused as e:
        print(f"PROMOTE: REFUSED ({e})")
        return 3
    except (OSError, ValueError) as e:
        print(f"PROMOTE: REFUSED (input unreadable: {type(e).__name__})")
        return 3
    print(f"batch {rep['batch']}: add {len(rep['added'])}, already pinned {len(rep['already_pinned'])}, "
          f"skipped {len(rep['skipped'])}; licences {rep['licence_ids']}")
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if rep["added"] and not a.dry_run:
        mpath.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"PROMOTE: OK (manifest updated; review and release it as a rule-7.a batch)")
    else:
        print("PROMOTE: OK (nothing written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
