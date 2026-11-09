#!/usr/bin/env python3
"""GOV-PD-1108: refresh config/us_gov_domains_federal.json from CISA's dotgov-data (CC0-1.0).

One fetch of dataset.source_url (config/us_federal_pd.json) through vera.public_fetch: robots.txt first (fail closed),
pinned public address, https only, byte caps. Keeps only four columns (domain, type, organization, suborganization); the contact email
and city/state columns are dropped. Prints a diff summary against the current snapshot. Writes the snapshot only with
--write. Offline use: --csv PATH reads a local copy instead (e.g. from a shallow `git clone` of the dataset repo) and
records --source-url as given.

    python3 -B -I scripts/refresh_us_gov_domains.py            # fetch, print diff, write nothing
    python3 -B -I scripts/refresh_us_gov_domains.py --write    # fetch and write the snapshot
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vera import gov_public_domain as gpd  # noqa: E402

REQUIRED = ("Domain name", "Domain type", "Organization name")


def parse(text: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or any(k not in rows[0] for k in REQUIRED):
        raise SystemExit(f"FAIL: unexpected CSV header; need columns {REQUIRED}")
    out = []
    for r in rows:
        d = str(r.get("Domain name") or "").strip().lower().rstrip(".")
        if d.endswith(".gov") and "." in d:
            out.append({"domain": d, "type": str(r.get("Domain type") or "").strip(),
                        "organization": str(r.get("Organization name") or "").strip()[:160],
                        "suborganization": str(r.get("Suborganization name") or "").strip()[:160]})
    if len(out) < 100:  # safety bound: a truncated or wrong file must never replace the snapshot
        raise SystemExit(f"FAIL: only {len(out)} .gov rows parsed; refusing to write")
    return sorted(out, key=lambda x: x["domain"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--csv", help="local CSV copy instead of a fetch")
    ap.add_argument("--source-url", help="recorded source URL when --csv is used")
    a = ap.parse_args(argv)
    cfg = json.loads(gpd.CONFIG_PATH.read_text(encoding="utf-8"))
    ds = cfg["dataset"]
    if a.csv:
        text, src = Path(a.csv).read_text(encoding="utf-8"), a.source_url or ("file:" + Path(a.csv).name)
    else:
        from vera import public_fetch
        print(f"fetching {ds['source_url']} (robots.txt first)", flush=True)
        try:
            got = public_fetch.fetch(ds["source_url"])
        except public_fetch.FetchRefused as e:
            print(f"FAIL: fetch refused: {e.reason}", file=sys.stderr)
            return 2
        text, src = got["text"], got["url"]
    rows = parse(text)
    types = sorted({r["type"] for r in rows})
    fed_types = set(ds["federal_types"])
    unknown = [t for t in types if t.startswith("Federal") and t not in fed_types]
    if unknown:
        print(f"WARN: federal-looking types not in dataset.federal_types (not admitted): {unknown}")
    snap_path = ROOT / ds["path"]
    old = {}
    try:
        old = {r["domain"]: r["type"] for r in json.loads(snap_path.read_text(encoding="utf-8"))["domains"]}
    except (OSError, ValueError, KeyError):
        pass
    new = {r["domain"]: r["type"] for r in rows}
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    retyped = sorted(d for d in set(new) & set(old) if new[d] != old[d])
    print(f"rows {len(rows)}; types {types}; added {len(added)}; removed {len(removed)}; retyped {len(retyped)}")
    for label, xs in (("added", added), ("removed", removed), ("retyped", retyped)):
        for d in xs[:40]:
            print(f"  {label}: {d} {old.get(d, '')} -> {new.get(d, '')}")
    snap = {"_doc": "GOV-PD-1108 vendored snapshot of CISA dotgov-data (CC0-1.0), reduced to domain, Domain type, "
                    "Organization and Suborganization name (contact email, city and state dropped). Data, not a hand list: regenerate "
                    "with scripts/refresh_us_gov_domains.py; never edit rows by hand.",
            "source_url": src, "project_url": ds.get("project_url", ""), "licence": "CC0-1.0",
            "retrieved_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "sha256_note": "of the fetched text after line whitespace normalisation (vera.public_fetch)",
            "row_count": len(rows), "type_counts": {t: sum(1 for r in rows if r["type"] == t) for t in types},
            "domains": rows}
    if a.write:
        head = json.dumps({k: v for k, v in snap.items() if k != "domains"}, indent=1, ensure_ascii=False)[:-2]
        body = ",\n".join("  " + json.dumps(r, ensure_ascii=False) for r in rows)  # one row per line: readable diffs
        snap_path.write_text(head + ',\n "domains": [\n' + body + "\n ]\n}\n", encoding="utf-8")
        json.loads(snap_path.read_text(encoding="utf-8"))  # fail loudly on a malformed write
        print(f"wrote {snap_path.relative_to(ROOT)}")
    else:
        print("dry run: nothing written (use --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
