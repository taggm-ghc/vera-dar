"""Licence research queue and proposal writer (LICENCE-ALLOWLIST-1104; owner 2026-11-04: "1, plus research to learn
about new license types as we encounter them"). Stdlib only. It never fetches the web and never connects to a database.

Deny-all: this script only ever writes PROPOSED rows (status "proposed", approved_by null) to a proposals file that
you name. It refuses to write the allowlist itself. vera.licence_gate ignores every row whose status is not "active",
so nothing here can make a licence storable. A proposal becomes active only when the owner or orchestrator approves it
and the row is copied into config/licence_allowlist.json (or, once migration 016 is applied, inserted as an active DB
row) with status "active" and approved_by set.

Procedure (bounded; one run handles at most --max licences, default 10):
  1. Export the queue (owner, or an agent with read-only DB access when one is authorised): run QUEUE_SQL below
     through the read-only account and save the rows as JSON lines {"licence", "count", "first_seen", "last_seen"}.
     Unknown licences come from vera_vjay.licence_observations rows with reason_code 'licence_unknown' (written on the
     /ask admission path when licence_observations_enabled is true). No DDL is needed.
  2. python scripts/licence_research.py queue --observations <export.jsonl> --out work/<task>/queue.json
     Drops strings any allowlist row already matches (any status, so pending proposals are not re-researched), merges
     spellings that normalise alike, ranks by count, keeps --max, and writes one research brief per licence.
  3. For each brief, the orchestrating agent spawns a `research-reader` child BY DEFINITION NAME (sanctioned route,
     AGENTS.md section 8; at most 4 children, 8 per tree, per research_config.json). The child fetches the licence's
     canonical text from its owner (the licensor's own site, SPDX, creativecommons.org) and returns the FINDING fields
     below. The agent saves the findings as JSON lines.
  4. python scripts/licence_research.py propose --findings <findings.jsonl> --out work/<task>/proposals.jsonl
     Classifies each finding against the owner's criteria (R1 2026-10-03: reject ONLY no-derivatives,
     no-educational-use and strictly-for-fee; plain NC accepted; storable only with an explicit store-and-redistribute
     grant; anything unclear is link-only) and appends PROPOSED rows with their evidence.
  5. The owner or orchestrator reviews the proposals and approves or declines each one (approval gate).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vera.licence_gate import ALLOWLIST_PATH, norm_id  # noqa: E402

PROPOSED = "proposed"
QUEUE_SQL = (
    "SELECT spdx AS licence, count(*) AS count, min(observed_at) AS first_seen, max(observed_at) AS last_seen "
    "FROM vera_vjay.licence_observations WHERE reason_code = 'licence_unknown' AND governing AND spdx IS NOT NULL "
    "GROUP BY spdx ORDER BY count(*) DESC")
FINDING_FIELDS = ("licence", "canonical_id", "evidence_url", "evidence_read_state", "quote",
                  "grants_store_and_redistribute", "nd", "nc", "sa", "no_educational_use", "fee_only",
                  "attribution_required")
MAX_QUOTE_WORDS = 25
DEFAULT_MAX = 10


class ResearchError(ValueError):
    pass


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                raise ResearchError(f"{path}:{n}: not JSON ({e})") from e
    return rows


def known_keys(allowlist_path: Path = ALLOWLIST_PATH) -> set[str]:
    """Normalised ids and aliases of EVERY allowlist row (any status)."""
    keys = set()
    for e in json.loads(allowlist_path.read_text(encoding="utf-8"))["entries"]:
        keys.update(norm_id(a) for a in [e["id"], *e.get("aliases", [])])
    keys.discard("")
    return keys


def build_queue(observations: list[dict], allowlist_path: Path = ALLOWLIST_PATH, max_items: int = DEFAULT_MAX) -> list[dict]:
    if not 1 <= max_items <= 50:
        raise ResearchError("--max must be 1..50 (bounded run)")
    known, merged = known_keys(allowlist_path), {}
    for o in observations:
        raw = str(o.get("licence") or "").strip()
        key = norm_id(raw)
        if not key or key in known:
            continue
        m = merged.setdefault(key, {"key": key, "spellings": [], "count": 0, "first_seen": None, "last_seen": None})
        if raw not in m["spellings"]:
            m["spellings"].append(raw)
        m["count"] += int(o.get("count") or 1)
        for f, pick in (("first_seen", min), ("last_seen", max)):
            v = o.get(f)
            if v:
                m[f] = v if m[f] is None else pick(m[f], v)
    queue = sorted(merged.values(), key=lambda m: (-m["count"], m["key"]))[:max_items]
    for m in queue:
        m["brief"] = (f"Licence string(s) {m['spellings']!r}. Find the licence's canonical text from its owner "
                      "(licensor site, SPDX or creativecommons.org). Return JSON with fields "
                      f"{list(FINDING_FIELDS)}; booleans true/false/null (null = not stated). Quote <= "
                      f"{MAX_QUOTE_WORDS} words. State the read level (full text, summarizer, snippet).")
    return queue


def classify(f: dict) -> tuple[bool, str, str]:
    """(storable, outcome, reason) under the owner's criteria. outcome is 'storable', 'link-only' or 'reject'."""
    if f.get("nd") is True:
        return False, "reject", "no derivatives (R1 reject class)"
    if f.get("no_educational_use") is True:
        return False, "reject", "forbids educational use (R1 reject class)"
    if f.get("fee_only") is True:
        return False, "reject", "strictly for fee (R1 reject class)"
    if f.get("grants_store_and_redistribute") is True and f.get("nd") is False:
        nc = " NonCommercial accepted (R1)." if f.get("nc") is True else ""
        return True, "storable", "explicit store-and-redistribute grant, derivatives allowed." + nc
    return False, "link-only", "no explicit store-and-redistribute grant or terms unclear (deny-all)"


def proposal(f: dict, now: str | None = None) -> dict:
    missing = [k for k in ("licence", "evidence_url", "evidence_read_state") if not f.get(k)]
    if missing:
        raise ResearchError(f"finding {f.get('licence')!r} lacks {missing}")
    if not str(f["evidence_url"]).startswith("https://") and not str(f["evidence_url"]).startswith("http://"):
        raise ResearchError(f"finding {f['licence']!r}: evidence_url is not an http(s) URL")
    if len(str(f.get("quote") or "").split()) > MAX_QUOTE_WORDS:
        raise ResearchError(f"finding {f['licence']!r}: quote over {MAX_QUOTE_WORDS} words")
    storable, outcome, why = classify(f)
    return {"id": f.get("canonical_id") or f"LicenseRef-{norm_id(f['licence'])}", "status": PROPOSED,
            "storable": storable, "proposed_outcome": outcome, "attribution_required": f.get("attribution_required"),
            "nc": f.get("nc"), "nd": f.get("nd"), "sa": f.get("sa"), "reason": f"PROPOSED: {why}",
            "evidence_url": f["evidence_url"], "evidence_read_state": f["evidence_read_state"],
            "evidence_quote": f.get("quote"), "approved_by": None, "aliases": [f["licence"]], "urls": [],
            "proposed_at": now or datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _guard_out(out: Path, allowlist_path: Path = ALLOWLIST_PATH) -> None:
    if out.resolve() == allowlist_path.resolve() or out.resolve().parent == allowlist_path.resolve().parent:
        raise ResearchError("refused: proposals never go into config/ (the allowlist is changed only on approval)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("queue")
    q.add_argument("--observations", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--max", type=int, default=DEFAULT_MAX)
    p = sub.add_parser("propose")
    p.add_argument("--findings", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        _guard_out(a.out)
        if a.cmd == "queue":
            items = build_queue(_read_jsonl(a.observations), max_items=a.max)
            a.out.write_text(json.dumps(items, indent=1) + "\n", encoding="utf-8")
            print(f"licence_research: queued {len(items)} unknown licence(s) -> {a.out}")
        else:
            rows = [proposal(f) for f in _read_jsonl(a.findings)]
            with a.out.open("a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            print(f"licence_research: wrote {len(rows)} PROPOSED row(s) -> {a.out}; none is active until approved")
    except (ResearchError, OSError, KeyError) as e:
        print(f"licence_research: FAILED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
