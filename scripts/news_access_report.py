"""News-access exceptions report (H19H21-DESIGN-1106 T7). Read-only and offline.

    python3 -I -B scripts/news_access_report.py [--config PATH] [--today YYYY-MM-DD]

Reads the outlet rows from config/reporting_lane.json (the seed; the DB list views are NOT read here, so the script never
opens a connection, reads no env file and makes no network call). For each row that has an `access` block whose derived
mode (vera.news_access.mode) is link_only it prints one line: host, failing tokens, checked age, evidence URL.
Exit 0 = report printed, 2 = usage / news_access (T1) not available.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO / "config" / "reporting_lane.json"


def _split(res):
    """Accept a str mode, (mode, reasons), or an object/dict with mode and reasons."""
    if isinstance(res, str):
        return res, []
    if isinstance(res, (tuple, list)) and len(res) == 2:
        return res[0], list(res[1] or [])
    if isinstance(res, dict):
        return res.get("mode"), list(res.get("reasons") or [])
    return getattr(res, "mode", None), list(getattr(res, "reasons", None) or [])


def _checked(access: dict) -> str | None:
    dates = [v.get("checked") for v in access.values() if isinstance(v, dict) and isinstance(v.get("checked"), str)]
    return max(dates) if dates else None


def _evidence(access: dict) -> str:
    for k in ("licence", "nc_eligible", "terms_ai_clause"):
        v = access.get(k)
        if isinstance(v, dict) and v.get("evidence_url"):
            return str(v["evidence_url"])
    return "-"


def report(rows, mode_fn, today: dt.date) -> list[str]:
    """One line per row with an `access` block whose mode is not full_text (fail closed: unknown mode counts as link_only)."""
    out = []
    for row in rows or []:
        if not isinstance(row, dict) or not isinstance(row.get("access"), dict):
            continue
        host = str(row.get("host", "?"))
        mode, reasons = _split(mode_fn("https://" + host + "/"))
        if mode == "full_text":
            continue
        chk = _checked(row["access"])
        try:
            age = f"{(today - dt.date.fromisoformat(chk)).days}d" if chk else "never"
        except ValueError:
            age = "bad-date"
        out.append(f"{host}\tlink_only\t{','.join(reasons) or 'no_reason_given'}\tchecked={chk or '-'} ({age})\t{_evidence(row['access'])}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--today", default=None)
    a = ap.parse_args(argv)
    try:
        today = dt.date.fromisoformat(a.today) if a.today else dt.date.today()
        cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
        rows = cfg.get("outlets", [])
    except (OSError, ValueError) as e:
        print(f"news_access_report: cannot read input ({type(e).__name__})", file=sys.stderr)
        return 2
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    try:
        from vera import news_access
    except ImportError:
        print("news_access_report: vera.news_access (T1) is not available", file=sys.stderr)
        return 2
    def mode_fn(url):  # news_access.mode returns a str; reasons() is separate; both take the same cfg dict
        return news_access.mode(url, cfg), news_access.reasons(url, cfg)

    lines = report(rows, mode_fn, today)
    print(f"news-access exceptions: {len(lines)} row(s) with an access block are link_only")
    for ln in lines:
        print(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
