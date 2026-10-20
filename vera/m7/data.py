"""Where the demo gets its report. Real runs come from vera_vjay.runs.eval_metrics (Postgres,
not Render's ephemeral filesystem). If none can be read, return the clearly-labelled FIXTURE and
say exactly why, so a demo never silently shows invented data as a real run."""
from __future__ import annotations

import json
import os


def load_latest_report() -> tuple[dict, str]:
    """Return (report, source_note)."""
    if os.getenv("VERA_DEMO_SOURCE") == "fixture":
        from vera.m7.fixture import build_fixture_report
        return build_fixture_report(), "VERA_DEMO_SOURCE=fixture"
    reason = ""
    try:
        from sqlalchemy import text
        from vera.db import get_session
        with get_session() as s:
            row = s.execute(text("SELECT run_id::text, eval_metrics FROM vera_vjay.runs "
                                 "WHERE eval_metrics IS NOT NULL ORDER BY evaluated_at DESC NULLS LAST "
                                 "LIMIT 1")).first()
        if row is not None:
            rep = row[1] if isinstance(row[1], dict) else json.loads(row[1])
            return rep, f"vera_vjay.runs run_id={row[0]}"
        reason = "vera_vjay.runs has no evaluated run yet (M6 has not been run)"
    except Exception as e:  # report verbosely, never swallow silently
        reason = f"database unavailable: {type(e).__name__}: {str(e)[:200]}"
    from vera.m7.fixture import build_fixture_report
    return build_fixture_report(), "FIXTURE fallback because " + reason
