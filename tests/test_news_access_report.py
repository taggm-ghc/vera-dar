"""T7 (H19H21-DESIGN-1106): the exceptions report is offline and lists every unsatisfied access row with its tokens."""
import datetime as dt
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("news_access_report", ROOT / "scripts" / "news_access_report.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

TODAY = dt.date(2026, 11, 6)
ROWS = [
    {"host": "a.example", "access": {"licence": {"evidence_url": "https://a.example/l", "checked": "2026-11-01"}}},
    {"host": "b.example", "access": {"licence": {"checked": "2026-11-05"}}},
    {"host": "c.example"},  # no access block: unchanged non-ND behaviour, not reported
    {"host": "d.example", "access": {}},
]


def fake(url):
    h = url.split("//")[1].split("/")[0]
    return {"a.example": ("link_only", ["acc_nc_unresolved", "acc_robots"]), "b.example": ("full_text", [])}.get(h, "weird")


def test_lists_every_unsatisfied_access_row_with_tokens():
    out = R.report(ROWS, fake, TODAY)
    assert len(out) == 2
    assert "a.example" in out[0] and "acc_nc_unresolved,acc_robots" in out[0] and "(5d)" in out[0] and "https://a.example/l" in out[0]
    assert out[1].startswith("d.example") and "no_reason_given" in out[1] and "never" in out[1]  # unknown mode fails closed
    assert not any("b.example" in o or "c.example" in o for o in out)


def test_split_accepts_shapes():
    assert R._split("full_text") == ("full_text", [])
    assert R._split({"mode": "link_only", "reasons": ["x"]}) == ("link_only", ["x"])


def test_script_is_offline_no_db_no_env():
    src = (ROOT / "scripts" / "news_access_report.py").read_text()
    for bad in ("import socket", "urllib", "requests", "httpx", "psycopg", "sqlalchemy", "os.environ", "config/env", ".env"):
        assert bad not in src


def test_019_draft_pair_exists_and_is_inert():
    up = (ROOT / "db/migrations/pending/019-DRAFT-news-access-exceptions.sql.draft").read_text()
    down = (ROOT / "db/migrations/rollback/019-DRAFT-news-access-exceptions-rollback.sql.draft").read_text()
    assert "NOT EXECUTED" in up and "NOT EXECUTED" in down
    assert "vera_vjay.news_access_exceptions" in up and "DROP TABLE IF EXISTS vera_vjay.news_access_exceptions" in down
    assert "CASCADE" not in down.replace("No CASCADE", "") and "DELETE" not in up.split("GRANT SELECT")[1]
