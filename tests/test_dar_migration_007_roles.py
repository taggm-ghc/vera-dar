"""VERA-DAR 007: role-dependency tests. Static only (no DB). The dynamic proof lives in the lead's mig007 scratch evidence."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIG = ROOT / "db" / "migrations" / "pending" / "007-DRAFT-domain-registry.sql.draft"
ACCOUNTS = ROOT / "db" / "accounts" / "create_vera_accounts.sql"
README = ROOT / "db" / "README.md"


def test_roles_script_creates_roles_007_requires():
    text = ACCOUNTS.read_text(encoding="utf-8")
    for role in ("vera_vjay_wo", "vera_vjay_ro"):
        assert f"CREATE ROLE {role}" in text


def test_007_preflight_precedes_first_statement_that_creates_anything():
    code = "\n".join(l.split("--", 1)[0] for l in MIG.read_text(encoding="utf-8").splitlines())
    code = "\n".join(l for l in code.splitlines() if not l.startswith("\\set "))
    first_stmt = code.strip().split(";", 1)[0]
    assert first_stmt.lstrip().startswith("DO $$")  # preflight is the first statement
    assert "RAISE EXCEPTION" in code.split("CREATE SCHEMA")[0]


def test_readme_documents_order():
    text = README.read_text(encoding="utf-8")
    assert re.search(r"create_vera_accounts\.sql.*001-005.*007", text, re.S)
    assert "fail" in text.lower() and "007" in text
