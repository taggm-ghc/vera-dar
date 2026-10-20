"""safety/destructive.py (PLAN-ENV v2.1 §3 E5, §5): decorator, context manager, static scan with pinned allowlist."""
from pathlib import Path

import pytest

from safety import destructive as d
from safety import endpoint_class as ec
from safety import sql_guard as sg

ROOT = Path(__file__).resolve().parents[1]

PINNED_ALLOWLIST = {
    "db/migration_guard.py", "db/apply_migrations.py", "scripts/live_catalog_check.py", "vera/dar_store.py",
    "vera/m6/test_m6_schema_sync.py",
}


class _Tagged:
    def __init__(self, verdict):
        self.info = {sg.VERDICT_KEY: verdict}


def test_decorator_refuses_production_and_untagged():
    calls = []

    @d.require_disposable
    def wipe(conn, x):
        calls.append(x)

    for conn in (_Tagged(ec.Verdict(ec.PRODUCTION, ("inconclusive",))), object(), _Tagged(None)):
        with pytest.raises(d.DestructiveRefused) as ei:
            wipe(conn, 1)
        assert ei.value.code == "not_disposable"
    assert calls == []
    assert wipe.__name__ == "wipe"


def test_context_manager_refuses():
    with pytest.raises(d.DestructiveRefused):
        with d.destructive_guard(_Tagged(ec.Verdict(ec.PRODUCTION, ()))):
            raise AssertionError("body must not run")


def test_destructive_refused_is_the_sql_guard_class():
    assert d.DestructiveRefused is sg.DestructiveRefused


def test_allowlist_is_pinned_and_each_entry_has_a_reason():
    assert set(d.ALLOWLIST) == PINNED_ALLOWLIST
    assert all(isinstance(r, str) and r.strip() for r in d.ALLOWLIST.values())


def test_repo_has_no_unallowed_destructive_sql_literals():
    assert d.unallowed_findings(ROOT) == {}


def test_scan_detects_literals_and_docstrings(tmp_path):
    (tmp_path / "vera").mkdir()
    (tmp_path / "vera" / "bad.py").write_text('"""DROP TABLE x"""\nq = "delete from t where a=1"\nok = "SELECT 1"\n')
    (tmp_path / "vera" / "fine.py").write_text("x = 'select 1'  # DELETE FROM t in a comment is not a literal\n")
    assert d.scan_tree(tmp_path) == {"vera/bad.py": [1, 2]}
    assert d.scan_source("x = 'GRANT SELECT ON t TO r'") == [1]
    assert d.scan_source("x = 'please grant access'") == []
