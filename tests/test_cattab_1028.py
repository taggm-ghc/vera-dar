"""CATTAB-1028: classifier category table in the database (offline: fakes only, no DB, no network).
The SQL trigger/CHECK logic is NOT exercised here (no PostgreSQL); only the migration text is inspected."""
import copy
import json
import logging
import re
from pathlib import Path

import pytest

from vera import scope_router as sr

ROOT = Path(__file__).resolve().parent.parent
MIG = ROOT / "db" / "migrations" / "pending" / "008-DRAFT-classifier-categories.sql.draft"
SEED = sr.load_categories()


def _rows(**over):
    rows = [{"code": c.code, "description": c.description, "disposition": c.disposition, "refusal_rank": c.refusal_rank,
             "aliases": list(c.aliases), "origin": "seed", "state": "active", "version": 1} for c in SEED]
    for code, patch in over.items():
        for r in rows:
            if r["code"] == code:
                r.update(patch)
    return rows


def _runtime(code="reading_group", desc="notes about a local seminar timetable", state="active", disp="off_topic", ver=1):
    return {"code": code, "description": desc, "disposition": disp, "refusal_rank": None, "aliases": [],
            "origin": "runtime", "state": state, "version": ver}


ON = {"read_from_db": True}


def test_toggles_default_off_and_shipped_config_is_off():
    s = sr.category_table_settings()
    assert s.get("read_from_db") is False and s.get("registration_enabled") is False
    assert s.get("registration_state") == "active"
    assert sr.CATEGORY_TABLE == SEED  # shipped behaviour unchanged
    with pytest.raises(sr.RegistrationRefused):
        sr.enrol_nonrefusing_entry("new_cat", "something entirely different here", settings={}, rows=_rows(), writer=lambda r: 1 / 0)
    with pytest.raises(sr.RegistrationRefused):  # only explicit boolean true enables
        sr.enrol_nonrefusing_entry("new_cat", "something entirely different here", settings={"registration_enabled": "true"},
                             rows=_rows(), writer=lambda r: 1 / 0)


def test_read_from_db_off_never_calls_reader():
    assert sr.load_category_table({"read_from_db": "yes"}, reader=lambda: 1 / 0) == SEED


def test_db_path_loads_active_rows_in_order():
    cats = sr.load_category_table(ON, reader=lambda: _rows())
    assert cats == SEED
    cats = sr.load_category_table(ON, reader=lambda: _rows() + [_runtime()])
    assert [c.code for c in cats][-1] == "reading_group" and len(cats) == 7
    assert "reading_group" not in sr.REFUSE_CATEGORIES


def test_absent_table_falls_back_to_seed_and_logs_once(caplog):
    sr._FALLBACK_LOGGED = False
    with caplog.at_level(logging.INFO, logger="vera"):
        assert sr.load_category_table(ON, reader=lambda: None) == SEED
        assert sr.load_category_table(ON, reader=lambda: None) == SEED
    assert sum("using the JSON seed" in r.getMessage() for r in caplog.records) == 1
    sr._FALLBACK_LOGGED = False


def test_invalid_table_fails_closed():
    bad = _rows()
    bad[0]["disposition"] = "bogus"
    for rows in ([], bad, _rows(harmful={"disposition": "off_topic", "refusal_rank": None}),
                 _rows(harmful={"description": "a request to write or fix code"}),
                 _rows(harmful={"origin": "other"}), _rows(harmful={"version": 0})):
        with pytest.raises(sr.ScopeConfigError):
            sr.load_category_table(ON, reader=lambda rows=rows: rows)


def test_core_refusals_must_be_active():
    for code in ("harmful", "rule_change", "advice", "programming_task"):
        for st in ("draft", "suspended", "retired"):
            with pytest.raises(sr.ScopeConfigError):
                sr.load_category_table(ON, reader=lambda c=code, s=st: _rows(**{c: {"state": s}}))
    with pytest.raises(sr.ScopeConfigError):  # a missing core row
        sr.load_category_table(ON, reader=lambda: [r for r in _rows() if r["code"] != "advice"])
    with pytest.raises(sr.ScopeConfigError):  # changed rank
        sr.load_category_table(ON, reader=lambda: _rows(advice={"refusal_rank": 9}))


def test_non_active_runtime_rows_ignored_but_validated():
    for st in ("draft", "suspended", "retired"):
        cats = sr.load_category_table(ON, reader=lambda s=st: _rows() + [_runtime(state=s)])
        assert cats == SEED
    # a runtime row may not be refusing, even if retired
    bad = _runtime(state="retired")
    bad.update(disposition="refuse", refusal_rank=7)
    with pytest.raises(sr.ScopeConfigError):
        sr.load_category_table(ON, reader=lambda: _rows() + [bad])
    bad = _runtime()
    bad["aliases"] = ["some_label"]
    with pytest.raises(sr.ScopeConfigError):
        sr.load_category_table(ON, reader=lambda: _rows() + [bad])


def test_version_logged_no_secrets(caplog):
    with caplog.at_level(logging.INFO, logger="vera"):
        sr.load_category_table(ON, reader=lambda: _rows() + [_runtime(ver=4)])
    assert any("max version 4" in r.getMessage() for r in caplog.records)


def _new_version(code, ver, state, **patch):
    r = next(dict(x) for x in _rows() if x["code"] == code)
    r.update(version=ver, state=state, **patch)
    return r


def test_new_version_activation_replaces_old_version():
    rows = _rows(harmful={"state": "retired"})
    desc = "asking for hands-on assistance to injure somebody"
    rows.append(_new_version("harmful", 2, "active", description=desc))
    cats = sr.load_category_table(ON, reader=lambda: rows)
    h = next(c for c in cats if c.code == "harmful")
    assert h.description == desc and h.disposition == "refuse" and h.refusal_rank == 1 and len(cats) == 6


def test_draft_new_version_is_ignored_old_stays_active():
    rows = _rows() + [_new_version("harmful", 2, "draft", description="asking for hands-on assistance to injure somebody")]
    assert sr.load_category_table(ON, reader=lambda: rows) == SEED


def test_two_active_rows_for_one_code_and_duplicate_version_fail_closed():
    rows = _rows() + [_new_version("harmful", 2, "active", description="asking for hands-on assistance to injure somebody")]
    with pytest.raises(sr.ScopeConfigError):
        sr.load_category_table(ON, reader=lambda: rows)
    rows = _rows() + [_new_version("harmful", 1, "retired", description="asking for hands-on assistance to injure somebody")]
    with pytest.raises(sr.ScopeConfigError):
        sr.load_category_table(ON, reader=lambda: rows)


def test_new_version_cannot_turn_a_refusal_off_or_change_rank():
    for code in ("harmful", "rule_change", "advice"):
        rows = _rows(**{code: {"state": "retired"}})
        rows.append(_new_version(code, 2, "active", disposition="off_topic", refusal_rank=None,
                                 description="totally unrelated words about gardens"))
        with pytest.raises(sr.ScopeConfigError):
            sr.load_category_table(ON, reader=lambda rows=rows: rows)
    rows = _rows(advice={"state": "retired"}) + [_new_version("advice", 2, "active", refusal_rank=9,
                                                              description="totally unrelated words about gardens")]
    with pytest.raises(sr.ScopeConfigError):
        sr.load_category_table(ON, reader=lambda: rows)
    rows = _rows(harmful={"state": "retired"})  # no active row left for harmful
    with pytest.raises(sr.ScopeConfigError):
        sr.load_category_table(ON, reader=lambda: rows)


def test_dissimilarity_spans_other_codes_any_state_but_not_own_older_versions():
    own = _rows(harmful={"state": "retired"}) + [_new_version("harmful", 2, "active")]  # same text as own v1: allowed
    assert [c.code for c in sr.load_category_table(ON, reader=lambda: own)] == [c.code for c in SEED]
    other = _rows() + [_runtime(code="copycat", desc="a request to write or fix code", state="retired")]
    with pytest.raises(sr.ScopeConfigError):
        sr.load_category_table(ON, reader=lambda: other)


class _FakeConn:
    def __init__(self, exists, rows, log):
        self.exists, self.rows, self.log = exists, rows, log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.log.append(sql)
        self.last = sql
        return self

    def scalar(self):
        return "vera_vjay.classifier_categories" if self.exists else None

    def mappings(self):
        return self

    def all(self):
        return self.rows


class _FakeEngine:
    def __init__(self, exists=True, rows=None):
        self.log, self.exists, self.rows = [], exists, rows or []

    def connect(self):
        return _FakeConn(self.exists, self.rows, self.log)


def test_reader_uses_select_only_through_engine():
    eng = _FakeEngine(True, _rows())
    out = sr.read_category_rows(eng)
    assert len(out) == 6 and all(q.lstrip().upper().startswith("SELECT") for q in eng.log)
    assert sr.read_category_rows(_FakeEngine(False)) is None


def test_reader_without_db_config_returns_none():
    assert sr.read_category_rows(env={}) is None  # RuntimeError (env unset) -> unavailable, no URL printed


def test_registration_ok_and_state_from_config():
    got = []
    row = sr.enrol_nonrefusing_entry("reading_group", "notes about a local seminar timetable",
                               settings={"registration_enabled": True}, rows=_rows(), writer=got.append)
    assert got == [row] and row["disposition"] == "off_topic" and row["state"] == "active"
    sr.enrol_nonrefusing_entry("reading_group", "notes about a local seminar timetable", "in_scope",
                         settings={"registration_enabled": True, "registration_state": "draft"}, rows=_rows(), writer=got.append)
    assert got[-1]["state"] == "draft" and got[-1]["disposition"] == "in_scope"
    with pytest.raises(sr.RegistrationRefused):
        sr.enrol_nonrefusing_entry("reading_group", "notes about a local seminar timetable",
                             settings={"registration_enabled": True, "registration_state": "deleted"}, rows=_rows(), writer=got.append)


@pytest.mark.parametrize("disp", ["refuse", "unsupported_use", "bogus"])
def test_registration_rejects_refusing_dispositions(disp):
    with pytest.raises(sr.RegistrationRefused):
        sr.enrol_nonrefusing_entry("reading_group", "notes about a local seminar timetable", disp,
                             settings={"registration_enabled": True}, rows=_rows(), writer=lambda r: 1 / 0)


def test_registration_rejects_similar_description_including_retired_rows():
    on = {"registration_enabled": True}
    with pytest.raises(sr.RegistrationRefused):  # near copy of an active description
        sr.enrol_nonrefusing_entry("other_code", "a request to write or fix code", settings=on, rows=_rows(), writer=lambda r: 1 / 0)
    retired = _runtime(state="retired")
    with pytest.raises(sr.RegistrationRefused):  # retired description cannot return under a new code
        sr.enrol_nonrefusing_entry("other_code", retired["description"], settings=on, rows=_rows() + [retired], writer=lambda r: 1 / 0)
    with pytest.raises(sr.RegistrationRefused):  # existing code
        sr.enrol_nonrefusing_entry("harmful", "notes about a local seminar timetable", settings=on, rows=_rows(), writer=lambda r: 1 / 0)
    with pytest.raises(sr.RegistrationRefused):  # bad code / description
        sr.enrol_nonrefusing_entry("Bad-Code", "notes about a local seminar timetable", settings=on, rows=_rows(), writer=lambda r: 1 / 0)
    with pytest.raises(sr.RegistrationRefused):
        sr.enrol_nonrefusing_entry("ok_code", 'has "quotes"', settings=on, rows=_rows(), writer=lambda r: 1 / 0)
    with pytest.raises(sr.RegistrationRefused):  # table absent (reader returns None): nothing written
        sr.enrol_nonrefusing_entry("ok_code", "notes about a local seminar timetable", settings=on, rows=None,
                                   writer=lambda r: 1 / 0) if sr.read_category_rows(env={}) is None else None


# ---- migration draft: text-level checks only (the SQL is NOT run; triggers untested against a real PostgreSQL) ----
SQL = MIG.read_text()


def test_migration_seed_matches_json_seed_with_version_and_active():
    m = re.search(r"INSERT INTO vera_vjay\.classifier_categories.*?VALUES\s*(.*?)ON CONFLICT", SQL, re.S)
    rows = re.findall(r"\('([a-z_]+)', 1, '([^']*)', '([a-z_]+)', (NULL|\d+), '\{\}', 'seed', 'active'\)", m.group(1))
    assert len(rows) == 6
    got = {r[0]: (r[1], r[2], None if r[3] == "NULL" else int(r[3])) for r in rows}
    assert got == {c.code: (c.description, c.disposition, c.refusal_rank) for c in SEED}
    assert "version        INTEGER NOT NULL DEFAULT 1" in SQL and "state          TEXT NOT NULL DEFAULT 'active'" in SQL
    assert "ON CONFLICT (code, version) DO NOTHING" in SQL


def test_migration_text_privileges_and_checks():
    assert "CHECK (state IN ('draft','active','suspended','retired'))" in SQL
    assert "CHECK (disposition IN ('in_scope','off_topic','refuse','unsupported_use'))" in SQL
    assert "REVOKE UPDATE, DELETE, TRUNCATE" in SQL
    assert "GRANT SELECT ON vera_vjay.classifier_categories TO vera_vjay_ro" in SQL
    assert "GRANT SELECT, INSERT ON vera_vjay.classifier_categories TO vera_vjay_rw" in SQL
    assert not re.search(r"GRANT[^;]*(UPDATE|DELETE|TRUNCATE)[^;]*TO", SQL)
    assert "origin <> 'runtime' OR (disposition IN ('in_scope','off_topic')" in SQL
    assert "TIMESTAMPTZ" in SQL and "UNCONFIRMED" in SQL and "007" in SQL and "OWNER-APPROVED DDL" in SQL
    assert "UNTESTED against a real PostgreSQL" in SQL


def test_migration_text_versions_as_rows():
    assert "UNIQUE (code, version)" in SQL
    assert re.search(r"UNIQUE INDEX[^;]*\(code\)\s+WHERE state = 'active'", SQL)
    assert re.search(r"UNIQUE INDEX[^;]*\(refusal_rank\)\s+WHERE state = 'active' AND refusal_rank IS NOT NULL", SQL)
    assert "content is immutable" in SQL and "NEW.description IS DISTINCT FROM OLD.description" in SQL
    assert "SET state = 'retired', modified_at = now()" in SQL  # activation retires the previous active row
    assert "BEFORE INSERT OR UPDATE ON vera_vjay.classifier_categories\n  FOR EACH ROW EXECUTE FUNCTION vera_vjay.classifier_categories_activate" in SQL
    assert "app roles may insert only brand-new codes" in SQL
    assert "OLD.version + 1" not in SQL  # the in-place version bump is gone
    assert "c.state = 'active' AND c.code <> NEW.code" in SQL  # aliases unique across active rows of other codes


def test_no_secret_in_draft():
    assert "postgres://" not in SQL and "postgresql://" not in SQL
