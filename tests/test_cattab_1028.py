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


def test_unreachable_or_unconfigured_db_fails_closed_not_seed():
    """Review D5: with read_from_db true, only 'table absent' may fall back to the seed."""
    def boom():
        raise sr.ScopeConfigError("classifier category table unreadable (OperationalError); failing closed")
    with pytest.raises(sr.ScopeConfigError):
        sr.load_category_table(ON, reader=boom)
    with pytest.raises(sr.ScopeConfigError):  # no DB URL configured
        sr.read_category_rows(env={})
    class Bad(_FakeEngine):
        def connect(self):
            raise OSError("host unreachable secret-host")
    with pytest.raises(sr.ScopeConfigError) as exc:
        sr.read_category_rows(Bad())
    assert "secret-host" not in str(exc.value) and "OSError" in str(exc.value)
    class SelectFails(_FakeEngine):
        def connect(self):
            c = _FakeConn(True, [], self.log)
            c.all = lambda: 1 / 0
            return c
    with pytest.raises(sr.ScopeConfigError):
        sr.read_category_rows(SelectFails())


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


def test_invalid_runtime_rows_are_skipped_and_logged_without_content(caplog):
    """Review D6: a runtime row can never refuse (DB constraint), so an invalid one is skipped, not fatal."""
    refusing = _runtime(state="retired")
    refusing.update(disposition="refuse", refusal_rank=7)
    aliased = _runtime(code="aliased_one")
    aliased["aliases"] = ["some_label"]
    quoted = _runtime(code="quoted_one", desc='has "quotes" in it')
    badcode = _runtime(code="Bad Code")
    copycat = _runtime(code="copycat", desc="a request to write or fix code", state="retired")
    with caplog.at_level(logging.WARNING, logger="vera"):
        cats = sr.load_category_table(ON, reader=lambda: _rows() + [refusing, aliased, quoted, badcode, copycat, _runtime()])
    assert [c.code for c in cats] == [c.code for c in SEED] + ["reading_group"]
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "5 invalid runtime row(s) skipped" in msg and "aliased_one" not in msg and "some_label" not in msg


def test_invalid_seed_or_admin_rows_still_fail_closed():
    bad = _rows()
    bad[2]["origin"] = "seed"
    bad[2]["disposition"] = "bogus"
    with pytest.raises(sr.ScopeConfigError):
        sr.load_category_table(ON, reader=lambda: bad)
    seed_alias = _rows(advice={"aliases": ["dangerous"]})
    seed_alias.append(dict(_rows()[3], code="harmful2", description="an unrelated gardening topic", refusal_rank=9,
                           aliases=["dangerous"]))
    with pytest.raises(sr.ScopeConfigError):
        sr.load_category_table(ON, reader=lambda: seed_alias)
    with pytest.raises(sr.ScopeConfigError):  # structural fault in a runtime-labelled row still fails closed
        sr.load_category_table(ON, reader=lambda: _rows() + [{"origin": "runtime", "code": "x"}])


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
    other = _rows() + [dict(_runtime(code="copycat", desc="a request to write or fix code", state="retired"), origin="seed")]
    with pytest.raises(sr.ScopeConfigError):  # a non-runtime row too similar to another code still fails closed
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
    assert len(out) == 6
    assert eng.log[0] == "SET TRANSACTION READ ONLY"  # D8: read-only transaction first
    assert all(q.lstrip().upper().startswith("SELECT") for q in eng.log[1:])
    assert sr.read_category_rows(_FakeEngine(False)) is None  # table absent -> None (seed fallback)


def test_reader_without_db_config_fails_closed():
    with pytest.raises(sr.ScopeConfigError):
        sr.read_category_rows(env={})  # RuntimeError (env unset) -> unreadable, no URL printed


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
    with pytest.raises(sr.RegistrationRefused):  # table absent (rows None): nothing written
        sr.enrol_nonrefusing_entry("ok_code", "notes about a local seminar timetable", settings=on, rows=None,
                                   writer=lambda r: 1 / 0, engine=_FakeEngine(False))


def test_registration_description_syntax_limits():
    """Review D7: cap 120 chars, printable ASCII only (DB CHECK character set), no newline or control chars."""
    on = {"registration_enabled": True}
    ok = "x" * 3 + " notes about a local seminar timetable"
    sr.enrol_nonrefusing_entry("ok_code", ok, settings=on, rows=_rows(), writer=lambda r: None)
    for bad in ("a" * 121, "line one\nline two seminar", "tab\tseparated seminar notes", "bell\x07 seminar notes",
                "caf\u00e9 seminar notes", "del\x7f seminar notes", "brace { seminar notes", "\n"):
        with pytest.raises(sr.RegistrationRefused):
            sr.enrol_nonrefusing_entry("ok_code", bad, settings=on, rows=_rows(), writer=lambda r: 1 / 0)
    sr.enrol_nonrefusing_entry("ok_code", "q" * 120, settings=on, rows=_rows(), writer=lambda r: None)
    assert "NEVER call this with raw user text" in sr.enrol_nonrefusing_entry.__doc__


@pytest.mark.xfail(reason="TODO D7: semantic overlap check against refusing descriptions not built; REQUIRED before "
                          "classifier_category_table.registration_enabled may be enabled", strict=True)
def test_registration_rejects_synonym_paraphrase_of_a_refusing_description():
    harmful_syn = "requesting operational assistance in causing harms"  # scores 0.429 lexically: passes today
    with pytest.raises(sr.RegistrationRefused):
        sr.enrol_nonrefusing_entry("helper_cat", harmful_syn, settings={"registration_enabled": True}, rows=_rows(),
                                   writer=lambda r: None)


class _TxnConn(_FakeConn):
    def execute(self, stmt, params=None):
        self.log.append((str(stmt), params))
        self.last = str(stmt)
        return self


class _TxnEngine:
    def __init__(self, exists=True, rows=None):
        self.log, self.exists, self.rows, self.committed = [], exists, rows or [], False

    def begin(self):
        eng = self

        class Ctx(_TxnConn):
            def __exit__(self, et, ev, tb):
                eng.committed = et is None
                return False
        return Ctx(self.exists, self.rows, self.log)


def test_enrolment_runs_lock_read_insert_in_one_transaction():
    """Review D6: advisory lock first, then re-read, then INSERT, all on one transaction-scoped connection."""
    eng = _TxnEngine(True, _rows())
    row = sr.enrol_nonrefusing_entry("reading_group", "notes about a local seminar timetable",
                                     settings={"registration_enabled": True}, engine=eng)
    sqls = [q for q, _ in eng.log]
    assert sqls[0] == sr.DB_CATEGORY_SQL_LOCK and "pg_advisory_xact_lock" in sqls[0]
    assert eng.log[0][1] == {"k": sr.ENROL_ADVISORY_LOCK_KEY} == {"k": 10281}
    assert sqls[1] == sr.DB_CATEGORY_SQL_EXISTS and sqls[2] == sr.DB_CATEGORY_SQL_ROWS
    assert sqls[-1] == sr.DB_CATEGORY_SQL_INSERT and eng.committed and row["code"] == "reading_group"
    # a refused candidate under the lock writes nothing and rolls back
    eng2 = _TxnEngine(True, _rows())
    with pytest.raises(sr.RegistrationRefused):
        sr.enrol_nonrefusing_entry("other_code", "a request to write or fix code",
                                   settings={"registration_enabled": True}, engine=eng2)
    assert sr.DB_CATEGORY_SQL_INSERT not in [q for q, _ in eng2.log] and not eng2.committed


# ---- migration draft: text-level checks only (the SQL is NOT run; triggers untested against a real PostgreSQL) ----
SQL = MIG.read_text()


def test_migration_seed_matches_json_seed_with_version_and_active():
    m = re.search(r"INSERT INTO vera_vjay\.classifier_categories.*?\(VALUES\s*(.*?)\)\s+AS v\(", SQL, re.S)
    rows = re.findall(r"\('([a-z_]+)', 1, '([^']*)', '([a-z_]+)', (NULL::integer|\d+), '\{\}'::text\[\], 'seed', 'active'\)", m.group(1))
    assert len(rows) == 6
    got = {r[0]: (r[1], r[2], None if r[3].startswith("NULL") else int(r[3])) for r in rows}
    assert got == {c.code: (c.description, c.disposition, c.refusal_rank) for c in SEED}
    assert "state          TEXT NOT NULL DEFAULT 'active'" in SQL
    assert "WHERE NOT EXISTS (SELECT 1 FROM vera_vjay.classifier_categories t WHERE t.code = v.code)" in SQL
    assert "ON CONFLICT" not in SQL and "Idempotent per code" in SQL


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


def test_migration_review_fixes_d1_to_d4_d8_d9_text():
    # D1: retire-UPDATE only when another active row of the code exists
    act = SQL[SQL.index("FUNCTION vera_vjay.classifier_categories_activate"):SQL.index("DROP TRIGGER IF EXISTS classifier_categories_a_guard")]
    assert re.search(r"AND EXISTS \(SELECT 1 FROM vera_vjay\.classifier_categories c\s+WHERE c\.code = NEW\.code AND c\.state = 'active'", act)
    assert act.index("AND EXISTS") < act.index("UPDATE vera_vjay.classifier_categories")
    # D2: default-deny, owner-only, no pg_has_role/MEMBER and no is_app_role
    assert "pg_has_role" not in SQL and "MEMBER" not in SQL and "is_app_role" not in SQL
    assert "pg_catalog.pg_get_userbyid(c.relowner) = current_user" in SQL and "MUST BE RUN AS THE OWNER ROLE" in SQL
    assert "DEFAULT-DENY" in SQL and "vera_vjay_admin" not in SQL
    # D3: no default on version; NULL auto-assigns; supplied must be max+1
    assert not re.search(r"version\s+INTEGER NOT NULL DEFAULT", SQL)
    assert "IF NEW.version IS NULL THEN" in SQL and "NEW.version <> nxt" in SQL
    # D4: every function pins search_path
    funcs = re.findall(r"CREATE OR REPLACE FUNCTION [^\n]*\n?[^$]*?\$f\$", SQL)
    assert len(funcs) == 5 and all("SET search_path = pg_catalog, pg_temp" in f for f in funcs)
    # D8 / D9 documentation
    assert "wo: NO grant at all" in SQL and "006 and 007 grant wo INSERT" in SQL
    assert "compares DISPOSITION and REFUSAL RANK" in SQL and "It does NOT compare description or aliases" in SQL


def test_rollback_draft_drops_in_dependency_order_and_is_owner_gated():
    rb = (ROOT / "db" / "migrations" / "rollback" / "008-DRAFT-classifier-categories-rollback.sql.draft").read_text()
    assert "NOT EXECUTED" in rb and "DESTRUCTIVE" in rb and "owner" in rb.lower()
    order = [rb.index(x) for x in ("DROP TRIGGER IF EXISTS classifier_categories_a_guard", "DROP TABLE IF EXISTS vera_vjay.classifier_categories",
                                   "DROP FUNCTION IF EXISTS vera_vjay.classifier_categories_guard", "classifier_categories_is_owner()")]
    assert order == sorted(order)
    assert "DROP SCHEMA" not in rb and "postgres://" not in rb
