"""safety/sql_guard.py (PLAN-ENV v2.1 §3 E8/C7, L3, §5): allowlist on PRODUCTION-tagged connections. Offline."""
from pathlib import Path

import pytest

from safety import endpoint_class as ec
from safety import sql_guard as sg
from safety.sql_guard import DestructiveRefused, check_sql

ROOT = Path(__file__).resolve().parents[1]

# The three real app UPDATEs (O10), verbatim apart from the f-string schema substitution in run_mvp.py.
APP_UPDATES = [
    ("vera/pipeline_store.py", "gate_c_decision=:g where run_id=:r",
     "update vera_vjay.runs set engineered_response=:f, final_answer_id=:a, verification_status=:v, "
     "gate_c_decision=:g where run_id=:r"),
    ("vera/run_mvp.py", "evaluated_at = now() WHERE run_id = :id",
     "UPDATE vera_vjay.runs SET baseline_response = :b, "
     "eval_metrics = CAST(:m AS jsonb), evaluated_at = now() WHERE run_id = :id"),
    ("vera/m6/m6_runner.py", "evaluated_at = now() WHERE run_id::text = :id",
     "UPDATE vera_vjay.runs SET baseline_response = :b, "
     "eval_metrics = CAST(:m AS jsonb), evaluated_at = now() WHERE run_id::text = :id"),
]


@pytest.mark.parametrize("path,fragment,sql", APP_UPDATES)
def test_real_app_updates_pass(path, fragment, sql):
    assert fragment in (ROOT / path).read_text(), "app UPDATE changed; re-pin it here"
    check_sql(sql)


@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "select a, b from vera_vjay.runs where run_id = %s",
    "(SELECT 1) UNION (SELECT 2)",
    "INSERT INTO vera_vjay.t (a) VALUES (:a) ON CONFLICT (a) DO UPDATE SET a = excluded.a",
    "INSERT INTO vera_vjay.t (a) VALUES (:a) ON CONFLICT DO NOTHING RETURNING id",
    "WITH x AS (SELECT 1 AS a) SELECT a FROM x",
    "WITH u AS (UPDATE t SET a = 1 WHERE id = :i RETURNING id) SELECT id FROM u",
    "SELECT * FROM t WHERE id = 1 FOR UPDATE",
    "SELECT * FROM t FOR NO KEY UPDATE",
    "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT s1", "RELEASE SAVEPOINT s1", "ROLLBACK TO SAVEPOINT s1",
    "SET LOCAL statement_timeout = 5000", "SHOW transaction_read_only",
    "SELECT 'DELETE FROM x' AS label",
    "SELECT $$DROP TABLE t$$",
    "SELECT \"delete\" FROM t",
    "SELECT set_config('x.y', 'v', true)",
    "SELECT 1;",
    "select current_setting('transaction_read_only')",
])
def test_allowed(sql):
    check_sql(sql)


@pytest.mark.parametrize("sql,code", [
    ("/*x*/DELETE FROM t", "verb_not_allowed"),
    ("DELETE FROM t WHERE id = 1", "verb_not_allowed"),
    ("WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d", "forbidden_keyword"),
    ("WITH u AS (UPDATE t SET a = 1 RETURNING id) SELECT id FROM u", "update_without_where"),
    ("SELECT 1; DROP TABLE t", "multi_statement"),
    ("SELECT 1; SELECT 2", "multi_statement"),
    ("UPDATE t SET a=1 -- WHERE x = 1", "update_without_where"),
    ("UPDATE t SET a=1 /* WHERE x = 1 */", "update_without_where"),
    ("UPDATE t SET a='WHERE x'", "update_without_where"),
    ("UPDATE t SET a=1", "update_without_where"),
    ("UPDATE t SET a=1 WHERE true", "tautology"),
    ("UPDATE t SET a=1 WHERE TRUE RETURNING id", "tautology"),
    ("UPDATE t SET a=1 WHERE 1=1", "tautology"),
    ("UPDATE t SET a=1 WHERE (1 = 1)", "tautology"),
    ("UPDATE t SET a=1 WHERE id = id", "tautology"),
    ("UPDATE t SET a=1 WHERE 'a' = 'a'", "tautology"),
    ("UPDATE t SET a=1 WHERE id = 3 OR true", "tautology"),
    ("UPDATE t SET a=1 WHERE id = 3 OR 1=1", "tautology"),
    ("UPDATE t SET a=1 WHERE 2 > 1", "tautology"),
    ("TRUNCATE t", "verb_not_allowed"),
    ("DO $$ BEGIN DELETE FROM t; END $$", "verb_not_allowed"),
    ("COPY t TO STDOUT", "verb_not_allowed"),
    ("GRANT ALL ON t TO x", "verb_not_allowed"),
    ("REVOKE ALL ON t FROM x", "verb_not_allowed"),
    ("CREATE TABLE t (a int)", "verb_not_allowed"),
    ("ALTER TABLE t ADD COLUMN b int", "verb_not_allowed"),
    ("DROP TABLE t", "verb_not_allowed"),
    ("VACUUM", "verb_not_allowed"),
    ("CALL p()", "verb_not_allowed"),
    ("MERGE INTO t USING s ON true WHEN MATCHED THEN DELETE", "verb_not_allowed"),
    ("SET statement_timeout = 0", "session_set"),
    ("SET SESSION AUTHORIZATION x", "session_set"),
    ("RESET ALL", "verb_not_allowed"),
    ("INSERT INTO t SELECT * FROM s; DELETE FROM s", "multi_statement"),
    ("SELECT pg_terminate_backend(1)", "denied_function"),
    ("SELECT dblink_exec('x', 'y')", "denied_function"),
    ("SELECT * FROM dblink('x', 'y') AS t(a int)", "denied_function"),
    ("SELECT lo_unlink(1)", "denied_function"),
    ("SELECT lo_import('/etc/x')", "denied_function"),
    ("SELECT pg_read_file('x')", "denied_function"),
    ("SELECT pg_cancel_backend(1)", "denied_function"),
    ("SELECT set_config('role', 'x', false)", "denied_function"),
    ("SELECT set_config('role', 'x')", "denied_function"),
    ("SELECT 'unterminated", "unterminated"),
    ("SELECT 1 /* open", "unterminated"),
    ("SELECT $a$ open", "unterminated"),
    ("", "empty"),
    ("  ;  ", "empty"),
    ("EXPLAIN ANALYZE DELETE FROM t", "verb_not_allowed"),
    ("SELECT 1 FROM t WHERE x IN (DELETE FROM t RETURNING x)", "forbidden_keyword"),
])
def test_refused(sql, code):
    with pytest.raises(DestructiveRefused) as ei:
        check_sql(sql)
    assert ei.value.code == code
    assert sql.strip() == "" or sql not in str(ei.value) + repr(ei.value)


def test_nested_comment_and_escape_string_stripping():
    check_sql("SELECT /* a /* b */ c */ 1")
    check_sql(r"SELECT E'it\'s'")
    with pytest.raises(DestructiveRefused):
        check_sql(r"SELECT E'x\''; DELETE FROM t")


class _FakeCursor:
    def __init__(self, log):
        self.log = log

    def execute(self, q, v=None):
        self.log.append(("execute", q))

    def executemany(self, q, vs):
        self.log.append(("executemany", q))

    def fetchone(self):
        return (1,)

    def __exit__(self, *e):
        return False


class _FakeRaw:
    def __init__(self):
        self.log = []
        self.autocommit = False

    def cursor(self, *a, **k):
        return _FakeCursor(self.log)


def test_raw_cursor_wrapper_same_rules():
    raw = _FakeRaw()
    conn = sg.wrap_connection(raw, ec.Verdict(ec.PRODUCTION, ("inconclusive",)))
    cur = conn.cursor()
    cur.execute("SELECT 1")
    with pytest.raises(DestructiveRefused):
        cur.execute("/*x*/DELETE FROM t")
    with pytest.raises(DestructiveRefused):
        cur.executemany("UPDATE t SET a=%s", [(1,)])
    with pytest.raises(DestructiveRefused):
        cur.callproc("f")
    with pytest.raises(DestructiveRefused):
        cur.copy_expert("COPY t TO STDOUT", None)
    assert raw.log == [("execute", "SELECT 1")]
    conn.autocommit = True
    assert raw.autocommit is True and cur.fetchone() == (1,)
    assert "inconclusive" not in repr(conn)


class _Dialect:
    def __init__(self, name):
        self.name = name


class _SAConn:
    def __init__(self, name, verdict=None):
        self.dialect = _Dialect(name)
        self.info = {} if verdict is None else {sg.VERDICT_KEY: verdict}


def test_listener_skips_sqlite_and_checks_everything_else():
    sg.before_cursor_execute(_SAConn("sqlite"), None, "DELETE FROM t", None, None, False)
    with pytest.raises(DestructiveRefused):
        sg.before_cursor_execute(_SAConn("postgresql"), None, "DELETE FROM t", None, None, False)
    with pytest.raises(DestructiveRefused):     # untagged = PRODUCTION (C1)
        sg.before_cursor_execute(_SAConn("postgresql", ec.Verdict(ec.PRODUCTION, ())), None, "TRUNCATE t",
                                 None, None, False)
    sg.before_cursor_execute(_SAConn("postgresql"), None, "SELECT 1", None, None, False)


def test_listener_through_sqlalchemy_on_sqlite_is_untouched():
    from sqlalchemy import create_engine, text
    sg.install_listener()
    try:
        eng = create_engine("sqlite://")
        with eng.begin() as c:
            c.execute(text("CREATE TABLE t (a int)"))
            c.execute(text("DELETE FROM t"))
    finally:
        sg.uninstall_listener()
