"""TOPICREG-1028: topic registry migration draft 011 (static text checks, SQL never executed) and the DB-first loader
(offline, fakes only). No DB, no network."""
import copy
import json
import re
from pathlib import Path

import pytest

from vera import acq_config, micro_topic as mt, topic_registry as tr

ROOT = Path(__file__).resolve().parent.parent
MIG = ROOT / "db" / "migrations" / "pending" / "011-DRAFT-topic-registry.sql.draft"
ROLLBACK = ROOT / "db" / "migrations" / "rollback" / "011-DRAFT-topic-registry-rollback.sql.draft"
SEED_JSON = json.loads((ROOT / "config" / "micro_topic_anchors.json").read_text(encoding="utf-8"))["entries"]


# OD-1028(50) (owner, 2026-10-28): the DB objects were renamed to the topreg_ prefix with type suffixes (work/additive-1028/NAMING.md).
# The guard-test literals below were updated EXPLICITLY for that rename (vera_vjay.topic_registry -> vera_vjay.topreg_topics and the
# object names; the regexes now match topreg_ as well as topic_registry); every assertion is kept.
def _sql():
    return MIG.read_text(encoding="utf-8")


def _code(text):
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("--"))


def _seed_rows():
    code = _code(_sql())
    block = code[code.index("FROM (VALUES"):code.index(") AS v(")]
    rows = []
    for ln in block.splitlines()[1:]:
        if not ln.strip():
            continue
        m = re.fullmatch(r"\s*\('((?:[^']|'')*)', (\d+), '((?:[^']|'')*)', '((?:[^']|'')*)', '((?:[^']|'')*)', '((?:[^']|'')*)', "
                         r"'(\w+)', '(\w+)'\),?", ln)
        assert m, ln
        g = [x.replace("''", "'") if isinstance(x, str) else x for x in m.groups()]
        rows.append(g)
    return rows


# --- migration draft (static) ---
def test_drafts_exist_and_match_no_runner_glob():
    assert MIG.is_file() and ROLLBACK.is_file()
    for p in (MIG, ROLLBACK):
        assert "_vera_" not in p.name and not p.name.endswith(".sql")
    assert list((ROOT / "db" / "migrations").glob("011_vera_*.sql")) == []


def test_header_states_owner_gate_batch_and_unconfirmed_schema():
    head = _sql().split("BEGIN;")[0]
    assert "NOT EXECUTED" in head and "OWNER-APPROVED DDL" in head
    assert "BATCHED with 008 and 010" in head and "PG 18" in head and "UNCONFIRMED" in head
    assert "PostgreSQL 16 or newer" in head and "PG 18.6 on 2026-10-21" in head and "UNCONFIRMED since" in head  # D1
    assert "008 = schema_migrations bootstrap" in head and "009 = portal discovery" in head and "010 = dynamic state" in head
    assert "011 = this topic registry" in head and "012 = classifier categories" in head  # D5
    assert "switch" not in head.lower()  # owner decision: no note that the default may be switched to active


def test_seed_equals_json_and_is_active():
    rows = _seed_rows()
    assert len(rows) == len(SEED_JSON) == 10
    for g, e in zip(rows, SEED_JSON):
        key, ver, label, field, anchor, parent, origin, state = g
        assert (key, label, field, anchor, parent) == (e["id"], e["label"], e["field"], e["anchor_text"], e["parent_openalex_topic"])
        assert (ver, origin, state) == ("1", "seed", "active")


def test_seed_is_idempotent_without_on_conflict():
    code = _code(_sql())
    assert "ON CONFLICT" not in code.upper()
    assert "WHERE NOT EXISTS (SELECT 1 FROM vera_vjay.topreg_topics t WHERE t.topic_key = v.topic_key)" in code


def test_version_has_no_default_and_state_default_is_draft():
    code = _code(_sql())
    col = re.search(r"\n\s+version\s+INTEGER[^,]*?CHECK \(version >= 1\),", code, re.S).group(0)
    assert "DEFAULT" not in col
    assert re.search(r"state\s+TEXT NOT NULL DEFAULT 'draft'", code)
    assert "CHECK (state IN ('draft','active','suspended','retired','banned','prohibited','secrets'))" in code
    assert "UNIQUE (topic_key, version)" in code
    assert "WHERE state = 'active'" in code  # one active per key
    assert "created_at" in code and "modified_at" in code


def test_every_function_pins_search_path_and_no_membership_test():
    code = _code(_sql())
    funcs = re.findall(r"CREATE FUNCTION .*?\$f\$;?", code, re.S)
    assert len(funcs) == 5
    for f in funcs:
        assert "SET search_path = pg_catalog, pg_temp" in f, f[:80]
    assert code.count("pg_has_role") == 2 and "pg_has_role" not in code[code.index("FUNCTION vera_vjay.topreg_is_owner_fn"):]
    assert "pg_get_userbyid(c.relowner) = current_user" in code  # default deny: only the table owner is privileged


def test_app_role_insert_is_draft_only_and_new_key_only():
    code = _code(_sql())
    g = code[code.index("FUNCTION vera_vjay.topreg_guard_trgfn"):code.index("FUNCTION vera_vjay.topreg_forbid_truncate_trgfn")]
    assert "NOT is_owner AND TG_OP = 'DELETE'" in g
    assert "nxt <> 1" in g and "brand-new keys" in g
    assert "NEW.state NOT IN ('draft','banned','prohibited','secrets')" in g and "NEW.origin <> 'runtime'" in g and "no OpenAlex parent topic" in g  # an 'active' app insert raises
    assert "version % is wrong" in g and "NEW.version IS NULL" in g


def test_app_role_update_only_runtime_draft_to_active():
    code = _code(_sql())
    g = code[code.index("FUNCTION vera_vjay.topreg_guard_trgfn"):code.index("FUNCTION vera_vjay.topreg_forbid_truncate_trgfn")]
    assert ("NOT is_owner AND TG_OP = 'UPDATE' AND (OLD.origin <> 'runtime' OR OLD.state <> 'draft' "
            "OR NEW.state NOT IN ('active','suspended'))") in g  # suspended->active, seed rows, other transitions: owner only
    assert g.index("only move a runtime draft row to active or suspended") < g.index("content is immutable")  # immutability applies to all
    assert "SECURITY DEFINER" not in code.upper()
    assert "UPDATE (state)" in code and len(re.findall(r"GRANT UPDATE", code)) == 1


def test_activate_trigger_updates_only_if_another_active_row_exists():
    code = _code(_sql())
    a = code[code.index("FUNCTION vera_vjay.topreg_activate_trgfn"):code.index("CREATE TRIGGER topreg_a_guard_trg")]
    assert a.index("IF EXISTS") < a.index("UPDATE vera_vjay.topreg_topics")
    assert "SET state = 'retired'" in a


def test_exact_grants_column_update_state_only_for_rw():
    code = _code(_sql())
    grants = re.findall(r"GRANT .*?;", code)
    assert grants == ["GRANT SELECT ON vera_vjay.topreg_topics TO vera_vjay_ro;",
                      "GRANT SELECT ON vera_vjay.topreg_topics TO vera_vjay_rw;",
                      "GRANT INSERT (topic_key, label, field, anchor_text, parent_topic_key, origin, state) "
                      "ON vera_vjay.topreg_topics TO vera_vjay_rw;",
                      "GRANT UPDATE (state) ON vera_vjay.topreg_topics TO vera_vjay_rw;"]  # D3: no table-wide INSERT
    assert "REVOKE ALL ON vera_vjay.topreg_topics FROM PUBLIC, vera_vjay_wo, vera_vjay_rw, vera_vjay_ro;" in code
    assert code.index("REVOKE ALL ON vera_vjay.topreg_topics") < code.index("GRANT SELECT ON vera_vjay.topreg_topics")
    ins = grants[2]
    for forbidden in ("topic_row_id", "version", "created_at", "modified_at", "parent_openalex_topic"):
        assert forbidden not in ins
    assert "BEFORE TRUNCATE" in code


def test_free_text_is_ascii_printable_and_capped():
    code = _code(_sql())
    for col in ("label", "field", "anchor_text"):
        assert f"{col} ~ '^[ -~]+$'" in code
    assert "length(anchor_text) BETWEEN 20 AND 1200" in code and "length(label) BETWEEN 3 AND 120" in code
    assert "topic_key ~ '^[a-z0-9][a-z0-9-]{1,63}$'" in code


def test_rollback_drops_table_and_functions_only():
    r = _code(ROLLBACK.read_text(encoding="utf-8"))
    drops = re.findall(r"DROP (TABLE|FUNCTION) IF EXISTS (vera_vjay\.\w+)", r)
    assert ("TABLE", "vera_vjay.topreg_topics") in drops and len(drops) == 6
    assert not re.search(r"\b(DELETE|TRUNCATE|ALTER|UPDATE|INSERT)\b", r.upper())
    assert re.search(r"^BEGIN;$", r, re.M) and r.rstrip().endswith("COMMIT;") and "CASCADE" not in r and "DROP SCHEMA" not in r
    assert "DELETES the seed rows" in ROLLBACK.read_text(encoding="utf-8") and "psql -X -v ON_ERROR_STOP=1" in ROLLBACK.read_text(encoding="utf-8")


# --- toggles and app-side SQL guard ---
def test_toggles_default_off_and_proposals_draft():
    s = acq_config.load_config()["micro_topic"]["registry_table"]
    assert s["read_from_db"] is False and s["registration_enabled"] is False and s["proposal_state"] == "draft"
    assert tr.proposal_state(s) == "draft"
    for bad in ("active", "suspended", "", None, 1):
        with pytest.raises(tr.TopicRegistryError):
            tr.proposal_state({"proposal_state": bad})
    assert tr.proposal_state({}) == "draft"


def test_app_code_sql_statements_against_the_registry_are_exactly_the_allowed_set():
    import ast
    allowed = {tr.TOPIC_SQL_PARENT_ACTIVE, tr.TOPIC_SQL_EXISTS, tr.TOPIC_SQL_ROWS, tr.TOPIC_SQL_KEY_EXISTS, tr.TOPIC_SQL_MATCHABLE, tr.TOPIC_SQL_INSERT,
               tr.TOPIC_SQL_ACTIVATE, tr.TOPIC_SQL_SUSPEND, tr.TOPIC_SQL_LOCK, tr.TOPIC_SQL_OWNER, tr.TOPIC_SQL_INSERT_REFUSED}
    verbs = re.compile(r"^\s*(SELECT|INSERT|UPDATE|DELETE|TRUNCATE|ALTER|COPY|DROP|WITH|MERGE)\b", re.I)
    mentioning = []
    for p in (ROOT / "vera").rglob("*.py"):
        src = p.read_text(encoding="utf-8")
        if "topic_registry" in src and p.name not in ("topic_registry.py",):
            mentioning.append(p.name)
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "vera_vjay.topreg_topics" in node.value \
                    and verbs.match(node.value):
                assert p.name == "topic_registry.py" and node.value in allowed, (p.name, node.value)
    assert mentioning == ["micro_topic.py"]  # only the delegating hook; no other module touches the registry
    assert tr.TOPIC_SQL_INSERT.startswith("INSERT") and "'draft'" in tr.TOPIC_SQL_INSERT and "'runtime'" in tr.TOPIC_SQL_INSERT
    assert tr.TOPIC_SQL_ACTIVATE.startswith("UPDATE") and tr.TOPIC_SQL_ACTIVATE.count("SET") == 1
    assert "SET state = 'active'" in tr.TOPIC_SQL_ACTIVATE and "origin = 'runtime'" in tr.TOPIC_SQL_ACTIVATE
    assert "SET state = 'suspended'" in tr.TOPIC_SQL_SUSPEND and tr.TOPIC_SQL_SUSPEND.count("SET") == 1
    assert "state = 'draft'" in tr.TOPIC_SQL_SUSPEND and "origin = 'runtime'" in tr.TOPIC_SQL_SUSPEND
    for s in allowed:
        assert not re.search(r"\b(DELETE|TRUNCATE|ALTER|COPY|DROP)\b|ON\s+CONFLICT", s, re.I), s
    assert "topreg_topics" not in (ROOT / "vera" / "scope_router.py").read_text(encoding="utf-8")


# --- loader (fakes) ---
class _Res:
    def __init__(self, scalar=None, rows=None):
        self._s, self._r = scalar, rows

    def scalar(self):
        return self._s

    def mappings(self):
        return self

    def all(self):
        return self._r

    def first(self):
        return self._r[0] if self._r else None


class _Conn:
    def __init__(self, log, exists, rows, fail, owner):
        self.log, self.exists, self.rows, self.fail, self.owner = log, exists, rows, fail, owner

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, stmt):
        s = str(stmt)
        self.log.append(s)
        if self.fail == "exists" and "to_regclass" in s:
            raise RuntimeError("boom with secret host")
        if "pg_get_userbyid" in s:
            return _Res(rows=[self.owner])
        if "to_regclass" in s:
            return _Res(scalar="vera_vjay.topreg_topics" if self.exists else None)
        if s.startswith("SELECT topic_key"):
            return _Res(rows=self.rows)
        return _Res()


GOOD_OWNER = {"owner": "vera_admin", "app_memberships": 0, "current_reaches_owner": False, "session_reaches_owner": False}


class _Engine:
    def __init__(self, exists=True, rows=(), fail=None, owner=None):
        self.log, self._a = [], (exists, list(rows), fail, owner or dict(GOOD_OWNER))

    def connect(self):
        return _Conn(self.log, *self._a)


def _row(key, ver=1, state="active", origin="seed", label=None, anchor="alpha beta gamma delta epsilon zeta eta theta"):
    return {"topic_key": key, "version": ver, "label": label or f"Label {key}", "field": "f1", "anchor_text": anchor,
            "parent_openalex_topic": "none", "parent_topic_key": None, "origin": origin, "state": state}


@pytest.fixture(autouse=True)
def _clean():
    mt.reset_for_tests()
    yield
    mt.reset_for_tests()


ON = {"registry_table": {"read_from_db": True}}


def test_default_off_reads_json_and_never_touches_db():
    called = []
    reg = mt.load_registry(copy.deepcopy(acq_config.load_config()["micro_topic"]), reader=lambda: called.append(1))
    assert len(reg.entries) == 10 and called == []
    assert mt.load_registry({"registry_table": {"read_from_db": "true"}}, reader=lambda: called.append(1)).entries and called == []


def test_db_first_uses_only_active_rows_latest_version():
    rows = [_row("a-one", 1, "retired"), _row("a-one", 2, "active"), _row("b-two", 1, "draft", "runtime"), _row("c-three", 1, "suspended")]
    reg = mt.load_registry(ON, reader=lambda: (True, rows))
    assert [e["id"] for e in reg.entries] == ["a-one"] and reg.version == "db:v2"


def test_absent_table_falls_back_to_seed():
    reg = mt.load_registry(ON, reader=lambda: (False, []))
    assert len(reg.entries) == 10


def test_other_errors_and_invalid_tables_fail_closed():
    def boom():
        raise tr.TopicRegistryError("x")
    with pytest.raises(tr.TopicRegistryError):
        mt.load_registry(ON, reader=boom)
    bad = [[], [_row("a-one", 1, "draft")], [_row("a-one", 1), _row("a-one", 2)],  # two active for one key
           [_row("A_bad")], [_row("a-one", state="live")], [_row("a-one", label="two\nlines")],
           [_row("a-one", anchor="x" * 1201)], [{**_row("a-one"), "extra": 1}], [_row("a-one"), _row("a-one")]]
    for rows in bad:
        mt.reset_for_tests()
        with pytest.raises(tr.TopicRegistryError):
            mt.load_registry(ON, reader=lambda r=rows: (True, r))


def test_read_topic_rows_is_read_only_txn_and_hides_driver_text():
    eng = _Engine(rows=[_row("a-one")])
    exists, rows = tr.read_topic_rows(eng)
    assert exists and len(rows) == 1
    assert eng.log[0] == "SET TRANSACTION READ ONLY" and all(s.startswith(("SET", "SELECT")) for s in eng.log)
    assert tr.read_topic_rows(_Engine(exists=False)) == (False, [])
    with pytest.raises(tr.TopicRegistryError) as ei:
        tr.read_topic_rows(_Engine(fail="exists"))
    assert "secret" not in str(ei.value) and "RuntimeError" in str(ei.value)
    with pytest.raises(tr.TopicRegistryError):  # no rw URL configured: enabled + unreadable = fail closed, not the seed
        tr.read_topic_rows(env={})


def test_key_for_provenance_does_not_swallow_registry_error():
    def boom():
        raise tr.TopicRegistryError("x")
    cfg = {"registry_table": {"read_from_db": True}, "count_min_score": {"anchor_bm25": {"scholarly": 2.0}}}
    import unittest.mock as um
    with um.patch.object(tr, "read_topic_rows", boom), pytest.raises(tr.TopicRegistryError):
        mt.key_for_provenance({}, "some text", cfg)


def test_keys_normalize():
    assert tr.normalize_topic_key("  Quantum  Biology! ") == "quantum-biology"
    assert tr.normalize_topic_key("Ｑuantum Biology") == "quantum-biology"
    with pytest.raises(ValueError):
        tr.normalize_topic_key("!!")


# --- runtime registration (fake transactional DB; real concurrency and the SQL triggers are NOT exercised) ---
class _RR:
    def __init__(self, rows=None, rowcount=1):
        self._rows, self.rowcount = rows or [], rowcount

    def first(self):
        return self._rows[0] if self._rows else None

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _RDB:
    """In-memory registry with the verbs register_proposed_topic uses; commit on clean exit, rollback on error."""
    def __init__(self, rows=()):
        self.rows, self.log = [dict(r) for r in rows], []

    def begin(self):
        db = self

        class Tx:
            def __enter__(s):
                s.snap = [dict(r) for r in db.rows]
                return s

            def __exit__(s, et, ev, tb):
                if et:
                    db.rows[:] = s.snap
                return False

            def execute(s, stmt, params=None):
                q = str(stmt)
                db.log.append(q)
                if q == tr.TOPIC_SQL_LOCK:
                    return _RR()
                if q == tr.TOPIC_SQL_KEY_EXISTS:
                    return _RR([1] if any(r["topic_key"] == params["k"] for r in db.rows) else [])
                if q == tr.TOPIC_SQL_PARENT_ACTIVE:
                    return _RR([1] if any(r["topic_key"] == params["p"] and r["state"] == "active" for r in db.rows) else [])
                if q == tr.TOPIC_SQL_INSERT:
                    db.rows.append({"topic_key": params["k"], "label": params["label"], "anchor_text": params["anchor"],
                                    "parent_topic_key": params["parent"], "state": "draft", "origin": "runtime", "version": 1})
                    return _RR()
                if q == tr.TOPIC_SQL_MATCHABLE:
                    return _RR([r for r in db.rows if r["state"] in ("active", "retired", "banned", "prohibited", "secrets")])
                if q == tr.TOPIC_SQL_INSERT_REFUSED:
                    db.rows.append({"topic_key": params["k"], "label": params["label"], "field": params["field"], "anchor_text": params["anchor"],
                                    "parent_topic_key": None, "state": params["state"], "origin": "runtime", "version": 1})
                    return _RR()
                if q in (tr.TOPIC_SQL_ACTIVATE, tr.TOPIC_SQL_SUSPEND):
                    new = "active" if q == tr.TOPIC_SQL_ACTIVATE else "suspended"
                    hit = [r for r in db.rows if r["topic_key"] == params["k"] and r["state"] == "draft" and r["origin"] == "runtime"]
                    for r in hit:
                        r["state"] = new
                    return _RR(rowcount=len(hit))
                raise AssertionError(q)
        return Tx()


REG_ON = {"registration_enabled": True, "proposal_state": "draft", "similarity_max": 0.5}
EXISTING = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
OTHER = "quasar pulsar nebula galaxy orbit comet asteroid planet cosmos stellar"


def _db(state="active", anchor=EXISTING):
    return _RDB([{"topic_key": "old-topic", "label": "Old topic", "anchor_text": anchor, "state": state, "origin": "seed", "version": 1}])


def _reg(db, anchor, label="New Topic One", **kw):
    return tr.register_proposed_topic(label, "field one", anchor, settings=REG_ON, engine=db, **kw)


def test_registration_toggle_off_by_default_and_writes_nothing():
    cfg = acq_config.load_config()["micro_topic"]["registry_table"]
    db = _db()
    with pytest.raises(tr.RegistrationRefused):
        tr.register_proposed_topic("New Topic", "f1", OTHER, settings=cfg, engine=db)
    with pytest.raises(tr.RegistrationRefused):
        tr.register_proposed_topic("New Topic", "f1", OTHER, settings={"registration_enabled": "true"}, engine=db)
    assert db.log == []


def test_dissimilar_proposal_is_written_draft_then_activated_under_lock():
    db = _db()
    out = _reg(db, OTHER)
    assert out == {"key": "new-topic-one", "state": "active"}
    assert db.log[0] == tr.TOPIC_SQL_LOCK  # advisory lock before any read or write
    i = db.log.index(tr.TOPIC_SQL_INSERT)
    assert i < db.log.index(tr.TOPIC_SQL_MATCHABLE) < db.log.index(tr.TOPIC_SQL_ACTIVATE)  # draft first, check, then activate
    assert [r["state"] for r in db.rows] == ["active", "active"]


def _states(db):
    return [(r["topic_key"], r["state"]) for r in db.rows]


def test_similar_to_an_active_topic_is_suspended_and_kept(caplog):
    db = _db()
    with caplog.at_level("INFO", logger="vera"):
        out = _reg(db, EXISTING + " mu")
    assert out["state"] == "suspended" and tr.TOPIC_SQL_SUSPEND in db.log and tr.TOPIC_SQL_ACTIVATE not in db.log
    assert _states(db) == [("old-topic", "active"), ("new-topic-one", "suspended")]
    assert "equivalent to a active topic (matched key old-topic)" in caplog.text
    assert "Greek" not in caplog.text and "New Topic" not in caplog.text and "alpha" not in caplog.text


def test_similar_only_to_a_retired_topic_is_suspended(caplog):
    db = _db("retired")
    with caplog.at_level("INFO", logger="vera"):
        out = _reg(db, EXISTING + " mu")
    assert out["state"] == "suspended" and _states(db)[1] == ("new-topic-one", "suspended")
    assert "equivalent to a retired topic (matched key old-topic)" in caplog.text


def test_similar_to_both_active_and_retired_logs_active(caplog):
    db = _RDB([{"topic_key": "ret-topic", "label": "r", "anchor_text": EXISTING, "state": "retired", "origin": "seed", "version": 1},
               {"topic_key": "act-topic", "label": "a", "anchor_text": EXISTING, "state": "active", "origin": "seed", "version": 1}])
    with caplog.at_level("INFO", logger="vera"):
        out = _reg(db, EXISTING + " mu")
    assert out["state"] == "suspended" and "equivalent to a active topic (matched key act-topic)" in caplog.text


@pytest.mark.parametrize("state", ["draft", "suspended"])
def test_similar_only_to_a_draft_or_suspended_topic_activates(state):
    db = _db(state)
    assert _reg(db, EXISTING + " mu")["state"] == "active"


def test_no_match_activates():
    assert _reg(_db(), OTHER)["state"] == "active"


def test_check_that_cannot_run_keeps_the_draft(caplog):
    def broken(a, b):
        raise RuntimeError("metric exploded")
    for metric in (broken, lambda a, b: None, lambda a, b: float("nan"), lambda a, b: True):
        db = _db()
        with caplog.at_level("WARNING", logger="vera"):
            out = _reg(db, OTHER, similarity=metric)
        assert out["state"] == "draft" and _states(db)[1] == ("new-topic-one", "draft")
        assert tr.TOPIC_SQL_ACTIVATE not in db.log and tr.TOPIC_SQL_SUSPEND not in db.log
    assert "could not run" in caplog.text and "exploded" not in caplog.text


def test_concurrent_similar_proposals_serialise_at_most_one_active():
    db = _db(anchor=OTHER)
    first = _reg(db, EXISTING, label="Greek Letters One")
    second = _reg(db, EXISTING + " mu", label="Greek Letters Two")
    assert (first["state"], second["state"]) == ("active", "suspended")
    assert sum(r["state"] == "active" and r["origin"] == "runtime" for r in db.rows) == 1
    assert db.log.count(tr.TOPIC_SQL_LOCK) == 2  # each registration takes the lock first; real concurrency not exercised


def test_pluggable_metric_and_threshold_and_no_user_text_in_log(caplog):
    db = _db()
    with caplog.at_level("INFO", logger="vera"):
        out = _reg(db, OTHER, similarity=lambda a, b: 0.9, label="Secret Words Topic")
    assert out["state"] == "suspended"
    assert "Secret" not in caplog.text and "Words" not in caplog.text and OTHER.split()[0] not in caplog.text
    for bad in (0, 1.5, "x", True):
        with pytest.raises(tr.RegistrationRefused):
            tr.register_proposed_topic("New Topic", "f1", OTHER, settings={**REG_ON, "similarity_max": bad}, engine=_db())


def test_bad_input_and_existing_key_refused_before_any_write():
    for label, field, anchor in (("ab", "f1", OTHER), ("Good label", "f1", "short"), ("Good label", "f1", "two\nlines " * 5),
                                 ("Good label", "", OTHER), ("!!!", "f1", OTHER)):
        db = _db()
        with pytest.raises(tr.RegistrationRefused):
            tr.register_proposed_topic(label, field, anchor, settings=REG_ON, engine=db)
        assert db.log == []
    db = _db()
    with pytest.raises(tr.RegistrationRefused):
        tr.register_proposed_topic("Old topic", "f1", OTHER, settings=REG_ON, engine=db)
    assert len(db.rows) == 1


def test_proposal_state_must_stay_draft():
    with pytest.raises(tr.TopicRegistryError):
        tr.register_proposed_topic("New Topic", "f1", OTHER, settings={**REG_ON, "proposal_state": "active"}, engine=_db())


def test_failed_transition_rolls_the_whole_transaction_back():
    db = _db()
    orig = db.begin

    def begin():
        tx = orig()
        ex = type(tx).execute

        def execute(s, stmt, params=None):
            if str(stmt) == tr.TOPIC_SQL_ACTIVATE:
                raise RuntimeError("driver text with secret")
            return ex(s, stmt, params)
        type(tx).execute = execute
        return tx
    db.begin = begin
    with pytest.raises(tr.TopicRegistryError) as ei:
        _reg(db, OTHER)
    assert "secret" not in str(ei.value) and len(db.rows) == 1


def test_lexical_metric_is_labelled_provisional_and_lexical():
    assert "LEXICAL" in tr.lexical_similarity.__doc__ and "PROVISIONAL" in tr.lexical_similarity.__doc__
    assert tr.lexical_similarity(EXISTING, EXISTING) >= 0.99 and tr.lexical_similarity(EXISTING, OTHER) < 0.5


def test_loader_fails_closed_when_table_owner_is_an_app_role():
    for owner in ({**GOOD_OWNER, "owner": "vera_vjay_rw"}, {**GOOD_OWNER, "owner": "vera_vjay_wo"},
                  {**GOOD_OWNER, "owner": "someone", "app_memberships": 1}):
        eng = _Engine(rows=[_row("a-one")], owner=owner)
        with pytest.raises(tr.TopicRegistryError) as ei:
            tr.read_topic_rows(eng)
        assert "owned by an app role" in str(ei.value)
    eng = _Engine(rows=[_row("a-one")])
    assert tr.read_topic_rows(eng)[0] is True
    assert tr.TOPIC_APP_ROLES == ("vera_vjay_wo", "vera_vjay_rw", "vera_vjay_ro")
    with pytest.raises(tr.TopicRegistryError):  # absent owner row (cannot verify) = fail closed too
        class E2(_Engine):
            pass
        e = E2(rows=[_row("a-one")])
        e._a = (True, [_row("a-one")], None, None)
        orig = _Conn.execute
        try:
            _Conn.execute = lambda s, stmt: _Res(rows=[]) if "pg_get_userbyid" in str(stmt) else orig(s, stmt)
            tr.read_topic_rows(e)
        finally:
            _Conn.execute = orig


def test_loader_fails_closed_when_session_can_reach_the_owner():  # D2
    for k in ("current_reaches_owner", "session_reaches_owner"):
        for bad in (True, None, 1):
            eng = _Engine(rows=[_row("a-one")], owner={**GOOD_OWNER, k: bad})
            with pytest.raises(tr.TopicRegistryError) as ei:
                tr.read_topic_rows(eng)
            assert "can reach the topic registry table owner" in str(ei.value) and "vera_admin" not in str(ei.value)
    missing = {k: v for k, v in GOOD_OWNER.items() if k != "session_reaches_owner"}
    with pytest.raises(tr.TopicRegistryError):
        tr.read_topic_rows(_Engine(rows=[_row("a-one")], owner=missing))
    q = tr.TOPIC_SQL_OWNER
    assert "pg_has_role(current_user, c.relowner, 'MEMBER')" in q and "pg_has_role(session_user, c.relowner, 'MEMBER')" in q


def test_membership_counts_only_usable_rows_pg16():  # D1
    assert "(m.inherit_option OR m.set_option)" in tr.TOPIC_SQL_OWNER
    pre = _code(_sql())
    pre = pre[pre.index("DO $$ BEGIN", pre.index("requires roles")) + 1:pre.index("CREATE SCHEMA")]
    assert "AND (m.inherit_option OR m.set_option)" in pre


def test_migration_preflight_refuses_app_role_runner():
    code = _code(_sql())
    pre = code[code.index("DO $$ BEGIN", code.index("requires roles")) + 1:code.index("CREATE SCHEMA")]
    assert "current_user IN ('vera_vjay_wo', 'vera_vjay_rw', 'vera_vjay_ro')" in pre
    assert "pg_catalog.pg_auth_members" in pre and "RAISE EXCEPTION" in pre
    assert "pg_has_role(a.rolname, current_user, 'MEMBER')" in pre and "pg_has_role(a.rolname, session_user, 'MEMBER')" in pre  # D2
    assert code.index("must not run as") < code.index("CREATE SCHEMA")
    assert "vera_vjay_wo" in (ROOT / "db" / "accounts" / "create_vera_accounts.sql").read_text(encoding="utf-8")  # names exist


def test_no_dynamic_or_orm_construction_of_the_registry_table_name_in_vera_or_scripts():
    import ast
    own = {"topic_registry.py"}
    hits = []
    for base in ("vera", "scripts"):
        for p in (ROOT / base).rglob("*.py"):
            src = p.read_text(encoding="utf-8")
            tree = ast.parse(src)
            for node in ast.walk(tree):
                def consts(n):
                    return [c.value for c in ast.walk(n) if isinstance(c, ast.Constant) and isinstance(c.value, str)]
                joined = "".join(consts(node)) if isinstance(node, (ast.BinOp, ast.JoinedStr)) else ""
                if isinstance(node, (ast.BinOp, ast.JoinedStr)) and re.search(r"top(?:ic_)?reg|vera_vjay\.top", joined):
                    hits.append((p.name, "built string", joined[:60]))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("format", "format_map", "join") \
                        and re.search(r"top(?:ic_)?reg|vera_vjay\.top", "".join(consts(node))):
                    hits.append((p.name, "format/join", ""))
                fn = node.func.id if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) else \
                    node.func.attr if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) else ""
                if fn in ("Table", "Column", "automap_base", "declarative_base", "reflect") and \
                        re.search(r"top(?:ic_)?reg", "".join(consts(node))):
                    hits.append((p.name, "orm", fn))
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and "vera_vjay.topreg_topics" in node.value \
                        and p.name not in own:
                    hits.append((p.name, "sql string outside topic_registry.py", node.value[:60]))
    assert hits == [], hits
    assert "sqlalchemy" not in "".join(tr.__doc__ or "")  # sanity: the module is the only SQL home


# --- OD-1028(49): parent_topic_key (static SQL text; the triggers are never executed) and caller-supplied parent_key ---
def test_parent_topic_key_column_check_trigger_and_immutability():
    code = _code(_sql())
    assert "parent_topic_key TEXT CONSTRAINT topreg_parent_key_chk CHECK (parent_topic_key IS NULL OR parent_topic_key ~ '^[a-z0-9][a-z0-9-]{1,63}$')" in code
    assert "parent_topic_key TEXT NOT NULL" not in code  # null is allowed
    g = code[code.index("FUNCTION vera_vjay.topreg_guard_trgfn"):code.index("FUNCTION vera_vjay.topreg_forbid_truncate_trgfn")]
    assert "IF NEW.parent_topic_key IS NOT NULL THEN" in g
    assert "p.topic_key = NEW.parent_topic_key AND p.state = 'active'" in g and "must reference an existing active topic key" in g
    assert "cannot be its own parent" in g
    assert g.index("IF NEW.parent_topic_key IS NOT NULL") < g.index("IF NEW.version IS NULL")  # insert branch
    assert "NEW.parent_topic_key IS DISTINCT FROM OLD.parent_topic_key" in g.split("RETURN NEW;\n  END IF;")[1]  # immutable (UPDATE branch)
    assert "parent_topic_key" in tr.TOPIC_SQL_ROWS and "parent_topic_key" in tr.TOPIC_SQL_INSERT


def test_no_similarity_inference_of_parent_and_match_means_equivalence():
    import inspect
    src = inspect.getsource(tr.register_proposed_topic)
    assert "parent_key" in src and "semantic equivalence" in tr.register_proposed_topic.__doc__
    assert "NEVER inferred" in tr.register_proposed_topic.__doc__
    assert "'DISTINCT" in _sql() or "DISTINCT micro-topic" in _sql()


def test_register_with_valid_parent_stores_it_and_null_is_allowed():
    db = _db()
    out = _reg(db, OTHER, parent_key="old-topic")
    assert out["state"] == "active" and db.rows[-1]["parent_topic_key"] == "old-topic"
    assert tr.TOPIC_SQL_PARENT_ACTIVE in db.log and db.log.index(tr.TOPIC_SQL_PARENT_ACTIVE) < db.log.index(tr.TOPIC_SQL_INSERT)
    db = _db()
    _reg(db, OTHER)
    assert db.rows[-1]["parent_topic_key"] is None and tr.TOPIC_SQL_PARENT_ACTIVE not in db.log


def test_register_with_invalid_or_nonactive_parent_is_refused_and_rolled_back():
    for state in ("draft", "suspended", "retired"):
        db = _db(state)
        with pytest.raises(tr.RegistrationRefused):
            _reg(db, OTHER, parent_key="old-topic")
        assert len(db.rows) == 1 and tr.TOPIC_SQL_INSERT not in db.log
    for bad in ("missing-topic", "Bad_Key", "x", "new-topic-one", 5):  # unknown, malformed, self, non-string
        db = _db()
        with pytest.raises(tr.RegistrationRefused):
            _reg(db, OTHER, parent_key=bad)
        assert len(db.rows) == 1


def test_narrower_or_broader_nonequivalent_topic_is_distinct_and_activated():
    db = _db()
    out = _reg(db, OTHER, parent_key="old-topic", similarity=lambda a, b: 0.3)  # related but not equivalent
    assert out["state"] == "active" and db.rows[-1]["parent_topic_key"] == "old-topic"


def test_loader_rows_validate_parent_topic_key():
    ok = {**_row("a-one"), "parent_topic_key": None}
    child = {**_row("b-two", label="Label b-two"), "parent_topic_key": "a-one"}
    reg = mt.load_registry(ON, reader=lambda: (True, [ok, child]))
    assert {e["id"]: e["parent_topic_key"] for e in reg.entries} == {"a-one": None, "b-two": "a-one"}
    for bad in ("Bad_Key", 3, "b-two"):
        mt.reset_for_tests()
        with pytest.raises(tr.TopicRegistryError):
            mt.load_registry(ON, reader=lambda b=bad: (True, [{**child, "parent_topic_key": b}]))


# --- D4: cache ---
def test_db_registry_and_absent_fallback_are_cached_with_ttl():
    t = [0.0]
    calls = []
    mt._now = lambda: t[0]
    try:
        cfg = {"registry_table": {"read_from_db": True, "cache_ttl_seconds": 10}}
        mt.load_registry(cfg, reader=lambda: calls.append(1) or (False, []))
        mt.load_registry(cfg, reader=lambda: calls.append(1) or (False, []))
        assert len(calls) == 1  # absent fallback cached: not re-queried
        t[0] = 11.0
        reg = mt.load_registry(cfg, reader=lambda: calls.append(1) or (True, [_row("z-new")]))
        assert len(calls) == 2 and [e["id"] for e in reg.entries] == ["z-new"]  # runtime activation visible after TTL
        t[0] = 12.0
        assert mt.load_registry(cfg, reader=lambda: calls.append(1) or (True, [])) is reg and len(calls) == 2
        mt.reset_for_tests()
        cfg0 = {"registry_table": {"read_from_db": True, "cache_ttl_seconds": 0}}
        mt.load_registry(cfg0, reader=lambda: calls.append(1) or (False, []))
        mt.load_registry(cfg0, reader=lambda: calls.append(1) or (False, []))
        assert len(calls) == 4
    finally:
        mt._now = __import__("time").monotonic
    assert mt._db_ttl({}) == mt.DEFAULT_DB_TTL and mt._db_ttl({"registry_table": {"cache_ttl_seconds": -1}}) == mt.DEFAULT_DB_TTL


# --- OD-1028(50): strictly additive, transactional, collision preflight, names derived from the file (mirrors 012) ---
def test_forward_is_strictly_additive_transactional_with_full_preflight():
    sql = _sql()
    code = _code(sql)
    assert re.search(r"^BEGIN;$", sql, re.M) and sql.rstrip().endswith("COMMIT;") and [l for l in sql.splitlines() if l == "COMMIT;"] == ["COMMIT;"]
    assert not re.search(r"CREATE OR REPLACE", code) and not re.search(r"CREATE (TABLE|UNIQUE INDEX|INDEX|FUNCTION|TRIGGER) IF NOT EXISTS", code)
    assert "CREATE SCHEMA IF NOT EXISTS vera_vjay;" in code
    assert "DROP " not in code and "\\set" not in code and "psql -X -v ON_ERROR_STOP=1" in sql
    body = code.split("BEGIN;", 1)[1]
    assert body.index("already applied or name collision") < body.index("CREATE SCHEMA") < body.index("CREATE TABLE")
    pre = body[:body.index("CREATE SCHEMA")]
    tables = re.findall(r"CREATE TABLE (vera_vjay\.\w+)", code)
    funcs = re.findall(r"CREATE FUNCTION vera_vjay\.(\w+)", code)
    idx = re.findall(r"CREATE UNIQUE INDEX (\w+)", code) + re.findall(r"CONSTRAINT (\w+) (?:PRIMARY KEY|UNIQUE)", code)
    trg = re.findall(r"CREATE TRIGGER (\w+)", code)
    chk = re.findall(r"CONSTRAINT (\w+)\s+CHECK", code)
    assert tables == ["vera_vjay.topreg_topics"] and len(funcs) == 5 and len(idx) == 4 and len(trg) == 4 and len(chk) == 9
    assert "to_regclass('%s')" % tables[0] in pre
    for n in funcs + idx + trg:  # derived from the file, so the preflight list cannot drift
        assert "'%s'" % n in pre, n
    assert all(re.fullmatch(r"topreg_\w+_(fn|trgfn)", f) for f in funcs)
    assert all(re.fullmatch(r"topreg_\w+_trg", t) for t in trg)
    assert all(re.fullmatch(r"topreg_\w+_(uq|pk)", i) for i in idx)
    assert all(re.fullmatch(r"topreg_\w+_chk", c) for c in chk)
    assert "pg_proc" in pre and "pg_namespace" in pre and "pg_trigger" in pre and "postgres://" not in pre
    assert "topic_registry" not in code


def test_rollback_drops_every_created_object_by_name():
    code = _code(_sql())
    r = _code(ROLLBACK.read_text(encoding="utf-8"))
    for n in re.findall(r"CREATE FUNCTION vera_vjay\.(\w+)", code):
        assert "DROP FUNCTION IF EXISTS vera_vjay.%s()" % n in r, n
    for n in re.findall(r"CREATE TRIGGER (\w+)", code):
        assert "DROP TRIGGER IF EXISTS %s ON vera_vjay.topreg_topics" % n in r, n
    for n in re.findall(r"CREATE UNIQUE INDEX (\w+)", code):
        assert "DROP INDEX IF EXISTS vera_vjay.%s;" % n in r, n


# --- OD-1028(51)/(52): topical states ---
def test_refused_states_db_rules_in_the_migration_text():
    code = _code(_sql())
    g = code[code.index("FUNCTION vera_vjay.topreg_guard_trgfn"):code.index("FUNCTION vera_vjay.topreg_forbid_truncate_trgfn")]
    assert "'banned','prohibited','secrets'" in g  # app INSERT allow-list
    assert "OLD.state <> 'draft' OR NEW.state NOT IN ('active','suspended')" in g  # leaving banned/prohibited/secrets is owner-only
    assert "leaving banned/prohibited/secrets is owner-only" in g
    assert "refused-topic row (banned/prohibited/secrets) has no parent topic key" in g
    assert re.findall(r"GRANT UPDATE", code) == ["GRANT UPDATE"]  # the app's only UPDATE stays column state, draft -> active/suspended


ROWS_CASE = [_row("act-one"), _row("ban-one", state="banned", origin="runtime", label="Neutral banned label")]


@pytest.mark.parametrize("state", ["banned", "prohibited", "secrets", "suspended", "retired", "draft"])
def test_non_active_rows_are_never_loaded_as_active(state):
    data = tr.registry_data_from_rows([_row("act-one"), _row("other-one", state=state, origin="runtime", label="Neutral other label")])
    assert [e["id"] for e in data["entries"]] == ["act-one"]


def test_refused_rows_never_reach_the_registry_or_acquisition_users():
    reg = mt.load_registry(ON, reader=lambda: (True, ROWS_CASE))
    assert [e["id"] for e in reg.entries] == ["act-one"]
    import ast
    for name in ("acq_rounds.py", "catstep.py"):  # acquisition callers only see Registry entries, never refused rows
        src = (ROOT / "vera" / name).read_text(encoding="utf-8")
        assert "REFUSED_STATES" not in src and "refused_entries_from_rows" not in src and "register_refused_topic" not in src
        ast.parse(src)
    only_refused = [_row("ban-one", state="banned", origin="runtime", label="Neutral banned label")]
    with pytest.raises(tr.TopicRegistryError):  # a table with only refused rows has no active row: fail closed
        tr.registry_data_from_rows(only_refused)


MAP = {"harmful": "banned", "advice": "prohibited", "rule_change": "prohibited", "secret_leak": "secrets"}
MAP_LIST = [{"category_code": c, "state": st} for c, st in MAP.items()]
REF_ON = {**REG_ON, "refused_state_map": MAP_LIST}


def test_config_mapping_is_provisional_and_valid():
    s = acq_config.load_config()["micro_topic"]["registry_table"]
    assert s["refused_state_map"] == [{"category_code": c, "state": st} for c, st in MAP.items()] and s["refused_state_mapping_status"] == "PROVISIONAL"
    assert tr.refused_state_mapping(s) == MAP
    tr.validate_registry_settings(s)
    dup = [{"category_code": "a_b", "state": "banned"}] * 2
    for bad in ({}, {"refused_state_map": []}, {"refused_state_map": [{"category_code": "harmful", "state": "active"}]},
                {"refused_state_map": [{"category_code": "Bad Code", "state": "banned"}]}, {"refused_state_map": "x"},
                {"refused_state_map": dup}, {"refused_state_map": [{"category_code": "harmful"}]}):
        with pytest.raises(tr.TopicRegistryError):
            tr.refused_state_mapping(bad)
        if "refused_state_map" in bad:
            with pytest.raises(tr.TopicRegistryError):
                tr.validate_registry_settings(bad)  # validated at load


@pytest.mark.parametrize("code,state", sorted(MAP.items()))
def test_register_refused_topic_stores_category_and_neutral_label_only(code, state, caplog):
    db = _db()
    with caplog.at_level("INFO", logger="vera"):
        out = tr.register_refused_topic(None, code, "Neutral placeholder label", settings=REF_ON, engine=db)
        out2 = tr.register_refused_topic(state, code, "Another neutral label", settings=REF_ON, engine=db)
    assert out == {"key": "neutral-placeholder-label", "state": state} and out2["state"] == state
    row = db.rows[1]
    assert (row["state"], row["field"], row["label"], row["parent_topic_key"]) == (state, code, "Neutral placeholder label", None)
    assert row["anchor_text"] == f"refused topic category {code}: Neutral placeholder label"  # only code + label: no question text
    assert "placeholder" not in caplog.text and tr.TOPIC_SQL_INSERT not in db.log


def test_register_refused_topic_refusals_write_nothing():
    for st, code, label, cfg in ((None, "unknown_cat", "Neutral label", REF_ON), ("banned", "advice", "Neutral label", REF_ON),
                                 (None, "harmful", "ab", REF_ON), (None, "harmful", "Is this a question?", REF_ON),
                                 (None, "harmful", "x" * 61, REF_ON), (None, "harmful", "Line\nbreak label", REF_ON),
                                 (None, "harmful", " leading space", REF_ON), (None, "harmful", "Neutral  label", REF_ON),
                                 (None, "harmful", "Old topic", REF_ON), (None, "harmful", "Neutral label", {**REF_ON, "registration_enabled": False}),
                                 (None, "harmful", "Neutral label", {**REG_ON}), (None, 5, "Neutral label", REF_ON)):
        db = _db()
        with pytest.raises((tr.RegistrationRefused, tr.TopicRegistryError)):
            tr.register_refused_topic(st, code, label, settings=cfg, engine=db)
        assert not any(q in (tr.TOPIC_SQL_INSERT_REFUSED, tr.TOPIC_SQL_INSERT) for q in db.log), (code, label)
        assert len(db.rows) == 1


def test_register_refused_topic_signature_takes_no_question_text():
    import inspect
    p = inspect.signature(tr.register_refused_topic).parameters
    assert list(p)[:3] == ["state", "category_code", "label"] and not {"text", "question", "query", "content"} & set(p)


@pytest.mark.parametrize("kind", ["banned", "prohibited", "secrets", "active", "retired"])
def test_draft_similar_to_any_of_these_goes_to_suspended(kind, caplog):
    db = _db(kind)
    with caplog.at_level("INFO", logger="vera"):
        out = _reg(db, EXISTING + " mu")
    assert out["state"] == "suspended" and tr.TOPIC_SQL_SUSPEND in db.log and tr.TOPIC_SQL_ACTIVATE not in db.log
    if kind in tr.REFUSED_STATES:
        assert f"equivalent to a {kind} topic" in caplog.text and "old-topic" not in caplog.text


def test_matching_a_refused_topic_is_a_suspension_not_a_refusal():
    db = _db("banned")
    out = _reg(db, EXISTING + " mu")  # returns normally: no RegistrationRefused is raised for a refused-topic match
    assert out["state"] == "suspended"


# scrutiny_flag: pure; extra scrutiny, never an automatic refusal
ENTRIES = (("harmful", EXISTING), ("secret_leak", OTHER))


def test_scrutiny_flag_pure_and_returns_category_or_none():
    assert tr.scrutiny_flag(EXISTING + " mu", ENTRIES) == {"category_code": "harmful"}
    assert tr.scrutiny_flag(OTHER + " mu", ENTRIES) == {"category_code": "secret_leak"}
    assert tr.scrutiny_flag("completely unrelated cooking recipe words here", ENTRIES) == {"category_code": None}
    assert tr.scrutiny_flag("anything", ()) == {"category_code": None} and tr.scrutiny_flag(None, ENTRIES) == {"category_code": None}
    assert tr.scrutiny_flag("x", ENTRIES, similarity=lambda a, b: 1 / 0) == {"category_code": None}
    rows = [_row("a-one"), {**_row("b-two", state="banned", origin="runtime"), "field": "harmful"}]
    assert tr.refused_entries_from_rows(rows) == (("harmful", rows[1]["anchor_text"]),)


def test_scrutiny_flag_is_pure_no_io_and_used_by_no_refusal_constructor():
    import ast
    tree = ast.parse((ROOT / "vera" / "topic_registry.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "scrutiny_flag")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert not names & {"engine", "conn", "execute", "connect", "open", "db", "get_role_engine", "begin", "logging", "sqlalchemy"}
    users = []
    for p in list((ROOT / "vera").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
        src = p.read_text(encoding="utf-8")
        if "scrutiny_flag" in src and p.name != "topic_registry.py":
            users.append(p.name)
    assert users == []  # not wired into scope_router.refusal_token, catstep, the refusal path or anything else in this task
    own = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
           if any(isinstance(c, ast.Name) and c.id == "scrutiny_flag" for c in ast.walk(n)) and n.name != "scrutiny_flag"]
    assert own == []
    assert "refus" not in "".join(n.name for n in ast.walk(ast.parse((ROOT / "vera" / "scope_router.py").read_text(encoding="utf-8")))
                                  if isinstance(n, ast.FunctionDef) and "scrutiny" in n.name)


# --- reviewer LOW: cache_ttl_seconds capped 0..3600, rejected at load ---
@pytest.mark.parametrize("ttl", [0, 1, 300, 3600, 3600.0])
def test_cache_ttl_in_range_loads(ttl):
    mt.reset_for_tests()
    cfg = {"registry_table": {"read_from_db": True, "cache_ttl_seconds": ttl}}
    assert [e["id"] for e in mt.load_registry(cfg, reader=lambda: (True, [_row("z-new")])).entries] == ["z-new"]


@pytest.mark.parametrize("ttl", [-1, 3601, 10 ** 9, "300", None, True, float("nan")])
def test_cache_ttl_out_of_range_is_rejected_at_load(ttl):
    mt.reset_for_tests()
    cfg = {"registry_table": {"read_from_db": True, "cache_ttl_seconds": ttl}}
    with pytest.raises(tr.TopicRegistryError):
        mt.load_registry(cfg, reader=lambda: (True, [_row("z-new")]))
    assert acq_config.load_config()["micro_topic"]["registry_table"]["cache_ttl_seconds"] == 300
