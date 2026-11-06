"""CONTRACT-FIX-1105: SQL contract collector fixes and the deny-by-default pending-migration exemption.

No DB. Synthetic trees live under tmp_path (the project runner's private tmpfs). See tests/sql_contract_support.py.
"""
import hashlib
import warnings
from pathlib import Path

from tests import sql_contract_support as sc

REPO = Path(__file__).resolve().parent.parent
# The 158 (table, column) pairs extracted at 2c0a323, which passed the live catalog check on 2026-10-31 (sha256 of the
# sorted "table.column\n" lines). The required contract at the candidate must be exactly this set.
CONTRACT_2C0A323_COUNT = 158
CONTRACT_2C0A323_FROZEN_SHA256 = "84d028093cc8e770d732a2c7ad3095313e50a6295582709b2555a28fad511e73"  # old collector at 2c0a323


def _sha(pairs) -> str:
    return hashlib.sha256("".join(f"{t}.{c}\n" for t, c in sorted(pairs)).encode()).hexdigest()


def _tree(tmp_path, drafts: dict, allow=None, applied: dict | None = None):
    pend = tmp_path / "db" / "migrations" / "pending"
    pend.mkdir(parents=True)
    for name, text in drafts.items():
        (pend / name).write_text(text)
    for name, text in (applied or {}).items():
        (tmp_path / "db" / "migrations" / name).write_text(text)
    return tmp_path


DRAFT_T = ("-- CREATE TABLE vera_vjay.ghost (nope int);  (comment: never parsed)\n"
           "DO $$ BEGIN RAISE NOTICE 'x'; END $$;\n"
           "CREATE TABLE vera_vjay.newtab (\n  id BIGINT PRIMARY KEY,\n  label TEXT CHECK (label ~ '^[a-z](x)$'),\n"
           "  CONSTRAINT newtab_uq UNIQUE (label)\n);\n"
           "CREATE VIEW vera_vjay.newtab_v WITH (security_barrier = true) AS\n"
           "  SELECT t.id, t.label ->> 'k' AS shape FROM vera_vjay.newtab t;\n"
           "CREATE FUNCTION vera_vjay.newfn() RETURNS TABLE (n bigint) LANGUAGE sql AS $f$ SELECT 1 $f$;\n")
FOUND = {("newtab", "id"): ["x"], ("newtab", "label"): ["x"], ("newtab", "bogus"): ["x"],
         ("newtab_v", "shape"): ["x"], ("ghost", "nope"): ["x"], ("appraisals", "source_id"): ["x"]}


# ------------------------------------------------------------------ (1) function calls are not tables
def test_function_call_is_not_a_table():
    assert sc.extract_columns("SELECT events_deleted, requests_deleted FROM {S}.purge_expired_trace()") == set()
    assert sc.extract_columns("SELECT x FROM vera_vjay.some_fn (:a) AS f") == set()
    cols = sc.extract_columns("SELECT a.x FROM {S}.appraisals a JOIN {S}.fn_t(:p) f ON f.k = a.source_id")
    assert cols == {("appraisals", "x"), ("appraisals", "source_id")}
    # INSERT INTO t (cols) keeps its column list (the '(' after INTO is a column list, not a call)
    assert {("answers", "run_id"), ("answers", "stage")} <= sc.extract_columns(
        "INSERT INTO {S}.answers (run_id, stage) VALUES (:r, :s)")


def test_purge_function_absent_from_contract():
    assert not any(t == "purge_expired_trace" for t, _ in sc.collect_all())


# ------------------------------------------------------------------ (2) docstrings are not parsed
class _Src:
    def __init__(self, text):
        self.text = text

    def read_text(self):
        return self.text


def test_docstrings_not_parsed():
    src = ('"""Module: SELECT owner, url FROM vera_vjay.doc_tab WHERE a = 1."""\n'
           'def f():\n    """SELECT y FROM vera_vjay.doc_tab2"""\n    return 1\n'
           'class C:\n    """UPDATE vera_vjay.doc_tab3 SET z = 1"""\n'
           'Q = "SELECT real_col FROM vera_vjay.code_tab"\n')
    strings = [s for s, _ in sc._py_strings(_Src(src))]
    assert strings == ["SELECT real_col FROM vera_vjay.code_tab"]


def test_retention_purge_docstring_not_parsed():
    strings = [s for s, _ in sc._py_strings(REPO / "vera" / "retention_purge.py")]
    assert not any("the ONLY SQL is" in s for s in strings)


# ------------------------------------------------------------------ (3) deny-by-default exemption
def test_exempt_pair_requires_listed_draft_that_creates_it(tmp_path, monkeypatch):
    root = _tree(tmp_path, {"099-DRAFT-new.sql.draft": DRAFT_T})
    monkeypatch.setattr(sc, "PENDING_DRAFT_ALLOWLIST", ("099-DRAFT-new.sql.draft",))
    objs = sc.pending_objects(root)
    assert set(objs) == {"newtab", "newtab_v"}                       # function and commented-out table excluded
    assert objs["newtab"][1] == {"id", "label"} and objs["newtab_v"][1] == {"id", "shape"}
    ex = sc.pending_exemptions(FOUND, root)
    assert ex == {("newtab", "id"): "099-DRAFT-new.sql.draft", ("newtab", "label"): "099-DRAFT-new.sql.draft",
                  ("newtab_v", "shape"): "099-DRAFT-new.sql.draft"}
    # a column the draft does not create, an object no draft creates, and an existing table all stay required
    assert ("newtab", "bogus") not in ex and ("ghost", "nope") not in ex and ("appraisals", "source_id") not in ex
    assert sc.pending_exempt("newtab", root) and sc.pending_exempt("NEWTAB_V", root)
    assert not sc.pending_exempt("newfn", root) and not sc.pending_exempt("ghost", root)


def test_pair_with_no_draft_fails(tmp_path, monkeypatch):
    root = _tree(tmp_path, {})
    monkeypatch.setattr(sc, "PENDING_DRAFT_ALLOWLIST", ("099-DRAFT-new.sql.draft",))  # listed but file absent
    assert sc.pending_objects(root) == {} and sc.pending_exemptions(FOUND, root) == {}


def test_removing_allowlist_entry_makes_it_fail(tmp_path, monkeypatch):
    root = _tree(tmp_path, {"099-DRAFT-new.sql.draft": DRAFT_T})
    monkeypatch.setattr(sc, "PENDING_DRAFT_ALLOWLIST", ())
    assert sc.pending_exemptions(FOUND, root) == {} and not sc.pending_exempt("newtab", root)


def test_unlisted_or_badly_named_or_applied_drafts_prove_nothing(tmp_path, monkeypatch):
    root = _tree(tmp_path, {"099-DRAFT-new.sql.draft": DRAFT_T, "newtab.sql": DRAFT_T},
                 applied={"099_applied.sql": "CREATE TABLE vera_vjay.newtab_v (shape text);"})
    monkeypatch.setattr(sc, "PENDING_DRAFT_ALLOWLIST", ("newtab.sql",))           # wrong name shape: ignored
    assert sc.pending_objects(root) == {}
    monkeypatch.setattr(sc, "PENDING_DRAFT_ALLOWLIST", ("099-DRAFT-new.sql.draft",))
    assert set(sc.pending_objects(root)) == {"newtab"}                            # applied migration creates newtab_v


def test_block_commented_create_does_not_exempt(tmp_path, monkeypatch):
    draft = ("/* outer /* nested */ CREATE TABLE vera_vjay.newtab (id int, label text); still comment */\n"
             "/* unterminated: CREATE VIEW vera_vjay.newtab_v AS SELECT 1 AS shape")
    root = _tree(tmp_path, {"099-DRAFT-new.sql.draft": draft})
    monkeypatch.setattr(sc, "PENDING_DRAFT_ALLOWLIST", ("099-DRAFT-new.sql.draft",))
    assert sc.pending_objects(root) == {} and sc.pending_exemptions(FOUND, root) == {}
    assert sc.draft_objects("/* x */ CREATE TABLE vera_vjay.real_t (a int);") == {"real_t": ("table", frozenset({"a"}))}


def test_symlinked_parent_directory_proves_nothing(tmp_path, monkeypatch):
    real = _tree(tmp_path / "real", {"099-DRAFT-new.sql.draft": DRAFT_T})
    root = tmp_path / "linked"
    (root / "db").mkdir(parents=True)
    (root / "db" / "migrations").symlink_to(real / "db" / "migrations", target_is_directory=True)
    monkeypatch.setattr(sc, "PENDING_DRAFT_ALLOWLIST", ("099-DRAFT-new.sql.draft",))
    assert set(sc.pending_objects(real)) == {"newtab", "newtab_v"}
    assert sc.pending_objects(root) == {}


def _fn_tree(tmp_path, code: str, applied: str = "", draft: str = ""):
    (tmp_path / "vera").mkdir()
    (tmp_path / "vera" / "m.py").write_text(code)
    _tree(tmp_path, {"099-DRAFT-fn.sql.draft": draft} if draft else {}, applied={"001_x.sql": applied})
    return tmp_path


def test_every_called_function_is_created_by_applied_migration_or_listed_draft():
    assert sc.unknown_function_calls() == {}
    assert set(sc.PENDING_FUNCTION_DRAFT_ALLOWLIST) == {"018-DRAFT-trace-purge-erasure.sql.draft"}
    assert "purge_expired_trace" in sc.collect_function_calls()


def test_unknown_function_call_fails(tmp_path, monkeypatch):
    code = ('Q = "SELECT a FROM {S}.ghost_fn(:x)"\nI = "INSERT INTO vera_vjay.answers (run_id) VALUES (1)"\n'
            'D = "CALL vera_vjay.pending_fn()"\n')
    fn_draft = "CREATE FUNCTION vera_vjay.pending_fn() RETURNS int LANGUAGE sql AS $f$ SELECT 1 $f$;"
    root = _fn_tree(tmp_path, code, applied="CREATE OR REPLACE FUNCTION vera_vjay.other_fn() RETURNS int AS $$ $$;",
                    draft=fn_draft)
    monkeypatch.setattr(sc, "PENDING_FUNCTION_DRAFT_ALLOWLIST", ("099-DRAFT-fn.sql.draft",))
    assert set(sc.collect_function_calls(root)) == {"ghost_fn", "pending_fn"}       # INSERT INTO t (..) is not a call
    assert set(sc.unknown_function_calls(root)) == {"ghost_fn"}
    monkeypatch.setattr(sc, "PENDING_FUNCTION_DRAFT_ALLOWLIST", ())                  # unlisted draft proves nothing
    assert set(sc.unknown_function_calls(root)) == {"ghost_fn", "pending_fn"}
    monkeypatch.setattr(sc, "PENDING_DRAFT_ALLOWLIST", ("099-DRAFT-fn.sql.draft",))  # table list does not cover functions
    assert set(sc.unknown_function_calls(root)) == {"ghost_fn", "pending_fn"}


def test_real_tree_exemptions_are_exactly_016_and_017_and_reported():
    ex = sc.pending_exemptions()
    assert set(sc.PENDING_DRAFT_ALLOWLIST) == {"016-DRAFT-lists-to-db.sql.draft", "017-DRAFT-topic-tags.sql.draft"}
    assert {t for t, _ in ex} == {"lstreg_active_lists_v", "lstreg_active_items_v", "toptag_document_topics"}
    assert len(ex) == 15
    report = sc.exemption_report()
    assert report.startswith("SQL contract pending-migration EXEMPTION: 15 (table, column) pairs exempt")
    assert all(f"exempt: {t}.{c} ({d})" in report for (t, c), d in ex.items())
    warnings.warn(report, UserWarning)  # loud on every suite run (shown in the pytest warnings summary)


def test_016_drafts_byte_identical_to_reviewed_staged_copies():
    want = {"pending/016-DRAFT-lists-to-db.sql.draft":
            "1734f6518f64c8cd2caec2bcf56fcf7073a09b47019b34f5022f70c27ccc4340",
            "rollback/016-DRAFT-lists-to-db-rollback.sql.draft":
            "d5905b9a992227c8de3a69d108594b2999be176bc98cb9c6edc20bdd2881e6f4"}
    for rel, sha in want.items():
        assert hashlib.sha256((REPO / "db" / "migrations" / rel).read_bytes()).hexdigest() == sha, rel


# ------------------------------------------------------------------ (5) the 10-31 contract is unchanged
def test_required_contract_equals_2c0a323():
    req = sc.collect_required()
    assert sc.collect() == req
    assert len(req) == CONTRACT_2C0A323_COUNT
    assert _sha(req) == CONTRACT_2C0A323_FROZEN_SHA256
    assert set(req) | set(sc.pending_exemptions()) == set(sc.collect_all())
