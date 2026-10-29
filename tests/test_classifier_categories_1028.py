"""OOE-APRIME-C-1028 (OD-1028(38), owner "a' + c"): the owner-curated, read-only layer-2 category table
config/classifier_categories.json is the single source of truth; it is validated on load and fails closed."""
import copy
import dataclasses
import itertools
import json
import re
from pathlib import Path

import pytest

from vera import scope_router as sr

ROOT = Path(__file__).resolve().parent.parent
RAW = json.loads((ROOT / "config" / "classifier_categories.json").read_text())


def test_table_maps_the_six_categories_to_their_current_behaviour():
    assert sr.LAYER2_CATEGORIES == ("in_scope", "off_topic", "advice", "harmful", "programming_task", "rule_change")
    assert sr.REFUSE_CATEGORIES == frozenset({"advice", "harmful", "programming_task", "rule_change"})
    assert sr._REFUSE_ORDER == ("harmful", "rule_change", "advice", "programming_task")
    assert sr._CATS == r"(harmful|rule[ _]?change|advice|programming[ _]?task)"
    assert sr._RAW_REFUSAL.pattern == r"(?<![a-z])(harmful|rule[ _-]?change|advice|programming[ _-]?task)(?![a-z])"
    assert sr._CAT_SPELLINGS == {"harmful": "harmful", "advice": "advice", "rule_change": "rule_change",
                                 "rulechange": "rule_change", "programming_task": "programming_task",
                                 "programmingtask": "programming_task"}
    assert sr.UNSUPPORTED_CATEGORY == "programming_task"
    assert sr.CODING_HINT_CONFIRM == frozenset({"programming_task", "off_topic", "in_scope"})
    assert dict(sr.CATEGORY_ALIASES) == {} and all(c.aliases == () for c in sr.CATEGORY_TABLE)
    assert "OD-1028(38)" in RAW["_note"] and "owner-curated" in RAW["_note"].lower()


def test_prompt_category_section_is_rendered_from_the_table():
    p = sr.load_config().layer2["system_prompt"]
    assert ('"category": "in_scope" or "off_topic" or "advice" or "harmful" or "programming_task" or "rule_change",'
            in p)
    for c in sr.CATEGORY_TABLE:
        assert f'"{c.code}": {c.description}' in p
    assert "{{" not in p
    tmpl = json.loads((ROOT / "config" / "scope_router.json").read_text())["layer2"]["system_prompt"]
    assert "programming_task" not in tmpl and "rule_change" not in tmpl  # no second category list in the prompt
    with pytest.raises(sr.ScopeConfigError):
        sr.render_category_prompt("no slots here")


def test_table_is_read_only_at_runtime():
    assert isinstance(sr.CATEGORY_TABLE, tuple) and isinstance(sr.LAYER2_CATEGORIES, tuple)
    assert isinstance(sr.REFUSE_CATEGORIES, frozenset)
    row = sr.CATEGORY_TABLE[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.code = "new_route"
    assert isinstance(row.aliases, tuple)
    with pytest.raises(TypeError):
        sr.CATEGORY_ALIASES["x"] = "in_scope"
    # no code path writes the JSON seed file
    for f in (ROOT / "vera").rglob("*.py"):
        src = f.read_text(errors="ignore")
        if "clscat_categories" in src:
            assert f.name == "scope_router.py", f
            assert "CATEGORIES_PATH = " in src and "write_text" not in src and "json.dump" not in src
            assert not re.search(r"open\([^)]*['\"][wa]", src)


# OD-1028(38) ("the table is read-only at runtime; no writer exists", the earlier regex guard) is SUPERSEDED IN PART by
# OD-1028(39)/(42): the DB table may receive runtime INSERTs of brand-new, non-refusing codes through ONE gated function.
# The regex guard was defeated by naming the function enrol_nonrefusing_entry, so it is replaced by explicit assertions.
import ast  # noqa: E402

_SQL_VERBS = re.compile(r"\b(UPDATE|DELETE|TRUNCATE|ALTER|COPY|ON\s+CONFLICT)\b", re.I)
# OWNER DECISION 2026-10-28 ("the tables and funtions should be disambiguated by prefix & suffix as appropriate"):
# the table is renamed classifier_categories -> clscat_categories; the guards below track the new literal explicitly (sanctioned rename, not an evasion).
_TABLE_SQL = re.compile(r"\b(FROM|INTO|UPDATE|TABLE|JOIN|COPY|ON)\s+(vera_vjay\.)?clscat_categories\b|vera_vjay\.clscat_categories", re.I)
_SOURCES = sorted((ROOT / "vera").rglob("*.py")) + [ROOT / "main.py"]
_GUARD_SOURCES = _SOURCES + sorted((ROOT / "scripts").rglob("*.py"))  # R2: scripts/ too


def _docstring_nodes(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body \
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
            out.add(id(n.body[0].value))
    return out


def _string_constants():
    for f in _SOURCES:
        tree = ast.parse(f.read_text(errors="ignore"))
        skip = _docstring_nodes(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip:
                yield f, n.value


def test_guard_a_only_three_sql_texts_name_the_table_and_none_mutate_it():
    assert not hasattr(sr, "DB_CATEGORY_SQL_UPDATE") and not hasattr(sr, "DB_CATEGORY_SQL_DELETE")
    allowed = {sr.DB_CATEGORY_SQL_EXISTS, sr.DB_CATEGORY_SQL_ROWS, sr.DB_CATEGORY_SQL_INSERT}
    named = {v for f, v in _string_constants() if _TABLE_SQL.search(v)}
    assert named == allowed, named ^ allowed
    # every other SQL-like text near the table in any vera/ source: no mutating verb. Per file that names the table,
    # no string constant (other than the three) holds a mutating SQL verb or ON CONFLICT.
    for f, v in _string_constants():
        if v in allowed or "clscat_categories" not in f.read_text(errors="ignore"):
            continue
        assert not _SQL_VERBS.search(v) or not re.search(r"\b(SELECT|INSERT|SET|FROM|INTO|WHERE)\b", v), (f.name, v[:60])
    for v in allowed - {sr.DB_CATEGORY_SQL_INSERT}:
        assert not _SQL_VERBS.search(v)
    ins = sr.DB_CATEGORY_SQL_INSERT
    assert ins.startswith("INSERT INTO vera_vjay.clscat_categories") and "ON CONFLICT" not in ins.upper()
    assert not _SQL_VERBS.search(ins.replace("INSERT", ""))
    # the lock statement names no table and no mutation
    assert "clscat_categories" not in sr.DB_CATEGORY_SQL_LOCK


def test_guard_b_insert_is_brand_new_non_refusing_only():
    ins = " ".join(sr.DB_CATEGORY_SQL_INSERT.split())
    assert "(code, version, description, disposition, aliases, origin, state)" in ins
    assert "VALUES (:code, 1, :description, :disposition, '{}', 'runtime', :state)" in ins
    assert "refusal_rank" not in ins and ":version" not in ins and ":origin" not in ins and ":aliases" not in ins
    assert re.findall(r":(\w+)", ins) == ["code", "description", "disposition", "state"]  # no app-supplied version/origin/rank


def test_guard_c_only_enrol_nonrefusing_entry_references_the_insert():
    refs = []
    for f in _SOURCES:
        tree = ast.parse(f.read_text(errors="ignore"))
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = ast.dump(fn)
                if "DB_CATEGORY_SQL_INSERT" in body and not any(
                        isinstance(o, (ast.FunctionDef, ast.AsyncFunctionDef)) and o is not fn and "DB_CATEGORY_SQL_INSERT" in ast.dump(o)
                        for o in ast.walk(fn)):
                    refs.append((f.name, fn.name))
        # module-level (non-function) uses other than the definition itself
        for n in tree.body:
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and "DB_CATEGORY_SQL_INSERT" in ast.dump(n):
                assert isinstance(n, ast.Assign) and n.targets[0].id == "DB_CATEGORY_SQL_INSERT", (f.name, ast.unparse(n)[:80])
    assert refs == [("scope_router.py", "enrol_nonrefusing_entry")], refs
    on = {"registration_enabled": True}
    for disp in ("refuse", "unsupported_use"):
        with pytest.raises(sr.RegistrationRefused):
            sr.enrol_nonrefusing_entry("ok_code", "notes about a local seminar timetable", disp, settings=on, rows=[],
                                       writer=lambda r: 1 / 0)
    for flag in (None, False, "true", 1, "yes"):
        with pytest.raises(sr.RegistrationRefused):
            sr.enrol_nonrefusing_entry("ok_code", "notes about a local seminar timetable", settings={"registration_enabled": flag},
                                       rows=[], writer=lambda r: 1 / 0)


def test_guard_d_shipped_config_registration_disabled():
    cfg = json.loads((ROOT / "config" / "scope_router.json").read_text())["classifier_category_table"]
    assert cfg["registration_enabled"] is False and cfg["read_from_db"] is False
    assert sr.category_table_settings()["registration_enabled"] is False


def test_guard_e_docstring_states_the_write_limit():
    doc = " ".join(sr.enrol_nonrefusing_entry.__doc__.split())
    assert "Runtime writes are limited to brand-new, non-refusing codes" in doc
    assert "OD-1028(9)/(39)/(42)" in doc


def _with(mutate):
    raw = copy.deepcopy(RAW)
    mutate(raw["categories"])
    return raw


@pytest.mark.parametrize("mutate,msg", [
    (lambda rows: rows.append(dict(rows[0], description="totally new wording for questions here")), "duplicate"),
    (lambda rows: rows[2].update(disposition="redirect_advice"), "unknown disposition"),
    (lambda rows: rows[0].update(aliases=["off_topic"]), "collides"),
    (lambda rows: (rows[0].update(aliases=["dangerous"]), rows[1].update(aliases=["dangerous"])), "collides"),
    (lambda rows: rows[2].update(refusal_rank=None), "refusal_rank"),
    (lambda rows: rows[2].update(refusal_rank=1), "unique"),
    (lambda rows: rows[0].update(refusal_rank=5), "must be null"),
    (lambda rows: rows[0].update(code="In Scope"), "must match"),
    (lambda rows: rows[0].pop("aliases"), "exactly"),
    (lambda rows: rows[0].update(description='say "x"'), "description"),
    (lambda rows: rows.pop(4), "unsupported_use"),
    (lambda rows: rows.clear(), "non-empty"),
], ids=["dup-code", "bad-disposition", "alias-is-code", "alias-twice", "rank-missing", "rank-dup", "rank-on-nonrefusing",
        "bad-code", "missing-field", "quote-in-description", "no-unsupported-row", "empty"])
def test_invalid_table_fails_closed(mutate, msg):
    with pytest.raises(sr.ScopeConfigError, match=msg):
        sr.parse_categories(_with(mutate))


def test_invalid_table_file_is_a_load_error():
    with pytest.raises(sr.ScopeConfigError):
        sr.load_categories(Path(__file__).resolve().parent / "no_such_categories.json")


def test_shipped_descriptions_pass_with_calibrated_margin():
    """Owner rule 2026-10-28: 'a new C entry must have a significantly dissimlar description from all other
    descriptions'. Measure (review D1): max(Jaccard of suffix-stemmed content tokens, character-trigram Jaccard over
    sorted content tokens); lowercased, stopwords incl. 'defined', 'below' removed. Calibration on the six shipped
    descriptions: pairwise max 0.203 (harmful vs rule_change), next 0.172 (advice vs harmful).
    DESCRIPTION_SIMILARITY_MAX = 0.50 = max + 0.297 margin, so all six pass. Re-calibrate when a row is added."""
    scores = sorted(((round(sr.description_similarity(a.description, b.description), 3), a.code, b.code)
                     for a, b in itertools.combinations(sr.CATEGORY_TABLE, 2)), reverse=True)
    assert scores[0] == (0.203, "harmful", "rule_change") and sr.DESCRIPTION_SIMILARITY_MAX == 0.50
    assert scores[0][0] + 0.25 <= sr.DESCRIPTION_SIMILARITY_MAX


# review D1 paraphrase probes (R/work/ooe-review-1028/probe.py): word-form paraphrases of a shipped description
CAUGHT = {
    "advice": "requests seeking personalised medical, legal, or financial advising",
    "programming_task": "requests to write or fixing coding",
    "rule_change": "requests changing rules and revealing configurations",
    "off_topic": "harmless questions that fall outside those fields",
}
KNOWN_LIMIT = {"harmful": "requesting operational assistance in causing harms"}  # synonym swap: scores 0.429


@pytest.mark.parametrize("code", sorted(CAUGHT))
def test_word_form_paraphrase_probe_rejected(code):
    with pytest.raises(sr.ScopeConfigError, match=code):
        sr.parse_categories(_with(_new_row(CAUGHT[code])))


def test_synonym_paraphrase_probe_is_a_recorded_known_limit():
    """Known limit (recorded in the config note): a synonym paraphrase ('assistance' for 'help') is not caught by a
    lexical measure; owner curation of the table remains the control."""
    s = sr.description_similarity(KNOWN_LIMIT["harmful"], next(c.description for c in sr.CATEGORY_TABLE
                                                                   if c.code == "harmful"))
    assert round(s, 3) == 0.429 < sr.DESCRIPTION_SIMILARITY_MAX
    assert sr.parse_categories(_with(_new_row(KNOWN_LIMIT["harmful"])))[-1].code == "new_cat"


def test_category_table_is_validated_at_service_start():
    """Review D2: main.py imports the scope router at module level (process start), so an invalid table raises
    ScopeConfigError before the app serves; request-path lazy imports remain a second line."""
    import ast
    tree = ast.parse((ROOT / "main.py").read_text())
    top = [n for n in tree.body if isinstance(n, ast.Import)]
    assert any(a.name == "vera.scope_router" for n in top for a in n.names)
    calls = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and ast.unparse(n.value.func) == "_scope_router.load_config"]
    assert calls, "FINAL-REVIEW-1028 D3: main.py must call _scope_router.load_config() at startup"
    import main
    assert main._scope_router is sr and sr.CATEGORY_TABLE


def _new_row(desc):
    return lambda rows: rows.append({"code": "new_cat", "description": desc, "disposition": "refuse",
                                     "refusal_rank": 5, "aliases": []})


def test_near_duplicate_description_rejected_naming_both_codes_and_score():
    with pytest.raises(sr.ScopeConfigError) as exc:
        sr.parse_categories(_with(_new_row("a request for operational help causing harm (defined below)")))
    m = str(exc.value)
    assert "harmful" in m and "new_cat" in m and re.search(r"score 0\.\d{3}", m)


def test_clearly_distinct_new_description_accepted():
    rows = sr.parse_categories(_with(_new_row("questions seeking private details of a named person")))
    assert rows[-1].code == "new_cat" and isinstance(rows, tuple)


# ---- CATTAB-FIX2 R2: hardened AST guard (string building, sqlalchemy Table, scripts/) ----
def _name_violations(tree, allowed, fname):
    skip = _docstring_nodes(tree)
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip \
                and "clscat_categories" in n.value and n.value not in allowed:
            bad.append((fname, "constant", n.value[:50]))
        if isinstance(n, ast.Call) and ast.unparse(n.func).split(".")[-1] == "Table" and n.args \
                and isinstance(n.args[0], ast.Constant) and n.args[0].value == "clscat_categories":
            bad.append((fname, "Table", "clscat_categories"))
    return bad


def _allowed_texts():
    # the three SQL texts plus two non-SQL labels (JSON seed filename, error-message prefix) that name no table access
    return {sr.DB_CATEGORY_SQL_EXISTS, sr.DB_CATEGORY_SQL_ROWS, sr.DB_CATEGORY_SQL_INSERT,
            "classifier_categories.json", "clscat_categories (db)"}


def test_guard_f_no_other_string_names_the_table_in_vera_main_scripts():
    bad = []
    for f in _GUARD_SOURCES:
        allowed = _allowed_texts() if (f.name == "scope_router.py" and f.parent.name == "vera") else set()
        bad += _name_violations(ast.parse(f.read_text(errors="ignore")), allowed, f.name)
    assert bad == [], bad
    assert any(f.parent.name == "scripts" for f in _GUARD_SOURCES)


@pytest.mark.parametrize("src", [
    'q = "SELECT * FROM vera_vjay." + "clscat_categories"',
    'q = "DELETE FROM x.clscat_categories WHERE {}".format(1)',
    'name = f"UPDATE clscat_categories SET a={x}"',
    'from sqlalchemy import Table\nt = Table("clscat_categories", md)',
    'import sqlalchemy as sa\nt = sa.Table("clscat_categories", md)',
], ids=["plus", "format", "fstring", "Table", "sa-Table"])
def test_guard_f_detects_evasions(src):
    assert _name_violations(ast.parse(src), _allowed_texts(), "x.py")


def test_guard_f_ignores_docstrings_and_allowed_texts():
    assert _name_violations(ast.parse('"""clscat_categories"""\nx = 1'), set(), "x.py") == []
    assert _name_violations(ast.parse(f"x = {sr.DB_CATEGORY_SQL_ROWS!r}"), _allowed_texts(), "x.py") == []


# ---- CATTAB-FIX2 R1: owner / migration preflight ----
DRAFT = (ROOT / "db" / "migrations" / "pending" / "012-DRAFT-classifier-categories.sql.draft").read_text()


def test_app_role_list_is_one_constant_and_matches_the_draft():
    assert sr.APP_ROLE_NAMES == frozenset({"vera_vjay_wo", "vera_vjay_rw", "vera_vjay_ro", "vera_claude_code_rw",
                                           "vera_pipeline_wo", "vera_eval_ro"})
    pre = DRAFT.split("R1(a) preflight", 1)[1].split("CREATE TABLE", 1)[0]
    assert set(re.findall(r"'(vera_\w+)'", pre)) >= {"vera_vjay_wo", "vera_vjay_rw", "vera_vjay_ro"}
    assert "current_setting('vera012.rw')" in pre  # logins come from the psql variables (defaults = the three names below)
    head = DRAFT.split("BEGIN;", 1)[0]
    assert {"vera_claude_code_rw", "vera_pipeline_wo", "vera_eval_ro"} <= set(re.findall(r"vera_\w+", head))
    assert "pg_auth_members" in pre and "current_user" in pre and "RAISE EXCEPTION" in pre
    assert "(m.inherit_option OR m.set_option)" in pre  # PG16+ ADMIN-only rows must not lock out the owner
    assert "PG 18.6" in pre and "dedicated owner role" in pre
    src = (ROOT / "vera" / "scope_router.py").read_text()
    assert src.count("vera_claude_code_rw") == 1  # no scattered literals


class _OwnerConn:
    def __init__(self, owner, exists=True):
        self.owner, self.exists, self.log = owner, exists, []

    def execute(self, stmt, params=None):
        self.cur = str(stmt)
        self.log.append((self.cur, params))
        return self

    def scalar(self):
        if "to_regclass" in self.cur:
            return "vera_vjay.clscat_categories" if self.exists else None
        return self.owner if "relowner" in self.cur else None

    def mappings(self):
        return self

    def all(self):
        return []


@pytest.mark.parametrize("owner", sorted(sr.APP_ROLE_NAMES) + [None])
def test_owner_app_role_or_missing_fails_closed_without_host(owner):
    with pytest.raises(sr.ScopeConfigError, match="owner") as e:
        sr._select_rows(_OwnerConn(owner))
    assert "@" not in str(e.value) and "host" not in str(e.value).lower()


def test_non_app_owner_passes_and_owner_query_names_no_table():
    c = _OwnerConn("vera_owner_admin")
    assert sr._select_rows(c) == []
    assert c.log[1][1] == {"t": "vera_vjay.clscat_categories"}
    assert "clscat_categories" not in sr.DB_CATEGORY_SQL_OWNER
    assert sr._select_rows(_OwnerConn("x", exists=False)) is None


def test_read_category_rows_propagates_owner_failure_as_scope_error():
    class Eng:
        def connect(self):
            return _Ctx(_OwnerConn("vera_vjay_rw"))

    class _Ctx:
        def __init__(self, c):
            self.c = c

        def __enter__(self):
            return self.c

        def __exit__(self, *a):
            return False

    with pytest.raises(sr.ScopeConfigError, match="owner"):
        sr.read_category_rows(Eng())
