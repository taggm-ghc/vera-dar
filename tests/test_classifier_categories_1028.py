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
    # no writer exists: the table file is named only by the loader's read path, and no module opens it for writing
    for f in (ROOT / "vera").rglob("*.py"):
        src = f.read_text(errors="ignore")
        if "classifier_categories" in src:
            assert f.name == "scope_router.py", f
            assert "CATEGORIES_PATH = " in src and "write_text" not in src and "json.dump" not in src
            assert not re.search(r"open\([^)]*['\"][wa]", src)
    assert not [n for n in dir(sr) if re.match(r"(save|write|add|set|update|register)_?categor", n, re.I)]


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
