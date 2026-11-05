"""DIGEST-IMPROVE-1105: regression tests for three corpus-digest gaps (offline: fake LLM, fake engine, inert data).
(1) chunk-stored full documents are in the facts and the title sample, and the share label says what is counted;
(2) REVERTED by orchestrator decision (verify.md D1): "main" and "major" stay denied as at dd5a6a1;
(3) percent_of_sum lets "48% of the sample" pass only when code re-computes it from the right figures, bound to the
clusters the sentence names and to the population it names (verify.md D2-D4); the old percent op gets the same checks."""
import pytest

from tests.digest_fakes import FakeLLM
from vera import corpus_description as cd
from vera import sources_sidebar as sb

KIND = cd.FULLDOC_CHUNKS_KIND
OV = {"source_count": 209, "corpus_updated_at": "2026-11-03T10:00:00+00:00"}
TITLES = [f"Influenza transmission study {w}" for w in "alpha bravo charlie delta echo foxtrot golf".split()] + \
         [f"Coral reef bleaching survey {w}" for w in "hotel india juliet kilo lima".split()] + \
         [f"Unrelated archival note {w}" for w in "mike november oscar papa quebec romeo sierra tango uniform victor "
                                                  "whiskey xray yankee".split()]
assert len(TITLES) == 25
PROF = {"titles": TITLES, "excerpts": [""] * 25, "published_years": ["2005", "2026"], "added_last_30_days": 12,
        "fulldoc_total": 9, "fulldoc_in_sample": 9}
CL = [("influenza epidemiology", list(range(1, 8))), ("coral reef ecology", list(range(8, 13)))]  # 7 + 5 = 12 of 25 = 48%


def run(sentences, prof=PROF, **kw):
    f = FakeLLM(CL, sentences, **kw)
    return cd.run_digest_pipeline(OV, prof, f, {}), f


def kept(sentences, **kw):
    try:
        return run(sentences, **kw)[0]["summary"]
    except cd.DigestError:
        return ""


def fact_id(phrase, prof=PROF):
    facts = cd.grounded_facts(OV, prof)
    return next(f["id"] for f in cd.approved_facts(facts, [{"label": l, "members": m, "count": len(m)} for l, m in CL],
                                                    len(prof["titles"])) if f["phrase"] == phrase)


GOOD = "Influenza epidemiology and coral reef ecology are the subjects."


# --- (1) full documents stored as passages ------------------------------------------------------------------------
class _Res:
    def __init__(self, rows):
        self.rows = rows

    def one(self):
        return self.rows[0]

    def all(self):
        return list(self.rows)

    def scalar(self):
        return self.rows[0][0]

    def __iter__(self):
        return iter(self.rows)


class _Conn:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        q = str(sql)
        self.log.append((q, params))
        if q.startswith("SELECT count(*), max(created_at)"):
            return _Res([(5, None, 2)])
        if "md5(s.source_id::text)" in q:  # the fake returns rows in the order the query asks for: full documents first
            return _Res([("Curated document one", None, KIND), ("Curated document one", None, KIND),
                         ("Curated document two", None, KIND), ("Abstract title", "abs", "abstract_metadata")])
        if "min(substring" in q:
            return _Res([("2020", "2026")])
        if "interval '30 days'" in q:
            return _Res([(1,)])
        if "AS y" in q:
            return _Res([("2020", 1)])
        return _Res([("x", 2)])


class _Eng:
    def __init__(self):
        self.log = []

    def connect(self):
        return _Conn(self.log)


def test_fulldocs_counted_in_facts_and_sampled_first_once_each():
    eng = _Eng()
    overview, profile = cd.stamp_and_profile(eng, 10)
    head = eng.log[0]
    assert "f.provenance->>'stored_text_kind' = :fd_kind" in head[0] and head[1]["fd_kind"] == KIND
    sample_sql = [q for q, _ in eng.log if "md5(s.source_id::text)" in q][0]
    assert "<> :fd_kind" not in sample_sql and "LEFT JOIN vera_vjay.candidates" in sample_sql
    assert profile["titles"] == ["Curated document one", "Curated document two", "Abstract title"]  # duplicate row once
    assert profile["excerpts"] == ["", "", "abs"]
    assert profile["fulldoc_total"] == 2 and profile["fulldoc_in_sample"] == 2
    facts = cd.grounded_facts(overview, profile)
    assert facts["full_documents_stored_as_passages"] == 2 and facts["full_documents_in_sample"] == 2
    phrases = [f["phrase"] for f in cd.approved_facts(facts, [], len(profile["titles"]))]
    assert "2 full documents stored as passages" in phrases


def test_pipeline_share_label_says_full_documents_are_counted():
    out, _ = run([GOOD])
    assert out["topic_shares"]["full_documents_in_sample"] == 9
    line = sb._share_lines(out["topic_shares"])[0]
    assert "share of the 25 sampled sources, including 9 full documents stored as passages" in line
    assert "not in the sample" not in line


def test_share_label_without_fulldocs_and_for_older_digests():
    none = sb._share_lines(cd.topic_shares([{"label": "a", "count": 4}], 5, full_documents_in_sample=0))[0]
    assert "share of the 5 sampled sources, each counted once" in none and "full document" not in none
    one = sb._share_lines(cd.topic_shares([{"label": "a", "count": 4}], 5, full_documents_in_sample=1))[0]
    assert "including 1 full document stored as passages," in one
    legacy = sb._share_lines(cd.topic_shares([{"label": "a", "count": 4}], 5))[0]  # stored before the change
    assert sb.SAMPLE_EXCLUSION in legacy


# --- (2) reverted: ranking words stay denied as at HEAD dd5a6a1 ----------------------------------------------------
@pytest.mark.parametrize("s", ["The main themes are influenza epidemiology and coral reef ecology.",
                               "Major topics include influenza transmission and reef bleaching.",
                               "The main focus is influenza epidemiology.",           # verify.md D1 probes
                               "The major emphasis is influenza epidemiology.",
                               "The main themes are influenza epidemiology, well ahead of reef ecology.",
                               "The main themes are influenza epidemiology; reef ecology is secondary.",
                               "The main theme is influenza epidemiology.",
                               "Most sources concern influenza epidemiology.",
                               "Influenza epidemiology is the largest cluster.",
                               "The majority of sources concern influenza."])
def test_ranking_words_denied_as_before(s):
    assert s not in kept([s, GOOD])


def test_numeric_word_config_is_head_content():
    import json
    import pathlib
    cfg = json.loads((pathlib.Path(cd.__file__).resolve().parent.parent / "config/corpus_digest_numeric_words.json").read_text())
    assert set(cfg) == {"schema", "item", "note", "numeric_words"} and {"main", "major"} <= cd.load_numeric_words()
    assert not hasattr(cd, "load_rank_qualifiers")


# --- (3) percent_of_sum ---------------------------------------------------------------------------------------------
S48 = "Influenza epidemiology and coral reef ecology together make up 48% of the sample."


def pos(num, den, value="48%", rounding="integer"):
    return {S48: [{"value": value, "op": "percent_of_sum", "operands": {"numerator": num, "denominator": den},
                   "rounding": rounding}]}


def test_percent_of_sum_from_cluster_count_facts_kept():
    num = [fact_id("influenza epidemiology: 7 sources"), fact_id("coral reef ecology: 5 sources")]
    assert S48 in kept([S48], derivations=pos(num, fact_id("25 sampled sources")))


def test_percent_of_sum_from_title_ids_kept():
    assert S48 in kept([S48], derivations=pos([f"T{i}" for i in range(1, 13)], fact_id("25 sampled sources")))


def test_plain_percent_op_cannot_express_it():
    num = [fact_id("influenza epidemiology: 7 sources"), fact_id("coral reef ecology: 5 sources")]
    d = {S48: [{"value": "48%", "op": "percent", "operands": num + [fact_id("25 sampled sources")], "rounding": "integer"}]}
    assert S48 not in kept([S48, GOOD], derivations=d)


@pytest.mark.parametrize("case", ["wrong_value", "corpus_denominator", "mixed", "duplicate", "uncited_cluster",
                                  "percent_fact", "list_operands", "extra_key", "foreign_title", "over_100"])
def test_percent_of_sum_denied(case):
    c1, c2 = fact_id("influenza epidemiology: 7 sources"), fact_id("coral reef ecology: 5 sources")
    sample = fact_id("25 sampled sources")
    sentence, cited = S48, [1, 2]
    if case == "wrong_value":
        d = pos([c1, c2], sample, value="50%")
        d = {S48.replace("48%", "50%"): d[S48]}
        sentence = S48.replace("48%", "50%")
    elif case == "corpus_denominator":
        d = pos([c1, c2], fact_id("the corpus holds 209 sources"))
    elif case == "mixed":
        d = pos([c1, "T8"], sample)
    elif case == "duplicate":
        d = pos([c1, c1], sample)
    elif case == "uncited_cluster":
        sentence, cited = "Influenza epidemiology and another subject make up 48% of the sample.", [1]
        d = {sentence: pos([c1, c2], sample)[S48]}
    elif case == "percent_fact":
        d = pos([fact_id("influenza epidemiology: 28%"), fact_id("coral reef ecology: 20%")], sample)
    elif case == "list_operands":
        d = {S48: [{"value": "48%", "op": "percent_of_sum", "operands": [c1, c2, sample], "rounding": "integer"}]}
    elif case == "extra_key":
        d = {S48: [{"value": "48%", "op": "percent_of_sum", "rounding": "integer",
                    "operands": {"numerator": [c1, c2], "denominator": sample, "scale": 1}}]}
    elif case == "foreign_title":
        sentence, cited = "Influenza epidemiology makes up 48% of the sample.", [1]
        d = {sentence: pos([f"T{i}" for i in range(1, 13)], sample)[S48]}
    else:  # numerator larger than the denominator
        d = pos([fact_id("the corpus holds 209 sources")], fact_id("12 sources added in the last 30 days"))
    assert sentence not in kept([(sentence, cited), GOOD], derivations=d)


def test_check_prompt_documents_percent_of_sum():
    assert "percent_of_sum" in cd.CHECK_SYSTEM and '"numerator"' in cd.CHECK_SYSTEM
    assert "compares clusters" not in cd.CHECK_SYSTEM  # intent 2 reverted


# --- verify.md probes (D2-D5): each must be DENIED; the honest counterparts pass --------------------------------------
def d(op, operands, value, rounding="integer"):
    return [{"value": value, "op": op, "operands": operands, "rounding": rounding}]


def F():
    return (fact_id("influenza epidemiology: 7 sources"), fact_id("coral reef ecology: 5 sources"),
            fact_id("25 sampled sources"), fact_id("the corpus holds 209 sources"),
            fact_id("9 full documents stored as passages"))


PROBES = {
    "subset_titles": ("Influenza epidemiology makes up 12% of the sample.", [1],
                      lambda C1, C2, SA, CO, FD: d("percent_of_sum", {"numerator": ["T1", "T2", "T3"], "denominator": SA}, "12%")),
    "wrong_referent": ("Influenza epidemiology makes up 48% of the sample.", [1, 2],
                       lambda C1, C2, SA, CO, FD: d("percent_of_sum", {"numerator": [C1, C2], "denominator": SA}, "48%")),
    "titles_wrong_referent": ("Influenza epidemiology makes up 48% of the sample.", [1, 2],
                              lambda C1, C2, SA, CO, FD: d("percent_of_sum", {"numerator": [f"T{i}" for i in range(1, 13)],
                                                                             "denominator": SA}, "48%")),
    "sample_as_corpus_pos": ("Influenza epidemiology makes up 28% of the corpus.", [1],
                             lambda C1, C2, SA, CO, FD: d("percent_of_sum", {"numerator": [C1], "denominator": SA}, "28%")),
    "sample_as_corpus_percent": ("Influenza epidemiology makes up 28% of the corpus.", [1],
                                 lambda C1, C2, SA, CO, FD: d("percent", [C1, SA], "28%")),
    "sample_as_collection": ("Influenza epidemiology makes up 28% of the collection.", [1],
                             lambda C1, C2, SA, CO, FD: d("percent", [C1, SA], "28%")),
    "mixed_pop_percent": ("Influenza epidemiology makes up 3% of the corpus.", [1],
                          lambda C1, C2, SA, CO, FD: d("percent", [C1, CO], "3%")),
    "mixed_pop_pos": ("Influenza epidemiology makes up 3% of the corpus.", [1],
                      lambda C1, C2, SA, CO, FD: d("percent_of_sum", {"numerator": [C1], "denominator": CO}, "3%")),
    "fd_over_sample_percent": ("Full documents are 36% of the sample.", [1],
                               lambda C1, C2, SA, CO, FD: d("percent", [FD, SA], "36%")),
    "fd_over_sample_pos": ("Full documents are 36% of the sample.", [1],
                           lambda C1, C2, SA, CO, FD: d("percent_of_sum", {"numerator": [FD], "denominator": SA}, "36%")),
    "nearest5_false_precision": ("Influenza epidemiology and coral reef ecology make up 46% of the sample.", [1, 2],
                                 lambda C1, C2, SA, CO, FD: d("percent_of_sum", {"numerator": [C1, C2], "denominator": SA},
                                                              "46%", "nearest_5pct")),
    "list_denominator": ("Influenza epidemiology makes up 28% of the sample.", [1],
                         lambda C1, C2, SA, CO, FD: d("percent_of_sum", {"numerator": [C1], "denominator": [SA]}, "28%")),
    "percent_no_population_named": ("Influenza epidemiology makes up 28%.", [1],
                                    lambda C1, C2, SA, CO, FD: d("percent", [C1, SA], "28%")),
    "ratio_mixed_pop": ("Influenza epidemiology sources equal 0.03 of the corpus.", [1],
                        lambda C1, C2, SA, CO, FD: d("ratio", [C1, CO], "0.03", "1dp")),
    "sum_names_other_cluster": ("Coral reef ecology accounts for 7 of the 25 sampled sources.", [1, 2],
                                lambda C1, C2, SA, CO, FD: d("sum", [C1], "7")),
}


@pytest.mark.parametrize("name", sorted(PROBES))
def test_verify_probe_denied(name):
    s, cited, mk = PROBES[name]
    assert s not in kept([(s, cited), GOOD], derivations={s: mk(*F())})


@pytest.mark.parametrize("s,cited,mk", [
    ("Influenza epidemiology makes up 28% of the sample.", [1], lambda C1, C2, SA, CO, FD: d("percent", [C1, SA], "28%")),
    ("Influenza epidemiology makes up 28% of the sample.", [1],
     lambda C1, C2, SA, CO, FD: d("percent_of_sum", {"numerator": [f"T{i}" for i in range(1, 8)], "denominator": SA}, "28%")),
    ("Influenza epidemiology and coral reef ecology make up about 50% of the sample.", [1, 2],
     lambda C1, C2, SA, CO, FD: d("percent_of_sum", {"numerator": [C1, C2], "denominator": SA}, "50%", "nearest_5pct")),
    ("Full documents are 4% of the corpus.", [1], lambda C1, C2, SA, CO, FD: d("percent", [FD, CO], "4%")),
])
def test_honest_bound_derivations_kept(s, cited, mk):
    assert s in kept([(s, cited)], derivations={s: mk(*F())})


def test_fulldoc_count_is_distinct_titles_in_sql():
    eng = _Eng()
    cd.stamp_and_profile(eng, 10)
    q = eng.log[0][0]
    assert "count(DISTINCT lower(btrim(regexp_replace(coalesce(f.provenance->>'title', g.title)" in q
    assert "f.provenance->>'stored_text_kind' = :fd_kind" in q
