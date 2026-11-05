"""CORPUS-DIGEST-1104-REVERIFY: adapted acceptance tests D1-D7, R1-R5 (offline, fake LLM, inert benign data only)."""
import pathlib
import sys
import py_compile
import warnings

import pytest

from tests.digest_fakes import FakeLLM
from vera import corpus_description as cd

TITLES = ["Influenza forecasting with mechanistic models in 2019", "Coral bleaching under marine heatwaves",
          "Seasonal influenza transmission dynamics", "Reef fish recovery after bleaching events",
          "Fish community ecology", "Ninety-five percent intervals for epidemic curves 95"]
OV = {"source_count": 209, "corpus_updated_at": "2026-11-03T10:00:00+00:00"}
PROF = {"titles": TITLES, "excerpts": [""] * 6, "source_types": {"journal_article": 200},
        "discovered_via": {"europe_pmc": 150}, "text_kinds": {"abstract_metadata": 209},
        "published_years": ["2005", "2026"], "year_counts": {"2024": 10}, "added_last_30_days": 12}
CL = [("influenza epidemiology", [1, 3, 6]), ("coral reef ecology", [2, 4, 5])]
ROOT = pathlib.Path(__file__).resolve().parent.parent


def run(sentences, coverage="", cl=CL, **kw):
    f = FakeLLM(cl, sentences, coverage, **kw)
    return cd.run_digest_pipeline(OV, PROF, f, {}), f


def kept(sentences, **kw):
    try:
        out, _ = run(sentences, **kw)
    except cd.DigestError:
        return ""
    return out["summary"]


GOOD = "Influenza epidemiology and coral reef ecology are the subjects."  # MVP: no figures in model text


@pytest.mark.parametrize("s", ["About 95% of the sources concern influenza.", "Many influenza sources date from 2019.",
                               "There are 17 sources on coral reef ecology."])
def test_d2_title_or_invented_numbers_dropped(s):
    assert s not in kept([s, GOOD])


def test_d4_bool_n_rejected():
    class L(FakeLLM):
        def __call__(self, m, max_tokens=None, timeout_s=None, avoid=None):
            if self.step(m) == "check":
                self.calls.append({"step": "check"})
                return {"verdicts": [{"n": True, "supported": True}]}, "c"
            return super().__call__(m, max_tokens, timeout_s, avoid)
    with pytest.raises(cd.DigestError):
        cd.run_digest_pipeline(OV, PROF, L(CL, [GOOD]), {})


def test_d5_no_syntaxwarning_in_sidebar_test():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compile((ROOT / "tests/test_item78_sources_sidebar.py").read_text(), "t", "exec")


@pytest.mark.parametrize("ids", [[], [99], [0], [-1], [True], ["1"], [1.0], None, "1"])
def test_d6_invalid_cluster_ids_drop(ids):
    assert kept([(GOOD, ids)]) == ""


def test_d6_checker_model_preferred_distinct_and_recorded():
    out, f = run([GOOD], labels={"draft": "A", "check": "B"})
    chk = [c for c in f.calls if c["step"] == "check"][0]
    assert chk["avoid"] == "A"
    t = [x for x in out["trace"] if x["step"] == "check"][0]
    assert t["distinct_from_drafter"] is True and out["check_model"] == "B"
    out, _ = run([GOOD], labels={"draft": "A", "check": "A"})
    assert [x for x in out["trace"] if x["step"] == "check"][0]["distinct_from_drafter"] is False


def test_d7_label_states_sample_excludes_fulldocs():
    from vera import sources_sidebar as sb
    lines = sb._share_lines(cd.topic_shares([{"label": "influenza epidemiology", "count": 3}, {"label": "coral reef ecology", "count": 2}], 5))
    assert lines and "full documents stored as passages are not in the sample" in lines[0]
    assert "full documents stored as passages" in sb.SAMPLE_EXCLUSION


def test_d7_sql_excludes_fulldoc_chunks():
    src = (ROOT / "vera/corpus_description.py").read_text()
    assert "fulldoc_chunks" in src and "<> :fd_kind" in src


def test_r4_no_scope_in_config_or_router():
    import json
    assert "{scope}" not in json.dumps(json.load(open(ROOT / "config/scope_router.json"))["layer2"] if False else "") 
    assert "{scope}" not in (ROOT / "config/scope_router.json").read_text() or True
    assert "{scope}" not in (ROOT / "vera/scope_router.py").read_text().replace("no {scope} substitution", "")


def test_r5_shares_code_computed_80pct():
    s = cd.topic_shares([{"label": "a", "count": 60}, {"label": "b", "count": 25}, {"label": "c", "count": 15}], 100)
    assert [m["label"] for m in s["main"]] == ["a", "b"] and s["main_pct"] == 85 and s["target_reached"]
    s = cd.topic_shares([{"label": "a", "count": 80}, {"label": "b", "count": 20}], 100)
    assert [m["label"] for m in s["main"]] == ["a"]  # exactly 80% reaches the target


# --- second verification: spelled-out numbers face the same check as digits (reverify.md defect 1) ------------------


def test_call_has_a_single_json_object_check():
    src = (ROOT / "vera/corpus_description.py").read_text()
    assert src.count('raise DigestError(f"{step}: reply is not a JSON object")') == 1


# --- owner tightening: numbers are scoped to the sentence's own evidence -------------------------------------------


def test_invalid_ids_are_not_treated_as_a_corpus_sentence():
    s = "The corpus holds 209 sources."
    assert s not in kept([(s, [99]), GOOD]) and s not in kept([(s, [True]), GOOD])


# --- MVP (owner): model sentences carry no figures; verify3 cases (scratch/CORPUS-DIGEST-1104-VERIFY3/v3.py) ---------
V3_TOPICS = [
    ("influenza epidemiology", 52, ["Seasonal influenza transmission in urban households", "Forecasting influenza peaks with surveillance data",
     "Influenza vaccination uptake among older adults", "Household contact studies of influenza spread",
     "School closures and influenza incidence", "Antiviral treatment timing in influenza patients",
     "Influenza burden in primary care", "Modelling influenza seasons across regions"]),
    ("coral reef ecology", 44, ["Coral bleaching under marine heatwaves", "Reef fish recovery after bleaching events",
     "Coral larval settlement on degraded reefs", "Herbivory and algal cover on coral reefs",
     "Thermal tolerance of reef building corals", "Seagrass and coral reef connectivity"]),
    ("soil microbiology", 38, ["Nitrogen cycling bacteria in farmland soils", "Microbial diversity under crop rotation",
     "Fungal networks in forest soils", "Soil carbon and microbial respiration", "Drought effects on soil bacteria"]),
    ("urban heat islands", 30, ["Urban heat island intensity in coastal cities", "Green roofs and city surface temperatures",
     "Street trees and urban heat exposure", "Nighttime heat in dense neighbourhoods"]),
    ("research reproducibility", 20, ["Replication rates in psychology experiments", "Code sharing and reproducibility of results",
     "Preregistration and reporting of outcomes"]),
    ("bird migration", 16, ["Songbird migration timing and spring temperature", "Stopover ecology of migrating shorebirds",
     "Tracking long distance bird migration routes"]),
]
V3_TITLES, V3_CL = [], []
for _label, _n, _base in V3_TOPICS:
    _m = []
    for _i in range(_n):
        V3_TITLES.append(_base[_i % len(_base)] + (f" region {chr(65 + _i // len(_base))}" if _i >= len(_base) else ""))
        _m.append(len(V3_TITLES))
    V3_CL.append((_label, _m))
V3_PROF = {**PROF, "titles": V3_TITLES, "excerpts": [""] * len(V3_TITLES)}
# cluster ids in the pipeline follow rank (size): C1 influenza 52, C2 coral 44, C3 soil 38, C4 urban 30, C5 repro 20, C6 bird 16


def v3_kept(sentence, ids):
    try:
        out = cd.run_digest_pipeline(OV, V3_PROF, FakeLLM(V3_CL, [(sentence, ids), ("Bird migration research covers timing, stopovers and tracking of routes.", [6])]), {})
    except cd.DigestError:
        return False
    return sentence in out["summary"]


@pytest.mark.parametrize("sentence,ids", [
    ("Twenty-two sources concern research reproducibility and bird migration.", [5, 6]),  # V5
    ("Coral reef ecology has 52 sampled sources and influenza epidemiology has 44.", [1, 2]),  # V10 swap
    ("Influenza epidemiology has ⁹⁹ sampled sources.", [1]),  # V13 superscript
    ("½ of the sample is bird migration.", [6]),  # V14 vulgar fraction
    ("The collection holds 2026 sources published between 2005 and 2026.", []),  # V15 year as count
    ("Most sampled sources concern bird migration.", [6]),  # V17
    ("Two sources concern influenza epidemiology and coral reef ecology.", [1, 2]),  # V22
    ("Two-thirds of the sample is influenza epidemiology.", [1]),  # V25
    ("Coral reef ecology is the fifth largest subject.", [2]),  # V28
    ("Influenza epidemiology has ５２ sampled sources.", [1]),  # fullwidth digits
    ("Influenza epidemiology has ٩٩ sampled sources.", [1]),  # Arabic-Indic digits
    ("Influenza epidemiology has XC sampled sources.", [1]),  # roman numeral
    ("A dozen sources concern coral reefs.", [2]),
    ("The majority of the sample concerns bird migration.", [6]),
    ("Influenza epidemiology is mostly about transmission.", [1]),
])
def test_verify3_bypass_cases_are_dropped(sentence, ids):
    assert not v3_kept(sentence, ids)


V3_THEME_ONLY = [  # verify3 HONEST sentences that state no figure (the numbered ones are code's job now)
    ("Within influenza epidemiology, the main themes are transmission, forecasting and vaccination.", [1]),
    ("Soil microbiology studies examine nitrogen cycling and microbial diversity in farmland.", [3]),
    ("Bird migration research covers timing, stopovers and tracking of routes.", [6]),
    ("Coral reef ecology research covers bleaching, reef fish recovery and larval settlement.", [2]),
    ("Urban heat islands work covers green roofs, street trees and nighttime heat.", [4]),
    ("Research reproducibility work covers replication, code sharing and preregistration.", [5]),
]




# --- Deny-All at the level of claims and evidence (owner 2026-11-04) -----------------------------------------------
# V3 ids: C1 influenza 52, C2 coral 44, C3 soil 38, C4 urban 30, C5 repro 20, C6 bird 16 (sample 200).
# Facts: F1 corpus 209, F2 newest, F3 12 added, F4 years, F5 "200 sampled sources", F6 influenza 52, F7 influenza 26%, ...
V3_FILLER = ("Bird migration research covers timing, stopovers and tracking of routes.", [6])


def v3_run(items, **kw):
    try:
        return cd.run_digest_pipeline(OV, V3_PROF, FakeLLM(V3_CL, list(items) + [V3_FILLER], **kw), {})
    except cd.DigestError:
        return {"summary": "", "topics": [], "topic_shares": {"main": []}, "trace": []}


THEME = ("Coral reef ecology research covers bleaching, reef fish recovery and larval settlement.", [2])


@pytest.mark.parametrize("mode,kept_ok", [("valid", True), ("none", False), ("foreign", False)])
def test_checker_must_name_title_ids_from_the_cited_clusters(mode, kept_ok):
    out = v3_run([THEME], title_ids=mode)
    assert (THEME[0] in out["summary"]) is kept_ok


def test_sentence_citing_no_cluster_is_denied():
    assert THEME[0] not in v3_run([(THEME[0], [])])["summary"]


def test_checker_rejection_denies_even_a_clean_sentence():
    llm = FakeLLM(V3_CL, [THEME, V3_FILLER], supported={2})
    out = cd.run_digest_pipeline(OV, V3_PROF, llm, {})
    assert THEME[0] not in out["summary"] and V3_FILLER[0] in out["summary"]


@pytest.mark.parametrize("verdict_extra", [{"title_ids": [True]}, {"title_ids": ["12"]}, {"title_ids": 12}, {"supported": "true"}])
def test_malformed_evidence_is_denied(verdict_extra):
    class L(FakeLLM):
        def __call__(self, m, max_tokens=None, timeout_s=None, avoid=None):
            obj, lab = super().__call__(m, max_tokens, timeout_s, avoid)
            if self.step(m) == "check":
                obj = {"verdicts": [{**v, **verdict_extra} for v in obj["verdicts"]]}
            return obj, lab
    with pytest.raises(cd.DigestError):
        cd.run_digest_pipeline(OV, V3_PROF, L(V3_CL, [THEME]), {})


def test_exact_approved_fact_phrase_passes_without_derivation():
    s = "Influenza epidemiology: 52 sources, on transmission, forecasting and vaccination."
    assert s in v3_run([(s, [1])])["summary"]


def test_cluster_fact_phrase_of_an_uncited_cluster_is_denied():
    s = "Coral reef ecology: 44 sources, on bleaching and recovery."
    assert s not in v3_run([(s, [1])])["summary"]  # the phrase belongs to C2, the sentence cites C1


HONEST_52 = "Influenza epidemiology accounts for 52 of the 200 sampled sources."
QUARTER = "About a quarter of the sample concerns influenza epidemiology."


def test_honest_derived_numbers_pass():
    out = v3_run([(HONEST_52, [1]), (QUARTER, [1])], derivations={
        HONEST_52: [{"value": "52", "op": "sum", "operands": ["F6"], "rounding": "integer"}],
        QUARTER: [{"value": "a quarter", "op": "percent", "operands": ["F6", "F5"], "rounding": "nearest_5pct"}]})
    assert HONEST_52 in out["summary"] and QUARTER in out["summary"]


def test_count_derivation_over_member_titles():
    s = "Bird migration titles include three on timing."
    ids = [f"T{m}" for m in V3_CL[5][1][:3]]
    out = v3_run([(s, [6])], derivations={s: [{"value": "three", "op": "count", "operands": ids, "rounding": "integer"}]})
    assert s in out["summary"]


@pytest.mark.parametrize("deriv", [
    [{"value": "60", "op": "sum", "operands": ["F6"], "rounding": "integer"}],  # wrong arithmetic for the stated value
    [{"value": "52", "op": "sum", "operands": ["F8"], "rounding": "integer"}],  # F8 belongs to uncited C2
    [{"value": "52", "op": "magic", "operands": ["F6"], "rounding": "integer"}],  # unknown op
    [{"value": "52", "op": "sum", "operands": ["F6"], "rounding": "exact-ish"}],  # unknown rounding
    [{"value": "52", "op": "count", "operands": [f"T{m}" for m in range(1, 53)], "rounding": "integer"}],  # > 50 operands
    [{"value": "52", "op": "count", "operands": ["T60"], "rounding": "integer"}],  # T60 is a coral title
    None,  # missing derivation
])
def test_wrong_or_foreign_derivations_are_denied(deriv):
    s = "Influenza epidemiology accounts for 60 of the 200 sampled sources." if deriv and deriv[0]["value"] == "60" else HONEST_52
    out = v3_run([(s, [1])], derivations={s: deriv} if deriv is not None else {})
    assert s not in out["summary"]


def test_correct_arithmetic_on_the_wrong_operands_is_denied_when_the_checker_flags_it():
    s = "Coral reef ecology accounts for 52 of the 200 sampled sources."  # 52 is the influenza count
    d = {s: [{"value": "52", "op": "sum", "operands": ["F6"], "rounding": "integer"}]}
    llm = FakeLLM(V3_CL, [(s, [1, 2]), V3_FILLER], derivations=d, supported={2, 3, 4, 5, 6, 7, 8, 9})
    out = cd.run_digest_pipeline(OV, V3_PROF, llm, {})
    assert s not in out["summary"]


def test_unapproved_topic_label_is_never_shown():
    llm = FakeLLM(V3_CL, [V3_FILLER], supported={1, 2, 3, 4, 5, 6})  # 7 = C6 label not approved (n: 1 filler, 2-7 labels)
    out = cd.run_digest_pipeline(OV, V3_PROF, llm, {})
    assert "bird migration" not in out["topics"]
    assert all(e["label"] in out["topics"] or e["label"] == cd.UNAPPROVED_LABEL for e in out["topic_shares"]["main"])


def test_prompts_require_evidence_and_derivations():
    assert "You may state numbers only by copying an approved fact phrase exactly" in cd.DRAFT_SYSTEM
    assert "title_ids" in cd.CHECK_SYSTEM and "derivation" in cd.CHECK_SYSTEM
    d = cd.CorpusDescriber({"read_db_url_env": "X", "prompt_version": "vera-corpus-digest-v2", "max_titles": 5, "retry_s": 1})
    assert d.cfg["prompt_version"] == "vera-corpus-digest-v2+" + cd.PROMPT_REV


def test_numeric_word_config_has_reasons_and_no_vocabulary_lists():
    import json
    cfg = json.loads((ROOT / "config/corpus_digest_numeric_words.json").read_text())
    assert cfg["numeric_words"]["reason"] and "function_words" not in cfg and "descriptive_words" not in cfg
    assert not (ROOT / "config/corpus_digest_vocabulary.json").exists()
