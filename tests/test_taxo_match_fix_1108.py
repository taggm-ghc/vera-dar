"""TAXO-MATCH-FIX-1108: chunk-to-subject matching is strict and the Corpus page says exactly how many documents did not match.

Defect (Skeptical QA work/TAXO-MATCH-REVIEW-1108, "MIXED, mostly ARTEFACT"): the version-1 chunk rule matched almost any scholarly
text (one generic word such as "data" reached the BM25 acceptance value; one chunk of 16 assigned the document), so the page showed
"Not matched to a listed subject area: under 1%". Fixtures are synthetic paragraphs (no external source text); the five OFF_TOPIC
scholarly paragraphs and the recipe are the review probe's texts (scratch/TAXO-MATCH-REVIEW-1108/probe.py)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vera import chunk_topics, corpus_description as cd, sources_sidebar as sb

ROOT = Path(__file__).resolve().parent.parent
SEED = json.loads((ROOT / "config" / "micro_topic_anchors.json").read_text())["entries"]  # same 5 domains as live
T = 2.0

OFF_TOPIC = {  # the review probe's texts, verbatim
    "astrophysics": "We present new observations of the galaxy cluster using the telescope array. The sample of 120 spectra was reduced with standard pipelines and the redshift distribution shows a clear trend with mass. Uncertainty in the photometric calibration is the dominant error term; we estimate it with a bootstrap over the survey fields. Results indicate the dark matter halo profile is consistent with simulations, although the small sample size limits the accuracy of the fit. Future data from the next survey will improve the number of resolved sources.",
    "constitutional_law": "The court reviewed whether the statute violated the separation of powers. Counsel argued that the legislative record showed no evidence of the intended purpose, and that the trial judge applied the wrong standard of review. The majority opinion held that the provision was valid, noting the long history of similar statutes and the lack of any substantive change in practice. The dissent warned about the impacts on individual rights and the risk that future courts would use the decision to expand executive authority without adequate oversight.",
    "medieval_history": "Records from the monastery show that grain yields fell sharply after the flood of 1315. The abbot's accounts list the number of tenants, the rents collected and the deaths recorded in each parish. Historians have used these sources to estimate population change across the region, although the survival of documents is uneven and introduces bias. The trend in prices suggests a prolonged crisis, and the management of the estate shifted toward pasture as labour became scarce.",
    "economics": "We estimate the effect of the minimum wage increase on employment using a difference-in-differences design. The sample covers 2,400 firms observed over ten years. Results show a small negative association between the wage change and hours worked, with confidence intervals that include zero. Robustness checks using alternative control groups and adjusted standard errors give similar estimates. The findings are relevant to policy debates about labour market interventions and their distributional outcomes.",
    "materials_chemistry": "Thin films were deposited by atomic layer deposition at 200 C and characterized by X-ray diffraction and electron microscopy. The growth rate per cycle was stable across the substrate and the film quality improved after annealing. We report the band gap, the carrier mobility and the density of defects. Sample-to-sample variability was small, and the measurements were repeated three times to estimate uncertainty. These results suggest a route to scalable fabrication of transparent conductors.",
    "recipe": "Preheat the oven to 180 degrees. Mix the flour, sugar and butter until the dough is smooth, then rest it for thirty minutes. Roll it out, cut the shapes and bake for twelve minutes until golden. Let them cool on a rack before adding the icing.",
}

OFF_TOPIC_HELDOUT = {  # written before the generic list was checked against them; not used to choose any list word
    "linguistics": "We analyse vowel duration in spontaneous speech from 40 speakers of two dialects. Recordings were segmented by hand and the acoustic measures were compared with mixed-effects regression. Speakers from the northern dialect produced longer vowels before voiced consonants, and the difference was larger in stressed syllables. We discuss the implications for theories of phonological contrast and sound change.",
    "finance": "This paper examines how bond spreads respond to central bank announcements. Using daily prices for sovereign debt over fifteen years, we find that spreads narrow after unexpected easing and widen after tightening, with larger responses for issuers with high leverage. The results are robust to alternative event windows and to controlling for global risk appetite.",
    "neuroscience": "Mice were trained on a visual discrimination task while we recorded from neurons in the primary visual cortex with two-photon imaging. Responses to the rewarded stimulus increased over training sessions, whereas responses to the unrewarded stimulus were unchanged. Silencing feedback from higher areas abolished the learning-related enhancement, suggesting a top-down mechanism.",
    "sociology": "Drawing on interviews with 60 tenants in three cities, we examine how eviction shapes family routines and school attendance. Respondents described frequent moves, loss of belongings and strained ties with neighbours. Children in evicted households missed more school days, and parents reported that informal support networks weakened after each move.",
}

ON_TOPIC = {  # one synthetic abstract-style paragraph per listed subject area (natural prose, not keyword lists)
    "epi_outbreak": "During the 2019 measles outbreak, surveillance teams recorded weekly incidence in each district and traced contacts of confirmed cases. Vaccination coverage below 90% was associated with higher transmission, and the estimated reproduction number fell below one after a catch-up vaccine campaign. Seroprevalence surveys suggested that many infections were mild and unreported.",
    "epi_cohort": "In a prospective cohort of 12,000 adults followed for ten years, we examined whether smoking and diet predicted cardiovascular disease and diabetes. Hazard ratios were adjusted for physical activity, obesity and other confounding exposures. Higher incidence of chronic disease among smokers persisted after follow-up, and mortality was highest among participants with several risk factors.",
    "eco_biodiversity": "We surveyed bird and butterfly species richness along transects in forest, wetland and grassland habitats inside and outside protected areas. Abundance of endangered species was higher in protected sites, while invasive species dominated disturbed edges. Diversity index values declined with distance from intact habitat, which supports conservation of habitat corridors for wildlife.",
    "eco_function": "We measured nutrient cycling in a freshwater watershed after wildfire. Soil microbial decomposition of plant litter slowed in burned plots, nitrogen and phosphorus exports to the river rose during the first wet season, and primary production of riparian vegetation recovered within three years. Restoration plantings increased carbon storage and stabilised biogeochemical fluxes.",
    "clim_warming": "Global mean surface temperature has risen by about 1.1 degrees since pre-industrial times, driven mainly by greenhouse gas emissions and carbon dioxide. Attribution studies link the increased frequency of heatwaves and drought to this warming, while sea level rise reflects ocean heat uptake and melting ice sheets and glaciers. Emissions pathways consistent with the remaining carbon budget require rapid mitigation.",
    "clim_modelling": "We evaluate an ensemble of CMIP6 general circulation models against reanalysis over the monsoon region. Statistical downscaling with bias correction improved the representation of precipitation extremes, but cloud feedback and aerosol parameterization remained the largest sources of spread in the simulation. Internal variability linked to ENSO limited forecast skill beyond one season in the regional climate model.",
    "repro_research": "Replication attempts of 100 published psychology experiments succeeded in fewer than half of cases. Low statistical power, p-hacking and publication bias are plausible explanations. Preregistration, registered reports, open data and code availability are proposed remedies, and journals increasingly adopt transparency guidelines and reporting standards to strengthen research integrity and reproducibility.",
    "repro_software": "Research software engineering teams increasingly adopt version control with git, continuous integration and unit tests to keep scientific software maintainable. Our survey of scientists found that documentation and code review are rare in small labs, and that technical debt accumulates when developers lack training in testing and dependency management. Software citation could reward sustainability work.",
    "aicode_assistants": "We ran a controlled experiment with 80 students who completed programming exercises with and without an AI coding assistant based on a large language model. Copilot users finished tasks faster and code completion suggestions reduced syntax bugs, but generated programs more often contained security vulnerabilities. Novices relied on the autocomplete more heavily than experienced developers.",
    "aicode_agents": "We evaluate LLM agents on repository-level issue resolution from SWE-bench. Agents that combine test generation with automated program repair resolved more issues, yet generated patches often passed tests while introducing regressions. Hallucination of nonexistent APIs and limited debugging autonomy remain the main limitations; fine-tuning on refactoring tasks helped maintainability.",
}

EXPECTED_DOMAIN = {"epi": "epidemiology", "eco": "ecology", "clim": "climate science",
                   "repro": "reproducibility and software practice", "aicode": "AI coding assistants"}


@pytest.fixture(scope="module")
def strict():
    return chunk_topics.ChunkIndex(SEED, chunk_topics.load_rules())


def _domain(idx, key):
    return idx.by_id[key]["field"] if key else None


@pytest.mark.parametrize("name", sorted(OFF_TOPIC))
def test_review_off_topic_paragraphs_are_unmatched(strict, name):
    assert chunk_topics.assign_chunk(strict, OFF_TOPIC[name], T) is None


def test_review_off_topic_paragraphs_matched_under_the_old_rule():  # the defect the fix removes (regression evidence)
    old = chunk_topics.ChunkIndex(SEED)
    assert sum(chunk_topics.assign_chunk(old, OFF_TOPIC[n], T) is not None for n in OFF_TOPIC if n != "recipe") == 5


@pytest.mark.parametrize("name", sorted(OFF_TOPIC_HELDOUT))
def test_held_out_off_topic_paragraphs_are_unmatched(strict, name):
    assert chunk_topics.assign_chunk(strict, OFF_TOPIC_HELDOUT[name], T) is None


@pytest.mark.parametrize("name", sorted(ON_TOPIC))
def test_on_topic_fixtures_still_match_their_subject_area(strict, name):
    r = chunk_topics.assign_chunk(strict, ON_TOPIC[name], T)
    assert r is not None and _domain(strict, r[0]) == EXPECTED_DOMAIN[name.split("_")[0]]


def test_recipe_document_with_one_economics_chunk_is_unmatched(strict):
    doc = [OFF_TOPIC["recipe"]] * 15 + [OFF_TOPIC["economics"]]
    assert chunk_topics.document_primary(strict, doc, T, 1) is None
    doc_on = [OFF_TOPIC["recipe"]] * 15 + [ON_TOPIC["epi_outbreak"]]  # one genuine chunk of 16 is below the vote share too
    assert chunk_topics.document_primary(strict, doc_on, T, 1) is None
    genuine = [ON_TOPIC["epi_outbreak"]] * 4 + [OFF_TOPIC["recipe"]] * 12  # 4 of 16 = 25% of sampled chunks
    assert _domain(strict, chunk_topics.document_primary(strict, genuine, T, 1)) == "epidemiology"


def test_required_votes_and_share():
    assert chunk_topics.required_votes(16, 1, 0.25) == 4
    assert chunk_topics.required_votes(1, 1, 0.25) == 1
    assert chunk_topics.required_votes(5, 1, 0.25) == 2
    assert chunk_topics.required_votes(16, 6, 0.25) == 6
    assert chunk_topics.document_vote([("a", 3.0)] + [None] * 7, 1, 0.25) is None
    assert chunk_topics.document_vote([("a", 3.0), ("a", 3.0)] + [None] * 6, 1, 0.25) == "a"


def test_generic_terms_never_count_and_single_term_match_rejected(strict):
    d = chunk_topics.match_detail(strict, "data results sample uncertainty study analysis")
    assert d["shared"] == [] and d["score"] == 0.0
    assert chunk_topics.assign_chunk(strict, "data results sample uncertainty study analysis", T) is None
    assert chunk_topics.assign_chunk(strict, "the vaccine trial results", T) is None  # one content term only


def test_rules_are_data_and_fail_closed(tmp_path):
    raw = json.loads((ROOT / "config" / "chunk_topic_terms.json").read_text())
    assert "PROVISIONAL" in raw["_calibration_note"] and "run_induction.sh --no-llm" in raw["_calibration_note"]
    r = chunk_topics.load_rules()
    assert {"data", "uncertainty", "results", "the"} <= r["generic"] and r["min_distinct_terms"] == 2
    for bad in ({**raw, "min_lead_ratio": 0.5}, {**raw, "stopwords": []}, {**raw, "min_distinct_terms": True}):
        p = tmp_path / "r.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            chunk_topics.load_rules(p)


def test_classifier_version_bumped_for_phase_2_provenance():
    assert chunk_topics.CLASSIFIER_VERSION == "2"
    assert chunk_topics.provenance("db:v3", 2.0)["classifier_version"] == "2:db:v3:t2"


# --- Corpus Description page wording --------------------------------------------------------------------------------------
def _sa(n, unmatched, domains=None):
    return {"basis_documents": n, "domains": domains if domains is not None else [{"label": "ecology", "share": round((n - unmatched) / n, 4),
            "topics": [], "more_topics": 0}], "other_domains": {}, "unmatched_share": round(unmatched / n, 4),
            "unmatched_count": unmatched}


def test_remainder_is_one_last_other_mixed_line_never_a_not_matched_headline():
    for u, want in ((1, "under 1"), (120, 33), (0, None)):  # CORPUS-PCT-B-1108: replaces the exact not-matched count line
        lines = sb._subject_lines(_sa(365, u))
        assert not any("matched" in ln.lower() for ln in lines[1:])
        assert (lines[-1] == f"Other / mixed: {want}%") if want else lines[-1].startswith("Ecology")


def test_subject_lines_say_automatic_approximate_nearest_subject():
    head = sb._subject_lines(_sa(10, 2))[0]
    assert "automatic and approximate" in head.lower() and "nearest" in head.lower()
    assert "incomplete and still growing" in head  # OD-1108-HITL item 8: the listed areas are not the complete set


def test_subject_areas_reports_unmatched_count():
    entries = [{"id": "a", "label": "A", "field": "ecology", "parent_topic_key": None}]
    sa = cd.subject_areas(entries, {1: "a", 2: None, 3: None})
    assert sa["unmatched_count"] == 2 and sa["basis_documents"] == 3


# --- 020 provenance (migration 020 applied 2026-11-08; the backfill writes these into tax_chunk_subtopic / tax_chunk_topic) ----
def test_provenance_passes_020_link_row_checks(strict):
    rules = chunk_topics.load_rules()
    prov = chunk_topics.provenance("active10", 2.0, rules)
    assert prov["assigned_via"] == "backfill" and prov["classifier_name"] == "chunk_bm25" and prov["score_kind"] == "bm25"
    assert prov["classifier_version"].startswith("2:active10:t2:r") and len(prov["classifier_version"]) <= 128
    key, score = chunk_topics.assign_chunk(strict, ON_TOPIC["epi_outbreak"], T)
    assert chunk_topics.link_row_problems(prov, score) == []
    long = chunk_topics.provenance("x" * 300 + "é\n", 2.0, rules)  # long or non-ASCII registry text stays within cver_chk
    assert len(long["classifier_version"]) <= 128 and long["classifier_version"].endswith(":t2:r" + chunk_topics.rules_signature(rules))
    assert chunk_topics.link_row_problems(long, score) == []
    changed = {**rules, "min_lead_ratio": 2.0}  # a changed rule is a new classifier_version
    assert chunk_topics.provenance("active10", 2.0, changed)["classifier_version"] != prov["classifier_version"]


def test_link_row_problems_mirror_020_checks():
    ok = chunk_topics.provenance("v", 2.0)
    assert chunk_topics.link_row_problems({**ok, "classifier_name": "Chunk BM25"}, 1.0) == ["cname_chk"]
    assert chunk_topics.link_row_problems({**ok, "classifier_version": None}, 1.0) == ["prov_chk", "cpair_chk"]
    assert chunk_topics.link_row_problems(ok, None) == ["spair_chk"]
    assert chunk_topics.link_row_problems(ok, float("inf")) == ["score_chk"]
    assert chunk_topics.link_row_problems({**ok, "assigned_via": "agent"}, 1.0) == ["via_chk"]


# --- Corpus page document assignment ----------------------------------------------------------------------------------------
def test_document_with_read_chunks_gets_no_loose_document_level_second_try(monkeypatch):
    from vera import micro_topic
    reg = micro_topic.Registry({"entries": SEED})
    rows = [(1, "Measles outbreak vaccination", "outbreak vaccination incidence transmission surveillance", None),
            (2, "Measles outbreak vaccination", "outbreak vaccination incidence transmission surveillance", None),
            (3, "Off-topic", "", None)]
    chunks = {1: [OFF_TOPIC["recipe"]] * 4, 3: [ON_TOPIC["clim_warming"]] * 2}

    class Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, q, p=None):
            class R:
                def all(_s): return rows
            return R()

    class Eng:
        def connect(self): return Conn()
    monkeypatch.setattr(cd, "document_chunks", lambda c, ids, s: chunks)
    out = cd.document_primary_topics(Eng(), reg, {"count_min_score": {"anchor_bm25": {"scholarly": 2.0}}}, 10, 5000,
                                     {"chunk_votes": True})
    assert out[1] is None  # chunks read, none passed: no title + window retry
    assert reg.by_id[out[2]]["field"] == "epidemiology"  # no chunks: strict title + window match
    assert reg.by_id[out[3]]["field"] == "climate science"


def test_induction_uses_the_same_strict_rules():
    from vera import taxonomy_induction as ti
    prim, _ = ti.assign_corpus(SEED, {"d": [OFF_TOPIC["economics"]], "e": [ON_TOPIC["repro_research"]]}, T, 1)
    assert prim["d"] is None and SEED and prim["e"] is not None
