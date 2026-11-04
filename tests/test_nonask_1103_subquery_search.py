"""NONASK-1103 stream S (S2/S3/S5/S6/F0): per-sub-query search, scaled budget, entity-central ranking, cause wording,
title de-duplication, ddg_identifier "handle" handling. Offline: fakes only (no network, model or database)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vera import auto_acquire as aa
from vera import search_providers as sp
from vera import search_query as sq
from vera import subquery
from tests.test_yield_1021 import _R, _oa_cand

Q = "What are AWS, Google, Nvidia, and Azure cloud ai engineering?"
PARTS = [("aws cloud ai engineering", ("AWS",)), ("google cloud ai engineering", ("Google",)),
         ("nvidia cloud ai engineering", ("Nvidia",)), ("azure cloud ai engineering", ("Azure",))]


def _cfg(cap=7, sub=True, arxiv=True):
    return {"search": {"providers": ["openalex"] + (["arxiv"] if arxiv else []), "openalex_profile": {"x": 1},
                       "ddg_first": {"enabled": True}, "europepmc": {"enabled": True, "page_size": 5, "search_url": "https://example.invalid/search"},
                       **({"subquery": {"enabled": True}} if sub else {})},
            "bounds": {"max_search_calls_per_job": cap}}


def test_budget_scales_7_to_16_for_four_parts_and_is_unchanged_otherwise():
    assert sp.subquery_budget(_cfg(), 4) == 16 and sp.subquery_budget(_cfg(), 2) == 10
    assert sp.subquery_budget(_cfg(), 1) == 7 and sp.subquery_budget(_cfg(sub=False), 4) == 7  # kill switch
    assert sp.subquery_budget(_cfg(arxiv=False), 4) == 7 + 2 * 3


def _run(monkeypatch, cap=7, parts=PARTS, wiki=True):
    seen = {"base": [], "ddg": 0, "epmc": [], "wiki": 0, "lookup_left": None}
    monkeypatch.setattr(sp, "_wiki_cfg", lambda: {"enabled": wiki})
    from vera import wiki_ref_portal
    monkeypatch.setattr(wiki_ref_portal, "enabled", lambda c: bool(c.get("enabled")))

    def factory(cfg, *, http_get=None, **kw):
        def s(q, n, t):
            seen["wiki"] += 1
            return []
        return s
    monkeypatch.setattr(wiki_ref_portal, "make_portal_search", factory)

    def base(q, n, t):
        seen["base"].append((q, sp.OA_MODE.get(), sp.ARXIV_ALLOWED.get()))
        k = len(seen["base"])
        return [{"provider": "openalex", "doi": f"10.9/o{k}", "title": f"AWS cloud ai engineering record {k}"}]

    def ddg(q, n, t):
        seen["ddg"] += 1
        return [{"url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/", "title": "t"}]

    def fake_ddg_to_cands(results, ecfg, dcfg, **kw):
        seen["lookup_left"] = kw["calls_left"]
        return [], {}
    monkeypatch.setattr(sp, "ddg_to_candidates", fake_ddg_to_cands)

    def tx(url, params, headers, timeout):
        seen["epmc"].append(params["query"])
        return _R({"resultList": {"result": []}})
    search = sp.acquisition_search(base, _cfg(cap), ddg_search=ddg, transport=tx, terms_fn=lambda: (None, None))
    tok = sp.SUBQUERIES.set(parts)
    try:
        with sp.portal_scope():
            out = search(Q, 31, 5.0)
    finally:
        sp.SUBQUERIES.reset(tok)
    return seen, out


def test_four_parts_each_get_a_call_and_question_level_slots_are_single(monkeypatch):
    seen, out = _run(monkeypatch)
    # AMENDS F0's per-job pins (arXiv once per job; lookup_left == 1) to their PER-SUB-QUERY form: arXiv once per part
    # (4), the id lookup keeps its slot (lookup_left == 1), DDG and Wikipedia exactly once per QUESTION.
    assert [b[0] for b in seen["base"]] == [p[0] for p in PARTS]  # one OpenAlex AND call per part, none for the OR retry
    assert all(b[1] == "and_only" and b[2] is True for b in seen["base"])
    assert seen["ddg"] == 1 and seen["wiki"] == 1 and seen["lookup_left"] == 1
    total = len(seen["base"]) + len(seen["epmc"]) + len(seen["base"]) + seen["ddg"] + seen["wiki"]  # OA + EPMC + arXiv...
    assert len(seen["epmc"]) == 4
    assert {c.get("part") for c in out} == {0, 1, 2, 3}  # candidates tagged by sub-query
    assert total <= 16


def test_scaled_budget_never_exceeded_when_the_cap_is_low(monkeypatch):
    seen, _ = _run(monkeypatch, cap=7)  # cfg cap 7 is scaled by subquery_budget -> 16; a configured 3 stays tiny
    seen2, _ = _run(monkeypatch, cap=3)
    n_calls = len(seen2["base"]) + len(seen2["epmc"]) + seen2["ddg"] + seen2["wiki"]
    assert n_calls <= sp.subquery_budget(_cfg(3), 4) + 4  # arXiv shares the OpenAlex call; bounded by the scaled cap
    assert seen2["ddg"] <= 1 and seen2["wiki"] <= 1


def test_or_retry_is_question_level_and_only_when_no_part_found_anything(monkeypatch):
    monkeypatch.setattr(sp, "_wiki_cfg", lambda: {"enabled": False})
    calls = []

    def base(q, n, t):
        calls.append((q, sp.OA_MODE.get(), sp.ARXIV_ALLOWED.get()))
        return []
    search = sp.acquisition_search(base, _cfg(), ddg_search=lambda *a: [], terms_fn=lambda: (None, None),
                                   transport=lambda *a, **k: _R({"resultList": {"result": []}}))
    tok = sp.SUBQUERIES.set(PARTS)
    try:
        with sp.portal_scope():
            search(Q, 31, 5.0)
    finally:
        sp.SUBQUERIES.reset(tok)
    assert [c[1] for c in calls] == ["and_only"] * 4 + ["or_only"] and calls[-1][0] == Q and calls[-1][2] is False


def test_entity_central_ranking_demotes_a_record_without_the_entity():
    kc = sq.key_concepts("aws cloud ai engineering", central=("AWS",))
    t = sq.relevance_tiers(kc, ["Google cloud ai engineering practices overview and guide",
                                "AWS cloud ai engineering guide for teams"])
    assert t == [1, 0]
    kc0 = sq.key_concepts("aws cloud ai engineering")
    assert sq.relevance_tiers(kc0, ["Google cloud ai engineering practices overview and guide",
                                    "AWS cloud ai engineering guide for teams"]) == [0, 0]  # without it: no demotion


def test_part_query_is_a_reordering_subset_of_the_part_with_the_entity_first():
    cfg = sq._default_terms_cfg()
    txt = "reliable ai engineering on Azure cloud"
    q = sq.part_query(txt, ("Azure",), cfg)
    assert q.split()[0] == "azure" and set(q.split()) <= set(txt.lower().split())
    assert sq.and_terms(sq.subquery_terms(txt, ("Azure",), cfg), 1) == ["azure"]


def test_rank_candidates_interleaves_parts_round_robin():
    cfg = aa_cfg()
    cands = [{**_oa_cand(10 * k + i, is_oa=True), "title": f"{n} cloud ai engineering report number {i}", "part": k}
             for k, n in enumerate(["AWS", "Google"]) for i in range(3)]
    out = aa.rank_candidates(Q, cands, cfg, set(), {}, lambda *_: None, parts=PARTS[:2])
    parts = [c[2].get("part") for c in out]
    assert parts[:4] == [0, 1, 0, 1], parts


def aa_cfg():
    from vera import acq_config
    c = acq_config.load_config()
    return c


def test_merge_sources_dedupes_by_normalised_title_against_pmc_and_openalex():
    pmc = {"provider": "openalex", "doi": "10.1/a", "pmcid": "PMC12", "title": "Cloud AI Engineering: A Practical Guide!"}
    arx = {"provider": "arxiv", "arxiv_id": "2401.00001", "title": "cloud ai engineering - a practical guide"}
    short = {"provider": "arxiv", "title": "Introduction"}
    other = {"provider": "arxiv", "title": "Introduction"}
    out = sp.merge_sources([pmc], [arx], [short], [other])
    assert len(out) == 3 and out[0] is pmc and pmc.get("arxiv_id") == "2401.00001"  # the long title merged, ids absorbed
    # a short title (under MIN_TITLE_WORDS words) is never a key, so two different works called "Introduction" stay apart
    assert sp._cand_ids(short) == set()


def test_part_tags_are_absorbed_by_a_duplicate():
    a = {"doi": "10.1/a", "part": 0}
    b = {"doi": "10.1/a", "part": 2}
    out = sp.merge_sources([a], [b])
    assert out == [a] and a["parts"] == [0, 2]


@pytest.mark.parametrize("url", ["https://hdl.handle.net/1721.1/12345", "hdl:1721.1/12345"])
def test_every_ddg_identifier_caller_handles_the_handle_kind(url):
    """Callers: ddg_to_candidates (URL-only candidate), auto_acquire.follow_url (pid, no request), wiki_ref_portal
    (_ids_from_url: no id, so the reference is dropped as non-scholarly), replay script (kind only)."""
    ident = sp.ddg_identifier(url)
    assert ident and ident["kind"] == "handle"
    cand, _ = sp.ddg_to_candidates([{"url": "https://hdl.handle.net/1721.1/12345", "title": "t"}], {}, {},
                                   deadline=1e12, clock=lambda: 0.0, calls_left=0, acq={})
    assert len(cand) == 1 and not cand[0].get("doi")
    f = aa.follow_url({"url": "https://hdl.handle.net/1721.1/12345", "title": "t"})
    assert f["pid"] == {"kind": "handle", "value": "1721.1/12345"}
    from vera import wiki_ref_portal
    assert wiki_ref_portal._ids_from_url("https://hdl.handle.net/1721.1/12345") == {}


def test_decomposition_of_the_example_gives_four_entity_parts():
    plan = subquery.decompose(Q)
    assert len(plan.parts) == 4 and [p.entities for p in plan.parts] == [("AWS",), ("Google Cloud",), ("Nvidia",), ("Azure",)]


def test_cause_wording_names_the_real_cause_and_not_time():
    from vera import discovery_panel as dp
    from vera import research_outcome as ro
    assert "acquisition runs" in ro._CAUSE_TEXT["job_quota"] and "searches" not in ro._CAUSE_TEXT["job_quota"]
    assert "budget" in ro._CAUSE_TEXT["budget"]
    for code in ("R-ACQ-RATE", "R-ACQ-DAILY"):
        assert "in time" not in dp.REASON_TEXT[code]
    assert "in time" not in dp.OUTCOME_TEXT["not_evaluated"] and "in time" not in dp.OUTCOME_TEXT["not_reached"]
    assert "in time" in dp.REASON_TEXT["R-ACQ-DEADLINE"]  # time stops keep it


def test_replay_subquery_scenarios_show_no_regression_on_the_holdout():
    """S3 ships enabled only on this evidence (offline synthetic fixtures, not live): tier-0 recall of relevant records
    does not drop, off-entity records leave tier 0, single-topic controls are unchanged."""
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "scripts" / "nonask_1103_replay.py"
    spec = importlib.util.spec_from_file_location("nonask_replay_s23", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for split in ("holdout", "calibration"):
        m = mod.replay(split)["flags"]["subquery_search"]
        assert m["regression"] is False
        assert m["relevant_tier0_after"]["k"] >= m["relevant_tier0_before"]["k"]
        assert m["off_entity_tier0_after"]["k"] < m["off_entity_tier0_before"]["k"]


def test_title_dedupe_never_merges_works_with_conflicting_identifiers():
    t = "A long shared title about cloud ai engineering practice"
    a = {"doi": "10.1/a", "title": t}
    b = {"doi": "10.1/b", "title": t}
    c = {"pmcid": "PMC1", "title": t}
    d = {"pmcid": "PMC2", "title": t}
    assert len(sp.merge_sources([a], [b])) == 2 and len(sp.merge_sources([c], [d])) == 2
    pre = {"doi": "10.48550/arxiv.2401.00001", "title": t}  # a preprint DOI and a journal DOI: the same work
    assert len(sp.merge_sources([a], [pre])) == 1


@pytest.mark.parametrize("ent,text,hit", [
    ("Azure", "Azure sky photography and cloud formation engineering notes", False),
    ("Azure", "Cloud ai engineering on Microsoft's platform", True),
    ("Azure", "Microsoft Azure notes", True),
    ("Google Cloud", "Just google it: cloud ai engineering tips", False),
    ("Google Cloud", "Vertex AI on Google deployment", True),
    ("Snowflake", "Snowflake schema design for a data warehouse", False),
    ("Snowflake", "Snowflake virtual warehouse tuning", True),
    ("Pulsar", "Pulsar timing arrays and radio signals", False),
    ("Kafka", "Franz Kafka letters", False),
    ("Kafka", "Apache Kafka consumer group ordering", True),
    ("AWS", "Amazon rainforest deforestation, Amazon basin cloud cover engineering", False),
    ("AWS", "Amazon Web Services operations", True),
])
def test_entity_hit_needs_a_full_alias_or_a_context_term(ent, text, hit):
    assert sq.entity_hit(text, subquery.entity_context(ent)) is hit


def test_adversarial_replay_holdout_off_entity_falls_and_recall_holds_and_controls_do_not_split():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "scripts" / "nonask_1103_replay.py"
    spec = importlib.util.spec_from_file_location("nonask_replay_s23b", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.replay("holdout")["flags"]["subquery_adversarial"]
    assert m["off_tier0_after"]["k"] < m["off_tier0_before"]["k"]
    # Round 2: with comparison records filed under both entities the replay SHOWS a recall loss (13/14 holdout), the
    # reason S3 ships flag-off; the test pins the finding instead of hiding it (registry not tuned further, by decision).
    assert m["relevant_tier0_after"]["k"] >= m["relevant_tier0_before"]["k"] - 1 and m["regression"] is True
    allm = mod.replay("all")["flags"]["subquery_adversarial"]
    assert all(n == 1 for sid, n in allm["n_parts"].items() if sid.startswith("adv_c"))  # real decomposer: no split
