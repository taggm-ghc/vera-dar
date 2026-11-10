"""DDG-REDUCE-V3-1110: review fixes F1 (C4 only when new_documents is the sole trigger), F2 (C6 counts only candidates that
pass the existing pre-filters; the near-deadline skip is a named config value) and gaps G1-G4. Offline: fake transports,
the REAL DuckDuckGoLiteProvider, the ROUTE-R2 world harness. G1 and G4 run with the SHIPPED config/ddg_reduce.json (the
module's fixtures never patch ddg_reduce.load unless a test says so)."""
import json
import time
from types import SimpleNamespace

import pytest
import requests

import vera.search_providers as sp
from tests.test_ddg_meter_1110 import (  # noqa: F401  (rig is a fixture)
    CFG, GROUNDING, Q_NEWS, Q_PLAIN, _freshness_attempt, posts_by_site, reading_list, rig, set_flags)
from vera import ddg_meter, ddg_reduce, grounded_ask, inline_acquire
from vera import research_record as rr

C4 = dict(answered_question_no_ddg=True, ddg_fallback_only=True, ddg_fallback_min_results=3, one_group_per_call=True)


# ------------------------------------------------------------------------------------------------------ F1 (per trigger)
TRIGGER_SETS = [
    (["new_documents"], True),
    (["insufficient"], False),
    (["stale"], False),
    (["freshness_unknown"], False),
    (["insufficient", "new_documents"], False),
    (["stale", "new_documents"], False),
    (["freshness_unknown", "new_documents"], False),
    (["insufficient", "stale", "freshness_unknown", "new_documents"], False),
    ([], False),
    (None, False),          # unknown triggers: never skip
]


@pytest.mark.parametrize("triggers,skips", TRIGGER_SETS)
def test_f1_c4_skips_duckduckgo_only_when_new_documents_is_the_sole_trigger(rig, monkeypatch, triggers, skips):
    """The DDG slot of the OD-1109-SIX search is disabled (NO_DDG) only for a sufficient, fresh question."""
    from vera import ask_service
    seen = []
    set_flags(monkeypatch, **C4)
    monkeypatch.setattr(ask_service, "_layer1_clean", lambda q: True)
    monkeypatch.setattr(ask_service, "inline_need_s", lambda acq, g: 1.0)
    monkeypatch.setattr(ask_service, "_doc_keys", lambda x: [])
    from vera import acq_discovered, fulldoc_corpus
    monkeypatch.setattr(fulldoc_corpus, "flag_enabled", lambda c: True)
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda: {})
    monkeypatch.setattr(inline_acquire, "icfg_of", lambda acq: {"in_scope": True})
    monkeypatch.setattr(inline_acquire, "inline_enabled", lambda acq: False)
    monkeypatch.setattr(inline_acquire, "news_question", lambda q: False)
    monkeypatch.setattr(inline_acquire, "acquire", lambda q, acq, **kw: (seen.append(ddg_reduce.NO_DDG.get()) or
                                                                         SimpleNamespace(status="ok", public=lambda: {})))
    kw = {} if triggers is None else {"triggers": triggers}
    ask_service._research_attempt(Q_PLAIN, None, {"x": 1}, {}, {}, SimpleNamespace(inline_used=False,
                                                                                    deadline=time.monotonic() + 100), **kw)
    assert seen == [skips]


@pytest.mark.parametrize("triggers,skips", TRIGGER_SETS[:8])
def test_f1_fallback_can_fire_unless_the_sole_trigger_is_new_documents(rig, monkeypatch, triggers, skips):
    """Through the REAL provider: scholarly routes return nothing; the DDG fallback POSTs (1) for every trigger set except
    the sole new_documents, which spends none."""
    set_flags(monkeypatch, **C4)
    monkeypatch.setattr(inline_acquire, "news_question", lambda q: False)
    _freshness_attempt(rig, monkeypatch, Q_PLAIN, triggers=triggers)
    assert len(rig.posts) == (0 if skips else 1)
    if not skips:
        assert posts_by_site() == {"inline": 1} or sum(posts_by_site().values()) == 1


def test_f1_news_question_keeps_duckduckgo_even_for_the_sole_new_documents_trigger(rig, monkeypatch):
    set_flags(monkeypatch, **C4)
    _freshness_attempt(rig, monkeypatch, Q_NEWS, triggers=["new_documents"])
    assert len(rig.posts) >= 1


def test_f1_real_trigger_list_of_an_insufficient_answer(rig):
    """rr.triggers(False, ...) is ["insufficient"]: the shape _freshness_research hands to _research_attempt."""
    assert rr.triggers(False, {"counts": {}}) == [rr.TRIGGER_INSUFFICIENT]
    assert rr.triggers(False, {"counts": {"stale": 1}}) == [rr.TRIGGER_INSUFFICIENT, rr.TRIGGER_STALE]


# ------------------------------------------------------------------------------------------------------ F2 (relevant only)
REL = "Developer productivity gains from AI coding assistants: a randomized field study"
OFF1 = "Pedestrian flow in crowded urban transit stations during rush hours"
OFF2 = "Soil carbon sequestration under long-term tillage regimes in semiarid grasslands"


def _acq(rig, titles, timeout=5.0, **flags):
    ddg_fn = lambda q, n, t: rig.ddg.parse(rig.ddg.call(q, n, t), n)  # noqa: E731
    cfg = {"search": {"ddg_first": {"enabled": True}, "europepmc": {"enabled": False}}, "bounds": {"max_search_calls_per_job": 6}}
    base = lambda q, n, t: [{"provider": "openalex", "doi": f"10.9/{i}", "url": f"https://doi.org/10.9/{i}",  # noqa: E731
                             "title": ti, "snippet": ""} for i, ti in enumerate(titles)]
    with ddg_meter.site("inline"), sp.portal_scope() as recs:
        sp.acquisition_search(base, cfg, ddg_search=ddg_fn)(Q_PLAIN, 5, timeout)
    return recs


def test_f2_the_existing_pre_filter_flags_the_off_topic_titles():
    from vera import search_query
    kc = search_query.key_concepts(Q_PLAIN)
    assert search_query.relevance_tiers(kc, [REL, OFF1, OFF2]) == [0, 2, 2]


def test_f2_loosely_matched_hits_do_not_suppress_the_fallback(rig, monkeypatch):
    set_flags(monkeypatch, **C4, ddg_fallback_relevant_only=True)
    _acq(rig, [REL, OFF1, OFF2])            # 3 raw candidates, 1 passes the pre-filter
    assert len(rig.posts) == 1


def test_f2_three_relevant_candidates_still_suppress_it(rig, monkeypatch):
    set_flags(monkeypatch, **C4, ddg_fallback_relevant_only=True)
    _acq(rig, [REL, REL + " (replication)", REL + " (2)"])
    assert rig.posts == []


def test_f2_switch_off_counts_raw_candidates(rig, monkeypatch):
    set_flags(monkeypatch, **C4, ddg_fallback_relevant_only=False)
    _acq(rig, [REL, OFF1, OFF2])
    assert rig.posts == []


def test_f2_the_same_rule_in_the_reading_list_fan_out(rig, monkeypatch):
    import vera.search_and_fetch as sf
    set_flags(monkeypatch, **C4, ddg_fallback_relevant_only=True)
    titles = iter([REL, OFF1, OFF2])
    scholar = rig.chain[1]
    scholar.parse = lambda resp, n: [{"url": f"https://doi.org/10.1/{i}", "title": t, "snippet": "x", "source_type": "journal_article"}
                                     for i, t in enumerate([REL, OFF1, OFF2])]
    sf.search_question(Q_PLAIN, num_results=5, providers=rig.chain, min_candidates=1, mode="fan_out",
                       deadline_s=30, timeout_s=5, max_retries=1)
    assert len(rig.posts) == 1


def test_f2_usable_count_never_raises_and_counts_all_on_error(monkeypatch):
    set_flags(monkeypatch, ddg_fallback_relevant_only=True)
    import vera.search_query as sq
    monkeypatch.setattr(sq, "key_concepts", lambda *a, **k: 1 / 0)
    assert ddg_reduce.usable_count(Q_PLAIN, [{"title": REL}, {"title": OFF1}]) == 2


# ------------------------------------------------------------------------------------------------------ G2 near deadline
def test_f2_deadline_skip_is_a_named_config_value():
    cfgd = json.loads(ddg_reduce.CONFIG_PATH.read_text(encoding="utf-8"))
    assert cfgd["ddg_fallback_min_remaining_s"] == 1.0 and "_ddg_fallback_min_remaining_s_note" in cfgd
    assert ddg_reduce.fallback_min_remaining_s() == 1.0


@pytest.mark.parametrize("bad", [None, "x", -1, True])
def test_f2_bad_deadline_value_falls_back_to_one_second(monkeypatch, bad):
    set_flags(monkeypatch, ddg_fallback_min_remaining_s=bad)
    assert ddg_reduce.fallback_min_remaining_s() == 1.0


def test_g2_c6_skips_the_fallback_with_less_than_the_named_time_left(rig, monkeypatch):
    set_flags(monkeypatch, **C4, ddg_fallback_min_remaining_s=1.0)
    recs = _acq(rig, [], timeout=0.5)                  # 0 candidates, only 0.5 s of deadline: DuckDuckGo is not started
    assert rig.posts == [] and "ddg" not in [r["portal"] for r in recs if r["status"] == "completed"]


def test_g2_c6_runs_when_the_named_value_is_lowered(rig, monkeypatch):
    set_flags(monkeypatch, **C4, ddg_fallback_min_remaining_s=0.1)
    _acq(rig, [], timeout=0.5)
    assert len(rig.posts) == 1


def test_g2_c6_runs_with_enough_time_and_no_candidates(rig, monkeypatch):
    set_flags(monkeypatch, **C4)
    _acq(rig, [], timeout=5.0)
    assert len(rig.posts) == 1


# ------------------------------------------------------------------------------------------------------ G3 failures on the fallback path
class _R:
    def __init__(self, text, status=200):
        self.text, self.status_code, self.headers = text, status, {}


@pytest.mark.parametrize("variant,expect", [("timeout", "timeout"), ("quota", "own_cap"), ("challenge", "failed"),
                                            ("read_timeout", "timeout")])
def test_g3_ddg_failure_on_the_acquisition_fallback_path(rig, monkeypatch, variant, expect):
    set_flags(monkeypatch, **C4)

    def ddg_fn(q, n, t):
        if variant == "timeout":
            raise TimeoutError("ddg read timed out")
        if variant == "read_timeout":
            raise requests.exceptions.ReadTimeout("Read timed out.")
        if variant == "quota":
            raise sp.ProviderUnavailable("duckduckgo_lite daily query cap (quota) of 8 spent")
        raise sp.ProviderResponseError("duckduckgo challenge page; not evaded")
    cfg = {"search": {"ddg_first": {"enabled": True}, "europepmc": {"enabled": False}}, "bounds": {"max_search_calls_per_job": 6}}
    with sp.portal_scope() as recs:
        sp.acquisition_search(lambda q, n, t: [], cfg, ddg_search=ddg_fn)(Q_PLAIN, 5, 5.0)
    st = [r["status"] for r in recs if r["portal"] == "ddg"]
    assert len(st) == 1 and st[0] == expect, recs
    assert "completed" not in st


def test_g3_challenge_page_on_the_fan_out_fallback_is_recorded_and_not_retried(rig, monkeypatch):
    import vera.search_and_fetch as sf
    set_flags(monkeypatch, **C4)
    rig.scholar_n = 0
    rig.ddg._transport = lambda url, data, headers, timeout: (rig.posts.append(data["q"]) or
                                                              _R('<html><div class="anomaly-modal">robot?</div></html>'))
    res = sf.search_question(Q_PLAIN, num_results=5, providers=rig.chain, min_candidates=1, mode="fan_out",
                             deadline_s=30, timeout_s=5, max_retries=1)
    ports = {p["portal"]: p["status"] for p in res.meta["portals"]}
    assert ports.get("duckduckgo_lite") not in (None, "completed") or "duckduckgo_lite" in res.meta["provider_failures"]
    assert len(rig.posts) == 1                              # a challenge is never retried or evaded


def test_g3_ddg_timeout_on_the_fan_out_fallback_leaves_the_scholarly_result(rig, monkeypatch):
    import vera.search_and_fetch as sf
    set_flags(monkeypatch, **C4)
    rig.scholar_n = 2

    def boom(url, data, headers, timeout):
        rig.posts.append(data["q"])
        raise requests.exceptions.ReadTimeout("Read timed out.")
    rig.ddg._transport = boom
    res = sf.search_question(Q_PLAIN, num_results=5, providers=rig.chain, min_candidates=1, mode="fan_out",
                             deadline_s=30, timeout_s=5, max_retries=1)
    assert len(list(res)) == 2 and res.meta["route_used"] == "openalex"
    assert "duckduckgo_lite" in res.meta["provider_failures"]


# ------------------------------------------------------------------------------------------------------ G4 the real config, all on
def test_g4_the_shipped_config_has_every_switch_on():
    cfgd = ddg_reduce.load()
    for k in ("domain_branches_scholarly_only", "variant_rounds_no_ddg", "one_group_per_call", "answered_question_no_ddg",
              "longer_normalized_cache", "ddg_fallback_only", "ddg_fallback_relevant_only"):
        assert cfgd.get(k) is True, k
    assert ddg_reduce.fallback_min() == 3 and ddg_reduce.fallback_min_remaining_s() == 1.0


def test_g4_real_config_reading_list_enough_candidates_no_post_too_few_one_post(rig):
    ddg_meter.reset()
    rig.scholar_n = 3
    reading_list(rig, Q_PLAIN)
    assert rig.posts == []
    rig.scholar_n = 0
    reading_list(rig, Q_PLAIN)
    assert len(rig.posts) == 1


def test_g4_real_config_answered_question_by_trigger(rig, monkeypatch):
    monkeypatch.setattr(inline_acquire, "news_question", lambda q: False)
    _freshness_attempt(rig, monkeypatch, Q_PLAIN, triggers=["new_documents"])
    assert rig.posts == []
    _freshness_attempt(rig, monkeypatch, Q_PLAIN, triggers=["insufficient"])
    assert len(rig.posts) == 1
