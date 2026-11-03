"""P1-STAGEA-1031: documents the CURRENT effective inline fresh-read ceiling and the unchanged caps (OD-1031-P1 item 3:
DP-7, OD-DOCSCAP-1022, DBW-1023 preserved). inline_fresh_docs is further clamped by docs_budget() -> max_docs_per_job,
so the effective ceiling is the daily budget remaining, at most 31 (OD-1031-CEIL31); 2 with the kill switch. Offline; reads shipped config only."""
from __future__ import annotations

from vera import acq_config, auto_acquire as aa


def test_shipped_caps_unchanged():
    b = acq_config.load_config()["bounds"]
    assert (b["jobs_per_hour"], b["jobs_per_day"], b["max_docs_per_job"]) == (30, 180, 2)
    assert b["docs_per_day"] is None and b["max_candidates_resolved"] == 31
    assert acq_config.load_config()["inline"]["deadline_s"] == 45


def test_per_prompt_ceiling_on_by_default_31_background_stays_2():  # OD-1031-CEIL31, OD-1031-NO500
    cfg = acq_config.load_config()
    assert cfg["inline"]["per_prompt_ceiling_enabled"] is True
    assert cfg["bounds"]["max_docs_per_prompt"] == 31 and "inline_fresh_docs" not in cfg["bounds"]
    assert acq_config.inline_fresh_docs(cfg) == 31
    assert aa.docs_budget(cfg, None, per_job=acq_config.inline_fresh_docs(cfg)) == 31
    assert aa.docs_budget(cfg, None) == 2  # background jobs: max_docs_per_job unchanged
    n = cfg["bounds"]["_max_docs_per_prompt_note"]
    assert "OD-1031-CEIL31" in n and "NOT calibrated" in n and "statistical descriptiveness" in n and "5,580" in n


def test_kill_switch_restores_two_docs_and_only_true_enables():
    cfg = acq_config.load_config()
    cfg["inline"]["per_prompt_ceiling_enabled"] = False
    assert acq_config.inline_fresh_docs(cfg) == 2
    cfg["inline"]["per_prompt_ceiling_enabled"] = "yes"
    assert acq_config.inline_fresh_docs(cfg) == 2
    cfg["inline"]["per_prompt_ceiling_enabled"] = True
    cfg["bounds"]["inline_fresh_docs"] = 3  # legacy key still overrides, only with the flag on
    assert acq_config.inline_fresh_docs(cfg) == 3


def test_inline_budget_respects_a_daily_cap_when_one_is_set():
    cfg = acq_config.load_config()
    cfg["bounds"]["docs_per_day"] = 5
    cfg["inline"]["per_prompt_ceiling_enabled"] = True

    class L:
        def docs_last_24h(self):
            return 3

    class D:
        lookup = L()
    aa.reset_for_tests()
    assert aa.docs_budget(cfg, D(), per_job=acq_config.inline_fresh_docs(cfg)) == 2  # remaining daily budget still bounds it


def test_page_size_supplies_31_candidates_per_provider_in_one_request():
    c = acq_config.load_config()
    assert c["search"]["results_per_provider"] == 31 and c["search"]["europepmc"]["page_size"] == 31
    assert c["bounds"]["max_search_calls_per_job"] == 7 and c["search"]["max_calls_per_provider_per_job"] == 1


def test_europepmc_one_request_uses_page_size_31():
    from vera import search_providers as sp
    seen = []

    class R:
        status_code, text = 200, '{"resultList": {"result": []}}'

    def tx(url, params, headers, timeout):
        seen.append(params)
        return R()
    sp.europepmc_search("zebrafish heart regeneration", 31, 5.0, acq_config.load_config()["search"]["europepmc"], transport=tx)
    assert len(seen) == 1 and seen[0]["pageSize"] == 31


def test_shortlist_ceiling_is_31_and_discovered_max_does_not_clip_it():  # OD-1031-SHORTLIST31
    from vera import acq_discovered
    c = acq_config.load_config()
    assert int(acq_config.bound(c, "max_candidates_resolved")) == 31
    assert acq_discovered.discovered_max(c) == 31
    assert c["bounds"]["max_search_calls_per_job"] == 7 and c["inline"]["per_doc_timeout_s"] == 25
    assert c["inline"]["fetch_workers"] == 4
    assert "OD-1031-SHORTLIST31" in c["bounds"]["_max_candidates_resolved_note"]
    assert "NOT yet calibrated" in c["bounds"]["_max_candidates_resolved_note"]


def test_rank_candidates_cut_follows_the_ceiling():
    import copy
    from vera import auto_acquire as aa
    c = copy.deepcopy(acq_config.load_config())
    for n in (6, 31):
        c["bounds"]["max_candidates_resolved"] = n
        cands = [{"doi": f"10.9999/p1.{i}", "title": "t", "url": f"https://doi.org/10.9999/p1.{i}"} for i in range(40)]
        out = aa.rank_candidates("test question", cands, c, set(), {}, lambda *_: None)
        assert len(out) == n


def test_job_caps_od_1031_promptlim_and_unchanged_companions():
    b = acq_config.load_config()["bounds"]
    assert (b["jobs_per_hour"], b["jobs_per_day"]) == (30, 180)
    assert (b["max_docs_per_job"], b["queue_max"]) == (2, 4) and aa.BG_MIN_FREE_HOURLY_JOBS == 2
    assert "OD-1031-PROMPTLIM" in b["_jobs_per_hour_day_note"]


def test_inline_pass_stores_up_to_31_with_flag_on_2_with_kill_switch():  # end-to-end through inline_acquire, offline fakes
    from tests import acq_support as A
    from tests import fulltext_synth as fs
    from vera import inline_acquire
    import pytest  # noqa: F401
    dois = [f"10.9999/p1c.{i}" for i in range(5)]
    web = A.Web({x: A.epmc_record(x, pmcid=f"PMC74000{i}") for i, x in enumerate(dois)},
                {f"PMC74000{i}": fs.jats(f"p1c{i}", "epi", x) for i, x in enumerate(dois)})
    res = {}
    for flag in (False, True):
        aa.reset_for_tests()
        inline_acquire.reset_for_tests()
        cfg = A.cfg(max_candidates_resolved=31)
        cfg["inline"]["per_prompt_ceiling_enabled"] = flag
        d = A.deps([A.cand(x) for x in dois], web)
        o = inline_acquire.acquire(A.Q_EPI, cfg, env=A.ENV_ON, clock=__import__("time").monotonic,
                                   deps_factory=lambda a, e: d)
        res[flag] = o.stored
    assert res[False] == 2 and res[True] == 5, res
