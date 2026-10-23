"""DAR-B-1023 item 8: the per-job document cap holds on the SHIPPED bounds (docs_per_day null = no daily cap since
2026-10-22, max_docs_per_job 2) when several discovered records are all eligible. Before this file no test failed when
both enforcement points (auto_acquire.docs_budget null-cap branch and inline_acquire job_max) ignored max_docs_per_job.
Offline: fake Europe PMC router, fake store; no network, DB or env read."""
from __future__ import annotations

from tests import acq_support as A
from tests import fulltext_synth
from tests.test_acq_discovered import Router, _direct, cfg_on, oa_src, run  # noqa: F401  (_direct: autouse reset)
from vera import acq_config


def _three_good():
    dois = [f"10.5555/darb.cap.{i}" for i in range(3)]
    pmcs = [f"PMC90000{i}" for i in range(3)]
    records = {d: A.epmc_record(d, pmcid=p) for d, p in zip(dois, pmcs)}
    jats = {p: fulltext_synth.jats(f"darb_cap_{i}", "epi", d, seed=i + 1) for i, (d, p) in enumerate(zip(dois, pmcs))}
    srcs = [oa_src(doi=d, wid=f"https://openalex.org/W90{i}", title=f"Cap candidate {i}") for i, d in enumerate(dois)]
    return Router(records, jats), srcs


def test_shipped_bounds_are_the_ones_under_test():
    c = acq_config.load_config()
    assert acq_config.docs_per_day_cap(c) is None
    assert int(acq_config.bound(c, "max_docs_per_job")) == 2


def test_three_eligible_discovered_records_store_at_most_max_docs_per_job():
    web, srcs = _three_good()
    cfg = cfg_on()
    assert acq_config.docs_per_day_cap(cfg) is None  # shipped null cap: only max_docs_per_job bounds the job
    o, store = run(cfg, srcs, web)
    job_max = int(acq_config.bound(cfg, "max_docs_per_job"))
    assert o.stored <= job_max and len(store.plans) <= job_max
    outcomes = [e["outcome"] for e in o.public()["discovered"]]
    assert outcomes.count("admitted") == job_max, outcomes
    assert len(outcomes) == 3 and all(x in ("admitted", "not_used", "not_evaluated") for x in outcomes), outcomes


def test_cap_of_one_is_respected():
    web, srcs = _three_good()
    cfg = cfg_on()
    cfg["bounds"]["max_docs_per_job"] = 1
    o, store = run(cfg, srcs, web)
    assert o.stored == 1 and len(store.plans) == 1
    assert [e["outcome"] for e in o.public()["discovered"]].count("admitted") == 1
