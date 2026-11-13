"""PREON-ENFORCE-1112: the `_pre_on_condition` of config/permitted_originals.json enforced in code. In mode "on" the
ORIGINAL_HTML route claims, fetches and stores only URLs that are members of the harvested snapshot's entries; a missing,
invalid, stale, future-dated, unparseable or empty index admits nothing. Shadow and off behave as before. Offline."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from tests import acq_support as A
from tests.test_original_html_wpg_1111 import (DOE, DOE_HOST, NASA, SHIPPED, Web, cand, code, page, run, index_snapshot,
                                               set_mode)
from vera import acq_config, acq_discovered, auto_acquire, freshness, original_html

HOST = acq_config.reason(A.cfg(), "host")
NOTOA = "R-ACQ-NODOI"  # evaluate_candidate: the route does not claim it, the default route refuses (NOROUTE-CODE-1112: no DOI or PMCID, nothing sent)


def direct(web, c=None):
    """original_html.evaluate called directly (a caller that skipped the claim): must still refuse with `host`."""
    from vera import fulltext_loader
    cfg = A.cfg()
    icfg = acq_config.ingest_cfg(fulltext_loader.load_config(), cfg)
    with pytest.raises(auto_acquire._Stop) as e:
        original_html.evaluate(A.job(question="What are quarks and gluons?"), 1.0, "", c or cand(), None, cfg,
                               A.deps([], web), icfg, auto_acquire._fd_cfg(), clock=A.clock_at())
    return e.value.code
OTHER = "https://www.energy.gov/science/doe-explainsneutrinos"


@pytest.fixture(autouse=True)
def _fresh():
    original_html.reset_for_tests()
    acq_discovered.reset_for_tests()
    yield
    original_html.reset_for_tests()


def use(monkeypatch, snap, mode="on", **limits):
    c = set_mode(monkeypatch, "shadow", **limits)
    c["mode"] = mode
    monkeypatch.setattr(original_html, "load_snapshot", lambda cfg=None: snap)
    return c


def iso(delta):
    return (datetime.now(timezone.utc) + delta).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------- gate: on
def test_on_url_not_in_index_is_refused_before_any_request(monkeypatch):
    use(monkeypatch, index_snapshot([OTHER]))
    web = Web({DOE: page()})
    assert code(web) == NOTOA and web.urls == []
    assert direct(web) == HOST and web.urls == []
    assert original_html.claims_url(DOE) is False


def test_on_url_in_index_is_claimed_and_stored(monkeypatch):
    use(monkeypatch, index_snapshot([DOE]))
    assert original_html.claims_url(DOE) is True
    plan, _ = run(Web({DOE: page()}))
    assert plan["canonical_url"] == DOE


@pytest.mark.parametrize("snap", [
    None,
    index_snapshot([]),
    index_snapshot([DOE], harvested_at=""),
    index_snapshot([DOE], harvested_at="not a date"),
    index_snapshot([DOE], harvested_at="2026-11-01T00:00:00"),  # naive
])
def test_on_missing_empty_or_unparseable_index_refuses(monkeypatch, snap):
    use(monkeypatch, snap)
    assert original_html.claims_url(DOE) is False
    web = Web({DOE: page()})
    assert code(web) == NOTOA and direct(web) == HOST and web.urls == []


def test_on_stale_and_future_index_refuse_and_fresh_is_allowed(monkeypatch):
    use(monkeypatch, index_snapshot([DOE], harvested_at=iso(timedelta(days=-31))))
    assert original_html.claims_url(DOE) is False
    use(monkeypatch, index_snapshot([DOE], harvested_at=iso(timedelta(days=-29))))
    assert original_html.claims_url(DOE) is True
    original_html.reset_for_tests()
    use(monkeypatch, index_snapshot([DOE], harvested_at=iso(timedelta(hours=1))))
    assert original_html.claims_url(DOE) is False


@pytest.mark.parametrize("age", [0, -1, 366, 1.5, "30", True, None])
def test_bad_max_age_refuses_instead_of_falling_back(monkeypatch, age):
    use(monkeypatch, index_snapshot([DOE]), index_max_age_days=age)
    assert original_html.claims_url(DOE) is False


def test_max_age_override_is_honoured(monkeypatch):
    use(monkeypatch, index_snapshot([DOE], harvested_at=iso(timedelta(days=-3))), index_max_age_days=2)
    assert original_html.claims_url(DOE) is False
    use(monkeypatch, index_snapshot([DOE], harvested_at=iso(timedelta(days=-3))), index_max_age_days=5)
    assert original_html.claims_url(DOE) is True


def test_shipped_empty_snapshot_in_on_mode_refuses(monkeypatch):
    c = set_mode(monkeypatch, "shadow")
    c["mode"] = "on"  # real load_snapshot, shipped empty seed
    assert original_html.claims_url(DOE) is False
    assert code(Web({DOE: page()})) == NOTOA and direct(Web({DOE: page()})) == HOST


def test_snapshot_path_outside_config_dir_refuses(monkeypatch):
    c = use(monkeypatch, index_snapshot([DOE]))
    c["snapshot"] = {"path": "../outside.json"}
    assert original_html.index_urls(c) is None
    c["snapshot"] = {"path": "/etc/passwd"}
    assert original_html.index_urls(c) is None
    c["snapshot"] = {"path": "config/permitted_catalogue.json"}
    assert original_html.index_urls(c) == frozenset({DOE})


def test_redirect_to_an_unindexed_same_row_url_is_refused(monkeypatch):
    use(monkeypatch, index_snapshot([DOE]))
    web = Web({DOE: page(), OTHER: page()},
              over={DOE: A.Resp(status=302, headers={"Location": OTHER})})
    with pytest.raises(auto_acquire._Stop):
        run(web)
    assert OTHER not in web.docs()


def test_redirect_to_an_indexed_url_is_followed(monkeypatch):
    use(monkeypatch, index_snapshot([DOE, OTHER]))
    web = Web({OTHER: page()}, over={DOE: A.Resp(status=302, headers={"Location": OTHER})})
    plan, _ = run(web)
    assert plan["canonical_url"] == DOE or OTHER in plan["canonical_url"]


@pytest.mark.parametrize("variant", [
    DOE + "/", DOE.upper(), DOE + "?x=1", DOE + "#frag", DOE.replace("energy.gov", "energy.gov:443") + "x",
    "http://www.energy.gov/science/doe-explainsquarks-and-gluons", DOE + "%2f", "https://www.energy.gov/science/./doe-explainsquarks-and-gluons-x",
    "https://www.energy.gov/science/doe-explainsquarks-and-gluons/../doe-explainsneutrinos",
    "https://user@www.energy.gov/science/doe-explainsquarks-and-gluons", DOE + ";a=b", DOE + "%00",
    "https://www.energy.gov/science/doe-explainsQuarks-and-gluons", "https://www.energy.gov.evil.example/science/doe-explainsquarks-and-gluons",
])
def test_unindexed_variants_never_match(monkeypatch, variant):
    use(monkeypatch, index_snapshot([DOE]))
    got = original_html.in_index(variant)
    canon = original_html.normalise_url(variant)
    assert got == (canon == DOE)  # only the one canonical form of the entry can match
    if canon != DOE:
        assert original_html.claims_url(variant) is False


def test_percent_encoded_unreserved_and_host_case_normalise_to_the_entry(monkeypatch):
    use(monkeypatch, index_snapshot([DOE]))
    assert original_html.claims_url("https://WWW.Energy.GOV/science/doe-explains%71uarks-and-gluons") is True
    assert original_html.claims_url(DOE + "/") is True  # trailing slash normalises to the same canonical URL


def test_cand_url_never_disagrees_with_claims_cand(monkeypatch):
    use(monkeypatch, index_snapshot([DOE]))
    c = {"url": OTHER, "discovered_url": DOE}
    assert original_html.claims_cand(c) is True and original_html.cand_url(c) == DOE
    c2 = {"url": OTHER, "discovered_url": OTHER}
    assert original_html.claims_cand(c2) is False and original_html.cand_url(c2) is None


def test_match_subject_returns_only_indexed_entries(monkeypatch):
    snap = index_snapshot([DOE])
    snap["entries"][0].update(title="Quarks and gluons", slug_terms=["quarks", "gluons"])
    snap["sha256"] = original_html.snapshot_digest(snap)
    use(monkeypatch, snap)
    assert [h["url"] for h in original_html.match_subject("quarks")] == [DOE]


# ---------------------------------------------------------------------------- shadow and off unchanged
@pytest.mark.parametrize("snap", [None, index_snapshot([]), index_snapshot([OTHER]), index_snapshot([DOE], harvested_at="")])
def test_shadow_ignores_the_index_entirely(monkeypatch, snap):
    use(monkeypatch, snap if snap is not None else original_html.load_snapshot(), mode="shadow")
    assert original_html.claims_url(DOE) is True
    assert original_html.cand_url({"url": DOE}) == DOE
    assert code(Web({DOE: page()})) == acq_config.reason(A.cfg(), "original_shadow")


def test_shadow_does_not_call_the_index_code(monkeypatch):
    use(monkeypatch, index_snapshot([]), mode="shadow")
    monkeypatch.setattr(original_html, "index_urls", lambda cfg=None: pytest.fail("index consulted in shadow"))
    assert original_html.claims_url(DOE) is True
    assert code(Web({DOE: page()})) == acq_config.reason(A.cfg(), "original_shadow")


def test_off_claims_nothing(monkeypatch):
    use(monkeypatch, index_snapshot([DOE]), mode="off")
    monkeypatch.setattr(original_html, "index_urls", lambda cfg=None: pytest.fail("index consulted in off"))
    assert original_html.claims_url(DOE) is False


def test_path_checker_without_index_is_unchanged():
    row = SHIPPED["rows"][0]
    chk = original_html.path_checker(row)
    assert chk(DOE) is True and chk(OTHER) is True and chk(DOE + "?a=1") is False
    assert original_html.path_checker(row, frozenset({DOE}))(OTHER) is False


def test_no_row_mode_was_changed_by_this_task():
    assert SHIPPED["mode"] in ("off", "shadow")
    assert {r["id"]: r["store"] for r in SHIPPED["rows"]} == {"energy_gov_doe_explains": True, "nasa_science_explainers": False}
