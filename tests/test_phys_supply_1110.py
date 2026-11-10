"""PHYS-SUPPLY-1110 (diag work/HADRONS-B3-DIAG-1109 fix 4b): a journal-DOI candidate (e.g. a Phys. Rev. D DOI) whose
metadata names an arXiv version (INSPIRE arxiv_eprints, an OpenAlex arxiv.org location, or the arXiv record's own URL)
falls back to the arXiv full-text route when Europe PMC (still tried first) has no open-access record (not_oa). The arXiv version's own licence decides:
CC families admitted, arXiv's default licence held (fair-use tier off), ND rejected; robots Crawl-delay 15 kept.

SYNTHETIC data only (invented 10.9999 DOIs; the arXiv fixture is tests/arxiv_synth.py). Offline fakes: no socket, no
DB, no model."""
from __future__ import annotations

import copy
import json
import pathlib

import pytest

from tests import acq_support as A
from tests import arxiv_synth as X
from tests.test_acq_discovered import Router, _direct, cfg_on, disc, run  # noqa: F401  (_direct: autouse)
from vera import arxiv_fulltext, auto_acquire, fulltext_routes as FR, portal_inspire as pi
from vera import search_providers as sp

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRD = "10.9999/synth.physrevd.1"
ROBOTS = "https://arxiv.org/robots.txt"


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch):
    waits = []
    monkeypatch.setattr(arxiv_fulltext, "_sleep", lambda s: waits.append(s))
    pi.reset_for_tests()
    yield waits
    pi.reset_for_tests()


# ------------------------------------------------------------------------------------------------ metadata sources
def test_openalex_arxiv_location_is_recorded_as_metadata_only():
    w = {"open_access": {"is_oa": True}, "best_oa_location": {"license": "cc-by", "is_oa": True,
                                                              "landing_page_url": f"https://doi.org/{PRD}"},
         "locations": [{"is_oa": True, "landing_page_url": f"http://arxiv.org/abs/{X.AID}",
                        "pdf_url": f"http://arxiv.org/pdf/{X.AID}"}], "ids": {}}
    assert sp.openalex_oa_info(w)["arxiv_id"] == X.AID
    w["locations"] = [{"is_oa": True, "landing_page_url": f"https://arxiv.org/abs/{X.AID}{X.VER}"}]
    assert sp.openalex_oa_info(w)["arxiv_id"] == X.AID + X.VER
    w["locations"] = [{"is_oa": True, "landing_page_url": "https://example.org/arxiv.org/abs/2101.00001"}]
    assert "arxiv_id" not in sp.openalex_oa_info(w)  # only an arxiv.org URL counts; the key is absent otherwise


DOC = {"hits": {"hits": [{"metadata": {
    "dois": [{"value": PRD}], "arxiv_eprints": [{"value": X.AID}], "emails": ["synthetic@example.org"],
    "references": [{"reference": {"dois": ["10.9999/synth.ref.1"], "arxiv_eprint": "2101.05555"}},
                   {"reference": {"dois": ["10.9999/synth.ref.2"]}},
                   {"reference": {"arxiv_eprint": "hep-ph/9901001"}}]}}]}}


class _Resp:
    status_code, headers = 200, {}

    def __init__(self, body):
        self._b = body

    def iter_content(self, n):
        yield self._b

    def close(self):
        pass


def test_inspire_keeps_one_candidate_per_work_with_its_eprint_as_arxiv_alt():
    c = json.loads((ROOT / "config" / "reference_portals.json").read_text(encoding="utf-8"))
    counts: dict = {}
    out = pi.adapter("What are hadrons?", None, c, None, counts, clock=lambda: 0.0,
                     get=lambda u, p, t: _Resp(json.dumps(DOC).encode()))
    got = [(x.get("doi"), (x.get("arxiv_alt") or {}).get("id"), x.get("arxiv")) for x in out]
    assert got == [(PRD, X.AID, None), ("10.9999/synth.ref.1", "2101.05555", None), ("10.9999/synth.ref.2", None, None),
                   ("10.48550/arxiv.hep-ph/9901001", None, "hep-ph/9901001")]
    assert "synthetic@example.org" not in json.dumps(out)
    assert all(x["acq_source"] == "reference_portal" for x in out)


# ------------------------------------------------------------------------------------------------ route fallback
def _cfg(on=True):
    c = A.cfg()
    c["arxiv_route"]["enabled"] = on
    return c


class _Hit(Exception):
    pass


@pytest.fixture
def fakes(monkeypatch):
    """Europe PMC resolve -> the given stop code; arXiv evaluate -> records (doi, cand) and raises _Hit."""
    from vera import acq_config
    st = {"epmc": "not_oa", "arxiv": []}

    def resolve(doi, cfg, deps, clock=None, deadline=None, pmcid=None, chain=None):
        raise auto_acquire._Stop(acq_config.reason(cfg, st["epmc"]))

    def arx(job, score, doi, cand, *a):
        st["arxiv"].append((doi, copy.deepcopy(cand)))
        raise _Hit()
    monkeypatch.setattr(auto_acquire, "resolve_oa", resolve)
    monkeypatch.setattr(arxiv_fulltext, "evaluate", arx)
    return st


def _deps(known=()):
    class L:
        @staticmethod
        def known(doi, url, isbn):
            return doi in known
    return type("D", (), {"lookup": L})()


def _go(cand, cfg=None, known=()):
    from vera import acq_config
    try:
        auto_acquire.evaluate_candidate(auto_acquire.Job("What are hadrons?", "t"), 0.5, PRD, cand, None,
                                        cfg or _cfg(), _deps(known), {}, {}, clock=lambda: 0.0)
    except _Hit:
        return "arxiv"
    except auto_acquire._Stop as e:
        return e.code
    raise AssertionError("no stop")


@pytest.mark.parametrize("cand", [
    A.cand(PRD, arxiv_alt={"id": X.AID + X.VER, "source": "inspire"}),
    A.cand(PRD, oa={"is_oa": True, "has_oa_location": True, "arxiv_id": X.AID + X.VER}),
    A.cand(PRD, arxiv_alt={"id": X.AID + X.VER}, pmcid="PMC7300009"),  # PMC record but no OA text: still falls back
])
def test_not_oa_journal_doi_falls_back_to_its_arxiv_version(fakes, cand):
    before = copy.deepcopy(cand)
    assert FR.select(cand, PRD, _cfg())[0] is FR.EUROPEPMC  # Europe PMC FIRST (peer-reviewed text), unchanged
    assert _go(cand) == "arxiv"
    doi, c = fakes["arxiv"][0]
    assert doi == X.DOI and c["doi"] == X.DOI and c["journal_doi"] == PRD and c["arxiv_via"] == "journal_doi_alt"
    assert c["arxiv"] == {"id": X.AID, "version": X.VER} and c["url"] == X.ABS_URL
    assert cand == before  # never mutated
    assert FR.routable(cand, _cfg()) is True and FR.reroutes_to_arxiv(cand, _cfg()) is True


def test_no_fallback_without_arxiv_version_route_off_or_other_stop(fakes):
    from vera import acq_config
    c = _cfg()
    assert _go(A.cand(PRD)) == acq_config.reason(c, "not_oa")  # no arXiv version named
    assert _go(A.cand(PRD, arxiv_alt={"id": "not-an-id"})) == acq_config.reason(c, "not_oa")
    assert _go(A.cand(PRD, arxiv_alt={"id": X.AID}), _cfg(False)) == acq_config.reason(c, "not_oa")  # route off
    fakes["epmc"] = "no_licence"
    assert _go(A.cand(PRD, arxiv_alt={"id": X.AID})) == acq_config.reason(c, "no_licence")  # only not_oa falls back
    assert fakes["arxiv"] == []


def test_arxiv_version_already_in_corpus_stops_before_any_request(fakes):
    from vera import acq_config
    assert _go(A.cand(PRD, arxiv_alt={"id": X.AID}), known={X.DOI}) == acq_config.reason(_cfg(), "in_corpus")
    assert fakes["arxiv"] == []


# ------------------------------------------------------------------------------------------------ end to end (offline)
def _src(**kw):
    return {"provider": "openalex", "doi": PRD, "url": f"https://doi.org/{PRD}", "title": X.TITLE,
            "snippet": "transmission incidence outbreak pathogen", "openalex_id": "https://openalex.org/W91",
            "stored_text_kind": "abstract_metadata", **kw}


def test_prd_doi_with_cc_by_arxiv_version_is_read_from_arxiv_and_admitted():
    web = Router({}, {}, X.routes())
    o, store = run(cfg_on(), [_src(oa={"is_oa": True, "has_oa_location": True, "arxiv_id": X.AID + X.VER})], web)
    urls = [u for u, _h, _k in web.calls]
    assert X.ABS_URL in urls and X.HTML_URL in urls
    epmc = [i for i, u in enumerate(urls) if "ebi.ac.uk" in u]
    assert epmc and max(epmc) < urls.index(X.ABS_URL)  # Europe PMC asked first (not_oa), then the arXiv version
    assert disc(o)["outcome"] == "admitted", disc(o)
    a = store.plans[0]["source"]["provenance"]["acquisition"]["arxiv"]
    assert a["licence"] == "cc by" and (a["id"], a["version"]) == (X.AID, X.VER)


def test_prd_doi_with_default_licence_arxiv_version_is_held_and_html_never_fetched():
    web = Router({}, {}, X.routes(licence=X.ARXIV_DEFAULT))
    o, store = run(cfg_on(), [_src(arxiv_alt={"id": X.AID + X.VER, "source": "inspire"})], web)
    assert disc(o)["reason"] == "R-ACQ-ARXIV_LICENCE_HOLD" and store.plans == []  # fair-use tier ships off
    assert X.HTML_URL not in [u for u, _h, _k in web.calls]


def test_prd_doi_with_nd_arxiv_version_is_rejected():
    web = Router({}, {}, X.routes(licence=X.CC_BY_ND))
    o, store = run(cfg_on(), [_src(arxiv_alt={"id": X.AID + X.VER})], web)
    assert disc(o)["outcome"] != "admitted" and store.plans == []
    assert X.HTML_URL not in [u for u, _h, _k in web.calls]


def test_prd_doi_without_arxiv_version_stops_not_oa_on_europe_pmc():
    web = Router({}, {}, X.routes())
    o, store = run(cfg_on(), [_src()], web)
    assert not any("arxiv.org" in u for u, _h, _k in web.calls) and store.plans == []


def test_arxiv_crawl_delay_15_is_kept_for_the_rerouted_doi(monkeypatch):
    """The same moving fake clock as tests/test_p0dta_arxiv_ids.py: every arXiv request after the first waits the full
    robots Crawl-delay (15 s); the delay is never shortened for a re-routed journal DOI."""
    from vera import acq_discovered, inline_acquire

    class Clk:
        t = 1000.0

        def __call__(self):
            return self.t
    clk, waits = Clk(), []
    monkeypatch.setattr(arxiv_fulltext, "_sleep", lambda s: (waits.append(s), setattr(clk, "t", clk.t + float(s))))
    acq_discovered._ROBOTS.clear()
    arxiv_fulltext._HOST_LAST.clear()
    web = Router({}, {}, {**X.routes(), ROBOTS: A.Resp(body=b"User-agent: *\nCrawl-delay: 15\n", ctype="text/plain")})
    d = A.deps([], web, store=A.Store())
    try:
        o = inline_acquire.acquire(A.Q_EPI, cfg_on(), env=A.ENV_ON, deps_factory=lambda a, e: d,
                                   discovered=[_src(arxiv_alt={"id": X.AID + X.VER})], clock=clk)
    finally:
        arxiv_fulltext._HOST_LAST.clear()
        acq_discovered._ROBOTS.clear()
    urls = [u for u, _h, _k in web.calls]
    assert X.ABS_URL in urls and X.HTML_URL in urls, (urls, disc(o))
    assert waits and min(waits) >= 15 - 1e-6 and len(waits) >= 2  # abs and html each waited the full Crawl-delay
