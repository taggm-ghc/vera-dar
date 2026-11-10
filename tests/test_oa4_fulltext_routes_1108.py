"""OA-4 (OA37-1108, owner OD-1108-HITL D3): FulltextRoute registry, behaviour-preserving refactor.

Part 1 is a characterisation suite written and run GREEN on the pre-refactor code (origin/main 2d529ef) before any
change: it pins which full-text route evaluate_candidate dispatches to (and with which arguments) and what
routable_fulltext answers, over a fixed candidate matrix. Part 2 covers the registry itself.

OFFLINE: no network, DB or model. Route bodies are replaced by recorders that raise, so nothing is fetched."""
import ast
import copy
import pathlib
import types

import pytest

from tests import acq_support as A
from tests import arxiv_synth as X
from vera import acq_config, arxiv_fulltext, auto_acquire

ROOT = pathlib.Path(__file__).resolve().parents[1]


class _Hit(Exception):
    def __init__(self, route, **kw):
        super().__init__(route)
        self.route, self.kw = route, kw


@pytest.fixture
def rec(monkeypatch):
    def fake_arxiv(job, score, doi, cand, isbn, cfg, deps, icfg, fdcfg, clock, deadline, dur, min_spans):
        raise _Hit("arxiv", doi=doi, arxiv=dict(cand.get("arxiv") or {}), url=cand.get("url"), min_spans=min_spans,
                   score=score)

    def fake_resolve(doi, cfg, deps, clock=None, deadline=None, pmcid=None, chain=None):
        raise _Hit("europepmc", doi=doi, pmcid=pmcid, chain=chain is not None)

    monkeypatch.setattr(arxiv_fulltext, "evaluate", fake_arxiv)
    monkeypatch.setattr(auto_acquire, "resolve_oa", fake_resolve)


def _deps(known=False):
    return types.SimpleNamespace(lookup=types.SimpleNamespace(known=lambda *a, **k: known))


def _dispatch(cand, doi, cfg, known=False, min_spans=None):
    job = auto_acquire.Job(A.Q_EPI, "test")
    try:
        auto_acquire.evaluate_candidate(job, 0.5, doi, cand, None, cfg, _deps(known), {}, {}, clock=lambda: 0.0,
                                        min_spans=min_spans)
    except _Hit as h:
        return (h.route, h.kw)
    except auto_acquire._Stop as s:
        return ("stop", {"code": s.code})
    raise AssertionError("no route reached")


ARX = {"url": X.ABS_URL, "doi": X.DOI, "identifier": f"arXiv:{X.AID}{X.VER}", "title": "SYNTHETIC preprint",
       "snippet": ""}
PMC_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7300002/"

# (id, candidate, doi passed in, arxiv_route enabled, expected dispatch)
DISPATCH = [
    ("plain_doi", A.cand("10.9999/oa4.none"), "10.9999/oa4.none", True,
     ("europepmc", {"doi": "10.9999/oa4.none", "pmcid": None, "chain": False})),
    ("pmcid_only", {"url": "https://example.org/x", "pmcid": "PMC7300001", "title": "t"}, None, True,
     ("europepmc", {"doi": None, "pmcid": "PMC7300001", "chain": False})),
    ("doi_and_pmcid", A.cand("10.9999/oa4.pmc", pmcid="PMC7300003"), "10.9999/oa4.pmc", True,
     ("europepmc", {"doi": "10.9999/oa4.pmc", "pmcid": None, "chain": False})),
    ("arxiv_search_hit", dict(ARX), X.DOI, True,
     ("arxiv", {"doi": X.DOI, "arxiv": {"id": X.AID, "version": X.VER}, "url": X.ABS_URL, "min_spans": None,
                "score": 0.5})),
    ("arxiv_search_hit_no_doi", dict(ARX, doi=None), None, True,
     ("arxiv", {"doi": X.DOI, "arxiv": {"id": X.AID, "version": X.VER}, "url": X.ABS_URL, "min_spans": None,
                "score": 0.5})),
    ("arxiv_route_off", dict(ARX), X.DOI, False,
     ("europepmc", {"doi": X.DOI, "pmcid": None, "chain": False})),
    ("arxiv_dict_route_off", dict(ARX, arxiv={"id": X.AID, "version": X.VER}), X.DOI, False,
     ("arxiv", {"doi": X.DOI, "arxiv": {"id": X.AID, "version": X.VER}, "url": X.ABS_URL, "min_spans": None,
                "score": 0.5})),
    # PHYS-SUPPLY-1110: still Europe PMC FIRST (the peer-reviewed text); only a not_oa stop falls back to the arXiv
    # version (tests/test_phys_supply_1110.py)
    ("arxiv_url_journal_doi", dict(ARX, doi="10.1234/journal.x"), "10.1234/journal.x", True,
     ("europepmc", {"doi": "10.1234/journal.x", "pmcid": None, "chain": False})),
    ("discovered_doi", dict(A.cand("10.9999/oa4.disc"), discovery={"provider": "openalex"}), "10.9999/oa4.disc", True,
     ("europepmc", {"doi": "10.9999/oa4.disc", "pmcid": None, "chain": True})),
    ("discovered_arxiv", dict(ARX, discovery={"provider": "openalex"}), X.DOI, True,
     ("arxiv", {"doi": X.DOI, "arxiv": {"id": X.AID, "version": X.VER}, "url": X.ABS_URL, "min_spans": None,
                "score": 0.5})),
]


@pytest.mark.parametrize("cid,cand,doi,arx_on,want", DISPATCH, ids=[d[0] for d in DISPATCH])
def test_char_dispatch_unchanged(rec, cid, cand, doi, arx_on, want):
    cfg = A.cfg()
    cfg["arxiv_route"]["enabled"] = arx_on
    before = copy.deepcopy(cand)
    assert _dispatch(cand, doi, cfg) == want
    assert cand == before  # the caller's candidate is never mutated


def test_char_known_stops_before_any_route(rec):
    cfg = A.cfg()
    for cand, doi in ((A.cand("10.9999/oa4.k"), "10.9999/oa4.k"), (dict(ARX), X.DOI)):
        assert _dispatch(cand, doi, cfg, known=True) == ("stop", {"code": acq_config.reason(cfg, "in_corpus")})


def test_char_min_spans_passed_to_arxiv(rec):
    assert _dispatch(dict(ARX), X.DOI, A.cfg(), min_spans=1)[1]["min_spans"] == 1


# (id, candidate, arxiv_route enabled, expected routable_fulltext)
ROUTABLE = [
    ("none", A.cand("10.9999/r.none"), True, False),
    ("is_oa", A.cand("10.9999/r.isoa", oa={"is_oa": True}), True, True),
    ("is_oa_false", A.cand("10.9999/r.isoaf", oa={"is_oa": False}), True, False),
    ("has_oa_location", A.cand("10.9999/r.loc", oa={"has_oa_location": True}), True, True),
    ("epmc_ft", A.cand("10.9999/r.ft", oa={"epmc_ft": True}), True, True),
    ("pmcid", A.cand("10.9999/r.pmc", pmcid="PMC7300001"), True, True),
    ("oa_pmcid", A.cand("10.9999/r.oapmc", oa={"pmcid": "PMC7300004"}), True, True),
    ("pmc_url_only", {"url": PMC_URL, "title": "t"}, True, True),
    ("arxiv_hit", dict(ARX), True, True),
    ("arxiv_hit_route_off", dict(ARX), False, False),
    ("arxiv_url_only", {"url": X.ABS_URL, "title": "t"}, True, True),
    ("arxiv_url_only_route_off", {"url": X.ABS_URL, "title": "t"}, False, False),
    ("arxiv_dict_route_off", dict(ARX, arxiv={"id": X.AID, "version": X.VER}), False, True),
    ("arxiv_url_journal_doi", dict(ARX, doi="10.1234/journal.x"), True, True),  # PHYS-SUPPLY-1110 (was False)
    ("arxiv_url_journal_doi_route_off", dict(ARX, doi="10.1234/journal.x"), False, False),
    ("doi_url_only", {"url": "https://doi.org/10.9999/r.u", "title": "t"}, True, False),
    ("no_ident_url", {"url": "https://example.org/page", "title": "t"}, True, False),
    ("oa_not_dict", dict(A.cand("10.9999/r.bad"), oa="yes"), True, False),
    ("not_a_dict", "https://arxiv.org/abs/2605.23989v1", True, False),
    ("none_value", None, True, False),
]


@pytest.mark.parametrize("cid,cand,arx_on,want", ROUTABLE, ids=[r[0] for r in ROUTABLE])
def test_char_routable_unchanged(cid, cand, arx_on, want):
    cfg = A.cfg()
    cfg["arxiv_route"]["enabled"] = arx_on
    assert auto_acquire.routable_fulltext(cand, cfg) is want


def test_char_routable_arxiv_error_is_false(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(arxiv_fulltext, "search_candidate", boom)
    assert auto_acquire.routable_fulltext({"url": "https://example.org/p", "title": "t"}, A.cfg()) is False


# ---------------------------------------------------------------------------------------------- Part 2: registry
from vera import fulltext_routes as FR  # noqa: E402

_NET = {"requests", "httpx", "urllib", "urllib3", "http", "socket", "aiohttp", "ssl"}
_FETCH = {"_guarded_get", "_get", "_get_locked", "_b_get", "fetch", "fetch_candidate", "fetch_html", "pinned_get",
          "urlopen", "urlretrieve"}  # dict .get is fine; network modules are refused by import above


def test_registry_order_and_names():
    assert [r.name for r in FR.ROUTES] == ["arxiv", "europepmc"]
    assert FR.ROUTES[-1] is FR.EUROPEPMC and FR.by_name("arxiv") is FR.ARXIV and FR.by_name("plos") is None
    assert set(FR.ROUTABLE_ORDER) == set(FR.ROUTES)


def test_routes_declare_api_permission_ids():
    assert FR.EUROPEPMC.api_permission_id == "europepmc_rest"  # API-ALLOW-1107 entry id
    assert FR.ARXIV.api_permission_id is None  # website pages: robots/TDM gate, not an API entry


def test_default_route_claims_everything():
    for c in ({}, {"url": "x"}, A.cand("10.9999/z")):
        assert FR.EUROPEPMC.claims(c, A.cfg()) is True


def test_select_does_not_mutate_candidate():
    c = dict(ARX)
    before = copy.deepcopy(c)
    r, out, doi = FR.select(c, X.DOI, A.cfg())
    assert r is FR.ARXIV and out is not c and c == before and doi == X.DOI


def test_registry_module_makes_no_requests():
    """Every route fetches only through auto_acquire._guarded_get inside its route body; the registry itself imports
    no network module and calls no fetch primitive."""
    tree = ast.parse((ROOT / "vera" / "fulltext_routes.py").read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert not {a.name.split(".")[0] for a in n.names} & _NET
        if isinstance(n, ast.ImportFrom):
            assert (n.module or "").split(".")[0] not in _NET
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            assert name not in _FETCH, name


def test_evaluate_candidate_dispatches_only_through_registry():
    src = ast.get_source_segment((ROOT / "vera" / "auto_acquire.py").read_text(), next(
        n for n in ast.parse((ROOT / "vera" / "auto_acquire.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "evaluate_candidate"))
    assert "fulltext_routes.select" in src and "arxiv_fulltext" not in src and "resolve_oa" not in src


# ---------------------------------------------------------------------------------------------- Part 3: gates on every route
# Rebased onto a9b3af4 (OA12-1108 fail-closed rate limiting, API-ALLOW-R3-1108 api_permissions). Both gates live in
# the shared fetch (auto_acquire._guarded_get + default_deps transport), which every registered route uses; these
# pin that each route still reaches them, end to end through evaluate_candidate.
from vera import api_permissions as AP  # noqa: E402
from vera import rate_limit  # noqa: E402


@pytest.fixture
def policy_reset():
    yield
    AP.set_policy_for_tests(None)


def _eval(cand, doi, deps, cfg=None):
    try:
        auto_acquire.evaluate_candidate(auto_acquire.Job(A.Q_EPI, "test"), 0.5, doi, cand, None, cfg or A.cfg(), deps,
                                        {}, {}, clock=A.clock_at())
    except auto_acquire._Stop as s:
        return s.code
    raise AssertionError("expected a stop")


def test_europepmc_route_is_api_gated_before_any_request(policy_reset):
    from tests.test_api_permissions_1107 import _deny_epmc
    _deny_epmc()
    web = A.Web({}, {})
    deps = A.deps([], AP.gate_api_urls(web))
    assert FR.select(A.cand("10.9999/oa4.api"), "10.9999/oa4.api", A.cfg())[0] is FR.EUROPEPMC
    assert _eval(A.cand("10.9999/oa4.api"), "10.9999/oa4.api", deps) == AP.DENIED
    assert web.urls == []  # neither the REST API nor robots.txt was requested


def test_route_permission_ids_match_the_allowlist():
    pol = AP.get_policy()
    for r in FR.ROUTES:
        if r.api_permission_id is not None:
            assert AP.match(f"https://{pol.entries[r.api_permission_id]['host']}"
                            f"{pol.entries[r.api_permission_id]['path_prefix']}x") is not None
    assert "europepmc_rest" in pol.entries
    assert not AP.is_api_url(X.ABS_URL)  # arXiv abs pages: website content (robots/TDM gate), no API entry
    assert AP.is_api_url("https://arxiv.org/api/query")  # the arXiv /api path is an API entry (metadata only)


class _Unlisted:
    """A transport that behaves like rate_limit.limited_by_url(..., strict=True) for a host with no portal."""
    api_gated = False

    def __init__(self):
        self.calls = 0

    def __call__(self, url, **kw):
        self.calls += 1
        raise rate_limit.UnlistedHost(url)


@pytest.mark.parametrize("which", ["europepmc", "arxiv"])
def test_every_route_fails_closed_on_an_unlisted_host(monkeypatch, which):
    monkeypatch.setattr(arxiv_fulltext, "_sleep", lambda s: None)
    from vera import acq_discovered
    monkeypatch.setattr(acq_discovered, "_ROBOTS", {})
    monkeypatch.setattr(acq_discovered, "_TDMREP", {})
    t = _Unlisted()
    deps = A.deps([], t)
    if which == "europepmc":
        code = _eval(A.cand("10.9999/oa4.rl"), "10.9999/oa4.rl", deps)
    else:
        code = _eval(dict(ARX), X.DOI, deps)
    assert code == "R-ACQ-HOST", code  # refused with the host code; never fetched unlimited
    assert t.calls == 1  # the transport refused the first request; nothing was retried around it


def test_live_wiring_is_strict_and_api_gated():
    src = (ROOT / "vera" / "auto_acquire.py").read_text()
    assert "api_permissions.gate_api_urls(api_permissions.limited_transport(selective_fetch.pinned_get, strict=True))" in src
