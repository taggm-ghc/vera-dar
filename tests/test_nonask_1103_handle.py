"""NONASK-1103 WP-D: Handle (hdl.handle.net, hdl:) recognition. OFFLINE; requests are made to fail."""
import pytest

from vera import auto_acquire, research_outcome
from vera import search_providers as sp


@pytest.mark.parametrize("u,v", [
    ("https://hdl.handle.net/20.500.12345/abc-1", "20.500.12345/abc-1"),
    ("http://hdl.handle.net/1721.1/12345", "1721.1/12345"),
    ("https://handle.net/2027/mdp.39015", "2027/mdp.39015"),
    ("https://hdl.handle.net/api/handles/1721.1/99?noredirect", "1721.1/99"),
    ("hdl:1721.1/12345", "1721.1/12345"),
    ("HDL:20.500.1/x%2Fy", "20.500.1/x/y"),
])
def test_handle_forms(u, v):
    assert sp.ddg_identifier(u) == {"kind": "handle", "value": v}


def test_doi_in_handle_url_is_still_a_doi():
    assert sp.ddg_identifier("https://hdl.handle.net/10.1000/xyz123") == {"kind": "doi", "value": "10.1000/xyz123"}


@pytest.mark.parametrize("u", ["hdl:", "hdl:1721.1", "hdl:abc/def", "https://hdl.handle.net/", "https://hdl.handle.net/1721.1/",
                               "https://hdl.handle.net/foo/bar", "hdl:1721.1/ ", "https://example.org/hdl.handle.net/1721.1/9"])
def test_malformed_handles_are_none(u):
    assert sp.ddg_identifier(u) is None


def test_other_kinds_unchanged():
    assert sp.ddg_identifier("https://arxiv.org/abs/2401.00001v2")["kind"] == "arxiv"
    assert sp.ddg_identifier("https://doi.org/10.1234/abc")["kind"] == "doi"


@pytest.fixture
def no_requests(monkeypatch):
    import requests

    def boom(*a, **k):
        raise AssertionError("a request was made")
    for n in ("get", "post", "request", "head"):
        monkeypatch.setattr(requests, n, boom)
    monkeypatch.setattr(requests.Session, "request", boom)


def test_follow_url_marks_handle_makes_no_request_and_invents_no_doi(no_requests):
    c = {"url": "https://hdl.handle.net/1721.1/12345", "title": "t"}
    f = auto_acquire.follow_url(c)
    assert f["pid"] == {"kind": "handle", "value": "1721.1/12345"} and "doi" not in f and "pmcid" not in f
    assert auto_acquire._handle_pid(f) and auto_acquire._handle_pid(c)
    assert auto_acquire.follow_url({"url": "https://hdl.handle.net/10.1000/x"})["doi"] == "10.1000/x"  # DOI wins


def test_landing_route_is_off_so_no_fetch_route_exists():
    from vera import acq_config
    assert acq_config.load_config()["inline"]["landing_route"] is False


def test_rank_candidates_reason_is_pid_noroute_not_nodoi(no_requests):
    from tests import acq_support as A
    seen = []
    cands = [{"url": "https://hdl.handle.net/1721.1/12345", "title": "LLM orchestration"},
             {"url": "hdl:20.500.1/x", "title": "x"}, {"url": "https://example.org/a", "title": "y"}]
    out = auto_acquire.rank_candidates("llm orchestration", cands, A.cfg(), set(), auto_acquire._fd_cfg(), seen.append)
    assert out == [] and seen.count("R-ACQ-PID-NOROUTE") == 2 and seen.count("R-ACQ-NODOI") == 1


def test_research_outcome_maps_handle_to_no_route():
    assert research_outcome.reason_from_code("R-ACQ-PID-NOROUTE") == "no_route"


def test_ddg_result_with_handle_url_becomes_url_only_candidate_without_lookup(no_requests):
    out, counts = sp.ddg_to_candidates([{"url": "https://hdl.handle.net/1721.1/12345", "title": "T"}], {}, {},
                                       deadline=1e12, clock=lambda: 0.0, calls_left=1)
    assert len(out) == 1 and out[0]["doi"] is None and out[0]["ddg_id_kind"] == "url" and counts["ddg_url_only"] == 1
