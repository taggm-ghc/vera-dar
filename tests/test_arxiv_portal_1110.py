"""ARXIV-PORTAL-1110. Owner 2026-11-10, verbatim: "arxiv is a portal to sources, not a source, itself".

(1) arXiv text of any licence (CC-BY here) never becomes stored evidence, whatever route found the record: search
    (DuckDuckGo-mapped), the INSPIRE and Wikipedia reference portals, and discovered records. This holds even with
    config arxiv_route.enabled flipped back to true.
(2) An arXiv record is followed to its published version and its references through INSPIRE metadata. Nothing is
    fetched from arxiv.org. The new candidates go through the normal gates and carry portal_via "via arXiv <id>".
(3) The reading list shows the arXiv record as a portal entry (R-ACQ-ARXIV_PORTAL).

SYNTHETIC data only (invented DOIs; arXiv fixture tests/arxiv_synth.py). Offline fakes: no socket, no DB, no model."""
from __future__ import annotations

import copy
import json

import pytest

from tests import acq_support as A
from tests import arxiv_synth as X
from tests.test_acq_discovered import DOI, PMC, Router, _direct, cfg_on, epmc_web, run  # noqa: F401  (_direct: autouse)
from vera import acq_config, arxiv_fulltext, arxiv_portal, auto_acquire, discovery_panel, portal_inspire
from vera import reference_portal as rp
from vera import wiki_ref_portal as w

AID = X.AID
CODE = "R-ACQ-ARXIV_PORTAL"
_REAL_OA = arxiv_portal.openalex_lookup  # captured before the autouse fixture replaces it with a fake


class _Resp:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code, self._b, self.headers = status, body, headers or {}

    def iter_content(self, n):
        yield self._b

    def close(self):
        pass


def _inspire_body(own=(DOI,), refs=("10.9999/synth.ref.1", "10.9999/synth.ref.2")):
    return json.dumps({"hits": {"hits": [{"metadata": {
        "dois": [{"value": d} for d in own], "emails": ["synthetic@example.org"],
        "references": [{"reference": {"dois": [r]}} for r in refs] + [{"reference": {"arxiv_eprint": "2101.05555"}}]}}]}}
                      ).encode()


@pytest.fixture(autouse=True)
def inspire(monkeypatch):
    """Every INSPIRE call is a fake (no socket). Records (url, params)."""
    calls = []
    st = {"body": _inspire_body(), "status": 200}

    def fake(url, params, timeout_s):
        calls.append((url, dict(params)))
        return _Resp(st["status"], st["body"])
    monkeypatch.setattr(portal_inspire, "_default_get", fake)
    oa_calls = []  # OD-1110-ARXIV-OPENALEX: the default OpenAlex fallback is a recording fake (no socket)
    monkeypatch.setattr(arxiv_portal, "openalex_lookup", lambda aid, *a, **k: oa_calls.append(aid) or None)
    st["oa_calls"] = oa_calls
    arxiv_portal.reset_for_tests()
    rc = copy.deepcopy(rp.load_config())
    rc["mode"] = "on"  # the shipped mode (the test session forces the portal off by default)
    monkeypatch.setattr(rp, "load_config", lambda path=None: rc)
    st["rcfg"] = rc
    portal_inspire.reset_for_tests()
    st["calls"] = calls
    yield st
    portal_inspire.reset_for_tests()


def _arxiv_requests(web):
    return [u for u, _h, _k in web.calls if "arxiv.org" in u]


# ------------------------------------------------------------------------------------------------ (1) evidence off
def test_shipped_route_is_off_in_config_and_closed_in_code():
    acq = acq_config.load_config()
    assert acq["arxiv_route"]["enabled"] is False and arxiv_fulltext.PORTAL_ONLY is True
    on = copy.deepcopy(acq)
    on["arxiv_route"]["enabled"] = True  # flipped by mistake
    assert arxiv_fulltext.route_on(on) is False


def test_evaluate_refuses_before_any_request_even_with_the_config_flag_flipped():
    on = A.cfg()
    on["arxiv_route"]["enabled"] = True
    web = Router({}, {}, X.routes())
    cand = {"doi": X.DOI, "url": X.ABS_URL, "arxiv": {"id": AID, "version": X.VER}}
    with pytest.raises(auto_acquire._Stop) as e:
        arxiv_fulltext.evaluate(auto_acquire.Job(A.Q_EPI, "t"), 0.5, X.DOI, cand, None, on, A.deps([], web), {}, {})
    assert e.value.code == CODE and web.calls == []


def _ddg():
    return {"doi": X.DOI, "pmcid": None, "url": f"https://doi.org/{X.DOI}", "discovered_url": X.ABS_URL,
            "acq_source": "ddg", "ddg_id_kind": "arxiv", "provider": "ddg", "arxiv_id": AID + X.VER,
            "identifier": f"arXiv:{AID}{X.VER}", "title": X.TITLE, "snippet": "transmission incidence outbreak"}


def _portal(adapter):
    c = w.to_candidate({"arxiv": AID}, 0)
    c.update({"provider": f"portal:{adapter}", "acq_source": "reference_portal", "portal": adapter})
    return c


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("source", ["search_ddg", "inspire", "wikipedia", "discovered"])
def test_cc_by_arxiv_candidate_is_never_stored(source, flip, inspire):
    inspire["status"] = 404  # no published version found: nothing else to read
    c = cfg_on()
    c["arxiv_route"]["enabled"] = flip
    web = Router({}, {}, X.routes())  # a CC-BY abs page and LaTeXML HTML are available, but must never be read
    own = {"search_ddg": [_ddg()], "inspire": [_portal("inspire_refs")], "wikipedia": [_portal("wikipedia_title")],
           "discovered": []}[source]
    disc = [X.discovery_src()] if source == "discovered" else []
    o, store = run(c, disc, web, own=own)
    assert store.plans == [] and _arxiv_requests(web) == []
    reasons = [r.get("reason") for r in o.public().get("discovered") or ()] + [
        r.get("reason") for r in (o.public().get("candidates") or ())]
    if source == "discovered":
        assert CODE in reasons, o.public()


# ------------------------------------------------------------------------------------------------ (2) portal role
def test_extend_follows_a_seed_to_its_published_version_and_capped_references(inspire, caplog):
    import logging
    caplog.set_level(logging.INFO)
    seed = _ddg()
    journal = A.cand("10.9999/other.journal")
    out = arxiv_portal.extend("What are hadrons?", [seed, journal], A.cfg(), clock=lambda: 0.0)
    assert "autoacq arxivportal" in caplog.text and "arxivportal_calls=1" in caplog.text  # counts only
    assert out[:2] == [seed, journal]  # nothing removed; ranking drops the seed as a portal entry
    new = out[2:]
    assert [(c["doi"], c["portal_kind"]) for c in new] == [
        (DOI, "published_version"), ("10.9999/synth.ref.1", "reference"), ("10.9999/synth.ref.2", "reference")]
    assert all(c["portal_via"] == f"via arXiv {AID}" and c["acq_source"] == "arxiv_portal" and not c.get("arxiv")
               for c in new)
    assert "synthetic@example.org" not in json.dumps(out)  # no email field is read
    (url, params), = inspire["calls"]
    assert url == "https://inspirehep.net/api/literature" and params["q"] == f"arxiv:{AID}"
    assert set(params["fields"].split(",")) == {"dois.value", "references.reference.dois"}
    arxiv_portal.extend("What are hadrons?", [seed], A.cfg(), clock=lambda: 1.0)
    assert len(inspire["calls"]) == 1  # cached


def test_reference_cap_record_cap_and_off_switches(inspire, monkeypatch):
    c = A.cfg()
    c["arxiv_portal"]["max_refs_per_record"] = 1
    c["arxiv_portal"]["max_records_per_call"] = 2
    seeds = [dict(_ddg(), arxiv_id=a, identifier=f"arXiv:{a}", doi=f"10.48550/arxiv.{a}", url=f"https://doi.org/10.48550/arxiv.{a}",
                  discovered_url=f"https://arxiv.org/abs/{a}") for a in ("2601.00001", "2601.00002", "2601.00003")]
    out = arxiv_portal.extend("q", seeds, c, clock=lambda: 0.0)
    assert len(inspire["calls"]) == 2  # max_records_per_call
    assert sum(1 for x in out if x.get("portal_kind") == "reference") == 1  # max_refs_per_record (DOIs already seen skip)
    for off in ("config", "adapter", "permit"):
        portal_inspire.reset_for_tests()
        inspire["calls"].clear()
        c2 = A.cfg()
        with monkeypatch.context() as m:
            if off == "config":
                c2["arxiv_portal"]["enabled"] = False
            elif off == "adapter":
                rc = copy.deepcopy(inspire["rcfg"])
                rc["adapters"]["inspire_refs"]["enabled"] = False
                m.setattr(rp, "load_config", lambda path=None: rc)
            else:
                m.setattr(rp, "inspire_permitted", lambda: False)
            assert arxiv_portal.extend("q", seeds, c2, clock=lambda: 0.0) == seeds and inspire["calls"] == []


def test_429_starts_the_shared_cooldown_and_is_not_retried(inspire):
    inspire["status"] = 429
    seed = _ddg()
    assert arxiv_portal.extend("q", [seed], A.cfg(), clock=lambda: 0.0) == [seed]
    assert arxiv_portal.extend("q", [seed], A.cfg(), clock=lambda: 2.0) == [seed]  # cooldown >= 5 s
    assert len(inspire["calls"]) == 1


def test_seeds_and_non_seeds():
    assert arxiv_portal.seed_id(_ddg()) == AID
    assert arxiv_portal.seed_id(_portal("inspire_refs")) == AID
    assert arxiv_portal.seed_id(A.cand("10.9999/j.1")) is None  # a journal DOI is a source, not a portal
    assert arxiv_portal.seed_id(dict(_ddg(), doi="10.9999/j.2")) is None  # arXiv record carrying its journal DOI
    assert arxiv_portal.seed_id(A.cand("10.9999/j.3", pmcid="PMC7300001")) is None


def test_discovered_arxiv_record_leads_to_its_published_version_read_on_europe_pmc(inspire):
    web, _raw = epmc_web()  # the published version (DOI) is open in Europe PMC; arXiv is never requested
    web.routes.update(X.routes())
    o, store = run(cfg_on(), [X.discovery_src()], web)
    assert _arxiv_requests(web) == [] and len(store.plans) == 1, o.public()
    prov = store.plans[0]["source"]["provenance"]
    assert prov["doi"] == DOI and prov["acquisition"]["portal_via"] == f"via arXiv {AID}"
    assert len(inspire["calls"]) == 1
    seed_row = next(e for e in o.public()["discovered"] if e.get("service") == "arxiv")
    assert seed_row["reason"] == CODE


# ------------------------------------------------------------------------------------------------ (3) reading list
def test_reading_list_label_is_a_portal_entry():
    t = discovery_panel.REASON_TEXT[CODE]
    assert t.startswith("arXiv record:") and "published version" in t and "not used as evidence" in t
    assert "not in" not in t.lower()


# ------------------------------------------------------------------------------------------------ R3: answer-only web route
ARXIV_FAMILY = ["https://arxiv.org/html/2101.00001v1", "https://export.arxiv.org/abs/2101.00001",
                "https://ar5iv.labs.arxiv.org/html/2101.00001", "https://ar5iv.org/abs/2101.00001",
                "https://www.arxiv-vanity.com/papers/2101.00001/", "https://www.alphaxiv.org/abs/2101.00001",
                "https://de.arxiv.org/abs/2101.00001", "https://xxx.lanl.gov/abs/hep-ph/9901001"]


@pytest.mark.parametrize("url", ARXIV_FAMILY)
def test_r3_arxiv_and_mirrors_are_never_web_candidates_nor_any_hop(url):
    from vera import web_route
    assert web_route.is_web_candidate({"url": url, "title": "t"}) is False
    assert web_route.make_hop_gate("https://example.org/page", cfg={})(url) == "read_arxiv_portal"
    assert web_route.make_hop_gate(url, cfg={})(url) == "read_arxiv_portal"  # also as the start URL


def test_r3_redirect_hop_to_an_arxiv_mirror_is_refused_before_any_request():
    from vera import public_fetch, web_route
    sent = []

    def fetch(url, hop_gate=None, robots_cache=None):  # a public_fetch stand-in: gate each hop, then "request" it
        for hop in (url, "https://ar5iv.labs.arxiv.org/html/2101.00001"):  # the start URL redirects to a mirror
            why = hop_gate(hop)
            if why:
                raise public_fetch.FetchRefused(why)
            sent.append(hop)
        raise AssertionError("the mirror hop was requested")
    res = web_route.read_web("What are hadrons?", [{"url": "https://example.org/hadrons", "title": "Hadrons"}],
                             fetch=fetch, cfg={**web_route.load_config(), "topic_order": {"enabled": False}})
    assert sent == ["https://example.org/hadrons"] and res.pages == [] and len(res.listed) == 1
    assert res.dev[0]["gate"]["read"] == "read_arxiv_portal"
    assert "arXiv is used only to find original sources" in json.dumps(res.listed[0])


# ------------------------------------------------------------------------------------------------ OD-1110-ARXIV-ABSTRACT
def _rec(**kw):
    return {"title": "Hadron spectroscopy of quarks", "snippet": "We study hadrons made of quarks and gluons.",
            "published": "2025-01-01", "source_type": "scholarly", "provider": "openalex", **kw}


@pytest.mark.parametrize("arx", [
    _rec(doi="10.48550/arXiv.2101.00001", url="https://doi.org/10.48550/arXiv.2101.00001"),  # arXiv DataCite DOI
    _rec(url="https://arxiv.org/abs/2101.00001"),  # no DOI, arXiv landing
    _rec(url="https://ar5iv.org/abs/2101.00001"),  # a mirror
    _rec(provider="arxiv", url="https://example.org/x"),
])
def test_arxiv_only_abstract_is_never_grounding_and_is_listed_for_acquisition(arx):
    from tests.test_dar_pipeline_acceptance import _ground_cfg
    from vera import grounded_ask
    journal = _rec(doi="10.9999/synth.prd.2", url="https://doi.org/10.9999/synth.prd.2", title="Hadron masses of quarks")
    published_on_arxiv_url = _rec(doi="10.9999/synth.prd.3", url="https://arxiv.org/abs/2101.00003",
                                  title="Quark hadron journal version")
    out, unused = grounded_ask.retrieve_all("What are hadrons made of quarks?", _ground_cfg(),
                                            search=lambda q: [arx, journal, published_on_arxiv_url])
    assert {s.get("doi") for s in out} == {"10.9999/synth.prd.2", "10.9999/synth.prd.3"}  # journal DOIs ground
    assert [u["url"] for u in unused] == [arx["url"]] and "snippet" not in unused[0]


def test_is_arxiv_only_work():
    assert arxiv_portal.is_arxiv_only_work(_rec(doi="https://doi.org/10.48550/ARXIV.2101.00001")) is True
    assert arxiv_portal.is_arxiv_only_work(_rec(doi="10.1103/PhysRevD.1.1", url="https://arxiv.org/abs/2101.1")) is False
    assert arxiv_portal.is_arxiv_only_work(_rec(url="https://example.org/paper")) is False


# ------------------------------------------------------------------------------------------------ OD-1110-ARXIV-OPENALEX
class _OAResp:
    status_code = 200

    def __init__(self, doc, status=200):
        self._d, self.status_code = doc, status

    def json(self):
        return self._d


class _OAProvider:
    """Stands in for the shared /ask OpenAlexProvider: records metadata_get params and budget spends."""
    name = "openalex"

    def __init__(self, work, refs, status=200):
        self.work, self.refs, self.status, self.calls = work, refs, status, []

    def metadata_get(self, params, timeout_s):
        self.calls.append(dict(params))
        if self.status != 200:
            return _OAResp({}, self.status)
        if params["filter"].startswith("doi:"):
            return _OAResp({"results": [self.work] if self.work else []})
        return _OAResp({"results": [{"id": f"https://openalex.org/W{i}", "doi": d} for i, d in enumerate(self.refs)]})


WORK = {"id": "https://openalex.org/W1", "doi": f"https://doi.org/10.48550/arxiv.{AID}",
        "locations": [{"landing_page_url": f"https://arxiv.org/abs/{AID}"},
                      {"landing_page_url": "https://doi.org/10.9999/synth.cs.journal"}],
        "referenced_works": [f"https://openalex.org/W{i}" for i in range(20, 35)]}


def test_openalex_targeted_lookup_published_version_and_capped_references(monkeypatch):
    from vera import search_providers as sp
    monkeypatch.setattr(sp._QUOTA, "blocked", lambda now: False)
    p = _OAProvider(WORK, ["https://doi.org/10.9999/synth.ref.a", "https://doi.org/10.48550/arxiv.2101.1",
                           "https://doi.org/10.9999/synth.ref.b"])
    counts = {}
    own, refs = _REAL_OA(AID, 10, 5.0, counts, clock=lambda: 0.0, provider=p)
    assert own == ["10.9999/synth.cs.journal"] and refs == ["10.9999/synth.ref.a", "10.9999/synth.ref.b"]
    assert p.calls[0]["filter"] == f"doi:https://doi.org/10.48550/arxiv.{AID}" and "api_key" not in p.calls[0]
    assert set(p.calls[0]["select"].split(",")) == {"id", "doi", "locations", "referenced_works"}
    assert p.calls[1]["filter"].count("|") == 9 and p.calls[1]["select"] == "id,doi"  # capped at max_refs (10)
    assert len(p.calls) == 2 and counts["oa_calls"] == 2
    _REAL_OA(AID, 10, 5.0, counts, clock=lambda: 1.0, provider=p)
    assert len(p.calls) == 2  # cached


def test_openalex_quota_error_starts_the_shared_cooldown_and_unknown_work_yields_nothing(monkeypatch):
    from vera import search_providers as sp
    blocked = []
    monkeypatch.setattr(sp._QUOTA, "blocked", lambda now: False)
    monkeypatch.setattr(sp._QUOTA, "block", lambda now, s: blocked.append(s))
    assert _REAL_OA("2601.00009", 10, 5.0, {}, clock=lambda: 0.0, provider=_OAProvider(WORK, [], status=429)) is None
    assert blocked
    assert _REAL_OA("2601.00008", 10, 5.0, {}, clock=lambda: 0.0, provider=_OAProvider(None, [])) == ([], [])
    monkeypatch.setattr(sp._QUOTA, "blocked", lambda now: True)
    p = _OAProvider(WORK, [])
    assert _REAL_OA("2601.00007", 10, 5.0, {}, clock=lambda: 0.0, provider=p) is None and p.calls == []


def test_openalex_runs_only_when_inspire_does_not_know_the_record(inspire, monkeypatch):
    seed = _ddg()
    inspire["body"] = json.dumps({"hits": {"hits": []}}).encode()  # INSPIRE: unknown e-print
    monkeypatch.setattr(arxiv_portal, "openalex_lookup",
                        lambda aid, *a, **k: inspire["oa_calls"].append(aid) or (["10.9999/synth.cs.journal"], []))
    out = arxiv_portal.extend("q", [seed], A.cfg(), clock=lambda: 0.0)
    assert inspire["oa_calls"] == [AID] and len(inspire["calls"]) == 1
    new = out[1:]
    assert [(c["doi"], c["portal_kind"], c["portal_lookup"], c["portal_via"]) for c in new] == [
        ("10.9999/synth.cs.journal", "published_version", "openalex", f"via arXiv {AID}")]
    portal_inspire.reset_for_tests()
    arxiv_portal.reset_for_tests()
    inspire["oa_calls"].clear()
    inspire["body"] = _inspire_body()  # INSPIRE knows it: no OpenAlex call
    arxiv_portal.extend("q", [seed], A.cfg(), clock=lambda: 0.0)
    assert inspire["oa_calls"] == []
    off = A.cfg()
    off["arxiv_portal"]["openalex"]["enabled"] = False
    portal_inspire.reset_for_tests()
    inspire["body"] = json.dumps({"hits": {"hits": []}}).encode()
    arxiv_portal.extend("q", [seed], off, clock=lambda: 0.0)
    assert inspire["oa_calls"] == []


def test_generic_openalex_rw_adapter_stays_off():
    import json as _j
    from pathlib import Path
    c = _j.loads((Path(__file__).resolve().parents[1] / "config" / "reference_portals.json").read_text())
    assert c["adapters"]["openalex_rw"]["enabled"] is False


def test_metadata_get_uses_the_shared_budget_split_and_optional_key(monkeypatch):
    from vera import request_origin, search_providers as sp
    sent = []
    from tests.test_searchcaps_1110_openalex_key import CFG
    prov = sp.OpenAlexProvider(CFG["openalex"], transport=lambda u, p, h, t: sent.append(dict(p)) or _OAResp({"results": []}))
    monkeypatch.setenv("OPENALEX_API_KEY", "synthetic-test-key-000")
    prov.metadata_get({"filter": "doi:x", "select": "id"}, 1.0)
    assert sent[-1]["api_key"] == "synthetic-test-key-000" and prov._used == 1
    monkeypatch.delenv("OPENALEX_API_KEY")
    prov.metadata_get({"filter": "doi:x", "select": "id"}, 1.0)
    assert "api_key" not in sent[-1] and prov._used == 2
    prov._used = 10**6  # the seed share and the cap are the provider's own (_spend)
    with request_origin.scope(request_origin.SEED), pytest.raises(sp.ProviderUnavailable):
        prov.metadata_get({"filter": "doi:x"}, 1.0)
