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
