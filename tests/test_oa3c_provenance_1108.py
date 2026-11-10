"""OA-3c (OA37-1108): discovery events and preprint/published link in the provenance JSON (additive, no DDL), active
only with config/work_key.json mode "on" (shipped "shadow": nothing changes). OFFLINE."""
import copy
import types

import pytest

from tests import acq_support as A
from tests import arxiv_synth as X
from vera import arxiv_fulltext, auto_acquire, search_providers, work_key

# ARXIV-PORTAL-1110: these tests pin the DORMANT arXiv evidence route (closed in code: arxiv_fulltext.PORTAL_ONLY);
# the fixture reopens it per test. Portal-only behaviour: tests/test_arxiv_portal_1110.py.
pytestmark = pytest.mark.usefixtures("legacy_arxiv_route")

PMC = "PMC7300031"
E1 = {"title": "SYNTHETIC alpha", "doi": "10.9999/oa3c.a", "pmcid": PMC, "provider": "europepmc",
      "acq_source": "europepmc", "url": "https://doi.org/10.9999/oa3c.a"}
E2 = {"title": "SYNTHETIC alpha copy", "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{PMC}/", "provider": "ddg",
      "acq_source": "ddg"}


@pytest.fixture
def mode(monkeypatch):
    def set_(m):
        monkeypatch.setattr(work_key, "mode", lambda path=None: m)
    return set_


def test_on_keeps_every_discovery_event(mode):
    mode("on")
    out = search_providers.merge_sources([copy.deepcopy(E1)], [copy.deepcopy(E2), copy.deepcopy(E2)])
    assert len(out) == 1
    ev = out[0]["discovered_by"]
    assert [e["provider"] for e in ev] == ["epmc", "ddg"]  # search_providers.acq_source labels  # the identical repeat is kept once
    assert ev[0]["record_id"] == "doi:10.9999/oa3c.a" and ev[1]["record_id"] == f"pmcid:{PMC}"
    assert ev[1]["url"] == E2["url"]


@pytest.mark.parametrize("m", ["off", "shadow"])
def test_off_and_shadow_add_no_field(mode, m):
    mode(m)
    out = search_providers.merge_sources([copy.deepcopy(E1)], [copy.deepcopy(E2)])
    assert all("discovered_by" not in c for c in out)


def _plan():
    return {"source": {"provenance": {"acquisition": {"acquired_at": "2026-11-08T08:00:00+00:00"}}}}


def test_annotate_provenance_fields():
    plan = _plan()
    cand = {"arxiv": {"id": "2605.23989"}, "doi": "10.1234/journal.5", "url": X.ABS_URL, "acq_source": "openalex",
            "discovered_by": [{"provider": "openalex", "record_id": "doi:10.1234/journal.5", "url": X.ABS_URL}]}
    work_key.annotate_provenance(plan, cand)
    p = plan["source"]["provenance"]["acquisition"]
    assert set(plan["source"]["provenance"]) == {"acquisition"}  # nothing at the top level (reader compares it)
    assert p["work_key"] == "doi:10.1234/journal.5"
    assert p["discovered_by"] == [{"provider": "openalex", "record_id": "doi:10.1234/journal.5", "url": X.ABS_URL,
                                   "seen_at": "2026-11-08T08:00:00+00:00"}]
    assert p["related_versions"] == {"preprint": "arxiv:2605.23989", "published": "doi:10.1234/journal.5"}


def test_annotate_provenance_single_event_and_no_link():
    plan = _plan()
    work_key.annotate_provenance(plan, {"doi": "10.9999/z", "acq_source": "openalex"})
    p = plan["source"]["provenance"]["acquisition"]
    assert p["discovered_by"][0]["record_id"] == "doi:10.9999/z" and "related_versions" not in p
    work_key.annotate_provenance({"source": {}}, {"doi": "10.9999/z"})  # no provenance: nothing, no error
    work_key.annotate_provenance(_plan(), None)
    bad = {"source": {"provenance": {"acquisition": "x"}}}
    work_key.annotate_provenance(bad, {"doi": "10.9999/z"})
    assert bad == {"source": {"provenance": {"acquisition": "x"}}}


@pytest.mark.parametrize("m,annotated", [("on", True), ("shadow", False), ("off", False)])
def test_evaluate_candidate_annotates_only_when_on(monkeypatch, mode, m, annotated):
    mode(m)
    monkeypatch.setattr(arxiv_fulltext, "evaluate", lambda *a: (_plan(), None))
    deps = types.SimpleNamespace(lookup=types.SimpleNamespace(known=lambda *a, **k: False))
    cand = {"url": X.ABS_URL, "doi": X.DOI, "identifier": f"arXiv:{X.AID}{X.VER}", "title": "SYNTHETIC preprint",
            "acq_source": "openalex"}
    plan, _ = auto_acquire.evaluate_candidate(auto_acquire.Job(A.Q_EPI, "test"), 0.5, X.DOI, cand, None, A.cfg(), deps,
                                              {}, {}, clock=lambda: 0.0)
    p = plan["source"]["provenance"]["acquisition"]
    assert ("discovered_by" in p) is annotated
    if annotated:
        assert p["work_key"] == f"arxiv:{X.AID}" and p["discovered_by"][0]["provider"] == "openalex"


def test_new_keys_stay_inside_reader_ignored_acquisition_record():
    """Verifier finding (OA37-1108-VERIFY-3c): fulldoc_db's acquired-path reader compares top-level provenance minus
    ACQ_EXTRA_KEYS with the re-derived record, so OA-3c writes only under "acquisition"."""
    from vera import fulldoc_db
    assert "acquisition" in fulldoc_db.ACQ_EXTRA_KEYS
    assert not {"work_key", "discovered_by", "related_versions"} & set(fulldoc_db.ACQ_EXTRA_KEYS)


# ---------------------------------------------------------------------------------------------- write -> read round trip
def _acquire_rows(mode_set, m):
    from tests import test_fulldoc_e2e as e2e
    mode_set(m)
    web, _ = A.epi_setup()
    store = A.Store()
    auto_acquire.run_job(A.job(e2e.Q_EPI), A.cfg(), A.deps([A.cand("10.9999/acq.epi4")], web, store=store),
                         clock=A.clock_at(), env=A.ENV_ON)
    assert store.plans, "setup: no acquired plan was stored"
    return A.rows_from_plans(store.plans)


def _load(src, spans):
    from vera import fulldoc_db
    from vera import fulldoc_corpus as fc
    return fulldoc_db.load_acquired_rows(copy.deepcopy(src), copy.deepcopy(spans), fc.load_config(), A.cfg())


@pytest.mark.parametrize("m", ["on", "shadow"])
def test_round_trip_acquired_row_is_served(mode, m):
    """Verifier finding (OA37-1108-VERIFY-3c): the stored row with the OA-3c keys must still load through the
    acquired-path reader (which compares provenance with the re-derived record)."""
    import json
    src, spans = _acquire_rows(mode, m)
    acq = json.loads(src[0]["provenance"])["acquisition"]
    assert ("work_key" in acq) is (m == "on")
    if m == "on":
        assert acq["work_key"] == "doi:10.9999/acq.epi4" and acq["discovered_by"][0]["record_id"] == acq["work_key"]
        assert acq["discovered_by"][0]["seen_at"] == acq["acquired_at"]
    assert _load(src, spans).acquired


def test_round_trip_would_refuse_top_level_keys(mode):
    """Sensitivity check: the same keys at the TOP level of provenance are refused by the reader, so the test above
    proves the placement, not just the presence."""
    import json
    src, spans = _acquire_rows(mode, "on")
    prov = json.loads(src[0]["provenance"])
    prov["discovered_by"] = prov["acquisition"]["discovered_by"]
    src[0]["provenance"] = json.dumps(prov)
    assert not _load(src, spans).acquired
