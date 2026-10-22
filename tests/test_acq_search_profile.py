"""ACQ-SEARCH-1021, OFFLINE: the acquisition-only OpenAlex profile (PMC-held filter + AND of distinctive terms).
Recorded fixture: tests/fixtures/openalex_boxing_mma_pmc.json (live OpenAlex response, 2026-10-21). No network."""
import json
from pathlib import Path

from vera import search_providers as sp
from vera import auto_acquire
from tests import acq_support as A

ROOT = Path(__file__).resolve().parent.parent
Q = "What are key differences between boxing and mma?"
SEARCH = json.loads((ROOT / "config" / "vera_eval_run.json").read_text())["search"]
QT = SEARCH["query_terms"]
PROFILE = A.cfg()["search"]["openalex_profile"]
FIX = (ROOT / "tests" / "fixtures" / "openalex_boxing_mma_pmc.json").read_text()


class _HTTP:
    def __init__(self, text):
        self.status_code, self.text, self.headers = 200, text, {"Content-Type": "application/json"}


def _prov(calls):
    def tx(url, params, headers, timeout):
        calls.append(dict(params))
        return _HTTP(FIX)
    return sp.OpenAlexProvider(SEARCH["openalex"], transport=tx, query_terms=QT)


def test_default_query_is_unchanged_all_or():
    assert sp.build_openalex_search(Q, QT) == ("(key OR differences OR boxing OR mma)", "extra_only")


def test_profile_ands_distinctive_terms_and_drops_generic():
    assert sp.build_openalex_search(Q, QT, profile=PROFILE)[0] == "boxing AND mma"


def test_profile_caps_terms_and_falls_back_to_all_when_all_generic():
    p = {"and_extra_max": 2, "generic_terms": []}
    assert sp.build_openalex_search("alpha beta gamma delta", QT, profile=p)[0].count(" AND ") == 1
    allg = {"and_extra_max": 4, "generic_terms": ["key", "differences"]}
    assert sp.build_openalex_search("key differences", QT, profile=allg)[0] == "key AND differences"


def test_core_phrase_question_query_is_not_rewritten():
    q = "Does github copilot improve developer productivity?"
    assert sp.build_openalex_search(q, QT, profile=PROFILE) == sp.build_openalex_search(q, QT)


def test_provider_without_profile_sends_no_extra_filter():
    calls = []
    _prov(calls).call(Q, 10, 5)
    assert "S2764455111" not in calls[0]["filter"] and "OR" in calls[0]["search"]


def test_make_search_applies_profile_only_inside_the_acquisition_search_and_resets():
    calls = []
    prov = _prov(calls)
    out = auto_acquire.make_search(A.cfg(), chain_fn=lambda: [prov])(Q, 10, 5)
    assert len(calls) == 1 and calls[0]["search"] == "boxing AND mma", calls
    assert calls[0]["filter"].endswith("locations.source.id:S2764455111"), calls
    assert calls[0]["filter"].startswith("from_publication_date:2023-01-01,to_publication_date:"), calls  # date gate kept
    assert calls[0]["per_page"] == 10  # cap unchanged
    assert [c["doi"] for c in out][:1] == ["10.1097/iop.0000000000002510"] and len(out) == 4
    assert sp.OPENALEX_PROFILE.get() is None
    assert prov.last_query is None
    _prov(calls).call(Q, 10, 5)  # a later /ask call is back to the default
    assert "OR" in calls[-1]["search"] and "S2764455111" not in calls[-1]["filter"]


def test_removing_the_config_block_restores_previous_behaviour():
    c = A.cfg()
    c["search"].pop("openalex_profile")
    calls = []
    auto_acquire.make_search(c, chain_fn=lambda: [_prov(calls)])(Q, 10, 5)
    assert "OR" in calls[0]["search"] and "S2764455111" not in calls[0]["filter"]
