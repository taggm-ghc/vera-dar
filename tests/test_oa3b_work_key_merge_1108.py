"""OA-3b (OA37-1108): merge_sources and search_and_fetch._Deduper with the canonical work key, shadow first.
OFFLINE, synthetic records."""
import copy
import json
import logging
import pathlib

import pytest

from vera import search_and_fetch, search_providers, work_key

PMC = "PMC7300021"
A = {"title": "SYNTHETIC alpha", "doi": "10.9999/oa3b.a", "pmcid": PMC, "provider": "europepmc"}
B = {"title": "SYNTHETIC alpha (web copy)", "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{PMC}/",
     "provider": "ddg"}  # same work, seen by URL only: the old keys (DOI, PMCID field) do not match it
C = {"title": "SYNTHETIC gamma", "doi": "10.9999/oa3b.c", "provider": "openalex"}


@pytest.fixture
def mode(monkeypatch):
    def set_(m):
        monkeypatch.setattr(work_key, "mode", lambda path=None: m)
    return set_


def _merge():
    return search_providers.merge_sources([copy.deepcopy(A)], [copy.deepcopy(B), copy.deepcopy(C)])


def test_shipped_mode_is_shadow():
    cfg = json.loads((pathlib.Path(work_key.__file__).resolve().parents[1] / "config" / "work_key.json").read_text())
    assert cfg["mode"] == "shadow" and work_key.mode() == "shadow"


def test_merge_sources_shadow_equals_off_and_logs_count(mode, caplog):
    mode("off")
    off = _merge()
    mode("shadow")
    with caplog.at_level(logging.INFO, logger="vera.work_key"):
        sh = _merge()
    assert sh == off and len(off) == 3
    msgs = [r.getMessage() for r in caplog.records if r.name == "vera.work_key"]
    assert msgs == ["work_key shadow site=merge_sources would_merge=1"]
    assert PMC not in msgs[0] and "10.9999" not in msgs[0]  # counts only


def test_merge_sources_on_merges_same_work(mode):
    mode("on")
    out = _merge()
    assert [c["title"] for c in out] == ["SYNTHETIC alpha", "SYNTHETIC gamma"]
    assert len(out[0]["found_by"]) == 2  # the losing copy's discovery (acq_source label) is kept


def test_merge_sources_on_keeps_distinct_works(mode):
    mode("on")
    out = search_providers.merge_sources([{"doi": "10.9999/x1", "title": "one"}], [{"doi": "10.9999/x2", "title": "two"}])
    assert len(out) == 2


def _dd_run():
    dd = search_and_fetch._Deduper()
    res = []
    for i, it in enumerate((A, B, C)):
        hit = dd.find(it)
        res.append(hit)
        if hit is None:
            dd.add(it, i)
    return res


def test_deduper_shadow_equals_off(mode, caplog):
    mode("off")
    off = _dd_run()
    mode("shadow")
    with caplog.at_level(logging.INFO, logger="vera.work_key"):
        sh = _dd_run()
    assert sh == off == [None, None, None]
    assert [r.getMessage() for r in caplog.records if r.name == "vera.work_key"] == [
        "work_key shadow site=deduper would_merge=1"]


def test_deduper_on_matches_by_work_key(mode):
    mode("on")
    assert _dd_run() == [None, (0, "wk"), None]


def test_shadow_leaves_inputs_and_dedup_index_identical_to_off(mode):
    def run(m):
        mode(m)
        lists = ([copy.deepcopy(A)], [copy.deepcopy(B), copy.deepcopy(C), {"bad": ("x", 1), "url": 5}])
        out = search_providers.merge_sources(*lists)
        dd = search_and_fetch._Deduper()
        for i, it in enumerate((A, B, C)):
            if dd.find(it) is None:
                dd.add(it, i)
        return repr(out), repr(lists), dict(dd.hard)
    assert run("shadow") == run("off")


def test_on_merges_same_normalised_url(mode):
    """Intended in "on": the same normalised URL is the same record (utm and trailing slash dropped)."""
    mode("on")
    out = search_providers.merge_sources([{"url": "https://ex.org/p/", "title": "a"}],
                                         [{"url": "https://ex.org/p?utm_source=x", "title": "b"}])
    assert len(out) == 1
