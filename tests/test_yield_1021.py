"""YIELD-1021: licence/OA pre-filter before the top-N cut, Europe PMC as a second acquisition search, DuckDuckGo Lite
first (owner 2026-10-21) with scholarly-identifier mapping, call caps, failure isolation, reading-list providers.
SYNTHETIC fixtures only: DOIs are 10.9999/..., PMC ids are fake, titles are invented; no network, DB or model."""
from __future__ import annotations

import json

import pytest

from tests import acq_support as A
from vera import auto_acquire as aa
from vera import search_providers as sp

Q = "How do synthetic zqx widgets ferment in a test vat?"


class _R:
    def __init__(self, body, status=200, headers=None):
        self.text = body if isinstance(body, str) else json.dumps(body)
        self.status_code, self.headers = status, headers or {}
        self._b = self.text.encode()

    def iter_content(self, n):
        yield self._b

    def close(self):
        pass


def _oa_cand(i, **oa):
    c = {"doi": f"10.9999/yield.{i}", "url": f"https://doi.org/10.9999/yield.{i}", "title": f"Synthetic widget {i}",
         "snippet": "synthetic widgets ferment in a vat"}
    if oa:
        c["oa"] = oa
    return c


def _rank(cands, cfg=None):
    codes = {}

    def count(c):
        codes[c] = codes.get(c, 0) + 1
    out = aa.rank_candidates(Q, cands, cfg or A.cfg(), set(), aa._fd_cfg(), count)
    return out, codes


# ---- 1. pre-filter -----------------------------------------------------------------------------------------------
def test_prefilter_drops_nd_and_non_oa_keeps_unknown_before_the_cut():
    nd = [_oa_cand(i, licence="cc-by-nc-nd", has_oa_location=True, is_oa=True) for i in range(6)]
    closed = [_oa_cand(10 + i, licence=None, has_oa_location=False, is_oa=False) for i in range(3)]
    keep = [_oa_cand(20, licence="cc-by", has_oa_location=True, is_oa=True),
            _oa_cand(21, licence="cc-by-nc", has_oa_location=True, is_oa=True),  # plain NC is accepted
            _oa_cand(22, licence="other-oa", has_oa_location=True, is_oa=True),  # unknown licence stays in
            _oa_cand(23)]  # no OA metadata at all: unknown, stays in
    ranked, codes = _rank(nd + closed + keep)
    dois = {d for _s, d, _c, _i in ranked}
    assert dois == {"10.9999/yield.20", "10.9999/yield.21", "10.9999/yield.22", "10.9999/yield.23"}
    assert codes.get("R-ACQ-NOLIC") == 6 and codes.get("R-ACQ-NOTOA") == 3


def test_openalex_oa_info_from_metadata_only():
    pmc = {"source": {"id": "https://openalex.org/S2764455111"}, "license": "cc-by-nd", "is_oa": True,
           "landing_page_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC0000001"}
    w = {"open_access": {"is_oa": True}, "best_oa_location": {"license": "cc-by", "is_oa": True}, "locations": [pmc],
         "ids": {}}
    info = sp.openalex_oa_info(w)
    assert info["licence"] == "cc-by-nd" and info["pmcid"] == "PMC0000001"  # the PMC location's licence wins
    assert sp.acq_prefilter({"oa": info}) == "nd"
    assert sp.openalex_oa_info({"id": "x", "doi": None}) is None  # /ask select: no OA fields -> no "oa" key
    none = sp.openalex_oa_info({"open_access": {"is_oa": False}, "best_oa_location": None, "locations": []})
    assert sp.acq_prefilter({"oa": none}) == "not_oa"


# ---- reading list unchanged ----------------------------------------------------------------------------------------
class _Tx:
    def __init__(self, works):
        self.works, self.params = works, []

    def __call__(self, url, params, headers, timeout):
        self.params.append(dict(params))
        return _R({"results": self.works})


def _oa_provider(tx):
    s = {"endpoint": "https://openalex.invalid/works", "max_response_bytes": 1_000_000, "snippet_max_chars": 300,
         "min_interval_s": 0, "daily_search_budget": 99, "from_publication_date": "1900-01-01",
         "to_publication_date": "2100-01-01"}
    return sp.OpenAlexProvider(s, transport=tx)


def test_ask_reading_list_select_and_records_unchanged_oa_fields_only_inside_acquisition():
    work = {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.9999/yield.1", "title": "Synthetic widget",
            "open_access": {"is_oa": True}, "best_oa_location": {"license": "cc-by"}}
    tx = _Tx([work])
    p = _oa_provider(tx)
    rec = p.parse(p.call(Q, 5, 5), 5)[0]
    assert tx.params[-1]["select"] == sp.OPENALEX_SELECT  # /ask: unchanged select
    tok = sp.OPENALEX_OA_FIELDS.set(True)
    try:
        p.call(Q, 5, 5)
    finally:
        sp.OPENALEX_OA_FIELDS.reset(tok)
    assert tx.params[-1]["select"] == sp.OPENALEX_SELECT + "," + sp.OPENALEX_OA_SELECT_EXTRA
    # without the extra select the real API omits the fields; parse adds "oa" only when they are present
    del work["open_access"], work["best_oa_location"]
    rec2 = p.parse(_R({"results": [work]}), 5)[0]
    assert "oa" not in rec2 and "oa" in rec


# ---- 2. merge / dedupe -------------------------------------------------------------------------------------------
def test_merge_dedupes_by_doi_or_pmcid_and_keeps_first_place():
    a = _oa_cand(1)
    b = {"doi": None, "pmcid": "PMC0000009", "url": "https://europepmc.org/article/PMC/PMC0000009", "title": "s",
         "provider": "europepmc", "oa": {"pmcid": "PMC0000009", "epmc_ft": True}}
    a2 = {**_oa_cand(2), "oa": {"pmcid": "PMC0000009"}}
    e1 = {"doi": "10.9999/YIELD.1", "pmcid": "PMC0000001", "provider": "europepmc", "url": "u", "title": "t",
          "oa": {"epmc_ft": True, "pmcid": "PMC0000001"}}
    out = sp.merge_sources([a, a2], [e1, b])
    assert [c["doi"] for c in out] == ["10.9999/yield.1", "10.9999/yield.2"]
    assert out[0]["oa"]["epmc_ft"] is True and out[1]["oa"]["epmc_ft"] is True


# ---- Europe PMC parse ----------------------------------------------------------------------------------------------
def test_europepmc_search_query_and_parse():
    seen = {}

    def tx(url, params, headers, timeout):
        seen.update(params)
        return _R({"resultList": {"result": [
            {"doi": "10.9999/yield.5", "pmcid": "PMC0000005", "isOpenAccess": "Y", "source": "MED", "title": "Syn"},
            {"doi": "10.9999/yield.6", "pmcid": None, "isOpenAccess": "Y", "source": "MED", "title": "NoPMC"},
            {"doi": "10.9999/yield.7", "pmcid": "PMC0000007", "isOpenAccess": "Y", "source": "PPR", "title": "Pre"}]}})
    ecfg = A.cfg()["search"]["europepmc"]
    out = sp.europepmc_search(Q, 10, 5, ecfg, transport=tx)
    assert [c["doi"] for c in out] == ["10.9999/yield.5"]
    assert "OPEN_ACCESS:y AND HAS_FT:y" in seen["query"] and seen["resultType"] == "lite" and seen["pageSize"] == 10


# ---- DDG mapping -------------------------------------------------------------------------------------------------
def test_ddg_identifier_mapping():
    assert sp.ddg_identifier("https://onlinelibrary.wiley.com/doi/10.9999/yield.42/full") == {
        "kind": "doi", "value": "10.9999/yield.42"}
    assert sp.ddg_identifier("https://pmc.ncbi.nlm.nih.gov/articles/PMC0000042/") == {"kind": "pmcid",
                                                                                      "value": "PMC0000042"}
    assert sp.ddg_identifier("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC0000043") == {"kind": "pmcid",
                                                                                         "value": "PMC0000043"}
    assert sp.ddg_identifier("https://pubmed.ncbi.nlm.nih.gov/99999999/") == {"kind": "pmid", "value": "99999999"}
    assert sp.ddg_identifier("https://europepmc.org/article/MED/99999998") == {"kind": "pmid", "value": "99999998"}
    assert sp.ddg_identifier("https://arxiv.org/pdf/9999.00001v2") == {"kind": "arxiv", "value": "9999.00001"}
    assert sp.ddg_identifier("https://example.org/blog/widgets") is None


def _ddg_results():
    return [{"url": "https://example.org/blog/widgets", "title": "Blog", "discovery_group": "news"},  # held
            {"url": "https://portal.example.org/article/77", "title": "Portal", "discovery_group": "portals"},
            {"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC0000044/", "title": "PMC", "discovery_group": "portals"},
            {"url": "https://journals.example.org/doi/10.9999/yield.45", "title": "DOI", "discovery_group": "portals"},
            {"url": "https://arxiv.org/abs/9999.00002", "title": "Arx", "discovery_group": "portals"}]


def _land_get(url, **kw):
    return _R('<html><head><meta name="citation_doi" content="10.9999/yield.46"></head></html>')


def _lookup_tx(url, params, headers, timeout):
    assert "PMCID:PMC0000044" in params["query"]
    return _R({"resultList": {"result": [{"doi": "10.9999/YIELD.47", "pmcid": "PMC0000044"}]}})


def test_ddg_to_candidates_maps_doi_pmc_arxiv_landing_and_holds_unclear():
    import time
    out, counts = sp.ddg_to_candidates(_ddg_results(), {"search_url": "https://epmc.invalid/s"},
                                       {"max_landing_fetches": 3, "landing_groups": ["portals"]},
                                       deadline=time.monotonic() + 30, clock=time.monotonic, calls_left=1,
                                       transport=_lookup_tx, get=_land_get, url_check=lambda u: (True, ""))
    dois = sorted(c["doi"] for c in out)
    assert dois == ["10.48550/arxiv.9999.00002", "10.9999/yield.45", "10.9999/yield.46", "10.9999/yield.47"]
    assert counts["ddg_held"] == 1 and counts["ddg_landing_get"] == 1
    assert all(c["acq_source"] == "ddg" for c in out)


def test_ddg_pmc_without_a_call_left_is_held():
    import time
    out, counts = sp.ddg_to_candidates(_ddg_results()[2:3], {"search_url": "x"}, {}, deadline=time.monotonic() + 30,
                                       clock=time.monotonic, calls_left=0, transport=_lookup_tx)
    assert out == [] and counts["ddg_held"] == 1


def test_landing_page_ssrf_guard_blocks_private_hosts():
    calls = []
    assert sp.landing_doi("http://10.0.0.1/x", 1, 1000, get=lambda *a, **k: calls.append(1),
                          url_check=lambda u: (False, "private")) is None and not calls


def test_ddg_candidates_rank_first_and_arxiv_last():
    ddg = {**_oa_cand(50), "title": "unrelated", "snippet": "", "acq_source": "ddg", "ddg_id_kind": "doi"}
    arx = {**_oa_cand(51), "acq_source": "ddg", "ddg_id_kind": "arxiv"}
    others = [_oa_cand(i) for i in range(52, 56)]
    ranked, _ = _rank(sp.merge_sources([ddg, arx], others))
    order = [d for _s, d, _c, _i in ranked]
    assert order[0] == "10.9999/yield.50" and order[-1] == "10.9999/yield.51"


# ---- caps and failure isolation -------------------------------------------------------------------------------------
def _wrap(cap, *, base=None, ddg=None, epmc=None):
    cfg = A.cfg(max_search_calls_per_job=cap)
    calls = {"base": 0, "ddg": 0, "epmc": 0}

    def b(q, n, t):
        calls["base"] += 1
        if base == "fail":
            raise RuntimeError("openalex down")
        return [_oa_cand(1)]

    def d(q, n, t):
        calls["ddg"] += 1
        if ddg == "fail":
            raise RuntimeError("ddg down")
        return [{"url": "https://journals.example.org/doi/10.9999/yield.60", "title": "D", "discovery_group": "news"}]

    def tx(url, params, headers, timeout):
        calls["epmc"] += 1
        if epmc == "fail":
            raise RuntimeError("epmc down")
        return _R({"resultList": {"result": [{"doi": "10.9999/yield.61", "pmcid": "PMC0000061", "isOpenAccess": "Y",
                                              "source": "MED", "title": "E"}]}})
    fn = sp.acquisition_search(b, cfg, transport=tx, terms_fn=lambda: (None, None), ddg_search=d)
    return fn, calls


def test_call_caps_respected():
    fn, calls = _wrap(2)
    assert [c["doi"] for c in fn(Q, 10, 5)] == ["10.9999/yield.1"] and calls == {"base": 1, "ddg": 0, "epmc": 0}
    fn, calls = _wrap(3)
    assert [c["doi"] for c in fn(Q, 10, 5)] == ["10.9999/yield.60", "10.9999/yield.1"]
    assert calls == {"base": 1, "ddg": 1, "epmc": 0}
    fn, calls = _wrap(5)
    assert [c["doi"] for c in fn(Q, 10, 5)] == ["10.9999/yield.60", "10.9999/yield.1", "10.9999/yield.61"]
    assert calls == {"base": 1, "ddg": 1, "epmc": 1}


@pytest.mark.parametrize("which,expect", [
    ("base", ["10.9999/yield.60", "10.9999/yield.61"]),
    ("ddg", ["10.9999/yield.1", "10.9999/yield.61"]),
    ("epmc", ["10.9999/yield.60", "10.9999/yield.1"])])
def test_one_provider_failing_returns_the_others(which, expect):
    fn, _ = _wrap(5, **{which: "fail"})
    assert [c["doi"] for c in fn(Q, 10, 5)] == expect


def test_all_failing_raises():
    fn, _ = _wrap(5, base="fail", ddg="fail", epmc="fail")
    with pytest.raises(RuntimeError):
        fn(Q, 10, 5)


def test_config_caps_and_providers():
    c = A.cfg()
    assert c["bounds"]["max_search_calls_per_job"] == 5
    assert c["search"]["europepmc"]["enabled"] is True and c["search"]["ddg_first"]["enabled"] is True


# ---- reading list includes DDG links -------------------------------------------------------------------------------
def test_offtopic_reading_list_includes_duckduckgo_lite():
    from vera.scope_router import offtopic_providers
    assert "duckduckgo_lite" in offtopic_providers()


# ---- OpenAlex 429 / quota: unavailable for the cooldown, others still answer, counted -------------------------------
@pytest.fixture
def _quota_reset():
    sp._QUOTA.reset()
    yield
    sp._QUOTA.reset()


def test_openalex_429_marks_it_unavailable_and_other_sources_still_produce_candidates(_quota_reset, caplog):
    import logging
    cfg = A.cfg(max_search_calls_per_job=5)
    calls = {"base": 0}

    def b(q, n, t):
        calls["base"] += 1
        raise RuntimeError("openalex: attempt 1: HTTP 429 'Rate limit exceeded'")

    def d(q, n, t):
        return [{"url": "https://journals.example.org/doi/10.9999/yield.70", "title": "D", "discovery_group": "news"}]

    def tx(url, params, headers, timeout):
        return _R({"resultList": {"result": [{"doi": "10.9999/yield.71", "pmcid": "PMC0000071", "isOpenAccess": "Y",
                                              "source": "MED", "title": "E"}]}})
    fn = sp.acquisition_search(b, cfg, transport=tx, terms_fn=lambda: (None, None), ddg_search=d)
    with caplog.at_level(logging.INFO, logger="vera.auto_acquire"):
        first = fn(Q, 10, 5)
        second = fn(Q, 10, 5)
    assert [c["doi"] for c in first] == [c["doi"] for c in second] == ["10.9999/yield.70", "10.9999/yield.71"]
    assert calls["base"] == 1  # no second OpenAlex call during the cooldown (no retry storm)
    text = caplog.text
    assert "openalex_429=1" in text and "openalex_skipped_quota=1" in text


def test_openalex_api_key_is_optional_sent_only_from_env_and_redacted(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    tx = _Tx([])
    p = _oa_provider(tx)
    p.call(Q, 5, 5)
    assert "api_key" not in tx.params[-1]
    monkeypatch.setenv("OPENALEX_API_KEY", "synthkey12345678")
    p.call(Q, 5, 5)
    assert tx.params[-1]["api_key"] == "synthkey12345678"

    def boom(url, params, headers, timeout):
        raise RuntimeError(f"connection failed for {url}?api_key={params['api_key']}")
    p2 = _oa_provider(boom)
    with pytest.raises(RuntimeError) as ei:
        p2.call(Q, 5, 5)
    assert "synthkey12345678" not in str(ei.value) and "[redacted]" in str(ei.value)


def test_landing_doi_respects_the_deadline():
    calls = []
    t = [100.0]
    assert sp.landing_doi("https://portal.example.org/a", 5, 1000, get=lambda *a, **k: calls.append(1),
                          url_check=lambda u: (True, ""), deadline=100.1, clock=lambda: t[0]) is None and not calls
