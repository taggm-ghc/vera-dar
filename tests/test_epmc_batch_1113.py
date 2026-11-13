"""EPMC-BATCH-1113: option (d) of work/NOROUTE-CODE-1112/PLAN.md (W2.3). One paced Europe PMC lite batch request per
round for DOI-only candidates; a DOI with no qualifying open-access row in a USABLE answer ends R-ACQ-NOROUTE with no
per-DOI request; everything else (hits, any failed envelope) resolves per DOI as before. Flag resolve.batch_precheck.mode
ships "off".

KNOWN LIMIT: the transport here is a fake. It returns what the test author believes Europe PMC returns, so these tests
cannot show zero false skips on real data; that is the separate real-data paired comparison (plan W2.5 M0), which gates
enabling the flag. Offline: synthetic fixtures, example DOIs, no network, no DB."""
import json
import re
from urllib.parse import unquote_plus

import pytest

from tests import acq_support as A
from tests import fulltext_synth
from tests.test_acq_discovered import PUB, Router, _direct, _pub_is_original, cfg_on, oa_src, run, step_b_web  # noqa: F401
from vera import acq_config, auto_acquire, inline_acquire, rate_limit, stream_ctx

HIT = "10.9999/acq.epi4"
HIT_PMC = "PMC1234567"
MISS = ["10.1007/jhep07(2024)184", "10.9999/miss.two", "10.9999/miss.three"]
NOROUTE = "R-ACQ-NOROUTE"


class BatchWeb(Router):
    """Router plus a lite batch answer built from `records` (DOI -> core record), overridable by `batch_resp`."""

    def __init__(self, records, jats, routes=None, batch_resp=None, batch_rows=None):
        super().__init__(records, jats, routes)
        self.batch_resp, self.batch_rows, self.batch_urls = batch_resp, batch_rows, []

    def __call__(self, url, *a, **kw):
        if "/rest/search" in url and "resultType=lite" in url:
            self.batch_urls.append(url)
            self.urls.append(url)
            if isinstance(self.batch_resp, Exception):
                raise self.batch_resp
            if self.batch_resp is not None:
                return self.batch_resp
            if self.batch_rows is not None:
                rows = self.batch_rows
            else:
                q = unquote_plus(url)
                rows = [{"id": "1", "source": r["source"], "pmcid": r["pmcid"], "doi": r["doi"],
                         "isOpenAccess": r["isOpenAccess"]}
                        for d, r in self.records.items() if f'doi:"{d.lower()}"' in q.lower()]
            return A.Resp(body=json.dumps({"hitCount": len(rows), "resultList": {"result": rows}}).encode())
        return super().__call__(url, *a, **kw)

    def core(self):
        return [u for u in self.urls if "/rest/search" in u and "resultType=core" in u]


def web_for(hit=HIT, **kw):
    raw = fulltext_synth.jats("acq_epi", "epi", hit)
    return BatchWeb({hit: A.epmc_record(hit, pmcid=HIT_PMC)}, {HIT_PMC: raw}, **kw)


def cfg_batch(mode="on", step_b=False, **bp):
    c = cfg_on(step_b=step_b)
    c["resolve"]["batch_precheck"].update({"mode": mode, **bp})
    return c


def sources(dois):
    return [oa_src(doi=d, wid=f"https://openalex.org/W9{i}") for i, d in enumerate(dois)]


def by_doi(o):
    return {e.get("doi") or e.get("record_id"): e for e in o.public()["discovered"]}


def reasons(o):
    return sorted(str(e.get("reason")) for e in o.public()["discovered"])


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(inline_acquire, "ephemeral_corpus", lambda *a, **k: object())
    yield
    stream_ctx.CTX.set(None)


# ------------------------------------------------------------------------------------------------ config
def test_shipped_flag_is_off_and_modes_parse():
    c = acq_config.load_config()
    assert c["resolve"]["batch_precheck"]["mode"] == "off"
    assert auto_acquire.batch_precheck_mode(c) == "off"
    for v, want in (("on", "on"), ("shadow", "shadow"), ("ON", "on"), ("yes", "off"), (None, "off"), (True, "off")):
        c2 = {"resolve": {"batch_precheck": {"mode": v}}}
        assert auto_acquire.batch_precheck_mode(c2) == want
    assert auto_acquire.batch_precheck_mode({}) == "off"


def test_off_sends_no_batch_and_every_doi_resolves_per_doi():
    web = web_for()
    o, _s = run(cfg_batch("off"), sources([HIT] + MISS), web)
    assert web.batch_urls == [] and len(web.core()) == 4
    assert o.stored == 1


# ------------------------------------------------------------------------------------------------ the skip (through the real pool)
def test_on_skips_misses_resolves_the_hit_and_one_hit_does_not_void_the_batch():
    web = web_for()
    o, store = run(cfg_batch("on"), sources([HIT] + MISS), web)
    assert len(web.batch_urls) == 1 and len(web.core()) == 1 and HIT in web.core()[0].replace("%2F", "/")
    assert o.stored == 1 and len(store.plans) == 1
    got = [e for e in o.public()["discovered"] if e["outcome"] == "rejected"]
    assert len(got) == 3 and {e["reason"] for e in got} == {NOROUTE}
    q = unquote_plus(web.batch_urls[0])
    for d in [HIT] + MISS:
        assert f'DOI:"{d}"' in q
    assert "resultType=lite" in web.batch_urls[0] and "pageSize=100" in web.batch_urls[0]


def test_on_batch_is_gated_like_resolve_oa_robots_first_and_no_direct_request():
    web = web_for()
    run(cfg_batch("on"), sources([HIT] + MISS), web)
    first_ebi = next(u for u in web.urls if "www.ebi.ac.uk" in u)
    assert first_ebi.endswith("/robots.txt")  # the same hop gate: robots.txt before the first Europe PMC request


# ------------------------------------------------------------------------------------------------ fallback: nothing skipped
BAD = {
    "hitcount_missing": lambda: A.Resp(body=b'{"resultList": {"result": []}}'),
    "hitcount_str": lambda: A.Resp(body=b'{"hitCount": "0", "resultList": {"result": []}}'),
    "hitcount_bool": lambda: A.Resp(body=b'{"hitCount": true, "resultList": {"result": [{"doi": "x"}]}}'),
    "hitcount_mismatch": lambda: A.Resp(body=b'{"hitCount": 5, "resultList": {"result": []}}'),
    "hitcount_over_pagesize": lambda: A.Resp(body=json.dumps(
        {"hitCount": 101, "resultList": {"result": [{"doi": "d"}] * 101}}).encode()),
    "not_json": lambda: A.Resp(body=b"<html>nope</html>"),
    "duplicate_key": lambda: A.Resp(body=b'{"hitCount": 0, "hitCount": 0, "resultList": {"result": []}}'),
    "result_not_list": lambda: A.Resp(body=b'{"hitCount": 0, "resultList": {"result": {}}}'),
    "resultlist_missing": lambda: A.Resp(body=b'{"hitCount": 0}'),
    "row_not_dict": lambda: A.Resp(body=b'{"hitCount": 1, "resultList": {"result": ["x"]}}'),
    "non_200": lambda: A.Resp(status=500, body=b"{}"),
    "transport_error": lambda: OSError("boom"),
    "rate_limited": lambda: rate_limit.RateLimited("europepmc", "pacing"),
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_fallback_triggers_skip_nothing_and_resolve_per_doi(name):
    web = web_for(batch_resp=BAD[name]())
    o, _s = run(cfg_batch("on"), sources([HIT] + MISS), web)
    assert len(web.batch_urls) == 1  # one attempt, then today's path
    assert len(web.core()) == 4  # all four DOIs resolved per DOI (the three misses end NOROUTE on their own request)
    assert o.stored == 1 and reasons(o).count(NOROUTE) == 3


def test_fallback_is_attempted_once_per_ask_not_at_every_rank_site():
    memo = {}
    web = web_for(batch_resp=A.Resp(status=500, body=b"{}"))
    ranked = _ranked([HIT] + MISS)
    cfg = cfg_batch("on")
    d = A.deps([], web)
    assert auto_acquire.batch_precheck(ranked, cfg, d, memo)["fallback"] == "envelope" or memo.get("__failed__")
    n = len(web.batch_urls)
    out = auto_acquire.batch_precheck(ranked, cfg, d, memo)
    assert len(web.batch_urls) == n and out["sent"] is False


def _ranked(dois, **cand_kw):
    return [(1.0, d, {"doi": d, "title": "SYNTHETIC outbreak transmission study", **cand_kw}, None) for d in dois]


def test_deadline_stop_during_the_batch_falls_back_and_skips_nothing():
    web = web_for()
    clock = lambda: 1000.0  # noqa: E731
    out = auto_acquire.batch_precheck(_ranked([HIT] + MISS), cfg_batch("on"), A.deps([], web), {}, clock=clock,
                                      deadline=900.0)  # already past
    assert out["sent"] and out["fallback"] == acq_config.reason(cfg_batch("on"), "deadline")
    assert web.batch_urls == []  # nothing was even sent past the deadline


def test_size_refusal_falls_back(monkeypatch):
    big = A.Resp(body=b" " * 2_100_000)
    web = web_for(batch_resp=big)
    out = auto_acquire.batch_precheck(_ranked([HIT] + MISS), cfg_batch("on"), A.deps([], web), {})
    assert out["fallback"] and out["miss"] == 0


def test_api_permission_refusal_falls_back(monkeypatch):
    web = web_for()
    web.api_gated = True
    from vera import api_permissions
    monkeypatch.setattr(api_permissions, "is_api_url", lambda u: True)
    monkeypatch.setattr(api_permissions, "decide", lambda u, p: type("D", (), {
        "allowed": False, "code": "R-API-DENIED", "entry": "x", "why": "test"})())
    out = auto_acquire.batch_precheck(_ranked([HIT] + MISS), cfg_batch("on"), A.deps([], web), {})
    assert out["fallback"] == "R-API-DENIED" and out["miss"] == 0 and web.batch_urls == []


# ------------------------------------------------------------------------------------------------ per-DOI rule
def _row(doi, oa="Y", pmcid="PMC7654321", source="MED"):
    return {"id": "1", "source": source, "pmcid": pmcid, "doi": doi, "isOpenAccess": oa}


def test_require_open_access_true_a_non_oa_or_pmcid_less_row_is_a_miss():
    web = web_for(batch_rows=[_row(HIT, pmcid=HIT_PMC), _row(MISS[0], oa="N"), _row(MISS[1], pmcid=""),
                              _row(MISS[2].upper() + "x")])
    o, _s = run(cfg_batch("on"), sources([HIT] + MISS), web)
    assert len(web.core()) == 1 and reasons(o).count(NOROUTE) == 3  # lookalike DOI row (suffix x) does not rescue MISS[2]


def test_require_open_access_false_skips_only_a_doi_with_no_row_at_all():
    c = cfg_batch("on")
    c["resolve"]["require_open_access"] = False
    web = web_for(batch_rows=[_row(HIT, pmcid=HIT_PMC), _row(MISS[0], oa="N")])
    run(c, sources([HIT] + MISS), web)
    cores = unquote_plus(" ".join(web.core()))
    assert HIT in cores and MISS[0] in cores  # the row-bearing DOI is a hit and resolves per DOI
    assert MISS[1] not in cores and MISS[2] not in cores  # no row at all: skipped


def test_mixed_case_doi_and_parentheses_match_the_lowercased_row():
    upper = "10.1007/JHEP07(2024)184"
    web = web_for(batch_rows=[_row(upper.lower(), pmcid=HIT_PMC)])
    ranked = _ranked([upper] + MISS[1:] + ["10.9999/miss.four"])
    out = auto_acquire.batch_precheck(ranked, cfg_batch("on"), A.deps([], web), {})
    assert out["hit"] == 1 and out["miss"] == 3
    assert ranked[0][2]["epmc_precheck"] == "hit" and ranked[1][2]["epmc_precheck"] == "miss"
    assert 'DOI:"10.1007/jhep07(2024)184"' in unquote_plus(web.batch_urls[0])  # sent lowercased, parentheses intact


def test_a_doi_with_a_quote_is_refused_from_the_batch():
    ranked = _ranked(['10.9999/bad"quote'] + MISS + ["10.9999/ok.four"])
    web = web_for(batch_rows=[])
    auto_acquire.batch_precheck(ranked, cfg_batch("on"), A.deps([], web), {})
    assert "bad" not in unquote_plus(web.batch_urls[0]) and "epmc_precheck" not in ranked[0][2]


# ------------------------------------------------------------------------------------------------ eligibility, sizes, memo
def test_pmcid_epmc_ft_arxiv_and_oa_hits_are_not_batched():
    ranked = _ranked(MISS) + _ranked(["10.9999/has.pmcid"], pmcid="PMC111") \
        + _ranked(["10.9999/has.ft"], oa={"epmc_ft": True}) + _ranked(["10.9999/has.arxiv"], arxiv={"id": "x"})
    web = web_for(batch_rows=[])
    out = auto_acquire.batch_precheck(ranked, cfg_batch("on"), A.deps([], web), {})
    q = unquote_plus(web.batch_urls[0])
    assert out["miss"] == 3 and "has." not in q


def test_max_dois_overflow_goes_per_doi_and_min_dois_sends_no_batch():
    dois = [f"10.9999/many.{i}" for i in range(6)]
    web = web_for(batch_rows=[])
    ranked = _ranked(dois)
    out = auto_acquire.batch_precheck(ranked, cfg_batch("on", max_dois=4), A.deps([], web), {})
    q = unquote_plus(web.batch_urls[0])
    assert out["miss"] == 4 and "many.3" in q and "many.4" not in q
    assert "epmc_precheck" not in ranked[4][2] and "epmc_precheck" not in ranked[5][2]
    web2 = web_for(batch_rows=[])
    out2 = auto_acquire.batch_precheck(_ranked(dois[:2]), cfg_batch("on", min_dois=3), A.deps([], web2), {})
    assert out2["sent"] is False and web2.batch_urls == []


def test_verdicts_are_memoised_per_ask_across_rank_sites():
    memo = {}
    web = web_for(batch_rows=[])
    cfg, d = cfg_batch("on"), A.deps([], web)
    r1 = _ranked(MISS)
    auto_acquire.batch_precheck(r1, cfg, d, memo)
    r2 = _ranked(MISS)  # a later rank site: fresh candidate dicts, same DOIs
    out = auto_acquire.batch_precheck(r2, cfg, d, memo)
    assert len(web.batch_urls) == 1 and out["sent"] is False
    assert all(c["epmc_precheck"] == "miss" for _s, _d, c, _i in r2)  # the verdict still reaches the new dicts


def test_mode_off_changes_nothing():
    ranked = _ranked(MISS)
    web = web_for()
    out = auto_acquire.batch_precheck(ranked, cfg_batch("off"), A.deps([], web), {})
    assert out["sent"] is False and web.urls == [] and all("epmc_precheck" not in c for _s, _d, c, _i in ranked)


# ------------------------------------------------------------------------------------------------ landing route, url_chain
def test_skipped_doi_still_enters_landing_and_its_url_chain_holds_the_batch_marker():
    lweb, raw = step_b_web(doi=MISS[1])
    web = BatchWeb({}, {}, routes=lweb.routes, batch_rows=[])
    o, store = run(cfg_batch("on", step_b=True), sources([MISS[1], MISS[0], MISS[2]]), web)
    assert web.core() == []  # no per-DOI Europe PMC request for any skipped DOI
    assert any(u == f"https://doi.org/{MISS[1]}" for u in web.urls)  # landing was entered for the skipped DOI
    chain = store.plans[0]["source"]["provenance"]["acquisition"]["url_chain"]
    assert chain[0] == auto_acquire.BATCH_HOP and not chain[0].startswith("http")


def test_skipped_doi_without_landing_is_noroute_with_the_marker_hop():
    web = web_for(batch_rows=[])
    o, _s = run(cfg_batch("on"), sources(MISS), web)
    assert reasons(o) == [NOROUTE] * 3 and web.core() == []


# ------------------------------------------------------------------------------------------------ modes: streamed, background, shadow
def test_streamed_path_skips_the_same_way():
    stream_ctx.CTX.set(stream_ctx.StreamCtx(300))
    web = web_for()
    o, _s = run(cfg_batch("on"), sources([HIT] + MISS), web)
    assert len(web.batch_urls) == 1 and len(web.core()) == 1 and reasons(o).count(NOROUTE) == 3


def test_background_job_skips_the_same_way():
    cands = [A.cand(d) for d in [HIT] + MISS]
    web = web_for()
    c = cfg_batch("on")
    d = A.deps(cands, web)
    auto_acquire.run_job(A.job(), c, d, clock=A.clock_at(), env=A.ENV_ON)
    assert len(web.batch_urls) == 1 and len(web.core()) == 1


def test_shadow_sends_the_batch_and_every_per_doi_request_and_counts_agreement():
    web = web_for()
    o, _s = run(cfg_batch("shadow"), sources([HIT] + MISS), web)
    assert len(web.batch_urls) == 1 and len(web.core()) == 4  # nothing is skipped
    assert auto_acquire.shadow_counts() == {"compared": 4, "disagree": 0}
    assert o.stored == 1


def test_shadow_counts_a_disagreement_and_logs_only_the_public_doi(caplog):
    rows = [_row(d, pmcid="PMC7654321") for d in [HIT] + MISS]  # the batch claims every DOI is a hit; core says miss
    web = web_for(batch_rows=rows)
    with caplog.at_level("WARNING", logger="vera.autoacq"):
        run(cfg_batch("shadow"), sources([HIT] + MISS), web)
    assert auto_acquire.shadow_counts() == {"compared": 4, "disagree": 3}
    msgs = [r.getMessage() for r in caplog.records if "shadow disagreement" in r.getMessage()]
    assert len(msgs) == 3 and all(re.search(r"doi=10\.\S+ batch=hit per_doi=miss$", m) for m in msgs)
    assert "outbreak" not in " ".join(msgs)  # no question or title text


@pytest.fixture(autouse=True)
def _reset_shadow():
    auto_acquire.reset_for_tests()
    yield
    auto_acquire.reset_for_tests()


# ------------------------------------------------------------------------------------------------ pacing invariants
def test_batch_goes_through_the_europe_pmc_lane_at_0_1_per_second(monkeypatch):
    from tests.test_epmc_clock_1112 import World
    w = World(monkeypatch)
    w.use_token()  # the acquisition's own search just spent the burst token
    deps = A.deps([], w.http)
    out = auto_acquire.batch_precheck(_ranked(MISS), cfg_batch("on"), deps, {}, clock=w.clk)
    assert out["sent"]
    times = [t for t, u in w.sent if "/rest/search" in u and "resultType=lite" in u]
    assert len(times) == 1 and times[0] - 1000.0 >= 10.0 - 1e-6  # waited for the lane, not a burst past it
    cfg = rate_limit.load_config()
    p = cfg["portals"]["europepmc"]
    assert p["rate_per_s"] == 0.1 and p["burst"] == 1 and p["max_concurrency"] == 1
