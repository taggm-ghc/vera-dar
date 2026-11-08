"""NONASK-1103 A6 D1 (A6-2, A6-5): the inline acquisition round loop. Offline: every dependency is a stub, the
background worker never starts, no network, no database, no env files."""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from vera import acq_rounds, auto_acquire as aa, inline_acquire as ia, query_variants6, scope_router, stream_ctx

ROOT = pathlib.Path(__file__).resolve().parents[1]
Q = "What are the de facto status quo best practices agent archetypes used in multi-agent orchestration?"
ENV = {"VERA_DB_URL_RO": "dummy-not-a-url", "VERA_DB_URL_WO": "dummy-not-a-url"}
MARK = "LANE-PAGE-TEXT-MARKER"


def cfg(**inline):
    a = json.loads((ROOT / "config/auto_acquire.json").read_text())
    a["enabled"] = True
    a["inline"].update({"enabled": True, "acq6_rounds_enabled": True, **inline})
    a["bounds"]["docs_per_day"] = None
    return a


def cand(tag, usable=True, lane=None):
    c = {"doi": f"10.99999/v1103.{tag}", "title": f"Synthetic {tag}", "_usable": usable}
    if lane:
        c.update({"lane": lane, "answer_only": True})
    return c


class H:
    def __init__(self, mp, pools, *, endless=False, search_cost=0.01, eval_cost=0.0, on_eval=None):
        aa.reset_for_tests()
        ia.reset_for_tests()
        acq_rounds.reset_for_tests()
        self.t = 1000.0
        self.pools, self.endless, self.search_cost, self.eval_cost, self.on_eval = list(pools), endless, search_cost, eval_cost, on_eval
        self.searches, self.evaluated, self.writes, self.eval_t = [], [], [], []
        deps = SimpleNamespace(search=self.search, corpus_keys=lambda: set(), stored_rows=lambda cus: ([], []),
                               store=SimpleNamespace(write=lambda p: self.writes.append(p) or {"status": "stored"}),
                               close=None, refresh=None, url_check=None, http_get=None,
                               lookup=SimpleNamespace(known=lambda *a, **k: False, docs_last_24h=lambda: 0))
        self.deps = deps
        mp.setattr(scope_router, "intent_clean", lambda q, *a, **k: True)
        mp.setattr(aa, "_worker_loop", lambda *a, **k: None)
        mp.setattr(aa, "_topic_fit_active", lambda *a, **k: None)
        mp.setattr(aa, "rank_candidates", lambda q, cs, *a, **k: [(1.0, c.get("doi"), c, None) for c in cs])
        mp.setattr(aa, "reserve_reads", lambda ranked, n, *a, **k: list(ranked)[:max(0, int(n))])
        mp.setattr(aa, "evaluate_candidate", self.evaluate)
        mp.setattr(ia, "_stored_for", lambda ranked, *a, **k: (([], []), list(ranked)))
        mp.setattr(ia, "ephemeral_corpus", lambda plans, *a, **k: SimpleNamespace(plans=list(plans)) if plans else None)
        mp.setattr(ia, "_continue", lambda *a, **k: None)

    def clock(self):
        return self.t

    def search(self, query, *a, **k):
        i = len(self.searches)
        assert i < 200, "unbounded loop"
        from vera import search_providers as _sp
        self.searches.append({"q": query, "t": self.t, "jobs": len(aa._S.jobs_hour), "timeout": a[1] if len(a) > 1 else None,
                              "scholarly_only": _sp.SCHOLARLY_ONLY.get()})
        self.t += self.search_cost
        if i < len(self.pools):
            return [dict(c) for c in self.pools[i]]
        return [cand(f"endless{i}", usable=False)] if self.endless else []

    def evaluate(self, job, score, doi, c, isbn, acq, deps, icfg, fdcfg, clock, deadline, d, min_spans=1, **k):
        self.evaluated.append(doi)
        self.eval_t.append(self.t)
        self.t += self.eval_cost
        if self.on_eval:
            self.on_eval(self, doi)
        if not c.get("_usable"):
            raise aa._Stop("R-ACQ-IRRELEVANT")
        text = (MARK + " " if c.get("lane") else "") + f"Synthetic passage for {doi}."
        plan = {"canonical_url": f"https://example.invalid/{doi}", "doi": doi, "admitted_spans": 3,
                "spans": [{"start_index": 0, "end_index": len(text), "text": text}],
                "source": {"content": text, "content_hash": doi, "provenance": {}}}
        plan.update({k2: c[k2] for k2 in ("lane", "answer_only") if k2 in c})
        return plan, None

    def run(self, acq, stream=False, stall_s=300.0):
        self.ctx, tok = None, None
        if stream:
            self.ctx = stream_ctx.StreamCtx(stall_s, clock=self.clock)
            tok = stream_ctx.CTX.set(self.ctx)
        try:
            return ia.acquire(Q, acq, env=ENV, clock=self.clock, deps_factory=lambda a, e: self.deps)
        finally:
            if tok is not None:
                stream_ctx.CTX.reset(tok)


def test_n1_stream_always_progressing_stops_at_variant_exhaustion(monkeypatch):
    v = query_variants6.variants(Q)
    # every round yields one new usable document (so the no-progress stop never fires) and the target is out of reach
    h = H(monkeypatch, [[cand(f"s{i}")] for i in range(len(v) + 1)], search_cost=20.0)  # far past 45 s: no bound
    a = cfg(deadline_s=45)
    a["bounds"]["acq_round_target_new_docs"] = 1000
    o = h.run(a, stream=True)
    assert v and len(h.searches) == len(v) + 1  # round 0 + every variant, then variants_exhausted
    # The bound scales with the variant count: elapsed = search_cost * (rounds), so derive it from len(v) instead of a
    # fixed multiple of the deadline (QUERY-SHAPE-1108: the neutral, de-duplicated variant list is shorter than the old
    # 12+). The point is that the streamed path runs past the 45 s deadline (OD-1031-NODEADLINE).
    assert h.t - 1000.0 == 20.0 * (len(v) + 1) and h.t - 1000.0 > 45.0
    assert o.acquisition["stop"] == "variants_exhausted" and o.acquisition["rounds"] == len(v) + 1
    assert o.acquisition["status"] != "met"


def test_nonstream_absolute_bound_with_fetch_reserve_every_round(monkeypatch):
    h = H(monkeypatch, [[cand("n0", usable=False)]], endless=True, search_cost=10.0, eval_cost=1.0)
    h.run(cfg(deadline_s=45, per_doc_timeout_s=5), stream=False)
    t0, per_doc, dl = 1000.0, 5.0, 45.0
    starts = [s["t"] - t0 for s in h.searches]
    assert len(starts) >= 2
    assert all(t + per_doc < dl for t in starts), starts  # N5: each search ends a fetch reserve before the deadline
    assert all(e - t0 < dl for e in h.eval_t)  # no fetch starts after the deadline
    assert h.t - t0 <= dl + per_doc  # absolute: deadline_s + one straggler timeout


def test_variant_only_documents_are_answer_only_never_stored_or_cached(monkeypatch):
    h = H(monkeypatch, [[cand("a0"), cand("a1")], [cand("a2"), cand("a3")], [cand("a4"), cand("a5")]])
    o = h.run(cfg(), stream=False)
    assert o.admitted == 6 and o.acquisition["status"] == "met" and len(h.searches) == 3
    assert o.stored == 2 and o.acquisition["stored"] == 2
    assert sorted(p["doi"] for p in h.writes) == ["10.99999/v1103.a0", "10.99999/v1103.a1"]  # S5: round-0 docs only
    # never written; they passed the licence gate, so like below_persist_min plans they may be cached (a provider-chain
    # retry must not re-run every round); lane/web pages (R-037) are never cached (test_lane_* and verify_1103_acq6)
    assert not o.no_cache
    assert len(o.corpus.plans) == 6  # but they answer this request


def test_lane_pages_count_toward_target_but_never_stored(monkeypatch):
    h = H(monkeypatch, [[cand("n0"), cand("n1"), cand("L0", lane="official")] + [cand(f"n{i}") for i in range(2, 5)]])
    o = h.run(cfg(), stream=True)
    assert o.admitted == 6 and len(h.searches) == 1 and o.acquisition["status"] == "met"
    assert o.stored == 5 and not [p for p in h.writes if p.get("lane") or MARK in json.dumps(p)]
    assert not ia._I.results


def test_one_job_per_round(monkeypatch):
    h = H(monkeypatch, [[cand("j0", usable=False)]], endless=True)
    before = len(aa._S.jobs_hour)
    h.run(cfg(), stream=True)
    assert [s["jobs"] for s in h.searches] == list(range(before + 1, before + 1 + len(h.searches)))
    assert len(aa._S.jobs_hour) - before == len(h.searches)


@pytest.mark.parametrize("prefill,rounds", [(0, 4), (3, 1)])
def test_floor_keeps_two_free_hourly_jobs(monkeypatch, prefill, rounds):
    h = H(monkeypatch, [[cand(f"f{i}")] for i in range(10)])  # one usable doc per round: no-progress never fires
    a = cfg()
    a["bounds"]["jobs_per_hour"] = 6
    for _ in range(prefill):
        aa._S.jobs_hour.append(h.t)
        aa._S.jobs_day.append(h.t)
    o = h.run(a, stream=True)
    assert len(h.searches) == rounds
    assert 6 - len(aa._S.jobs_hour) >= aa.BG_MIN_FREE_HOURLY_JOBS
    if rounds > 1:
        assert o.acquisition["stop"] == "rate"


def test_reserve_job_floor_unit():
    aa.reset_for_tests()
    c = cfg()
    c["bounds"]["jobs_per_hour"] = 3
    assert aa.reserve_job(c, "q", floor=2) is None  # 3 - 0 - 1 = 2 >= 2
    assert aa.reserve_job(c, "q", floor=2) is not None  # 3 - 1 - 1 = 1 < 2
    assert aa.reserve_job(c, "q") is None  # floor 0 = the old rule


def test_disconnect_stops_further_rounds(monkeypatch):
    h = H(monkeypatch, [[cand("d0", usable=False)]], endless=True,
          on_eval=lambda h, doi: h.ctx.cancel() if doi.endswith("d0") else None)
    o = h.run(cfg(), stream=True)
    assert len(h.searches) == 1 and o.acquisition["status"] == "incomplete_disconnected"


def test_tried_keys_never_refetched_and_met_stops_rounds(monkeypatch):
    h = H(monkeypatch, [[cand("t0"), cand("t1")], [cand("t0"), cand("t2"), cand("t3")], [cand("t1"), cand("t4"), cand("t5")],
                        [cand("t6")]])
    o = h.run(cfg(), stream=True)
    assert len(h.evaluated) == len(set(h.evaluated)) == 6
    assert len(h.searches) == 3 and o.acquisition["status"] == "met"


def test_variant_queries_have_no_raw_sentence_operators_or_hosts(monkeypatch):
    h = H(monkeypatch, [[cand("q0", usable=False)]], endless=True)
    h.run(cfg(), stream=True)
    extra = [s["q"] for s in h.searches[1:]]
    assert extra
    keys = [aa._qkey(q) for q in extra]
    assert len(keys) == len(set(keys)) and aa._qkey(Q) not in keys
    for q in extra:
        low = q.lower()
        assert "?" not in q and "://" not in q and "site:" not in low and '"' not in q and "(" not in q
        assert Q.lower() not in low and len(q.split()) <= 16


def test_kill_switch_off_is_the_single_search(monkeypatch):
    h = H(monkeypatch, [[cand("k0"), cand("k1")], [cand("k2")]], endless=True)
    o = h.run(cfg(acq6_rounds_enabled=False), stream=True)
    assert len(h.searches) == 1 and len(aa._S.jobs_hour) == 1
    assert o.admitted == 2 and o.stored == 2 and not o.no_cache
    assert o.acquisition == {"status": "resource_limited", "stored": 2, "target": 6}


def test_shipped_config_enables_rounds():
    assert json.loads((ROOT / "config/auto_acquire.json").read_text())["inline"]["acq6_rounds_enabled"] is True


def test_n3_variant_rounds_are_scholarly_only_ddg_once(monkeypatch):
    """Review N3: round 0 is the full search (DDG once per question); every variant round runs with
    search_providers.SCHOLARLY_ONLY set, so DDG, its id lookup and Wikipedia are skipped there."""
    h = H(monkeypatch, [[cand("s0", usable=False)]], endless=True)
    h.run(cfg(), stream=True)
    assert len(h.searches) >= 2
    assert h.searches[0]["scholarly_only"] is False
    assert all(x["scholarly_only"] is True for x in h.searches[1:])


def test_n3_scholarly_only_budget_drops_ddg_wiki_lookup():
    from vera import search_providers as sp
    seen = []
    c = {"search": {"ddg_first": {"enabled": True}, "europepmc": {"enabled": True}, "providers": ["openalex", "arxiv"]},
         "bounds": {"max_search_calls_per_job": 7}}

    def base(q, n, t):
        return []

    def ddg(*a, **k):
        seen.append("ddg")
        return []
    fn = sp.acquisition_search(base, c, ddg_search=ddg, transport=lambda *a, **k: None, get=lambda *a, **k: None,
                               url_check=lambda *a, **k: True, wiki_get=lambda *a, **k: None)
    try:
        fn("agent orchestration patterns", 5, 2.0)  # control: DDG runs on the full (round-0) search
    except Exception:  # noqa: BLE001
        pass
    assert "ddg" in seen, "control: the DDG stub was never called, so the check below would be vacuous"
    seen.clear()
    tok = sp.SCHOLARLY_ONLY.set(True)
    try:
        try:
            fn("agent orchestration patterns", 5, 2.0)
        except Exception:  # noqa: BLE001  stub providers may fail; only the DDG call matters
            pass
    finally:
        sp.SCHOLARLY_ONLY.reset(tok)
    assert "ddg" not in seen


def test_no_progress_stop_after_two_empty_rounds(monkeypatch):
    """Orchestrator bound: 2 consecutive variant rounds with zero new usable candidates end the loop (no_progress),
    so an empty-result question does not drain the hourly jobs to the floor."""
    h = H(monkeypatch, [[cand("p0", usable=False)]], endless=False)  # every variant search returns nothing
    before = len(aa._S.jobs_hour)
    o = h.run(cfg(), stream=True)
    assert len(h.searches) == 3, len(h.searches)  # round 0 + 2 empty variant rounds
    assert o.acquisition["stop"] == "no_progress"
    assert len(aa._S.jobs_hour) - before == 3  # one job per search, then no more


def test_no_progress_counter_resets_on_a_round_with_new_candidates(monkeypatch):
    pools = [[cand("r0", usable=False)], [], [cand("r2")], [], [], []]
    h = H(monkeypatch, pools)
    o = h.run(cfg(), stream=True)
    assert len(h.searches) == 5 and o.acquisition["stop"] == "no_progress"  # r2 usable resets the count
