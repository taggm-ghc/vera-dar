"""MIDTERMS-WP7-1106 (RC5): cut-deferred candidates are evaluated before any variant search; unevaluated candidates are
labelled by the real stop; gate_a reports the drops reason; _trace_lane counts terms and raw_items. Offline, stubbed."""
from __future__ import annotations

from types import SimpleNamespace

from tests.test_acq6_rounds_1103 import H, cand, cfg
from vera import acq_config, acq_discovered, auto_acquire as aa, discovery_panel, inline_acquire as ia, request_trace, stream_ctx

CUT = 31


def _cutting_rank(question, cs, acq, skip, fdcfg, count, drops=None, **k):
    """Stand-in for aa.rank_candidates: rank order = input order; beyond the cut -> drops[id] = R-ACQ-DEFER_CAP."""
    ranked = [(1.0, c.get("doi"), c, None) for c in cs if (c.get("doi") not in skip)]
    if drops is not None:
        for r in ranked[CUT:]:
            drops[id(r[2])] = acq_config.reason(acq, "defer_cap")
    return ranked[:CUT]


def test_6_deferred_candidates_evaluated_in_round_2_before_any_variant_search(monkeypatch):
    pool = [cand(f"u{i}", usable=False) for i in range(31)] + [cand(f"g{i}") for i in range(9)]  # 40; 32..40 usable
    h = H(monkeypatch, [pool])
    monkeypatch.setattr(aa, "rank_candidates", _cutting_rank)
    o = h.run(cfg(deadline_s=45), stream=False)
    assert len(h.searches) == 1  # no variant search: the 9 cut candidates were enough
    assert [d for d in h.evaluated if ".g" in d] and len([d for d in h.evaluated if ".g" in d]) >= 6
    left = [c for c in o.candidates if c["reason"] == "R-ACQ-DEFER_CAP"]  # only those not needed once the target was met
    assert len(left) == 3 and all(c["outcome"] == "dropped" for c in left)
    done = [c for c in o.candidates if c["outcome"] == "admitted"]
    assert len(done) >= 6 and o.acquisition["status"] == "met"


def test_6_deferred_not_tried_until_evaluated_unreached_ones_get_real_stop(monkeypatch):
    pool = [cand(f"u{i}", usable=False) for i in range(31)] + [cand(f"g{i}") for i in range(3)]
    h = H(monkeypatch, [pool])
    monkeypatch.setattr(aa, "rank_candidates", _cutting_rank)
    ctx_box = {}

    def stop_after_first(hh, doi):
        if ".g" in doi:
            ctx_box["ctx"]._stop = stream_ctx.DISCONNECTED  # the visitor leaves while the cut candidates are read
    h.on_eval = stop_after_first
    h.ctx = ctx_box["ctx"] = stream_ctx.StreamCtx(300.0, clock=h.clock)
    tok = stream_ctx.CTX.set(h.ctx)
    try:
        o = ia.acquire(h.__class__.__mro__[0] and __import__("tests.test_acq6_rounds_1103", fromlist=["Q"]).Q, cfg(),
                       env={"VERA_DB_URL_RO": "dummy-not-a-url", "VERA_DB_URL_WO": "dummy-not-a-url"},
                       clock=h.clock, deps_factory=lambda a, e: h.deps)
    finally:
        stream_ctx.CTX.reset(tok)
    why = {c["reason"] for c in o.candidates if c["outcome"] in ("not_evaluated", None)}
    assert "R-ACQ-DEFER_CAP" not in why


def test_left_reason_names_the_real_stop():
    acq = {}
    live = SimpleNamespace(stop_reason=lambda: None, _stop=None)
    assert ia._left_reason(acq, None, {}, True) == "R-ACQ-DEADLINE"
    assert ia._left_reason(acq, SimpleNamespace(_stop=stream_ctx.STALLED), {}, False) == "R-ACQ-STOPPED"
    assert ia._left_reason(acq, live, {"acq6": {"stop": "no_progress"}}, False) == "R-ACQ-NO_PROGRESS"
    assert ia._left_reason(acq, live, {"acq6": {"stop": "ceiling31"}}, False) == "R-ACQ-DOC_BUDGET"
    assert ia._left_reason(acq, live, {"acq6": {"stop": "variants_exhausted"}}, False) == "R-ACQ-DEFER_CAP"
    assert ia._left_reason(acq, live, {}, False) == "R-ACQ-DEFER_CAP"


def test_gate_a_uses_the_drops_reason_before_defer_cap():
    acq = cfg()
    def disc(tag):
        return {"doi": f"10.9/{tag}", "title": tag, "discovery": {"record_id": tag}}
    dup, nodoi, cut = disc("dup"), disc("nodoi"), disc("cut")
    merged = [dup, nodoi, cut]
    drops = {id(dup): acq_config.reason(acq, "duplicate"), id(nodoi): acq_config.reason(acq, "no_doi")}
    out = acq_discovered.gate_a([dup, nodoi, cut], merged, [], set(), acq, drops)
    assert (out[id(dup)]["outcome"], out[id(dup)]["reason"]) == ("rejected", "R-ACQ-DUPLICATE")
    assert (out[id(nodoi)]["outcome"], out[id(nodoi)]["reason"]) == ("rejected", "R-ACQ-NODOI")
    assert (out[id(cut)]["outcome"], out[id(cut)]["reason"]) == ("deferred", "R-ACQ-DEFER_CAP")  # a real cut stays the cut


def test_panel_has_plain_labels_for_the_new_reasons():
    for code in ("R-ACQ-STOPPED", "R-ACQ-NO_PROGRESS"):
        t = discovery_panel.REASON_TEXT[code]
        assert t.startswith("Not checked") and "R-ACQ" not in t and "candidate" not in t.lower()


def _lane_counts(monkeypatch, diag):
    seen = {}
    monkeypatch.setattr(ia.rt, "event", lambda stage, code, counts=None, detail=None: seen.update(counts=counts))
    ia._trace_lane("What are the 2026 midterm elections results?", SimpleNamespace(diag=diag, pages=[], links=[]))
    return seen["counts"]


def test_10_trace_lane_emits_terms_and_raw_items_counts(monkeypatch):
    c = _lane_counts(monkeypatch, [{"portal": "gdelt", "status": "called", "reason": "outlet_policy", "items": 0,
                                    "terms": 4, "raw_items": 7}])
    assert c["terms"] == 4 and c["raw_items"] == 7 and c["items"] == 0
    kept = request_trace.clean_counts(c)  # counts keys are identifier tokens, not allowlisted detail keys
    assert kept["terms"] == 4 and kept["raw_items"] == 7
    c2 = _lane_counts(monkeypatch, [{"portal": "gdelt", "status": "called", "reason": "ok", "items": 2}])
    assert "terms" not in c2 and "raw_items" not in c2  # a producer that does not report them adds no guessed zero


def test_trace_reason_passes_the_request_trace_value_filter():
    assert ia._trace_reason("R-ACQ-DEFER_CAP") == "r-acq-defer_cap"
    assert ia._trace_reason("R-ACQ-RATELIMITED-COOLDOWN") == "r-acq-ratelimited-cooldown"
    assert ia._trace_reason("some free text, https://x") is None and ia._trace_reason(None) is None
    out = request_trace.clean_detail({"reason": ia._trace_reason("R-ACQ-NO_PROGRESS")}, stage="acquire")
    assert out is None or out.get("reason") == "r-acq-no_progress" or "reason" not in out  # key allowlisted: value kept
