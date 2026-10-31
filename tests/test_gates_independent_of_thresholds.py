"""OD-1030-HITL2-3: the provenance, integrity and evidence gates are enforced independently of the two admission
thresholds. Each case feeds ONE candidate that fails only one gate, with min_relevance and min_question_coverage both
at the shipped 0.34 and at 0 (the weakest setting), and asserts the candidate is still rejected for that gate's reason.
A clean candidate is admitted at both values, so the rejections are not vacuous. Offline; no network or DB."""
import copy

import pytest

from tests.fulldoc_bundle_synth import synth_cfg, write_bundle
from tests.test_fulldoc_admission import CFG, GOOD, Q, corp, span
from tests.test_fulldoc_answer import CFG as ACFG, GCFG, Stub, bundle as _unused, checker  # noqa: F401
from vera import fulldoc_answer as fa
from vera import fulldoc_corpus as fc
from vera.fulldoc_admission import admit

THRESHOLDS = (0.34, 0.0)


def cfg_at(t, base=CFG):
    c = copy.deepcopy(base)
    c["admission"].update(min_relevance=t, min_question_coverage=t)
    return c


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("FD_TEST_KEY", "x")


@pytest.mark.parametrize("t", THRESHOLDS)
def test_control_clean_candidate_is_admitted(t):
    b = admit(Q, corp([span("d1", "s1", 10, GOOD)]), cfg_at(t))
    assert b.status == "admitted" and [a["span_id"] for a in b.admitted] == ["s1"]


def _unlicensed(t):
    return admit(Q, corp([span("d1", "s1", 10, GOOD)], licences={"d1": "hold"}), cfg_at(t))


def _not_acquired(t):
    return admit(Q, corp([span("dX", "s1", 0, GOOD)]), cfg_at(t))


def _span_text_mismatch(t):
    s = span("d1", "s1", 10, GOOD)
    s["text"] = GOOD.replace("improved", "worsened")  # same length, still matches the question words
    return admit(Q, corp([s]), cfg_at(t))


def _non_body_range(t):
    c = corp([span("d1", "s1", 10, GOOD)])
    c.acquired[0]["evidence_ranges"] = [[10 ** 4, 10 ** 5]]  # the span lies outside every body range
    return admit(Q, c, cfg_at(t))


@pytest.mark.parametrize("t", THRESHOLDS)
@pytest.mark.parametrize("make,reason", [(_unlicensed, "licence"), (_not_acquired, "not_acquired"),
                                         (_span_text_mismatch, "span_mismatch"), (_non_body_range, "not_body_section")])
def test_admission_gates_reject_at_both_thresholds(make, reason, t):
    b = make(t)
    assert b.admitted == () and not b.grounded and b.status == "insufficient"
    assert [r["reason"] for r in b.rejected] == [reason]


class _Pinned:
    """Corpus over a bundle with a tampered text file; retrieve() returns spans taken from the clean bundle."""

    def __init__(self, corpus, cands):
        self._c, self._cands = corpus, cands

    def retrieve(self, q, k=None, evidence_only=False):
        return list(self._cands)

    def __getattr__(self, name):
        return getattr(self._c, name)


@pytest.mark.parametrize("t", THRESHOLDS)
def test_sha_mismatch_document_is_never_admitted(tmp_path, t):
    q = "temperature radiative forcing ensemble"
    clean = fc.load_bundle(write_bundle(tmp_path / "clean"), synth_cfg())
    cands = [s for s in clean.retrieve(q, 12) if s["document_id"] == "syn_clim"]
    assert cands and clean.status == "ok"
    cfg = cfg_at(t, base=copy.deepcopy(synth_cfg()))
    ok = admit(q, _Pinned(clean, cands), cfg)  # control: the same spans on the untampered corpus are admitted
    assert ok.admitted, "control failed: the clean candidate must be admitted"
    bad = fc.load_bundle(write_bundle(tmp_path / "bad", tamper={"syn_clim": " TAMPERED"}), synth_cfg())
    assert {u["document_id"]: u["code"] for u in bad.unusable} == {"syn_clim": "R-SUMS"}
    b = admit(q, _Pinned(bad, cands), cfg)
    assert b.admitted == () and not b.grounded
    assert {r["reason"] for r in b.rejected} == {"not_acquired"}


@pytest.mark.parametrize("t", THRESHOLDS)
def test_unsupported_claim_is_not_answered(t):
    ev = admit(Q, corp([span("d1", "s1", 10, GOOD)]), cfg_at(t))
    assert ev.grounded
    stub = Stub("Forecasting rainfall accuracy improved [1].")
    out = fa.answer(Q, ev, "m", None, chained={}, grounding_cfg=GCFG, cfg=ACFG, base_tokens=0, base_cost=0.0,
                    complete_fn=stub, check_call=checker(("unsupported",)))
    assert out.result is None and out.fallback == "claim_check_insufficient"
    ok = fa.answer(Q, ev, "m", None, chained={}, grounding_cfg=GCFG, cfg=ACFG, base_tokens=0, base_cost=0.0,
                   complete_fn=Stub("Forecasting rainfall accuracy improved [1]."), check_call=checker(("supported",)))
    assert ok.fallback is None and ok.result is not None  # control: supported claim passes
