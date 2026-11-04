import itertools

from vera import adequacy_gate as ag
from vera import answer_contract, fulldoc_answer


def test_verdicts_and_field():
    r = ag.evaluate([1, 2], [1, 2], {1: "a"})
    assert r["verdict"] == "finalize" and r["all_declared_parts_supported"] is True and r["parts"][0]["label"] == "a"
    r = ag.evaluate([1, 2], [1])
    assert r["verdict"] == "qualify" and r["all_declared_parts_supported"] is False
    assert [p["supported"] for p in r["parts"]] == [True, False]
    assert ag.evaluate([1, 2], [])["verdict"] == "abstain"
    assert ag.evaluate([], [])["all_declared_parts_supported"] is False
    assert ag.evaluate([1], [1], resp_partial=True)["verdict"] == "qualify"


def test_record_is_separate_from_claim_records():
    claims = [{"id": 0, "sentence": "s", "verdict": "supported", "part": 1}]
    r = ag.evaluate([1], [1])
    assert claims == [{"id": 0, "sentence": "s", "verdict": "supported", "part": 1}]
    assert "sentence" not in str(r) and "id" not in r


def test_parity_with_answer_status_and_resp_partial():
    rcfg = {}
    for declared, sup_mask, flavours in itertools.product([(1,), (1, 2), (1, 2, 3)], range(8), [("yes",), ("partial",), ("partial", "yes")]):
        sup = [k for i, k in enumerate(declared) if sup_mask >> i & 1]
        kept = [{"responsive": f} for f in flavours] if sup else []
        rp = fulldoc_answer._resp_partial(kept, rcfg)
        unanswered = [k for k in declared if k not in sup]
        legacy_complete = not unanswered and not rp  # fulldoc_answer.py "complete = not unanswered and not resp_partial"
        rec = ag.evaluate(declared, sup, resp_partial=rp)
        if sup:
            status = answer_contract.answer_status("answer", "ok", ag.completeness(rec))
            assert (status == "complete") == legacy_complete == (rec["verdict"] == "finalize")
            assert (status == "partial") == (not legacy_complete)
        else:
            assert rec["verdict"] == "abstain"
            assert answer_contract.answer_status("insufficient", "ok", None) == "not_answered"


def test_es2_four_vendor_two_supported_qualify():
    subs = [{"label": f"v{i}", "declared": [1], "supported": [1] if i < 2 else []} for i in range(4)]
    r = ag.evaluate([1], [1], subqueries=subs)
    assert r["verdict"] == "qualify"
    assert [s["not_found"] for s in r["subqueries"]] == [False, False, True, True]
    assert [s["verdict"] for s in r["subqueries"]] == ["finalize", "finalize", "abstain", "abstain"]
    assert ag.evaluate([1], [], subqueries=[{"label": "a"}, {"label": "b"}])["verdict"] == "abstain"
    full = [{"label": "a", "declared": [1], "supported": [1]}, {"label": "b", "declared": [1], "supported": [1]}]
    assert ag.evaluate([1], [1], subqueries=full)["verdict"] == "finalize"


def test_es2_single_part_parity():
    for d, s_, rp in [([1], [1], False), ([1, 2], [1], False), ([1], [], False), ([1], [1], True)]:
        base = ag.evaluate(d, s_, resp_partial=rp)
        one = ag.evaluate(d, s_, resp_partial=rp, subqueries=[{"label": "x", "declared": d, "supported": s_}])
        assert one == base and "subqueries" not in one
