"""FA8 (IMPL-1028-FA8): claim_check switches context_title (FA8-2) and keep_partial (FA8-3), per-claim reason in
records (FA8-1), and the full-document sidebar wording (Fix 8). LLM stubbed; UNCALIBRATED switches, tested by
position only."""
import json

import pytest

from vera import claim_check as cc

CC = {"enabled": True, "timeout_s": 1, "max_claims": 30, "partial_marker": " (partly)", "removed_note": "REMOVED",
      "checkers": [{"provider": "a", "base_url": "x", "model": "first", "api_key_env": "KEY_A", "family": "qwen"}]}
SRC = [{"n": 1, "title": "Police defensive tactics", "snippet": "Officers trained in DT."},
       {"n": 2, "title": "Boxing", "snippet": "Boxers jab."}]
ANS = "Alpha [1]. Beta [2]. Gamma [1]."
V = {0: ("supported", "ok"), 1: ("partial", "weak"), 2: ("unsupported", "wrong subject")}


def _call(seen):
    def f(checker, msgs):
        seen.append(msgs)
        return json.dumps({"claims": [{"id": i, "verdict": v, "reason": r} for i, (v, r) in V.items()]})
    return f


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("KEY_A", "k")


def _run(cfg):
    seen, recs = [], []
    text, summary = cc.check_answer(ANS, SRC, cfg, call=_call(seen), records=recs)
    return text, summary, recs, seen[0]


def test_defaults_reproduce_current_output():
    base = _run(CC)
    explicit = _run(dict(CC, context_title=False, keep_partial=True))
    assert base[0] == "Alpha [1]. Beta [2]. (partly)\n\nREMOVED"
    assert base[1] == {"checked": 3, "supported": 1, "partial": 1, "removed": 1, "checker": "a:first", "outcome": "checked"}
    assert base[:2] == explicit[:2] and base[3] == explicit[3]
    user = json.loads(base[3][1]["content"])
    assert user["sources"] == {"1": "Officers trained in DT.", "2": "Boxers jab."}  # snippet strings, no titles
    assert base[3][0]["content"] == cc.SYSTEM
    assert [{k: r[k] for k in ("id", "sentence", "citations", "verdict")} for r in base[2]][1]["verdict"] == "partial"


def test_fa8_1_reason_kept_in_records():
    recs = _run(CC)[2]
    assert [r["reason"] for r in recs] == ["ok", "weak", "wrong subject"]


def test_fa8_1_missing_reason_is_empty():
    recs = []
    cc.check_answer("A [1].", SRC, CC, records=recs,
                    call=lambda c, m: json.dumps({"claims": [{"id": 0, "verdict": "supported"}]}))
    assert recs[0]["reason"] == ""


def test_context_title_on_adds_titles_and_rubric():
    msgs = _run(dict(CC, context_title=True))[3]
    user = json.loads(msgs[1]["content"])
    assert user["sources"]["1"] == {"title": "Police defensive tactics", "text": "Officers trained in DT."}
    assert "different subject, population or activity" in msgs[0]["content"] and msgs[0]["content"].startswith(cc.SYSTEM)
    assert "partial\", not \"unsupported" in msgs[0]["content"] and "never supports a claim by itself" in msgs[0]["content"]


def test_context_title_is_capped():
    src = [dict(SRC[0], title="T" * 1000), SRC[1]]
    seen = []
    cc.check_answer(ANS, src, dict(CC, context_title=True), call=_call(seen), records=[])
    t = json.loads(seen[0][1]["content"])["sources"]["1"]["title"]
    assert t == "T" * cc.TITLE_MAX_CHARS == "T" * 300


def test_portal_recorder_removed_partial_not_cited():
    from vera import portal_recorder as pr
    c = {"outcome": "admitted", "n_chunks": 1, "url": "u"}
    srcs = [{"n": 1, "url": "u"}]
    kept = [{"verdict": "partial", "citations": [1]}]
    gone = [{"verdict": "partial", "citations": [1], "partial_removed": True}]
    assert pr.stage_outcome(c, srcs, kept, {}, True)["cited"] is True
    assert pr.stage_outcome(c, srcs, gone, {}, True)["cited"] is False


def test_context_title_non_bool_is_off():
    assert _run(dict(CC, context_title="true"))[3] == _run(CC)[3]


def test_context_title_with_responsiveness():
    seen = []
    cfg = dict(CC, context_title=True, responsiveness={"enabled": True})
    rows = lambda c, m: (seen.append(m), json.dumps({"claims": [{"id": 0, "verdict": "supported", "reason": "r", "responsive": "yes"}]}))[1]
    cc.check_answer("A [1].", SRC, cfg, call=rows, records=[], question="q?")
    s = seen[0][0]["content"]
    assert cc.TITLE_SYSTEM in s and cc.RESPONSIVE_SYSTEM in s


def test_keep_partial_false_removes_partial():
    text, summary, recs, _ = _run(dict(CC, keep_partial=False))
    assert text == "Alpha [1].\n\nREMOVED"
    assert summary["partial"] == 0 and summary["removed"] == 2 and summary["supported"] == 1
    assert recs[1]["verdict"] == "partial" and recs[1]["partial_removed"] is True  # kept for measurement
    assert "partial_removed" not in recs[0] and "partial_removed" not in recs[2]


def test_keep_partial_false_only_partial_fails_closed():
    text, summary = cc.check_answer("B [2].", SRC, dict(CC, keep_partial=False), records=[],
                                    call=lambda c, m: json.dumps({"claims": [{"id": 0, "verdict": "partial", "reason": "x"}]}))
    assert text is None and summary["outcome"] == "no_supported_claims"


def test_keep_partial_false_with_responsiveness_counts():
    cfg = dict(CC, keep_partial=False, responsiveness={"enabled": True})
    rep = json.dumps({"claims": [{"id": 0, "verdict": "supported", "reason": "r", "responsive": "yes"},
                                 {"id": 1, "verdict": "partial", "reason": "r", "responsive": "no"}]})
    text, summary = cc.check_answer("A [1]. B [2].", SRC, cfg, call=lambda c, m: rep, records=[], question="q?")
    assert text == "A [1].\n\nREMOVED" and summary["removed"] == 1 and summary["nonresponsive_removed"] == 0


def test_fulldoc_checked_only_and_enforce_skip_removed_partials():
    from vera import fulldoc_answer as fa
    recs = [{"id": 0, "sentence": "A [1].", "citations": [1], "verdict": "supported"},
            {"id": 1, "sentence": "B [2].", "citations": [2], "verdict": "partial", "partial_removed": True},
            {"id": 2, "sentence": "C [1].", "citations": [1], "verdict": "partial"}]
    cfg = {"answer": {"quote_max_chars": 200}}
    kept, _, _ = fa.enforce_verbatim(recs, SRC, None, cfg)
    assert [r["id"] for r in kept] == [0, 2]
    text, _ = fa.checked_only("A [1]. B [2]. C [1].", recs, SRC, {"partial_marker": " (p)", "removed_note": "R"},
                              {"answer": {"quote_max_chars": 200, "unchecked_note": "U"}}, {})
    assert text == "A [1].\n\nC [1]. (p)"


class _St:
    def __init__(self): self.cap = []
    def caption(self, t): self.cap.append(t)


def test_sidebar_wording_fulldoc_vs_abstract():
    import inspect
    from vera import sources_sidebar as ss
    src = inspect.getsource(ss)
    assert "its full text was not acquired for this answer" in src
    assert "no abstract was available for it, so not used" in src  # non-fulldoc wording unchanged
