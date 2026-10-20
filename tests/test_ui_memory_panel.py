"""Item #84: UI display of VERA-DAR's recalled findings from memory."""
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent


def test_memory_panel_renders_with_recalled_findings():
    """Test that recalled findings are displayed with claim text, source link, and saved date."""
    from vera.memory_panel import render_memory_panel

    mock_st = mock.MagicMock()
    data = {
        "answer": "Test answer",
        "sources": [],
        "recalled": [
            {
                "claim_text": "Earlier finding about quantum physics",
                "source_title": "Quantum Computing Basics",
                "source_url": "https://arxiv.org/abs/2301.00001",
                "saved_at": "2026-10-07T10:30:00"
            }
        ],
        "memory_written": 0
    }

    render_memory_panel(mock_st, data)

    # Should call markdown with the heading
    markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list]
    assert any("From VERA-DAR's memory" in str(call) for call in markdown_calls)
    # Should call caption with memory description
    caption_calls = [call[0][0] for call in mock_st.caption.call_args_list]
    assert any("confirmed by vera-dar's owner" in str(call).lower() for call in caption_calls)  # D6


def test_memory_panel_empty_when_no_recalled_and_no_written():
    """Test that memory panel is not rendered when there are no recalled items and memory_submitted is 0."""
    from vera.memory_panel import render_memory_panel

    mock_st = mock.MagicMock()
    data = {
        "answer": "Test answer",
        "sources": [],
        "recalled": [],
        "memory_written": 0
    }

    render_memory_panel(mock_st, data)

    # Should not call markdown at all
    mock_st.markdown.assert_not_called()
    mock_st.success.assert_not_called()
    mock_st.info.assert_not_called()


def test_memory_panel_shows_memory_submitted_count():
    """memory_submitted displays the submission line (st.info, not a success box)."""
    from vera.memory_panel import render_memory_panel

    mock_st = mock.MagicMock()
    data = {
        "answer": "Test answer",
        "sources": [],
        "recalled": [],
        "memory_submitted": 3
    }

    render_memory_panel(mock_st, data)

    mock_st.success.assert_not_called()
    info_calls = [call[0][0] for call in mock_st.info.call_args_list]
    from vera.memory_panel import _submitted_line
    assert info_calls == [_submitted_line(3)]  # bare line: count is not displayed (owner ruling BARE)
    assert "3" not in info_calls[0] and "submitted" in info_calls[0]


def test_memory_panel_handles_missing_keys():
    """Test that the panel gracefully handles missing fields in recalled items."""
    from vera.memory_panel import render_memory_panel

    mock_st = mock.MagicMock()
    data = {
        "answer": "Test answer",
        "sources": [],
        "recalled": [
            {
                "claim_text": "Finding without URL",
                # source_url and source_title missing
            },
            {
                "source_title": "Title without URL and claim",
                "saved_at": "2026-10-07T10:30:00"
                # claim_text and source_url missing
            }
        ],
        "memory_submitted": 1
    }

    # Should not raise an exception
    render_memory_panel(mock_st, data)

    # Should have called markdown for heading
    markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list]
    assert any("From VERA-DAR's memory" in str(call) for call in markdown_calls)
    info_calls = [call[0][0] for call in mock_st.info.call_args_list]
    assert any("submitted" in str(call) for call in info_calls)


def test_memory_panel_handles_missing_recalled_and_memory_written_keys():
    """Test backward compatibility when recalled and memory_written keys are absent (older API)."""
    from vera.memory_panel import render_memory_panel

    mock_st = mock.MagicMock()
    data = {
        "answer": "Test answer",
        "sources": []
        # no recalled or memory_written keys at all
    }

    render_memory_panel(mock_st, data)

    # Should not call anything (graceful no-op)
    mock_st.markdown.assert_not_called()
    mock_st.success.assert_not_called()
    mock_st.caption.assert_not_called()


def test_memory_panel_with_multiple_recalled_items():
    """Test that multiple recalled items are all displayed."""
    from vera.memory_panel import render_memory_panel

    mock_st = mock.MagicMock()
    data = {
        "answer": "Test answer",
        "sources": [],
        "recalled": [
            {
                "claim_text": "First finding",
                "source_title": "First Source",
                "source_url": "https://example.com/1",
                "saved_at": "2026-10-07T10:00:00"
            },
            {
                "claim_text": "Second finding",
                "source_title": "Second Source",
                "source_url": "https://example.com/2",
                "saved_at": "2026-10-07T11:00:00"
            }
        ],
        "memory_written": 0
    }

    render_memory_panel(mock_st, data)

    # Should have called markdown for heading
    markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list]
    assert any("From VERA-DAR's memory" in str(call) for call in markdown_calls)
    # Should have called caption multiple times for each item
    caption_calls = mock_st.caption.call_args_list
    assert len(caption_calls) >= 2  # At least 2 captions for the items


def _panel_text(mock_st):
    calls = (mock_st.markdown.call_args_list + mock_st.caption.call_args_list + mock_st.info.call_args_list)
    return " ".join(str(c[0][0]) for c in calls).lower()


def test_memory_panel_text_names_no_backend_component():
    """OD-SIDEBAR-2: visitor-facing memory panel text names no gate or internal module."""
    from vera.memory_panel import render_memory_panel

    mock_st = mock.MagicMock()
    render_memory_panel(mock_st, {"recalled": [{"claim_text": "A finding", "source_title": "T",
                                                "source_url": "https://arxiv.org/abs/2301.00001",
                                                "saved_at": "2026-10-07"}], "memory_submitted": 2})
    text = _panel_text(mock_st)
    assert "passed vera-dar's automated checks" in text
    for term in ("write gate", "gate", "memory_gate"):
        assert term not in text


def test_recalled_caption_claims_owner_confirmation_only_when_required():
    """DEC-EI1: the caption must not claim owner confirmation when confirm_required is off."""
    from vera.memory_panel import _recalled_caption

    on, off = _recalled_caption(True).lower(), _recalled_caption(False).lower()
    assert "confirmed by vera-dar's owner" in on and "not individually confirmed" in off
    for c in (on, off):
        assert "gate" not in c and "operator" not in c and "checked finding" not in c



# ---------------------------------------------------------------- K1-M1 panel tests
import re
import pytest

OWNER_SENTENCE = "Findings are submitted, and those that pass wait for owner review."
FORBIDDEN = ["saved", "stored", "remembered", "verified", "checked finding", "gate", "pending", "queue", "queued",
             "kept", "operator", "owner key", "d6", "memory_gate", "confirm_required",
             "admitted by", "dec-", "vera_vjay", "milestone", "gate a", "gate b", "gate c", "licence gate", "write gate"]
_INTERNAL_RE = re.compile(r"#\d{2,3}\b|\bM[1-7]\b|\bitem \d")


def _infos(st):
    return [c[0][0] for c in st.info.call_args_list]


def test_panel_hidden_when_nothing_recalled_and_nothing_submitted():
    from vera.memory_panel import render_memory_panel
    st = mock.MagicMock()
    render_memory_panel(st, {"recalled": [], "memory_submitted": 0})
    st.info.assert_not_called(); st.markdown.assert_not_called(); st.caption.assert_not_called()


def test_panel_shows_submitted_line_when_submitted_positive_and_nothing_recalled():
    from vera.memory_panel import render_memory_panel
    st = mock.MagicMock()
    render_memory_panel(st, {"recalled": [], "memory_submitted": 2})
    assert len(_infos(st)) == 1 and st.markdown.call_count == 0


def test_panel_submitted_line_uses_owner_approved_sentence():
    from vera.memory_panel import _submitted_line
    assert _submitted_line(2, True) == OWNER_SENTENCE  # exact full-string equality (owner ruling BARE)
    assert _submitted_line(1, True) == OWNER_SENTENCE
    assert "Some or all may not pass" not in _submitted_line(2, True)
    assert "owner" not in _submitted_line(2, False).lower()


def test_panel_submitted_line_singular_plural():
    from vera.memory_panel import _submitted_line
    assert "from this answer" not in _submitted_line(1, True) and "from this answer" not in _submitted_line(2, True)
    assert not re.search(r"\d", _submitted_line(2, True))


def test_panel_ignores_legacy_memory_written_constant():
    from vera.memory_panel import render_memory_panel
    st = mock.MagicMock()
    render_memory_panel(st, {"memory_written": 5})
    st.info.assert_not_called(); st.success.assert_not_called()


@pytest.mark.parametrize("bad", ["abc", None, -1, True, 2.7])
def test_panel_submitted_non_integer_or_negative_is_treated_as_zero(bad):
    from vera.memory_panel import render_memory_panel
    st = mock.MagicMock()
    render_memory_panel(st, {"recalled": [], "memory_submitted": bad})
    st.info.assert_not_called()


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("confirm", [True, False])
def test_panel_submitted_line_forbidden_words(n, confirm):
    from vera.memory_panel import _submitted_line
    text = _submitted_line(n, confirm).lower()
    for term in FORBIDDEN:
        assert term not in text, term
    assert not _INTERNAL_RE.search(text)


def test_panel_submitted_line_claims_owner_review_only_when_confirm_required():
    from vera.memory_panel import _submitted_line
    assert "owner" in _submitted_line(2, True).lower()
    assert "owner" not in _submitted_line(2, False).lower()


def test_panel_submitted_line_noun_parameter():
    from vera.memory_panel import _submitted_line
    one, two = _submitted_line(1, True, noun="record"), _submitted_line(2, True, noun="record")
    assert one == two == "Records are submitted, and those that pass wait for owner review."
    assert _submitted_line(1, True) == OWNER_SENTENCE


def test_panel_renders_nothing_for_a0_zero_submitted():
    from vera.memory_panel import render_memory_panel
    st = mock.MagicMock()
    render_memory_panel(st, {"memory_submitted": 0, "claim_check": None})
    st.info.assert_not_called()


def test_panel_renders_no_claim_text_or_ids_for_submitted_only():
    from vera.memory_panel import _submitted_line, render_memory_panel
    st = mock.MagicMock()
    render_memory_panel(st, {"memory_submitted": 3, "claims": [{"sentence": "SECRET CLAIM", "id": 99}]})
    assert _infos(st) == [_submitted_line(3)]
    assert "SECRET" not in _infos(st)[0]


# ---------------------------------------------------------------- K1-M1 API contract tests
from fastapi.testclient import TestClient
from vera.ask_service import AskResult

HEAD_KEYS = {"answer", "model", "tokens_used", "cost_usd", "sources", "traced_sources", "recalled",
             "memory_written", "claim_check", "result_kind", "candidates", "contract", "recalled_discovery",
             "discovery_note_submitted"}  # K2.I6 added the last five; memory_submitted is the K1 field


class _Mem:
    def __init__(self, reader=True, store=True, cap=3, cfg_raises=False):
        self.reader, self.store = (object() if reader else None), (object() if store else None)
        self._cap, self._raises = cap, cfg_raises
        self.writes = []

    @property
    def cfg(self):
        if self._raises:
            raise RuntimeError("cfg boom")
        return {"max_writes_per_answer": self._cap}

    def recall(self, q):
        return []

    def write_claims(self, records, sources, question):
        self.writes.append(records)
        return 0, {}


def _claims(*verdicts):
    return tuple({"id": i, "sentence": f"s{i}", "citations": [1], "verdict": v} for i, v in enumerate(verdicts))


def _ask(monkeypatch, mem, accepted=True, claims=(), mem_raises=False, kind="answer"):
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    import main
    from vera import public_mode as pm
    pm.reset_limiter(None)
    monkeypatch.setattr(main, "answer_via_chain",
                        lambda q, e, g: (AskResult("A [1]", 5, 0.0, accepted=accepted, claims=claims, result_kind=kind), "groq:m"))
    monkeypatch.setattr(main, "store_from_env", lambda cfg: None)

    def fake_mem(block):
        if mem_raises:
            raise RuntimeError("no memory")
        return mem
    monkeypatch.setattr(main, "memory_from_env", fake_mem)
    r = TestClient(main.app).post("/ask", json={"question": "q"}, headers={"X-API-Key": "k" * 20})
    assert r.status_code == 200
    return r.json()


S, N = "supported", "not_supported"


def test_ask_memory_submitted_counts_supported_claims_when_accepted_and_memory_available(monkeypatch):
    assert _ask(monkeypatch, _Mem(), claims=_claims(S, S))["memory_submitted"] == 2


def test_ask_memory_submitted_is_zero_when_write_store_absent(monkeypatch):
    assert _ask(monkeypatch, _Mem(store=False), claims=_claims(S))["memory_submitted"] == 0
    assert _ask(monkeypatch, _Mem(reader=False), claims=_claims(S))["memory_submitted"] == 0


def test_ask_memory_submitted_excludes_unsupported_claims(monkeypatch):
    assert _ask(monkeypatch, _Mem(), claims=_claims(S, N, N, S))["memory_submitted"] == 2


def test_ask_memory_submitted_is_zero_when_accepted_and_claims_empty(monkeypatch):
    assert _ask(monkeypatch, _Mem(), claims=())["memory_submitted"] == 0


def test_ask_memory_submitted_is_zero_when_declined_out_of_scope(monkeypatch):
    assert _ask(monkeypatch, _Mem(), accepted=False, claims=_claims(S))["memory_submitted"] == 0


def test_ask_memory_submitted_is_zero_when_no_claims(monkeypatch):
    assert _ask(monkeypatch, _Mem(), claims=_claims(N))["memory_submitted"] == 0


def test_ask_memory_submitted_is_zero_when_memory_unavailable(monkeypatch):
    assert _ask(monkeypatch, None, claims=_claims(S))["memory_submitted"] == 0
    assert _ask(monkeypatch, None, claims=_claims(S), mem_raises=True)["memory_submitted"] == 0


def test_ask_memory_submitted_capped_at_max_writes_per_answer(monkeypatch):
    assert _ask(monkeypatch, _Mem(cap=2), claims=_claims(S, S, S, S))["memory_submitted"] == 2


def test_ask_memory_written_still_present_and_zero(monkeypatch):
    assert _ask(monkeypatch, _Mem(), claims=_claims(S))["memory_written"] == 0


def test_ask_response_has_no_ids_text_or_global_count_fields(monkeypatch):
    body = _ask(monkeypatch, _Mem(), claims=_claims(S))
    assert set(body) - HEAD_KEYS == {"memory_submitted"}
    assert type(body["memory_submitted"]) is int


def test_ask_never_fails_when_submitted_computation_raises(monkeypatch):
    assert _ask(monkeypatch, _Mem(cfg_raises=True), claims=_claims(S))["memory_submitted"] == 0
