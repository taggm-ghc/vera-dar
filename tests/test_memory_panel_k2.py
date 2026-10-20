"""K2.I5b: memory panel wording for evidence findings (no discovery rendering, no cap or old licence text)."""
from pathlib import Path
from unittest import mock

from vera import disclosure, memory_panel as mp

SRC = Path(mp.__file__).read_text()
ON = ("Findings from earlier sessions that passed VERA-DAR's automated checks against the text they cite, which may be an "
      "abstract or index record rather than the full document, and were confirmed by VERA-DAR's owner (confirmed is not "
      "proof of truth). Never your question or who asked.")
OFF = ("Findings from earlier sessions that passed VERA-DAR's automated checks against the text they cite, which may be an "
       "abstract or index record rather than the full document. They were not individually confirmed by VERA-DAR's owner. "
       "Never your question or who asked.")


def _render(data):
    st = mock.MagicMock()
    mp.render_memory_panel(st, data)
    return st


def test_recalled_caption_confirm_on_exact():
    assert mp._recalled_caption(True) == ON


def test_recalled_caption_confirm_off_exact():
    assert mp._recalled_caption(False) == OFF


def test_caption_never_claims_full_document_checked():
    for c in (ON, OFF):
        assert "may be an abstract or index record rather than the full document" in mp._recalled_caption(c == ON)
        low = mp._recalled_caption(c == ON).lower()
        assert "full paper was" not in low and "full document was checked" not in low


def test_item_date_caption_has_no_saved_word():
    st = _render({"recalled": [{"claim_text": "A finding.", "source_title": "T", "source_url": "https://x.org/a",
                                "saved_at": "2026-10-01"}]})
    caps = [c.args[0] for c in st.caption.call_args_list]
    from vera.sources_sidebar import plain
    assert f"From an earlier session ({plain('2026-10-01')})" in caps
    assert not any("Saved" in c for c in caps)


def test_module_has_no_cap_literal_or_old_licence_sentence():
    for bad in ("1,500", "1500", "icence", "rather than the full papers", "grounded", "abstracts it found", "Saved"):
        assert bad not in SRC, bad


def test_panel_ignores_discovery_fields():
    st = _render({"recalled_discovery": [{"kind": "discovery_note", "note": "n", "urls": ["https://a.org"]}],
                  "discovery_note_submitted": 2})
    assert not st.method_calls


def test_caption_consistent_with_disclosure_evidence_sentence():
    assert "abstract or index record" in ON and "abstract or index record" in OFF
    assert "abstract or index record" in disclosure._EVIDENCE_FINDINGS
