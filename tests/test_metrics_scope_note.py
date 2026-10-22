from pathlib import Path

SRC = (Path(__file__).resolve().parent.parent / "streamlit_app.py").read_text()


def test_metrics_scope_note_defined_and_rendered():
    assert "METRICS_SCOPE_NOTE = (" in SRC
    assert "st.caption(METRICS_SCOPE_NOTE)" in SRC
    assert SRC.index('st.metric("Cost"') < SRC.index("st.caption(METRICS_SCOPE_NOTE)")
