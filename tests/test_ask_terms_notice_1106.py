"""H14 (OD-1106-HITL): one short Terms notice under the Ask box, wording from config, no consent claim."""
from unittest.mock import MagicMock

from vera import help_pages


def test_config_wording_short_and_no_consent_claim():
    label = help_pages.load()["ask_terms_notice"]["label"]
    assert label == "By asking, you agree to the Terms"
    assert len(label) < 60 and "consented" not in label.lower()


def test_notice_links_to_terms_page_with_config_label():
    st = MagicMock()
    help_pages.render_ask_terms_notice(st, {"ask_terms_notice": {"label": "X Terms"}})
    st.page_link.assert_called_once_with(help_pages.TERMS_PAGE, label="X Terms")


def test_notice_falls_back_to_default_and_to_caption():
    st = MagicMock()
    st.page_link.side_effect = RuntimeError("no page")
    help_pages.render_ask_terms_notice(st, None)
    st.caption.assert_called_once_with(help_pages.ASK_TERMS_DEFAULT + ".")


def test_ask_page_source_calls_notice_after_form():
    src = open(help_pages.HELP_CONFIG_PATH.parent.parent / "streamlit_app.py", encoding="utf-8").read()
    assert src.index('st.form_submit_button("Ask"') < src.index("render_ask_terms_notice(st)")
