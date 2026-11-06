"""Streamlit multipage entry: VERA-DAR Terms of Access, Browsing, and Use (PRIVACY-1104), including the privacy notice.
Logic is in vera/help_pages.py; prose is in config/ui_help_pages.json `terms`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from vera.help_pages import render_terms  # noqa: E402
from vera.sources_sidebar import render_nav, render_sources_sidebar  # noqa: E402

st.set_page_config(page_title="VERA-DAR Terms of Access, Browsing, and Use", layout="wide")
render_nav(st)
render_sources_sidebar(st)
render_terms(st)
