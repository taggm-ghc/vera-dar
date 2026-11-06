"""Streamlit multipage entry: VERA-DAR How to use page (OD-1029-UI); logic is in vera/help_pages.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from vera.help_pages import render_how_to_use  # noqa: E402
from vera.sources_sidebar import render_nav, render_sources_sidebar  # noqa: E402

st.set_page_config(page_title="VERA-DAR How to use", layout="wide")
render_nav(st)
render_sources_sidebar(st)
render_how_to_use(st)
