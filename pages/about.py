"""Streamlit multipage entry: VERA-DAR About page (OD-1029-UI). The full disclosure lives here; logic is in vera/help_pages.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from vera.disclosure import render_disclosure  # noqa: E402
from vera.help_pages import render_about  # noqa: E402
from vera.sources_sidebar import render_nav, render_sources_sidebar  # noqa: E402

st.set_page_config(page_title="VERA-DAR About", layout="wide")
render_nav(st)
render_sources_sidebar(st)
render_disclosure(st)  # item #74: the required notice on every page
render_about(st)
