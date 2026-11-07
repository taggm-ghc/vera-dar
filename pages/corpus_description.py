"""Streamlit multipage entry: VERA-DAR Corpus Description page (CORPUS-DESC-1107, owner 2026-11-07: the corpus description
is its own page, not part of About). Logic is in vera/help_pages.py render_corpus_page."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from vera.help_pages import render_corpus_page  # noqa: E402
from vera.sources_sidebar import render_nav, render_sources_sidebar  # noqa: E402

st.set_page_config(page_title="VERA-DAR Corpus Description", layout="wide")
render_nav(st)
render_sources_sidebar(st)
render_corpus_page(st)
