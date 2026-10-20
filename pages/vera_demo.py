"""Streamlit multipage entry: VERA-DAR demo (M7). All logic lives in vera/m7/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from vera.disclosure import render_disclosure  # noqa: E402
from vera.m7.data import load_latest_report  # noqa: E402
from vera.m7.demo_ui import render_demo  # noqa: E402
from vera.sources_sidebar import render_nav, render_sources_sidebar  # noqa: E402

st.set_page_config(page_title="VERA-DAR example report", layout="wide")
render_nav(st)
render_sources_sidebar(st)
render_disclosure(st)  # item #74
_rep, _ = load_latest_report()  # the loader's note can carry an env-var name, a schema name or exception text: never shown
render_demo(_rep, "Example report" if _rep.get("is_fixture") else "Stored report")
