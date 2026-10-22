"""Streamlit multipage entry: VERA-DAR report page. Live mode (default) runs a real /ask through the Ask page's client
(vera.public_mode.post_ask) and renders vera/m7/live_report.py; the offline mode keeps the stored evaluation report or
the clearly labelled illustrative fixture (vera/m7/data.py + demo_ui.py). All logic lives in vera/m7/."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402
import streamlit as st  # noqa: E402

from vera import public_mode  # noqa: E402
from vera.disclosure import ask_help_text, render_disclosure  # noqa: E402
from vera.env_files import load_vera_env  # noqa: E402
from vera.m7 import live_report as lr  # noqa: E402
from vera.m7.data import load_latest_report  # noqa: E402
from vera.m7.demo_ui import render_demo  # noqa: E402
from vera.sources_sidebar import render_nav, render_sources_sidebar  # noqa: E402

load_vera_env()  # same loader as streamlit_app.py

st.set_page_config(page_title="VERA-DAR report", layout="wide")
render_nav(st)
render_sources_sidebar(st)
render_disclosure(st)  # item #74

LIVE, OFFLINE = "Live report (ask VERA-DAR)", "Offline example (illustrative, not a real run)"
PUBLIC = public_mode.is_public_mode()
_LOCAL_URL_FILE = Path(__file__).resolve().parent.parent / ".vera-local-url"

mode = st.radio("Report source", (LIVE, OFFLINE), horizontal=True)

if mode == OFFLINE:
    st.warning(lr.FIXTURE_FALLBACK_NOTE)
    _rep, _ = load_latest_report()  # the loader's note can carry an env-var name, a schema name or exception text: never shown
    render_demo(_rep, "Example report" if _rep.get("is_fixture") else "Stored report")
else:
    st.title("VERA-DAR report")
    typed = ""
    if not PUBLIC:  # dev mode: a typed key only, never pre-filled; the server key is not used (as on the Ask page)
        typed = st.sidebar.text_input("API Key (X-API-Key)", value="", type="password",
                                      help="Local/dev only. Blank by default; never pre-filled.")
    base_url, api_key = lr.resolve_api(PUBLIC, local_url_file=_LOCAL_URL_FILE, typed_key=typed)  # never rendered
    r = lr.redactor(base_url)
    recent = st.session_state.get(lr.RECENT_KEY, [])
    question = st.text_input("Research question for a live report:", placeholder="What is VERA-DAR?",
                             help=ask_help_text())
    if st.button("Run report", type="primary"):
        if not question:
            st.warning("Please enter a question.")
        elif not api_key:
            st.warning("The service is not configured." if PUBLIC else "No API key set. Type one in the sidebar.")
        else:
            with st.spinner("Running VERA-DAR..."):
                t0 = time.monotonic()
                try:
                    resp = public_mode.post_ask(base_url, question, "normal", api_key)
                except requests.exceptions.RequestException as e:
                    resp = None
                    st.error("The service is unavailable. Please try again later." if PUBLIC
                             else f"Request failed: {type(e).__name__}")
                latency = time.monotonic() - t0
            if resp is not None:
                if resp.status_code == 200:
                    try:
                        body = resp.json()
                    except ValueError:
                        body = None
                    if isinstance(body, dict):
                        recent = lr.remember(st.session_state, lr.build_live_report(body, question, latency))
                    else:
                        st.error("The service returned a response this page cannot read.")
                elif resp.status_code == 429:
                    st.error("Rate limit reached. Please try again later.")
                else:
                    st.error("The service could not produce a report. Please try again later."
                             + ("" if PUBLIC else f" (HTTP {int(resp.status_code)})"))
    if recent:
        pick = st.selectbox("Reports from this session (newest first)", range(len(recent)),
                            format_func=lambda i: r(recent[i].get("question"))[:120])
        lr.render_live_report(st, recent[pick], r)
    else:
        st.info("Enter a question and click Run report. For an offline illustration, choose the offline example above.")
