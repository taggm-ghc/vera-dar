# Streamlit UI for VERA-DAR `/ask` Endpoint
import os
from pathlib import Path

import requests
import streamlit as st

from vera.disclosure import render_disclosure
from vera.env_files import load_vera_env
from vera.discovery_panel import render_discovery
from vera.memory_panel import render_memory_panel
from vera.memory_review_panel import render_memory_review
from vera.public_mode import is_public_mode, post_ask, upstream_base_url
from vera.sources_sidebar import render_nav, render_sources, render_sources_sidebar
from vera.ui_safety import safe_markdown
from vera.version_info import UNKNOWN, deployed_commit, short_commit

load_vera_env()  # PLAN-ENV: same loader as main.py (once per process across reruns)

# run.sh records its actual host:port here on every start (dev-mode default URL only).
_LOCAL_URL_FILE = Path(__file__).resolve().parent / ".vera-local-url"

# Mode is read once per script run from VERA_PUBLIC_MODE; unset means PUBLIC (safe default).
PUBLIC = is_public_mode()

# Server-side only. Never passed as a widget value, never stored in st.session_state, never
# rendered. In public mode this is the only key the UI uses; in dev mode it is NOT used at all
# (a dev types their own key), so a user-editable URL can never receive the server key.
_SERVER_KEY = os.getenv("VERA_API_KEY", "")

def _show_triage(base_url: str) -> None:
    """Item #83: ordered, host-free triage after a failed call (ported from AI-Internship)."""
    from vera.api_triage import earliest_failure, triage_api

    with st.status("Checking the connection to VERA-DAR's API…", expanded=True) as box:
        steps = triage_api(base_url, "/ask")
        for step in steps:
            st.write(("✅ " if step["ok"] else "❌ ") + f"{step['step']}: {step['detail']}")
        first = earliest_failure(steps)
        box.update(label=f"Earliest failure: {first['step']}" if first else "The API is reachable now; ask again",
                   state="error" if first else "complete")


st.title("VERA-DAR - Question Answering Service")
render_disclosure(st)  # item #74: shown in public and dev mode

render_nav(st)  # K3.I: plain page-link labels; the default file-name list is hidden by .streamlit/config.toml

if PUBLIC:
    api_base_url = upstream_base_url(local_url_file=None)
    api_key = _SERVER_KEY
else:
    st.sidebar.header("Settings (local/dev mode)")
    api_base_url = st.sidebar.text_input(
        "API Base URL", value=upstream_base_url(local_url_file=_LOCAL_URL_FILE),
        help="Local/dev only. The key typed below is sent to this URL.")
    api_key = st.sidebar.text_input(
        "API Key (X-API-Key)", value="", type="password",
        help="Blank by default. Type the key for the API above; it is never pre-filled.")
    from vera.api_kickstart import ensure_api  # UI-1: dev only; gated by VERA_UI_AUTOSTART_API=1 + default local URL
    if "vera_api_kick" not in st.session_state:  # once per session: no 2s health block / respawn on reruns
        st.session_state["vera_api_kick"] = ensure_api(api_base_url, notify=st.sidebar.info)
    if not st.session_state["vera_api_kick"].ok:
        st.sidebar.warning(st.session_state["vera_api_kick"].message)
    if st.session_state["vera_api_kick"].status == "foreign_listener":  # Q2-6 / V1: never send the key to a non-VERA-DAR listener
        api_key = ""
    render_memory_review(st, api_base_url)  # item #84 D6: operator review, dev mode only (never in public mode)

# Force bad demo toggle
render_sources_sidebar(st)
st.sidebar.subheader("Guardrail Demo")
force_bad = st.sidebar.checkbox("Show a rejected-answer demonstration", value=False)

# Question input
st.subheader("Ask a Question")
question = st.text_input(
    "Enter your research question:",
    placeholder="What is VERA-DAR?",
    help="Ask anything. The service will return an answer with token usage and cost."
)

# Ask button
if st.button("Ask", type="primary"):
    if not question:
        st.warning("⚠️ Please enter a question.")
    elif not api_key:
        st.warning("The service is not configured." if PUBLIC else "No API key set. Type one in the sidebar.")
    else:
        with st.spinner("🤔 Thinking..."):
            try:
                # Make API request
                response = post_ask(api_base_url, question, "force_bad" if force_bad else "normal", api_key)

                if response.status_code == 200:
                    data = response.json()
                    # K2.I5 (fail-closed): the green Answer block only for an evidentiary answer; every other
                    # value (missing, unknown, reading_list, decline, ...) renders the reading-list branch.
                    if data.get("result_kind") == "answer":
                        if (data.get("claim_check") or {}).get("answer_completeness") == "partial":  # D8: never styled as complete
                            st.markdown("### ⚠️ Partial answer")
                            st.warning(safe_markdown(data["answer"]))
                        else:
                            st.markdown("### ✅ Answer")
                            st.success(safe_markdown(data["answer"]))
                        render_sources(st, data.get("sources") or [], data.get("traced_sources") or [], data.get("claim_check"))

                        # Display recalled findings from memory
                        render_memory_panel(st, data)
                    else:
                        render_discovery(st, data)

                    # Display metrics
                    st.markdown("### 💰 Metrics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Tokens Used", f"{data['tokens_used']:,}")
                    with col2:
                        st.metric("Cost", f"${data['cost_usd']:.6f}")

                    # K5.2.I: raw JSON is a dev-mode (owner debug) view only; public mode renders none.
                    if not PUBLIC:
                        with st.expander("📊 Full Response (JSON)"):
                            st.json(data)

                    # Next question suggestion
                    st.info("💡 Try asking about your capstone topic!")

                elif response.status_code == 429:
                    st.error("Rate limit reached. Please try again later.")

                elif PUBLIC:
                    st.error("The service could not answer this request. Please try again later.")
                    if response.status_code >= 500:  # item #83: find the earliest failing step
                        _show_triage(api_base_url)

                elif response.status_code == 401:
                    st.error("API key error. Check the key typed in the sidebar matches the server's VERA_API_KEY.")

                else:
                    st.error(f"Error {response.status_code}")

            except requests.exceptions.RequestException as e:
                if PUBLIC:
                    st.error("The service is unavailable. Please try again later.")
                    _show_triage(api_base_url)
                else:
                    st.error(f"Request failed: {type(e).__name__}")
                    st.info("Make sure the API is running: `./run.sh`")
else:
    st.info("👆 Click 'Ask' to get started.")


def _footer_commit(base_url: str) -> str:
    """Short deployed commit: this process's env first, else the API's /version (short timeout, never raises)."""
    commit = deployed_commit()
    if commit == UNKNOWN:
        try:
            commit = requests.get(f"{base_url.rstrip('/')}/version", timeout=3).json().get("commit", UNKNOWN)
        except Exception:  # noqa: BLE001  (footer only; never surface the error or the URL)
            commit = UNKNOWN
    return short_commit(commit)


st.caption(f"VERA-DAR build: {_footer_commit(api_base_url)}")
