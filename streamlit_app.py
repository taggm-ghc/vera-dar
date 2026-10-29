# Streamlit UI for VERA-DAR `/ask` Endpoint
import os
import secrets
from pathlib import Path

import requests
import streamlit as st

from vera.disclosure import ask_help_text, render_disclosure
from vera.env_files import load_vera_env
from vera.discovery_panel import render_discovery
from vera.m7.live_report import INLINE_STATUS
from vera.memory_panel import render_memory_panel
from vera.memory_review_panel import render_memory_review
from vera import progress, progress_ui, ui_wake
from vera.public_mode import get_progress, is_public_mode, post_ask, post_ask_tracked, upstream_base_url
from vera.sources_sidebar import cited_numbers, plain, render_nav, render_sources, render_sources_sidebar
from vera.ui_evidence import (is_fulldoc, render_answer_scope_note, render_citations, render_evidence_scope,
                              render_fallback_reason)
from vera.ui_safety import safe_markdown
from vera.version_info import UNKNOWN, deployed_commit, short_commit

load_vera_env()  # PLAN-ENV: same loader as main.py (once per process across reruns)

# TABTITLE-1023: must be the first st.* call; without it the browser tab falls back to the file name "streamlit_app".
st.set_page_config(page_title="VERA-DAR")

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
    help=ask_help_text()
)

METRICS_SCOPE_NOTE = ("Counts the answer-writing and scope-check model calls for this request, plus the answer checker's "
                      "cost when it is priced. It excludes the checker's tokens and any background indexing. "
                      "Calls to the free-tier provider are recorded at $0, and the answer checker is not priced. 0 tokens means no counted model call ran.")


def render_result(data: dict) -> None:
    """Render one 200 /ask body (W4b-1: pure extract of the former inline block, unchanged output)."""
    # K2.I5 (fail-closed): the green Answer block only for an evidentiary answer; every other
    # value (missing, unknown, reading_list, decline, ...) renders the reading-list branch.
    if data.get("result_kind") == "answer":
        if (data.get("claim_check") if isinstance(data.get("claim_check"), dict) else {}).get("answer_completeness") == "partial":  # D8: never styled as complete
            st.markdown("### ⚠️ Partial answer")
            st.warning(safe_markdown(data.get("answer")))
        else:
            st.markdown("### ✅ Answer")
            st.success(safe_markdown(data.get("answer")))
        full = is_fulldoc(data)
        render_evidence_scope(st, data)  # AE-1020 D3: N documents inspected, M passages admitted
        render_sources(st, data.get("sources") or [], data.get("traced_sources") or [],
                       data.get("claim_check"), fulldoc=full,
                       cited=cited_numbers(data.get("answer")))
        render_citations(st, data)  # AE-1020 D2: short quotes, span refs, CC BY attribution
        if full:
            render_answer_scope_note(st)

        # Display recalled findings from memory
        render_memory_panel(st, data)
    else:
        render_fallback_reason(st, data)  # AE-1020 G7: why there is no answer (no internal token)
        render_discovery(st, data)
        _ia = (data.get("contract") if isinstance(data.get("contract"), dict) else {}).get("inline_acquisition")
        if isinstance(_ia, dict) and _ia:  # P0-2: same status wording as the Report (live_report.INLINE_STATUS)
            st.caption("**New search for this question:** " + INLINE_STATUS.get(
                _ia.get("status"), "A new search was considered (outcome: other)."))
        _rs = (data.get("contract") if isinstance(data.get("contract"), dict) else {}).get("research")
        if isinstance(_rs, dict) and isinstance(_rs.get("user_text") or _rs.get("summary"), str) and (_rs.get("user_text") or _rs.get("summary")).strip():
            st.caption("**Research outcome:** " + plain(_rs.get("user_text") or _rs.get("summary")))  # FRESH-1023 C3
        _st = _rs.get("steps") if isinstance(_rs, dict) else None  # TEVAL: aggregate steps only
        if isinstance(_st, dict) and _rs.get("triggered"):
            _ans = {"generated_new_evidence": "generated from newly found passages selected as possible evidence",
                    "generated_existing_only": "generated from the existing collection only",
                    "abstained": "abstained", "not_attempted": "not attempted"}
            st.caption("**Search steps:** " + f"{int(_st.get('discovered') or 0)} found; search "
                       + ("ran" if _st.get("search") == "ran" else "not run")
                       + "; retrieval " + str(_st.get("acquisition", "not_attempted")).replace("_", " ")
                       + f"; {int(_st.get('admitted') or 0)} passed the initial relevance check; answer "
                       + _ans.get(_st.get("answer"), "not attempted") + ".")
        _dc = _rs.get("diag_status_counts") if isinstance(_rs, dict) else None  # P0DTA-DIAG: aggregate counts only
        if isinstance(_dc, dict) and any(isinstance(v, int) and v > 0 for v in _dc.values()):
            from vera.research_outcome import DIAG_LABELS
            st.caption("**Sources found:** " + "; ".join(f"{int(_dc[k])} {lab}" for k, lab in DIAG_LABELS.items()
                                                      if isinstance(_dc.get(k), int) and _dc[k] > 0))

    # Display metrics
    st.markdown("### 💰 Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tokens Used", f"{int(data.get('tokens_used') or 0):,}")
    with col2:
        st.metric("Cost", f"${float(data.get('cost_usd') or 0):.6f}")
    st.caption(METRICS_SCOPE_NOTE)

    # K5.2.I: raw JSON is a dev-mode (owner debug) view only; public mode renders none.
    if not PUBLIC:
        with st.expander("📊 Full Response (JSON)"):
            st.json(data)

    # Next question suggestion
    st.info("💡 Ask a follow-up question to explore further.")


def _send_question(question: str, force_bad: bool) -> None:
    """Send one /ask (after the wake-up) and render the result."""
    with st.spinner("🤔 Thinking..."):
        try:
            # Make API request
            response = post_ask(api_base_url, question, "force_bad" if force_bad else "normal", api_key)

            if response.status_code == 200:
                data = response.json()
                data = data if isinstance(data, dict) else {}
                render_result(data)

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


def _load_labels():
    """Progress UI labels; any problem keeps the progress path off (spinner path) rather than breaking the page."""
    try:
        return progress_ui.Labels.load()
    except Exception:  # noqa: BLE001  (fixed behavior: flag-off path, no text shown)
        return None


# PROG-1027: flag on only when config/progress.json enabled (and the kill switch is unset) and the labels validate.
LABELS = _load_labels() if progress.enabled() else None
PROGRESS_ON = LABELS is not None
_ses = st.session_state


def _render_terminal(lr) -> None:
    """Final text for a finished tracked run (labels come from config/progress-stages.json; nothing from exceptions)."""
    text = LABELS.terminal_text(lr.terminal, lr.rid)
    if lr.terminal in ("answered", "partial", "no_answer", "declined") and isinstance(lr.result, dict):
        st.caption(text)
        render_result(lr.result)
        if lr.queued > 0:  # background line carries no number; absent until PROG-L's set_queued call
            st.caption(LABELS.background_line)
    elif lr.terminal == "error" and not PUBLIC and lr.status == 401:
        st.error("API key error. Check the key typed in the sidebar matches the server's VERA_API_KEY.")
    else:
        st.error(text)
        # Item #83 (preserved on the flag-on public path): ordered triage after a 5xx or no response; a VERA 429 is the caps text.
        if PUBLIC and lr.terminal != "busy" and (lr.status is None or lr.status >= 500):
            _show_triage(api_base_url)


@st.fragment(run_every=progress.CFG.poll_interval_s if PROGRESS_ON else None)
def _poller(inf) -> None:
    """Client-driven timer (spike2 design B): the main script never blocks; polling ends with the client."""
    def fetch(after):
        return get_progress(api_base_url, api_key, inf.rid, _ses["vera_ptoken"], after)
    if progress_ui.poll_tick(inf, _ses, st, fetch, ui_wake.clock, progress.CFG, LABELS):
        st.rerun()  # full rerun drops run_every and renders the durable result


# Ask button. UIWAKE-1022: the question is queued in session state and sent only once GET /health answers,
# so a sleeping API is woken first and the question survives a rerun during the wait.
_inflight = _ses.get("vera_inflight") if PROGRESS_ON else None
_clicked = st.button("Ask", type="primary", disabled=_inflight is not None)  # evaluated first (R2-PROG-Q-8)
if _clicked and _inflight is not None:  # race: the click landed before the disabled render
    st.info(LABELS.already_running)
elif _clicked:
    if not question:
        st.warning("⚠️ Please enter a question.")
    elif not api_key:
        st.warning("The service is not configured." if PUBLIC else "No API key set. Type one in the sidebar.")
    else:
        ui_wake.queue_question(st.session_state, {"question": question, "force_bad": bool(force_bad)})

if PROGRESS_ON:
    if _ses.get("vera_ptoken") is None:
        _ses["vera_ptoken"] = secrets.token_urlsafe(32)  # per browser session; header-only, never rendered
    if _inflight is None and _ses.get(ui_wake.PENDING_KEY):
        _pending = ui_wake.take_ready_question(st, api_base_url, allow_restart_hook=PUBLIC)
        if _pending:
            _inflight = progress_ui.start_tracked(
                _ses, lambda: secrets.token_hex(16), _ses["vera_ptoken"], ui_wake.clock, post_ask_tracked,
                (api_base_url, _pending["question"], "force_bad" if _pending["force_bad"] else "normal", api_key),
                st, LABELS)
            st.rerun()  # the Sending label was drawn before the worker started; the rerun hands over to the poller
    _lr = _ses.get("vera_last_result")
    if _inflight is not None:
        _poller(_inflight)
    elif _lr is not None and not _lr.rendered:
        _render_terminal(_lr)
        _lr.rendered = True  # only after a render that did not raise (AC6)
    elif not _clicked:
        st.info("👆 Click 'Ask' to get started.")
        ui_wake.page_load_check(st, api_base_url, allow_restart_hook=PUBLIC)
else:
    if st.session_state.get(ui_wake.PENDING_KEY):
        _pending = ui_wake.take_ready_question(st, api_base_url, allow_restart_hook=PUBLIC)
        if _pending:
            _send_question(_pending["question"], _pending["force_bad"])
    elif not _clicked:
        st.info("👆 Click 'Ask' to get started.")
        ui_wake.page_load_check(st, api_base_url, allow_restart_hook=PUBLIC)  # at most once a minute per session


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
