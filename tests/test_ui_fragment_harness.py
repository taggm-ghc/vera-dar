"""PROG-1027 W4b-4 (partial): how AppTest (Streamlit 1.40.1) treats a `run_every` fragment, recorded as a test.

Finding (this file asserts it): a run_every fragment executes exactly once per script run under AppTest; no client timer
exists, so no tick follows by itself. Fragment-tick behaviour in a real client (incremental labels, completion only after
the holder is done, one POST across a rerun) is therefore NOT covered here: the spike2 websocket harness was not
re-run in this clone, so those behaviours are "untested (assumed)" until the owner smoke (plan s8, R3-PROG-Q-5, R6).
"""
import time

from streamlit.testing.v1 import AppTest

SCRIPT = """
import streamlit as st
st.session_state.setdefault("runs", 0)
st.session_state.runs += 1
@st.fragment(run_every=0.1)
def f():
    st.session_state["ticks"] = st.session_state.get("ticks", 0) + 1
    st.caption("tick")
f()
"""


def test_apptest_runs_a_run_every_fragment_once_per_script_run_and_never_ticks_alone():
    at = AppTest.from_string(SCRIPT, default_timeout=10).run()
    assert not at.exception
    assert at.session_state["ticks"] == 1
    time.sleep(0.5)  # a real client would have fired ~5 timer reruns by now
    assert at.session_state["ticks"] == 1
    at.run()
    assert at.session_state["ticks"] == 2 and at.session_state["runs"] == 2
