"""Required view (demo-1022/T): the Trace Eval page always shows a sanitized, no-secrets view. Offline; no provider, network or DB."""
import json
import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

from vera import trace_viz

ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "eval_results" / "trace_eval_historical_public.json"
VAL = ROOT / "eval_results" / "trace_eval_v1_validation.json"
_QF = ROOT / "config" / "trace_questions_v1.json"  # gitignored; the leak checks use it when present
QUESTION_TEXTS = [q["text"] for q in json.loads(_QF.read_text())["items"]] if _QF.is_file() else []


def _page(at):
    return "\n".join([*(m.value for m in at.markdown), *(c.value for c in at.caption), *(w.value for w in at.warning),
                      *(i.value for i in at.info), *(s.value for s in at.subheader)])


def _app(public, hist, offline, val):
    import functools
    from vera import trace_viz as tv
    from vera.trace_eval.ui import render_trace_eval
    orig = tv.public_view
    tv.public_view = functools.partial(orig, historical_path=hist, offline_path=offline, commit="b" * 40)
    try:
        render_trace_eval("/nonexistent/r.json", val, public)
    finally:
        tv.public_view = orig


def _render(public, hist, offline="/nonexistent/o.json", val="/nonexistent/v.json"):
    return AppTest.from_function(_app, args=(str(public), str(hist), str(offline), str(val)), default_timeout=30).run()

def test_sanitizer_drops_hostile_metadata():
    out = trace_viz._project_meta({"commit": "x y; rm -rf", "provider": "https://h.example/k", "mode": "weird", "models": ["ok-1", "bad model!"],
                                   "n": "5", "offline": "yes", "run_id": "ok_1"})
    assert out["commit"] is None and out["provider"] is None and out["mode"] is None and out["models"] == ["ok-1"]
    assert out["n"] is None and out["offline"] is None and out["run_id"] == "ok_1"
def test_historical_labelling_rule():
    meta = {"deployed_version": "b" * 40, "commit": "a" * 40}
    assert trace_viz.is_historical(meta, "bbbbbbb") is False
    assert trace_viz.is_historical(meta, "ccccccc") is True
    assert trace_viz.is_historical(meta, "unknown") is True
    assert trace_viz.is_historical(None, "bbbbbbb") is True
    assert trace_viz.is_historical({**meta, "offline": True}, "bbbbbbb") is True
def test_page_no_secrets_no_run_shows_shipped_historical_aggregates():
    at = _render("/nonexistent/p.json", ROOT / "eval_results" / "trace_eval_historical_public.json",
                 ROOT / "eval_results" / "trace_eval_offline_public.json", ROOT / "eval_results" / "trace_eval_v1_validation.json")
    text = _page(at)
    assert not at.exception
    assert "HISTORICAL (TRACE v1, 2026-10-04, earlier abstract-grounded variant, build not recorded, superseded)" in text
    assert "Method and limitations" in text and "Check accuracy, not answer pass rates" in text
    # the offline summary is optional: shown only when its static file ships (asserted in test_trace_auto.py)
    assert ("Offline suite" in text) == (ROOT / "eval_results" / "trace_eval_offline_public.json").is_file()
    assert len(at.dataframe) + len(at.table) >= 1  # aggregates shown without any key, secret or live run (DDFIX-1023 TEVAL: st.table)
    for q in QUESTION_TEXTS:
        assert q not in text
def test_page_never_blank_with_nothing_at_all():
    at = _render("/nonexistent/p.json", "/nonexistent/h.json")
    text = _page(at)
    assert not at.exception and "Method and limitations" in text and "Checker agreement with human labels (true-negative rate)" in text
def test_tnr_shown_as_counts_with_few_labels_note():
    from vera.trace_eval.ui import TNR_NOTE, TNR_SECTION, _tnr_cell
    assert _tnr_cell(6, 2, 0.75) == "6/8 (75%)" and _tnr_cell(1, 1, 0.5) == "1/2 (50%)" and _tnr_cell(0, 0, None) == "n/a"
    at = _render("/nonexistent/p.json", ROOT / "eval_results" / "trace_eval_historical_public.json", val=ROOT / "eval_results" / "trace_eval_v1_validation.json")
    text = _page(at)
    assert TNR_SECTION in text and "not answer pass rates" in text and TNR_NOTE.split(".")[0] in text
    cells = [str(v) for df in [*at.dataframe, *at.table] for v in df.value.get("TNR", [])]  # DDFIX-1023 TEVAL: st.table
    assert cells and all(re.match(r"^(n/a|\d+/\d+( \(\d+%\))?)$", c) for c in cells), cells
def test_shipped_historical_metadata():
    m = json.loads((ROOT / "eval_results" / "trace_eval_historical_public.json").read_text())["run_meta"]
    assert m["mode"] == "historical" and m["finished_at"] == "2026-10-04" and m["models"] == ["gpt-4.1-nano"]
    assert m["commit"] == "unknown" and m["deployed_version"] == "unknown"
    d = json.loads((ROOT / "eval_results" / "trace_eval_historical_public.json").read_text())
    assert "heldout" in next(iter(d["variants"].values()))["by_split"]  # owner accepted publishing held-out aggregates


def test_shipped_historical_file_is_sanitized_and_ignore_safe():
    t = HIST.read_text()
    d = json.loads(t)
    assert d["schema"] == trace_viz.SCHEMA and "traces" not in d
    for bad in ("http", "prompt", "messages", "Synthetic"):
        assert bad not in t, bad
    for q in QUESTION_TEXTS:
        assert q not in t


def test_shipped_historical_label_on_page_has_no_model_name():
    text = _page(_render("/nonexistent/p.json", HIST, val=VAL))
    assert "HISTORICAL (TRACE v1, 2026-10-04, earlier abstract-grounded variant, build not recorded, superseded)" in text
    assert "nano" not in text.lower() and "gpt" not in text.lower()
    for q in QUESTION_TEXTS:
        assert q not in text
