"""P0-2 fix 2: no 'No trace evaluation results yet' under a displayed public run; sidebar reads the public file."""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from vera import sources_sidebar as sb
from vera.trace_eval.ui import MISSING_RESULTS_TEXT

ROOT = Path(sb.__file__).resolve().parent.parent
HIST = ROOT / "eval_results" / "trace_eval_historical_public.json"


def _app(results_path, validation_path, public_path):
    from vera.trace_eval.ui import render_trace_eval
    render_trace_eval(results_path, validation_path, public_path, base_url="http://upstream.invalid:9")


def _page(at):
    return "\n".join(repr(el.proto) for el in list(at.main))


def _at(tmp_path, public):
    return AppTest.from_function(_app, kwargs={"results_path": str(tmp_path / "none.json"),
                                               "validation_path": str(tmp_path / "nov.json"),
                                               "public_path": str(public)}, default_timeout=30).run()


def test_no_empty_state_when_public_run_displayed(tmp_path):
    at = _at(tmp_path, HIST)  # historical file is shipped; public_path absent -> falls back to it
    text = _page(at)
    assert not at.exception
    assert "Latest run" in text
    assert MISSING_RESULTS_TEXT not in text


def test_empty_state_kept_when_no_run_anywhere(tmp_path):
    from vera import trace_viz
    from unittest import mock
    with mock.patch.object(trace_viz, "HISTORICAL_PATH", tmp_path / "h.json"):
        at = _at(tmp_path, tmp_path / "p.json")
    assert MISSING_RESULTS_TEXT in _page(at)


def test_sidebar_snapshot_reads_public_file(monkeypatch):
    s = sb.snapshot_summary()
    d = json.loads(HIST.read_text())
    assert s and s["count"] is None and s["questions"] == d["n_questions"] and s["captured"] == d["generated_at"][:10]
    assert set(s) <= sb.SNAPSHOT_FIELDS


def test_sidebar_caption_not_unavailable_with_public_file():
    from unittest import mock
    st = mock.MagicMock()
    with mock.patch.object(sb, "render_corpus"), mock.patch.object(sb, "_corpus"):
        sb.render_sources_sidebar(st)
    caps = " ".join(str(c.args[0]) for c in st.caption.call_args_list)
    assert "not available" not in caps.split("Trace Eval")[-1]
    assert "per-source counts are not published" in caps


def test_explicit_old_path_still_works(tmp_path):
    p = tmp_path / "v1.json"
    p.write_text(json.dumps({"traces": [{"id": "q1", "sources": [{"url": "https://arxiv.org/abs/2401.00001"}]}]}))
    s = sb.snapshot_summary(p)
    assert s["count"] == 1 and s["questions"] == 1
