"""ANS-LAYERS-1101 Stage 1 (UI half): layered result rendering. Offline; AppTest with a stubbed /ask."""
import pytest
from streamlit.testing.v1 import AppTest

from tests.test_ui_render_extract import _answer, ROOT
from vera import ui_evidence as ue
from vera.sources_sidebar import plain
import tests.test_ui_render_extract as base


def _texts(at):
    return [e.value for e in at.markdown] + [e.value for e in at.caption] + [e.value for e in at.warning] + [e.value for e in at.success]


def test_claim_check_wording_states_scope():
    cc = _answer()["claim_check"]
    t = ue.claim_check_text(cc, None, True, layered=True)
    assert t.startswith("Citation check: 1 of 1 cited claims are supported")
    assert "not whether the question was answered" in t
    assert "Claim check" not in t


def test_claim_check_legacy_wording_unchanged():
    assert ue.claim_check_text(_answer()["claim_check"], None, True).startswith("Claim check: 1 cited claims")


def test_has_layers_and_labels():
    assert not ue.has_layers({"answer": "x"})
    assert ue.has_layers({"answer_status": "complete"})
    assert ue.answer_status_label({"answer_status": "partial"}) == "Partial answer"
    assert ue.answer_status_label({"answer_status": "not_answered"}) == "Limited evidence"
    assert ue.answer_status_label({"answer_status": "complete"}) is None
    assert ue.answer_status_label({"answer_status": "weird"}) is None


def test_note_texts_tolerates_shapes():
    assert ue._note_texts(["a", {"kind": "k", "text": "b"}, 3, None, " "]) == ["a", "b"]
    assert ue._note_texts(None) == []


LIMIT = "The evidence found covers one specialised context."
NOTE = "Freshness: sources were checked today."
NOTE_MD = plain(NOTE)  # process notes render through plain(): the element holds Markdown-escaped source text


def _layered(status="complete", **kw):
    fields = {"limitations": [LIMIT], "process_notes": [{"kind": "freshness", "text": NOTE}]}
    fields.update(kw)  # a caller may override limitations/process_notes
    return _answer(answer_status=status, **fields)


def _all(at):
    return [e.value for e in list(at.markdown) + list(at.caption) + list(at.warning) + list(at.success)]


def _exp(at, label):
    return next(e for e in at.expander if e.label == label)


def test_layered_complete(monkeypatch):
    at = base._run(monkeypatch, _layered("complete"))
    assert at.success[0].value == "Batching reduces latency [1]."
    assert NOTE not in at.success[0].value
    assert any(LIMIT in v for v in _all(at))
    labels = [e.label for e in at.expander]
    assert "Research and verification" in labels and "Technical details" not in labels  # public mode
    ver = _exp(at, "Research and verification")
    assert any(NOTE_MD in c.value for c in ver.caption), "process note caption missing from 'Research and verification'"
    assert any(c.value.startswith("Citation check:") and "not whether the question was answered" in c.value for c in ver.caption)
    assert not any(c.value.startswith(("Claim check", "Citation check")) for c in at.caption if c not in list(ver.caption))
    assert not at.metric and not at.json
    assert ver.proto.expanded is False
    assert not any(NOTE_MD in v or NOTE in v for v in _all(at) if v not in [c.value for c in ver.caption])


@pytest.mark.parametrize("status,label", [("partial", "Partial answer"), ("not_answered", "Limited evidence")])
def test_layered_partial_label(monkeypatch, status, label):
    at = base._run(monkeypatch, _layered(status))
    assert any(label in m.value for m in at.markdown)
    assert not at.success
    assert at.warning[0].value == "Batching reduces latency [1]."
    assert not any("### ✅" in m.value for m in at.markdown)


def test_fallback_without_new_fields(monkeypatch):
    at = base._run(monkeypatch, _answer())
    assert at.success[0].value == "Batching reduces latency [1]."
    assert not any(e.label in ("Research and verification", "Technical details") for e in at.expander)
    assert any(c.value.startswith("Claim check:") for c in at.caption)
    assert at.metric


def test_layered_no_per_source_or_routing_in_layers_3_4(monkeypatch):
    at = base._run(monkeypatch, _layered("complete"))
    for lab in ("Research and verification",):
        blob = " ".join(e.value for e in list(_exp(at, lab).caption) + list(_exp(at, lab).markdown))
        assert base.TITLE not in blob and base.URL not in blob
        assert "gate" not in blob.lower()


def _run_dev(monkeypatch, body):
    """Dev mode reads the API key from the sidebar input, not VERA_API_KEY; base._run(public=False) never types one,
    so the page stops at 'No API key set' (the golden answer_dev_mode captured that state). Type it, then ask."""
    import copy, requests
    from types import SimpleNamespace
    from vera import ui_wake
    monkeypatch.setenv("VERA_PUBLIC_MODE", "0")
    monkeypatch.setattr(base.sb, "_corpus", lambda st: None)
    monkeypatch.setattr(requests, "post", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: copy.deepcopy(body)))
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: {"commit": "abcdef1234"}))
    monkeypatch.setattr(ui_wake, "http_get", lambda *a, **k: SimpleNamespace(status_code=200))
    # The stubbed GET is not a VERA-DAR /health, so ensure_api would report foreign_listener and (correctly, Q2-6)
    # blank the key. Report a healthy local API instead.
    from vera import api_kickstart
    monkeypatch.setattr(api_kickstart, "ensure_api", lambda *a, **k: api_kickstart.KickResult("ok", "", True))
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    next(t for t in at.sidebar.text_input if t.label == "API Key (X-API-Key)").set_value("k")
    next(t for t in at.main.text_input).set_value("render fixture question")
    at.button[0].click()
    at.run()
    assert not at.exception
    return at


def test_dev_mode_has_collapsed_technical_details(monkeypatch):
    at = _run_dev(monkeypatch, _layered("complete"))
    assert any(e.value == "Batching reduces latency [1]." for e in at.success), "dev-mode answer not rendered"
    labels = [e.label for e in at.expander]
    assert "Technical details" in labels, labels
    tech = _exp(at, "Technical details")
    assert any(m.label == "Tokens Used" for m in tech.metric)
    assert tech.proto.expanded is False
    # at.metric is recursive (includes the expander's own metrics): no metric may sit outside "Technical details"
    assert tech.metric and len(at.metric) == len(tech.metric), "metrics rendered outside the Technical details expander"


def test_public_mode_omits_expander_when_nothing_to_explain(monkeypatch):
    b = _layered("complete", process_notes=[])
    b["claim_check"] = {}
    at = base._run(monkeypatch, b)
    assert not any(e.label in ("Research and verification", "Technical details") for e in at.expander)


@pytest.mark.parametrize("status", ["unassessed", "bogus"])
def test_unassessed_and_unknown_are_neutral(monkeypatch, status):
    at = base._run(monkeypatch, _layered(status))
    assert not at.success and not at.warning
    assert any(i.value == "Batching reduces latency [1]." for i in at.info)
    assert not any("✅" in m.value for m in at.markdown)


def test_imported_names_exist():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    tree = ast.parse((root / "streamlit_app.py").read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("vera."):
            src = ast.parse((root / (n.module.replace(".", "/") + ".py")).read_text())
            defined = {x.name for x in ast.walk(src) if isinstance(x, (ast.FunctionDef, ast.ClassDef))}
            for t in ast.walk(src):
                if isinstance(t, ast.Assign):
                    defined |= {a.id for a in t.targets if isinstance(a, ast.Name)}
                if isinstance(t, (ast.Import, ast.ImportFrom)):
                    defined |= {(a.asname or a.name).split(".")[0] for a in t.names}
            missing = [a.name for a in n.names if a.name not in defined]
            assert not missing, (n.module, missing)


def test_public_nonanswer_outcome_sentence_static():
    """Static (streamlit not installed): _layer3_other shows the research-outcome sentence outside the PUBLIC guard."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "streamlit_app.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_layer3_other")
    body = ast.get_source_segment(src, fn)
    assert "_research_outcome_text(data)" in body and "Research outcome:" in body
    # the outcome caption is not nested under `if not PUBLIC`
    for n in ast.walk(fn):
        if isinstance(n, ast.If) and ast.unparse(n.test) == "not PUBLIC":
            assert "Research outcome:" not in ast.get_source_segment(src, n)
    # the owner-only captions no longer repeat it
    assert "_render_research_captions(data, outcome=False)" in body


def test_stream_path_renders_layers(monkeypatch):
    """Streaming ON: the page takes /ask/stream; the terminal 200 result body renders through the same layered
    render_result (answer box, limitations visible, process notes collapsed)."""
    import time
    from types import SimpleNamespace
    import requests
    from vera import progress, public_mode, ui_wake
    monkeypatch.setattr(progress, "stream_enabled", lambda: True)
    body = _layered("complete")
    lines = [{"v": "1.3", "type": "hb", "seq": 1, "elapsed_s": 1},
             {"v": "1.3", "type": "result", "status": 200, "body": body}]
    monkeypatch.setattr(public_mode, "iter_ask_stream", lambda *a, **k: iter(lines))
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    monkeypatch.setenv("VERA_API_KEY", "k")
    monkeypatch.setattr(base.sb, "_corpus", lambda st: None)
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: {"commit": "abcdef1234"}))
    monkeypatch.setattr(ui_wake, "http_get", lambda *a, **k: SimpleNamespace(status_code=200))
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    at.text_input[0].set_value("render fixture question")
    at.button[0].click()
    at.run()
    for _ in range(30):  # bounded: the worker thread finishes, then a rerun renders the terminal result once
        if at.success:
            break
        time.sleep(0.2)
        at.run()
    assert not at.exception
    assert at.success and at.success[0].value == "Batching reduces latency [1]." and NOTE not in at.success[0].value
    assert any(LIMIT in v for v in _all(at)), "limitations not in the visible section"
    ver = _exp(at, "Research and verification")
    assert any(NOTE_MD in c.value for c in ver.caption), "process note missing from the collapsed section"
    assert ver.proto.expanded is False
    assert not any(NOTE_MD in v or NOTE in v for v in _all(at) if v not in [c.value for c in ver.caption])
