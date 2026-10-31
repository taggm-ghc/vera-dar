"""TRACE-1028: pins the revisions section of the Trace Eval page (statuses derive from config; no AppTest)."""
import copy
import re
from unittest import mock

from vera.trace_eval import ui


def _by(cfgs=None):
    return {f: (s, t) for f, s, t in ui.revision_items(cfgs)}


def test_shipped_config_statuses():
    by = _by()
    for f in ("Category table in the database", "Topic registry in the database"):
        assert by[f][0].startswith("OFF"), f
    assert by["Topic equivalence judge"][0] == "OFF"
    assert by["Semantic similarity"][0] == "OFF"
    assert by["Progressive answer updates"][0] == "PLANNED"
    assert "SHADOW" in by["Unknown-category step"][0]
    assert by["Licence observations"][0].startswith("batch loader LIVE")  # fulltext_ingest.json true


def test_derived_rows_follow_config():
    base = ui._read_cfgs()
    by = _by(base)
    assert by["Scope handling"][0] == "LIVE"
    assert by["Safety patterns"][0] == "LIVE"
    assert by["Research topics"][0] == "LIVE (fixed seed)"
    c = copy.deepcopy(base)
    c["scope"]["offtopic_reading_list"] = False
    c["scope"]["injection_separation"] = False
    c["acq"]["micro_topic"]["registry_table"]["read_from_db"] = True
    by = _by(c)
    assert by["Scope handling"][0] == "OFF"
    assert by["Safety patterns"][0] == "LIVE; instruction separation OFF"
    assert by["Research topics"][0] == "LIVE (database)"
    c["scope"]["layer1"]["injection_patterns"] = []
    assert _by(c)["Safety patterns"][0] == "UNKNOWN"


def test_statuses_follow_config_not_text():
    base = ui._read_cfgs()
    c = copy.deepcopy(base)
    c["acq"]["micro_topic"]["registry_table"]["read_from_db"] = True
    c["sem"]["enabled"] = True
    c["sem"]["shadow"] = True
    assert _by(c)["Topic registry in the database"][0].startswith("LIVE")
    assert _by(c)["Semantic similarity"][0] == "SHADOW"
    empty = {"acq": {}, "scope": {}, "ingest": {}, "sem": {}}
    for f, (s, _) in _by(empty).items():
        assert not s.startswith("LIVE"), f  # unreadable config never shows LIVE (no hardcoded LIVE rows)


def test_text_has_no_internals_lists_or_admitted():
    items = ui.revision_items()
    text = " ".join([ui.REVISIONS_INTRO, ui.REVISIONS_NOTE] + [" ".join(i) for i in items])
    assert not re.search(r"render|groq|postgres|fastembed|bge|api key|localhost|https?://|\.json|\.sql|migration \d", text, re.I)
    assert "admit" not in text.lower()


def test_render_emits_section():
    st = mock.MagicMock()
    with mock.patch.object(ui, "st", st):
        ui._render_revisions()
    assert st.subheader.call_args[0][0] == ui.REVISIONS_SECTION
    assert st.table.called


def test_unknown_label_wording_follows_step():
    base = ui._read_cfgs()
    on = _by(base)["Question categories"]
    assert "researched, not refused" in on[1] and "safe decline" not in on[1]
    off = copy.deepcopy(base)
    off["scope"]["unknown_category_step"]["enabled"] = False
    assert "safe decline" in _by(off)["Question categories"][1]
    off["scope"].pop("unknown_category_step")
    assert _by(off)["Question categories"][0] == "UNKNOWN"
