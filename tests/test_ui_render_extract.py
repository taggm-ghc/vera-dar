"""PROG-E W4b-1: the result rendering of streamlit_app.py (render_result) is a pure extract.

Offline: stubbed requests, no DB or network. The golden file was captured from the original (pre-extract) page code;
each fixture's rendered element tree (type plus proto text, main and sidebar) must still equal it byte for byte.
"""
import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from vera import sources_sidebar as sb

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).resolve().parent / "ui_render_extract_golden.json"
TITLE, URL = "Render Fixture Title", "https://example.org/render-fixture"


def _answer(**kw):
    b = {"answer": "Batching reduces latency [1].", "model": "m", "tokens_used": 1234, "cost_usd": 0.001234,
         "result_kind": "answer", "evidence_status": "admitted",
         "sources": [{"n": 1, "title": TITLE, "url": URL, "published": "2024", "identifier": "doi:10/render",
                      "acquisition": "full_document_acquired", "span": "chars=10-90"}],
         "traced_sources": [], "recalled": [],
         "examined_full_documents": [{"document_id": "d1"}],
         "admitted_evidence": [{"span_id": "s1", "document_id": "d1", "span": "chars=10-90"}],
         "claim_check": {"outcome": "checked", "checked": 1, "supported": 1, "partial": 0, "removed": 0,
                         "uncited_dropped": 0, "unchecked_dropped": 0, "verbatim_dropped": 0,
                         "answer_completeness": "complete"},
         "citations": [{"claim_id": 1, "n": 1, "span_id": "s1", "span": "chars=10-90", "verdict": "supported",
                        "quote": "latency falls when requests are batched", "quote_span": "chars=20-60",
                        "attribution": {"creator": "B. Render", "title": TITLE, "source": URL,
                                        "licence": "CC BY 4.0",
                                        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
                                        "changes": "none"}}]}
    b.update(kw)
    return b


def _partial():
    b = _answer(answer="Only part of the question is answered [1].")
    b["claim_check"] = dict(b["claim_check"], answer_completeness="partial",
                            parts={"total": 2, "answered": [1], "unanswered": [2]})
    return b


def _legacy_answer():
    b = _answer(evidence_status=None, citations=[])
    b.pop("evidence_status")
    b["sources"][0].pop("acquisition")
    return b


def _reading_list():
    return {"answer": "", "model": "m", "tokens_used": 77, "cost_usd": 0.0, "result_kind": "reading_list",
            "evidence_status": "insufficient", "examined_full_documents": [{"document_id": "d1"}],
            "admitted_evidence": [],
            "candidates": [{"title": "Candidate Title", "url": "https://example.org/cand",
                            "acquisition": "not_acquired"}],
            "contract": {"layers": [], "fallback": "insufficient_full_document_evidence",
                         "inline_acquisition": {"status": "generated_existing_only"},
                         "research": {"user_text": "A new search found nothing usable.", "triggered": True,
                                      "steps": {"discovered": 4, "search": "ran", "acquisition": "not_attempted",
                                                "admitted": 0, "answer": "abstained"},
                                      "diag_status_counts": {"admitted": 0, "skipped": 2,
                                                             "discovered_not_selected": 1}}}}


def _decline():
    return {"answer": "", "model": "m", "tokens_used": 0, "cost_usd": 0.0, "result_kind": "decline"}


FIXTURES = {
    "answer": (_answer, True),
    "partial_answer": (_partial, True),
    "legacy_abstract_answer": (_legacy_answer, True),
    "reading_list_with_contract": (_reading_list, True),
    "decline": (_decline, True),
    "answer_dev_mode": (_answer, False),
    "reading_list_dev_mode": (_reading_list, False),
}


class _Env:
    """Minimal monkeypatch stand-in so the capture script can reuse _run without pytest."""

    def setenv(self, k, v):
        import os
        os.environ[k] = v

    def setattr(self, obj, name, value):
        setattr(obj, name, value)


def _run(monkeypatch, body, public=True):
    import requests
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1" if public else "0")
    monkeypatch.setenv("VERA_API_KEY", "k")
    monkeypatch.setattr(sb, "_corpus", lambda st: None)
    monkeypatch.setattr(requests, "post", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: copy.deepcopy(body)))
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: {"commit": "abcdef1234"}))
    from vera import ui_wake
    monkeypatch.setattr(ui_wake, "http_get", lambda *a, **k: SimpleNamespace(status_code=200))  # no wake backoff
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    at.text_input[0].set_value("render fixture question")
    at.button[0].click()
    at.run()
    assert not at.exception
    return at


_ID_RE = re.compile(r"\$\$ID-[0-9a-f]{32}")
_HASH_RE = re.compile(r'page_script_hash: "[0-9a-f]+"')


def _norm(text):
    """Streamlit puts md5(absolute path of streamlit_app.py) into widget ids and page_script_hash: path-dependent."""
    return _HASH_RE.sub('page_script_hash: "<H>"', _ID_RE.sub("$$ID-<H>", text))


def _snapshot(at):
    """Ordered (element type, proto text) pairs for the main and sidebar trees (path-dependent hashes normalized)."""
    out = []

    def walk(node):
        proto = getattr(node, "proto", None)
        out.append([type(node).__name__, _norm(str(proto)) if proto is not None else ""])
        for child in getattr(node, "children", {}).values():
            walk(child)

    walk(at.main)
    walk(at.sidebar)
    return out


def capture_all(monkeypatch):
    """Snapshot every fixture (used once to write the golden from the original code)."""
    result = {}
    for name, (make, public) in FIXTURES.items():
        result[name] = _snapshot(_run(monkeypatch, make(), public=public))
    return result


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_rendered_tree_matches_original_golden(monkeypatch, name):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    make, public = FIXTURES[name]
    assert _snapshot(_run(monkeypatch, make(), public=public)) == golden[name]


def test_golden_covers_every_fixture():
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert sorted(golden) == sorted(FIXTURES)
    assert all(golden[name] for name in FIXTURES)
