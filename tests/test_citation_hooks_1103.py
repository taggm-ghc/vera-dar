"""NONASK-1103 C2/C4/C5/C5a/C6 and R-037 storage: citation hooks (offline, fictitious works)."""
import copy
import json
from pathlib import Path

import pytest

from vera import apa7, discovery_panel, evidence_trace, grounded_ask, memory_store, references_ui, ui_evidence
from vera.m7 import live_report

ROOT = Path(__file__).resolve().parent.parent
SNIPPET = "SENTINEL-PAGE-TEXT unclear licence body"
SRC_A = {"n": 1, "title": "Alpha methods", "url": "https://example.org/a", "published": "2020-05-01", "snippet": SNIPPET,
         "source_type": "journal-article", "provider": "openalex", "identifier": "doi:10.1234/alpha",
         "doi": "10.1234/alpha", "authors": ["Ada Alpha", "Bo Beta"], "publisher": "Journal of Tests",
         "retrieved_at": "2026-10-15T00:00:00", "licence_decision": "allow"}
SRC_B = {"n": 2, "title": "Zeta notes", "url": "https://example.net/z", "published": "2018", "snippet": SNIPPET,
         "provider": "duckduckgo", "identifier": "https://example.net/z", "authors": "Zeta Zed, Zoe Zed",
         "licence_decision": "hold"}


@pytest.fixture(autouse=True)
def _offline_style(monkeypatch):
    monkeypatch.setattr(references_ui, "legacy", lambda: False)
    monkeypatch.setattr(apa7, "citation_style", lambda *a, **k: "apa7")


def test_public_sources_gain_only_reference_and_in_text():
    out = grounded_ask.public_sources([SRC_A, SRC_B])
    assert set(out[0]) == set(grounded_ask.PUBLIC_FIELDS)
    assert {"reference", "in_text"} <= set(grounded_ask.PUBLIC_FIELDS)
    for hidden in ("authors", "accessed", "snippet", "publisher", "doi"):
        assert hidden not in out[0]
    assert "Alpha" in out[0]["reference"] and "10.1234/alpha" in out[0]["reference"]
    assert out[0]["in_text"].startswith("(Alpha") or "Ada Alpha" in out[0]["in_text"]
    assert "Zeta Zed, Zoe Zed" in out[1]["reference"]  # joined string = ONE literal author, never split
    assert SNIPPET not in json.dumps(out)
    assert set(out[0]) - {"n", "title", "url", "published", "source_type", "provider", "identifier", "retrieved_at",
                          "licence_decision"} == {"reference", "in_text", "characterisation"}


def test_characterisation_public_verified_only_and_escaped():
    out = grounded_ask.public_sources([SRC_A, SRC_B, dict(SRC_B, n=3, title="W", url="https://news.example.org/x", authors=None)])
    assert out[0]["characterisation"] == "journal article · Journal of Tests · 2020"
    assert out[1]["characterisation"].startswith("Source") and "example.net" not in out[1]["characterisation"]
    assert "news.example.org" not in out[2]["characterisation"]  # never guessed from the domain
    esc = grounded_ask.public_sources([dict(SRC_A, publisher="<b>X*</b>")])[0]["characterisation"]
    assert "<b>" not in esc
    assert "characterisation" in grounded_ask.PUBLIC_CANDIDATE_FIELDS


def test_examined_documents_public_projection_has_no_authors():
    from vera import fulldoc_answer
    ev = type("E", (), {"status": "ok", "examined_full_documents": [{"document_id": 1, "title": "T", "authors": "A B"}],
                        "admitted": [], "corpus_source": None})()
    pub = fulldoc_answer.public_evidence(ev)
    assert "authors" not in fulldoc_answer._DOC_FIELDS and "A B" not in json.dumps(pub)


def test_public_candidates_carry_reference_in_text():
    out = grounded_ask.public_candidates([SRC_A])
    assert out[0]["reference"] and out[0]["in_text"] and SNIPPET not in json.dumps(out)


def test_citation_style_legacy_1103(monkeypatch):
    monkeypatch.setattr(references_ui, "legacy", lambda: True)
    out = grounded_ask.public_sources([SRC_A])[0]
    assert out["reference"] is None and out["in_text"] is None
    assert references_ui.display_answer("x [1]", [SRC_A]) == "x [1]"


def test_cited_set_unchanged_1103():
    """KC-3: the cited set comes from [n]; substitution writes a local display string only."""
    data = {"answer": "Claim one [1]. Claim two [2][1].", "sources": grounded_ask.public_sources([SRC_A, SRC_B])}
    before = copy.deepcopy(data)
    view = references_ui.answer_view(data)
    assert data == before and data["answer"] == "Claim one [1]. Claim two [2][1]."
    assert view["cited"] == {1, 2}
    assert "[1]" in view["display"] or "(Alpha" in view["display"]
    assert view["display"] != data["answer"] and "(Ada Alpha" in view["display"] and "Zeta Zed, Zoe Zed" in view["display"]
    from vera.sources_sidebar import cited_numbers
    assert view["cited"] == cited_numbers(data["answer"])
    assert len(view["references"]) == 2 and all(isinstance(r, str) for r in view["references"])
    assert references_ui.answer_view({"answer": "None cited.", "sources": data["sources"]})["references"] == []


class _St:
    def __init__(self):
        self.out = []
        self.session_state = {}

    sidebar = property(lambda self: self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def f(*a, **k):
            self.out.extend(str(x) for x in a if isinstance(x, str))
            return self if name in ("expander", "columns", "container", "status") else None
        return f


def test_reference_block_rendered_in_answer_area_with_house_caption():
    st = _St()
    references_ui.render_references(st, references_ui.reference_block(
        grounded_ask.public_sources([SRC_A, SRC_B])))
    text = "\n".join(st.out)
    assert "**References**" in text and apa7.HOUSE_CAPTION in text and "Alpha" in text
    st2 = _St()
    references_ui.render_references(st2, [])
    assert st2.out == []


def test_sidebar_no_per_source_1103():
    """S1: the sidebar renderer takes no body and shows no per-source title, author, reference or identifier."""
    import inspect

    from vera import sources_sidebar as sb
    assert list(inspect.signature(sb.render_sources_sidebar).parameters) == ["st"]
    body = inspect.getsource(sb.render_sources_sidebar)
    for banned in ("reference", "in_text", "authors", "references_ui", "apa7", '["sources"]', "get(\"sources\")"):
        assert banned not in body
    st = _St()
    sb.render_sources_sidebar(st)
    text = "\n".join(st.out)
    for s in (SRC_A, SRC_B):
        assert s["title"] not in text
    assert "References" not in text


def test_discovery_panel_reading_list_apa_and_double_escape_fix():
    st = _St()
    data = {"result_kind": "reading_list", "candidates": list(grounded_ask.public_candidates([SRC_A])),
            "answer": "x", "contract": {}}
    discovery_panel.render_discovery(st, data)
    text = "\n".join(st.out)
    assert "Alpha" in text and "10.1234/alpha" in text and "Source" in text or "2020" in text
    st = _St()
    discovery_panel._render_memory(st, {"recalled_discovery": [{"note": "a <b> & c", "urls": []}]})
    t = "\n".join(st.out)
    assert "&lt;b&gt;" in t and "\\&lt;" not in t and "&amp;lt;" not in t


def test_quotes_get_in_text_citation():
    st = _St()
    data = {"sources": grounded_ask.public_sources([SRC_A]),
            "citations": [{"n": 1, "verdict": "supported", "quote": "short quote", "span": "chars=1-2"}]}
    ui_evidence.render_citations(st, data)
    line = next(x for x in st.out if x.startswith("[1]"))
    assert line.startswith("[1] (") and "Alpha" in line and "supported" in line and "short quote" in line


def test_live_report_sources_and_reading_list_as_apa():
    body = {"result_kind": "answer", "answer": "A [1].", "sources": grounded_ask.public_sources([SRC_A]),
            "candidates": list(grounded_ask.public_candidates([SRC_B])), "citations": []}
    rep = live_report.build_live_report(body, "q", 1.0)
    assert "Alpha" in rep["groups"][0]["reference"] and rep["groups"][0]["in_text"]
    assert SNIPPET not in json.dumps(rep)
    st = _St()
    live_report.render_live_report(st, rep, r=lambda s: s, public=True)
    assert any(x.startswith("**[1]**") and "Alpha" in x for x in st.out)


def test_evidence_trace_row_carries_reference_not_page_text(monkeypatch):
    monkeypatch.setattr(evidence_trace, "_is_public_mode", lambda: False)
    Ev = {"admitted": [{"n": 1, "span_id": "s1", "document_id": "d", "text": SNIPPET}]}
    cites = [{"n": 1, "claim_id": 1, "quote": SNIPPET}]
    recs = [{"id": 1, "sentence": "Claim.", "citations": (1,), "verdict": "supported"}]
    tr = evidence_trace.project(Ev, recs, cites, None, None, True, sources=[SRC_A])
    assert tr is not None, "evidence_trace.project returned None: config/evidence_trace.json missing, invalid or disabled"
    assert "reference" not in tr["passages"][0]  # public rows stay URL-free (E allowlist)
    dev = evidence_trace.project(Ev, recs, cites, None, None, False, sources=[SRC_A])
    assert dev is not None and "Alpha" in dev["passages"][0]["reference"]
    lane = dict(SRC_A, answer_only=True)
    tr2 = evidence_trace.project(Ev, recs, cites, None, None, True, sources=[lane])
    assert tr2["passages"][0]["excerpt"] is None and SNIPPET not in json.dumps(tr2)


# ------------------------------------------------------------------ R-037 storage (plan section 8)

LANE = dict(SRC_A, answer_only=True, lane="official")


def test_licence_b_no_storage_1103(monkeypatch):
    """An R-037 (answer-only) source leaves no memory_store row and no page text in any projected/stored view.
    Sinks covered here: memory claims, memory discovery notes, evidence trace, live-report session state, public
    sources/candidates, claim records. Corpus, fetch/robots caches and logs belong to stream S (listed, not covered)."""
    assert memory_store.answer_only(LANE) and not memory_store.answer_only(SRC_A)
    written = []

    class Store:
        def __getattr__(self, n):
            def f(*a, **k):
                written.append((n, a, k))
                return 1
            return f

    ms = memory_store.MemoryService.__new__(memory_store.MemoryService)
    ms.reader, ms.store, ms.cfg = object(), Store(), {"max_writes_per_answer": 5, "fact_type": "f"}
    ms._lock, ms.clock = __import__("threading").Lock(), __import__("time").time
    ms._reserve = lambda: True
    ms._release = lambda: None
    rec = {"sentence": "Claim.", "verdict": "supported", "citations": [1]}
    assert ms._write_one(rec, [LANE], "q") == memory_store.SKIP_ANSWER_ONLY
    n, skipped = ms.write_claims([rec], [LANE], "q")
    assert n == 0 and skipped == {memory_store.SKIP_ANSWER_ONLY: 1} and written == []
    # projections: reference allowed; page text never
    pub = grounded_ask.public_sources([LANE]) + list(grounded_ask.public_candidates([LANE]))
    assert SNIPPET not in json.dumps(pub)
    rep = live_report.build_live_report({"result_kind": "answer", "answer": "A [1].", "sources": pub[:1],
                                         "candidates": pub[1:]}, "q", 1.0)
    session = {}
    live_report.remember(session, rep)
    assert SNIPPET not in json.dumps(session)
