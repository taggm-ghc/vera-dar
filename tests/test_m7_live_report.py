"""REPORT-1022: the report page's live mode, from synthetic /ask bodies only (no network, no model, no database).

Sentinels: the API base URL, its host and the API key are set in the env, echoed back inside the synthetic body, and
must never appear in the rendered page; the key must still reach the outgoing request header (server-side only)."""
import os
from types import SimpleNamespace

import pytest
import requests
from streamlit.testing.v1 import AppTest

from tests.public_terms import find_terms
from tests.test_item78_sources_sidebar import _tree_text
from vera import ui_evidence
from vera.m7 import live_report as lr

PAGE = os.path.join(os.path.dirname(__file__), "..", "pages", "vera_demo.py")
S_HOST = "sentinel-host-r1022.example.invalid"
S_URL = f"https://{S_HOST}"
S_KEY = "sk-SENTINELKEY1022abcdef"
S_MODEL = "sentinel-model-r1022"
TOKEN = "insufficient_full_document_evidence"
LONG = "Q" * 1000


def _body(kind="answer", **over):
    b = {"answer": f"Finding one [1]. Echo {S_URL}/ask key {S_KEY} host {S_HOST}.", "model": S_MODEL,
         "tokens_used": 1234, "cost_usd": 0.00042, "result_kind": kind,
         "claim_check": {"outcome": "checked", "checked": 2, "supported": 1, "partial": 1, "removed": 0},
         "contract": {"layers": ["L0"], "fallback": None if kind == "answer" else TOKEN,
                      "inline_acquisition": {"status": "ok", "admitted": 2, "stored": 1, "reused": 0,
                                             "n_answer_eligible": 2, "n_persisted": 1,
                                             "documents": [{"persisted": True},
                                                           {"persisted": False, "reason": "below_persist_min"}],
                                             "ms": {"search": 900, "evaluate": 300, "total": 1500, "secret_ms": 7}}},
         "sources": [{"n": 1, "title": "Doc <b>One</b>", "url": "https://example.org/one", "published": "2024",
                      "provider": "openalex", "acquisition": "full_document_acquired", "licence_decision": "allow",
                      "retrieved_at": "2030-01-01T00:00:00Z", "identifier": "doi:10.1/x"},
                     {"n": 2, "title": "Bad link", "url": "javascript:alert(1)", "provider": "duckduckgo_lite"},
                     {"n": 3, "title": "Host doc", "url": f"{S_URL}/doc", "provider": "zzz_unknown"}],
         "candidates": [{"n": 1, "title": "Cand", "url": "https://example.org/c", "provider": "europepmc",
                         "acquisition": "not_acquired", "licence_decision": "hold"}],
         "evidence_status": "ok", "corpus_source": {"kind": "db"},
         "examined_full_documents": [{"document_id": "d1", "title": "Doc One", "chunks_inspected": 9,
                                      "candidate_spans": 4, "admitted_spans": 2, "text_sha256": "f" * 64}],
         "admitted_evidence": [{"span_id": "s1", "document_id": "d1"}, {"span_id": "s2", "document_id": "d1"}],
         "citations": [{"claim_id": 1, "n": 1, "span_id": "s1", "document_id": "d1", "verdict": "supported",
                        "quote": LONG, "span": "chars=10-20",
                        "attribution": {"creator": "A. Author", "title": "Doc One", "source": "https://doi.org/10.1/x",
                                        "licence": "CC BY 4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
                                        "changes": "excerpted"}},
                       {"claim_id": 2, "n": 1, "span_id": "s2", "document_id": "d1", "verdict": "partial",
                        "quote": f"short quote {S_KEY}", "span": "chars=30-40",
                        "attribution": {"creator": "A. Author", "title": "Doc One", "source": "https://doi.org/10.1/x",
                                        "licence": "CC BY 4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
                                        "changes": "excerpted"}}]}
    b.update(over)
    return b


# --- projection ------------------------------------------------------------------------------------------------

def test_projection_is_allowlisted_and_quotes_are_capped():
    rep = lr.build_live_report(_body(), "Q?", 2.345)
    flat = repr(rep)
    for bad in (S_MODEL, "2030-01-01", "f" * 64, "openalex", "duckduckgo", "europepmc", "secret_ms", TOKEN):
        assert bad not in flat, bad
    cap = ui_evidence._quote_cap()
    assert all(len(c["quote"]) <= cap for c in rep["claims"]) and rep["claims"][0]["quote"] == "Q" * cap
    assert rep["search"]["by_service"] == {"scholarly index": 2, lr.OTHER: 1, "web search": 1}
    assert rep["search"]["inline"]["documents"] == {"added to VERA-DAR's collection": 1,
                                                    "used for this answer only (too few qualifying passages to keep)": 1}
    assert rep["usage"] == {"tokens": 1234, "cost_usd": 0.00042, "latency_s": 2.35}
    assert [g["n"] for g in rep["groups"]] == [1, 2, 3] and len(rep["groups"][0]["excerpts"]) == 2


def test_non_answer_uses_visitor_sentence_not_token():
    rep = lr.build_live_report(_body("reading_list"), "Q?", 1.0)
    assert rep["answer"] == "" and TOKEN not in rep["no_answer_reason"] and rep["no_answer_reason"]
    assert rep["reading_list"] and rep["reading_list"][0]["title"] == "Cand"


@pytest.mark.parametrize("junk", [None, [], "x", {"result_kind": "answer", "sources": "nope", "citations": [1, None],
                                                    "contract": [], "cost_usd": "abc", "tokens_used": True}])
def test_projection_never_raises_on_malformed(junk):
    rep = lr.build_live_report(junk, "Q?", "bad")
    assert rep["kind"] == "live" and rep["usage"]["latency_s"] == 0.0


def test_redactor_removes_url_host_and_key_with_existing_scrub():
    r = lr.redactor(S_URL, env={"VERA_API_BASE_URL": S_URL, "VERA_API_KEY": S_KEY})
    out = r(f"a {S_URL}/x b {S_HOST} c {S_KEY} d gsk_abcdefghij")
    assert S_HOST not in out and S_KEY not in out and "gsk_abcdefghij" not in out


def test_resolve_api_public_uses_server_env_dev_uses_typed_key_only():
    env = {"VERA_API_BASE_URL": S_URL, "VERA_API_KEY": S_KEY}
    assert lr.resolve_api(True, env=env) == (S_URL, S_KEY)
    assert lr.resolve_api(False, env=env, typed_key="") == (S_URL, "")
    assert lr.resolve_api(False, env=env, typed_key="typed")[1] == "typed"


def test_remember_keeps_newest_projected_reports_only():
    sess = {}
    for i in range(lr.RECENT_MAX + 2):
        lr.remember(sess, {"question": f"q{i}"})
    lr.remember(sess, {"question": "q3"})
    qs = [x["question"] for x in sess[lr.RECENT_KEY]]
    assert len(qs) == lr.RECENT_MAX and qs[0] == "q3" and qs.count("q3") == 1


# --- page (AppTest) --------------------------------------------------------------------------------------------

@pytest.fixture
def public_env(monkeypatch):
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    monkeypatch.setenv("VERA_API_BASE_URL", S_URL)
    monkeypatch.setenv("VERA_API_KEY", S_KEY)
    monkeypatch.setattr("vera.sources_sidebar._corpus", lambda st: None)
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("GET used")))
    monkeypatch.setattr("vera.db.get_session", lambda: (_ for _ in ()).throw(RuntimeError("no db in tests")))


def _run(monkeypatch, post, question="Does X affect Y?"):
    monkeypatch.setattr(requests, "post", post)
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    assert not at.exception
    at.text_input[0].input(question)
    at.button[0].click()
    return at.run()


def _ok(body, calls):
    def post(url, json=None, headers=None, timeout=None):
        calls.append((url, json, headers))
        return SimpleNamespace(status_code=200, json=lambda: body)
    return post


def _assert_clean(text):
    for bad in (S_HOST, S_KEY, S_URL, S_MODEL, TOKEN, "javascript:", "2030-01-01", "f" * 64, LONG[:300]):
        assert bad not in text, bad
    assert not find_terms(text), find_terms(text)


def test_live_report_renders_all_sections_and_leaks_nothing(monkeypatch, public_env):
    calls = []
    at = _run(monkeypatch, _ok(_body(), calls))
    assert not at.exception
    assert len(calls) == 1 and calls[0][0].startswith(S_URL) and calls[0][2]["X-API-Key"] == S_KEY  # server-side only
    text = _tree_text(at)
    _assert_clean(text)
    assert "LIVE REPORT" in text and "EXAMPLE REPORT" not in text
    heads = [h.value for h in at.header]
    assert heads == ["1. Answer", "2. Claim check", "3. Sources and short excerpts, by document", "4. Evidence scope",
                     "5. Search and acquisition", "6. Cost and latency"]
    assert "Evidence scope: 1 document inspected, 2 passages admitted as evidence (from 1 document)." in text
    assert "scholarly index: 2" in text and "licence unclear (held): 1" in text
    assert "Q" * ui_evidence._quote_cap() in text
    assert "](https://example.org/one)" in text
    assert "(javascript" not in text and "<b>" not in text
    assert any(m.label == "Latency (s)" for m in at.metric) and any(m.value == "1,234" for m in at.metric)
    for w in at.text_input:
        assert S_KEY not in str(w.value) and S_URL not in str(w.value)


def test_non_answer_report_is_clean(monkeypatch, public_env):
    at = _run(monkeypatch, _ok(_body("reading_list"), []))
    text = _tree_text(at)
    _assert_clean(text)
    assert "Reading list" in text


def test_connection_error_is_fixed_text(monkeypatch, public_env):
    def post(*a, **k):
        raise requests.exceptions.ConnectionError(f"cannot reach {S_URL} with {S_KEY}")
    at = _run(monkeypatch, post)
    text = _tree_text(at)
    _assert_clean(text)
    assert "The service is unavailable. Please try again later." in text and "ConnectionError" not in text


@pytest.mark.parametrize("code,msg", [(500, "could not produce a report"), (429, "Rate limit reached")])
def test_http_errors_are_fixed_text(monkeypatch, public_env, code, msg):
    at = _run(monkeypatch, lambda *a, **k: SimpleNamespace(status_code=code, json=lambda: {"detail": S_URL}))
    text = _tree_text(at)
    _assert_clean(text)
    assert msg in text and "HTTP 500" not in text


def test_dev_mode_never_uses_server_key(monkeypatch, public_env):
    monkeypatch.setenv("VERA_PUBLIC_MODE", "0")
    calls = []
    at = _run(monkeypatch, _ok(_body(), calls))
    assert calls == [] and any("No API key set" in w.value for w in at.warning)
    assert all(S_KEY not in str(w.value) for w in at.text_input)
    _assert_clean(_tree_text(at))


def test_offline_mode_is_labelled_fixture(monkeypatch, public_env):
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("POST used")))
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    at.radio[0].set_value(at.radio[0].options[1])
    at = at.run()
    assert not at.exception
    text = _tree_text(at)
    assert "OFFLINE FALLBACK" in text and "EXAMPLE REPORT" in text and S_HOST not in text and S_KEY not in text


# --- INT-1022-B: the licence label is each source's own licence --------------------------------------------------

def test_each_source_shows_its_own_licence_not_a_fixed_cc_by():
    b = _body()
    b["sources"][1]["licence"] = "cc by-nc-sa"
    b["sources"][2]["licence"] = "CC0 1.0"
    b["citations"][0]["attribution"]["licence"] = "CC BY-NC 4.0"
    b["citations"][1]["attribution"]["licence"] = "CC BY-NC 4.0"
    rep = lr.build_live_report(b, "Q?", 1.0)
    assert [g["licence"] for g in rep["groups"]] == ["", "cc by-nc-sa", "CC0 1.0"]
    assert rep["groups"][0]["attribution"]["licence"] == "CC BY-NC 4.0"
    texts = []

    class _St:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __getattr__(self, name):
            def f(*a, **k):
                texts.extend(str(x) for x in a)
                if name == "columns":
                    return [self] * (a[0] if isinstance(a[0], int) else len(a[0]))
                return self if name in ("expander", "container") else None
            return f
    lr.render_live_report(_St(), rep)
    joined = "\n".join(texts).replace("\\", "")  # drop markdown escapes
    assert "CC BY-NC 4.0" in joined and "cc by-nc-sa" in joined and "CC0 1.0" in joined
    assert "CC BY 4.0" not in joined


# --- B2-1022: out-of-scope override answer; one entry per document; stage vocabulary ----------------------------

def _b2_body(**over):
    rows = [{"n": i, "title": "Same Paper", "url": "https://example.org/p", "identifier": "doi:10.1/p",
             "span": f"chars={i}0-{i}9", "provider": "full_document"} for i in (1, 2, 3)]
    rows.append({"n": 4, "title": "Other Paper", "url": "https://example.org/q", "identifier": "doi:10.1/q",
                 "span": "chars=1-9", "provider": "full_document"})
    b = _body(sources=rows, candidates=[], contract={"layers": ["L0"], "fallback": None, "offtopic": True,
                                                     "acquired_only": True},
              claim_check={"outcome": "checked", "checked": 1, "supported": 1, "partial": 0, "removed": 0},
              admitted_evidence=[{"span_id": f"s{i}", "document_id": "d1" if i < 4 else "d2"} for i in range(1, 5)],
              citations=[{"claim_id": 1, "n": 1, "span_id": "s1", "document_id": "d1", "verdict": "supported",
                          "quote": "q", "span": "chars=10-19"}])
    b.update(over)
    return b


def test_b2_same_publication_is_one_entry_and_distinct_passages_are_kept():
    rep = lr.build_live_report(_b2_body(), "what did humpty dumpty do?", 1.0)
    assert [g["ns"] for g in rep["groups"]] == [[1, 2, 3], [4]]
    assert rep["groups"][0]["passages"] == ["chars=10-19", "chars=20-29", "chars=30-39"]
    assert rep["search"]["results"] == 2  # documents, not passages


def test_b2_scope_policy_and_stage_lines():
    rep = lr.build_live_report(_b2_body(), "q", 1.0)
    assert rep["scope_policy"] == lr.SCOPE_OVERRIDE_NOTE
    assert "Evidence authorized for answer generation: yes." in rep["pipeline"]
    assert "Answer generated: yes." in rep["pipeline"]
    assert "Passages admitted as evidence: 4, from 2 documents." in rep["pipeline"]
    assert "Claims checked: 1; supported by their source passage: 1." in rep["pipeline"]
    rl = lr.build_live_report(_body("reading_list", contract={"layers": ["L0"], "fallback": TOKEN, "offtopic": True}),
                              "q", 1.0)
    assert rl["scope_policy"] == lr.SCOPE_DECLINED_NOTE
    assert "Evidence authorized for answer generation: no." in rl["pipeline"] and "Answer generated: no." in rl["pipeline"]
    assert lr.build_live_report(_body(), "q", 1.0)["scope_policy"] == ""  # in-scope: no policy note


def test_b2_render_shows_grouped_entry_and_policy(monkeypatch, public_env):
    text = _tree_text(_run(monkeypatch, _ok(_b2_body(), [])))
    assert "[1, 2, 3]" in text and "Coverage policy: VERA-DAR's collection could not answer" in text  # OD-1028(6) wording
    assert "Evidence authorized for answer generation: yes." in text


def test_b2_in_scope_admitted_but_no_answer_is_still_authorized():
    body = _body("reading_list", claim_check={"outcome": "checked", "checked": 2, "supported": 0, "partial": 0,
                                              "removed": 2})
    rep = lr.build_live_report(body, "q", 1.0)
    assert "Evidence authorized for answer generation: yes." in rep["pipeline"]
    assert "Answer generated: no." in rep["pipeline"] and rep["scope_policy"] == ""


def test_p05b_licence_labels_keep_states_distinct():
    from vera.m7.live_report import licence_label
    meta = {"acquisition": "not_acquired", "licence_decision": "allow", "metadata_licence": "CC0-1.0"}
    assert licence_label(meta) == "Index metadata: CC0; article licence not verified"
    assert licence_label({**meta, "metadata_licence": "CC-BY-4.0"}) == "Index metadata: CC-BY-4.0; article licence not verified"
    nometa = {"acquisition": "not_acquired", "licence_decision": "allow"}
    assert licence_label(nometa) == "article licence not verified (link only)"
    assert "allows use" not in licence_label(meta) and "allows use" not in licence_label(nometa)
    assert licence_label({"acquisition": "full_document_acquired", "licence_decision": "allow",
                          "metadata_licence": "CC0-1.0"}) == "article licence allows use"
    assert licence_label({"acquisition": "not_acquired", "licence_decision": "hold"}) == "licence unclear (held)"


def test_p05b_report_tally_and_candidate_field():
    from vera.m7.live_report import build_live_report
    from vera.grounded_ask import public_candidates
    c = public_candidates([{"n": 1, "title": "t", "url": "https://example.org/a", "provider": "openalex",
                            "licence_decision": "allow",
                            "licence": {"metadata": {"id": "CC0-1.0"}, "content": None}}])[0]
    assert c["metadata_licence"] == "CC0-1.0" and c["acquisition"] == "not_acquired"
    rep = build_live_report(_body("reading_list", candidates=[c], sources=[]), "q", 1.0)
    assert rep["search"]["by_licence"] == {"Index metadata: CC0; article licence not verified": 1}
    assert rep["search"]["by_acquisition"] == {"not acquired (link only)": 1}
