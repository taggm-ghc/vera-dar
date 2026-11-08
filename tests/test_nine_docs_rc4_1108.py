"""NINE-DOCS-RC4-1108 (owner, recurring defect RC-4): "9 stored full documents shared words with this question" for
"from where did vera come?". Root cause: user-facing counts were taken from candidate_spans, the RAW retrieval
candidates; retrieval (BM25 over the domain-routing tokenizer) matches on any non-stopword token such as "did" or
"come", so every document of the pinned 9-document base set "matched". Counts now use relevant_spans (passages that
cleared the question's admission relevance rules), and reading-list results that share no content stem with the
question are dropped. Offline: shipped configs, fake corpus, no network, no LLM, no DB."""
import json
import re
from pathlib import Path

from vera import fulldoc_corpus, grounded_ask
from vera import ui_evidence as u
from vera.domain_identify import tokenize as route_tokenize
from vera.fulldoc_admission import admit

ROOT = Path(__file__).resolve().parents[1]
TOK = "insufficient_full_document_evidence"
CFG = fulldoc_corpus.load_config()
ROUTE_ICFG = json.loads((ROOT / "config" / "domain_routing.json").read_text())["identification"]

# Nine stored documents in three subjects; every one also uses "did" and "come" as ordinary prose.
FILLER = " Researchers did come back to this point later, and what they did come to see was more data. "
SUBJECTS = {
    "epi": "Epidemic forecasting models predict influenza incidence; ensemble epidemic forecasting improves accuracy "
           "of influenza incidence forecasts across seasons.",
    "soil": "Soil carbon sequestration under cover crops increases soil organic carbon; sequestration rates of soil "
            "carbon depend on tillage.",
    "code": "Reproducible research software needs version control; software testing and version control make "
            "research software reproducible.",
}
DOCS = {f"{s}{i}": FILLER + SUBJECTS[s] + FILLER for s in SUBJECTS for i in range(1, 4 if s != "code" else 4)}


def _doc(i):
    return {"document_id": i, "canonical_source_id": f"https://x.example/{i}", "title": f"Paper {i}",
            "url": f"https://x.example/{i}", "doi": None, "published": "2020", "authors": ["Author A"],
            "licence": "CC BY 4.0", "licence_id": "CC BY 4.0",
            "licence_url": "https://creativecommons.org/licenses/by/4.0/", "licence_decision": "allow",
            "text_sha256": "h" + i, "chars": len(DOCS[i]), "evidence_ranges": [[0, 10 ** 9]]}


class _Corpus:
    """Retrieval like DarCorpus: any shared routing token (stopwords dropped, but not "did"/"come") is a candidate."""
    status, source_kind, unusable = "ok", "db", []

    def __init__(self):
        self.acquired = [_doc(i) for i in DOCS]

    def text(self, i):
        return DOCS[i]

    def retrieve(self, q, k, evidence_only=False):
        qt = set(route_tokenize(q, ROUTE_ICFG))
        out = []
        for i, t in DOCS.items():
            for m in re.finditer(r"[^.;]+[.;]", t):
                if qt & set(route_tokenize(m.group(0), ROUTE_ICFG)):
                    out.append({"span_id": f"{i}:{m.start()}", "document_id": i, "start": m.start(), "end": m.end(),
                                "text": m.group(0), "score": 1.0})
        return out

    def inspection(self):
        return {i: {"chunks_inspected": 3} for i in DOCS}


def _response(question):
    ev = admit(question, _Corpus(), CFG)
    rows = [dict(d) for d in ev.examined_full_documents]
    sc = u.search_counts(rows, None)
    data = {"evidence_status": ev.status, "examined_full_documents": rows, "corpus_source": {"kind": "db"},
            "admitted_evidence": [], "contract": {"fallback": TOK, **({"search_counts": sc} if sc else {})}}
    head = u.fallback_text(TOK, CFG["answer"], n_inspected=len(rows), n_matched=u.matched_for(data))
    return ev, rows, head, u.evidence_scope_line(data)


def test_function_word_question_has_raw_candidates_everywhere_but_no_relevant_document():
    ev, rows, head, scope = _response("from where did vera come?")
    assert len(rows) == 9 and all(r["candidate_spans"] > 0 for r in rows)  # the old count: 9, always
    assert all(r["relevant_spans"] == 0 for r in rows)
    assert u.matched_for({"examined_full_documents": rows}) == 0
    assert head.startswith("No stored full document was relevant to this question")
    assert "9" not in head and "shared words" not in head
    assert "no stored document relevant to the question" in scope and "sharing words" not in scope


def test_stopword_only_question_makes_no_shared_words_claim():
    _ev, rows, head, scope = _response("what did they do there?")
    assert u.matched_for({"examined_full_documents": rows}) == 0
    for line in (head, scope):
        assert not re.search(r"\d+ stored (full )?documents? (shared|sharing) words", line)
        assert "shared words" not in line and "sharing words" not in line


def test_two_domain_questions_give_different_relevance_based_counts():
    _e1, r1, h1, s1 = _response("How accurate is ensemble epidemic forecasting of influenza incidence?")
    _e2, r2, h2, s2 = _response("What drives soil carbon sequestration rates?")
    n1 = u.matched_for({"examined_full_documents": r1})
    n2 = u.matched_for({"examined_full_documents": r2})
    assert {r["document_id"] for r in r1 if r["relevant_spans"]} <= {"epi1", "epi2", "epi3"}
    assert {r["document_id"] for r in r2 if r["relevant_spans"]} <= {"soil1", "soil2", "soil3"}
    assert 0 < n1 < 9 and 0 < n2 < 9
    # different subjects -> different document sets, never the base-set size
    assert {r["document_id"] for r in r1 if r["relevant_spans"]} != {r["document_id"] for r in r2 if r["relevant_spans"]}
    assert "9 stored" not in s1 + s2


def test_legacy_rows_without_relevant_spans_show_no_count():
    rows = [{"document_id": f"d{i}", "candidate_spans": 2, "admitted_spans": 0} for i in range(9)]
    assert u.matched_documents(rows) is None and u.search_counts(rows) is None
    data = {"evidence_status": "insufficient", "examined_full_documents": rows, "admitted_evidence": [],
            "contract": {"fallback": TOK}}
    head = u.fallback_text(TOK, CFG["answer"], n_inspected=9, n_matched=u.matched_for(data))
    assert not re.search(r"\b9\b", head) and not re.search(r"\b9\b", u.evidence_scope_line(data))


def test_reading_list_drops_function_word_only_titles():
    on = grounded_ask._content_matcher("from where did vera come?")
    assert on is not None
    assert not on("Come, Go") and not on("Where did they come from? Origins")
    assert on("VERA: a verified evidence research assistant")
    assert grounded_ask._content_matcher("what did they do?") is None  # nothing to judge on: list unchanged


def test_retrieve_all_never_lists_come_go():
    cfg = {"num_results": 5, "snippet_max_chars": 300, "resolve": {"enabled": False}}
    res = [{"title": "Come, Go", "url": "https://b.example/1", "snippet": "A study of motion verbs and going."},
           {"title": "The VERA project", "url": "https://b.example/2", "snippet": "VERA origins and design."}]
    out, _ = grounded_ask.retrieve_all("from where did vera come?", cfg, search=lambda q: list(res))
    assert [s["title"] for s in out] == ["The VERA project"] and out[0]["n"] == 1


# ---- the recurrence control (rca.md): every user-facing count string is relevance-based ----
# a COUNT of documents by shared words ("N stored documents sharing words", "{n_stored_phrase} shared words",
# f"...{n} document{'s'} sharing words"); writer prompts saying "not merely shared words" are not counts.
_BANNED = re.compile(r"(?:documents?|phrase\}|'\)\}|\}) shar(?:ed|ing|e)s? words", re.I)


def _user_facing_files():
    for p in sorted((ROOT / "vera").rglob("*.py")) + sorted((ROOT / "pages").rglob("*.py")) + [ROOT / "streamlit_app.py",
                                                                                              ROOT / "main.py"]:
        yield p
    for p in sorted((ROOT / "config").glob("*.json")):
        yield p


def test_no_user_facing_text_counts_documents_by_shared_words():
    hits = []
    for p in _user_facing_files():
        if not p.is_file():
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if _BANNED.search(line) and not line.lstrip().startswith("#") and '"_' not in line:
                hits.append(f"{p.relative_to(ROOT)}:{n}")
    assert not hits, f"user-facing 'shared words' document counts: {hits}"


def test_count_sites_read_relevant_spans_not_raw_candidates():
    """Every display-side count of 'documents for this question' keys on relevant_spans. A new display site that
    filters on candidate_spans fails here (the RC-4 recurrence was a second site the first fix did not cover)."""
    allowed = {"vera/fulldoc_admission.py", "vera/fulldoc_answer.py", "vera/fulldoc_evidence.py",
               "vera/ask_service.py"}  # producer, response field list, docstring; ask_service: freshness dates only
    bad = []
    for p in sorted((ROOT / "vera").rglob("*.py")):
        rel = str(p.relative_to(ROOT))
        if rel in allowed:
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"""\.get\(["']candidate_spans["']\)\s*or\s*0\)\s*>\s*0|\["candidate_spans"\]\s*>\s*0""", line):
                bad.append(f"{rel}:{n}")
    assert not bad, f"count sites on raw retrieval candidates: {bad}"
    src = (ROOT / "vera" / "ui_evidence.py").read_text()
    assert 'RELEVANT_FIELD = "relevant_spans"' in src


def test_relevant_spans_survives_the_api_response_model():
    """Verifier finding: the /ask response model dropped relevant_spans, so the Report and Trace Eval (which read the
    response rows) could never show a relevance-based count."""
    import main
    row = main.ExaminedDocument(document_id="d1", candidate_spans=3, relevant_spans=2).model_dump()
    assert row["relevant_spans"] == 2
