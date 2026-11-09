"""FAIRUSE-TIER-1108: no full-text redistribution. Greps the API routes, the response models and the UI for paths that
could emit stored source text, so that only capped quotes leave the system. Audit record (2026-11-08): every route below
was read; only /ask and /ask/stream carry source text, and only as citations[].quote (<= answer.quote_max_chars, budgeted
per response, per source for fair-use and ND sources) and contract.trace.passages[].excerpt (<= 200 each, < 1200 total;
a fair-use passage shows only its capped citation quote). A NEW route, response field or full-text reader fails this
test until it is audited and added here."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (method, path) -> what it can emit; every entry was audited in FAIRUSE-TIER-1108 (see the module docstring)
AUDITED_ROUTES = {
    ("GET", "/corpus-summary"): "model-written corpus digest; no titles, URLs or source text",
    ("GET", "/memory/pending"): "owner only: pending memory findings (model-written claims), not source text",
    ("POST", "/memory/{claim_id}/confirm"): "owner only: status change",
    ("POST", "/memory/{claim_id}/reject"): "owner only: status change",
    ("GET", "/health"): "status only",
    ("GET", "/health/dar"): "status and counts only",
    ("GET", "/trace/public"): "aggregates only",
    ("GET", "/trace/live"): "aggregates only",
    ("GET", "/trace/live/owner"): "owner only: aggregates",
    ("GET", "/trace/private"): "viz key: eval results file (question, 25-word answer excerpt, titles/URLs); no source text",
    ("GET", "/version"): "build id only",
    ("GET", "/trace/request/{trace_id}"): "owner only: request steps, strings cut to 200 chars; no passages by design",
    ("POST", "/ask"): "answer + capped quotes (citations[].quote) + capped trace excerpts",
    ("GET", "/diag/header-shape"): "owner only: header shapes",
    ("POST", "/ask/stream"): "same response object as /ask",
    ("GET", "/ask/progress/{rid}"): "stage events and counts only",
}
# response-model field names that would carry raw source text
_TEXT_FIELDS = {"text", "snippet", "abstract", "full_text", "fulltext", "body", "chunk", "chunk_text", "passage",
                "passage_text", "content", "document_text", "span_text", "excerpt"}
# modules allowed to read a stored document's full text (internal: admission, claim-check reference, corpus loaders)
_FULLTEXT_READERS = {"vera/ask_service.py", "vera/fulldoc_admission.py", "vera/fulldoc_answer.py",
                     "vera/fulldoc_corpus.py", "vera/fulldoc_db.py",
                     # FAIRUSE-A-REBASE-1109: internal readers added by batch 3/4 (not egress): acronym definitions are matched in
                     # the text in memory; the hybrid stored-vector guard hashes a slice to compare with the stored sha.
                     "vera/acronym_match.py", "vera/hybrid_retrieval.py"}
_FULLTEXT_READ = re.compile(r"\.text\((?:did|document_id|d\[|a\[|str\(|x\b|doc)|\b(?:corpus|c)\.text\(|\b_texts(?:\[|\.get\()")


def _routes(path: Path) -> set:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.decorator_list:
                if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and isinstance(d.func.value, ast.Name):
                    meth = d.func.attr.upper()
                    if meth in ("GET", "POST", "PUT", "DELETE", "PATCH", "API_ROUTE", "ROUTE", "WEBSOCKET"):
                        path = d.args[0] if d.args else next((k.value for k in d.keywords if k.arg == "path"), None)
                        out.add((meth, path.value if isinstance(path, ast.Constant) else f"<dynamic:{ast.dump(d)}>"))
    return out


def test_every_api_route_is_audited():
    assert _routes(ROOT / "main.py") == set(AUDITED_ROUTES)


def test_no_other_module_registers_routes_or_mounts():
    pat = re.compile(r"@(?:app|router)\.(?:get|post|put|delete|patch|api_route|websocket)\(|APIRouter\(|add_api_route\("
                     r"|include_router\(|\bapp\.mount\(")  # requests Session.mount("https://", adapter) is not a route
    hits = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "vera").rglob("*.py")) if pat.search(p.read_text("utf-8"))]
    assert hits == []
    main = (ROOT / "main.py").read_text(encoding="utf-8")
    for call in ("include_router(", ".mount(", "add_api_route(", "add_route(", "add_websocket_route(", "Router("):
        assert call not in main, call


def test_response_models_have_no_source_text_fields():
    import main

    seen, bad = set(), []

    def walk(model, where):
        if model in seen:
            return
        seen.add(model)
        for name, f in model.model_fields.items():
            if name in _TEXT_FIELDS:
                bad.append(f"{where}.{name}")
            for t in _types(f.annotation):
                if isinstance(t, type) and hasattr(t, "model_fields"):
                    walk(t, f"{where}.{name}")

    for m in (main.AskResponse, main.CorpusDescription, main.PendingList, main.MemoryDecision):
        walk(m, m.__name__)
    assert bad == []
    q = main.EvidenceCitation.model_fields
    assert "quote" in q and "attribution" in q  # the only quoted text field, always with attribution


def _types(ann):
    import typing
    out = [ann]
    for a in typing.get_args(ann):
        out += _types(a)
    return out


def test_full_text_is_read_only_by_internal_modules():
    hits = set()
    for p in [ROOT / "main.py", ROOT / "streamlit_app.py", *sorted((ROOT / "vera").rglob("*.py")),
              *sorted((ROOT / "pages").glob("*.py"))]:
        if p.exists() and _FULLTEXT_READ.search(p.read_text(encoding="utf-8").replace("st.text(", "st_text(")):
            hits.add(str(p.relative_to(ROOT)))
    assert hits <= _FULLTEXT_READERS, hits - _FULLTEXT_READERS


def test_public_projection_drops_snippets():
    from vera import fulldoc_answer, grounded_ask
    assert "snippet" not in grounded_ask.PUBLIC_CANDIDATE_FIELDS
    src = (ROOT / "vera" / "fulldoc_answer.py").read_text(encoding="utf-8")
    assert '"snippet` is internal' in src or "`snippet` is internal" in src
    assert callable(fulldoc_answer.public_evidence)


def test_quote_caps_agree_across_config_trace_and_ui():
    from vera import evidence_trace, fair_use
    a = json.loads((ROOT / "config" / "fulldoc_answer.json").read_text(encoding="utf-8"))["answer"]
    assert a["quote_max_chars"] == evidence_trace.QUOTE_CAP == 200
    assert a["max_total_quote_chars"] == evidence_trace.QUOTE_TOTAL_CAP == 1200
    assert fair_use.caps()["quote_max_chars"] <= a["quote_max_chars"]


def test_build_citations_never_exceeds_the_quote_cap():
    from vera.fulldoc_answer import build_citations
    long = "Alpha beta gamma delta. " * 60
    srcs = [{"n": 1, "snippet": long, "start": 0, "span_id": "d:0-1", "document_id": "d", "canonical_source_id": "c",
             "licence": "cc by 4.0", "span": "chars=0-1", "attribution": {"creator": "a", "title": "t", "source": "https://x",
                                                                          "licence": "cc by", "licence_url": "u"}}]
    recs = [{"id": 1, "verdict": "supported", "citations": [1], "sentence": "Alpha beta gamma delta."}]
    for c in build_citations(recs, srcs, 200, 1200):
        assert c["quote"] is None or (len(c["quote"]) <= 200 and c["quote"] in long)
