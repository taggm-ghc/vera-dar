"""Item #84 live drill (R1-approved, one-off). Never prints URLs, hosts or keys.

Ops class: ops-write (mode `ask` writes memory claims through the write-only account; `recall`/`recall-related`/`count`
read). Owner-gated. Installs connect_guard (app context) and refuses under a test runner.

Usage (env: OPENAI_API_KEY, GROQ_API_KEY, VERA_API_KEY, VERA_DB_URL_RO, VERA_DB_URL_WO already in the process):
  memory_recall_drill.py ask QID      # FastAPI TestClient; background tasks run before the call returns
  memory_recall_drill.py recall QID   # fresh process: ask a (related) question and print `recalled`
  memory_recall_drill.py recall-related QID "<related question>"  # ro reader recall ONLY: no /ask, no writes
  memory_recall_drill.py count MINUTES [QID...]  # ro: live memory claims created in the last MINUTES; placeholder check
Questions come from config/trace_questions_v1.json by id.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def question(qid: str) -> str:
    items = json.loads((ROOT / "config" / "trace_questions_v1.json").read_text())["items"]
    return next(i["text"] for i in items if i["id"] == qid)


def client():
    from fastapi.testclient import TestClient
    import main
    os.environ.pop("VERA_DB_URL_RW", None)  # never used by this drill (main's load_dotenv may have loaded it)
    return TestClient(main.app)


def split_ids(rows) -> dict:
    """V6: rows of (claim_id, n_evidence_spans) -> findings (more than 0 spans) and discovery notes (0 spans)."""
    return {"new_finding_ids": [r[0] for r in rows if int(r[1] or 0) > 0],
            "new_note_ids": [r[0] for r in rows if int(r[1] or 0) == 0]}


def ask(qid: str) -> dict:
    r = client().post("/ask", json={"question": question(qid)}, headers={"X-API-Key": os.environ["VERA_API_KEY"]})
    body = r.json() if r.status_code == 200 else {"detail": str(r.json().get("detail", ""))[:120]}
    return {"status": r.status_code, "recalled": body.get("recalled"), "memory_written": body.get("memory_written"),
            "recalled_discovery": (lambda v: len(v) if isinstance(v, list) else v)(body.get("recalled_discovery")),
            "discovery_note_submitted": body.get("discovery_note_submitted"),
            "n_sources": len(body.get("sources") or []), "claim_check": body.get("claim_check"),
            "detail": body.get("detail")}


def count(minutes: int, qids: tuple = ()) -> dict:
    from sqlalchemy import create_engine, text
    cfg = json.loads((ROOT / "config" / "ask-provider-chain.json").read_text())["memory"]["store"]
    eng = create_engine(os.environ[cfg["db_url_ro_env"]], hide_parameters=True)
    p = {"live": cfg["status_live"], "qt": cfg["question_text"], "m": minutes}
    with eng.connect() as c:
        rows = c.execute(text(
            "SELECT c.claim_id, jsonb_array_length(c.evidence_span_ids) FROM vera_vjay.claims c JOIN vera_vjay.answers a ON a.answer_id=c.answer_id "
            "WHERE c.detailed_status=:live AND a.response_text=:qt AND c.created_at > now() - make_interval(mins => :m) "
            "ORDER BY c.claim_id"), p).fetchall()
        ph = c.execute(text(
            "SELECT (SELECT count(*) FROM vera_vjay.questions WHERE research_question=:qt), "
            "(SELECT count(*) FROM vera_vjay.runs WHERE question=:qt), "
            "(SELECT count(*) FROM vera_vjay.questions q JOIN vera_vjay.runs r ON r.question_id=q.question_id "
            " JOIN vera_vjay.answers a ON a.run_id=r.run_id WHERE a.model=:g AND a.response_text=:qt "
            " AND (q.research_question<>:qt OR r.question<>:qt))"), {"qt": cfg["question_text"], "g": cfg["gate_version"]}).fetchone()
        real = [c.execute(text("SELECT (SELECT count(*) FROM vera_vjay.questions WHERE research_question=:t)"
                               " + (SELECT count(*) FROM vera_vjay.runs WHERE question=:t)"
                               " + (SELECT count(*) FROM vera_vjay.answers WHERE response_text=:t)"), {"t": question(q)}).scalar()
                for q in qids]
    return {"real_question_rows_per_qid": real, **split_ids(rows), "placeholder_questions": ph[0], "placeholder_runs": ph[1],
            "memory_answer_rows_with_nonplaceholder_question": ph[2]}


def recall_related(qid: str, related: str) -> dict:
    """ro reader's recall only (SELECT). Needs only the ro URL env var; never touches /ask or a write account."""
    from sqlalchemy import create_engine
    from vera.memory_store import RoMemoryReader
    cfg = json.loads((ROOT / "config" / "ask-provider-chain.json").read_text())["memory"]["store"]
    rd = RoMemoryReader(create_engine(os.environ[cfg["db_url_ro_env"]], hide_parameters=True), cfg)
    from vera import answer_contract
    out = {"evidence_recall_enabled": answer_contract.evidence_recall_enabled()}  # app path returns [] when False
    for label, q in (("qid", question(qid)), ("related", related)):
        rows = rd.recall(q)
        out[label] = {"terms": rd.query_terms(q), "recalled": len(rows), "first": [r["claim_text"][:80] for r in rows]}
    return out


def _ops_guard() -> None:
    """ENV-E (E7/C8): refuse under a test runner; classify-and-tag every connection as an APP context (never disposable)."""
    if os.environ.get("VERA_TEST_RUNNER") or os.environ.get("PYTEST_CURRENT_TEST"):
        sys.stderr.write("refused: ops script, not for test runs\n")
        raise SystemExit(3)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from safety import connect_guard
    connect_guard.install("production")


if __name__ == "__main__":
    _ops_guard()
    mode, arg = sys.argv[1], sys.argv[2]
    if mode == "recall-related":
        print(json.dumps(recall_related(arg, sys.argv[3]), default=str))
        sys.exit(0)
    out = count(int(arg), tuple(sys.argv[3:])) if mode == "count" else ask(arg)
    print(json.dumps(out, default=str))
