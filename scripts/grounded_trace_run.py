"""Item #79: run the 30 frozen trace questions through the grounded /ask path and measure R1's criteria.

Measures, per R1's grounded-default criteria: unresolvable citations (citation numbers with no retrieved
source, and cited source identifiers that do not resolve over HTTP) and the unsupported-claim rate from the
claim check, with Wilson 95% intervals (directional, small n). No corpus writes. Usage:
    PYTHONPATH=. .venv/bin/python scripts/grounded_trace_run.py [--limit N] [--out eval_results/grounded_ask_v1.json]
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "config" / "trace_questions_v1.json"
PACE_S = 4.0          # between questions: arXiv asks for spacing, Groq free tier has per-minute limits
RESOLVE_TIMEOUT_S = 8
MAX_RESOLVE_PER_ANSWER = 6


def _resolves(url: str) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=RESOLVE_TIMEOUT_S,
                          headers={"User-Agent": "VERA-trace-check/1.0"})
        if r.status_code in (403, 405):  # some hosts refuse HEAD; a GET of the landing page decides
            r = requests.get(url, allow_redirects=True, timeout=RESOLVE_TIMEOUT_S, stream=True,
                             headers={"User-Agent": "VERA-trace-check/1.0"})
        return r.status_code < 400 or r.status_code == 403  # 403: exists but bot-blocked (publisher pages)
    except requests.RequestException:
        return False


def _is_conn_error(exc: BaseException) -> bool:
    """C12: driver, SQLAlchemy and safety-layer errors may carry connection data in their text."""
    return any(c.__module__.split(".")[0] in ("sqlalchemy", "psycopg2", "psycopg", "safety") for c in type(exc).__mro__)


def _err_text(exc: BaseException) -> str:
    """C12: connection-class errors are shown by type name plus code only; others keep their text."""
    if _is_conn_error(exc):
        code = getattr(exc, "code", None)
        return type(exc).__name__ + (f"[{code}]" if isinstance(code, str) and code.replace("_", "").isalnum() else "")
    return f"{type(exc).__name__}: {str(exc)[:160]}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=str(ROOT / "eval_results" / "grounded_ask_v1.json"))
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from vera.env_files import load_vera_env
    load_vera_env()  # PLAN-ENV
    import main as api
    from vera.grounded_ask import retrieve_all
    from vera.inference_chain import answer_via_chain
    from vera.trace_eval.guidance import wilson, z_for

    doc = json.loads(QUESTIONS.read_text())
    items = doc["items"][: a.limit] if a.limit else doc["items"]
    records, start = [], time.monotonic()
    for i, q in enumerate(items, 1):
        t = time.monotonic()
        cache = {}

        def sources_fn():
            if "s" not in cache:
                cache["s"], cache["u"] = retrieve_all(q["text"], api.GROUNDING)
            return cache["s"]

        rec = {"id": q["id"], "split": q.get("split"), "category": q.get("category")}
        try:
            result, model = answer_via_chain(q["text"], api.ASK_CHAIN, (sources_fn, api.GROUNDING))
            srcs = list(result.sources)
            cited = sorted({int(n) for g in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", result.answer) for n in g.split(",")})
            bad_numbers = [n for n in cited if not 1 <= n <= len(srcs)]
            unresolved = [s["url"] for s in srcs if s["n"] in cited][:MAX_RESOLVE_PER_ANSWER]
            unresolved = [u for u in unresolved if not _resolves(u)]
            rec.update(model=model, answer=result.answer, n_sources=len(srcs), cited=cited,
                       bad_citation_numbers=bad_numbers, unresolved_cited_urls=unresolved,
                       claim_check=result.claim_check, sources=[{k: s.get(k) for k in ("n", "title", "url", "identifier",
                                                                                      "provider", "licence_decision")}
                                                               for s in srcs], error=None)
        except Exception as exc:  # noqa: BLE001  (recorded per question)
            rec.update(error=_err_text(exc))
        rec["latency_s"] = round(time.monotonic() - t, 2)
        records.append(rec)
        cc = rec.get("claim_check") or {}
        print(f"[{i}/{len(items)}] {q['id']} {q.get('category')}: {rec.get('model', '-')} sources={rec.get('n_sources', 0)} "
              f"cited={len(rec.get('cited') or [])} check={cc.get('outcome', '-')} {rec['latency_s']}s"
              + (f" ERROR {rec['error']}" if rec["error"] else ""), flush=True)
        time.sleep(PACE_S)

    z = z_for(0.05)
    answered = [r for r in records if (r.get("claim_check") or {}).get("outcome") == "checked"]
    checked = sum(r["claim_check"]["checked"] for r in answered)
    removed = sum(r["claim_check"]["removed"] for r in answered)
    partial = sum(r["claim_check"]["partial"] for r in answered)
    cited_total = sum(len(r.get("cited") or []) for r in records)
    unresolvable = sum(len(r.get("bad_citation_numbers") or []) + len(r.get("unresolved_cited_urls") or []) for r in records)
    outcomes = {}
    for r in records:
        key = "error" if r.get("error") else (r.get("claim_check") or {}).get("outcome") or ("no_sources" if not r.get("n_sources") else "ungrounded")
        outcomes[key] = outcomes.get(key, 0) + 1
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions": len(records), "questions_file": QUESTIONS.name, "outcomes": outcomes,
        "cited_sources": cited_total, "unresolvable_citations": unresolvable,
        "claims_checked": checked, "claims_removed_unsupported": removed, "claims_partial": partial,
        "unsupported_claim_rate": round(removed / checked, 4) if checked else None,
        "unsupported_claim_rate_wilson95": wilson(removed, checked, z) if checked else None,
        "partial_or_unsupported_rate": round((removed + partial) / checked, 4) if checked else None,
        "partial_or_unsupported_wilson95": wilson(removed + partial, checked, z) if checked else None,
        "checkers_used": sorted({r["claim_check"].get("checker") for r in answered if r["claim_check"].get("checker")}),
        "caveats": "Directional, small n. The claim check is a model judgement against abstracts only (its own "
                   "error rate is unmeasured here); questions are agent-authored; one run.",
        "elapsed_s": round(time.monotonic() - start, 1),
    }
    Path(a.out).write_text(json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
