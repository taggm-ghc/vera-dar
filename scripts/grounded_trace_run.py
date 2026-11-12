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


MAX_RESOLVE_REDIRECTS = 3


def _resolves(url: str, *, http_get=None, robots_ok=None):
    """True: the page exists. False: it does not (or is unreachable). None: not checked (robots.txt disallows or could
    not be read on some hop; no document request was sent). DENIAL-COVERAGE-1111: pinned GET only, manual redirects
    (at most MAX_RESOLVE_REDIRECTS), each hop through the access-denial guard and robots-first. 403 means the page
    exists (bot-blocked publisher pages); a URL already denied is not requested again and also counts as existing."""
    from urllib.parse import urljoin

    from vera import selective_fetch as sf
    get = http_get or sf.pinned_get
    ua = "VERA-trace-check/1.0"
    ok_robots = robots_ok or (lambda u: sf.robots_allowed(u, http_get=get, url_check=sf.is_public_url, user_agent=ua))
    cur = url
    try:
        for _hop in range(MAX_RESOLVE_REDIRECTS + 1):
            try:
                sf.document_request_guard(cur)
            except sf.DocumentDenied:
                return True
            ok, _why = sf.is_public_url(cur)
            if not ok:
                return False
            if not ok_robots(cur):
                return None
            r = get(cur, headers={"User-Agent": ua}, timeout=RESOLVE_TIMEOUT_S, allow_redirects=False, stream=True)
            try:
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
                    cur = urljoin(cur, r.headers["Location"])
                    continue
                return r.status_code < 400 or r.status_code == 403
            finally:
                r.close()
        return False  # redirect cap used up
    except (requests.RequestException, ValueError):
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
            unresolved = [u for u in unresolved if _resolves(u) is False]  # None = not checked
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
