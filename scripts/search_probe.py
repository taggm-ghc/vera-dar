"""Search probe (item #71): run search_question over trace question ids, write the top-5 as JSONL.

No DB, no LLM. It DOES make live network calls to the configured search providers when run, so run it
deliberately; the shared cached chain keeps each provider's rate gate across questions.

  venv/bin/python scripts/search_probe.py --label arxiv_only_fix --ids q01 q02 q06
  venv/bin/python scripts/search_probe.py --label after_fanout --all-substantive   # q01-q22

Output: tmp/search_eval/<label>.jsonl, one line per result (and one per failed question, with the error).
Bounded: at most MAX_QUESTIONS questions per run; each question is bounded by search_question's deadline.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTIONS_PATH = ROOT / "config" / "trace_questions_v1.json"
OUT_DIR = ROOT / "tmp" / "search_eval"
MAX_QUESTIONS = 30
TOP_N = 5


def load_questions() -> dict[str, str]:
    doc = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return {q["id"]: q["text"] for q in doc["items"]}


def result_row(qid: str, r: dict, meta: dict) -> dict:
    return {"question_id": qid, "id": r.get("url"), "provider": r.get("provider"),
            "providers": r.get("found_by_providers") or [r.get("provider")], "rank": r.get("rank"),
            "title": r.get("title"), "url": r.get("url"), "published": r.get("published"),
            "snippet": (r.get("snippet") or "")[:300],
            "source_type": r.get("source_type"), "discovery": r.get("discovery"), "mode": meta.get("mode")}


def main(argv=None, search=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True, help="output file stem under tmp/search_eval/")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ids", nargs="+", help="question ids, e.g. q01 q02")
    g.add_argument("--all-substantive", action="store_true", help="q01-q22")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", a.label):
        print("label must match [A-Za-z0-9_.-]+", file=sys.stderr)
        return 2
    qs = load_questions()
    ids = a.ids if a.ids else [f"q{i:02d}" for i in range(1, 23)]
    unknown = [i for i in ids if i not in qs]
    if unknown or len(ids) > MAX_QUESTIONS:
        print(f"unknown ids {unknown} or more than {MAX_QUESTIONS} questions; nothing run", file=sys.stderr)
        return 2
    if search is None:
        from vera.search_and_fetch import search_question as search
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{a.label}.jsonl"
    failed = 0
    with out_path.open("w", encoding="utf-8") as f:
        for qid in ids:  # the cached chain's RateGates space the calls
            try:
                res = search(qs[qid], num_results=TOP_N)
            except Exception as e:  # noqa: BLE001  (verbose, per question; the run continues, exit code says)
                failed += 1
                f.write(json.dumps({"question_id": qid, "error": f"{type(e).__name__}: {e}"}) + "\n")
                print(f"{qid}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
                continue
            meta = getattr(res, "meta", {}) or {}
            for r in list(res)[:TOP_N]:
                f.write(json.dumps(result_row(qid, r, meta), ensure_ascii=False) + "\n")
            print(f"{qid}: {len(res)} results via {meta.get('route_used')}", file=sys.stderr)
    print(f"wrote {out_path}; {failed} of {len(ids)} questions failed", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
