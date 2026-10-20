"""Capture raw answer traces for the item #70 trace evaluation.

Why: error analysis needs the full request, the sources shown, the answer and its cost, one JSONL
record per question, so later passes (open coding, checks) read evidence and not summaries.
Two variants: `baseline` sends exactly what /ask sends; `grounded_v1` adds search results and a
cite-only system prompt. The chat and search callables are injected so tests use fakes and the
live run is a separate, orchestrator-owned step.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

from openai import AuthenticationError

from vera.pricing.config import PricingRecord

# Why: recorded in every trace so a later prompt change cannot be confused with this one.
PROMPT_VERSION_BASELINE = "baseline-v1"
PROMPT_VERSION_GROUNDED = "grounded-v1"
PROMPT_VERSIONS = {"baseline": PROMPT_VERSION_BASELINE, "grounded_v1": PROMPT_VERSION_GROUNDED}

# Why: the questions file is only evidence once frozen; a draft may still change under us.
REQUIRED_STATUS = "FROZEN"

# Why: bounds one model call so a hung request fails verbosely instead of stalling the run.
CHAT_TIMEOUT_S = 60

# Why: the harness's total spend guard (plan item 70, section 8 step 2).
DEFAULT_MAX_COST_USD = 0.50

# Why: abstracts are all the sources carry; the prompt must make the model say so.
GROUNDED_SYSTEM_PROMPT_V1 = (
    "You are VERA, a research assistant on published evidence (2023-2026) about AI coding assistants "
    "and developer productivity.\n"
    "Use ONLY the numbered sources below. They are arXiv abstracts only, so methods, limitations and "
    "exact numbers may be missing; say so when that matters.\n"
    "Cite each claim as [n]. Never cite anything that is not listed.\n"
    "If the sources do not answer the question, or it is out of scope, say so plainly instead of "
    "answering from memory.\n"
    "If the question rests on a false premise, correct it.\n\n"
    "Sources:\n{sources}"
)

ChatFn = Callable[[str, list[dict]], "tuple[str, int, int]"]
SearchFn = Callable[[str], "list[dict]"]


def load_questions(path: str | Path) -> tuple[dict, str]:
    """Return (document, sha256 of the file bytes); refuse anything not FROZEN."""
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    if doc.get("status") != REQUIRED_STATUS:
        raise ValueError(f"questions file status is {doc.get('status')!r}, need {REQUIRED_STATUS!r}: {path}")
    return doc, hashlib.sha256(raw).hexdigest()


def baseline_messages(question: str) -> list[dict]:
    """Identical to vera.ask_service.answer_question's request."""
    return [{"role": "user", "content": question}]


from vera.source_labels import source_label  # noqa: E402


def _format_sources(sources: list[dict]) -> str:
    # item #71 R6-a / #72 R72-f: shared label helper (attribution/display; no behavioural effect is claimed).
    def label(s: dict) -> str:
        lab = source_label(s)
        return lab + " " if lab else ""
    return "\n\n".join(
        f"[{s['n']}] {label(s)}{s['title']} (published {s.get('published') or 'unknown'}) {s['url']}\n{s['snippet']}"
        for s in sources
    )


def grounded_v1_messages(question: str, sources: list[dict]) -> list[dict]:
    system = GROUNDED_SYSTEM_PROMPT_V1.format(sources=_format_sources(sources) or "(none found)")
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def _cost(pricing: PricingRecord, prompt_tokens: int, completion_tokens: int) -> float:
    return round(prompt_tokens / 1_000_000 * pricing.input + completion_tokens / 1_000_000 * pricing.output, 6)


def capture_one(q_item: dict, variant: str, *, chat: ChatFn, search: SearchFn, pricing: PricingRecord | None = None,
                model: str = "", questions_sha256: str = "") -> dict:
    """One trace record. Per-question failures are recorded; only authentication errors propagate."""
    question = q_item["text"]
    rec = {
        "id": q_item["id"], "split": q_item.get("split"), "category": q_item.get("category"),
        "variant": variant, "prompt_version": PROMPT_VERSIONS[variant], "questions_sha256": questions_sha256,
        "model": model, "messages": None, "sources": [], "answer": None, "prompt_tokens": None,
        "completion_tokens": None, "cost_usd": None, "latency_s": None, "search_meta": None, "error": None,
    }
    start = time.monotonic()
    try:
        if variant == "baseline":
            messages = baseline_messages(question)
        else:
            results = search(question)
            rec["search_meta"] = dict(getattr(results, "meta", None) or {})
            rec["sources"] = [
                {"n": i, "url": r.get("url", ""), "title": r.get("title", ""),
                 "published": r.get("published"), "snippet": r.get("snippet", ""),
                 "source_type": r.get("source_type"), "source_class": r.get("source_class"), "discovery": r.get("discovery")}
                for i, r in enumerate(results, 1)
            ]
            messages = grounded_v1_messages(question, rec["sources"])
        rec["messages"] = messages
        text, pt, ct = chat(model, messages)
        rec.update(answer=text, prompt_tokens=pt, completion_tokens=ct,
                   cost_usd=_cost(pricing, pt, ct) if pricing else None)
    except AuthenticationError:
        raise
    except Exception as e:  # noqa: BLE001  (recorded per question, never silent)
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    rec["latency_s"] = round(time.monotonic() - start, 3)
    return rec


def run_capture(questions: dict, variant: str, out_path: str | Path, *, chat: ChatFn, search: SearchFn,
                pricing: PricingRecord, model: str, max_cost_usd: float = DEFAULT_MAX_COST_USD,
                splits: set[str] | list[str] | None = None, questions_sha256: str = "",
                limit: int | None = None) -> dict:
    """Append one JSONL record per question; stop verbosely before the cost guard would be exceeded."""
    items = [q for q in questions["items"] if not splits or q.get("split") in set(splits)]
    if limit is not None:
        items = items[:limit]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    total, errors, done, worst, stopped = 0.0, 0, 0, 0.0, None
    with out.open("a", encoding="utf-8") as fh:
        for q in items:
            # Why: the next call's cost is unknown, so assume it can match the dearest call so far.
            if total + worst > max_cost_usd or total >= max_cost_usd:
                stopped = (f"cost guard: cumulative ${total:.6f} plus worst call so far ${worst:.6f} "
                           f"would exceed ${max_cost_usd:.2f}; stopped before {q['id']}")
                break
            rec = capture_one(q, variant, chat=chat, search=search, pricing=pricing, model=model,
                              questions_sha256=questions_sha256)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done += 1
            errors += 1 if rec["error"] else 0
            cost = rec["cost_usd"] or 0.0
            total += cost
            worst = max(worst, cost)
    return {"n": done, "planned": len(items), "errors": errors, "total_cost_usd": round(total, 6),
            "stopped": stopped}


def default_chat(model: str) -> ChatFn:
    from vera.ask_service import _get_client

    def chat(model_: str, messages: list[dict]) -> tuple[str, int, int]:
        c = _get_client().chat.completions.create(model=model_ or model, messages=messages, timeout=CHAT_TIMEOUT_S)
        return c.choices[0].message.content, c.usage.prompt_tokens, c.usage.completion_tokens

    return chat


# Why: search_question() without `providers` rebuilds the chain, and with it each provider's RateGate, on
# every call, so arXiv's min_interval_s is not enforced ACROSS questions. Found by item #70's first grounded
# capture (HTTP 429 from question 21 on). The harness therefore builds the chain once per process; the M2
# defect itself is fixed by p3m3 item #71 (search_question now caches its chain); this cache is kept for
# explicitness.
_CHAIN_CACHE: dict = {}


def default_search(question: str) -> list[dict]:
    from vera.search_and_fetch import default_chain_and_min, search_question

    if "chain" not in _CHAIN_CACHE:
        _CHAIN_CACHE["chain"], _CHAIN_CACHE["min"] = default_chain_and_min()
    return search_question(question, num_results=5, providers=_CHAIN_CACHE["chain"],
                           min_candidates=_CHAIN_CACHE["min"])
