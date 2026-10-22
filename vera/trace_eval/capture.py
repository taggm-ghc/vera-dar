"""Capture raw answer traces for the item #70 trace evaluation.

Why: error analysis needs the full request, the sources shown, the answer and its cost, one JSONL
record per question, so later passes (open coding, checks) read evidence and not summaries.
Three variants: `baseline` is the historical bare model call; `grounded_v1` adds search results and a
cite-only system prompt; `ask_api` (TRACEEVAL-1022) posts each question to VERA-DAR's real /ask route,
over HTTP or in-process, so the trace is what a user gets. The chat, search and ask callables are
injected so tests use fakes and the live run is a separate, owner-authorized step.
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
# Why: ask_api sends no prompt of its own; the server owns the prompt, so the version names the route instead.
PROMPT_VERSION_ASK_API = "ask-api-route"
PROMPT_VERSIONS = {"baseline": PROMPT_VERSION_BASELINE, "grounded_v1": PROMPT_VERSION_GROUNDED,
                   "ask_api": PROMPT_VERSION_ASK_API}
VARIANTS = tuple(PROMPT_VERSIONS)

# Why: the categories the v1 checks are gated on (vera.trace_eval.checks); a case set without them is adapted
# by explicit, config-held rules (config/trace_eval_guidance.json "case_sets"), never by guessing.
CHECK_CATEGORIES = ("in_scope_synthesis", "multi_part", "false_premise", "out_of_scope", "ambiguous_or_underspecified")

# Why: /ask fields copied into a trace source; the response carries no passage text by design (no abstracts).
ASK_SOURCE_FIELDS = ("n", "url", "title", "published", "source_type")

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
AskFn = Callable[[str], dict]  # question -> parsed /ask JSON body; raises AskHTTPError on a non-200 status


class AskHTTPError(RuntimeError):
    """A non-200 /ask status. Carries the status code only (never the body, URL or headers)."""

    def __init__(self, status: int):
        super().__init__(f"/ask returned HTTP {status}")
        self.status = status


# Why: these statuses mean the route cannot answer ANY question (bad key, misconfigured server); stop the run.
ASK_ABORT_STATUSES = frozenset({401, 403, 500, 503})


def load_case_set(path: str | Path, case_set: str, cfg: dict | None = None) -> tuple[dict, str]:
    """Return (questions document in capture shape, sha256 of the file bytes).

    `frozen`: the v1 trace question file (FROZEN status required; categories and splits as authored; the dev-split
    labels in docs/trace_eval/ were made on it). Any other name: an un-frozen set adapted by the rules in
    cfg["case_sets"][name] (first matching rule wins), recorded as not validated for the checks."""
    if case_set == "frozen":
        doc, sha = load_questions(path)
        return {**doc, "case_set": "frozen", "checks_validated": True}, sha
    rules = ((cfg or {}).get("case_sets") or {}).get(case_set)
    if not rules:
        raise ValueError(f"no case_sets rules for {case_set!r} in the guidance config")
    raw = Path(path).read_bytes()
    src = json.loads(raw)
    items = []
    for q in src.get(rules["items_key"], []):
        cat = next((r["category"] for r in rules["rules"]
                    if all(str(v).lower() in str(q.get(k, "")).lower() for k, v in r.get("match", {}).items())),
                   rules["default_category"])
        if cat not in CHECK_CATEGORIES:
            raise ValueError(f"case_sets rule maps to unknown category {cat!r}")
        items.append({"id": q["id"], "text": q["text"], "category": cat, "split": rules.get("split", "dev")})
    if not items:
        raise ValueError(f"case set {case_set!r} has no questions under {rules['items_key']!r}")
    return {"status": "ADAPTED", "items": items, "case_set": case_set, "checks_validated": False}, \
        hashlib.sha256(raw).hexdigest()


# Why: the /ask fields that show citation and evidence behaviour. Kept in the private trace only (traces.jsonl);
# trace_row and the closed public projection never copy them. Quotes are already short (per-response budget).
ASK_DETAIL_FIELDS = ("citations", "admitted_evidence", "examined_full_documents", "claim_check", "evidence_status", "contract")


def ask_record_fields(body: dict) -> dict:
    """Trace fields from one /ask JSON body: answer, numbered sources (no passage text), cost, outcome."""
    sources = [{k: s.get(k) for k in ASK_SOURCE_FIELDS} | {"snippet": ""} for s in body.get("sources") or []]
    tokens = body.get("tokens_used")
    return {"answer": body.get("answer"), "sources": sources, "cost_usd": body.get("cost_usd"),
            "prompt_tokens": None, "completion_tokens": None, "tokens_used": tokens,
            "result_kind": body.get("result_kind"), "outcome": body.get("outcome"),
            "source_text_available": False, "model": body.get("model") or "",
            "ask_detail": {k: body[k] for k in ASK_DETAIL_FIELDS if body.get(k) is not None}}


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


def capture_one(q_item: dict, variant: str, *, chat: ChatFn | None = None, search: SearchFn | None = None,
                pricing: PricingRecord | None = None, model: str = "", questions_sha256: str = "",
                ask: AskFn | None = None) -> dict:
    """One trace record. Per-question failures are recorded; only authentication errors propagate."""
    question = q_item["text"]
    rec = {
        "id": q_item["id"], "split": q_item.get("split"), "category": q_item.get("category"),
        "variant": variant, "prompt_version": PROMPT_VERSIONS[variant], "questions_sha256": questions_sha256,
        "model": model, "messages": None, "sources": [], "answer": None, "prompt_tokens": None,
        "completion_tokens": None, "cost_usd": None, "latency_s": None, "search_meta": None, "error": None,
    }
    start = time.monotonic()
    if variant == "ask_api":
        rec["messages"] = baseline_messages(question)  # what the route receives (the user turn only)
        try:
            rec.update(ask_record_fields(ask(question)))
        except AskHTTPError as e:
            if e.status in ASK_ABORT_STATUSES:
                raise
            rec["error"] = f"AskHTTPError: HTTP {e.status}"
        except Exception as e:  # noqa: BLE001  (recorded per question, type only: a message may carry a host)
            rec["error"] = type(e).__name__
        rec["latency_s"] = round(time.monotonic() - start, 3)
        return rec
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


def run_capture(questions: dict, variant: str, out_path: str | Path, *, chat: ChatFn | None = None,
                search: SearchFn | None = None, pricing: PricingRecord | None = None, model: str = "",
                max_cost_usd: float = DEFAULT_MAX_COST_USD, splits: set[str] | list[str] | None = None,
                questions_sha256: str = "", limit: int | None = None, ask: AskFn | None = None) -> dict:
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
                              questions_sha256=questions_sha256, ask=ask)
            rec["case_set"] = questions.get("case_set", "frozen")
            rec["checks_validated"] = questions.get("checks_validated", True)
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


def http_ask(base_url: str, api_key: str, timeout: int | None = None) -> AskFn:
    """POST /ask over HTTP through vera.public_mode.post_ask (the UI's own client). The base URL and key are
    closed over and never recorded, printed or raised."""
    from vera.public_mode import ASK_TIMEOUT_S, post_ask

    def ask(question: str) -> dict:
        r = post_ask(base_url, question, "normal", api_key, timeout=timeout or ASK_TIMEOUT_S)
        if r.status_code != 200:
            raise AskHTTPError(r.status_code)
        return r.json()

    return ask


def inprocess_ask(api_key: str) -> AskFn:
    """POST /ask in-process through FastAPI's TestClient on main.app: the real route, provider chain and grounding,
    with no listening socket. Importing main loads its configuration; LLM calls go through its provider chain."""
    from fastapi.testclient import TestClient

    import main  # noqa: PLC0415  (deliberately lazy: only an authorized run imports the API)

    client = TestClient(main.app)

    def ask(question: str) -> dict:
        r = client.post("/ask", json={"question": question, "mode": "normal"}, headers={"X-API-Key": api_key})
        if r.status_code != 200:
            raise AskHTTPError(r.status_code)
        return r.json()

    return ask
