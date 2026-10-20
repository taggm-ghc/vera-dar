"""M6 baseline: direct LLM call, no context engineering, same model as M5.

Three runs for consistency; the median response (by output length, the only
ordering available for free text) is returned. Total spend of all three runs is
reported separately from the median run's own cost/latency so the comparison
stays apples-to-apples while the true spend stays visible.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable

from vera.pricing.config import latest_pricing_for, load_model_pricing

DEFAULT_MODEL = "gpt-4.1-nano"  # cheapest acceptable; same model as M5 (R1 rule)
N_RUNS = 3
MAX_OUTPUT_TOKENS = 2500  # per-run step cap


def _openai_call(question: str, model: str) -> dict:
    from vera.ask_service import _get_client
    t0 = time.monotonic()
    c = _get_client().chat.completions.create(
        model=model, messages=[{"role": "user", "content": question}],
        max_tokens=MAX_OUTPUT_TOKENS)
    return {"text": c.choices[0].message.content or "",
            "prompt_tokens": c.usage.prompt_tokens,
            "completion_tokens": c.usage.completion_tokens,
            "latency_s": time.monotonic() - t0}


def _cost(model: str, ptok: int, ctok: int) -> float:
    rec = latest_pricing_for(model, load_model_pricing())
    if rec is None:
        raise RuntimeError(f"No pricing record for {model!r} in config/model-pricing.json; "
                           "refusing to report an unpriced cost as $0.")
    return round(ptok / 1e6 * rec.input + ctok / 1e6 * rec.output, 6)


def collect_baseline(question: str, model: str = DEFAULT_MODEL, *,
                     llm_call: Callable[[str, str], dict] | None = None,
                     n_runs: int = N_RUNS) -> dict:
    """Return the median baseline run plus per-run logs.

    `llm_call(question, model) -> {text, prompt_tokens, completion_tokens, latency_s}`
    is injectable so tests never touch the network.
    """
    call = llm_call or _openai_call
    runs = []
    for i in range(n_runs):  # bounded: exactly n_runs steps, no retry loop
        r = call(question, model)
        r["cost_usd"] = _cost(model, r["prompt_tokens"], r["completion_tokens"])
        r["tokens"] = r["prompt_tokens"] + r["completion_tokens"]
        r["run_index"] = i
        runs.append(r)
    order = sorted(range(len(runs)), key=lambda i: len(runs[i]["text"]))
    med = runs[order[len(order) // 2]]
    return {
        "response_text": med["text"],
        "model": model,
        "method": "direct call, no context engineering; median of "
                  f"{n_runs} runs by output length",
        "usage": {"api_calls": 1, "tokens": med["tokens"],
                  "cost_usd": med["cost_usd"], "latency_s": round(med["latency_s"], 3)},
        "all_runs_usage": {
            "api_calls": len(runs),
            "tokens": sum(r["tokens"] for r in runs),
            "cost_usd": round(sum(r["cost_usd"] for r in runs), 6),
            "latency_s": round(sum(r["latency_s"] for r in runs), 3),
            "output_length_median": statistics.median(len(r["text"]) for r in runs),
        },
        "runs": [{k: r[k] for k in ("run_index", "tokens", "cost_usd", "latency_s")}
                 for r in runs],
        "question": question,
    }
