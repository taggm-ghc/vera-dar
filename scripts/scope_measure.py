"""Measure the item #72 W1 scope router on the frozen scope set v1 (layer 1 only, and layer 1 + LLM).

Reports false-refusal rate (in-scope declined) and false-accept rate (out-of-scope accepted) with Wilson 95%
intervals, n per class, LLM calls and cost. Directional, small n; the set is agent-authored (see its authoring note).

    venv/bin/python scripts/scope_measure.py [--modes none,llm] [--cap-usd 0.05] [--force]

The OpenAI key is read the way the app reads it (the explicit R/config/env/.env-vera loader, or the process environment);
no value is printed. Missing/placeholder key with mode llm: exits 2 with a verbose message. Spend is capped:
before each call the worst-case cost of that call is added to the running total and the run stops if it would
exceed --cap-usd. Output: tmp/scope_eval/scope_measure_v1.json (gitignored); refuses to overwrite without --force.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "tmp" / "scope_eval" / "scope_measure_v1.json"
TRACE_Q = ROOT / "config" / "trace_questions_v1.json"
MAX_CALLS = 120           # hard step bound across all modes
MAX_CONSECUTIVE_ERRORS = 3
CHARS_PER_TOKEN_FLOOR = 3  # pessimistic estimate for the worst-case prompt token count


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, round(centre - half, 4)), min(1.0, round(centre + half, 4)))


def fail(msg: str, code: int = 2) -> None:
    print(f"scope_measure: FAILED: {msg}", file=sys.stderr)
    sys.exit(code)


def load_set(router_raw: dict) -> tuple[dict, str]:
    p = ROOT / router_raw["scope_set"]["path"]
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != router_raw["scope_set"]["sha256"]:
        fail(f"scope set sha256 {sha[:12]} != recorded {router_raw['scope_set']['sha256'][:12]}; set changed after freeze")
    doc = json.loads(raw)
    if doc.get("status") != "FROZEN":
        fail(f"scope set status is {doc.get('status')!r}, need 'FROZEN'")
    return doc, sha


def summarise(rows: list[dict]) -> dict:
    ins = [r for r in rows if r["label"] == "in_scope"]
    outs = [r for r in rows if r["label"] == "out_of_scope"]
    fr = sum(1 for r in ins if not r["accepted"])
    fa = sum(1 for r in outs if r["accepted"])
    by_layer: dict[str, int] = {}
    for r in rows:
        by_layer[r["layer"]] = by_layer.get(r["layer"], 0) + 1
    return {
        "n_in_scope": len(ins), "n_out_of_scope": len(outs),
        "false_refusals": fr, "false_refusal_rate": round(fr / len(ins), 4) if ins else None,
        "false_refusal_wilson95": wilson(fr, len(ins)),
        "false_accepts": fa, "false_accept_rate": round(fa / len(outs), 4) if outs else None,
        "false_accept_wilson95": wilson(fa, len(outs)),
        "decided_by_layer": by_layer,
        "false_refusal_ids": [r["id"] for r in ins if not r["accepted"]],
        "false_accept_ids": [r["id"] for r in outs if r["accepted"]],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="none,llm")
    ap.add_argument("--cap-usd", type=float, default=0.05)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    if OUT.exists() and not a.force:
        fail(f"{OUT.relative_to(ROOT)} exists; the measurement is run once (use --force to overwrite)")

    from vera import scope_router as sr
    from vera.pricing.config import latest_pricing_for, load_model_pricing, load_model_selection

    router_raw = json.loads(sr.CONFIG_PATH.read_text(encoding="utf-8"))
    doc, sha = load_set(router_raw)
    sel = load_model_selection()
    model = sel.selected_model if sel else None
    pricing = latest_pricing_for(model, load_model_pricing()) if model else None

    if "llm" in modes:
        from vera.env_files import load_vera_env

        from vera.auth import classify_openai_api_key
        load_vera_env()  # PLAN-ENV (the app's mechanism, main.py); prints nothing
        status = classify_openai_api_key(os.getenv("OPENAI_API_KEY"))
        if status != "ok":
            fail(f"OPENAI_API_KEY is {status} (value not shown); set it in R/config/env/.env-vera or the environment")
        if model is None or pricing is None:
            fail(f"no model selection or pricing record (model={model!r}); cannot cap spend")

    l2 = router_raw["layer2"]
    spent = 0.0
    calls = 0
    result: dict = {"schema": "vera-scope-measure-v1", "scope_set_sha256": sha, "router_config_version":
                    router_raw.get("version"), "model": model, "cap_usd": a.cap_usd,
                    "label": "directional, small n; agent-authored set (see scope_set_v1 authoring_note)",
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "modes": {}}

    for mode in modes:
        cfg = sr.parse_config(router_raw, layer2_mode=mode)
        rows, consecutive_err, stopped = [], 0, None
        for item in doc["items"]:
            chat = None
            if mode == "llm":
                if calls >= MAX_CALLS:
                    stopped = f"step bound MAX_CALLS={MAX_CALLS} reached"
                    break
                msgs = sr.classifier_messages(item["text"], cfg)
                est_in = sum(len(m["content"]) for m in msgs) // CHARS_PER_TOKEN_FLOOR + 20
                worst = est_in / 1e6 * pricing.input + l2["max_tokens"] / 1e6 * pricing.output
                if spent + worst > a.cap_usd:
                    stopped = f"spend cap: spent {spent:.6f} + worst-case {worst:.6f} > cap {a.cap_usd}"
                    break
            t0 = time.monotonic()
            try:
                d = sr.route(item["text"], model or "none", pricing, cfg=cfg, chat=chat)
                err = None
                consecutive_err = 0
            except Exception as exc:  # recorded verbosely; counted as a decline (fail closed)
                d = sr.ScopeDecision(False, "error", f"error:{type(exc).__name__}", 0, 0.0, 1)
                err = f"{type(exc).__name__}: {str(exc)[:200]}"
                consecutive_err += 1
            calls += d.llm_calls
            spent += d.cost_usd
            rows.append({"id": item["id"], "label": item["label"], "subcategory": item["subcategory"],
                         "accepted": d.accepted, "layer": d.layer, "reason": d.reason, "cost_usd": d.cost_usd,
                         "latency_s": round(time.monotonic() - t0, 3), "error": err})
            if consecutive_err >= MAX_CONSECUTIVE_ERRORS:
                stopped = f"{consecutive_err} consecutive errors; last: {err}"
                break
        q23 = next((q for q in json.loads(TRACE_Q.read_text())["items"] if q["id"] == "q23"), None)
        q23_dec = sr.layer1(q23["text"], cfg) if q23 else None
        result["modes"][mode] = {
            "complete": stopped is None and len(rows) == len(doc["items"]), "stopped": stopped,
            "n_routed": len(rows), "llm_calls": sum(r["layer"] in ("layer2_llm", "error") for r in rows),
            "errors": sum(1 for r in rows if r["error"]),
            "cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
            "trace_q23_layer1": q23_dec, **summarise(rows), "rows": rows,
        }
        if stopped:
            print(f"scope_measure: mode {mode} STOPPED early: {stopped}", file=sys.stderr)

    result["total_llm_calls"] = calls
    result["total_cost_usd"] = round(spent, 6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for mode, m in result["modes"].items():
        print(f"{mode}: n_in={m['n_in_scope']} n_out={m['n_out_of_scope']} "
              f"FRR={m['false_refusals']}/{m['n_in_scope']} {m['false_refusal_wilson95']} "
              f"FAR={m['false_accepts']}/{m['n_out_of_scope']} {m['false_accept_wilson95']} "
              f"calls={m['llm_calls']} errors={m['errors']} cost=${m['cost_usd']} complete={m['complete']} "
              f"q23={m['trace_q23_layer1']}")
    print(f"total calls={calls} cost=${round(spent, 6)} -> {OUT.relative_to(ROOT)}")
    return 0 if all(m["complete"] for m in result["modes"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
