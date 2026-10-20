"""Validate the deterministic TRACE checks against independent labels (p3m3 item #70, Week 4 "validate it").

Week 4's rule for any automated judge is to measure TPR and TNR against labels before trusting its metric; it
applies to deterministic checks too. Inputs: the committed results file (automated verdicts), the blind
per-criterion labels (docs/trace_eval/check_labels_v1.json) and the blind support labels
(docs/trace_eval/support_labels_v1.json). Output: eval_results/trace_eval_v1_validation.json.

Positive class = FAIL (a failure is present). TPR = share of labelled failures the check also fails;
TNR = share of labelled passes the check also passes. Dev split only (the labels cover dev only).

    venv/bin/python scripts/trace_validate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vera.trace_eval import checks  # noqa: E402

# Why: label criteria C1-C9 were written as the meaning-level versions of checks A1-A9 (same applicability).
CRITERION_TO_CHECK = {
    "C1": "A1_citation_present", "C2": "A2_citations_resolve", "C3": "A3_numbers_grounded",
    "C4": "A4_phantom_evidence", "C5": "A5_window", "C7": "A7_premise_corrected",
    "C8": "A8_scope_decline", "C9": "A9_clarify_or_assume",
}


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def confusion(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs = (label_fail, check_fail)."""
    tp = sum(1 for lf, cf in pairs if lf and cf)
    fn = sum(1 for lf, cf in pairs if lf and not cf)
    tn = sum(1 for lf, cf in pairs if not lf and not cf)
    fp = sum(1 for lf, cf in pairs if not lf and cf)
    return {"n": len(pairs), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "tpr": _rate(tp, tp + fn), "tnr": _rate(tn, tn + fp)}


def validate(results: dict, labels: dict, support_labels: dict, grounded_dev: list[dict]) -> dict:
    auto = {(t["id"], t["variant"]): {c["name"]: c for c in t["checks"]} for t in results["traces"]
            if t["split"] == "dev"}
    per: dict[str, list[tuple[bool, bool]]] = {}
    skipped = 0
    for item in labels["items"]:
        a = auto.get((item["trace"], item["variant"]))
        labs = item.get("labels")
        if not isinstance(labs, dict):  # e.g. "error": a capture-error trace has no labels
            skipped += 1
            continue
        for crit, lab in labs.items():
            name = CRITERION_TO_CHECK.get(crit)
            c = (a or {}).get(name or "")
            if not c or not c["applies"] or lab.get("label") not in ("PASS", "FAIL"):
                skipped += 1  # applicability mismatch or error trace: reported, never guessed
                continue
            per.setdefault(name, []).append((lab["label"] == "FAIL", not c["passed"]))
    per_check = {}
    for name, pairs in sorted(per.items()):
        d = confusion(pairs)
        d["note"] = "" if d["tpr"] is not None and d["tnr"] is not None else \
            "one class absent in labels: rate undefined"
        per_check[name] = d

    by_trace = {r["id"]: r for r in grounded_dev if not r.get("error")}
    heur = {}
    for r in by_trace.values():
        for i, j in enumerate(checks.support_judgements(r)):
            heur[(r["id"], i)] = j["passed"]
    sp, counts = [], {"supported": 0, "partially": 0, "unsupported": 0}
    for it in support_labels["items"]:
        counts[it["label"]] = counts.get(it["label"], 0) + 1
        h = heur.get((it["trace"], it["idx"]))
        if h is not None:
            sp.append((it["label"] != "supported", not h))  # partially counts as a failure (strict)
    sc = confusion(sp)
    return {
        "schema": "vera-trace-check-validation-v1", "positive_class": "FAIL",
        "checks_version": results.get("checks_version"),
        "labeller": labels.get("labeller", "") + "; support: " + support_labels.get("labeller", ""),
        "split": "dev", "skipped_label_cells": skipped,
        "per_check": per_check,
        "support": {"n": sc["n"], **counts, "heuristic_tpr": sc["tpr"], "heuristic_tnr": sc["tnr"],
                    "tp": sc["tp"], "tn": sc["tn"], "fp": sc["fp"], "fn": sc["fn"],
                    "note": "partially counted as unsupported (strict)"},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="eval_results/trace_eval_v1.json")
    ap.add_argument("--labels", default="docs/trace_eval/check_labels_v1.json")
    ap.add_argument("--support-labels", default="docs/trace_eval/support_labels_v1.json")
    ap.add_argument("--grounded-dev", default="tmp/trace_eval/grounded_dev.jsonl")
    ap.add_argument("--out", default="eval_results/trace_eval_v1_validation.json")
    a = ap.parse_args()
    rd = lambda p: json.loads((ROOT / p).read_text())  # noqa: E731
    gd = [json.loads(line) for line in (ROOT / a.grounded_dev).read_text().splitlines() if line.strip()]
    out = validate(rd(a.results), rd(a.labels), rd(a.support_labels), gd)
    (ROOT / a.out).write_text(json.dumps(out, indent=2) + "\n")
    for name, d in out["per_check"].items():
        print(f"{name}: n={d['n']} tp={d['tp']} fn={d['fn']} tn={d['tn']} fp={d['fp']} tpr={d['tpr']} tnr={d['tnr']}")
    s = out["support"]
    print(f"support heuristic: n={s['n']} tpr={s['heuristic_tpr']} tnr={s['heuristic_tnr']} "
          f"labels supported={s['supported']} partially={s['partially']} unsupported={s['unsupported']}")
    print(f"skipped label cells: {out['skipped_label_cells']}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
