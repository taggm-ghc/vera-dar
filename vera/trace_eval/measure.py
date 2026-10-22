"""Run the v1 trace checks over captured JSONL traces and build the committed results document.

Why: the results file is committed, so it carries per-trace check outcomes, a 25-word answer excerpt and
source ids/titles only; full answers and abstracts stay in the untracked raw traces.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from vera.trace_eval.checks import CHECKS_VERSION, run_checks, trace_outcome

SCHEMA = "vera-trace-eval-results-v1"

# Why: 95% interval, the level the plan (P2-R2) reports for every rate.
Z95 = 1.959963984540054

# Why: bounds how much answer text can reach the committed file.
EXCERPT_WORDS = 25


def wilson95(k: int, n: int) -> list[float] | None:
    """Wilson score interval for k successes of n at 95%; None when n == 0."""
    if n <= 0:
        return None
    p, z2 = k / n, Z95 ** 2
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _rate(k: int, n: int) -> dict:
    return {"rate": round(k / n, 4) if n else None, "wilson95": wilson95(k, n)}


def load_traces(paths: list[str | Path]) -> list[dict]:
    recs: list[dict] = []
    for p in paths:
        with Path(p).open(encoding="utf-8") as fh:
            recs += [json.loads(line) for line in fh if line.strip()]
    return recs


def trace_row(rec: dict) -> dict:
    results = run_checks(rec)
    outcome = trace_outcome(rec, results)
    return {
        "id": rec.get("id"), "split": rec.get("split"), "category": rec.get("category"),
        "variant": rec.get("variant"),
        "all_pass": None if outcome == "error" else outcome == "pass",
        "error": rec.get("error"),
        "checks": [{"name": r.name, "applies": r.applies, "passed": r.passed, "reason": r.reason} for r in results],
        "answer_excerpt": " ".join((rec.get("answer") or "").split()[:EXCERPT_WORDS]),
        "sources": [{"n": s.get("n"), "url": s.get("url"), "title": s.get("title")} for s in rec.get("sources") or []],
    }


def _per_dimension(per_check: dict, dimensions: dict | None) -> dict:
    """Pooled applies/passed per configured group of checks; a check in no group is its own dimension."""
    groups = dict(dimensions or {})
    grouped = {c for names in groups.values() for c in names}
    groups.update({n: [n] for n in per_check if n not in grouped})
    out = {}
    for dim, names in groups.items():
        hit = [per_check[n] for n in names if n in per_check]
        if not hit:
            continue
        k, n = sum(c["passed"] for c in hit), sum(c["applies"] for c in hit)
        out[dim] = {"checks": [x for x in names if x in per_check], "applies": n, "passed": k, **_rate(k, n)}
    return out


def _per_category(vrows: list[dict]) -> dict:
    out = {}
    for cat in sorted({str(r["category"]) for r in vrows}):
        crows = [r for r in vrows if str(r["category"]) == cat]
        k = sum(1 for r in crows if r["all_pass"])
        out[cat] = {"n": len(crows), "errors": sum(1 for r in crows if r["all_pass"] is None), "all_pass": k,
                    **_rate(k, len(crows))}
    return out


def summarize(rows: list[dict], dimensions: dict | None = None) -> dict:
    """Per-variant aggregates from trace rows."""
    out: dict = {}
    for variant in sorted({r["variant"] for r in rows}):
        vrows = [r for r in rows if r["variant"] == variant]
        by_split = {}
        for split in sorted({r["split"] for r in vrows}, key=str):
            srows = [r for r in vrows if r["split"] == split]
            errors = sum(1 for r in srows if r["all_pass"] is None)
            ok = [r for r in srows if r["all_pass"] is not None]
            k = sum(1 for r in ok if r["all_pass"])
            # Why: a capture error is a failed user request, not a missing data point; dropping it from the
            # denominator would flatter the variant that errs more (grounded search failures). The rate over
            # non-error traces is kept alongside, labelled, for diagnosis only.
            by_split[split] = {"n": len(srows), "errors": errors, "all_pass": k,
                               "all_pass_rate": _rate(k, len(srows))["rate"], "wilson95": wilson95(k, len(srows)),
                               "all_pass_rate_excluding_errors": _rate(k, len(ok))["rate"]}
        per_check: dict = {}
        for r in vrows:
            for c in r["checks"]:
                if c["applies"]:
                    d = per_check.setdefault(c["name"], {"applies": 0, "passed": 0})
                    d["applies"] += 1
                    d["passed"] += 1 if c["passed"] else 0
        for d in per_check.values():
            d.update(_rate(d["passed"], d["applies"]))
        out[variant] = {"by_split": by_split, "per_check": per_check,
                        "per_dimension": _per_dimension(per_check, dimensions), "per_category": _per_category(vrows)}
    return out


def _run_meta(records: list[dict]) -> dict:
    """Facts about the capture the page states as caveats (computed, never fixed text)."""
    case_sets = sorted({str(r.get("case_set") or "frozen") for r in records})
    return {"case_sets": case_sets,
            "checks_validated": all(r.get("checks_validated", True) for r in records),
            "capture_errors": sum(1 for r in records if r.get("error")),
            "source_text_unavailable": sum(1 for r in records if r.get("source_text_available") is False),
            "traces": len(records)}


def build_results(records: list[dict], *, questions_sha256: str | None = None, dimensions: dict | None = None) -> dict:
    rows = [trace_row(r) for r in records]
    shas = {r.get("questions_sha256") for r in records if r.get("questions_sha256")}
    return {
        "schema": SCHEMA, "checks_version": CHECKS_VERSION,
        "questions_sha256": questions_sha256 or (sorted(shas)[0] if len(shas) == 1 else sorted(shas) or None),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": _run_meta(records), "variants": summarize(rows, dimensions), "traces": rows,
    }


def configured_dimensions() -> dict | None:
    """The check groups from config/trace_eval_guidance.json; None (each check its own group) if unreadable."""
    from vera.trace_eval import guidance
    try:
        return guidance.load_config().get("dimensions")
    except (OSError, ValueError):
        return None


def measure(trace_paths: list[str | Path], out_path: str | Path, dimensions: dict | None = None) -> dict:
    dimensions = configured_dimensions() if dimensions is None else dimensions
    results = build_results(load_traces(trace_paths), dimensions=dimensions)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return results


def aggregate_lines(results: dict) -> list[str]:
    """Aggregate-only text for the CLI (no answers, no ids)."""
    lines = [f"checks_version={results['checks_version']} traces={len(results['traces'])}"]
    for v, d in results["variants"].items():
        for split, s in d["by_split"].items():
            lines.append(f"{v} {split}: n={s['n']} errors={s['errors']} all_pass={s['all_pass']} "
                         f"rate={s['all_pass_rate']} wilson95={s['wilson95']}")
        for name, c in d["per_check"].items():
            lines.append(f"  {name}: {c['passed']}/{c['applies']} wilson95={c['wilson95']}")
        for name, c in d.get("per_dimension", {}).items():
            lines.append(f"  dimension {name}: {c['passed']}/{c['applies']} wilson95={c['wilson95']}")
    return lines
