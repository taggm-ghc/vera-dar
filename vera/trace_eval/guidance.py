"""Decision guidance for the Trace Eval page: paired comparisons, check trust and indicators.

Pure functions over the results/validation JSON plus config/trace_eval_guidance.json; no API, DB or env.
Both variants answer the same questions, so comparisons are paired (exact McNemar on discordant questions),
not an overlap test of two independent intervals.
"""
import json
import math
from pathlib import Path
from statistics import NormalDist

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "trace_eval_guidance.json"

BETTER, WORSE, SAME, SMALL = "▲ better", "▼ worse", "≈ not distinguishable", "· too few to judge"
RELIABLE, OVERFLAGS, MISSES, MIXED = "reliable", "over-flags good answers", "misses failures", "mixed"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())


def z_for(alpha: float) -> float:
    return NormalDist().inv_cdf(1 - alpha / 2)


def wilson(k: int, n: int, z: float) -> tuple[float, float] | None:
    """Wilson score interval for k successes in n trials."""
    if n <= 0:
        return None
    p = k / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denom = 1 + z * z / n
    return max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: min(1, 2 * P[Bin(b+c, 1/2) <= min(b, c)])."""
    m = b + c
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(min(b, c) + 1)) / 2 ** m
    return min(1.0, 2 * tail)


def min_discordant_for_significance(alpha: float) -> int:
    """Smallest m where all m discordant questions going one way gives p < alpha (2 * 0.5^m < alpha)."""
    m = 1
    while 2 * 0.5 ** m >= alpha:
        m += 1
    return m


def _verdict(b: int, c: int, n_pairs: int, cfg: dict) -> tuple[float, str]:
    """b: questions only the candidate passed; c: only the baseline passed."""
    p = mcnemar_exact(b, c)
    if n_pairs < cfg["small_n"]:
        return p, SMALL
    if p < cfg["alpha"]:
        return p, BETTER if b > c else WORSE
    return p, SAME


def _by_question(traces: list[dict], variant: str) -> dict:
    return {t["id"]: t for t in traces if t["variant"] == variant}


def compare_overall(res: dict, cfg: dict) -> list[dict]:
    """Per candidate variant and split: paired all-pass comparison against the baseline (errors count as fails)."""
    traces, base = res.get("traces", []), cfg["baseline_variant"]
    base_q = _by_question(traces, base)
    rows = []
    for v in sorted({t["variant"] for t in traces} - {base}):
        cand_q = _by_question(traces, v)
        for split in sorted({t["split"] for t in base_q.values()}):
            ids = [q for q, t in base_q.items() if t["split"] == split and q in cand_q]
            bp = [bool(base_q[q]["all_pass"]) and not base_q[q].get("error") for q in ids]
            cp = [bool(cand_q[q]["all_pass"]) and not cand_q[q].get("error") for q in ids]
            b = sum(1 for x, y in zip(bp, cp) if y and not x)
            c = sum(1 for x, y in zip(bp, cp) if x and not y)
            p, verdict = _verdict(b, c, len(ids), cfg)
            rows.append({"variant": v, "split": split, "paired questions": len(ids),
                         f"{base} passed": sum(bp), f"{v} passed": sum(cp),
                         "only candidate passed (b)": b, "only baseline passed (c)": c,
                         "McNemar p": round(p, 4), "indicator": verdict})
    return rows


def _check_map(t: dict) -> dict:
    return {c["name"]: c for c in t.get("checks", [])}


def compare_checks(res: dict, cfg: dict) -> list[dict]:
    """Per check: paired comparison on questions where the check applies to both variants."""
    traces, base = res.get("traces", []), cfg["baseline_variant"]
    base_q = _by_question(traces, base)
    rows = []
    for v in sorted({t["variant"] for t in traces} - {base}):
        cand_q = _by_question(traces, v)
        names = sorted({n for t in cand_q.values() for n in _check_map(t)})
        for name in names:
            pairs = []
            for q, ct in cand_q.items():
                bt = base_q.get(q)
                cc, bc = _check_map(ct).get(name), _check_map(bt).get(name) if bt else None
                if cc and bc and cc.get("applies") and bc.get("applies"):
                    pairs.append((bool(bc.get("passed")), bool(cc.get("passed"))))
            b = sum(1 for x, y in pairs if y and not x)
            c = sum(1 for x, y in pairs if x and not y)
            p, verdict = _verdict(b, c, len(pairs), cfg) if pairs else (1.0, SMALL)
            rows.append({"variant": v, "check": name, "paired questions": len(pairs),
                         f"{base} passed": sum(x for x, _ in pairs), f"{v} passed": sum(y for _, y in pairs),
                         "b": b, "c": c, "McNemar p": round(p, 4), "indicator": verdict})
    return rows


def _trust(tpr: float | None, tnr: float | None, cfg: dict) -> str:
    t = cfg["trust"]
    if tpr is None or tpr <= t["unusable_max_tpr"]:
        return MISSES
    if tpr >= t["reliable_min_tpr"] and tnr is not None and tnr >= t["reliable_min_tnr"]:
        return RELIABLE
    if tpr >= t["reliable_min_tpr"]:
        return OVERFLAGS
    return MIXED


def _fmt_ci(ci) -> str:
    return f"{ci[0]:.2f} to {ci[1]:.2f}" if ci else "n/a"


def check_trust(val: dict, cfg: dict) -> list[dict]:
    """Per validated check: TPR = TP/(TP+FN), TNR = TN/(TN+FP) with Wilson intervals, and a trust indicator."""
    z = z_for(cfg["alpha"])
    rows = []
    items = list(val.get("per_check", {}).items())
    s = val.get("support")
    if s and all(k in s for k in ("tp", "tn", "fp", "fn")):
        items.append(("A6_support (heuristic)", s))
    for name, c in items:
        pos, neg = c["tp"] + c["fn"], c["tn"] + c["fp"]
        tpr = c["tp"] / pos if pos else None
        tnr = c["tn"] / neg if neg else None
        rows.append({"check": name, "real failures": pos, "caught (TP)": c["tp"],
                     "TPR": "n/a" if tpr is None else f"{tpr:.0%}", "TPR 95%": _fmt_ci(wilson(c["tp"], pos, z)),
                     "good answers": neg, "left alone (TN)": c["tn"],
                     "TNR": "n/a" if tnr is None else f"{tnr:.0%}", "TNR 95%": _fmt_ci(wilson(c["tn"], neg, z)),
                     "indicator": _trust(tpr, tnr, cfg) + (" (few labels)" if min(pos, neg) < cfg["small_n"] else "")})
    return rows


def supported_statements(overall: list[dict], checks: list[dict], trust: list[dict], cfg: dict) -> list[str]:
    """Plain-language statements derived from the computed tables."""
    out = []
    for r in overall:
        out.append(f"Overall ({r['split']}): {r['variant']} vs {cfg['baseline_variant']} is {r['indicator'].split(' ', 1)[1]}"
                   f" (b={r['only candidate passed (b)']}, c={r['only baseline passed (c)']}, p={r['McNemar p']}).")
    better = sorted({r["check"] for r in checks if r["indicator"] == BETTER})
    worse = sorted({r["check"] for r in checks if r["indicator"] == WORSE})
    small = sorted({r["check"] for r in checks if r["indicator"] == SMALL})
    if better:
        out.append("Real improvements (claimable): " + ", ".join(better) + ".")
    if worse:
        out.append("Real regressions (fix before claiming overall gains): " + ", ".join(worse) + ".")
    if small:
        out.append("Too few paired questions to judge (warning signs only): " + ", ".join(small) + ".")
    weak = [r["check"] for r in trust if r["indicator"].startswith(MISSES)]
    if weak:
        out.append("Checks that miss real failures (their passes do not show quality; replace or add a judge): "
                   + ", ".join(weak) + ".")
    over = [r["check"] for r in trust if r["indicator"].startswith(OVERFLAGS)]
    if over:
        out.append("Checks that over-flag good answers (pass rates may be understated): " + ", ".join(over) + ".")
    return out


def tldr(overall: list[dict], checks: list[dict], trust: list[dict], cfg: dict) -> str:
    """A short top-of-page summary assembled from the computed tables (not a model judgement)."""
    base = cfg["baseline_variant"]
    parts = []
    if overall:
        verdicts = {r["indicator"] for r in overall}
        variant = overall[0]["variant"]
        if verdicts == {SAME} or verdicts <= {SAME, SMALL}:
            parts.append(f"Overall, {variant} is not distinguishable from {base} on this question set "
                         f"(paired McNemar, p >= {cfg['alpha']}).")
        else:
            parts.append("Overall: " + "; ".join(f"{r['split']} {r['indicator'].split(' ', 1)[1]} (p={r['McNemar p']})"
                                                  for r in overall) + ".")
    better = [r for r in checks if r["indicator"] == BETTER]
    worse = [r for r in checks if r["indicator"] == WORSE]
    if better:
        parts.append("Real improvements: " + ", ".join(f"{r['check']} (p={r['McNemar p']})" for r in better) + ".")
    if worse:
        parts.append("Real regressions: " + ", ".join(f"{r['check']} (p={r['McNemar p']})" for r in worse) + ".")
    small = [r["check"] for r in checks if r["indicator"] == SMALL]
    if small:
        parts.append(f"Too few questions to judge: {len(small)} checks.")
    weak = [r["check"] for r in trust if r["indicator"].startswith(MISSES)]
    if weak:
        parts.append("Do not trust: " + ", ".join(weak) + " (misses real failures).")
    return " ".join(parts) or "Not enough data for a summary."
