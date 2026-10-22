"""Streamlit renderer for the TRACE results (item #70; decision guidance added 2026-10-07). Reads only the committed JSON files; no API, DB or env."""
import copy
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_PATH = "eval_results/trace_eval_v1.json"
VALIDATION_PATH = "eval_results/trace_eval_v1_validation.json"
# Why: a capture run writes its results to a work/ path; these optional env vars (paths, not secrets) point the page at
# it without editing code. Unset means the default paths above.
RESULTS_ENV, VALIDATION_ENV = "VERA_TRACE_EVAL_RESULTS", "VERA_TRACE_EVAL_VALIDATION"
ARXIV_PREFIX = "https://arxiv.org/"  # Why: the only links the page ever shows, and only if the file supplies them.

# K3.P2: fixed public texts. Nothing from the data (error text, labeller, paths, exceptions) is ever formatted into them.
CAPTURE_ERROR_TEXT = ("Capture error: this answer could not be captured, so it counts as a failure. "
                      "Details are kept off the public page.")
MISSING_RESULTS_TEXT = "No trace evaluation results yet."
# Fixed how-to for the empty state (no paths, hosts or script names on the public page).
HOW_TO_PRODUCE_TEXT = (
    "**How to produce them.** An authorized operator (1) captures answers to the frozen evaluation questions from "
    "VERA-DAR's answer route, (2) scores every answer with the rule-based checks, and (3) points this page at the "
    "results file that step writes. Capturing makes paid or rate-limited model calls, so it runs only with the "
    "owner's authorization; scoring is offline. The repository's trace-capture script documents the exact commands, "
    "including a dry run that makes no calls.")
VALIDATION_VERSION_TEXT = ("Check trust below was measured on an earlier version of the checks and on the "
                           "frozen question set's historical answers; treat it as indicative for this run.")
NO_BASELINE_TEXT = ("This run has a single variant, so no paired comparison is computed; the tables below report its "
                    "pass rates per check, per dimension and per question category.")
GUIDANCE_UNAVAILABLE_TEXT = "Decision guidance is unavailable right now."
RENDER_ERROR_TEXT = "This section could not be shown."
LABELLER_TEXT = "an independent AI agent, blind to the automated verdicts"

# Public display names (OD-SIDEBAR-2, deny by default): raw variant and check identifiers never render.
# Names not in these maps become "Variant N" / "Check N" (sorted order), so a new raw name cannot reach the page.
VARIANT_NAMES = {"baseline": "Baseline", "grounded_v1": "Grounded", "ask_api": "Live answers"}
CHECK_NAMES = {"A1_citation_present": "Citation present", "A2_citations_resolve": "Citations resolve",
               "A3_numbers_grounded": "Numbers grounded", "A4_phantom_evidence": "No phantom evidence",
               "A5_window": "Time window", "A6_support": "Support", "A7_premise_corrected": "False premise corrected",
               "A8_scope_decline": "Out-of-scope declined", "A9_clarify_or_assume": "Clarifies or states assumptions"}
HEURISTIC_RAW, HEURISTIC_SHOWN = "A6_support (heuristic)", "Support (heuristic)"


def _namer(known: dict, prefix: str, raw) -> dict:
    unknown = sorted({str(n) for n in raw if n is not None and str(n) not in known})
    return {**known, **{n: f"{prefix} {i}" for i, n in enumerate(unknown, 1)}}


def _public_labels(res: dict, val: dict | None, cfg: dict | None):
    """Relabelled deep copies of results, validation and guidance config; the inputs are never modified."""
    res, val, cfg = copy.deepcopy(res), copy.deepcopy(val), copy.deepcopy(cfg)
    traces = res.get("traces", [])
    vm = _namer(VARIANT_NAMES, "Variant",
                list(res.get("variants", {})) + [t.get("variant") for t in traces]
                + ([cfg.get("baseline_variant")] if cfg else []))
    raw_checks = [n for d in res.get("variants", {}).values() for n in d.get("per_check", {})]
    raw_checks += [c.get("name") for t in traces for c in t.get("checks", [])]
    raw_checks += list((val or {}).get("per_check", {}))
    cm = _namer(CHECK_NAMES, "Check", raw_checks)
    res["variants"] = {vm[str(k)]: {**d, "per_check": {cm[str(n)]: c for n, c in d.get("per_check", {}).items()}}
                       for k, d in res.get("variants", {}).items()}
    for t in traces:
        if "variant" in t:
            t["variant"] = vm[str(t["variant"])]
        for c in t.get("checks", []):
            if "name" in c:
                c["name"] = cm[str(c["name"])]
    if val:
        val["per_check"] = {cm[str(n)]: c for n, c in val.get("per_check", {}).items()}
    for d in res["variants"].values():
        dims = {}
        for dim, c in (d.get("per_dimension") or {}).items():
            shown = cm.get(str(dim), str(dim)) if str(dim) in cm else str(dim)
            dims[shown] = {**c, "checks": [cm.get(str(n), str(n)) for n in c.get("checks", [])]}
        d["per_dimension"] = dims
    if cfg:
        descs = cfg.get("variant_descriptions") or {}
        cfg["variant_descriptions"] = {vm[str(k)]: v for k, v in descs.items() if str(k) in vm}
    if cfg and cfg.get("baseline_variant") is not None:
        cfg["baseline_variant"] = vm[str(cfg["baseline_variant"])]
    return res, val, cfg


def _trust_rows(val: dict | None, cfg: dict) -> list:
    """g.check_trust rows with the check column relabelled (the guidance module adds its own heuristic row name)."""
    from vera.trace_eval import guidance as g
    if not val:
        return []
    rows = g.check_trust(val, cfg)
    for row in rows:
        if row.get("check") == HEURISTIC_RAW:
            row["check"] = HEURISTIC_SHOWN
    return rows


def _load(path) -> dict | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _ci(w) -> str:
    return f"{w[0]:.2f} to {w[1]:.2f}" if w else "n/a"


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _headline(res: dict) -> pd.DataFrame:
    rows = []
    for variant, d in res.get("variants", {}).items():
        for split, s in d.get("by_split", {}).items():
            rows.append({"variant": variant, "split": str(split), "n": s["n"], "errors": s["errors"],
                         "all-pass": s["all_pass"], "rate": _pct(s["all_pass_rate"]),
                         "Wilson 95%": _ci(s.get("wilson95"))})
    return pd.DataFrame(rows)


def _per_check(res: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = list(res.get("variants", {}))
    names = sorted({n for v in variants for n in res["variants"][v].get("per_check", {})})
    table, chart = [], []
    for n in names:
        row, crow = {"check": n}, {"check": n}
        for v in variants:
            c = res["variants"][v].get("per_check", {}).get(n)
            row[f"{v} passed/applies"] = f"{c['passed']}/{c['applies']}" if c else "n/a"
            row[f"{v} rate"] = _pct(c["rate"]) if c else "n/a"
            row[f"{v} Wilson 95%"] = _ci(c.get("wilson95")) if c else "n/a"
            crow[v] = float(c["rate"]) if c and c.get("rate") is not None else 0.0
        table.append(row)
        chart.append(crow)
    return pd.DataFrame(table), pd.DataFrame(chart).set_index("check") if chart else pd.DataFrame()


def _validation(val: dict) -> None:
    st.markdown("#### Check validation (dev labels)")
    st.caption(f"Positive class: {val.get('positive_class')}; labeller: {LABELLER_TEXT}.")
    rows = [{"check": n, "n": c["n"], "TP": c["tp"], "TN": c["tn"], "FP": c["fp"], "FN": c["fn"],
             "TPR": _pct(c.get("tpr")), "TNR": _pct(c.get("tnr")), "note": c.get("note", "")}
            for n, c in val.get("per_check", {}).items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    s = val.get("support")
    if s:
        st.markdown(f"**Support heuristic**: n={s['n']}, supported={s['supported']}, "
                    f"partially={s['partially']}, unsupported={s['unsupported']}, "
                    f"TPR={_pct(s.get('heuristic_tpr'))}, TNR={_pct(s.get('heuristic_tnr'))}. {s.get('note', '')}")


def _cfg():
    from vera.trace_eval import guidance as g
    try:
        return g.load_config()
    except (OSError, ValueError):
        return None


def _context(res: dict, cfg: dict | None = None) -> None:
    """1. Context: a short setup only (what was evaluated, which system, when); detail lives in Elaboration."""
    captured = str(res.get("generated_at") or "")[:10] or "unknown"
    n_q = len({t.get("id") for t in res.get("traces", []) if t.get("id")})
    descs = (cfg or {}).get("variant_descriptions") or {}
    parts = [f"{v} ({descs[v]})" if v in descs else v for v in res.get("variants", {})]
    run = res.get("run") or {}
    kind = "frozen agent-authored" if run.get("checks_validated", True) else "un-frozen, adapted"
    st.markdown(f"**Context.** {n_q} {kind} questions, each answered by: {'; '.join(parts)}. "
                "Rule-based checks score every answer; the system may have changed since capture.")
    st.caption(f"Snapshot captured {captured} (UTC); re-running the evaluation replaces these results and this date.")


def _run_caveats(res: dict) -> list[str]:
    """Caveats computed from the results file's run facts (absent facts = no caveat; never a fixed conclusion)."""
    run, out = res.get("run") or {}, []
    if run.get("checks_validated") is False:
        out.append("The checks were validated only on the frozen question set, not on this one.")
    if run.get("source_text_unavailable"):
        out.append(f"{run['source_text_unavailable']} answers came without source text, so the number and support "
                   "checks did not apply to them.")
    errors = run.get("capture_errors")
    if errors is None:  # older results files carry no run facts; count from the traces themselves
        errors = sum(1 for tr in res.get("traces", []) if tr.get("error"))  # truth test only
    if errors:
        out.append(f"{errors} answers could not be captured; they count as failures.")
    return out


def _tldr(res: dict, overall, checks, trust, cfg: dict) -> None:
    """2. TL;DR: computed findings with their uncertainty and the main caveat beside them (not later)."""
    from vera.trace_eval import guidance as g
    text = g.tldr(overall, checks, trust, cfg)
    n_q = len({t.get("id") for t in res.get("traces", []) if t.get("id")})
    st.info(f"**TL;DR (computed from the results below)**\n\n{text}\n\n**Main caveat:** {n_q} agent-authored "
            "questions (small n, self-preference risk); checks are heuristics. " + " ".join(_run_caveats(res)))
    st.caption("Assembled from the paired tests and check validation on this page; not a new model judgement.")


def _determinations(overall, checks, trust) -> None:
    """3. Computed determinations: the paired tests and check trust, each row with its indicator."""
    st.subheader("Computed determinations")
    st.markdown("**Is a variant better overall?** (paired exact McNemar test, every question)")
    st.dataframe(pd.DataFrame(overall), hide_index=True)
    st.markdown("**Which checks changed?** (paired, questions where the check applies to both variants)")
    st.dataframe(pd.DataFrame(checks), hide_index=True)
    if trust:
        st.markdown("**How far can each check be trusted?** (against blind labels; positive class = FAIL)")
        st.dataframe(pd.DataFrame(trust), hide_index=True)


def _decisioning(overall, checks, trust, cfg) -> None:
    """4. Decisioning: recommendations, each tied to a determination above (findings and recommendations kept apart)."""
    from vera.trace_eval import guidance as g
    st.subheader("Decisioning")
    st.caption("Recommendations derived from the determinations above with the page's configured thresholds; "
               "each names the row it rests on. Judgement, not a further finding.")
    for line in g.supported_statements(overall, checks, trust, cfg):
        st.markdown(f"- {line}")
    st.markdown(
        "**How to act on it**\n"
        "- **Claim** only ▲ rows, and say the overall result is ≈ unless the overall table shows ▲.\n"
        "- **Fix first**: ▼ rows, then checks marked *misses failures* (their passes say nothing about quality).\n"
        "- **Treat · rows as warning signs**: add questions in those categories before deciding.\n"
        "- **Discount** pass rates on checks that *over-flag*: they understate quality.\n"
        "- **Tune on dev only**; read held-out once at the end, or it stops being a fair test.\n"
        "- **Re-run after each change** and compare the paired tables, not just the headline.")


def _elaboration(cfg, val: dict | None, res: dict | None = None) -> None:
    """5. Elaboration: the math, check validation detail and the full caveats."""
    from vera.trace_eval import guidance as g
    st.subheader("Elaboration")
    alpha, m = cfg["alpha"], g.min_discordant_for_significance(cfg["alpha"])
    t = cfg["trust"]
    with st.expander("The math behind the indicators"):
        st.markdown("**Pass rate and its 95% range (Wilson score interval)**, for k passes in n questions, "
                    f"with z = Φ⁻¹(1 − α/2) = {g.z_for(alpha):.2f} at α = {alpha}:")
        st.latex(r"\hat p=\frac{k}{n},\qquad \frac{\hat p+\frac{z^2}{2n}\pm z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}}")
        st.markdown("**Paired comparison (exact McNemar test).** Both variants answer the same questions, so only "
                    "questions where they disagree carry information: b = only the candidate passed, "
                    "c = only the baseline passed.")
        st.latex(r"p=\min\Bigl(1,\;2\sum_{i=0}^{\min(b,c)}\binom{b+c}{i}\Bigl(\tfrac12\Bigr)^{b+c}\Bigr)")
        st.markdown(f"▲/▼ when p < α = {alpha}; ≈ otherwise; · when fewer than {cfg['small_n']} paired questions. "
                    f"Even if every disagreement goes one way, at least {m} disagreeing questions are needed "
                    f"for p < {alpha} (2·½^m < α).")
        st.markdown("**Check trust** (positive class = FAIL): TPR = TP / (TP + FN), the share of real failures the "
                    "check catches; TNR = TN / (TN + FP), the share of good answers it leaves alone; each with a "
                    f"Wilson range. Reliable: TPR ≥ {t['reliable_min_tpr']:.0%} and TNR ≥ {t['reliable_min_tnr']:.0%}. "
                    f"Over-flags: TPR ≥ {t['reliable_min_tpr']:.0%}, TNR lower. Misses failures: TPR ≤ "
                    f"{t['unusable_max_tpr']:.0%}. \"few labels\" when fewer than {cfg['small_n']} real failures or "
                    "good answers were labelled.")
    if val is None:
        st.info("Check validation not yet available")
    else:
        _validation(val)
    st.warning(" ".join(["Caveats: questions are agent-authored (self-preference risk); a single agent coded the checks "
                         "and labelled the validation set, both from the same model family; n is small; checks are "
                         "heuristics validated on dev only."] + _run_caveats(res or {})))


def _data(res: dict) -> None:
    """6. Data: raw summary tables, chart and the trace browser."""
    st.subheader("Data")
    st.markdown("**Headline: share of answers that passed every check**")
    st.caption("Directional, small n; capture errors count as failures.")
    st.dataframe(_headline(res), hide_index=True)
    st.markdown("**Per check**")
    table, chart = _per_check(res)
    st.dataframe(table, hide_index=True)
    if not chart.empty:
        st.bar_chart(chart)
    st.markdown("**Per dimension** (checks pooled by what they measure)")
    st.dataframe(_per_group(res, "per_dimension", "dimension"), hide_index=True)
    st.markdown("**Per question category** (share of answers that passed every applicable check)")
    st.dataframe(_per_group(res, "per_category", "category"), hide_index=True)
    _browser(res)


def _per_group(res: dict, key: str, label: str) -> pd.DataFrame:
    rows = []
    for v, d in res.get("variants", {}).items():
        for name, c in (d.get(key) or {}).items():
            den = c.get("applies", c.get("n"))
            num = c.get("passed", c.get("all_pass"))
            row = {"variant": v, label: str(name), "passed": f"{num}/{den}", "rate": _pct(c.get("rate")),
                   "Wilson 95%": _ci(c.get("wilson95"))}
            if "checks" in c:
                row["checks"] = ", ".join(c["checks"])
            rows.append(row)
    return pd.DataFrame(rows)


def _browser(res: dict) -> None:
    st.markdown("#### Trace browser")
    traces = res.get("traces", [])
    variants = sorted({t["variant"] for t in traces})
    if not variants:
        st.info("No traces in the results file.")
        return
    variant = st.selectbox("Variant", variants)
    ids = [t["id"] for t in traces if t["variant"] == variant]
    qid = st.selectbox("Question id", ids)
    t = next(t for t in traces if t["variant"] == variant and t["id"] == qid)
    st.markdown(f"Category: `{t['category']}`; split: `{t['split']}`; all-pass: `{t['all_pass']}`")
    if t.get("error"):
        st.error(CAPTURE_ERROR_TEXT)  # the error value is only tested for truth, never shown, sliced or logged
    st.dataframe(pd.DataFrame([{"check": c["name"], "applies": c["applies"],
                                "passed": "n/a" if c["passed"] is None else str(c["passed"]),
                                "reason": c["reason"]} for c in t["checks"]]), hide_index=True)
    st.markdown("**Answer excerpt (first 25 words)**")
    st.write(t.get("answer_excerpt") or "(none)")
    st.markdown("**Sources**")
    srcs = t.get("sources") or []
    if not srcs:
        st.write("(none)")
    for s in srcs:
        title, url = s.get("title") or "untitled", s.get("url") or ""
        st.markdown(f"- [{title}]({url})" if url.startswith(ARXIV_PREFIX) else f"- {title}")


def _resolve_paths(results_path, validation_path):
    return (results_path or os.environ.get(RESULTS_ENV) or RESULTS_PATH,
            validation_path or os.environ.get(VALIDATION_ENV) or VALIDATION_PATH)


def render_trace_eval(results_path=None, validation_path=None) -> None:
    """Order (literature-checked): context, TL;DR with caveat, computed determinations, decisioning,
    elaboration, data. Any exception in the body shows one fixed line (no type, text or traceback)."""
    st.title("VERA-DAR Trace Eval")
    try:
        _render_body(*_resolve_paths(results_path, validation_path))
    except Exception:  # noqa: BLE001  (Streamlit's rerun/stop signals derive from BaseException and pass through)
        st.error(RENDER_ERROR_TEXT)


def _render_body(results_path, validation_path) -> None:
    from vera.trace_eval import guidance as g
    res = _load(results_path)
    if res is None or not res.get("traces"):
        st.warning(MISSING_RESULTS_TEXT)
        st.info(HOW_TO_PRODUCE_TEXT)
        return
    val, cfg = _load(validation_path), _cfg()
    res, val, cfg = _public_labels(res, val, cfg)
    _context(res, cfg)
    if cfg is None:
        st.info(GUIDANCE_UNAVAILABLE_TEXT)
        _data(res)
        return
    overall, checks = g.compare_overall(res, cfg), g.compare_checks(res, cfg)
    trust = _trust_rows(val, cfg)
    if val and val.get("checks_version") != res.get("checks_version"):
        st.caption(VALIDATION_VERSION_TEXT)
    if cfg.get("baseline_variant") not in res.get("variants", {}):
        st.info(NO_BASELINE_TEXT)
    _tldr(res, overall, checks, trust, cfg)
    _determinations(overall, checks, trust)
    _decisioning(overall, checks, trust, cfg)
    _elaboration(cfg, val, res)
    _data(res)
