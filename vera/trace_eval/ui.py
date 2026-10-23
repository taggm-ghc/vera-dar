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
    st.markdown(f"#### {TNR_SECTION}: dev labels")
    st.caption(TNR_NOTE)
    st.caption(f"Positive class: {val.get('positive_class')}; labeller: {LABELLER_TEXT}.")
    rows = [{"check": n, "n": c["n"], "TP": c["tp"], "TN": c["tn"], "FP": c["fp"], "FN": c["fn"],
             "TPR": _pct(c.get("tpr")), "TNR": _tnr_cell(c.get("tn"), c.get("fp"), c.get("tnr")), "note": c.get("note", "")}
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
        st.markdown(f"**{TNR_SECTION}** (blind labels; positive class = FAIL). {TNR_NOTE}")
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
    # TRACEVIZ-1022: the per-trace browser (answer excerpts, source titles) is the private tier; see _render_private.


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


def _resolve_paths(results_path, validation_path):
    return (results_path or os.environ.get(RESULTS_ENV) or RESULTS_PATH,
            validation_path or os.environ.get(VALIDATION_ENV) or VALIDATION_PATH)


PRIVATE_DENIED_TEXT = "Detail is not available with that key."
PRIVATE_OFF_TEXT = "Trace detail could not be loaded right now."
PRIVATE_EMPTY_TEXT = "The key was accepted, but no per-trace detail exists on the server yet."


HISTORICAL_TEXT = ("HISTORICAL: this run was not made on the build deployed now (or its build is unknown), so it describes "
                   "an earlier state of the system and may not reflect current behaviour.")
CHECK_ACCURACY_TEXT = ("Check accuracy, not answer pass rates. This table measures how well each automated check agrees with "
                       "human labels (TPR = failures the check catches, TNR = passes it correctly leaves alone). It says nothing "
                       "about how often VERA-DAR's answers pass.")


METHOD_TEXT = ("**Method.** Each question is sent to VERA-DAR's answer route; every answer is scored by nine rule-based checks "
               "(citation present, citations resolve, numbers grounded, no phantom evidence, date window, claim support, false "
               "premise corrected, out-of-scope declined, clarification or stated assumption). A case passes only if every "
               "check that applies passes; a capture error counts as a failure. Rates carry 95% Wilson intervals. "
               "Per-case detail is private.")


def historical_label(run: dict) -> str:
    m = run.get("run_meta") or {}
    models = ", ".join(m.get("models") or []) or "model unknown"
    if m.get("mode") == "historical":
        return (f"HISTORICAL (TRACE v1, {m.get('finished_at') or 'date unknown'}, earlier abstract-grounded variant, "
                "build not recorded, superseded)")
    return f"HISTORICAL ({m.get('finished_at') or run.get('generated_at') or 'date unknown'}, {models})"


def _limitations(run: dict) -> list[str]:
    """Fixed limitation lines chosen from the run's recorded flags (no free text from the data)."""
    m = run.get("run_meta") or {}
    out = ["Rule-based checks A1-A9 are heuristics; a pass is not proof an answer is correct."]
    if m.get("offline"):
        out.append("Offline fixture run: synthetic answers, no model or search was called.")
    if m.get("checks_validated") is False:
        out.append("The question set is not the frozen set the checks were validated on.")
    if (run.get("n_errors") or 0) > 0:
        out.append("Capture errors count as failures in the pass rates.")
    if not m:
        out.append("No run metadata was recorded for this file, so its build and scoring version are unknown.")
    out.append("Rates come from a small question set; read the 95% intervals, not only the point values.")
    return out


TNR_SECTION = "Checker agreement with human labels (true-negative rate)"
TNR_NOTE = ("These figures come from few labelled examples and are not answer pass rates. n/N = good answers the check "
            "correctly left alone out of all good answers labelled by humans.")


def _tnr_cell(tn, fp, rate) -> str:
    """Counts first (TN/(TN+FP)); a bare percentage is never shown."""
    if tn is None or fp is None:
        return "n/a"
    n = int(tn + fp)
    return f"{int(tn)}/{n}" + (f" ({rate:.0%})" if rate is not None and n else "") if n else "n/a"


def _pv(x, fmt="{:.0%}") -> str:
    return "n/a" if x is None else fmt.format(x)


def _render_public(view: dict) -> None:
    """TRACEVIZ-1022 public tier: aggregates and structure from vera.trace_viz.public_view. No key, never blank."""
    st.caption(f"Build {view.get('build', 'unknown')}. This section shows aggregates only: no individual questions, "
               "answers or sources. Detail needs a viz key (bottom of the page).")
    st.subheader("Evaluation flow")
    st.markdown(" then ".join(f"{r['stage']}. {r['check']}" for r in view.get("stages", [])) or "(unavailable)")
    run = view.get("run")
    st.subheader("Latest run")
    if not run:
        st.info("No run aggregates are available on this deployment. The method, limitations and check accuracy below "
                "do not depend on a run.")
    else:
        meta = run.get("run_meta") or {}
        if view.get("historical"):
            st.warning(((historical_label(run) + ". ") if (meta.get("mode") == "historical") else "") + HISTORICAL_TEXT)
        same = "" if view.get("historical") or view.get("run_build") == view.get("build") else "; application code unchanged since the run build"
        st.caption(f"Run build {view.get('run_build') or 'unknown'} (deployed build {view.get('build', 'unknown')}{same}); "
                   f"run date {view.get('run_file_date') or 'unknown'} (UTC); scoring {meta.get('scoring_version') or 'unknown'}; "
                   f"eval set {(meta.get('eval_set_sha256') or 'unknown')[:12]}.")
        st.markdown(f"**{run.get('n_questions')}** questions, **{run.get('n_traces')}** answers, "
                    f"**{run.get('n_errors')}** capture errors.")
        rows = [{"variant": v, "check": c, "passed/applies": f"{_pv(x.get('passed'), '{:g}')}/{_pv(x.get('applies'), '{:g}')}",
                 "rate": _pv(x.get("rate"))}
                for v, d in (run.get("variants") or {}).items() for c, x in (d.get("per_check") or {}).items()]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        lat, tot = run.get("latency_s"), run.get("totals")
        if lat:
            st.markdown(f"Latency: median {_pv(lat.get('p50'), '{:.1f}')} s, 95th percentile {_pv(lat.get('p95'), '{:.1f}')} s (n={lat.get('n')}).")
        if tot:
            st.markdown(f"Totals: ${tot.get('cost_usd', 0):.2f}, {tot.get('tokens', 0)} tokens.")
        st.markdown("**Limitations**\n" + "\n".join(f"- {t}" for t in _limitations(run)))
    st.subheader("Method and limitations")
    st.markdown(METHOD_TEXT)
    st.markdown("\n".join(f"- {t}" for t in _limitations(run or {})))
    off = view.get("offline_suite")
    if off:
        st.subheader("Offline suite")
        st.caption("Synthetic fixture run through the same checks (no model called); it tests the pipeline, not VERA-DAR's answers.")
        st.markdown(f"**{off.get('n_traces')}** fixture answers, **{off.get('n_errors')}** simulated capture errors.")
    st.subheader(TNR_SECTION)
    val = view.get("check_validation")
    if not val:
        st.info("Check validation is not available on this deployment.")
    else:
        st.caption(f"Evidence dated {view.get('validation_file_date') or 'unknown'} (UTC). Positive class = FAIL. {CHECK_ACCURACY_TEXT} {TNR_NOTE}")
        st.dataframe(pd.DataFrame([{"check": c, "n": x.get("n"), "TP": x.get("tp"), "TN": x.get("tn"), "FP": x.get("fp"),
                                    "FN": x.get("fn"), "TPR": _pv(x.get("tpr")), "TNR": _tnr_cell(x.get("tn"), x.get("fp"), x.get("tnr"))}
                                   for c, x in val["checks"].items()]), hide_index=True)


def fetch_private(base_url: str, key: str, timeout: int = 15):
    """(status_code, body or None). The key goes in a header only; nothing is logged; errors give the status only."""
    import requests
    try:
        r = requests.get(f"{base_url.rstrip('/')}/trace/private", headers={"X-Viz-Key": key}, timeout=timeout)
        return r.status_code, (r.json() if r.status_code == 200 else None)
    except (requests.exceptions.RequestException, ValueError):
        return 0, None


def _render_private(base_url: str | None = None) -> None:
    """Private tier: the key is typed into a password field, kept in session_state only, never in a URL or message."""
    st.subheader("Trace detail (viz key required)")
    with st.form("trace_viz_form"):
        st.text_input("Viz key", type="password", key="trace_viz_key", value="", help="Sent only to this app's API in a header.")
        go = st.form_submit_button("Unlock detail")
    key = st.session_state.get("trace_viz_key") or ""
    if go and key:
        if base_url is None:
            from vera.public_mode import upstream_base_url
            base_url = upstream_base_url()
        status, body = fetch_private(base_url, key)
        st.session_state["trace_viz_detail"] = body if status == 200 else None
        st.session_state["trace_viz_status"] = status
    status, body = st.session_state.get("trace_viz_status"), st.session_state.get("trace_viz_detail")
    if status is None:
        st.caption("Per-trace detail (question ids, per-check reasons, answer excerpts, source titles) is private.")
    elif status in (401, 403, 429):
        st.warning(PRIVATE_DENIED_TEXT)
    elif status != 200:
        st.warning(PRIVATE_OFF_TEXT)
    elif not body or not body.get("available"):
        st.info(PRIVATE_EMPTY_TEXT)
    else:
        _private_browser(body.get("traces", []))


def _private_browser(traces: list) -> None:
    variants = sorted({str(t.get("variant")) for t in traces})
    variant = st.selectbox("Variant", variants)
    ids = [t["id"] for t in traces if str(t.get("variant")) == variant]
    qid = st.selectbox("Question id", ids)
    t = next(t for t in traces if str(t.get("variant")) == variant and t["id"] == qid)
    st.text(f"Category: {t.get('category')}; split: {t.get('split')}; all-pass: {t.get('all_pass')}")
    if t.get("question"):
        st.markdown("**Question**")
        st.text(t["question"])
    if t.get("capture_error"):
        st.error(CAPTURE_ERROR_TEXT)
    st.dataframe(pd.DataFrame([{"check": c.get("name"), "applies": c.get("applies"),
                                "passed": "n/a" if c.get("passed") is None else str(c.get("passed")),
                                "reason": c.get("reason")} for c in t.get("checks", [])]), hide_index=True)
    st.markdown("**Answer excerpt (first 25 words)**")
    st.text(t.get("answer_excerpt") or "(none)")
    st.markdown("**Sources**")
    for s_ in t.get("sources") or []:
        st.text(str(s_.get("title") or "untitled"))
    if not t.get("sources"):
        st.write("(none)")


def render_trace_eval(results_path=None, validation_path=None, public_path=None, base_url=None) -> None:
    """TRACEVIZ-1022: public tier first (always), then the computed analysis when a results file exists, then the
    key-gated detail. Any exception in a section shows one fixed line (no type, text or traceback)."""
    st.title("VERA-DAR Trace Eval")
    from vera import trace_viz
    public_has_run = False
    try:
        view = trace_viz.public_view(results_path, validation_path, public_path)
        public_has_run = bool(view.get("run"))
        _render_public(view)
    except Exception:  # noqa: BLE001
        st.error(RENDER_ERROR_TEXT)
    try:
        _render_body(*_resolve_paths(results_path, validation_path), public_has_run=public_has_run)
    except Exception:  # noqa: BLE001  (Streamlit's rerun/stop signals derive from BaseException and pass through)
        st.error(RENDER_ERROR_TEXT)
    try:
        _render_private(base_url)
    except Exception:  # noqa: BLE001
        st.error(RENDER_ERROR_TEXT)


def _render_body(results_path, validation_path, public_has_run: bool = False) -> None:
    from vera.trace_eval import guidance as g
    res = _load(results_path)
    if res is None or not res.get("traces"):
        if public_has_run:  # P0-2: the public view above already shows a run; no contradictory empty state
            return
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
