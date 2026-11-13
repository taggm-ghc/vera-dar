"""WP-C (CHUNK-ASSOC-1113, plan D6): turn the output of sql/chunk_assoc_coverage.sql into the coverage report and verdict.

Offline and read-only: it opens no database connection and no network socket; it reads three local files (the query's JSON line, the
optional owner limits file, config/chunk_taxo.json) and prints a report. Verdict values: PASS, FAIL, PENDING-CAL, ERROR.
PENDING-CAL is printed (never PASS) while any CTQ-1 limit is unset (OD-J), or while the job-counted exclusions are not available
or disagree (plan D6, passes 14-16), so the query cannot pass with zero association links.

  python3 -I scripts/chunk_assoc_coverage.py params  [--config config/chunk_taxo.json]     # psql -v arguments
  python3 -I scripts/chunk_assoc_coverage.py report  --raw OUT [--limits FILE] [--config config/chunk_taxo.json]

Limits file (JSON, owner-written after a calibration note; absent file or null values mean "unset"):
  {"age_limit_hours": <number>, "covered_floor": <0<x<=1>, "covered_basis": "stored_or_derived"|"stored", "calibration_note": "<text>"}
Exit status: 0 PASS, 1 FAIL, 3 PENDING-CAL, 2 ERROR.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

CLASSES = ("covered_all_stored", "covered_all_derived", "partial_depth", "no_vector_yet", "stale_vector",
           "new_since_build", "abstained", "unexplained")
EXCLUSIONS = ("portal", "portal_error", "no_source", "empty")  # report-only exclusions, counted by the job
AGED = ("no_vector_yet", "stale_vector", "new_since_build")
DEFAULT_CN = "chunk_taxo_auto"
BASES = ("stored_or_derived", "stored")
EXIT = {"PASS": 0, "FAIL": 1, "ERROR": 2, "PENDING-CAL": 3}


class CoverageError(ValueError):
    """Malformed input: the report prints ERROR, never a verdict."""


def _int(v, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (math.inf, -math.inf) or int(v) != v or v < 0:
        raise CoverageError(f"{what}: not a non-negative integer")
    return int(v)


def _inside(p: str, base: Path) -> Path:
    r = Path(p).resolve()
    if base != r and base not in r.parents:
        raise CoverageError("path is outside the working directory")
    return r


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    try:
        return {"max_abstain_share": float(cfg["display_gate"]["max_abstain_share"]),
                "model": str(cfg["cloudflare"]["model"]),
                "text_chars": int(cfg.get("job", {}).get("text_chars", 8000))}
    except (KeyError, TypeError, ValueError) as e:
        raise CoverageError("config/chunk_taxo.json lacks display_gate.max_abstain_share or cloudflare.model") from e


def load_limits(path: Path | None) -> tuple[dict, list[str]]:
    """Return (limits, pending reasons). Unset or incomplete limits are pending reasons, invalid ones are errors."""
    lim = {"age_limit_hours": None, "covered_floor": None, "covered_basis": "stored_or_derived"}
    pend: list[str] = []
    raw = {}
    if path is not None and path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise CoverageError("limits file is not a JSON object")
    a, f = raw.get("age_limit_hours"), raw.get("covered_floor")
    if a is not None:
        if isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a) or a <= 0:
            raise CoverageError("age_limit_hours must be a positive finite number")
        lim["age_limit_hours"] = float(a)
    if f is not None:
        if isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f) or not 0 < f <= 1:
            raise CoverageError("covered_floor must be a number in (0, 1]")
        lim["covered_floor"] = float(f)
    b = raw.get("covered_basis", "stored_or_derived")
    if b not in BASES:
        raise CoverageError("covered_basis must be one of " + ", ".join(BASES))
    lim["covered_basis"] = b
    if lim["age_limit_hours"] is None:
        pend.append("age_limit_hours unset (OD-J)")
    if lim["covered_floor"] is None:
        pend.append("covered_floor unset (OD-J)")
    note = raw.get("calibration_note")
    if (lim["age_limit_hours"] is not None or lim["covered_floor"] is not None) and not (isinstance(note, str) and note.strip()):
        pend.append("limits lack a calibration_note")
    return lim, pend


def parse_raw(text: str) -> dict:
    """The query prints exactly one JSON object line; take the last line that starts with '{'."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise CoverageError("no JSON line in the query output")
    try:
        raw = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        raise CoverageError("query output line is not valid JSON") from e
    if not isinstance(raw, dict) or "spans" not in raw or "classes" not in raw:
        raise CoverageError("query output lacks spans/classes")
    return raw


def _job_counts(row: dict | None, what: str) -> dict | None:
    """Exclusion counts of a run row as ints, or None when the row or any needed key is missing (then PENDING-CAL)."""
    if not row:
        return None
    c = row.get("counts") or {}
    need = [f"skipped_{k}" for k in EXCLUSIONS] + [f"excluded_with_vector_{k}" for k in EXCLUSIONS]
    if any(k not in c for k in need):
        return None
    out = {k: _int(c[k], f"{what}.{k}") for k in need}
    for k in EXCLUSIONS:
        if out[f"excluded_with_vector_{k}"] > out[f"skipped_{k}"]:
            raise CoverageError(f"{what}: excluded_with_vector_{k} exceeds skipped_{k}")
    out["quarantined"] = _int(c["quarantined"], f"{what}.quarantined") if "quarantined" in c else None
    return out


def evaluate(raw: dict, limits: dict, limit_pending: list[str], max_abstain_share: float) -> dict:
    for key in ("classes", "ages_desc", "build", "scan"):  # WP-I (WP-C V-a): a malformed section is an ERROR, never a crash or a FAIL
        if raw.get(key) is not None and not isinstance(raw.get(key), dict):
            raise CoverageError(f"{key}: not an object")
    spans = _int(raw["spans"], "spans")
    with_vec = _int(raw.get("with_vector", 0), "with_vector")
    cl = {k: 0 for k in CLASSES}
    for k, v in (raw["classes"] or {}).items():
        if k not in cl:
            raise CoverageError(f"unknown class {k!r} in query output")
        cl[k] = _int(v, f"classes.{k}")
    if sum(cl.values()) != spans:
        raise CoverageError("query classes do not sum to spans")
    ages_desc = raw.get("ages_desc") or {}
    pend, fail, notes = list(limit_pending), [], []

    build, scan = raw.get("build"), raw.get("scan")
    # Exclusion counts: the latest W4 build row; before any build, the latest complete-scan row (plan D6, Q2 subtraction rule).
    if scan is None:
        pend.append("no complete-scan run row yet (scan_complete 1)")
    src = build if build else scan
    jc = _job_counts(src, "build" if build else "scan")
    if src is not None and jc is None:
        pend.append("run row lacks the job-counted exclusion keys (WP-I not live)")
    if build and scan:
        sc = _job_counts(scan, "scan")
        if jc is not None and sc is None:
            pend.append("latest complete-scan row lacks the exclusion keys, so it cannot be compared with the W4 build row")
        elif jc is not None and any(jc[k] != sc[k] for k in jc if k != "quarantined"):
            pend.append("latest complete-scan exclusion counts differ from the latest W4 build row")
    excl = {k: 0 for k in EXCLUSIONS}
    excl_vec = {k: 0 for k in EXCLUSIONS}
    if jc is not None:
        excl = {k: jc[f"skipped_{k}"] for k in EXCLUSIONS}
        excl_vec = {k: jc[f"excluded_with_vector_{k}"] for k in EXCLUSIONS}
    adj = dict(cl)
    n_ex, n_ex_vec = sum(excl.values()), sum(excl_vec.values())
    adj["no_vector_yet"] -= n_ex - n_ex_vec  # excluded spans without a vector sit in no_vector_yet
    target = "abstained" if build else "unexplained"  # Q2: abstained once a build exists, else unexplained
    adj[target] -= n_ex_vec
    for k, v in adj.items():
        if v < 0:
            fail.append(f"class {k} would be negative after removing job-counted exclusions ({v})")
    eligible = spans - n_ex
    if sum(adj.values()) != eligible:
        fail.append("sum identity broken: classes do not sum to the eligible set")
    if eligible <= 0:
        pend.append("no eligible chunks")

    def share(n: int, d: int):
        return None if d <= 0 else n / d

    vec_elig = with_vec - n_ex_vec
    covered = adj["covered_all_stored"] + (0 if limits["covered_basis"] == "stored" else adj["covered_all_derived"])
    shares = {"covered_stored": share(adj["covered_all_stored"], eligible),
              "covered_derived": share(adj["covered_all_derived"], eligible),
              "covered_basis": share(covered, eligible),
              "partial_depth": share(adj["partial_depth"], eligible),
              "abstained": share(adj["abstained"], vec_elig)}  # denominator: vector-bearing eligible chunks
    if adj["unexplained"] > 0:
        fail.append(f"unexplained {adj['unexplained']} > 0")
    # Age of the oldest ELIGIBLE waiting chunk. Job-excluded spans cannot be told apart in SQL, so assume they are the oldest and skip
    # that many: a lower bound on the true maximum, so an age FAIL is never false (D1 of the WP-C verification).
    ages = {}
    for k in AGED:
        lst = ages_desc.get(k) or []
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and x >= 0 for x in lst):
            raise CoverageError(f"ages_desc.{k}: not a list of non-negative numbers")
        skip = (n_ex - n_ex_vec if k == "no_vector_yet" else n_ex_vec)
        if adj[k] <= 0:
            ages[k] = 0.0
        elif skip < len(lst):
            ages[k] = float(lst[skip])
        else:
            ages[k] = None
            if limits["age_limit_hours"] is not None:
                pend.append(f"{k}: age unknown (more job-excluded spans than ages listed)")
    if limits["age_limit_hours"] is not None:
        for k in AGED:
            if ages[k] is not None and adj[k] > 0 and ages[k] > limits["age_limit_hours"]:
                fail.append(f"{k} older than the age limit ({ages[k]:.2f} h > {limits['age_limit_hours']} h)")
    if shares["abstained"] is not None and shares["abstained"] > max_abstain_share:
        fail.append(f"abstained share {shares['abstained']:.4f} > display_gate.max_abstain_share {max_abstain_share}")
    if limits["covered_floor"] is not None and eligible > 0 and shares["covered_basis"] < limits["covered_floor"]:
        fail.append(f"covered share {shares['covered_basis']:.4f} < covered_floor {limits['covered_floor']}")
    job_abs = ((scan or {}).get("counts") or {}).get("abstained")
    if build and scan and scan.get("run_id") == build.get("run_id") and job_abs is not None:
        if _int(job_abs, "scan.abstained") != adj["abstained"]:
            fail.append(f"job abstain count {job_abs} differs from the query's {adj['abstained']}")
    elif job_abs is None:
        notes.append("job abstain count not reported (advisory A7 comparison skipped)")
    verdict = "FAIL" if fail else ("PENDING-CAL" if pend else "PASS")
    return {"verdict": verdict, "fail": fail, "pending": pend, "notes": notes, "spans": spans, "eligible": eligible,
            "classes": adj, "excluded": excl, "excluded_with_vector": excl_vec, "shares": shares, "ages_max_hours": ages,
            "vector_bearing_eligible": vec_elig, "quarantined_reported": (jc or {}).get("quarantined"),
            "sources_total": raw.get("sources_total"), "sources_with_covered": raw.get("sources_with_covered"),
            "build_age_hours": (build or {}).get("age_hours"), "scan_age_hours": (scan or {}).get("age_hours"),
            "scan_state": (scan or {}).get("state")}


def render(r: dict) -> str:
    f = lambda x: "n/a" if x is None else f"{x:.4f}"  # noqa: E731
    out = [f"COVERAGE spans={r['spans']} eligible={r['eligible']} (job-counted exclusions removed: {r['excluded']})",
           "classes: " + " ".join(f"{k}={v}" for k, v in r["classes"].items()),
           f"covered share: stored={f(r['shares']['covered_stored'])} derived={f(r['shares']['covered_derived'])} "
           f"(basis {f(r['shares']['covered_basis'])}); partial_depth share={f(r['shares']['partial_depth'])}; "
           f"abstained share={f(r['shares']['abstained'])} of {r['vector_bearing_eligible']} vector-bearing eligible",
           "max age (h): " + " ".join(f"{k}={v}" for k, v in r["ages_max_hours"].items()),
           f"sources: {r['sources_with_covered']} of {r['sources_total']} have a covered chunk; quarantined (job, inside no_vector_yet): "
           f"{r['quarantined_reported']}; build row age h={r['build_age_hours']}; complete-scan row age h={r['scan_age_hours']} "
           f"({r['scan_state']})"]
    out += [f"FAIL: {x}" for x in r["fail"]] + [f"PENDING-CAL: {x}" for x in r["pending"]] + [f"note: {x}" for x in r["notes"]]
    out.append(f"VERDICT: {r['verdict']}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("params", "report"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="config/chunk_taxo.json")
    sub.choices["report"].add_argument("--raw", required=True)
    sub.choices["report"].add_argument("--limits", default=None)
    a = ap.parse_args(argv)
    try:
        base = Path.cwd().resolve()
        cfg = load_config(_inside(a.config, base))
        if a.cmd == "params":
            print(f"-v cn={DEFAULT_CN} -v model={cfg['model']} -v text_chars={cfg['text_chars']}")
            return 0
        limits, pend = load_limits(_inside(a.limits, base) if a.limits else None)
        res = evaluate(parse_raw(_inside(a.raw, base).read_text(encoding="utf-8")), limits, pend, cfg["max_abstain_share"])
    except (CoverageError, OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError) as e:
        print(f"VERDICT: ERROR ({type(e).__name__}: {e})")
        return EXIT["ERROR"]
    print(render(res))
    return EXIT[res["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
