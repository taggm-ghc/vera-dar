"""TRACEVIZ-1022: two-tier view of the Trace Eval evidence.

Public tier: aggregates and structure only, from a closed allowlist of fields. Private tier: per-trace detail, served only
by the API behind VERA_TRACE_VIZ_KEY (X-Viz-Key). Nothing here reads a database, the network or a secret file.
"""
import hashlib
import hmac
import json
import math
import os
import re
import threading
import time
from pathlib import Path

from fastapi import Header, HTTPException, Request

ROOT = Path(__file__).resolve().parent.parent
VIZ_KEY_ENV = "VERA_TRACE_VIZ_KEY"
VIZ_KEY_MIN_LEN = 32  # generate with: python3 -c 'import secrets;print(secrets.token_urlsafe(32))'
DENIED_DETAIL = "Not available"  # identical for a missing, wrong or unset key
RESULTS_PATH = ROOT / "eval_results" / "trace_eval_v1.json"
VALIDATION_PATH = ROOT / "eval_results" / "trace_eval_v1_validation.json"
PUBLIC_PATH = ROOT / "eval_results" / "trace_eval_public.json"
# Shipped fallbacks (sanitized aggregates, no questions/traces): the superseded v1 run and the offline fixture suite summary.
HISTORICAL_PATH = ROOT / "eval_results" / "trace_eval_historical_public.json"
OFFLINE_PATH = ROOT / "eval_results" / "trace_eval_offline_public.json"
SCHEMA = "vera-trace-public-v1"
# P0-4: a results-only publication commit keeps a run current only when the application code is provably unchanged.
PUBLICATION_PATH = ROOT / "eval_results" / "trace_eval_publication.json"
PUBLICATION_SCHEMA = "vera-trace-publication-v1"
# Behaviour-determining files present in the runtime image (Dockerfile `COPY . .` minus .dockerignore). Excludes
# eval_results/, tests/, docs/, scripts/ and env/host templates, so publishing results never changes the fingerprint.
FP_FILES = ("main.py", "streamlit_app.py", "start.sh", "requirements.txt")
FP_DIRS = ("vera", "pages", "safety", "config", ".streamlit")
_FP_SKIP_DIRS = {"__pycache__", "bundle", "bundles", "env"}  # config/env/ holds env files: never read
_FP_SKIP = re.compile(r"(\.py[cod]$|\.log$|:Zone\.Identifier$|\.meta\.json$|\.SHA256SUMS$|^manifest-filled|-prv(\.|$)|^\.env|"
                      r"^live-hosts)")
FAIL_LIMIT, FAIL_WINDOW_S = 10, 600  # failed key attempts per client per window before 429 (generic)

_STAGE_ORDER = ("A1_citation_present", "A2_citations_resolve", "A3_numbers_grounded", "A4_phantom_evidence",
                "A5_window", "A6_support", "A7_premise_corrected", "A8_scope_decline", "A9_clarify_or_assume")


def _load(path) -> dict | None:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def _date(path) -> str | None:
    try:
        return time.strftime("%Y-%m-%d", time.gmtime(Path(path).stat().st_mtime))
    except OSError:
        return None


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _pct_pair(c: dict) -> dict:
    w = c.get("wilson95")
    return {"passed": _num(c.get("passed")), "applies": _num(c.get("applies")), "rate": _num(c.get("rate")),
            "wilson95": [_num(w[0]), _num(w[1])] if isinstance(w, list) and len(w) == 2 else None}


def _names():
    from vera.trace_eval.ui import CHECK_NAMES, VARIANT_NAMES, _namer  # lazy: keeps streamlit out of API start-up
    return CHECK_NAMES, VARIANT_NAMES, _namer


def _quantile(xs: list, q: float):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 3)


_CAT_RE = re.compile(r"^[a-z0-9_]{1,40}$")


def _name_map(known: dict, prefix: str, raw: list) -> dict:
    """Display names only: a known raw id -> its display name, an existing display name stays, anything else -> 'Prefix N'."""
    shown = set(known.values())
    out, n = {}, 0
    for r in raw:
        r = str(r)
        if r in known:
            out[r] = known[r]
        elif r in shown:
            out[r] = r
        else:
            n += 1
            out[r] = f"{prefix} {n}"
    return out


_META_STR = ("run_id", "mode", "trigger", "commit", "deployed_version", "config_hash", "eval_set_sha256", "scoring_version",
             "started_at", "finished_at", "provider")
_META_RE = re.compile(r"^[A-Za-z0-9._:+\-]{1,80}$")
_META_MODES = ("offline", "live", "historical")
_META_TRIGGERS = ("manual", "post-deploy", "scheduled")


def _project_meta(m) -> dict | None:
    """Closed run metadata: identifiers only (strict charset), numbers, booleans. Free text is dropped."""
    if not isinstance(m, dict):
        return None
    out = {k: (str(m[k]) if _META_RE.match(str(m.get(k) or "")) else None) for k in _META_STR}
    out["mode"] = out["mode"] if out["mode"] in _META_MODES else None
    out["trigger"] = out["trigger"] if out["trigger"] in _META_TRIGGERS else None
    out["models"] = [str(x) for x in (m.get("models") or []) if _META_RE.match(str(x))][:5] if isinstance(m.get("models"), list) else []
    for k in ("n", "planned"):
        out[k] = _num(m.get(k))
    for k in ("offline", "checks_validated"):
        out[k] = m.get(k) if isinstance(m.get(k), bool) else None
    return out


def _project(src: dict) -> dict:
    """The ONE closed projection used for built and loaded summaries: only known fields, numbers through _num, names
    through display maps. Anything else in `src` (extra keys, raw ids, strings in numeric slots) is dropped."""
    cn, vn, _n = _names()
    variants_in = src.get("variants") if isinstance(src.get("variants"), dict) else {}
    vm = _name_map(vn, "Variant", list(variants_in))
    cm = _name_map(cn, "Check", [n for d in variants_in.values() if isinstance(d, dict)
                                  for n in (d.get("per_check") or {})])
    variants = {}
    for k, d in variants_in.items():
        if not isinstance(d, dict):
            continue
        cats, i = {}, 0
        for n, c in (d.get("per_category") or {}).items():
            if not isinstance(c, dict):
                continue
            i += 1
            name = str(n) if _CAT_RE.match(str(n)) else f"Category {i}"
            cats[name] = _pct_pair({**c, "passed": c.get("passed", c.get("all_pass")), "applies": c.get("applies", c.get("n"))})
        variants[vm[str(k)]] = {
            "by_split": {(str(s) if _CAT_RE.match(str(s)) else "split"): {f: _num(v.get(f)) for f in ("n", "errors", "all_pass", "all_pass_rate")}
                         for s, v in (d.get("by_split") or {}).items() if isinstance(v, dict)},
            "per_check": {cm[str(n)]: _pct_pair(c) for n, c in (d.get("per_check") or {}).items() if isinstance(c, dict)},
            "per_category": cats}
    lat, tot = src.get("latency_s"), src.get("totals")
    return {"schema": SCHEMA, "generated_at": str(src.get("generated_at") or "")[:10] if re.match(r"^\d{4}-\d\d-\d\d", str(src.get("generated_at") or "")) else None,
            "n_questions": _num(src.get("n_questions")), "n_traces": _num(src.get("n_traces")), "n_errors": _num(src.get("n_errors")),
            "variants": variants,
            "latency_s": {f: _num(lat.get(f)) for f in ("p50", "p95", "n")} if isinstance(lat, dict) else None,
            "totals": {"cost_usd": _num(tot.get("cost_usd")), "tokens": _num(tot.get("tokens"))} if isinstance(tot, dict) else None,
            "run_meta": _project_meta(src.get("run_meta"))}


def build_public(results: dict | None, raw_traces: list | None = None, run_meta: dict | None = None) -> dict | None:
    """Aggregates from a results document (and optionally raw capture records for latency/cost), through _project."""
    if not results or not results.get("traces"):
        return None
    lat, cost, toks = [], 0.0, 0
    for r in raw_traces or []:
        if _num(r.get("latency_s")) is not None:
            lat.append(r["latency_s"])
        cost += _num(r.get("cost_usd")) or 0.0
        toks += int(_num(r.get("tokens_used")) or 0)
    return _project({"generated_at": results.get("generated_at"),
                     "n_questions": len({t.get("id") for t in results["traces"] if t.get("id")}),
                     "n_traces": len(results["traces"]), "n_errors": sum(1 for t in results["traces"] if t.get("error")),
                     "variants": results.get("variants") or {},
                     "latency_s": {"p50": _quantile(lat, 0.5), "p95": _quantile(lat, 0.95), "n": len(lat)} if lat else None,
                     "totals": {"cost_usd": round(cost, 4), "tokens": toks} if raw_traces else None,
                     "run_meta": run_meta})


def _validation_public(val: dict | None) -> dict | None:
    if not val:
        return None
    cn, _vn, namer = _names()
    cm = namer(cn, "Check", list((val.get("per_check") or {})))
    return {"checks": {cm[str(n)]: {f: _num(c.get(f)) for f in ("n", "tp", "tn", "fp", "fn", "tpr", "tnr")}
                       for n, c in (val.get("per_check") or {}).items()}}


def stage_flow() -> list[dict]:
    """The evaluation structure (static, from the frozen check catalogue): the 'per-stage flow' that is always shown."""
    cn, _vn, _n = _names()
    return [{"stage": i, "check": cn[c]} for i, c in enumerate(_STAGE_ORDER, 1)]


def run_build(meta: dict | None) -> str:
    """The build a run evaluated: the deployed version it recorded, else its commit; 'unknown' when neither is known."""
    from vera.version_info import short_commit
    for k in ("deployed_version", "commit"):
        v = (meta or {}).get(k)
        if v and v != "unknown":
            return short_commit(v)
    return "unknown"


def is_historical(meta: dict | None, deployed_build: str, publication=None, code_root=None) -> bool:
    """HISTORICAL unless the run's build is known and equals the deployed build (an unknown on either side is historical),
    or, for a later results-only commit, the publication marker proves the deployed code equals the evaluated code."""
    if (meta or {}).get("offline"):
        return True  # an offline fixture run evaluates no deployed build
    rb = run_build(meta)
    if rb == "unknown" or deployed_build == "unknown":
        return True
    return rb != deployed_build and not _same_code(meta, publication, code_root)


def _fp_paths(root: Path) -> list[str]:
    out = [f for f in FP_FILES if (root / f).is_file() and not (root / f).is_symlink()]
    for d in FP_DIRS:
        for dp, dns, fns in os.walk(root / d):  # os.walk does not follow directory symlinks
            dns[:] = [n for n in dns if n not in _FP_SKIP_DIRS and not n.endswith("-prv")]
            for n in fns:
                f = Path(dp) / n
                rel = f.relative_to(root).as_posix()
                if _FP_SKIP.search(n) or (Path(dp).name == "texts" and n.endswith(".txt")) or f.is_symlink() or not f.is_file():
                    continue
                out.append(rel)
    return sorted(out)


def code_fingerprint(root=None) -> str:
    """sha256 over the sorted (relative path, file sha256) of the behaviour-determining files under `root` (FP_FILES,
    FP_DIRS). Deterministic; identical for a clean checkout of a commit and that commit's runtime image. Publisher:
    python -I -c "from vera.trace_viz import code_fingerprint; print(code_fingerprint('.'))" """
    root = Path(root or ROOT).resolve()
    h = hashlib.sha256(b"vera-code-fp-v1\n")
    for rel in _fp_paths(root):
        h.update(rel.encode("utf-8") + b"\0" + hashlib.sha256((root / rel).read_bytes()).hexdigest().encode() + b"\n")
    return h.hexdigest()


def _same_code(meta: dict | None, publication, code_root) -> bool:
    """True only if a valid publication marker names this run's build and its code fingerprint equals ours. Any doubt: False."""
    try:
        m = _load(publication or PUBLICATION_PATH)
        if not m or m.get("schema") != PUBLICATION_SCHEMA:
            return False
        eb, fp = str(m.get("evaluated_build") or ""), str(m.get("code_fingerprint") or "")
        if not re.fullmatch(r"[0-9a-f]{40}", eb) or not re.fullmatch(r"[0-9a-f]{64}", fp):
            return False
        rv = next((str(meta[k]).lower() for k in ("deployed_version", "commit") if (meta or {}).get(k) not in (None, "", "unknown")), "")
        if not re.fullmatch(r"[0-9a-f]{7,40}", rv) or not (eb == rv if len(rv) == 40 else eb.startswith(rv)):
            return False
        return hmac.compare_digest(code_fingerprint(code_root), fp)
    except Exception:  # noqa: BLE001 - unreadable tree or marker: cannot establish equivalence
        return False


def public_view(results_path=None, validation_path=None, public_path=None, commit: str | None = None,
                historical_path=None, offline_path=None, publication_path=None, code_root=None) -> dict:
    """Always returns a dict with the structure and a `status`. Source order for `run`: the newest published run file, then a
    local results file, then the shipped historical file; if none exist the method and validation sections still render."""
    from vera.version_info import deployed_commit, short_commit
    pub_p, val_p = Path(public_path or PUBLIC_PATH), Path(validation_path or VALIDATION_PATH)
    res_p = Path(results_path or os.environ.get("VERA_TRACE_EVAL_RESULTS") or RESULTS_PATH)
    hist_p, off_p = Path(historical_path or HISTORICAL_PATH), Path(offline_path or OFFLINE_PATH)
    summary, src_date, source = None, None, None
    for kind, p in (("run", pub_p), ("historical_file", hist_p)):
        d = _load(p)
        if d and d.get("schema") == SCHEMA and d.get("variants"):
            summary, src_date, source = _project(d), _date(p), kind
            break
        if kind == "run":
            r = build_public(_load(res_p))
            if r:
                summary, src_date, source = r, _date(res_p), "run"
                break
    val = _load(val_p)
    build = short_commit(commit or deployed_commit())
    meta = (summary or {}).get("run_meta")
    fin = str((meta or {}).get("finished_at") or "")[:10]
    if re.match(r"^\d{4}-\d\d-\d\d$", fin):
        src_date = fin  # the run's own UTC date beats file mtime
    off = _load(off_p)
    off = _project(off) if off and off.get("schema") == SCHEMA else None
    return {"schema": SCHEMA, "status": "ok" if summary else "no_run_results", "build": build, "source": source,
            "run_build": run_build(meta), "historical": (True if source == "historical_file" else is_historical(meta, build, publication_path, code_root)) if summary else None,
            "stages": stage_flow(), "run": summary, "run_file_date": src_date if summary else None, "offline_suite": off,
            "check_validation": _validation_public(val), "validation_file_date": _date(val_p) if val else None}


def private_view(results_path=None) -> dict:
    """Per-trace detail. Only ever returned by the key-protected route."""
    res = _load(results_path or os.environ.get("VERA_TRACE_EVAL_RESULTS") or RESULTS_PATH)
    if not res or not res.get("traces"):
        return {"available": False, "traces": []}
    cn, vn, namer = _names()  # deny by default: raw variant/check ids never leave the API, only display names
    vm = namer(vn, "Variant", [t.get("variant") for t in res["traces"]])
    cm = namer(cn, "Check", [c.get("name") for t in res["traces"] for c in t.get("checks", [])])
    rows = []
    for t in res["traces"]:
        rows.append({"id": t.get("id"), "split": t.get("split"), "category": t.get("category"),
                     "variant": vm.get(str(t.get("variant")), "Variant"),
                     "all_pass": t.get("all_pass"), "capture_error": bool(t.get("error")), "question": t.get("question"),
                     "checks": [{"name": cm.get(str(c.get("name")), "Check"), "applies": c.get("applies"), "passed": c.get("passed"),
                                 "reason": c.get("reason")} for c in t.get("checks", [])],
                     "answer_excerpt": t.get("answer_excerpt"),
                     "sources": [{"title": s.get("title"), "url": s.get("url")} for s in (t.get("sources") or [])]})
    return {"available": True, "generated_at": str(res.get("generated_at") or "")[:10] or None, "traces": rows}


# --- key enforcement (server side) ---------------------------------------------------------------------------------

_fail_lock, _fails = threading.Lock(), {}


def reset_for_tests() -> None:
    with _fail_lock:
        _fails.clear()


def _client(request: Request) -> str:
    from vera.public_mode import client_id, get_limiter
    return client_id(request, get_limiter().caps)


def _too_many(cid: str, now: float) -> bool:
    with _fail_lock:
        recent = [t for t in _fails.get(cid, []) if now - t < FAIL_WINDOW_S]
        _fails[cid] = recent
        return len(recent) >= FAIL_LIMIT


def _note_fail(cid: str, now: float) -> None:
    with _fail_lock:
        _fails.setdefault(cid, []).append(now)
        if len(_fails) > 5000:  # bound memory: drop the oldest clients
            for k in list(_fails)[:1000]:
                _fails.pop(k, None)


def verify_viz_key(request: Request, x_viz_key: str | None = Header(default=None, alias="X-Viz-Key")) -> None:
    """Fail closed: unset/weak/API-key-equal config, a missing key and a wrong key all give the same 403. The key is
    checked first, so a correct key always passes (no lockout DoS: UI visitors share one client id). Failures are
    counted per client; an over-limit client still sees the same 403 (no 429, no configured-or-not oracle)."""
    expected = (os.environ.get(VIZ_KEY_ENV) or "").strip()
    usable = len(expected) >= VIZ_KEY_MIN_LEN and expected != (os.environ.get("VERA_API_KEY") or "").strip()
    if usable and x_viz_key and hmac.compare_digest(x_viz_key.encode(), expected.encode()):
        return
    if usable:
        cid, now = _client(request), time.time()
        if not _too_many(cid, now):  # over the limit: stop recording (bounded memory); the answer is unchanged
            _note_fail(cid, now)
    raise HTTPException(status_code=403, detail=DENIED_DETAIL)


def _cli(argv: list[str]) -> int:
    """python -I -m vera.trace_viz RESULTS.json OUT.json [RAW.jsonl ...]: write the public aggregates file."""
    if len(argv) < 2:
        print("usage: RESULTS.json OUT.json [RAW.jsonl ...]")
        return 2
    raw = []
    for p in argv[2:]:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            try:
                raw.append(json.loads(line))
            except ValueError:
                pass
    out = build_public(_load(argv[0]), raw)
    if out is None:
        print("no traces in results; nothing written")
        return 1
    Path(argv[1]).write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {argv[1]}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv[1:]))
