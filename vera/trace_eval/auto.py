"""TRACE eval automation (demo-1022/T): run metadata, aggregation, atomic public JSON, failure records, run compare.

Reuses vera.trace_eval.capture (capture, cost guard), .checks (scoring) and .measure (per-case results, Wilson intervals)
and vera.trace_viz (the closed public projection). Nothing here calls a provider or reads an env file; the live path in
scripts/trace_run.py loads credentials at runtime and passes callables in. Per-case results stay in a private,
gitignored directory; only sanitized aggregates are published.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from vera.trace_eval.checks import CHECKS_VERSION

from vera.trace_viz import ROOT  # noqa: E402  (project root; one definition)
# Why: tmp/trace_eval/ is already gitignored (raw traces hold full answers). Per-case results never leave it.
PRIVATE_DIR = ROOT / "tmp" / "trace_eval"
PUBLIC_OUT = ROOT / "eval_results" / "trace_eval_public.json"
OFFLINE_FIXTURE = ROOT / "tests" / "fixtures" / "trace_auto" / "offline_v1.json"
SCORING_VERSION = f"{CHECKS_VERSION}+agg1"  # bump agg suffix when aggregation arithmetic changes
MODES = ("offline", "live")
TRIGGERS = ("manual", "post-deploy", "scheduled")
_ID_RE = re.compile(r"^[A-Za-z0-9._:+\-]{1,80}$")
_HEX_RE = re.compile(r"^[0-9a-f]{7,64}$")
CONFIG_FILES = ("config/trace_eval_guidance.json", "config/ask-provider-chain.json", "config/model-selection.json")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_sha(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def config_hash(root: Path = ROOT) -> str:
    """Hash over the configs that shape an eval run (missing files count as absent, not as an error)."""
    h = hashlib.sha256()
    for rel in CONFIG_FILES:
        p = root / rel
        h.update(rel.encode() + b"\0" + (p.read_bytes() if p.is_file() else b"<absent>") + b"\0")
    return h.hexdigest()[:16]


def git_commit(root: Path = ROOT) -> str:
    """HEAD of the tooling checkout; 'unknown' when git or a repo is unavailable."""
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        v = out.stdout.strip().lower()
        return v if out.returncode == 0 and _HEX_RE.match(v) else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_metadata(*, run_id: str, mode: str, trigger: str, eval_set_sha256: str, n: int, planned: int, models: list[str],
                 provider: str, checks_validated: bool, started: datetime, finished: datetime, commit: str | None = None,
                 deployed_version: str | None = None, config: str | None = None) -> dict:
    from vera.version_info import deployed_commit
    return {
        "run_id": run_id, "mode": mode, "trigger": trigger,
        "commit": commit or git_commit(), "deployed_version": deployed_version or deployed_commit(),
        "config_hash": config or config_hash(), "eval_set_sha256": eval_set_sha256, "scoring_version": SCORING_VERSION,
        "started_at": started.isoformat(timespec="seconds"), "finished_at": finished.isoformat(timespec="seconds"),
        "n": n, "planned": planned, "provider": provider, "models": sorted({m for m in models if m}),
        "offline": mode == "offline", "checks_validated": bool(checks_validated)}


def aggregate(results: dict, raw_traces: list[dict], meta: dict) -> dict:
    """Public aggregate (through trace_viz's closed projection) with sanitized run metadata. No per-case rows."""
    from vera import trace_viz
    pub = trace_viz.build_public(results, raw_traces, run_meta=meta)
    if pub is None:
        raise ValueError("no traces to aggregate")
    return pub


def atomic_write_json(path: Path, obj: dict) -> None:
    """Temp file in the target directory, fsync, then rename: readers see the old file or the new one, never a part."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def publish_success(public: dict, public_out: Path = PUBLIC_OUT) -> Path:
    atomic_write_json(public_out, public)
    return Path(public_out)


def record_failure(run_dir: Path, run_id: str, reason: str, *, mode: str, stage: str) -> Path:
    """Private failure record. The reason is a short code/type string, never a message that could carry a host or key.
    It never touches the public file."""
    safe = reason if _ID_RE.match(reason or "") else "unspecified"
    rec = {"run_id": run_id, "status": "failed", "mode": mode, "stage": stage, "reason": safe,
           "recorded_at": utc_now().isoformat(timespec="seconds")}
    p = Path(run_dir) / "FAILED.json"
    atomic_write_json(p, rec)
    return p


def comparable(a: dict, b: dict) -> tuple[bool, str]:
    ma, mb = (a or {}).get("run_meta") or {}, (b or {}).get("run_meta") or {}
    if not ma.get("eval_set_sha256") or not mb.get("eval_set_sha256") or not ma.get("scoring_version") or not mb.get("scoring_version"):
        return False, "missing eval-set hash or scoring version"
    if ma["eval_set_sha256"] != mb["eval_set_sha256"]:
        return False, "eval-set hash differs"
    if ma["scoring_version"] != mb["scoring_version"]:
        return False, "scoring version differs"
    return True, ""


def compare_runs(old: dict, new: dict, min_delta: float = 0.0) -> dict:
    """Per variant and check: new minus old pass rate. 'incomparable' unless eval-set hash and scoring version match.
    A change is 'significant' only when the two Wilson 95% intervals do not overlap (indicative, not a test)."""
    ok, why = comparable(old, new)
    if not ok:
        return {"status": "incomparable", "reason": why, "regressions": [], "improvements": []}
    reg, imp = [], []
    for v, nd in (new.get("variants") or {}).items():
        od = (old.get("variants") or {}).get(v)
        if not od:
            continue
        for c, nc in (nd.get("per_check") or {}).items():
            oc = (od.get("per_check") or {}).get(c)
            if not oc or oc.get("rate") is None or nc.get("rate") is None:
                continue
            d = round(nc["rate"] - oc["rate"], 4)
            if abs(d) <= min_delta:
                continue
            ow, nw = oc.get("wilson95"), nc.get("wilson95")
            sig = bool(ow and nw and (nw[0] > ow[1] or ow[0] > nw[1]))
            row = {"variant": v, "check": c, "old": oc["rate"], "new": nc["rate"], "delta": d, "significant": sig}
            (imp if d > 0 else reg).append(row)
    return {"status": "comparable", "reason": "", "regressions": reg, "improvements": imp}


# --- execution (providers are injected; this module never builds one) -----------------------------------------------

def fixture_ask(fixture: dict):
    """Deterministic ask callable from the offline fixture: question text -> canned /ask body. No network, no model."""
    by_text = {q["text"]: q["id"] for q in fixture["questions"]["items"]}

    def ask(question: str) -> dict:
        body = fixture["bodies"][by_text[question]]
        if "__error" in body:
            raise ConnectionError("offline fixture error")
        return json.loads(json.dumps(body))

    return ask


def guarded_ask(ask, *, min_interval_s: float, total_timeout_s: float, clock=None, sleep=None):
    """Rate limit (minimum seconds between calls; concurrency is 1 because capture is sequential) and a total wall-clock
    deadline. On the deadline it sets state['timed_out'] and raises an abort-class AskHTTPError so the run stops."""
    import time as _t

    from vera.trace_eval.capture import AskHTTPError
    clock, sleep = clock or _t.monotonic, sleep or _t.sleep
    state = {"timed_out": False, "calls": 0, "last": None, "t0": clock()}

    def wrapped(question: str) -> dict:
        if clock() - state["t0"] > total_timeout_s:
            state["timed_out"] = True
            raise AskHTTPError(503)
        if state["last"] is not None:
            wait = min_interval_s - (clock() - state["last"])
            if wait > 0:
                sleep(wait)
        state["last"] = clock()
        state["calls"] += 1
        return ask(question)

    wrapped.state = state
    return wrapped


def select_items(doc: dict, limit: int | None) -> list[dict]:
    return doc["items"][:limit] if limit else list(doc["items"])


def eval_set_hash(file_sha: str, items: list[dict]) -> str:
    """Hash of the effective eval set: the question source plus the exact ids run (a subset is a different eval set)."""
    return canonical_sha({"source": file_sha, "ids": [q["id"] for q in items]})


def execute_run(*, doc: dict, source_sha: str, ask, mode: str, trigger: str, private_dir: Path, run_id: str,
                limit: int | None, max_cost: float, provider: str, public_out: Path | None, publish_out: Path | None,
                timed_out=lambda: False, **meta_kw) -> dict:
    """Capture -> score -> aggregate -> publish. Returns {status, run_dir, ...}. Any failure writes FAILED.json in the
    private run dir and leaves every public file untouched. Public files are written only after full success."""
    from vera.trace_eval import capture, measure
    started = utc_now()
    run_dir = Path(private_dir) / run_id
    if run_dir.exists():
        return {"status": "failed", "reason": "run-dir-exists", "run_dir": str(run_dir), "failure": None}
    run_dir.mkdir(parents=True)
    items = select_items(doc, limit)
    stage = "capture"

    def fail(reason: str) -> dict:
        p = record_failure(run_dir, run_id, reason, mode=mode, stage=stage)
        return {"status": "failed", "reason": reason, "run_dir": str(run_dir), "failure": str(p)}

    try:
        traces = run_dir / "traces.jsonl"
        sub = {**doc, "items": items}
        summary = capture.run_capture(sub, "ask_api", traces, ask=ask, max_cost_usd=max_cost, questions_sha256=source_sha)
        if timed_out():
            return fail("total-timeout")
        if summary["stopped"]:
            return fail("cost-guard")
        if summary["n"] == 0:
            return fail("no-cases-run")
        if summary["errors"] == summary["n"]:
            return fail("all-cases-errored")
        stage = "score"
        results = measure.measure([traces], run_dir / "cases.json")  # private per-case results
        raw = measure.load_traces([traces])
        stage = "aggregate"
        finished = utc_now()
        meta = run_metadata(run_id=run_id, mode=mode, trigger=trigger, eval_set_sha256=eval_set_hash(source_sha, items),
                            n=summary["n"], planned=summary["planned"], models=[r.get("model") or "" for r in raw],
                            provider=provider, checks_validated=bool(doc.get("checks_validated", True)),
                            started=started, finished=finished, **meta_kw)
        public = aggregate(results, raw, meta)
        stage = "publish"
        atomic_write_json(run_dir / "aggregate.json", public)
        if public_out:
            atomic_write_json(public_out, public)
        if publish_out:
            publish_success(public, publish_out)
        return {"status": "ok", "run_dir": str(run_dir), "public": str(publish_out or public_out or ""), "summary": summary,
                "failure": None}
    except BaseException as e:  # noqa: BLE001  (interrupts included: record, then re-raise non-Exception signals)
        out = fail("total-timeout" if timed_out() else type(e).__name__)
        if not isinstance(e, Exception):
            raise
        return out
