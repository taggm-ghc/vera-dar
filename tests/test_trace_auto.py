"""demo-1022/T: TRACE eval automation. Offline only: no provider call, no network, no production DB."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from vera import trace_viz
from vera.trace_eval import auto, capture, measure

ROOT = Path(__file__).resolve().parent.parent
FIX = json.loads((ROOT / "tests" / "fixtures" / "trace_auto" / "offline_v1.json").read_text())
QUESTION_TEXTS = [q["text"] for q in FIX["questions"]["items"]]
DOC = {**FIX["questions"], "case_set": "offline-fixture", "checks_validated": True}


def _run(tmp_path, run_id="r1", publish=None, ask=None, limit=None, max_cost=1.0, **kw):
    return auto.execute_run(doc=DOC, source_sha="s" * 64, ask=ask or auto.fixture_ask(FIX), mode="offline", trigger="manual",
                            private_dir=tmp_path / "priv", run_id=run_id, limit=limit, max_cost=max_cost,
                            provider="offline-fixture", public_out=None, publish_out=publish, commit="a" * 40,
                            deployed_version="b" * 40, **kw)


def test_aggregation_arithmetic(tmp_path):
    res = _run(tmp_path)
    assert res["status"] == "ok"
    pub = json.loads((tmp_path / "priv" / "r1" / "aggregate.json").read_text())
    cases = json.loads((tmp_path / "priv" / "r1" / "cases.json").read_text())
    assert pub["n_traces"] == 5 == len(cases["traces"]) and pub["n_errors"] == 1
    v = next(iter(pub["variants"].values()))
    split = v["by_split"]["dev"]
    passed = sum(1 for t in cases["traces"] if t["all_pass"])
    assert split["n"] == 5 and split["errors"] == 1 and split["all_pass"] == passed
    assert split["all_pass_rate"] == round(passed / 5, 4)
    for c in v["per_check"].values():
        assert c["rate"] == round(c["passed"] / c["applies"], 4) and 0 <= c["wilson95"][0] <= c["rate"] <= c["wilson95"][1] <= 1
    assert measure.wilson95(0, 0) is None


def test_metadata_recorded(tmp_path):
    _run(tmp_path)
    m = json.loads((tmp_path / "priv" / "r1" / "aggregate.json").read_text())["run_meta"]
    assert m["commit"] == "a" * 40 and m["deployed_version"] == "b" * 40 and m["scoring_version"] == auto.SCORING_VERSION
    assert m["n"] == 5 and m["provider"] == "offline-fixture" and m["models"] == ["offline-fixture"] and m["mode"] == "offline"
    assert re.match(r"^[0-9a-f]{64}$", m["eval_set_sha256"]) and m["config_hash"] and m["finished_at"].endswith("+00:00")


def test_offline_is_deterministic_and_subset_changes_hash(tmp_path):
    _run(tmp_path, "r1")
    _run(tmp_path, "r2")
    a, b = (json.loads((tmp_path / "priv" / r / "aggregate.json").read_text()) for r in ("r1", "r2"))
    assert a["variants"] == b["variants"] and a["run_meta"]["eval_set_sha256"] == b["run_meta"]["eval_set_sha256"]
    _run(tmp_path, "r3", limit=3)
    c = json.loads((tmp_path / "priv" / "r3" / "aggregate.json").read_text())
    assert c["run_meta"]["eval_set_sha256"] != a["run_meta"]["eval_set_sha256"]


def test_public_has_no_question_text_or_secrets(tmp_path):
    pub = tmp_path / "public.json"
    _run(tmp_path, publish=pub)
    text = pub.read_text()
    for q in QUESTION_TEXTS:
        assert q not in text
    for bad in ("messages", "prompt", "http", "Synthetic", '"traces"', "sk-", "ConnectionError"):
        assert bad not in text, bad
    # per-case rows (with question text in the raw traces) stay in the private dir only
    assert any(q in (tmp_path / "priv" / "r1" / "traces.jsonl").read_text() for q in QUESTION_TEXTS)




def test_atomic_write_no_partial_and_failure_keeps_last_good(tmp_path):
    good = tmp_path / "pub" / "trace_eval_public.json"
    assert _run(tmp_path, "ok1", publish=good)["status"] == "ok"
    before = good.read_bytes()

    def boom(q):
        raise KeyboardInterrupt  # interrupt mid-run: a BaseException, not an Exception

    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, "int1", publish=good, ask=boom)
    assert good.read_bytes() == before
    fr = json.loads((tmp_path / "priv" / "int1" / "FAILED.json").read_text())
    assert fr["status"] == "failed" and fr["reason"] == "KeyboardInterrupt"

    allerr = _run(tmp_path, "bad1", publish=good, ask=lambda q: (_ for _ in ()).throw(ConnectionError("host.example key")))
    assert allerr["status"] == "failed" and allerr["reason"] == "all-cases-errored" and good.read_bytes() == before
    assert "host.example" not in (tmp_path / "priv" / "bad1" / "FAILED.json").read_text()
    assert not list(good.parent.glob("*.tmp"))


def test_cost_guard_and_total_timeout_fail_without_publishing(tmp_path):
    good = tmp_path / "pub.json"

    def costly(q):
        return {**FIX["bodies"]["f01"], "cost_usd": 0.4}

    r = _run(tmp_path, "c1", publish=good, ask=costly, max_cost=0.5)
    assert r["status"] == "failed" and r["reason"] == "cost-guard" and not good.exists()
    t = [0.0]
    ask = auto.guarded_ask(auto.fixture_ask(FIX), min_interval_s=0, total_timeout_s=1, clock=lambda: t[0], sleep=lambda s: None)

    def tick(q):
        t[0] += 2
        return auto.fixture_ask(FIX)(q)
    ask2 = auto.guarded_ask(tick, min_interval_s=0, total_timeout_s=1, clock=lambda: t[0], sleep=lambda s: None)
    r = _run(tmp_path, "t1", publish=good, ask=ask2, timed_out=lambda: ask2.state["timed_out"])
    assert r["status"] == "failed" and r["reason"] == "total-timeout" and not good.exists() and ask.state["calls"] == 0


def test_rate_limit_sleeps_between_calls():
    waits, t = [], [0.0]
    ask = auto.guarded_ask(lambda q: {}, min_interval_s=5, total_timeout_s=100, clock=lambda: t[0], sleep=lambda s: (waits.append(s), t.__setitem__(0, t[0] + s)))
    ask("a"), ask("b")
    assert waits == [5]


def _pub(eval_sha="e1", scoring="s1", rate=0.5, lo=0.4, hi=0.6):
    return {"run_meta": {"eval_set_sha256": eval_sha, "scoring_version": scoring},
            "variants": {"V": {"per_check": {"C": {"rate": rate, "wilson95": [lo, hi]}}}}}


def test_compare_comparability_rule():
    assert auto.compare_runs(_pub(), _pub(eval_sha="e2"))["status"] == "incomparable"
    assert auto.compare_runs(_pub(), _pub(scoring="s2"))["status"] == "incomparable"
    assert auto.compare_runs({"variants": {}}, _pub())["status"] == "incomparable"
    r = auto.compare_runs(_pub(rate=0.9, lo=0.8, hi=1.0), _pub(rate=0.2, lo=0.1, hi=0.3))
    assert r["status"] == "comparable" and r["regressions"][0]["delta"] == -0.7 and r["regressions"][0]["significant"] and not r["improvements"]
    r = auto.compare_runs(_pub(), _pub(rate=0.55, lo=0.3, hi=0.7))
    assert r["improvements"][0]["delta"] == 0.05 and not r["improvements"][0]["significant"]




def _page(at):
    return "\n".join([*(m.value for m in at.markdown), *(c.value for c in at.caption), *(w.value for w in at.warning),
                      *(i.value for i in at.info), *(s.value for s in at.subheader)])


def _app(public, hist, offline, val):
    import functools
    from vera import trace_viz as tv
    from vera.trace_eval.ui import render_trace_eval
    orig = tv.public_view
    tv.public_view = functools.partial(orig, historical_path=hist, offline_path=offline, commit="b" * 40)
    try:
        render_trace_eval("/nonexistent/r.json", val, public)
    finally:
        tv.public_view = orig


def _render(public, hist, offline="/nonexistent/o.json", val="/nonexistent/v.json"):
    return AppTest.from_function(_app, args=(str(public), str(hist), str(offline), str(val)), default_timeout=30).run()


def test_page_current_run_vs_historical_label(tmp_path):
    _run(tmp_path, publish=tmp_path / "pub.json")
    at = _render(tmp_path / "pub.json", "/nonexistent/h.json")  # offline run: always historical
    assert not at.exception and "HISTORICAL" in _page(at) and "Method and limitations" in _page(at)
    # a live-mode run on the deployed build is NOT historical; on another build it is
    d = json.loads((tmp_path / "pub.json").read_text())
    d["run_meta"].update(mode="live", offline=False)
    (tmp_path / "cur.json").write_text(json.dumps(d))
    assert "HISTORICAL" not in _page(_render(tmp_path / "cur.json", "/nonexistent/h.json"))
    d["run_meta"]["deployed_version"] = "c" * 40
    (tmp_path / "old.json").write_text(json.dumps(d))
    assert "HISTORICAL" in _page(_render(tmp_path / "old.json", "/nonexistent/h.json"))






def test_shipped_public_files_are_sanitized():
    for name in ("trace_eval_historical_public.json", "trace_eval_offline_public.json"):
        t = (ROOT / "eval_results" / name).read_text()
        d = json.loads(t)
        assert d["schema"] == trace_viz.SCHEMA and "traces" not in d
        for bad in ("http", "Synthetic", "prompt", "messages"):
            assert bad not in t, (name, bad)
    assert datetime.now(timezone.utc)  # keep import used


def test_entry_script_guards(tmp_path, monkeypatch, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("trace_run", ROOT / "scripts" / "trace_run.py")
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    monkeypatch.delenv(tr.CONFIRM_ENV, raising=False)
    monkeypatch.setattr(auto, "git_commit", lambda *a, **k: "unknown")  # the test boundary refuses child processes
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be reached"))  # noqa: E731
    monkeypatch.setattr(capture, "http_ask", boom)
    monkeypatch.setattr(capture, "inprocess_ask", boom)
    assert tr.main(["--live"]) == 2  # missing limits
    full = ["--live", "--max-cases", "3", "--max-cost", "0.1", "--min-interval-s", "2", "--timeout-s", "30", "--total-timeout-s", "300"]
    assert tr.main(full) == 2  # no owner confirmation
    assert tr.main([*full[:2], "999", *full[3:]]) == 2  # over ceiling
    assert tr.main(["--dry-run", "--questions", str(tmp_path / "none.json")]) == 4
    q = tmp_path / "q.json"
    q.write_text(json.dumps(FIX["questions"]))
    assert tr.main(["--dry-run", "--questions", str(q)]) == 0
    assert "no call made" in capsys.readouterr().out
    assert tr.main(["--offline", "--private-dir", str(tmp_path / "p")]) == 0





def test_page_offline_suite_section():
    at = _render("/nonexistent/p.json", "/nonexistent/h.json", ROOT / "eval_results" / "trace_eval_offline_public.json")
    assert not at.exception and "Offline suite" in _page(at)
    t = (ROOT / "eval_results" / "trace_eval_offline_public.json").read_text()
    for bad in ("http", "Synthetic", "prompt", "messages"):
        assert bad not in t


def test_base_url_must_be_loopback(monkeypatch, capsys):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("trace_run_t", Path(__file__).resolve().parent.parent / "scripts" / "trace_run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m._loopback("http://127.0.0.1:8001") and m._loopback("http://localhost:8001/") and not m._loopback("https://example.org")
    assert not m._loopback("http://127.0.0.1@evil.example") and not m._loopback("ftp://127.0.0.1")
    monkeypatch.setenv(m.CONFIRM_ENV, m.CONFIRM_VALUE)
    argv = ["--live", "--base-url", "https://example.org", "--max-cases", "1", "--max-cost", "0.05", "--min-interval-s", "1",
            "--timeout-s", "60", "--total-timeout-s", "300"]
    assert m.main(argv) == 2
    assert "loopback" in capsys.readouterr().err
