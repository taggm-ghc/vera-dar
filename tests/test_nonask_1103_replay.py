"""NONASK-1103 WP-E: tests for the offline replay harness and its synthetic fixtures (offline, no network/DB)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import nonask_1103_replay as R  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "nonask_1103"


def test_wilson_known_values():
    w = R.wilson(0, 10)
    assert w["lo"] == 0.0 and abs(w["hi"] - 0.2775) < 1e-3
    w = R.wilson(5, 10)
    assert abs(w["lo"] - 0.2366) < 1e-3 and abs(w["hi"] - 0.7634) < 1e-3
    assert R.wilson(0, 0)["rate"] is None


def test_fixture_sizes_and_split_integrity():
    qs = json.loads((FIX / "questions.json").read_text())["questions"]
    assert len(qs) >= 70 and sum(q["label"] == "technical" for q in qs) >= 30
    assert {q["split"] for q in qs} == {"calibration", "holdout"}
    for fn, key in (("questions.json", "questions"), ("scenarios.json", "scenarios"), ("identifiers.json", "records")):
        for it in json.loads((FIX / fn).read_text())[key]:
            assert it["split"] == R.split_of(it["cluster"])
    assert any(q["tag"] == "marl_negative" for q in qs) and any(q["label"] == "general" for q in qs)


def test_scenarios_cover_reported_cases_and_are_clean():
    sc = json.loads((FIX / "scenarios.json").read_text())["scenarios"]
    ids = {s["id"] for s in sc}
    assert {"s01_orchestration", "s02_orchestration_only_offtopic", "s03_handle_record"} <= ids
    raw = "".join((FIX / f).read_text() for f in ("scenarios.json", "questions.json", "identifiers.json")).lower()
    assert "onrender" not in raw and "postgres://" not in raw
    s1 = next(s for s in sc if s["id"] == "s01_orchestration")
    titles = " ".join(d["title"].lower() for d in s1["docs"])
    assert "reinforcement" in titles and "container" in titles
    assert any(s.get("identifier_url", "").startswith("https://hdl.handle.net/") for s in sc)


def test_replay_runs_and_halves_partition_all():
    rep = {s: R.replay(s) for s in ("calibration", "holdout", "all")}
    for key in ("questions", "scenarios", "identifiers"):
        assert rep["calibration"]["n"][key] + rep["holdout"]["n"][key] == rep["all"]["n"][key]
    a = rep["all"]["flags"]
    assert set(a) == {"admission", "source_type_router", "handle_identifier", "subquery_search", "subquery_adversarial"}
    assert rep["all"]["label"] == "offline fixture baseline, not live"
    assert a["admission"]["on_topic_recall"]["n"] >= 9
    cm = a["source_type_router"]["confusion_label_by_route"]
    assert sum(sum(r.values()) for r in cm.values()) == 75 - 2  # 75 questions, 2 ambiguous excluded
    assert set(rep["all"]["ambiguous"]["questions"]) == {"q027", "q033"}
    assert len(rep["all"]["ambiguous"]["passages"]) == 2
    assert "synthetic fixtures; real-data calibration" in rep["all"]["caveat"]


def test_scenarios_share_half_with_their_question_and_clusters_are_grouped():
    qs = {q["id"]: q for q in json.loads((FIX / "questions.json").read_text())["questions"]}
    for s in json.loads((FIX / "scenarios.json").read_text())["scenarios"]:
        assert s["split"] == qs[s["question_id"]]["split"] and s["question"] == qs[s["question_id"]]["text"]
    by = {}
    for q in qs.values():
        by.setdefault(q["cluster"], set()).add(q["split"])
    assert all(len(v) == 1 for v in by.values())
    assert qs["q001"]["cluster"] == qs["q004"]["cluster"] == qs["q018"]["cluster"]
    assert qs["q049"]["cluster"] == qs["q051"]["cluster"]


def test_override_changes_admission():
    base = R.replay("all")["flags"]["admission"]["off_topic_admit_rate"]
    strict = R.replay("all", ["admission.min_relevance=0.99"])["flags"]["admission"]["off_topic_admit_rate"]
    assert strict["k"] <= base["k"]


def test_cli_runs_offline():
    p = subprocess.run([sys.executable, str(Path(R.__file__)), "--split", "holdout", "--json"],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["split"] == "holdout"
