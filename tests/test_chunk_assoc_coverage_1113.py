"""WP-C (CHUNK-ASSOC-1113): coverage report logic and static read-only checks on the query. Offline; no DB, no network.

Claimed here: the verdict logic (PENDING-CAL / PASS / FAIL), the subtraction rule, the limits handling and the static properties of
sql/chunk_assoc_coverage.sql. NOT run here: the SQL itself (no Postgres is available offline); its per-class behaviour (T17 negative
probes: whitespace span, span after the snapshot, link to a proposed or retired node, routing-only link, topic-only link; T41 scan_complete 0)
is verified only by the owner-run fixture rehearsal. Fixtures use the JSON shape the query prints.
"""
from __future__ import annotations

import copy
import re
import shutil
from pathlib import Path

import pytest

from scripts import chunk_assoc_coverage as cov

ROOT = Path(__file__).resolve().parents[1]
SQL = (ROOT / "sql" / "chunk_assoc_coverage.sql").read_text(encoding="utf-8")
MAXABS = 0.3
LIM = {"age_limit_hours": 48.0, "covered_floor": 0.5, "covered_basis": "stored_or_derived"}
UNSET = {"age_limit_hours": None, "covered_floor": None, "covered_basis": "stored_or_derived"}
ZERO = {f"skipped_{k}": 0 for k in cov.EXCLUSIONS} | {f"excluded_with_vector_{k}": 0 for k in cov.EXCLUSIONS}


def row(run_id=7, counts=None, age=2.0, state="finished"):
    return {"run_id": run_id, "state": state, "age_hours": age, "counts": dict(ZERO if counts is None else counts)}


def raw(classes, build="default", scan="default", with_vector=None, ages=None):
    n = sum(classes.values())
    return {"spans": n, "with_vector": n - classes.get("no_vector_yet", 0) if with_vector is None else with_vector,
            "sources_total": 3, "sources_with_covered": 2, "classes": dict(classes), "age_max_hours": ages or {},
            "build": row() if build == "default" else build, "scan": row() if scan == "default" else scan}


def run(r, lim=LIM, pend=()):
    return cov.evaluate(r, lim, list(pend), MAXABS)


FULL = {"covered_all_derived": 8, "partial_depth": 1, "abstained": 1}


def test_t18_fully_covered_with_limits_set_passes():
    res = run(raw(FULL))
    assert res["verdict"] == "PASS" and res["fail"] == [] and res["pending"] == []
    assert res["shares"]["covered_derived"] == 0.8


def test_t18_never_pass_while_limits_unset():
    res = cov.evaluate(raw(FULL), UNSET, ["age_limit_hours unset (OD-J)", "covered_floor unset (OD-J)"], MAXABS)
    assert res["verdict"] == "PENDING-CAL"


def test_t18_all_no_vector_yet_limits_unset_is_pending_not_pass():
    res = cov.evaluate(raw({"no_vector_yet": 10}, ages={"no_vector_yet": 1.0}), UNSET, ["age_limit_hours unset (OD-J)"], MAXABS)
    assert res["verdict"] == "PENDING-CAL" and res["verdict"] != "PASS"


def test_all_no_vector_yet_with_limits_fails_the_covered_floor():
    res = run(raw({"no_vector_yet": 10}, ages={"no_vector_yet": 1.0}))
    assert res["verdict"] == "FAIL" and any("covered_floor" in f for f in res["fail"])


def test_t18_unexplained_fails_even_with_limits_unset():
    res = cov.evaluate(raw({"covered_all_derived": 5, "unexplained": 1}, build=None), UNSET, ["covered_floor unset (OD-J)"], MAXABS)
    assert res["verdict"] == "FAIL" and any("unexplained" in f for f in res["fail"])


def test_zero_links_zero_vectors_cannot_pass():
    # the S4 case: every chunk waits for a vector; unexplained and abstained are 0, yet the verdict is not PASS
    for lim, pend in ((LIM, ()), (UNSET, ("age_limit_hours unset (OD-J)",))):
        assert cov.evaluate(raw({"no_vector_yet": 5250}), lim, list(pend), MAXABS)["verdict"] != "PASS"


def test_age_limit_fails_old_waiting_chunks():
    res = run(raw({"covered_all_derived": 9, "no_vector_yet": 1}, ages={"no_vector_yet": 100.0}))
    assert res["verdict"] == "FAIL" and any("no_vector_yet older" in f for f in res["fail"])
    assert run(raw({"covered_all_derived": 9, "no_vector_yet": 1}, ages={"no_vector_yet": 3.0}))["verdict"] == "PASS"


def test_abstained_share_over_display_gate_fails_and_denominator_is_vector_bearing():
    res = run(raw({"covered_all_derived": 6, "abstained": 4}))  # 4 of 10 vector-bearing
    assert res["verdict"] == "FAIL" and any("max_abstain_share" in f for f in res["fail"])
    # 4 abstained of 20 vector-bearing is 0.2, under the gate, even though many chunks wait for vectors
    ok = run(raw({"covered_all_derived": 16, "abstained": 4, "no_vector_yet": 0}))
    assert ok["shares"]["abstained"] == 0.2


def test_partial_depth_is_reported_not_failing():
    res = run(raw({"covered_all_derived": 6, "partial_depth": 4}))
    assert res["verdict"] == "PASS" and res["shares"]["partial_depth"] == 0.4


def test_stored_basis_does_not_count_derived_chunks():
    stored = dict(LIM, covered_basis="stored")
    assert run(raw({"covered_all_derived": 10}), stored)["verdict"] == "FAIL"
    assert run(raw({"covered_all_stored": 10}), stored)["verdict"] == "PASS"
    assert run(raw({"covered_all_stored": 5, "covered_all_derived": 5}))["shares"]["covered_stored"] == 0.5


def test_t37_deferred_scan_row_counts_are_read_and_age_printed():
    c = dict(ZERO, skipped_portal=2, excluded_with_vector_portal=1)
    r = raw({"covered_all_derived": 7, "no_vector_yet": 1, "abstained": 2}, build=None, scan=row(4, c, age=31.5, state="deferred"))
    res = run(r)
    assert res["excluded"]["portal"] == 2 and res["scan_state"] == "deferred" and res["scan_age_hours"] == 31.5
    assert "complete-scan row age h=31.5" in cov.render(res)


def test_t40_portal_span_with_vector_counted_once_and_abstained_not_inflated():
    # 10 spans: 7 covered, 1 portal span holding a vector (SQL put it in abstained after the build), 2 true abstained
    c = dict(ZERO, skipped_portal=1, excluded_with_vector_portal=1)
    res = run(raw({"covered_all_derived": 7, "abstained": 3}, build=row(7, c), scan=row(7, c)))
    assert res["eligible"] == 9 and res["classes"]["abstained"] == 2 and sum(res["classes"].values()) == 9
    assert res["excluded"]["portal"] == 1
    assert res["shares"]["abstained"] == pytest.approx(2 / 9)  # denominator: 9 vector-bearing eligible


def test_excluded_without_vector_leave_no_vector_yet():
    c = dict(ZERO, skipped_empty=2)
    res = run(raw({"covered_all_derived": 6, "no_vector_yet": 2, "abstained": 0}, build=row(7, c), scan=row(7, c)))
    assert res["classes"]["no_vector_yet"] == 0 and res["eligible"] == 6 and res["verdict"] == "PASS"


def test_t41_no_complete_scan_row_is_pending():
    res = run(raw(FULL, build=None, scan=None))
    assert res["verdict"] == "PENDING-CAL" and any("complete-scan" in p for p in res["pending"])


def test_t41_subtraction_goes_to_unexplained_before_first_build():
    c = dict(ZERO, skipped_portal=1, excluded_with_vector_portal=1)
    res = run(raw({"covered_all_derived": 5, "unexplained": 2}, build=None, scan=row(4, c)))
    assert res["classes"]["unexplained"] == 1 and res["classes"]["abstained"] == 0 and res["verdict"] == "FAIL"


def test_t41_subtraction_goes_to_abstained_after_a_build():
    c = dict(ZERO, skipped_portal=1, excluded_with_vector_portal=1)
    res = run(raw({"covered_all_derived": 9, "abstained": 1, "unexplained": 0}, build=row(7, c), scan=row(7, c)))
    assert res["classes"]["abstained"] == 0 and res["classes"]["unexplained"] == 0 and res["verdict"] == "PASS"


def test_t41_negative_class_after_subtraction_fails():
    c = dict(ZERO, skipped_portal=3, excluded_with_vector_portal=3)
    res = run(raw({"covered_all_derived": 7, "abstained": 1, "no_vector_yet": 0}, build=row(7, c), scan=row(7, c), with_vector=8))
    assert res["verdict"] == "FAIL" and any("negative" in f for f in res["fail"])


def test_t43_scan_counts_differing_from_build_counts_is_pending():
    b = row(7, dict(ZERO, skipped_portal=1, excluded_with_vector_portal=1))
    s = row(9, dict(ZERO, skipped_portal=2, excluded_with_vector_portal=1))
    res = run(raw({"covered_all_derived": 8, "abstained": 2}, build=b, scan=s))
    assert res["verdict"] == "PENDING-CAL" and any("differ" in p for p in res["pending"])
    assert res["excluded"]["portal"] == 1  # counts come from the build row (W4), not the later scan


def test_run_row_without_job_keys_is_pending():
    res = run(raw(FULL, build={"run_id": 7, "age_hours": 1, "counts": {}}, scan={"run_id": 7, "state": "finished", "age_hours": 1, "counts": {}}))
    assert res["verdict"] == "PENDING-CAL" and any("job-counted" in p for p in res["pending"])


def test_job_abstain_count_mismatch_fails_only_for_the_same_row():
    same = row(7, dict(ZERO, abstained=5))
    assert run(raw({"covered_all_derived": 8, "abstained": 2}, build=row(7), scan=same))["verdict"] == "FAIL"
    other = row(8, dict(ZERO, abstained=5))
    assert run(raw({"covered_all_derived": 8, "abstained": 2}, build=row(7), scan=other))["verdict"] == "PASS"


def test_malformed_input_is_error_not_a_verdict():
    bad = raw(FULL)
    bad["classes"]["covered_all_derived"] = 7  # does not sum to spans
    with pytest.raises(cov.CoverageError):
        run(bad)
    bad2 = raw(FULL)
    bad2["classes"]["mystery"] = 1
    with pytest.raises(cov.CoverageError):
        run(bad2)
    neg = raw(FULL, build=row(7, dict(ZERO, skipped_portal=1, excluded_with_vector_portal=2)))
    with pytest.raises(cov.CoverageError):
        run(neg)
    with pytest.raises(cov.CoverageError):
        cov.parse_raw("no json here")


def test_limits_validation_and_calibration_note(tmp_path_factory=None):
    d = ROOT / "tests" / "_cov_tmp_1113"
    d.mkdir(exist_ok=True)
    try:
        f = d / "limits.json"
        f.write_text('{"age_limit_hours": 24, "covered_floor": 0.9}', encoding="utf-8")
        lim, pend = cov.load_limits(f)
        assert lim["age_limit_hours"] == 24.0 and any("calibration_note" in p for p in pend)
        f.write_text('{"age_limit_hours": 24, "covered_floor": 0.9, "calibration_note": "n"}', encoding="utf-8")
        assert cov.load_limits(f)[1] == []
        f.write_text('{"age_limit_hours": null, "covered_floor": null}', encoding="utf-8")
        assert len(cov.load_limits(f)[1]) == 2
        for bad in ('{"age_limit_hours": true}', '{"age_limit_hours": -1}', '{"covered_floor": 1.5}', '{"covered_floor": "0.9"}',
                    '{"covered_basis": "x"}', "[1]"):
            f.write_text(bad, encoding="utf-8")
            with pytest.raises(cov.CoverageError):
                cov.load_limits(f)
        assert len(cov.load_limits(d / "absent.json")[1]) == 2  # absent file: unset
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_cli_exit_codes_and_params(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    d = ROOT / "tests" / "_cov_tmp_1113"
    d.mkdir(exist_ok=True)
    try:
        import json
        f = d / "raw.json"
        f.write_text(json.dumps(raw(FULL)) + "\n", encoding="utf-8")
        assert cov.main(["report", "--raw", str(f.relative_to(ROOT))]) == 3  # no limits file: PENDING-CAL
        assert "VERDICT: PENDING-CAL" in capsys.readouterr().out
        lim = d / "limits.json"
        lim.write_text('{"age_limit_hours": 48, "covered_floor": 0.5, "calibration_note": "x"}', encoding="utf-8")
        assert cov.main(["report", "--raw", str(f.relative_to(ROOT)), "--limits", str(lim.relative_to(ROOT))]) == 0
        f.write_text("garbage\n", encoding="utf-8")
        assert cov.main(["report", "--raw", str(f.relative_to(ROOT))]) == 2
        assert "VERDICT: ERROR" in capsys.readouterr().out.splitlines()[-1]
        assert cov.main(["report", "--raw", "/etc/hostname"]) == 2  # outside the working directory
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert cov.main(["params"]) == 0
    out = capsys.readouterr().out
    assert "-v cn=chunk_taxo_auto" in out and "-v model=@cf/baai/bge-m3" in out and "-v text_chars=8000" in out


def test_config_gate_value_is_reused_not_a_new_number():
    cfg = cov.load_config(ROOT / "config" / "chunk_taxo.json")
    assert cfg["max_abstain_share"] == 0.3


# ---- static checks on the SQL (T19 and the read-only guarantee) ----
def _code(sql: str) -> str:
    return "\n".join(ln for ln in sql.splitlines() if not ln.lstrip().startswith("--"))


def test_sql_is_read_only_select():
    code = _code(SQL)
    assert "BEGIN TRANSACTION READ ONLY" in code
    for kw in ("INSERT", "UPDATE", "DELETE", "TRUNCATE", "CREATE", "ALTER", "DROP", "GRANT", "REVOKE", "COPY", "VACUUM", "LOCK ", "CALL "):
        assert not re.search(rf"\b{kw.strip()}\b(?!_)", re.sub(r"has_table_privilege\([^)]*\)", "", code.replace("'INSERT'", "").replace("'UPDATE'", "").replace("'DELETE'", ""))), kw
    assert not re.search(r"^\s*\\(!|o|copy|i|include)(\b|$|!)", code, re.M | re.I)
    assert "SET LOCAL" in code and "statement_timeout" in code


def test_t19_sql_refuses_a_writable_login():
    guard = SQL[SQL.index("DO $g$"):SQL.index("END $g$")]
    assert "not the read-only login" in guard and "current_database() <> 'vera_dar_db'" in guard
    for need in ("vera_vjay.sources', 'INSERT'", "tax_job_runs', 'INSERT'", "tax_chunk_vectors', 'INSERT'", "tax_chunk_subtopic', 'UPDATE'"):
        assert need in guard


def test_sql_counting_rule_reader_predicate():
    code = _code(SQL)
    assert code.count("state = 'active'") >= 8  # links, entities and pairs
    assert "'cosine', 'cosine_lvl'" in code and "cosine_r" not in code and "cosine_soft" not in code
    for t in ("tax_topics", "tax_subtopics", "tax_topic_subtopic", "tax_domain_topic", "tax_domains"):
        assert f"vera_vjay.{t}" in code
    assert ":'cn'" in code and "classifier_name = :'cn'" in code


def test_sql_selects_no_text_title_url_or_label():
    out = SQL[SQL.index("SELECT json_build_object("):]
    assert not re.search(r"(?<!:)\b(text|title|url|label|slug|description)\b", out)


def test_sql_stale_hash_matches_the_job():
    assert "left(e.text, :text_chars)" in SQL and "convert_to" in SQL and "'UTF8'" in SQL and "sha256(" in SQL


def test_sql_emits_every_class_the_script_knows():
    for k in cov.CLASSES:
        if k != "covered_all_stored":  # stored needs a chunk-domain source (WP-D); the script treats it as 0
            assert f"'{k}'" in SQL
