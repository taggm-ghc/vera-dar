"""C4 contract test: every column named in SQL under vera/ and sql/ exists on its live table.

The live-catalog check moved to scripts/live_catalog_check.py (owner OD-S1, 2026-10-16); pytest never connects to
production. See sql_contract_support for extractor limits. The extractor unit tests below need no DB.
"""

from tests import sql_contract_support as sc


def test_extractor_unit():
    cols = sc.extract_columns("SELECT a.x, a.y FROM {S}.appraisals a JOIN {S}.sources s ON s.source_id=a.source_id")
    assert {("appraisals", "x"), ("appraisals", "y"), ("sources", "source_id")} <= cols
    cols = sc.extract_columns("UPDATE {S}.runs SET baseline_response=:b, eval_metrics=CAST(:m AS jsonb) WHERE run_id=:r")
    assert {("runs", "baseline_response"), ("runs", "eval_metrics"), ("runs", "run_id")} <= cols
    cols = sc.extract_columns("INSERT INTO {S}.answers (run_id, stage) VALUES (:r,:s) RETURNING answer_id")
    assert {("answers", "run_id"), ("answers", "stage"), ("answers", "answer_id")} <= cols
    assert ("appraisals", "bogus_col") in sc.extract_columns("SELECT bogus_col FROM vera_vjay.appraisals WHERE source_id=:s")


def test_extractor_finds_statements():
    found = sc.collect()
    assert len(found) > 40, f"extractor found only {len(found)} (table,column) pairs; parser regression?"
    assert ("appraisals", "rubric_version") in found
