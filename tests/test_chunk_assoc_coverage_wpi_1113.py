"""WP-I closes WP-C item V-a (CHUNK-ASSOC-1113 A5): a malformed section of the query output (a non-object classes, ages_desc, build or scan)
is an ERROR verdict with exit 1, never an uncaught exception. Helpers come from tests/test_chunk_assoc_coverage_1113.py (imported, not edited).
No DB, no network."""
from __future__ import annotations

import json

import pytest

from scripts import chunk_assoc_coverage as cov
from tests.test_chunk_assoc_coverage_1113 import FULL, LIM, MAXABS, raw


@pytest.mark.parametrize("key", ["classes", "ages_desc", "build", "scan"])
@pytest.mark.parametrize("bad", [[1, 2], "text", 7, [{"a": 1}]])
def test_va_non_object_section_raises_coverage_error(key, bad):
    r = raw(FULL)
    r[key] = bad
    with pytest.raises(cov.CoverageError):
        cov.evaluate(r, LIM, [], MAXABS)


def test_va_main_prints_error_verdict_and_exits_with_the_error_code(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "chunk_taxo.json").write_text(json.dumps({"display_gate": {"max_abstain_share": 0.3}, "cloudflare": {"model": "@cf/baai/bge-m3"}}))
    bad = raw(FULL)
    bad["ages_desc"] = ["x"]  # was an AttributeError (exit 1 with a traceback)
    (tmp_path / "out.json").write_text(json.dumps(bad) + "\n")
    rc = cov.main(["report", "--raw", "out.json"])
    out = capsys.readouterr().out
    assert rc == cov.EXIT["ERROR"] and "VERDICT: ERROR" in out and "Traceback" not in out


def test_va_well_formed_output_is_unchanged():
    assert cov.evaluate(raw(FULL), LIM, [], MAXABS)["verdict"] == "PASS"
