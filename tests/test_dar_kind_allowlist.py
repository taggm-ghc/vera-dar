"""GAP-DBKIND-1028: the DAR document reader and DarCorpus gate use a fail-closed stored_text_kind ALLOW-list
(config acquisition.full_document_text_kinds = ["full_text"]). Synthetic; no DB, network or env."""
import logging

import pytest

from vera import answer_contract
from vera.dar_corpus import DarCorpus
from vera.dar_store import DarReader
from tests.test_dar_corpus import doc, make_cfg
from tests.test_dar_store import FakeEngine

ROWS = {  # kind -> stored_text_kind of a fixture sources row (None = NULL provenance key)
    "full_text": "full_text", "full_content": "full_content", "link_metadata": "link_metadata",
    "abstract_v2": "abstract_v2", "fulldoc_chunks": "fulldoc_chunks", "null": None}


def _engine():
    """Fake DB that applies the bound allow-list the way the SQL does: kind = ANY(:kinds) (NULL never matches)."""
    def handler(sql, p):
        if "DISTINCT ON" in sql:
            kinds = p["kinds"]
            return [(f"k_{n}", i + 1, i + 1, "text" if k != "fulldoc_chunks" else "", "u", None, "T", "2024-05-01",
                     "journal", "2024-06-01", k) for i, (n, k) in enumerate(ROWS.items()) if k is not None and k in kinds]
        return []
    return FakeEngine(handler)


def _loaded(kinds):
    eng = _engine()
    snap = DarReader(eng, schema="s1").load_snapshot(kinds)
    sql, params = next(c for c in eng.calls if "DISTINCT ON" in c[0])
    return snap["documents"], sql, params


def test_reader_sql_is_allow_list_with_bound_params():
    docs, sql, params = _loaded(["full_text"])
    assert "s.provenance->>'stored_text_kind' = ANY(:kinds)" in sql and "NOT (" not in sql
    assert params == {"kinds": ["full_text"]}
    assert [d["stored_text_kind"] for d in docs] == ["full_text"]


def test_reader_skips_unknown_kinds_and_null_and_chunks():
    docs, _, _ = _loaded(answer_contract.full_document_kinds())
    got = {d["source_key"] for d in docs}
    assert got == {"k_full_text"}
    for bad in ("full_content", "link_metadata", "abstract_v2", "fulldoc_chunks", "null"):
        assert f"k_{bad}" not in got


def test_reader_empty_allow_list_loads_nothing():
    docs, _, params = _loaded([])
    assert docs == [] and params == {"kinds": []}
    assert _loaded(None)[0] == []


def test_shipped_config_allow_list():
    assert answer_contract.full_document_kinds() == ["full_text"]
    assert "fulldoc_chunks" not in answer_contract.full_document_kinds()


@pytest.mark.parametrize("kind", ["full_content", "link_metadata", "abstract_v2", "fulldoc_chunks", None, "", "FULL_TEXT"])
def test_gate_rejects_non_allowed_kinds(kind):
    c = DarCorpus(make_cfg())
    r = c.ingest_document(doc("g", "reef", 0, kind=kind))
    assert r["status"] == "skipped" and r["reason"] == "not_full_document"
    assert c.stats()["documents"] == 0


def test_gate_admits_full_text():
    c = DarCorpus(make_cfg())
    assert c.ingest_document(doc("ok", "reef", 0, kind="full_text"))["status"] != "skipped"
    assert c.stats()["documents"] == 1


@pytest.mark.parametrize("content", ["{}", '{"acquisition": {"full_document_text_kinds": []}}',
                                     '{"acquisition": {"full_document_text_kinds": "full_text"}}',
                                     '{"acquisition": {"full_document_text_kinds": [1]}}', "not json"])
def test_missing_or_invalid_config_fails_closed(tmp_path, content, caplog):
    p = tmp_path / "c.json"
    p.write_text(content)
    with caplog.at_level(logging.ERROR):
        assert answer_contract.full_document_kinds(p) == []
    assert "failing closed" in caplog.text
    cfg = make_cfg()
    cfg["corpus"]["contract_config_path"] = str(p)
    c = DarCorpus(cfg)
    assert c.ingest_document(doc("m", "reef", 0, kind="full_text"))["reason"] == "not_full_document"
    assert c.stats()["documents"] == 0


def test_absent_config_file_fails_closed(tmp_path):
    assert answer_contract.full_document_kinds(tmp_path / "nope.json") == []
