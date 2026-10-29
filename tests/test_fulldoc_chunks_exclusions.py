"""fulldoc_chunks source kind (empty sources.content, text in evidence_spans) must never shadow or leak. Offline: fake engines."""
import json
from pathlib import Path

from vera import answer_contract, corpus_description, dar_store, memory_store
from vera.dar_store import DarReader
from vera.memory_store import RoMemoryReader

KIND = "fulldoc_chunks"
ROOT = Path(__file__).resolve().parent.parent


class _Res:
    def __init__(self, rows): self.rows = rows
    def fetchall(self): return list(self.rows)
    def first(self): return self.rows[0] if self.rows else None
    def one(self): return self.rows[0] if self.rows else (0, None)


class _Conn:
    def __init__(self, eng): self.eng = eng
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, stmt, params=None):
        self.eng.calls.append((str(stmt), dict(params or {})))
        return _Res([])


class _Eng:
    def __init__(self): self.calls = []
    def connect(self): return _Conn(self)
    begin = connect


def _norm(sql): return " ".join(sql.split())


def test_contract_lists_kind_as_not_acquired():
    cfg = json.loads((ROOT / "config" / "discovery_contract.json").read_text(encoding="utf-8"))
    assert KIND in cfg["acquisition"]["non_document_text_kinds"]
    assert all(k in cfg["acquisition"]["non_document_text_kinds"] for k in ("abstract_metadata", "abstract", "snippet", "metadata", "index_record"))


def test_memory_source_filters_kind():
    eng = _Eng()
    assert RoMemoryReader(eng, {}).source("https://x/y") is None
    sql, p = eng.calls[0]
    assert "stored_text_kind" in sql and "<> :fd_kind" in sql and "ORDER BY s.version DESC" in sql
    assert p == {"u": "https://x/y", "fd_kind": KIND}


def test_dar_source_id_for_filters_kind():
    eng = _Eng()
    assert DarReader(eng).source_id_for(3) is None
    sql, p = eng.calls[0]
    assert "stored_text_kind" in sql and "<> :fd_kind" in sql
    assert p["fd_kind"] == KIND and p["c"] == 3


def test_dar_domain_chunks_max_version_ignores_kind():
    eng = _Eng()
    DarReader(eng).load_snapshot(["abstract_metadata"], span_text="x")
    chunk = [(s, p) for s, p in eng.calls if "FROM vera_vjay.domain_chunks" in s or "domain_chunks dk" in s]
    assert chunk, "domain_chunks query not issued"
    sql, p = _norm(chunk[0][0]), chunk[0][1]
    sub = sql.split("SELECT max(v.version)")[1].split(")")[0] if "SELECT max(v.version)" in sql else ""
    assert "v.provenance->>'stored_text_kind'" in sql and "<> :fd_kind" in sql and "v.canonical_source_id" in sub + sql
    assert p["fd_kind"] == KIND


def test_snapshot_documents_exclude_kind_when_contract_kinds_passed():
    eng = _Eng()
    kinds = answer_contract.full_document_kinds()  # GAP-DBKIND-1028: fail-closed allow-list (fulldoc_chunks absent)
    DarReader(eng).load_snapshot(kinds, span_text="x")
    doc = [(s, p) for s, p in eng.calls if "DISTINCT ON" in s][0]
    assert KIND not in doc[1]["kinds"] and "s.provenance->>'stored_text_kind' = ANY(:kinds)" in doc[0] and "NOT (s.provenance" not in doc[0]


def test_corpus_description_title_sample_excludes_kind():
    eng = _Eng()
    try:
        corpus_description.stamp_and_profile(eng, 5)
    except Exception:
        pass  # later fake results are not shaped for the rest of the function; only the first calls matter
    sql, p = eng.calls[1] if len(eng.calls) > 1 else eng.calls[0]
    assert "<> :fd_kind" in sql and "stored_text_kind" in sql and "md5(s.source_id::text)" in sql
    assert p["fd_kind"] == KIND and p["n"] == 5


def test_constants_agree():
    assert memory_store.FULLDOC_CHUNKS_KIND == dar_store.FULLDOC_CHUNKS_KIND == corpus_description.FULLDOC_CHUNKS_KIND == KIND
