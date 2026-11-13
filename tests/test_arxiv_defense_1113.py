"""ARXIV-DEFENSE-1113: portal (arXiv) records never leave for embedding, the memory gate refuses them, and the two flag-off
readers (memory evidence recall, DAR stored branch) apply the corpus-load portal check. Offline: FakeDB and helpers of
test_chunk_taxo_job_1109.py, FakeEngine of test_dar_store.py; real record shapes (doi.org canonical URLs, acquisition.arxiv.id,
fulltext_url); reserved values only (example.org, 10.1000 DOIs, synthetic arXiv ids). No DB, no network."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pytest

from tests.test_chunk_taxo_job_1109 import FakeDB, FakeEmb, SENTINEL_TEXT, make
from tests.test_dar_store import FakeEngine
from vera import chunk_taxo_job as job
from vera import chunk_taxo_store as store
from vera import fulltext_loader
from vera import memory_gate as mg
from vera import memory_store as ms
from vera.dar_store import DarReader
from vera.memory_gate import SourceRef, check_memory_write, load_gate_config

PORTAL_TAG = "PORTALSPANBODY"
OKTAG = "KEEPSPANBODY"
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=x"

# source id -> (canonical_url, provenance), real shapes
PORTAL_SOURCES = {
    201: ("https://doi.org/10.48550/arxiv.2101.00001", {"document_id": "acq-201", "item": {"doi": "10.48550/arXiv.2101.00001"}}),
    202: ("https://doi.org/10.1000/example.202",
          {"document_id": "acq-202", "acquisition": {"arxiv": {"id": "2101.00002", "abs_url": "https://arxiv.org/abs/2101.00002"}}}),
    203: ("https://doi.org/10.1000/example.203",
          {"document_id": "acq-203", "item": {"fulltext_url": "https://arxiv.org/pdf/2101.00003"}, "fulltext_url": "https://arxiv.org/pdf/2101.00003"}),
    204: (EPMC_SEARCH, {"document_id": "acq-204", "url": EPMC_SEARCH}),
}
OK_SOURCES = {
    301: ("https://doi.org/10.1000/example.1",
          {"document_id": "acq-301", "item": {"source": "MED", "fulltext_url": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1234567/fullTextXML"},
           "fulltext_url": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1234567/fullTextXML"}),
    302: ("https://example.org/papers/arxiv-survey", {"document_id": "acq-302", "title": "An arxiv survey of sparse attention"}),
    303: ("https://doi.org/10.1000/48550", {"document_id": "acq-303", "item": {"doi": "10.1000/48550"}}),
}


class RecEmb(FakeEmb):
    def __init__(self):
        super().__init__()
        self.texts = []

    def embed(self, texts):
        self.texts.extend(texts)
        return super().embed(texts)


def mixed_spans():
    """Spans 1..7; the LAST span (7) belongs to a portal source."""
    sids = [301, 201, 302, 202, 303, 203, 204]
    return [(i + 1, s, f"alpha {PORTAL_TAG if s in PORTAL_SOURCES else OKTAG} {i + 1}") for i, s in enumerate(sids)]


def mixed_db(spans=None):
    db = FakeDB(spans or mixed_spans())
    db.sources = {**PORTAL_SOURCES, **OK_SOURCES}
    return db


# ------------------------------------------------------------------------------------------------------ (a) taxonomy job
def test_portal_spans_never_embedded_or_labelled():
    db, e = mixed_db(), RecEmb()
    j = make(db, e=e, page_size=3, tick_row_cap=100)
    r = j.run()
    assert r["state"] != "error"
    assert e.texts and not [t for t in e.texts if PORTAL_TAG in t]  # nothing of a portal span reaches the embedder
    assert sum(OKTAG in t for t in e.texts) == 3
    portal_ids = {2, 4, 6, 7}
    assert not portal_ids & set(db.vectors)
    assert {1, 3, 5} <= set(db.vectors)
    for k in db.links:
        if k[0] in ("chunk_subtopic", "chunk_topic"):
            assert k[1] not in portal_ids
    assert j.skipped_portal == 4 and j.skipped_portal_error == 0 and j.skipped_no_source == 0
    assert j.vectors_missing == 0
    assert j._cursor == 7  # the page cursor reached the last span although the last span of a page is a portal span
    assert "sources_portal" in db.names()


def test_stored_vector_of_portal_span_not_reused():
    db = mixed_db()
    db.vectors[2] = ("fake", "stale", [1.0, 0.0])
    e = RecEmb()
    make(db, e=e).run()
    assert not [t for t in e.texts if PORTAL_TAG in t]
    assert all(k[1] != 2 for k in db.links if k[0] in ("chunk_subtopic", "chunk_topic"))


def test_lookalikes_are_not_refused():
    db = mixed_db([(1, 302, f"alpha {OKTAG} 1"), (2, 303, f"beta {OKTAG} 2"), (3, 301, f"alpha {OKTAG} 3")])
    j = make(db, e=RecEmb())
    j.run()
    assert j.skipped_portal == 0 and j.skipped_portal_error == 0 and {1, 2, 3} <= set(db.vectors)


def test_missing_source_row_counted_separately():
    db = mixed_db([(1, 301, f"alpha {OKTAG} 1"), (2, 999, f"alpha {OKTAG} 2")])
    db.sources[999] = None
    e = RecEmb()
    j = make(db, e=e)
    j.run()
    assert j.skipped_no_source == 1 and j.skipped_portal == 0 and 2 not in db.vectors and 1 in db.vectors
    assert not [t for t in e.texts if " 2" in t.split(OKTAG)[-1][:3]]


def test_unparseable_provenance_and_raising_verdict_fail_closed(monkeypatch):
    db = mixed_db([(1, 301, f"alpha {OKTAG} 1"), (2, 305, f"alpha {PORTAL_TAG} 2")])
    db.sources[305] = ("https://doi.org/10.1000/example.305", "{not json")
    e = RecEmb()
    j = make(db, e=e)
    j.run()
    assert j.skipped_portal_error == 1 and not [t for t in e.texts if PORTAL_TAG in t] and 2 not in db.vectors
    # a raising portal_verdict is an error verdict, never an embed
    db2 = mixed_db([(1, 301, f"alpha {PORTAL_TAG} 1")])
    monkeypatch.setattr(fulltext_loader, "portal_verdict", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    e2 = RecEmb()
    j2 = make(db2, e=e2)
    j2.run()
    assert j2.skipped_portal_error == 1 and not e2.texts
    monkeypatch.setattr(fulltext_loader, "portal_verdict", lambda *a, **k: "internal_error")
    j3 = make(mixed_db([(1, 301, f"alpha {PORTAL_TAG} 1")]), e=RecEmb())
    j3.run()
    assert j3.skipped_portal_error == 1


def test_none_provenance_still_runs_url_checks():
    v = job._source_verdicts([(1, "https://doi.org/10.48550/arxiv.2101.00001", None), (2, "https://doi.org/10.1000/example.2", None)])
    assert v[1] not in (None, "error") and v[2] is None
    assert job._source_verdicts([(3, "https://doi.org/10.1000/example.3", [1])]) == {3: "error"}


def test_failed_lookup_goes_to_error_path_and_embeds_nothing():
    db = mixed_db()
    db.fail_on = "sources_portal"
    e = RecEmb()
    j = make(db, e=e)
    assert j.run()["state"] == "error" and not e.texts


def test_robust_info_carries_counters_as_ints():
    db = mixed_db()
    j = make(db, e=RecEmb())
    j.run()
    info = j._robust_info()
    for k in ("skipped_portal", "skipped_portal_error", "skipped_no_source"):
        assert isinstance(info[k], int)
    assert info["skipped_portal"] == 4
    assert "PORTAL" not in json.dumps(info) and PORTAL_TAG not in json.dumps(info)


def test_counters_reset_per_pass_and_log_is_count_only(caplog):
    caplog.set_level(logging.INFO, logger="vera.chunk_taxo")
    j = make(mixed_db(), e=RecEmb())
    j.run()
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "portal=4" in msgs and PORTAL_TAG not in msgs and "arxiv" not in msgs.lower()
    j._reset_run()
    assert j.skipped_portal == 0 and j._verdicts == {}


def test_store_statement_is_allowlisted_read_only():
    sql = store.SQL["sources_portal"]
    assert sql.startswith("SELECT") and "vera_vjay.sources" in sql and "vera_vjay.canonical_sources" in sql
    assert all(w not in sql.upper() for w in ("INSERT", "UPDATE", "DELETE", "TRUNCATE"))


# ------------------------------------------------------------------------------------------------------ (b) memory gate
CFG = load_gate_config()
GOOD = "Sparse attention reduces memory cost on long sequences while retaining accuracy on benchmarks."
Q = "How does sparse attention affect long documents?"


def gate(src):
    return check_memory_write("cited_finding", GOOD, "supported", (src,), Q, CFG)


@pytest.mark.parametrize("src", [
    SourceRef("arxiv", "allow", "10.1000/example.1"),
    SourceRef("openalex+arxiv", "allow", "10.1000/example.1"),
    SourceRef(" ARXIV ", "allow", "10.1000/example.1"),
    SourceRef("openalex", "allow", "arXiv:2101.00001"),
    SourceRef("openalex", "allow", "arxiv.org/abs/2101.00001"),
    SourceRef("openalex", "allow", "10.48550/arXiv.2101.00001"),
    SourceRef("openalex", "allow", "doi:10.48550/arXiv.2101.00001"),
    SourceRef("openalex", "allow", "10.1000/example.1", url="https://arxiv.org/abs/2101.00001"),
    SourceRef("openalex", "allow", "https://openalex.org/W1"),
    SourceRef("openalex", "allow", "x", url="https://openalex.org/W1"),
])
def test_gate_refuses_portal_citations(src):
    r = gate(src)
    assert not r.allowed and mg.PORTAL_SOURCE in r.reasons


def test_gate_allows_original_source_and_documents_bare_id_non_refusal():
    ok = SourceRef("openalex", "allow", "doi:10.1000/example.1", url="https://doi.org/10.1000/example.1")
    assert gate(ok).allowed
    bare = SourceRef("openalex", "allow", "2101.00001", url="https://example.org/paper")
    assert mg.PORTAL_SOURCE not in gate(bare).reasons  # documented: a bare new-style id is ambiguous and not refused


def test_gate_host_role_failure_refuses(monkeypatch):
    from vera import host_role
    monkeypatch.setattr(host_role, "role_for_record", lambda rec: (_ for _ in ()).throw(RuntimeError("registry")))
    r = gate(SourceRef("openalex", "allow", "doi:10.1000/example.1", url="https://doi.org/10.1000/example.1"))
    assert not r.allowed and mg.PORTAL_SOURCE in r.reasons


def test_gate_real_config_still_listing_arxiv_still_refuses():
    assert "arxiv" in CFG.allowed_providers  # until the config follow-up; the code floor must refuse regardless
    assert mg.PORTAL_SOURCE in gate(SourceRef("arxiv", "allow", "10.1000/example.1")).reasons


def test_all_reasons_unique_and_discovery_helpers_unchanged():
    assert len(mg.ALL_REASONS) == len(set(mg.ALL_REASONS)) and mg.PORTAL_SOURCE in mg.ALL_REASONS
    from vera.discovery_memory import eligible_candidates
    c = {"title": "T", "url": "https://arxiv.org/abs/2401.00001", "provider": "arxiv", "licence_decision": "allow", "identifier": "arXiv:2401.00001"}
    assert eligible_candidates([c], CFG) == [c]  # discovery notes keep today's behaviour (config-driven, floor not applied)


@pytest.mark.xfail(strict=True, reason="config follow-up (after CORPUS-UNCAT-1113 merges): remove arxiv from memory.allowed_providers")
def test_config_providers_hold_no_portal_provider():
    assert not set(CFG.allowed_providers) & mg.PORTAL_PROVIDERS


# ------------------------------------------------------------------------------------------------------ (c) memory recall
def reader_with(claims, srcs):
    """claims: list of 5-column rows; srcs: (claim_id, provenance, canonical_url) rows for the second read."""
    shipped = json.load(open(ms.__file__.replace("vera/memory_store.py", "config/ask-provider-chain.json")))["memory"]
    r = ms.RoMemoryReader(object(), shipped["store"])
    calls = []

    def rows(sql, p):
        calls.append(sql)
        if "count(DISTINCT" in sql:
            return [(0,)]
        if "c.claim_id = ANY(:ids)" in sql:
            return [s for s in srcs if s[0] in p["ids"]]
        return claims
    r._rows = rows
    r.calls = calls
    return r


T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
Q2 = "quantized inference latency transformer"


def test_recall_keeps_non_portal_and_drops_portal_claims():
    claims = [("Claim ok one.", T0, "T1", "https://doi.org/10.1000/example.1", 1),
              ("Claim mixed two.", T0, "T2", "https://doi.org/10.1000/example.2", 2),
              ("Claim arxiv id three.", T0, "T3", "https://doi.org/10.1000/example.3", 3)]
    srcs = [(1, {"document_id": "acq-1", "url": "https://doi.org/10.1000/example.1"}, "https://doi.org/10.1000/example.1"),
            (2, {"document_id": "acq-2"}, "https://doi.org/10.1000/example.2"),
            (2, {"document_id": "acq-9"}, "https://doi.org/10.48550/arxiv.2101.00001"),
            (3, {"document_id": "acq-3", "acquisition": {"arxiv": {"id": "2101.00003"}}}, "https://doi.org/10.1000/example.3")]
    r = reader_with(claims, srcs)
    out = r.recall(Q2)
    assert [o["claim_text"] for o in out] == ["Claim ok one."]
    assert set(out[0]) == {"claim_text", "source_title", "source_url", "saved_at"}


def test_recall_fails_closed_on_unparseable_provenance_and_second_read_error():
    claims = [("Claim ok one.", T0, "T1", "u", 1)]
    r = reader_with(claims, [(1, "{bad", "https://doi.org/10.1000/example.1")])
    assert r.recall(Q2) == []
    r2 = reader_with(claims, [])
    inner = r2._rows

    def boom(sql, p):
        if "c.claim_id = ANY(:ids)" in sql:
            raise RuntimeError("db")
        return inner(sql, p)
    r2._rows = boom
    with pytest.raises(RuntimeError):
        r2.recall(Q2)


# ------------------------------------------------------------------------------------------------------ (d) DAR stored branch
PORTAL_KEY = "https://doi.org/10.48550/arxiv.2101.00001"
OK_KEY = "https://doi.org/10.1000/example.1"


def doc_row(key, sid, prov):
    return (key, sid, sid + 100, "full text", "u", "10.1000/x", "T", "2024-05-01", "journal", "2024-06-01", "full_text", None, None,
            None, prov)


def test_dar_snapshot_drops_portal_document_and_associations():
    def ro(sql, p):
        if "DISTINCT ON" in sql:
            return [doc_row(PORTAL_KEY, 7, {"document_id": "acq-7", "item": {"doi": "10.48550/arXiv.2101.00001"}}),
                    doc_row(OK_KEY, 8, {"document_id": "acq-8", "item": {"doi": "10.1000/example.1"}})]
        if "domain_corpus" in sql:
            return [("alpha", PORTAL_KEY, "dar", T0), ("alpha", OK_KEY, "dar", T0)]
        if "domain_chunks" in sql:
            return [("alpha", PORTAL_KEY, 0, 4, "dar", T0, 5), ("alpha", OK_KEY, 0, 4, "dar", T0, 6)]
        return []
    snap = DarReader(FakeEngine(ro), schema="s1").load_snapshot(["abstract_metadata"], span_text="whole abstract")
    assert [d["source_key"] for d in snap["documents"]] == [OK_KEY]
    assert [a["source_key"] for a in snap["corpus_assoc"]] == [OK_KEY]
    assert [a["source_key"] for a in snap["chunk_assoc"]] == [OK_KEY]
    assert snap["cursor"]["source_id"] == 8  # the cursor still uses the unfiltered rows


def test_dar_snapshot_fails_closed_on_unparseable_provenance_and_keeps_none_provenance():
    def ro(sql, p):
        if "DISTINCT ON" in sql:
            return [doc_row(OK_KEY, 8, "{bad"), doc_row("https://doi.org/10.1000/example.9", 9, None)]
        return []
    snap = DarReader(FakeEngine(ro), schema="s1").load_snapshot(["abstract_metadata"], span_text=None)
    assert [d["source_key"] for d in snap["documents"]] == ["https://doi.org/10.1000/example.9"]


def test_dar_delta_drops_portal_keyed_association_without_document():
    since = {"source_id": 10, "span_id": 5, "domain_id": 2, "assoc_ts": None}

    def ro(sql, p):
        if "domain_corpus" in sql:
            return [("alpha", PORTAL_KEY, "dar", T0)]
        if "domain_chunks" in sql:
            return [("alpha", PORTAL_KEY, 0, 4, "dar", T0, 6)]
        return []
    d = DarReader(FakeEngine(ro), schema="s1").load_delta(since, ["abstract_metadata"], span_text=None)
    assert d["documents"] == [] and d["corpus_assoc"] == [] and d["chunk_assoc"] == []
    assert not d.get("partial") and d["cursor"]["source_id"] == 10
