"""DARINT L4: flag-gated post-admission DAR hook in vera.corpus_admission. Offline; fake store; synthetic text."""
import copy
import json
from pathlib import Path

from vera import corpus_admission as ca, domain_registry
from vera.dar_corpus import DarCorpus
from vera.search_providers import OPENALEX_METADATA_LICENCE, licence_record

CFG = {"enabled": True, "db_url_env": "VERA_DB_URL_WO", "question_text": "FIXED",
       "gate_a": {"fetch_threshold": 0.6, "reject_threshold": 0.25, "max_fetch": 8}}
VOCAB = json.loads((Path(__file__).parent / "fixtures" / "dar" / "corpus_vocab.json").read_text())


def dar_cfg(hook=True, enabled=True):
    c = copy.deepcopy(domain_registry.load_config())
    c["enabled"] = enabled
    c["corpus"]["admission_hook_enabled"] = hook
    return c


class FakeStore:
    def admit(self, c):
        return "admitted"


def _llm(messages):
    rows = [{"id": 0, "query_fit": 0.9, "evidentiary_value": 0.9, "cost_risk": 0.9, "uncertainty": "low",
             "rationale": {"query_fit": "r", "evidentiary_value": "r", "cost_risk": "r"}}]
    return json.dumps({"scores": rows}), 1, 1


def _src():
    return {"n": 1, "title": "T", "url": "https://arxiv.org/abs/1", "snippet": "an abstract",
            "licence": licence_record(metadata=OPENALEX_METADATA_LICENCE), "stored_text_kind": "abstract_metadata"}


def test_shipped_flags_routing_on_hook_off():
    """OD-DARINT-DAR-ON: routing ON by default, OFF still settable; the admission hook stays OFF; with routing ON,
    domains:[] and the hook off, admit() matches baseline (no DAR effect)."""
    c = domain_registry.load_config()
    assert c["enabled"] is True and c["domains"] == [] and c["corpus"]["admission_hook_enabled"] is False
    assert domain_registry.routing_enabled({**c, "enabled": False}) is False
    dc = DarCorpus(copy.deepcopy(c))
    base = ca.admit("q", [_src()], _llm, CFG, FakeStore())
    on = ca.admit("q", [_src()], _llm, CFG, FakeStore(), dar_corpus=dc)
    assert {k: v for k, v in on.items() if not k.startswith("dar_")} == base and dc.stats()["documents"] == 0


def test_abstract_admission_is_skipped_and_recorded_not_indexed():
    dc = DarCorpus(dar_cfg())
    counts = ca.admit("q", [_src()], _llm, CFG, FakeStore(), dar_corpus=dc)
    assert counts["admitted"] == 1 and counts["dar_skipped"] == 1
    assert dc.stats()["documents"] == 0 and dc.skipped[0]["stored_text_kind"] == "abstract_metadata"


def test_hook_off_or_flag_off_is_noop():
    for dc in (DarCorpus(dar_cfg(hook=False)), DarCorpus(dar_cfg(enabled=False))):
        counts = ca.admit("q", [_src()], _llm, CFG, FakeStore(), dar_corpus=dc)
        assert not [k for k in counts if k.startswith("dar_")] and dc.skipped == []


def test_default_call_unchanged_without_dar_corpus():
    assert ca.admit("q", [_src()], _llm, CFG, FakeStore()) == {"admitted": 1}


def _cand(kind, text=None):
    return {"url": "https://example.org/d1", "canonical_url": "https://example.org/d1", "title": "D1", "published": "2024-05-01",
            "source_type": "journal", "snippet": text or " ".join(VOCAB["reef"] * 3),
            "stored_text_kind": "full_text", "provenance": {"stored_text_kind": kind, "retrieved_at": "2025-01-01"}}


def test_full_document_admit_is_indexed_and_persisted_via_callback():
    dc, plans, counts = DarCorpus(dar_cfg()), [], {}
    ca.dar_post_admit(dc, _cand("full_text"), "admitted", counts, dar_persist=plans.append)
    assert counts == {"dar_indexed": 1} and dc.stats()["documents"] == 1 and plans[0]["status"] == "indexed"


def test_gate_uses_stored_provenance_not_candidate_self_declared_kind():
    dc, counts = DarCorpus(dar_cfg()), {}
    ca.dar_post_admit(dc, _cand("abstract_metadata"), "admitted", counts)  # candidate field says full_text; provenance says abstract
    assert counts == {"dar_skipped": 1} and dc.stats()["documents"] == 0
    ca.dar_post_admit(dc, _cand("full_text"), "known", counts)  # not admitted this time
    assert counts["dar_skipped"] == 2 and dc.stats()["documents"] == 0


def test_hook_failure_never_raises_or_blocks_admission():
    class Boom:
        cfg = dar_cfg()

        def ingest_document(self, d):
            raise RuntimeError("x")
    counts = ca.admit("q", [_src()], _llm, CFG, FakeStore(), dar_corpus=Boom())
    assert counts["admitted"] == 1 and counts["dar_error"] == 1


def test_wo_store_publishes_db_ids_on_the_candidate_and_return_value_is_unchanged():
    class Eng:
        def begin(self):
            class Ctx:
                def __enter__(s):
                    return object()

                def __exit__(s, *a):
                    return False
            return Ctx()

    class S(ca.WoCorpusStore):
        def run_id(self):
            return 1

        def _insert_new(self, conn, sql, params):
            return 11 if "candidates" in sql else 22  # candidate_id, canonical_source_id

        def _one(self, conn, sql, params):
            return 33 if "INSERT INTO {S}.sources" in sql else 1  # source_id

    cand = {"url": "https://x.example/a", "canonical_url": "https://x.example/a", "title": "t", "snippet": "abs",
            "gate_a_decision": "fetch", "provenance": {"stored_text_kind": "abstract_metadata"}}
    assert S(Eng(), "q").admit(cand) == "admitted"
    assert cand["canonical_source_id"] == 22 and cand["source_id"] == 33
    from vera.dar_corpus import doc_from_admitted
    d = doc_from_admitted(cand, "admitted")
    assert d["db_source_id"] == 33 and d["db_canonical_source_id"] == 22


def test_stored_corpus_sources_are_never_readmitted():
    stored = dict(_src(), url="https://stored.example/doc", stored_text_kind="full_text",
                  discovered_via="stored_corpus")
    class Rec(FakeStore):
        def __init__(self):
            self.rows = []

        def admit(self, c):
            self.rows.append(c["url"])
            return "admitted"
    st = Rec()
    counts = ca.admit("q", [stored, _src()], _llm, CFG, st)
    assert st.rows == ["https://arxiv.org/abs/1"] and counts == {"admitted": 1}
    assert ca.is_stored_document_source(stored) and not ca.is_stored_document_source(_src())
