"""VERA-DAR registry: in-memory, DB-free, network-free (directive s9D tests 1-6, 12, 13 + flag-OFF pin).

Real constraint enforcement (PK/FK/unique) is CANNOT_VERIFY here; this checks the Python semantics only.
Stdlib and pytest only; reads the shipped config file, sets no environment.
"""
import json
from pathlib import Path

import pytest

from vera import domain_registry as dr

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = ROOT / "config" / "domain_routing.json"


def _reg():
    return dr.InMemoryDomainRegistry(clock=lambda: "2026-10-17T00:00:00+00:00")


# 1 unique registration
def test_register_domain_normalizes_name():
    r = _reg()
    d, created = r.register("  Machine   Learning ")
    assert created is True and d.name == "Machine Learning" and d.domain_id == 1
    assert d.name_norm == "machine learning"
    d2, created2 = r.register("MACHINE learning")
    assert created2 is False and d2 == d
    assert len(r.list_domains()) == 1


def test_normalize_rejects_empty_and_nonstring():
    for bad in ("", "   ", None, 5):
        with pytest.raises(ValueError):
            dr.normalize_domain_name(bad)  # type: ignore[arg-type]


def test_normalize_nfkc_casefold():
    assert dr.normalize_domain_name("ＡＢ Straße") == "ab strasse"


def test_register_nfkc_variants_resolve_to_one_domain():
    r = _reg()
    full, created = r.register("\uff2d\uff2c Research")  # full-width forms
    plain, created2 = r.register("ml research")
    composed, _ = r.register("Cafe\u0301 studies")  # decomposed e + combining acute
    precomposed, created3 = r.register("Caf\u00e9 Studies")
    assert created is True and created2 is False and full == plain
    assert composed.domain_id == precomposed.domain_id and created3 is False
    assert r.get("\uff2d\uff2c RESEARCH") == full and len(r.list_domains()) == 2


def test_register_rejects_unknown_origin():
    with pytest.raises(ValueError):
        _reg().register("x", origin="auto")


# 2 reuse
def test_register_existing_name_returns_same_id():
    r = _reg()
    a, _ = r.register("Law", "first")
    b, created = r.register("law", "second")
    assert b.domain_id == a.domain_id and created is False and b.description == "first"
    known, unknown = r.resolve([a.domain_id, "LAW", "law", 999, "nope"])
    assert known == [a] and unknown == ["999", "nope"]
    assert len(r.list_domains()) == 1  # resolve never registers


# 3 multiple domains per corpus
def test_one_source_many_domains_in_memory():
    r = _reg()
    d1, _ = r.register("A")
    d2, _ = r.register("B")
    assert r.associate_corpus(d1.domain_id, 10, "t") and r.associate_corpus(d2.domain_id, 10, "t")
    assert [d.name for d in r.domains_for_corpus(10)] == ["A", "B"]
    r.associate_corpus(d1.domain_id, 11, "t")
    assert r.corpora_for_domain(d1.domain_id) == [10, 11]  # one domain, many corpora


# 4 multiple domains per chunk
def test_one_span_many_domains_in_memory():
    r = _reg()
    d1, _ = r.register("A")
    d2, _ = r.register("B")
    r.associate_chunk(d1.domain_id, 5, "t")
    r.associate_chunk(d2.domain_id, 5, "t")
    r.associate_chunk(d1.domain_id, 6, "t")
    assert [d.name for d in r.domains_for_chunk(5)] == ["A", "B"]
    assert r.chunks_for_domain(d1.domain_id) == [5, 6]


# 5 independence
def test_corpus_label_never_implies_chunk_labels():
    r = _reg()
    d, _ = r.register("A")
    r.associate_corpus(d.domain_id, 1, "t")
    assert r.chunks_for_domain(d.domain_id) == [] and r.domains_for_chunk(1) == []
    assert r.chunk_state(1) == dr.UNCLASSIFIED and r.corpus_state(1) == dr.CLASSIFIED


def test_chunk_label_without_parent_corpus_label():
    r = _reg()
    d, _ = r.register("A")
    r.associate_chunk(d.domain_id, 1, "t")
    assert r.corpora_for_domain(d.domain_id) == [] and r.corpus_state(1) == dr.UNCLASSIFIED
    assert r.chunk_state(1) == dr.CLASSIFIED


def test_disassociation_is_independent():
    r = _reg()
    d, _ = r.register("A")
    r.associate_corpus(d.domain_id, 1, "t")
    r.associate_chunk(d.domain_id, 1, "t")  # same numeric id, different asset kind
    assert r.disassociate_corpus(d.domain_id, 1) is True
    assert r.chunks_for_domain(d.domain_id) == [1]
    assert r.disassociate_corpus(d.domain_id, 1) is False  # already gone
    assert r.disassociate_chunk(d.domain_id, 1) is True and r.disassociate_chunk(d.domain_id, 1) is False


# 6 duplicate association
def test_duplicate_association_is_idempotent_false():
    r = _reg()
    d, _ = r.register("A")
    assert r.associate_corpus(d.domain_id, 1, "first") is True
    assert r.associate_corpus(d.domain_id, 1, "second") is False
    assert r.associate_chunk(d.domain_id, 2, "first") is True
    assert r.associate_chunk(d.domain_id, 2, "second") is False
    assert r.corpora_for_domain(d.domain_id) == [1] and r.chunks_for_domain(d.domain_id) == [2]
    assert r.association_actor("corpus", d.domain_id, 1) == "first"  # original row kept


def test_association_needs_known_domain_and_actor():
    r = _reg()
    d, _ = r.register("A")
    with pytest.raises(KeyError):
        r.associate_corpus(99, 1, "t")
    with pytest.raises(KeyError):
        r.associate_chunk(99, 1, "t")
    for bad in ("", "  ", None):
        with pytest.raises(ValueError):
            r.associate_corpus(d.domain_id, 1, bad)  # type: ignore[arg-type]
    assert r.corpora_for_domain(d.domain_id) == []  # failed calls leave no row


# 12 empty registry
def test_empty_registry_behaviour():
    r = _reg()
    assert r.list_domains() == [] and r.coverage() == []
    assert r.resolve(["x", 1]) == ([], ["x", "1"])
    assert r.get("x") is None and r.get(1) is None and r.get("") is None
    assert r.domains_for_corpus(1) == [] and r.domains_for_chunk(1) == []
    assert r.corpus_state(1) == dr.UNCLASSIFIED and r.chunk_state(1) == dr.UNCLASSIFIED
    assert r.distinct_totals() == {"domains": 0, "corpora_classified": 0, "chunks_classified": 0}
    assert r.list_uncertain() == []


# 13 coverage without inflation
def test_coverage_counts_distinct_assets():
    r = _reg()
    a, _ = r.register("A")
    b, _ = r.register("B")
    for dom in (a, b):
        r.associate_corpus(dom.domain_id, 1, "t")
        r.associate_corpus(dom.domain_id, 1, "again")  # duplicate must not inflate
        r.associate_chunk(dom.domain_id, 7, "t")
    r.associate_corpus(a.domain_id, 2, "t")
    cov = {c["name"]: c for c in r.coverage()}
    assert cov["A"]["corpus_count"] == 2 and cov["B"]["corpus_count"] == 1
    assert cov["A"]["chunk_count"] == 1 and cov["B"]["chunk_count"] == 1
    assert set(cov["A"]) == {"domain_id", "name", "description", "corpus_count", "chunk_count"}
    # summing rows would give 3 corpora and 2 chunks; distinct totals count each asset once
    assert r.distinct_totals() == {"domains": 2, "corpora_classified": 2, "chunks_classified": 1}


# classification and non-admission semantics
def test_uncertain_classification_is_kept_not_registered():
    r = _reg()
    rec = r.record_uncertain("maybe bio or chem", "ambiguous", candidates=["Biology", "Chemistry"])
    assert rec["candidates"] == ["Biology", "Chemistry"]
    assert r.list_domains() == [] and len(r.list_uncertain()) == 1
    with pytest.raises(ValueError):
        r.record_uncertain(" ")


def test_domain_has_no_credibility_or_admission_field():
    fields = set(dr.Domain.__dataclass_fields__)
    assert fields == {"domain_id", "name", "name_norm", "description", "created_at", "origin"}
    row = _reg().coverage()
    assert row == []
    d, _ = _reg().register("A")
    banned = ("credib", "admit", "admission", "quality", "relevan", "score", "trust")
    assert not [f for f in fields if any(b in f for b in banned)] and d.origin == "manual"


def test_load_registry_seeds_domains_only():
    r = dr.load_registry({"domains": ["Law", {"name": "law "}, {"name": "Bio", "description": "d"}, 5, {}]})
    assert [d.name for d in r.list_domains()] == ["Law", "Bio"]
    assert {d.origin for d in r.list_domains()} == {"config"}
    assert r.distinct_totals()["corpora_classified"] == 0
    assert dr.load_registry({}).list_domains() == [] and dr.load_registry(None).list_domains() == []  # type: ignore[arg-type]


# flag OFF
def test_flag_shipped_on_settable_off_and_fail_closed(tmp_path):
    """OD-DARINT-DAR-ON: shipped default ON; OFF still settable; ON with domains:[] equals baseline."""
    from vera import domain_routing as routing
    cfg = json.loads(SHIPPED.read_text())
    assert cfg["enabled"] is True and cfg["domains"] == []
    assert dr.routing_enabled(cfg) is True
    assert dr.routing_enabled(dr.load_config(SHIPPED)) is True
    assert dr.routing_enabled({**cfg, "enabled": False}) is False  # OFF still settable
    base = [{"url": "https://a.invalid/1", "title": "t"}]
    out, trace = routing.route_search("any question", base, lambda q: [{"url": "https://b.invalid/2"}],
                                      dr.InMemoryDomainRegistry(), cfg)
    assert out is base and trace["routing"] == "no_domain"  # ON + domains:[] + empty registry == baseline
    for v in ("true", 1, "yes", None):
        assert dr.routing_enabled({"enabled": v}) is False
    assert dr.routing_enabled({"enabled": True}) is True
    assert dr.routing_enabled({}) is False
    assert dr.load_config(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert dr.load_config(bad) == {} and dr.routing_enabled(dr.load_config(bad)) is False
