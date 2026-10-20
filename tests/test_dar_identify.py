"""Tests for vera.domain_identify (synthetic documents only; no DB, network or env)."""
import copy
import json
from pathlib import Path

from vera.domain_identify import DomainIdentifier, tokenize
from vera.domain_registry import InMemoryDomainRegistry, load_registry

CFG = json.loads((Path(__file__).resolve().parent.parent / "config" / "domain_routing.json").read_text())
# Mechanics tests run on tiny corpora; the shipped assoc_salient_min_docs (derive nothing below that many documents,
# review HIGH-1) is exercised separately in test_below_salient_threshold_derives_nothing.
SHIPPED_MIN_DOCS = CFG["identification"]["assoc_salient_min_docs"]
CFG["identification"]["assoc_salient_min_docs"] = 2
SHIPPED_MIN_SUPPORT = CFG["identification"]["min_support"]
CFG["identification"]["min_support"] = 2  # the min-support PATH is exercised here; the shipped value (1) has its own tests

SEC = ("vulnerability exploit injection authentication encryption malware firewall patch credential attacker phishing "
       "sandbox privilege payload ransomware cipher intrusion exfiltration backdoor hardening").split()
DEV = ("developer workflow commit review refactoring editor shortcut automation pipeline debugging tooling focus "
       "latency productivity snippet template terminal linter backlog standup").split()
BIO = ("protein cell genome enzyme mutation membrane chromosome ribosome species organism metabolism "
       "mitochondria nucleotide receptor tissue bacteria evolution lipid peptide").split()


def noise(i):
    return "zq" + chr(97 + i % 26) + chr(97 + (i // 26) % 26) + "x"


def make_doc(words, i, n=50, extra=()):
    parts = [words[(i * 3 + j * 7 + j // len(words)) % len(words)] for j in range(n)]
    parts += [noise(i), noise(i + 40), noise(i + 80)]
    parts += list(extra)
    return " ".join(parts)


def fresh(cfg=None, domains=None):
    cfg = copy.deepcopy(cfg or CFG)
    reg = InMemoryDomainRegistry()
    for name in domains or []:
        reg.register(name, "", origin="config")
    return reg, DomainIdentifier(reg, cfg["identification"]), cfg["identification"]


def seed_topics(ident, per=2):
    out = {}
    for label, words in (("sec", SEC), ("dev", DEV), ("bio", BIO)):
        for i in range(per):
            ident.add_document(f"{label}{i}", make_doc(words, i), "t")
    for label in ("sec", "dev", "bio"):  # promotion is re-evaluated as documents arrive: read the final state
        out[label] = ident.registry.domains_for_corpus(f"{label}0")[0].domain_id
    return out


def test_tokenize_uses_config():
    ic = CFG["identification"]
    assert tokenize("The Cell, the CELL; ab 123 genome-x", ic) == ["cell", "cell", "genome"]


def test_determinism():
    runs = []
    for _ in range(2):
        reg, ident, _ic = fresh()
        ids = seed_topics(ident)
        runs.append((ids, [d.name for d in reg.list_domains()],
                     ident.classify(make_doc(BIO, 9)), ident.classify("protein enzyme", "question")))
    assert runs[0] == runs[1]


def test_topics_become_three_domains_named_from_content():
    reg, ident, _ = fresh()
    ids = seed_topics(ident)
    assert len(set(ids.values())) == 3
    assert [d.origin for d in reg.list_domains()] == ["derived"] * 3
    assert set(ident.profile_terms(ids["bio"], 5)) <= set(BIO) | {noise(i) for i in range(200)}
    assert reg.get(ids["sec"]).name.split()[0] in SEC


def test_multi_domain_doc_joins_two():
    reg, ident, _ = fresh()
    ids = seed_topics(ident)
    r = ident.add_document("mix", make_doc(SEC, 5, 30) + " " + make_doc(BIO, 5, 30), "t")
    got = {d["domain_id"] for d in r["domains"]}
    assert got == {ids["sec"], ids["bio"]}
    assert not r["unclassified"] and r["created"] == []


def test_chunk_classified_independently_of_doc():
    reg, ident, _ = fresh()
    ids = seed_topics(ident)
    ident.add_document("sec9", make_doc(SEC, 9), "t")
    chunk = make_doc(BIO, 11, 30)
    got = ident.classify(chunk, "chunk")
    assert got and got[0]["domain_id"] == ids["bio"]
    assert reg.domains_for_corpus("sec9")[0].domain_id == ids["sec"]


def test_classify_is_read_only():
    reg, ident, _ = fresh()
    seed_topics(ident)
    before = (ident._n, dict(ident._df), reg.coverage(), reg.list_uncertain(), copy.deepcopy(ident._profiles))
    for kind in ("chunk", "question"):
        ident.classify(make_doc(DEV, 3) + " brandnewterm", kind)
    after = (ident._n, dict(ident._df), reg.coverage(), reg.list_uncertain(), ident._profiles)
    assert before == after


def test_classify_question_vs_chunk_thresholds():
    reg, ident, ic = fresh()
    ids = seed_topics(ident)
    q = ident.classify("protein mutation", "question")
    assert q and q[0]["domain_id"] == ids["bio"]
    assert len(q) <= ic["max_question_domains"]
    assert ident.classify("") == []


def test_min_support_single_doc_not_registered():
    reg, ident, _ = fresh()
    ident.add_document("f1", make_doc(DEV, 3), "t")  # unrelated singletons so the corpus has >= 4 documents
    ident.add_document("f2", make_doc(BIO, 3), "t")
    r = ident.add_document("a", make_doc(SEC, 0), "t")
    assert r["pending"] and r["unclassified"] and r["uncertain"] and r["created"] == []
    assert reg.list_domains() == [] and len(reg.list_uncertain()) == 3
    r2 = ident.add_document("b", make_doc(SEC, 1), "t")
    assert len(r2["created"]) == 1 and r2["promoted"] == ["a"] and not r2["pending"]
    did = r2["created"][0]
    assert reg.corpora_for_domain(did) == ["a", "b"]
    assert [u["label"] for u in reg.list_uncertain()] == ["f1", "f2", "a", "b"]  # each pending doc recorded once


def test_twenty_docs_bounded_domains():
    reg, ident, _ = fresh()
    for i in range(20):
        ident.add_document(f"d{i}", make_doc(SEC, i, extra=[noise(i * 7 + 3), noise(i * 11 + 5)]), "t")
    assert len(reg.list_domains()) <= 3  # bounded (a one-topic corpus has no topical contrast, so 0 is acceptable)


def test_max_derived_domains_cap():
    cfg = copy.deepcopy(CFG)
    cfg["identification"]["max_derived_domains"] = 2
    reg, ident, _ = fresh(cfg)
    for label, words in (("sec", SEC), ("dev", DEV), ("bio", BIO)):
        for i in range(2):
            r = ident.add_document(f"{label}{i}", make_doc(words, i), "t")
    assert len(reg.list_domains()) == 2
    assert r["pending"] and r["unclassified"] and r["created"] == []
    r = ident.add_document("bio2", make_doc(BIO, 2), "t")
    assert len(reg.list_domains()) == 2 and r["unclassified"]


def _bio_cluster_name(a=0, b=1):
    reg, ident, _ = fresh()
    for i in range(30):  # a corpus large enough for the shared-vocabulary view (documents of another topic)
        ident.restore_document(f"f{i}", make_doc(SEC, i), [])
    ident.add_document("b0", make_doc(BIO, a), "t")
    ident.add_document("b1", make_doc(BIO, b), "t")
    return reg.list_domains()[0].name


def test_name_collision_suffix_when_dissimilar():
    # BIO docs use i=300/301 so their noise tokens do not collide with the restored SEC documents' noise tokens
    name = _bio_cluster_name(300, 301)
    reg, ident, _ = fresh(domains=[name])
    colliding = reg.get(name).domain_id
    for i in range(30):
        ident.restore_document(f"s{i}", make_doc(SEC, i), [colliding])
    ident.add_document("b0", make_doc(BIO, 300), "t")
    r = ident.add_document("b1", make_doc(BIO, 301), "t")
    assert len(r["created"]) == 1 and r["created"][0] != colliding
    assert reg.get(r["created"][0]).name == f"{name} (2)"
    assert reg.get(name).domain_id == colliding


def test_name_collision_reuse_when_similar():
    name = _bio_cluster_name()
    reg, ident, _ = fresh(domains=[name])
    colliding = reg.get(name).domain_id
    for i in range(5):
        ident.restore_document(f"r{i}", make_doc(BIO, i + 10), [colliding])
    ident.add_document("b0", make_doc(BIO, 0), "t")
    ident.add_document("b1", make_doc(BIO, 1), "t")
    assert len(reg.list_domains()) == 1


def test_too_short_doc_uncertain_unclassified():
    reg, ident, ic = fresh()
    r = ident.add_document("s", "protein cell", "t")
    assert r["unclassified"] and r["uncertain"] and not r["pending"] and r["domains"] == []
    assert reg.list_domains() == [] and ident._n == 0 and ic["min_doc_terms"] > 2


def test_config_domains_without_terms_do_not_error():
    reg = load_registry({"domains": ["Genomics", {"name": "Network Security", "description": "x"}]})
    ident = DomainIdentifier(reg, CFG["identification"])
    assert ident.classify("genomics research on genomics") [0]["domain_id"] == reg.get("Genomics").domain_id
    assert ident.classify("zzzz qqqq") == []
    r = ident.add_document("g", make_doc(BIO, 0, extra=["genomics"] * 8), "t")
    assert isinstance(r["domains"], list)
    assert ident.profile_terms(reg.get("Genomics").domain_id, 3)


def test_config_unchanged_after_new_topic():
    cfg = copy.deepcopy(CFG)
    before = json.dumps(cfg, sort_keys=True)
    reg, ident, _ = fresh(cfg)
    seed_topics(ident)
    ident.add_document("new", make_doc(["quantum", "photon", "qubit", "entangle", "laser", "optics", "wave"], 1), "t")
    assert json.dumps(cfg, sort_keys=True) == before
    assert json.dumps(ident.icfg, sort_keys=True) == json.dumps(CFG["identification"], sort_keys=True)


def test_restore_document_no_registry_write():
    reg, ident, _ = fresh()
    ids = seed_topics(ident)
    cov = reg.coverage()
    unc = reg.list_uncertain()
    n0 = ident._n
    ident.restore_document("old", make_doc(BIO, 20), [ids["bio"]])
    assert ident._n == n0 + 1
    assert reg.coverage() == cov and reg.list_uncertain() == unc
    # rebuilt identifier reproduces classification from restored docs alone
    reg2 = InMemoryDomainRegistry()
    for d in reg.list_domains():
        reg2.register(d.name, d.description, origin=d.origin)
    ident2 = DomainIdentifier(reg2, CFG["identification"])
    for label, words in (("sec", SEC), ("dev", DEV), ("bio", BIO)):
        for i in range(2):
            ident2.restore_document(f"{label}{i}", make_doc(words, i), [ids[label]])
    probe = make_doc(BIO, 31, 30)
    assert ident2.classify(probe)[0]["domain_id"] == ids["bio"]


# ---- fix round 2: D1 (diluted docs), review HIGH-1 (small-N boilerplate), order invariance ----
import itertools as _it
import random as _random

from vera import domain_registry as _dreg
from vera.dar_corpus import DarCorpus as _DarCorpus

_TOPICS = {
    "cardio": "heart cardiac artery blood pressure hypertension cholesterol vascular stroke myocardial infarction".split(),
    "soil": "soil nitrogen crop yield fertilizer irrigation farm wheat harvest agronomy".split(),
    "nets": "neural network training gradient layer transformer attention model learning dataset".split(),
    "orbit": "satellite orbit apogee perigee thruster propellant trajectory payload telemetry launch".split(),
}


def _enabled_cfg(min_docs=2, min_support=2):
    import copy
    c = copy.deepcopy(_dreg.load_config())
    c["enabled"] = True
    c["identification"]["assoc_salient_min_docs"] = min_docs
    c["identification"]["min_support"] = min_support
    return c


def _tiny_docs(per=3):
    docs = {}
    for tp, v in _TOPICS.items():
        for j in range(per):
            ws = list(v)
            _random.Random(f"{tp}{j}").shuffle(ws)
            docs[f"{tp}{j}"] = " ".join(ws * 3) + " study results"
    return docs


def _partition(order, docs):
    c = _DarCorpus(_enabled_cfg())
    for k in order:
        c.ingest_document({"source_key": k, "text": docs[k], "stored_text_kind": "full_text", "admission_passed": True})
    part = set()
    for d in c.registry.list_domains():
        part.add(frozenset(k for k in docs if d.domain_id in {x.domain_id for x in c.registry.domains_for_corpus(k)}))
    return part


def test_review_high1_boilerplate_does_not_merge_unrelated_docs():
    docs = _tiny_docs()
    part = _partition(["cardio0", "soil2", "nets1", "cardio1", "soil0", "nets0", "cardio2", "soil1", "nets2"], docs)
    for g in part:
        assert len({k.rstrip("0123456789") for k in g}) == 1, sorted(g)  # no cluster mixes topics


def test_review_high1_partition_is_permutation_invariant_on_small_fixture():
    docs = {k: v for k, v in _tiny_docs(2).items()}
    keys = sorted(docs)
    rng = _random.Random(5)
    orders = [rng.sample(keys, len(keys)) for _ in range(60)]  # bounded seeded sample of the 720 orders
    seen = {frozenset(_partition(p, docs)) for p in orders}
    assert len(seen) == 1, f"{len(seen)} distinct partitions across all orders"
    (only,) = seen
    assert {frozenset(k for k in keys if k.startswith(t)) for t in _TOPICS} == set(only)


def _generic_words(n=60):
    letters = "abcdefghijklmnopqrstuvwxyz"
    return [f"{letters[i % 26]}{letters[(i // 26) % 26]}{letters[(i * 7) % 26]}q" for i in range(n)]  # distinct alphabetic tokens


def _diluted_docs(per=3, topics=("cardio", "soil", "nets", "orbit"), seed=7):
    gen, rng, docs = _generic_words(), _random.Random(seed), {}
    for tp in topics:
        for j in range(per):
            top = _random.Random(f"{tp}{j}").sample(_TOPICS[tp] * 2, 10)
            body = rng.sample(gen * 2, 70) + top
            rng.shuffle(body)
            docs[f"{tp}{j}"] = " ".join(body)
    return docs


def _partition_of(order, docs, min_docs=None):
    min_docs = SHIPPED_MIN_DOCS if min_docs is None else min_docs
    c = _DarCorpus(_enabled_cfg(min_docs=min_docs))
    for k in order:
        c.ingest_document({"source_key": k, "text": docs[k], "stored_text_kind": "full_text", "admission_passed": True})
    prim = lambda d, k: (c.registry.association_actor("corpus", d.domain_id, k) or ":secondary").endswith(":secondary") is False  # noqa: E731
    return frozenset(frozenset(k for k in docs if prim(d, k)) for d in c.registry.list_domains()), c


def test_d1_diluted_documents_of_unrepresented_domain_form_a_derived_domain_without_config_change():
    import hashlib, json
    docs = _diluted_docs()
    cfg = _enabled_cfg(min_docs=SHIPPED_MIN_DOCS)
    before = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    order = [f"{tp}{j}" for tp in ("cardio", "soil", "orbit") for j in range(3)] + [f"nets{j}" for j in range(3)]
    c = _DarCorpus(cfg)
    for k in order:
        c.ingest_document({"source_key": k, "text": docs[k], "stored_text_kind": "full_text", "admission_passed": True})
    nets = [d for d in c.registry.list_domains()
            if sum(1 for k in ("nets0", "nets1", "nets2") if d in c.registry.domains_for_corpus(k)) >= 2]
    assert nets and all(d.origin == "derived" for d in nets)
    for d in nets:  # the derived domain EXCLUDES the other topics' documents
        members = {k for k in docs if d in c.registry.domains_for_corpus(k)}
        assert members and all(k.startswith("nets") for k in members), members
    assert hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() == before


def test_review_high1_diluted_corpus_partition_is_stable_across_sampled_orders():
    docs = _diluted_docs()
    keys, rng = sorted(docs), _random.Random(3)
    seen = set()
    for _ in range(8):
        order = list(keys)
        rng.shuffle(order)
        part, _c = _partition_of(order, docs)
        seen.add(part)
        for g in part:  # no mixed-topic domain, ever
            assert len({k.rstrip("0123456789") for k in g}) == 1, sorted(g)
    assert len(seen) <= 2, f"{len(seen)} distinct partitions over 8 sampled orders (residual order dependence must stay bounded)"


# ---- shipped config (min_support 1): domains form from the first documents; provisional until supported ----
import string as _string


def _moderate_docs(topics=("reef", "compiler", "orbit", "ecology"), per=(3, 3, 3, 1), seed=99):
    import json as _json
    voc = _json.loads((Path(__file__).parent / "fixtures" / "dar" / "corpus_vocab.json").read_text())
    voc.setdefault("ecology", "ecosystem predator prey population habitat biodiversity niche species trophic forecasting "
                              "migration resilience".split())
    g = _random.Random(seed)
    gen = sorted({"".join(g.choice(_string.ascii_lowercase) for _ in range(7)) for _ in range(400)})
    docs = {}
    for tp, n in zip(topics, per):
        for j in range(n):
            r = _random.Random(f"{tp}{j}")
            body = r.sample(gen, 50) + r.sample(voc[tp] * 2, 14) + ["study", "results", "method"]
            r.shuffle(body)
            docs[f"{tp}{j}"] = " ".join(body)
    return docs


def _shipped_cfg():
    import copy
    c = copy.deepcopy(_dreg.load_config())
    c["enabled"] = True
    return c


def _ingest_all(order, docs, cfg=None):
    c = _DarCorpus(cfg or _shipped_cfg())
    for k in order:
        c.ingest_document({"source_key": k, "text": docs[k], "stored_text_kind": "full_text", "admission_passed": True})
    return c


def _members(c, docs):
    return [frozenset(k for k in docs if d.domain_id in {x.domain_id for x in c.registry.domains_for_corpus(k)})
            for d in c.registry.list_domains()]


def test_shipped_config_domains_form_from_the_first_document_and_start_provisional():
    docs = _moderate_docs()
    c = _ingest_all(["reef0"], docs)
    assert len(c.registry.list_domains()) == 1  # one document is enough (no ten-document gate)
    (d,) = c.registry.list_domains()
    assert d.origin == "derived" and c.domain_status(d.domain_id) == "provisional"
    assert c.retrieve(" ".join(sorted(set(docs["reef0"].split()))[:6]))["candidates"]  # retrievable at once


def _demo():
    import json as _json
    base = Path(__file__).parent / "fixtures" / "dar"
    rows = _json.loads((base / "demo9.json").read_text()) + _json.loads((base / "eco1.json").read_text())
    return {r["doc_id"]: r["content"] for r in rows}


def _primary_partition(c, keys):
    prim = lambda d, k: not (c.registry.association_actor("corpus", d.domain_id, k) or ":secondary").endswith(":secondary")  # noqa: E731
    return frozenset(frozenset(k for k in keys if prim(d, k)) for d in c.registry.list_domains())


def test_shipped_config_demo9_three_unmixed_domains_plus_provisional_fourth_sampled_orders():
    docs = _demo()
    keys, rng = sorted(docs), _random.Random(4)
    seen = {}
    for _ in range(12):
        order = rng.sample(keys, len(keys))
        c = _ingest_all(order, docs)
        part = _primary_partition(c, keys)
        seen[part] = seen.get(part, 0) + 1
        for g in part:
            assert len({k[:2] for k in g if not k.startswith("E")}) <= 1, sorted(g)  # no two demo subjects share a domain
        for k in keys:
            assert c.registry.domains_for_corpus(k), k  # nothing is left unclassified
    expect = frozenset({frozenset({"D1-1", "D1-2", "D1-3"}), frozenset({"D2-1", "D2-2", "D2-3"}),
                        frozenset({"D3-1", "D3-2", "D3-3"}), frozenset({"E-1"})})
    assert seen.get(expect, 0) >= 10, {tuple(sorted(map(sorted, k))): v for k, v in seen.items()}  # bounded order effect
    c = _ingest_all(keys, docs)
    assert c.domain_status(c.registry.domains_for_corpus("E-1")[0].domain_id) == "provisional"


def test_shipped_config_new_unrepresented_topic_gets_a_provisional_domain_immediately_without_config_change():
    import hashlib, json
    docs = _demo()
    cfg = _shipped_cfg()
    before = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    c = _ingest_all([k for k in sorted(docs) if k != "E-1"], docs, cfg)
    known = {d.domain_id for d in c.registry.list_domains()}
    r = c.ingest_document({"source_key": "E-1", "text": docs["E-1"], "stored_text_kind": "full_text", "admission_passed": True})
    new = [d for d in c.registry.list_domains() if d.domain_id not in known]
    assert len(new) == 1 and c.domain_status(new[0].domain_id) == "provisional"
    assert r["status"] == "indexed" and any(x["created"] for x in r["domains"])
    q = " ".join(sorted(set(docs["E-1"].split()))[:8])
    assert any(x["source_key"] == "E-1" for x in c.retrieve(q)["candidates"])
    assert hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() == before
