"""HYBRID-1109 (OD-1109-HYBRID): hybrid stored-passage retrieval, offline. A fake embedder returns deterministic concept-bag
vectors in which "archetype" and "category"/"type"/"kind" share one concept; no network, no DB (an injected vector loader)."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re

import pytest

from vera import chunk_taxo, chunk_taxo_embed as emb, fulldoc_admission as fa, hybrid_retrieval as hr
from vera import domain_routing
from vera.dar_corpus import DarCorpus
from vera.fulldoc_corpus import FullDocCorpus, _routing_cfg, load_config

MODEL = "@cf/baai/fake-hybrid"
SYN = {"archetyp": "KIND", "categor": "KIND", "type": "KIND", "kind": "KIND"}
STOP = set("the a an of and or in on to for is are be by with what which used that this as at it from".split())


def fake_vec(t: str, dim: int = 64) -> list:
    v = [0.0] * dim
    counts: dict = {}
    for w in re.findall(r"[a-z]+", t.lower()):
        if w in STOP or len(w) < 3:
            continue
        c = next((x for k, x in SYN.items() if w.startswith(k)), w[:6])
        counts[c] = counts.get(c, 0) + 1
    for c, n in counts.items():
        h = int(hashlib.sha256(c.encode()).hexdigest(), 16)
        v[h % dim] += (1 + math.log(n)) * (8.0 if c == "KIND" else 1.0)
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class FakeEmbedder:
    name = MODEL

    def __init__(self, fail=None, fail_batch=None):
        self.calls, self.batches, self.fail, self.fail_batch, self.day_used = 0, [], fail, fail_batch, None

    def set_day_used(self, n):
        self.day_used = n

    def embed(self, texts):
        self.calls += 1
        self.batches.append(len(texts))
        if self.fail is not None:
            raise self.fail
        if self.fail_batch is not None and len(texts) > 1:
            raise self.fail_batch
        return [fake_vec(t) for t in texts]


PARA_A = ("The orchestration layer of a multi-agent system routes messages between agents and keeps the multi-agent "
          "orchestration state consistent across agent turns, so that every agent sees the orchestration plan. ")
INTRO_B = ("This survey reviews enterprise adoption, governance and deployment practice for multi-agent orchestration "
           "platforms; each agent platform is compared on orchestration cost and agent uptime. ")
ROLE_B = ["Below are the key categories and types of specialized agents found in multi-agent deployments today. ",
          "Worker agents are the most basic type; they carry out well-defined tasks such as retrieval pipelines. ",
          "Supervisor agents are a second kind; they assign work to workers and check results before release. ",
          "Critic agents are a third category; they review outputs and request revisions when quality is low. "]


def _texts():
    a = "".join(f"Section {i}. " + PARA_A * 2 + "\n\n" for i in range(12))
    b = INTRO_B * 8 + "\n\n" + "".join(p * 2 + "\n\n" for p in ROLE_B)
    c = "".join(f"Part {i}. " + PARA_A.replace("routes messages", "schedules tasks") * 2 + "\n\n" for i in range(12))
    return a, b, c


def _corpus(spans_b=None):
    a, b, c3 = _texts()
    cfg = load_config()
    dar = DarCorpus(_routing_cfg())
    acq, key_to_doc, texts = [], {}, {}
    for did, key, t in (("DA", "https://example.org/a", a), ("DB", "https://example.org/b", b), ("DC", "https://example.org/c", c3)):
        res = dar.ingest_document({"source_key": key, "url": key, "doi": None, "title": "Doc " + did, "published": "2026-01-01",
                                   "source_type": "preprint", "retrieved_at": "2026-11-09", "text": t,
                                   "stored_text_kind": "full_text", "admission_passed": True, "db_source_id": None,
                                   "db_canonical_source_id": None})
        assert res.get("status") == "indexed"
        acq.append({"document_id": did, "canonical_source_id": key, "title": "Doc " + did, "url": key, "doi": None,
                    "published": "2026-01-01", "authors": ["Author"], "licence": "CC-BY-4.0",
                    "licence_url": "https://creativecommons.org/licenses/by/4.0/", "licence_decision": "allow",
                    "fulltext_format": "PDF", "text_sha256": "x", "chars": len(t), "evidence_ranges": [[0, len(t)]],
                    "excluded_ranges": []})
        key_to_doc[key], texts[did] = did, t
    c = FullDocCorpus("ok", acq, [], texts, dar, key_to_doc, cfg)
    c.source_kind = "db"
    cut = b.index("Below are")
    parts = {"DA": [(0, len(a) // 2), (len(a) // 2, len(a))], "DB": spans_b or [(0, cut), (cut, len(b))],
             "DC": [(0, len(c3) // 2), (len(c3) // 2, len(c3))]}
    sid, stamps, table = 1000, {}, {}
    for did, rs in parts.items():
        lst = []
        for s, e in rs:
            sid += 1
            lst.append({"span_id": sid, "start": s, "end": e})
            table[sid] = (did, s, e)
        stamps[did] = {"corpus_added_at": None, "corpus_stamps": lst}
    c.stamps = stamps
    return c, table


class Loader:
    def __init__(self, corpus, table, stale=(), neurons=0.0):
        self.c, self.t, self.stale, self.neurons, self.calls = corpus, table, set(stale), neurons, 0

    def vectors(self, ids, model):
        self.calls += 1
        for i in ids:
            did, s, e = self.t[i]
            txt = self.c._texts[did][s:e]
            sha = hashlib.sha256(("x" if i in self.stale else txt[:8000]).encode()).hexdigest()
            yield i, sha, fake_vec(txt)

    def neurons_today(self):
        return self.neurons


Q = "Which agent archetypes are used in multi-agent orchestration?"
CF_ENV = {"CLOUDFLARE_ACCOUNT_ID": "acct_test", "CLOUDFLARE_AI_TOKEN": "tok-secret-123"}


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    hr.reset_for_tests()
    emb.reset_budget_for_tests()
    yield
    hr.reset_for_tests()
    emb.unregister("fakehybrid")
    emb.reset_budget_for_tests()


def setup_hybrid(monkeypatch, corpus, table, fake=None, **cfg):
    fake = fake or FakeEmbedder()
    emb.register("fakehybrid", lambda c: fake)
    tc = chunk_taxo.load_cfg()
    monkeypatch.setattr(hr, "_taxo_cfg", lambda: dict(tc, embedder="fakehybrid"))
    hr.set_config_for_tests(dict({"enabled": True}, **cfg))
    loader = Loader(corpus, table)
    hr.set_loader_for_tests(loader)
    idx, code = hr.ensure_index(corpus, MODEL, sync=True)
    assert code == "ok" and idx is not None
    return fake, loader


def _role(c, b0):
    return c["document_id"] == "DB" and c["end"] > b0


# ------------------------------------------------------------------------------------------------ flag off
def test_flag_off_no_new_keys_no_calls_and_same_candidates(monkeypatch):
    c, table = _corpus()
    called = []
    monkeypatch.setattr(hr, "vector_fn", lambda *a, **k: called.append(1) or (None, None))
    monkeypatch.setattr(hr, "query_vector", lambda *a, **k: called.append(2))
    assert hr.enabled() is False  # shipped config: OFF
    cands = c.retrieve(Q, 16, True)
    direct = c._evidence_dar().retrieve(Q, 16, max_per_doc=8)
    assert [x["span_id"] for x in cands] == [f"{c._key_to_doc[d['source_key']]}:{d['start']}-{d['end']}" for d in direct["candidates"]]
    assert "vector" not in direct["trace"]["branch_counts"]
    assert all("vec_sim" not in x for x in cands)
    b = fa.admit(Q, c, load_config())
    assert "hybrid" not in b.trace and not called


def test_enabled_but_unavailable_is_byte_identical_to_off(monkeypatch):
    c, table = _corpus()
    off = fa.admit(Q, c, load_config())
    off_c = c.retrieve(Q, 16, True)
    setup_hybrid(monkeypatch, c, table, fake=FakeEmbedder(fail=emb.EmbedError("http_500")))
    on_c = c.retrieve(Q, 16, True)
    on = fa.admit(Q, c, load_config())
    assert json.dumps(on_c, sort_keys=True) == json.dumps(off_c, sort_keys=True)
    t = dict(on.trace)
    h = t.pop("hybrid")
    assert h["status"].startswith("cooldown:") or h["status"].startswith("embed_error:")
    assert json.dumps(off.trace, sort_keys=True) == json.dumps(t, sort_keys=True)
    assert [a["span_id"] for a in on.admitted] == [a["span_id"] for a in off.admitted]


# ------------------------------------------------------------------------------------------------ fusion
def test_fuse_order_is_reciprocal_rank_fusion():
    items = [(1, "a"), (2, "b"), (3, "c"), (4, "d")]
    vs = {"c": 0.9, "d": 0.8}
    out = hr.fuse_order(items, lambda t: t[0], lambda t: vs.get(t[1]), 60)
    # a 1/61, b 1/62, c 1/63+1/61, d 1/64+1/62 => c, d, a, b
    assert [t[1] for t in out] == ["c", "d", "a", "b"]


def test_dar_retrieve_fuses_lexical_and_vector_as_peers():
    c, _ = _corpus()
    ev = c._evidence_dar()
    lex = ev.retrieve(Q, 40, max_per_doc=40)["candidates"]
    want_v = [ch for ch in ev._chunks if ch.doc["source_key"].endswith("/b")][-3:]

    def vfn(dar, scores):
        return want_v
    res = ev.retrieve(Q, 40, max_per_doc=40, vector_fn=vfn)
    assert res["trace"]["branch_counts"]["vector"] == 3
    ids = [x["span_id"] for x in res["candidates"]]
    lex_ids = [x["span_id"] for x in lex]
    groups, _ = domain_routing._rrf([("lexical", [{"span_id": s} for s in lex_ids]),
                                     ("vector", [{"span_id": ch.span_id} for ch in want_v])], 60, key_fn=lambda d: d["span_id"])
    groups.sort(key=lambda g: (-g["score"], g["best"]))
    assert ids == [g["key"] for g in groups][:len(ids)]
    assert {ch.span_id for ch in want_v} <= set(ids)
    top = groups[0]  # rank-1 tie (lexical 1 vs vector 1) keeps the word-matching item first
    assert ids[0] == top["key"] and (lex_ids[0] == ids[0] or "vector" in res["candidates"][0]["branches"])
    assert any("vector" in x["branches"] for x in res["candidates"])


# ------------------------------------------------------------------------------------------------ archetypes vs categories
def test_archetypes_reach_categories_passages_only_with_hybrid(monkeypatch):
    c, table = _corpus()
    b0 = c._texts["DB"].index("Below are")
    off = fa.admit(Q, c, load_config())
    assert not any(_role(a, b0) for a in off.admitted)  # word-only: the "categories" section loses to keyword-heavy text
    fake, _ = setup_hybrid(monkeypatch, c, table, chunks_per_partition=3)
    on = fa.admit(Q, c, load_config())
    assert on.trace["hybrid"]["status"] == "ok" and on.trace["hybrid"]["fused"] is True
    assert any(_role(a, b0) for a in on.admitted), on.trace
    assert fake.calls == 1


def test_vector_stem_floor_rule(monkeypatch):
    c, table = _corpus()
    b0 = c._texts["DB"].index("Worker agents")
    q = "Which agent archetypes?"  # content stems: agent, arche (a role passage matches "agent" only)
    fake, _ = setup_hybrid(monkeypatch, c, table, chunks_per_partition=4, vector_k=4)
    base = fa.admit(q, c, load_config())
    assert base.trace["hybrid"]["floor_by_vector"] == 0
    assert not any(_role(a, b0) for a in base.admitted)
    hr.set_config_for_tests({"enabled": True, "chunks_per_partition": 4, "vector_k": 4, "stem_floor_sim": 0.3})
    hr.set_loader_for_tests(Loader(c, table))
    hr.ensure_index(c, MODEL, sync=True)
    on = fa.admit(q, c, load_config())
    assert on.trace["hybrid"]["floor_by_vector"] >= 1
    assert any(_role(a, b0) for a in on.admitted)
    # never without a matched content stem: a floor of -1 still leaves zero-stem passages out
    hr.set_config_for_tests({"enabled": True, "chunks_per_partition": 4, "vector_k": 4, "stem_floor_sim": -1.0})
    hr.set_loader_for_tests(Loader(c, table))
    hr.ensure_index(c, MODEL, sync=True)
    any_floor = fa.admit("Which archetypes?", c, load_config())  # only "arche": no stored passage matches it
    assert any_floor.admitted == () and any_floor.trace["hybrid"]["floor_by_vector"] == 0


# ------------------------------------------------------------------------------------------------ fallback + budget
def test_fallback_on_error_then_cooldown(monkeypatch):
    c, table = _corpus()
    fake = FakeEmbedder(fail=emb.EmbedError("request_failed"))
    setup_hybrid(monkeypatch, c, table, fake=fake, cooldown_s=60)
    t = [1000.0]
    hr.set_clock_for_tests(clock=lambda: t[0])
    c.retrieve(Q, 16, True)
    assert hr.last_status()["status"] == "embed_error:request_failed"
    c.retrieve(Q + " again", 16, True)
    assert hr.last_status()["status"] == "cooldown:embed_error:request_failed" and fake.calls == 1
    t[0] += 61
    fake.fail = None
    out = c.retrieve(Q + " later", 16, True)
    assert hr.last_status()["status"] == "ok" and fake.calls == 2 and any(x.get("vec_sim") is not None for x in out)


def test_quota_defers_until_utc_midnight(monkeypatch):
    c, table = _corpus()
    fake = FakeEmbedder(fail=emb.Deferred("quota_reached", 5000.0))
    setup_hybrid(monkeypatch, c, table, fake=fake)
    t = [0.0]
    hr.set_clock_for_tests(clock=lambda: t[0])
    lex = [x["span_id"] for x in c.retrieve(Q, 16, True)]
    assert hr.last_status()["status"] == "deferred:quota_reached"
    t[0] = 4999.0
    c.retrieve(Q + "?", 16, True)
    assert hr.last_status()["status"].startswith("cooldown:deferred") and fake.calls == 1
    hr.set_config_for_tests({"enabled": False})
    assert [x["span_id"] for x in c.retrieve(Q, 16, True)] == lex  # the fallback list is the word-only list


def test_question_embedded_once_and_day_usage_passed(monkeypatch):
    c, table = _corpus()
    c2, _ = _corpus()
    fake, loader = setup_hybrid(monkeypatch, c, table)
    loader.neurons = 1234.0
    hr.set_loader_for_tests(Loader(c2, table, neurons=1234.0))
    hr.ensure_index(c2, MODEL, sync=True)
    c.retrieve(Q, 16, True)
    c2.retrieve(Q, 16, True)
    c.retrieve(Q, 16, False)
    assert fake.calls == 1 and fake.day_used == 1234.0


class _Resp:
    def __init__(self, body):
        self.body = body

    def read(self, n=-1):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_budget_job_reserve_and_ask_share(monkeypatch):
    cf = chunk_taxo.load_cfg()["cloudflare"]
    assert cf["reserve_neurons"] == 5000 and cf["daily_neuron_budget"] == 8000  # job 3000/day, ask share 5000
    calls = []

    def opener(req, timeout):
        calls.append(timeout)
        n = len(json.loads(req.data.decode())["text"])
        return _Resp(json.dumps({"success": True, "result": {"data": [[0.5, 0.5]] * n, "usage": {"total_tokens": 30}}}).encode())
    job = emb.CloudflareEmbedder(cf, env=dict(CF_ENV), opener=opener, clock=lambda: 1_700_000_000.0, sleep=lambda s: None)
    job.set_day_used(2999.99)
    with pytest.raises(emb.Deferred):
        job.embed(["x" * 300])  # the job may not eat into the ask-time reserve
    # ask-time adapter as built by hybrid_retrieval: whole budget, one attempt, short timeout
    monkeypatch.setattr(emb, "_default_open", opener)
    for k, v in CF_ENV.items():
        monkeypatch.setenv(k, v)
    hr.set_config_for_tests({"enabled": True, "timeout_s": 3.0})
    with hr._LOCK:
        hr._STATE["neurons_today"] = 2999.99
    v, code, _ms = hr.query_vector("a short question")
    assert code == "ok" and v is not None and calls[-1] == 3.0
    e = hr._STATE["emb"]
    assert e.cfg["max_attempts"] == 1 and e.cfg["reserve_neurons"] == 0 and e.cfg["timeout_s"] == 3.0
    with hr._LOCK:
        hr._STATE["neurons_today"] = 7999.999
    v2, code2, _ = hr.query_vector("another short question")
    assert v2 is None and code2 == "deferred:budget_reached"


def test_index_skips_stale_vectors_and_respects_size_cap(monkeypatch):
    c, table = _corpus()
    setup_hybrid(monkeypatch, c, table)
    stale_id = sorted(table)[-1]
    idx, code = hr.build_index(c, MODEL, Loader(c, table, stale={stale_id}))
    assert code == "ok" and len(idx) == len(table) - 1 and idx.stale == 1
    assert idx.mat.dtype.name == "float32"
    hr.set_config_for_tests({"enabled": True, "max_index_mb": 1e-6 + 1})
    big = Loader(c, table)
    big.vectors = lambda ids, model: ((i, hashlib.sha256(c._texts[table[i][0]][table[i][1]:table[i][2]][:8000].encode()).hexdigest(),
                                       [0.1] * 200000) for i in ids)
    assert hr.build_index(c, MODEL, big) == (None, "index_too_large")


def test_no_index_means_word_only_and_no_embedding(monkeypatch):
    c, table = _corpus()
    fake = FakeEmbedder()
    emb.register("fakehybrid", lambda cfg: fake)
    tc = chunk_taxo.load_cfg()
    monkeypatch.setattr(hr, "_taxo_cfg", lambda: dict(tc, embedder="fakehybrid"))
    hr.set_config_for_tests({"enabled": True})

    class Empty(Loader):
        def vectors(self, ids, model):
            return iter(())
    hr.set_loader_for_tests(Empty(c, table))
    hr.ensure_index(c, MODEL, sync=True)
    out = c.retrieve(Q, 16, True)
    assert hr.last_status()["status"] == "no_vectors" and fake.calls == 0 and all("vec_sim" not in x for x in out)


# ------------------------------------------------------------------------------------------------ privacy of the question
def test_no_question_text_in_logs_or_trace(monkeypatch, caplog):
    marker = "zqxjmarkerquestion"
    q = f"Which agent archetypes are used in {marker} multi-agent orchestration?"
    c, table = _corpus()
    caplog.set_level(logging.DEBUG)
    fake, _ = setup_hybrid(monkeypatch, c, table)
    ok = fa.admit(q, c, load_config())
    fake.fail = RuntimeError(q)  # even an exception carrying the text must not leak it
    hr.reset_for_tests()
    setup_hybrid(monkeypatch, c, table, fake=fake)
    bad = fa.admit(q + " 2", c, load_config())
    assert bad.trace["hybrid"]["status"] == "embed_error:RuntimeError"
    blob = " ".join(r.getMessage() + " " + repr(r.args) for r in caplog.records)
    assert marker not in blob and "tok-secret" not in blob
    assert marker not in json.dumps(ok.trace) and marker not in json.dumps(bad.trace)
    assert all(marker not in k for k in hr._QCACHE)


# ------------------------------------------------------------------------------------------------ stage 2: passage re-rank
ONE_PART = None  # doc B stored as ONE partition: stage 1 alone cannot tell its role passages from its intro


def _rr_corpus(monkeypatch, fake=None, **rr):
    c, table = _corpus(spans_b=[(0, len(_texts()[1]))])
    fake, loader = setup_hybrid(monkeypatch, c, table, fake=fake, chunks_per_partition=1,
                                rerank=dict({"enabled": True, "top_partitions": 3, "max_passages": 48}, **rr))
    return c, table, fake


def test_rerank_off_by_default_and_stage1_unchanged(monkeypatch):
    assert hr._clean({})["rerank"]["enabled"] is False
    assert json.loads(hr.CONFIG_PATH.read_text())["rerank"]["enabled"] is False
    c, table = _corpus(spans_b=[(0, len(_texts()[1]))])
    setup_hybrid(monkeypatch, c, table, chunks_per_partition=1)
    out = c.retrieve(Q, 16, True)
    assert all("psg_sim" not in x for x in out) and "rerank" not in hr.last_status()


def test_rerank_admits_role_passage_inside_a_large_partition(monkeypatch):
    c, table, fake = _rr_corpus(monkeypatch)
    b0 = c._texts["DB"].index("Below are")
    hr.set_config_for_tests({"enabled": True, "chunks_per_partition": 1})
    stage1 = fa.admit(Q, c, load_config())
    assert not any(_role(a, b0) for a in stage1.admitted)
    hr.set_config_for_tests({"enabled": True, "chunks_per_partition": 1, "rerank": {"enabled": True}})
    on = fa.admit(Q, c, load_config())
    rr = on.trace["hybrid"]["rerank"]
    assert rr["status"] == "ok" and 0 < rr["passages"] <= 48 and rr["neurons_est"] > 0
    assert any(_role(a, b0) for a in on.admitted), on.trace
    assert fake.batches.count(1) >= 1 and max(fake.batches) <= 48  # one question call, one batched passage call


def test_rerank_single_batched_call_and_cache(monkeypatch):
    c, table, fake = _rr_corpus(monkeypatch)
    c.retrieve(Q, 16, True)
    batches = [b for b in fake.batches if b > 1]
    assert len(batches) == 1
    n = len(hr._PCACHE)
    c.retrieve(Q + " (again)", 16, True)  # new question vector, same passages: served from the passage cache
    assert [b for b in fake.batches if b > 1] == batches and len(hr._PCACHE) == n
    assert hr.last_status()["rerank"]["status"] == "ok"


def test_rerank_cache_is_bounded(monkeypatch):
    c, table, fake = _rr_corpus(monkeypatch, cache_max=5)
    c.retrieve(Q, 16, True)
    assert hr.last_status()["rerank"]["status"] == "ok" and hr.last_status()["rerank"]["passages"] > 5
    assert len(hr._PCACHE) == 5
    hr.set_config_for_tests({"enabled": True, "chunks_per_partition": 1, "rerank": {"enabled": True, "cache_max": 0}})
    hr._PCACHE.clear()
    c.retrieve(Q + "?", 16, True)
    assert len(hr._PCACHE) == 0


def test_rerank_error_falls_back_to_stage1(monkeypatch):
    c, table, fake = _rr_corpus(monkeypatch, fake=FakeEmbedder(fail_batch=emb.EmbedError("request_failed")))
    got = [x["span_id"] for x in c.retrieve(Q, 16, True)]
    st = hr.last_status()
    assert st["status"] == "ok" and st["rerank"]["status"] == "embed_error:request_failed"
    hr.set_config_for_tests({"enabled": True, "chunks_per_partition": 1})
    want = [x["span_id"] for x in c.retrieve(Q, 16, True)]
    assert got == want  # the stage-1 list, unchanged
    hr.set_config_for_tests({"enabled": True, "chunks_per_partition": 1, "rerank": {"enabled": True}})
    n = len(fake.batches)
    c.retrieve(Q + "!", 16, True)
    assert hr.last_status()["rerank"]["status"].startswith("cooldown:embed_error") and [b for b in fake.batches[n:] if b > 1] == []


def test_rerank_quota_and_ask_share(monkeypatch):
    c, table, fake = _rr_corpus(monkeypatch, fake=FakeEmbedder(fail_batch=emb.Deferred("quota_reached", 3600.0)))
    c.retrieve(Q, 16, True)
    assert hr.last_status()["rerank"]["status"] == "deferred:quota_reached"
    hr.reset_for_tests()
    c, table, fake = _rr_corpus(monkeypatch)
    tc = chunk_taxo.load_cfg()
    monkeypatch.setattr(hr, "_taxo_cfg", lambda: dict(tc, embedder="fakehybrid", cloudflare=dict(tc["cloudflare"], reserve_neurons=0.5)))
    c.retrieve(Q, 16, True)
    st = hr.last_status()["rerank"]
    assert st["status"] == "deferred:ask_share_reached" and st["neurons_est"] > 0.5
    assert [b for b in fake.batches if b > 1] == []  # nothing sent over the share


def test_neurons_estimate_bge_m3():
    class E:
        cfg = {"chars_per_token": 3.0}
        mcfg = {"neurons_per_m": 1075}
    est = hr.neurons_estimate(["x" * 400] * 48, E())
    assert abs(est - 48 * 400 / 3.0 * 1075 / 1e6) < 1e-9 and 6.8 < est < 6.9


def test_rerank_no_text_in_logs(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    marker = "zqxjrerankmarker"
    c, table, fake = _rr_corpus(monkeypatch, fake=FakeEmbedder(fail_batch=RuntimeError("Worker agents " + marker)))
    b = fa.admit(f"Which agent archetypes {marker} are used in multi-agent orchestration?", c, load_config())
    assert b.trace["hybrid"]["rerank"]["status"] == "embed_error:RuntimeError"
    blob = " ".join(r.getMessage() + " " + repr(r.args) for r in caplog.records)
    assert marker not in blob and "Worker agents" not in blob and marker not in json.dumps(b.trace)
    assert all(isinstance(k[0], str) and len(k[0]) == 64 for k in hr._PCACHE)
