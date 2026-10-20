"""DARINT owner acceptance test (TASK DARINT-ACC), driven through the INTEGRATED application.

Owner requirement (verbatim): "The acceptance test should demonstrate that: Distinct subjects generate distinct
provisional domain associations. A document can associate with multiple domains where warranted. New content becomes
retrievable immediately, regardless of whether a domain has formed. /ask actually retrieves stored corpus evidence and
retains provenance. Cross-domain retrieval works through the integrated application."

Flow per test: DarCorpus.ingest_document -> vera.dar_store.persist -> (fake DB) -> vera.dar_runtime.get_corpus
(snapshot restore through DarReader, injected engine_factory) -> POST /ask through fastapi TestClient(main.app), where
main.py's real hookup (lazy dar_runtime.get_corpus() -> retrieve_all(dar_corpus=...)) is used unchanged.

DB: FAKE-ENGINE (NOT POSTGRES-INTEGRATED). The test boundary refuses real DB connections, so FakeDB below is an
in-memory engine that implements exactly the SQL dar_store.py issues. A separate engineer runs the same flow on
ephemeral Postgres outside the runner.

Overrides and stubs (all via monkeypatch, in-process; no config file is edited):
  * NO CONFIG OVERRIDE (owner D-041): the ACTUAL SHIPPED config/domain_routing.json is used (enabled=true,
    corpus.stored_branch_enabled=true); a test asserts both flags so a regression of the shipped value fails loudly.
  * SCOPE ROUTER: vera.scope_router.route is patched to an in-scope decision (scope widening is owner-held).
  * LLM: ask_service.complete returns a fixed cited text; claim_check.check_answer returns a fixed pass; the L3 seam
    answer_contract.examined_full_documents is opened (as tests/test_k2_integrated_contract.py does) so /ask yields
    result_kind="answer" and the evidence the response carries is exactly what retrieval selected.
  * EXTERNAL SEARCH: main.retrieve_all is wrapped so the real retrieve_all runs with a stub search (world.external,
    default []): no provider is contacted; with the default every returned source must come from the stored corpus.
  * main.store_from_env / memory_from_env / admit_to_corpus are stubbed (no corpus store, no memory, no background write).
  * VERA_DB_URL_RO is set to an unreachable .invalid URL; the injected engine_factory never connects to it.
"""
import copy
import functools
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

from vera import answer_contract as ac
from vera import ask_service, claim_check, dar_runtime, domain_registry, scope_router
from vera.ask_service import AskResult
from vera.dar_corpus import DarCorpus
from vera.dar_store import DarReader, DarWriter, persist
from vera.grounded_ask import load_grounding, retrieve_all

FIX = Path(__file__).resolve().parent / "fixtures" / "dar_acceptance"
RETRIEVED_AT = "2026-10-19T00:00:00+00:00"
KEY = "k" * 20
SCHEMA = "vera_vjay"


# ------------------------------------------------------------------ fake DB (fake-engine, not Postgres)
class _Res:
    def __init__(self, rows):
        self.rows = rows

    def first(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class _SP:
    def commit(self):
        pass

    def rollback(self):
        pass


class _Conn:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def begin_nested(self):
        return _SP()

    def execute(self, stmt, params=None):
        return _Res(self.db.run(" ".join(str(stmt).split()), dict(params or {})))


class FakeDB:
    """In-memory tables + the exact statements of vera/dar_store.py. Doubles as engine (connect/begin/dispose)."""

    def __init__(self):
        self.canonical, self.sources = {}, []          # canonical_source_id -> url ; source rows
        self.domains, self.spans, self.dcorp, self.dchunks = [], [], [], []
        self.disposed = 0

    # engine surface
    def connect(self):
        return _Conn(self)

    begin = connect

    def dispose(self):
        self.disposed += 1

    def add_source(self, n, d):
        self.canonical[n] = d["url"]
        self.sources.append({"source_id": n, "canonical_source_id": n, "version": 1, "content": d["content"],
                             "prov": {"url": d["url"], "doi": d["doi"], "title": d["title"],
                                      "published": str(d["year"]), "source_type": d["source_type"],
                                      "retrieved_at": RETRIEVED_AT, "stored_text_kind": d["stored_text_kind"]}})

    def _dup(self):
        return IntegrityError("stmt", {}, Exception("duplicate key"))

    def run(self, sql, p):  # noqa: C901  (one dispatcher per statement shape)
        sql = sql.replace(SCHEMA + ".", "")
        if sql.startswith("INSERT INTO research_domains"):
            if any(d["name_norm"] == p["nn"] for d in self.domains):
                raise self._dup()
            self.domains.append({"domain_id": len(self.domains) + 1, "name": p["n"], "name_norm": p["nn"],
                                 "description": p["d"], "origin": p["o"]})
            return [(len(self.domains),)]
        if sql.startswith("INSERT INTO evidence_spans"):
            if any((s["source_id"], s["start"], s["end"]) == (p["s"], p["a"], p["b"]) for s in self.spans):
                raise self._dup()
            self.spans.append({"span_id": len(self.spans) + 1, "source_id": p["s"], "text": p["t"],
                               "start": p["a"], "end": p["b"]})
            return [(len(self.spans),)]
        if sql.startswith("INSERT INTO domain_corpus"):
            if any((r["d"], r["c"]) == (p["d"], p["c"]) for r in self.dcorp):
                raise self._dup()
            self.dcorp.append({"d": p["d"], "c": p["c"], "by": p["a"]})
            return [(1,)]
        if sql.startswith("INSERT INTO domain_chunks"):
            if any((r["d"], r["s"]) == (p["d"], p["s"]) for r in self.dchunks):
                raise self._dup()
            self.dchunks.append({"d": p["d"], "s": p["s"], "by": p["a"]})
            return [(1,)]
        assert sql.startswith("SELECT"), sql  # the store never UPDATEs/DELETEs
        if sql.startswith("SELECT domain_id FROM research_domains WHERE name_norm"):
            return [(d["domain_id"],) for d in self.domains if d["name_norm"] == p["n"]]
        if sql.startswith("SELECT span_id FROM evidence_spans WHERE source_id"):
            return [(s["span_id"],) for s in self.spans
                    if (s["source_id"], s["start"], s["end"]) == (p["s"], p["a"], p["b"])]
        if sql.startswith("SELECT canonical_source_id FROM canonical_sources WHERE canonical_url"):
            return [(c,) for c, u in self.canonical.items() if u == p["u"]]
        if sql.startswith("SELECT source_id FROM sources WHERE canonical_source_id"):
            rows = sorted((s for s in self.sources if s["canonical_source_id"] == p["c"]), key=lambda s: -s["version"])
            return [(s["source_id"],) for s in rows[:1]]
        if sql.startswith("SELECT domain_id, name, name_norm, description, origin FROM research_domains"):
            return [(d["domain_id"], d["name"], d["name_norm"], d["description"], d["origin"]) for d in self.domains]
        if "SELECT DISTINCT ON (cs.canonical_source_id)" in sql:
            out = []
            for s in sorted(self.sources, key=lambda s: (s["canonical_source_id"], -s["version"])):
                kind = s["prov"].get("stored_text_kind")
                if kind is None or kind in p["kinds"] or any(o[2] == s["canonical_source_id"] for o in out):
                    continue
                pr = s["prov"]
                out.append((self.canonical[s["canonical_source_id"]], s["source_id"], s["canonical_source_id"],
                            s["content"], pr["url"], pr["doi"], pr["title"], pr["published"], pr["source_type"],
                            pr["retrieved_at"], kind))
            return out
        if sql.startswith("SELECT d.name_norm, cs.canonical_url, dc.asserted_by"):
            nm = {d["domain_id"]: d["name_norm"] for d in self.domains}
            return [(nm[r["d"]], self.canonical[r["c"]], r["by"]) for r in sorted(self.dcorp, key=lambda r: (r["d"], r["c"]))]
        if sql.startswith("SELECT d.name_norm, cs.canonical_url, es.start_index"):
            nm = {d["domain_id"]: d["name_norm"] for d in self.domains}
            sp = {s["span_id"]: s for s in self.spans}
            sid = {s["source_id"]: s["canonical_source_id"] for s in self.sources}
            out = []
            for r in sorted(self.dchunks, key=lambda r: (r["d"], r["s"])):
                s = sp[r["s"]]
                if p["span_text"] is not None and s["start"] == 0 and s["text"] == p["span_text"]:
                    continue
                out.append((nm[r["d"]], self.canonical[sid[s["source_id"]]], s["start"], s["end"], r["by"]))
            return out
        raise AssertionError("FakeDB: unsupported SQL: " + sql)


# ------------------------------------------------------------------ world builders
def _load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


DEMO9, ECO, CROSS = _load("demo9.json"), _load("eco1.json"), _load("cross.json")
QUERIES = {q["query_id"]: q["query"] for q in _load("queries.json")}
BY_URL = {d["url"]: d for d in DEMO9 + ECO + CROSS}
CROSS_URL = CROSS[0]["url"]
ECO_URL = ECO[0]["url"]


def _cfg():
    return copy.deepcopy(_REAL_CFG)                 # shipped config, unmodified


_REAL_CFG = domain_registry.load_config()


class World:
    def __init__(self, cfg):
        self.cfg, self.db, self.ctx, self.n = cfg, FakeDB(), {}, 0
        self.ingest = DarCorpus(cfg)                    # the ingest-side corpus (separate from the serving corpus)
        self.span_text = dar_runtime._span_text(ac.CHAIN_CONFIG_PATH)

    def add(self, d):
        """ingest_document -> persist, for one fixture document."""
        self.n += 1
        self.db.add_source(self.n, d)
        plan = self.ingest.ingest_document({
            "source_key": d["url"], "url": d["url"], "doi": d["doi"], "title": d["title"], "text": d["content"],
            "year": d["year"], "published": str(d["year"]), "source_type": d["source_type"],
            "retrieved_at": RETRIEVED_AT, "stored_text_kind": d["stored_text_kind"],
            "admission_passed": d["admission_passed"], "db_source_id": self.n, "db_canonical_source_id": self.n})
        assert plan["status"] == "indexed", plan
        res = persist(plan, DarWriter(self.db, reader=DarReader(self.db), span_text=self.span_text),
                      span_text=self.span_text, ctx=self.ctx)
        assert res["status"] == "persisted", res
        return plan


@pytest.fixture
def app_env(monkeypatch):
    """Patches shared by every test; returns a function that builds the base world (demo9 + cross doc)."""
    import main
    from vera import public_mode as pm
    from tests.test_item79_claim_check import PRICING

    cfg = _cfg()
    assert cfg.get("enabled") is True and cfg["corpus"].get("stored_branch_enabled") is True, "shipped flags not ON"
    monkeypatch.setenv("VERA_API_KEY", KEY)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)                      # fake; every model call is stubbed
    monkeypatch.setenv("VERA_DB_URL_RO", "postgresql://ro:x@db.invalid:5432/none")  # never connected
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    pm.reset_limiter(None)
    dar_runtime.reset_for_tests()

    world = World(cfg)
    world.external = []
    factory_calls = []

    def factory(url, **kw):
        factory_calls.append(url)
        return world.db

    # main.py calls dar_runtime.get_corpus() with no arguments: inject the engine factory at the module seam only
    orig_get = dar_runtime.get_corpus
    monkeypatch.setattr(dar_runtime, "get_corpus", functools.partial(orig_get, engine_factory=factory))
    real_retrieve = retrieve_all
    grounding = dict(load_grounding())
    monkeypatch.setattr(main, "GROUNDING", grounding)
    monkeypatch.setattr(main, "retrieve_all",
                        lambda q, cfg_, **kw: real_retrieve(q, cfg_, search=lambda question: [dict(r) for r in world.external], **kw))
    monkeypatch.setattr(scope_router, "route", lambda *a, **k: scope_router.ScopeDecision(True, "layer1", "ok"))
    monkeypatch.setattr(ac, "examined_full_documents", lambda c: ["examined"])
    monkeypatch.setattr(ask_service, "complete", lambda *a, **k: AskResult("Stored evidence [1] [2] [3].", 10, 0.0))
    monkeypatch.setattr(claim_check, "check_answer", lambda t, s, c: (
        "Stored evidence [1] [2] [3].", {"outcome": "checked", "checked": 1, "supported": 1, "partial": 0,
                                         "removed": 0, "detail": "stub"}))
    monkeypatch.setattr(main, "answer_via_chain", lambda q, entries, g: (
        ask_service.answer_in_scope(q, "m", PRICING, grounding=g), "stub:m"))
    monkeypatch.setattr(main, "store_from_env", lambda c: None)
    monkeypatch.setattr(main, "memory_from_env", lambda block: None)
    monkeypatch.setattr(main, "admit_to_corpus", lambda *a: None)
    monkeypatch.setattr(main, "llm_call_for", lambda e: None)

    def build(with_cross=True):
        for d in DEMO9 + (CROSS if with_cross else []):
            world.add(d)
        return world

    world.client = TestClient(main.app)
    world.factory_calls, world.factory = factory_calls, factory
    world.build = build
    return world


def _ask(world, question):
    r = world.client.post("/ask", json={"question": question}, headers={"X-API-Key": KEY})
    assert r.status_code == 200, r.text
    return r.json()


def _serving(world):
    corpus = dar_runtime.get_corpus()
    assert corpus is not None, "dar_runtime.get_corpus() returned None (DAR off or restore failed)"
    return corpus


def _members(corpus):
    """{domain_id: [source urls]} from the restored (persisted) associations."""
    return {d.domain_id: sorted(corpus.registry.corpora_for_domain(d.domain_id)) for d in corpus.registry.list_domains()}


def _fx(url):
    return BY_URL[url]["fixture_domain"]


def _src_urls(body):
    return [s["url"] for s in body["sources"]]


# ------------------------------------------------------------------ the five owner criteria
def test_crit1_distinct_subjects_distinct_domains(app_env):
    w = app_env
    w.build()
    corpus = _serving(w)
    members = _members(corpus)
    demo_domains = {did: [u for u in us if u != CROSS_URL] for did, us in members.items()}
    demo_domains = {did: us for did, us in demo_domains.items() if us}
    assert len(demo_domains) >= 3, f"expected >=3 domains with demo9 members, got {len(demo_domains)}: {demo_domains}"
    for did, us in demo_domains.items():       # no mixing (cross doc excluded)
        subjects = {_fx(u) for u in us}
        assert len(subjects) == 1, f"domain {did} mixes subjects {sorted(subjects)}: {us}"
    for d in DEMO9:                            # every document is associated, and each subject has its own domains
        assert corpus.registry.domains_for_corpus(d["url"]), f"{d['doc_id']} has no domain"
    per_subject = {}
    for did, us in demo_domains.items():
        per_subject.setdefault(_fx(us[0]), set()).add(did)
    assert set(per_subject) == {"D1", "D2", "D3"}, per_subject
    status = {did: corpus.domain_status(did) for did in demo_domains}
    assert all(s in ("provisional", "established", "configured") for s in status.values()), status


def test_crit2_cross_subject_doc_has_multiple_domains(app_env):
    w = app_env
    w.build()
    corpus = _serving(w)
    doms = [d.domain_id for d in corpus.registry.domains_for_corpus(CROSS_URL)]
    assert len(doms) >= 2, f"cross-subject doc X-1 associated with {len(doms)} domain(s): {doms}"
    subjects = set()
    for did in doms:
        subjects |= {_fx(u) for u in corpus.registry.corpora_for_domain(did) if u != CROSS_URL}
    assert len(subjects) >= 2, f"X-1's domains only relate to subjects {sorted(subjects)}"


def test_crit3_new_content_retrievable_immediately_without_restart(app_env):
    w = app_env
    w.build()
    q = QUERIES["q_D4"]
    before = _ask(w, q)
    assert ECO_URL not in _src_urls(before), "eco doc present before it was ingested"
    w.add(ECO[0])                                      # new content arrives (ingest + persist); app NOT restarted
    assert hasattr(dar_runtime, "refresh"), "no dar_runtime refresh mechanism exists"
    dar_runtime.refresh(engine_factory=w.factory)      # the runtime refresh mechanism (forced rebuild from the DB)
    after = _ask(w, q)                                 # same TestClient / same main.app instance
    assert ECO_URL in _src_urls(after), f"new eco doc not retrievable via /ask right after refresh: {_src_urls(after)}"
    # regardless of domain formation: the domain-independent (unrestricted) branch finds the chunk
    cands = _serving(w).retrieve(q)["candidates"]
    eco = [c for c in cands if c["source_key"] == ECO_URL]
    from vera import domain_routing
    assert eco and domain_routing.FALLBACK in eco[0]["branches"], eco


def test_crit4_ask_returns_stored_evidence_with_provenance(app_env):
    w = app_env
    w.build()
    body = _ask(w, QUERIES["q_D1"])
    assert body["result_kind"] == "answer", body["result_kind"]
    assert body["sources"], "no sources in the /ask response"
    for s in body["sources"]:
        d = BY_URL.get(s["url"])
        assert d is not None, f"source is not from the stored corpus: {s}"   # external search returned nothing
        assert s["title"] == d["title"], s
        assert s["identifier"] == "doi:" + d["doi"].lower(), s
        assert s["retrieved_at"] == RETRIEVED_AT, s
        assert s["url"].startswith("https://fixtures.invalid/"), s
        required = ("url", "title", "identifier", "retrieved_at", "canonical_source_id", "source_id", "span",
                    "admission_passed", "acquisition", "licence_decision", "provider")
        missing = [k for k in required if s.get(k) in (None, "")]
        assert not missing, f"stored-evidence item lacks fields {missing}; emitted keys={sorted(s)}"
        assert s["provider"] == "stored_corpus", s["provider"]
        assert s["admission_passed"] is True and s["acquisition"] == ac.ACQ_ACQUIRED, (s["admission_passed"], s["acquisition"])
        assert s["licence_decision"] not in ("unknown", "unverified"), s["licence_decision"]
        assert re.search(r"\d+\s*-\s*\d+", str(s["span"])), s["span"]


def test_crit5_cross_domain_question_returns_two_domains(app_env):
    w = app_env
    w.build()
    body = _ask(w, QUERIES["q_cross_D1_D3"])
    urls = [u for u in _src_urls(body) if u != CROSS_URL]
    assert urls, "no stored sources returned"
    subjects = {_fx(u) for u in urls}
    assert len(subjects) >= 2, f"evidence from only {sorted(subjects)}: {urls}"
    corpus = _serving(w)
    dar_domains = set()
    for u in urls:
        dar_domains |= {d.domain_id for d in corpus.registry.domains_for_corpus(u)}
    assert len(dar_domains) >= 2, f"returned evidence spans only DAR domains {sorted(dar_domains)}"


def test_crit6_nonexclusion_external_base_not_displaced(app_env):
    w = app_env
    w.build()
    w.external = [{"title": f"External Result {i}", "url": f"https://external.example/r{i}", "doi": None,
                   "snippet": f"External abstract text number {i} about forecasting uncertainty and models.",
                   "published": "2025-01-01", "source_type": "journal_article"} for i in range(1, 6)]
    ext = [r["url"] for r in w.external]
    urls = _src_urls(_ask(w, QUERIES["q_D1"]))
    stored = [u for u in urls if u in BY_URL]
    assert stored, f"no stored results in sources: {urls}"
    assert [u for u in urls if u in ext], f"no external results in sources: {urls}"
    assert urls[:3] == ext[:3], f"base ranks 1-3 displaced: got {urls[:3]}, expected {ext[:3]}"
