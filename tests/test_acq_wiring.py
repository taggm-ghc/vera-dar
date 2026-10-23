"""WED-ACQ2 v4 wiring tests, OFFLINE: every layer of the acquisition path is the REAL function driven through a fake
TRANSPORT (never a mocked function), so a wiring defect like v1-v3's `max_retries=0` (zero provider calls) is caught.
  search  : real make_search -> search_and_clear -> vera.search_and_fetch.search_question -> real OpenAlexProvider with a
            fake HTTP transport;
  resolve : real resolve_oa / _guarded_get through `requests.get` replaced by a fake (the production http_get object);
  SSRF    : the real selective_fetch.is_public_url with socket.getaddrinfo faked;
  DB      : the real WoStore / RoLookup through a fake SQLAlchemy-style engine that checks every statement's bind names
            against the parameters and every INSERT / RETURNING against the write-only grants in
            db/accounts/create_vera_accounts.sql.
No network, no DB, no model."""
import json
import re
from pathlib import Path

import pytest

from tests import acq_support as A
from vera import acq_config, auto_acquire, fulldoc_corpus
from vera import search_providers as sp

ROOT = Path(__file__).resolve().parent.parent
DOI = "10.9999/acq.epi4"


@pytest.fixture(autouse=True)
def _reset():
    auto_acquire.reset_for_tests()
    fulldoc_corpus.reset_for_tests()
    yield
    auto_acquire.reset_for_tests()


class _HTTP:
    def __init__(self, body: dict):
        self.status_code, self.text, self.headers = 200, json.dumps(body), {"Content-Type": "application/json"}


def _openalex_work(doi=DOI):
    words = "SYNTHETIC outbreak transmission incidence pathogen study".split()
    return {"id": "https://openalex.org/W1", "doi": f"https://doi.org/{doi}", "title": " ".join(words),
            "publication_year": 2024, "publication_date": "2024-01-01", "type": "article",
            "primary_location": {"landing_page_url": f"https://example.org/{doi}", "source": {"display_name": "J"}},
            "abstract_inverted_index": {w: [i] for i, w in enumerate(words)},
            "authorships": [{"author": {"display_name": "A. Synth"}}]}


def _provider(calls):
    s = json.loads((ROOT / "config" / "vera_eval_run.json").read_text())["search"]

    def transport(url, params, headers, timeout):
        calls.append((url, dict(params)))
        return _HTTP({"results": [_openalex_work()], "meta": {"count": 1}})
    return sp.OpenAlexProvider(s["openalex"], transport=transport, query_terms=s.get("query_terms"))


def test_real_search_calls_the_provider_exactly_once_and_candidates_reach_the_resolver():
    calls = []
    prov = _provider(calls)
    web, _ = A.epi_setup()
    store = A.Store()
    d = A.deps([], web, store=store)
    d.search = auto_acquire.make_search(A.cfg(), chain_fn=lambda: [prov])  # REAL search_question underneath
    out = auto_acquire.run_job(A.job(), A.cfg(), d, clock=A.clock_at(), env=A.ENV_ON)
    assert len(calls) == 1 and calls[0][0] == "https://api.openalex.org/works", calls  # one real provider call
    assert any("/rest/search" in u and "acq.epi4" in u.replace("%2F", "/") for u in web.urls)  # reached the resolver
    assert out["stored"] == 1 and out["codes"] == {"ACQ-STORED": 1}, out
    assert prov.last_query is None  # this job's route record removed
    assert A.SENTINEL not in json.dumps(store.plans)


def test_real_search_with_zero_attempts_would_have_been_caught():
    """Regression guard for v1-v3: the attempt loop is range(1, max_retries + 1)."""
    import inspect
    src = inspect.getsource(auto_acquire.search_and_clear)
    m = re.search(r"max_retries=(\d+)", src)
    assert m and int(m.group(1)) >= 1


def test_real_resolver_and_fetch_through_the_production_http_object(monkeypatch):
    import requests
    seen = []
    web, _ = A.epi_setup()

    def fake_get(url, **kw):
        seen.append(kw)
        return web(url, **kw)
    monkeypatch.setattr(requests, "get", fake_get)
    d = A.deps([A.cand(DOI)], requests.get)  # the object default_deps passes in production
    d.http_get = requests.get
    out = auto_acquire.run_job(A.job(), A.cfg(), d, clock=A.clock_at(), env=A.ENV_ON)
    assert out["stored"] == 1, out
    assert len(seen) == 2 and all(k["allow_redirects"] is False and k["stream"] is True and k["timeout"] == 20.0
                                  and "User-Agent" in k["headers"] for k in seen)


def test_real_ssrf_guard_with_faked_dns(monkeypatch):
    import socket
    web, _ = A.epi_setup()
    d = A.deps([A.cand(DOI)], web)
    d.url_check = None  # default: the real selective_fetch.is_public_url
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("193.62.193.80", 443))])
    out = auto_acquire.run_job(A.job(), A.cfg(), d, clock=A.clock_at(), env=A.ENV_ON)
    assert out["stored"] == 1, out
    auto_acquire.reset_for_tests()
    web2, _ = A.epi_setup()
    d2 = A.deps([A.cand(DOI)], web2)
    d2.url_check = None
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("10.0.0.5", 443))])
    out = auto_acquire.run_job(A.job(), A.cfg(), d2, clock=A.clock_at(), env=A.ENV_ON)
    assert out["codes"] == {"R-ACQ-HOST": 1} and web2.urls == []


# ------------------------------------------------------------------------------------------------ DB through a fake engine
def _wo_grants():
    sql = (ROOT / "db" / "accounts" / "create_vera_accounts.sql").read_text()
    ins = set()
    for m in re.finditer(r"GRANT INSERT ON (.*?) TO vera_vjay_wo;", sql, re.S):
        ins |= {t.strip().split(".")[-1] for t in m.group(1).split(",")}
    sel = {}
    for m in re.finditer(r"GRANT SELECT \(([^)]*)\) ON vera_vjay\.(\w+) TO vera_vjay_wo;", sql):
        sel[m.group(2)] = {c.strip() for c in m.group(1).split(",")}
    return ins, sel


class _Engine:
    """SQLAlchemy-engine stand-in: runs nothing; checks binds, statement kind and (for the WO role) the grants."""

    def __init__(self, role, rows=None):
        self.role, self.rows, self.calls = role, rows or {}, []
        self.ins, self.sel = _wo_grants()

    def connect(self):
        return _Conn(self)

    def begin(self):
        return _Conn(self)


class _Conn:
    def __init__(self, eng):
        self.eng, self.n = eng, 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def begin_nested(self):
        class SP:
            def commit(self):
                pass

            def rollback(self):
                pass
        return SP()

    def execute(self, stmt, params=None):
        from sqlalchemy.sql.elements import TextClause
        assert isinstance(stmt, TextClause), "statements go through sqlalchemy.text"
        binds = set(stmt._bindparams)
        params = dict(params or {})
        assert binds == set(params), (binds, set(params), stmt.text)  # every bind supplied, nothing extra
        sql = stmt.text
        assert "{S}" not in sql and ("vera_vjay." in sql or "pg_advisory_xact_lock" in sql)
        e = self.eng
        if e.role == "wo":
            up = sql.lstrip().upper()
            assert up.startswith(("INSERT", "SELECT PG_ADVISORY_XACT_LOCK")), sql
            if up.startswith("INSERT"):
                table = re.search(r"INSERT INTO vera_vjay\.(\w+)", sql).group(1)
                assert table in e.ins, f"no INSERT grant on {table}"
                ret = re.search(r"RETURNING (\w+)", sql)
                if ret:
                    assert ret.group(1) in e.sel.get(table, set()), f"no SELECT grant on {table}.{ret.group(1)}"
        else:
            assert sql.lstrip().upper().startswith("SELECT"), sql
        e.calls.append((sql, params))
        self.n += 1
        key = next((k for k in e.rows if k in sql), None)
        val = e.rows.get(key, [(self.n,)])

        class R:
            def first(self_inner):
                return val[0] if val else None

            def fetchall(self_inner):
                return list(val)
        return R()


def _plan():
    web, _ = A.epi_setup()
    store = A.Store()
    auto_acquire.run_job(A.job(), A.cfg(), A.deps([A.cand(DOI)], web, store=store), clock=A.clock_at(), env=A.ENV_ON)
    return store.plans[0]


def test_real_wo_store_statements_fit_the_write_only_grants_and_binds():
    plan = _plan()
    wo, ro = _Engine("wo"), _Engine("ro", rows={"canonical_url = :u": [], "LIMIT 1": []})
    out = auto_acquire.WoStore(wo, ro, A.cfg()).write(plan)
    assert out["status"] == "stored"
    tables = [re.search(r"INSERT INTO vera_vjay\.(\w+)", s).group(1) for s, _ in wo.calls if s.lstrip().startswith("INSERT")]
    assert tables[:5] == ["questions", "runs", "candidates", "canonical_sources", "sources"]
    assert tables[5:] == ["evidence_spans"] * len(plan["spans"])
    q = [p for s, p in wo.calls if "INTO vera_vjay.questions" in s][0]
    assert q == {"q": acq_config.load_config()["question_text"]}
    src = [p for s, p in wo.calls if "INTO vera_vjay.sources" in s][0]
    assert json.loads(src["p"]) == plan["source"]["provenance"] and src["t"] == "" and src["v"] == 1


def test_real_ro_lookup_statements_and_result_parsing():
    ro = _Engine("ro", rows={"count(*)": [(3,)], "canonical_url = :u": [], "provenance->>'isbn'": [(9,)]})
    lk = auto_acquire.RoLookup(ro, A.cfg())
    assert lk.docs_last_24h() == 3
    assert lk.known(DOI, None, None) is False
    assert lk.known(DOI, None, "9780306406157") is True
    ro2 = _Engine("ro", rows={"canonical_source_id FROM vera_vjay.canonical_sources": [(7,)],
                               "max(version)": [(2, False)], "LIMIT 1": []})
    assert auto_acquire.RoLookup(ro2, A.cfg()).canonical_state("https://doi.org/" + DOI, "a" * 64) == (7, 2, False, False)


def test_production_composition_default_deps_end_to_end_through_fake_transports(monkeypatch):
    """default_deps itself (the live wiring): real make_search over the /ask chain, real requests.get object, real
    RoLookup/WoStore over engines from sqlalchemy.create_engine; only the transports below are fakes."""
    import requests
    import sqlalchemy

    from vera import search_and_fetch
    calls = []
    prov = _provider(calls)
    monkeypatch.setattr(search_and_fetch, "cached_chain", lambda *a, **k: ([prov], 1, "fan_out"))
    web, _ = A.epi_setup()
    monkeypatch.setattr(requests, "get", lambda url, **kw: web(url, **kw))
    from vera import selective_fetch  # CMF-1022-I2: the production transport is now the pinned getter
    monkeypatch.setattr(selective_fetch, "pinned_get", lambda url, **kw: web(url, **kw))
    import socket
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("193.62.193.80", 443))])
    engines = {}

    def create_engine(url, **kw):
        role = "wo" if "wo.invalid" in url else "ro"
        e = _Engine(role, rows={"count(*)": [(0,)], "canonical_url = :u": [], "LIMIT 1": []} if role == "ro" else None)
        e.dispose = lambda: engines.setdefault("disposed", []).append(role)
        engines[role] = e
        return e
    monkeypatch.setattr(sqlalchemy, "create_engine", create_engine)
    d = auto_acquire.default_deps(A.cfg(), A.ENV_ON)
    out = auto_acquire.run_job(A.job(), A.cfg(), d, clock=A.clock_at(), env=A.ENV_ON)
    d.close()
    assert out["stored"] == 1 and out["codes"] == {"ACQ-STORED": 1}, out
    assert len(calls) == 1 and prov.last_query is None
    assert any("INTO vera_vjay.sources" in s for s, _ in engines["wo"].calls)
    assert sorted(engines["disposed"]) == ["ro", "wo"]
