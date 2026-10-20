"""Item #84 D6: operator-only confirmation of memory writes (retires D-029). No DB: fake engines/readers/stores."""
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vera import memory_store as ms
from vera.memory_gate import load_gate_config
from vera.memory_operator import OWNER_KEY_MIN_LENGTH, classify_owner_key
from vera.memory_store import (CONFIRMED, NOT_ALLOWED, NOT_FOUND, REJECTED, MemoryService, MemoryUnavailable,
                               RoMemoryReader, WoMemoryStore)

ROOT = Path(__file__).resolve().parent.parent
_SHIPPED = json.loads((ROOT / "config" / "ask-provider-chain.json").read_text())["memory"]
# R-028 (2026-10-19): the shipped config has memory.store.enabled=false; these tests exercise the store/operator code itself,
# so they run it enabled (only that flag differs from the shipped block; before: BLOCK was the shipped block as-is).
BLOCK = {**_SHIPPED, "store": {**_SHIPPED["store"], "enabled": True}}
CFG = BLOCK["store"]
OWNER = "o" * 8 + "K3y-for-tests-only-0123456789abcdef"
API = "api-key-for-tests-0123456789"
QUESTION = "What does quantized inference change about transformer latency?"
SRC = [{"n": 1, "url": "https://arxiv.org/abs/2401.00001", "provider": "arxiv", "identifier": "arXiv:2401.00001",
        "licence_decision": "allow", "title": "T"}]
GOOD = {"id": 0, "sentence": "Int8 weight quantization reduced memory use by roughly half in the cited study [1].",
        "citations": [1], "verdict": "supported"}


# --- fakes -------------------------------------------------------------------------------------------------
class _Res:
    def __init__(self, rows): self.rows = rows
    def fetchall(self): return self.rows
    def scalar_one(self): return self.rows[0][0]
    def first(self): return self.rows[0] if self.rows else None


class FakeEngine:
    """Records (sql, params); returns `rows` for every statement."""
    def __init__(self, rows=()):
        self.calls, self.rows = [], list(rows)

    def _conn(self):
        eng = self

        class C:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, sql, params=None):
                eng.calls.append((str(sql), dict(params or {})))
                return _Res(eng.rows)
            def begin_nested(self): return self
        return C()

    def connect(self): return self._conn()
    def begin(self): return self._conn()


class StateReader:
    def __init__(self, status=None, pending_n=0, expired=False):
        self.status, self.pending_n, self.expired = status, pending_n, expired
    def claim_status(self, cid): return None if self.status is None else (self.status, self.expired)
    def pending(self): return [{"claim_id": 1, "claim_text": "x", "created_at": "", "state": "pending", "sources": []}]
    def pending_count(self): return self.pending_n
    def source(self, u): return (7, 100)
    def span(self, s, e): return 11
    def answer(self): return 5
    def duplicate(self, t, s): return False


class FakeStore:
    def __init__(self): self.claims, self.updates = [], []
    def span_id(self, s, e): return 11
    def answer_id(self): return 5
    def add_claim(self, a, t, s): self.claims.append((a, t, s)); return 1
    def set_status(self, cid, st): self.updates.append((cid, st)); return cid


def svc(reader=None, store=None, **over):
    return MemoryService(dict(CFG, **over), load_gate_config(), reader or StateReader(), store or FakeStore())


# --- config and write path ---------------------------------------------------------------------------------
def test_config_confirmation_on_and_distinct_labels():
    assert CFG["confirm_required"] is True
    labels = [CFG["status_live"], CFG["status_pending"], CFG["status_rejected"], CFG["status_tombstone"]]
    assert len(set(labels)) == 4
    assert CFG["status_live"] not in CFG["status_legacy_auto"]  # legacy auto rows are no longer recalled


def test_new_rows_are_pending_by_default_and_when_key_missing():
    eng = FakeEngine(rows=[(9,)])
    WoMemoryStore(eng, CFG).add_claim(5, "Claim.", [11])
    assert eng.calls[-1][1]["d"] == CFG["status_pending"]
    cfg = {k: v for k, v in CFG.items() if k != "confirm_required"}  # deny-by-default
    WoMemoryStore(eng, cfg).add_claim(5, "Claim.", [11])
    assert eng.calls[-1][1]["d"] == CFG["status_pending"]
    WoMemoryStore(eng, dict(CFG, confirm_required=False)).add_claim(5, "Claim.", [11])
    assert eng.calls[-1][1]["d"] == CFG["status_live"]
    assert ms.confirm_required({"confirm_required": "false"}) is True  # only a real false switches it off


def test_recall_reads_confirmed_only():
    eng = FakeEngine(rows=[])
    RoMemoryReader(eng, CFG).recall(QUESTION)
    sql, params = eng.calls[-1]
    assert "c.detailed_status = :live" in sql and params["live"] == CFG["status_live"]


def test_duplicate_covers_every_known_state_so_rejected_is_never_requeued():
    eng = FakeEngine(rows=[])
    RoMemoryReader(eng, CFG).duplicate("Claim.", [11])
    sts = eng.calls[-1][1]["sts"]
    for s in (CFG["status_pending"], CFG["status_live"], CFG["status_rejected"], CFG["status_tombstone"],
              *CFG["status_legacy_auto"]):
        assert s in sts


def test_public_write_lands_pending_and_queue_cap_skips():
    store = FakeStore()
    assert svc(store=store).write_claims([GOOD], SRC, QUESTION) == (1, {})
    assert len(store.claims) == 1
    store = FakeStore()
    n, skipped = svc(reader=StateReader(pending_n=CFG["max_pending"]), store=store).write_claims([GOOD], SRC, QUESTION)
    assert n == 0 and skipped == {ms.SKIP_QUEUE_FULL: 1} and not store.claims


# --- transitions -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("status,action,outcome,target", [
    ("memory:g1:pending", "confirm", CONFIRMED, "status_live"),
    ("memory:g1", "confirm", CONFIRMED, "status_live"),            # legacy auto row
    ("memory:g1:confirmed", "confirm", NOT_ALLOWED, None),
    ("memory:rejected", "confirm", NOT_ALLOWED, None),
    ("memory:tombstoned", "confirm", NOT_ALLOWED, None),
    (None, "confirm", NOT_FOUND, None),                            # not a memory row
    ("something:else", "reject", NOT_FOUND, None),
    ("memory:g1:pending", "reject", REJECTED, "status_rejected"),
    ("memory:g1:confirmed", "reject", REJECTED, "status_rejected"),
    ("memory:rejected", "reject", NOT_ALLOWED, None),
])
def test_transitions(status, action, outcome, target):
    store = FakeStore()
    assert getattr(svc(reader=StateReader(status), store=store), action)(42) == outcome
    assert store.updates == ([(42, CFG[target])] if target else [])


def test_operator_actions_need_both_accounts():
    s = MemoryService(CFG, load_gate_config(), StateReader("memory:g1:pending"), None)
    with pytest.raises(MemoryUnavailable):
        s.confirm(1)
    with pytest.raises(MemoryUnavailable):
        s.list_pending()


def test_wo_sql_is_update_by_claim_id_only_and_ro_reads_are_select():
    eng = FakeEngine(rows=[(42,)])
    WoMemoryStore(eng, CFG).set_status(42, CFG["status_live"])
    sql = eng.calls[-1][0]
    assert sql.startswith("UPDATE") and "WHERE claim_id = :id RETURNING claim_id" in sql
    assert "detailed_status =" in sql.split("WHERE")[0] and "detailed_status" not in sql.split("WHERE")[1]
    ro = FakeEngine(rows=[])
    r = RoMemoryReader(ro, CFG)
    r.pending(); r.pending_count(); r.claim_status(1)
    for sql, _ in ro.calls:
        assert sql.lstrip().upper().startswith("SELECT")
    src = Path(ms.__file__).read_text()
    assert not re.search(r"\bDELETE\s+FROM\b", src, re.I) and "TRUNCATE" not in src.upper()


def test_pending_listing_shape():
    from datetime import datetime, timezone
    row = (3, "Claim.", datetime(2026, 10, 10, tzinfo=timezone.utc), "memory:g1", False,
           [{"title": "T", "url": "https://x"}])
    out = RoMemoryReader(FakeEngine(rows=[row]), CFG).pending()
    assert out == [{"claim_id": 3, "claim_text": "Claim.", "created_at": "2026-10-10T00:00:00+00:00",
                    "state": "legacy_auto", "expired": False, "sources": [{"title": "T", "url": "https://x"}]}]


# --- owner key ---------------------------------------------------------------------------------------------
def test_classify_owner_key():
    assert classify_owner_key(None) == "missing"
    assert classify_owner_key("  ") == "missing"
    assert classify_owner_key("changeme") == "placeholder"
    assert classify_owner_key("x" * (OWNER_KEY_MIN_LENGTH - 1)) == "weak"
    assert classify_owner_key(OWNER, OWNER) == "same_as_api_key"
    assert classify_owner_key(OWNER, API) == "ok"


@pytest.fixture
def client(monkeypatch):
    import main
    calls = []

    class Svc:
        def list_pending(self): calls.append("list"); return [{"claim_id": 1, "claim_text": "C.", "state": "pending"}]
        def confirm(self, cid): calls.append(("confirm", cid)); return {1: CONFIRMED, 2: NOT_ALLOWED}.get(cid, NOT_FOUND)
        def reject(self, cid): calls.append(("reject", cid)); return REJECTED

    monkeypatch.setattr(main, "memory_from_env", lambda block: Svc())
    monkeypatch.setenv("VERA_API_KEY", API)
    return TestClient(main.app), calls


ROUTES = [("get", "/memory/pending"), ("post", "/memory/1/confirm"), ("post", "/memory/1/reject")]


@pytest.mark.parametrize("method,path", ROUTES)
@pytest.mark.parametrize("owner_env", [None, "", "changeme", "short-key", "SAME"])
def test_routes_disabled_unless_strong_distinct_owner_key(client, monkeypatch, method, path, owner_env):
    c, calls = client
    if owner_env is None:
        monkeypatch.delenv("VERA_OWNER_KEY", raising=False)
    else:
        monkeypatch.setenv("VERA_OWNER_KEY", API if owner_env == "SAME" else owner_env)
    r = getattr(c, method)(path, headers={"X-Owner-Key": API, "X-API-Key": API})
    assert r.status_code == 503 and calls == []
    assert API not in r.text


@pytest.mark.parametrize("method,path", ROUTES)
def test_routes_refuse_api_key_and_wrong_owner_key(client, monkeypatch, method, path):
    c, calls = client
    monkeypatch.setenv("VERA_OWNER_KEY", OWNER)
    for headers in ({}, {"X-API-Key": API}, {"X-Owner-Key": API}, {"X-Owner-Key": OWNER[:-1]}):
        r = getattr(c, method)(path, headers=headers)
        assert r.status_code == 401, headers
    assert calls == []


def test_operator_happy_path_and_errors(client, monkeypatch):
    c, calls = client
    monkeypatch.setenv("VERA_OWNER_KEY", OWNER)
    h = {"X-Owner-Key": OWNER}
    r = c.get("/memory/pending", headers=h)
    assert r.status_code == 200 and r.json()["pending"][0]["claim_id"] == 1
    assert c.post("/memory/1/confirm", headers=h).json() == {"claim_id": 1, "outcome": "confirmed"}
    assert c.post("/memory/2/confirm", headers=h).status_code == 409
    assert c.post("/memory/3/confirm", headers=h).status_code == 404
    assert c.post("/memory/5/reject", headers=h).json()["outcome"] == "rejected"
    assert c.post("/memory/abc/reject", headers=h).status_code == 422


def test_store_unavailable_is_503_without_detail(client, monkeypatch):
    import main
    c, _ = client
    monkeypatch.setenv("VERA_OWNER_KEY", OWNER)
    monkeypatch.setattr(main, "memory_from_env", lambda block: None)
    assert c.get("/memory/pending", headers={"X-Owner-Key": OWNER}).status_code == 503

    class Boom:
        def list_pending(self): raise RuntimeError("postgresql://secret-host/db")
    monkeypatch.setattr(main, "memory_from_env", lambda block: Boom())
    r = c.get("/memory/pending", headers={"X-Owner-Key": OWNER})
    assert r.status_code == 503 and "secret-host" not in r.text


def test_every_operator_route_declares_the_owner_dependency():
    import main
    for route in main.app.routes:
        if getattr(route, "path", "").startswith("/memory"):
            deps = [d.call for d in route.dependant.dependencies]
            assert main.verify_owner_key in deps, route.path
            assert main.verify_api_key not in deps, route.path  # the public key never opens these routes


def test_review_panel_only_in_dev_mode_and_never_reads_env_key():
    src = (ROOT / "streamlit_app.py").read_text()
    public_branch, dev_branch = src.split("if PUBLIC:\n", 1)[1].split("else:\n", 1)
    assert "render_memory_review" not in public_branch
    assert "render_memory_review(st, api_base_url)" in dev_branch.split("# Force bad demo toggle")[0]
    panel = (ROOT / "vera" / "memory_review_panel.py").read_text()
    assert "os.environ" not in panel and "getenv" not in panel and "session_state" not in panel


def test_confirm_refused_past_ttl_but_reject_allowed():
    store = FakeStore()
    assert svc(reader=StateReader("memory:g1:pending", expired=True), store=store).confirm(7) == NOT_ALLOWED
    assert svc(reader=StateReader("memory:g1:pending", expired=True), store=store).reject(7) == REJECTED
    assert store.updates == [(7, CFG["status_rejected"])]


def test_missing_d6_keys_fail_at_load():
    ms._SERVICE.clear()
    bad = dict(BLOCK, store={k: v for k, v in CFG.items() if k != "max_pending"})
    with pytest.raises(ValueError, match="max_pending"):
        ms.memory_from_env(bad, env={"VERA_DB_URL_RO": "x"}, engine_factory=lambda *a, **k: FakeEngine())
    ms._SERVICE.clear()


def test_non_ascii_owner_header_is_401_and_padded_api_key_reuse_is_503(client, monkeypatch):
    c, calls = client
    monkeypatch.setenv("VERA_OWNER_KEY", OWNER)
    r = c.get("/memory/pending", headers={"X-Owner-Key": ("\u00e9" * 40).encode("latin-1")})
    assert r.status_code == 401
    monkeypatch.setenv("VERA_API_KEY", "  " + OWNER + " ")
    assert c.get("/memory/pending", headers={"X-Owner-Key": OWNER}).status_code == 503
    assert calls == []
