"""Item #97 H-23 (VERA-DAR side): licence observations at admission, behind a default-off flag. Fakes only; no DB."""
import json
import re
from pathlib import Path

from vera import corpus_admission as ca
from vera.search_providers import OPENALEX_METADATA_LICENCE, licence_record
from tests.test_item79_corpus_admission import CFG, CC0, FakeStore, _llm, _src

ROOT = Path(__file__).resolve().parent.parent
MIG = ROOT / "db" / "migrations" / "pending" / "006_vera_licence_observations.sql"
ROLLBACK = ROOT / "db" / "migrations" / "rollback" / "006_vera_licence_observations_rollback.sql"


class ObsStore(FakeStore):
    def __init__(self, fail=False):
        super().__init__()
        self.obs, self.fail = [], fail

    def record_licence_observations(self, cand, run_id=None):
        if self.fail:
            raise RuntimeError("db down")
        self.obs.append(ca.licence_observation_rows(cand, run_id))


def _run(cfg, store):
    return ca.admit("q", [_src(1, "https://example.org/a"), _src(2, "https://example.org/b", licence=licence_record())],
                    _llm([0.9, 0.9]), cfg, store)


def test_flag_default_off_and_in_config():
    assert json.loads((ROOT / "config" / "ask-provider-chain.json").read_text())["corpus"]["licence_observations_enabled"] is True  # R1 decision 16 (2026-10-10): switched on


def test_flag_off_writes_no_observations():
    st = ObsStore()
    counts = _run(CFG, st)
    assert st.obs == [] and "licence_obs_written" not in counts and len(st.rows) == 2


def test_flag_on_one_pair_of_rows_per_candidate_and_nulls_when_undetermined():
    st = ObsStore()
    counts = _run({**CFG, "licence_observations_enabled": True}, st)
    assert counts["licence_obs_written"] == 2 and len(st.obs) == 2
    first, second = st.obs
    assert [r["licence_scope"] for r in first] == ["metadata", "content"]
    assert first[0]["governing"] is True and first[0]["spdx"] == CC0["metadata"]["id"] and first[0]["decision"] == "allow"
    assert first[0]["url"] == "https://example.org/a" and first[0]["host"] == "example.org"
    assert second[0]["spdx"] is None and second[0]["decision"] == "hold" and second[0]["url"] is None
    assert all(len(r["url_sha256"]) == 64 and r["nc"] is None and r["family"] is None for r in first + second)
    assert not any(k in first[0] for k in ("title", "snippet", "text", "content"))


def test_db_failure_never_blocks_admission():
    st = ObsStore(fail=True)
    counts = _run({**CFG, "licence_observations_enabled": True}, st)
    assert counts["licence_obs_failed"] == 2 and len(st.rows) == 2 and "error" not in counts


def test_wo_store_inserts_via_fake_engine_plain_insert():
    class Conn:
        def __init__(self, log): self.log = log
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, stmt, params): self.log.append((str(stmt), params))

    class Eng:
        def __init__(self): self.log = []
        def begin(self): return Conn(self.log)

    eng = Eng()
    n = ca.WoCorpusStore(eng, "q").record_licence_observations({"url": "https://example.org/a", "licence": CC0,
        "licence_decision": "allow", "gate_a_decision": "fetch", "stored_text_kind": "abstract_metadata"}, 7)
    assert n == 2 and len(eng.log) == 2
    sql = eng.log[0][0]
    assert "vera_vjay.licence_observations" in sql and "RETURNING" not in sql.upper() and "UPDATE" not in sql.upper()


def test_migration_is_append_only():
    sql = MIG.read_text()
    assert "licence_observations_append_only" in sql and "BEFORE UPDATE OR DELETE" in sql and "BEFORE TRUNCATE" in sql
    grants = re.findall(r"GRANT\s+(.*?)\s+ON", sql, re.S)
    assert grants and all(not re.search(r"UPDATE|DELETE|TRUNCATE|ALL", g) for g in grants)
    assert "vera_vjay_wo" in sql and "vera_vjay_rw" in sql and "vera_vjay_ro" in sql and "'vera-admission'" in sql
    assert "DESTRUCTIVE" in ROLLBACK.read_text() and not list((ROOT / "db" / "migrations").glob("006_vera_*rollback*"))


def test_pending_migration_not_matched_by_runner_glob():
    mig = ROOT / "db" / "migrations"
    assert not list(mig.glob("006_vera_*.sql")) and (mig / "pending" / "006_vera_licence_observations.sql").exists()
