"""LICENCE-ALLOWLIST-1104: an unknown licence held on the full-text ingest path is recorded in licence_observations
(reason_code licence_unknown) before the hold. Fake DB engine only; nothing connects anywhere."""
import pytest

from tests import acq_support as A
from vera import auto_acquire, fulltext_loader as fl
from vera.corpus_admission import LICENCE_OBS_INSERT

CFG = {**fl.load_config(), "licence_hold_observations_enabled": True}
ITEM = {"fulltext_url": "https://example.org/x.xml", "landing_url": "https://example.org/x", "doi": "10.9999/x"}
META = {"meta": {"licence": {"id": None, "url": None}}}


class FakeConn:
    def __init__(self, log, fail):
        self.log, self.fail = log, fail

    def execute(self, sql, params=None):
        if self.fail:
            raise RuntimeError("db down")
        self.log.append((str(sql), dict(params or {})))


class FakeEngine:
    def __init__(self, fail=False):
        self.log, self.fail = [], fail

    def begin(self):
        eng = self

        class _Ctx:
            def __enter__(self):
                return FakeConn(eng.log, eng.fail)

            def __exit__(self, *a):
                return False
        return _Ctx()


def _refused(licence, url=None, cfg=CFG):
    with pytest.raises(fl.Refused) as ei:
        fl.check_licence({**ITEM, "licence": licence, "licence_url": url}, META, cfg)
    return ei.value


def test_unknown_licence_hold_carries_the_candidate():
    e = _refused("Mystery Publisher Terms v3")
    assert e.code == CFG["reason_codes"]["licence_hold"] and e.licence_obs["licence_reason_code"] == "licence_unknown"


@pytest.mark.parametrize("lid", ["All rights reserved", "", "cc by-nd"])
def test_known_holds_and_rejects_carry_nothing(lid):
    assert _refused(lid).licence_obs is None


def test_record_writes_two_append_only_rows_before_the_hold():
    eng = FakeEngine()
    e = _refused("Mystery Publisher Terms v3")
    assert fl.record_licence_hold(eng, e, CFG, run_id=7, now="2026-11-04T00:00:00+00:00") == 2
    assert all(sql.strip().startswith("INSERT INTO vera_vjay.licence_observations") for sql, _ in eng.log)
    assert all("UPDATE" not in sql and "DELETE" not in sql for sql, _ in eng.log)
    gov = [p for _, p in eng.log if p["governing"]][0]
    assert gov["spdx"] == "Mystery Publisher Terms v3" and gov["reason_code"] == "licence_unknown"
    assert gov["decision"] == "hold" and gov["url"] is None and gov["doc_ref"] == "10.9999/x"
    assert set(gov) >= {k.strip(" :") for k in LICENCE_OBS_INSERT.split("VALUES")[1].strip(" ()").split(",")
                        if k.strip().startswith(":")}


def test_record_is_off_without_the_flag_and_never_raises():
    e = _refused("Mystery Publisher Terms v3")
    assert fl.record_licence_hold(FakeEngine(), e, {**CFG, "licence_hold_observations_enabled": False}) == 0
    assert fl.record_licence_hold(None, e, CFG) == 0
    assert fl.record_licence_hold(FakeEngine(fail=True), e, CFG) == 0
    assert fl.record_licence_hold(FakeEngine(), fl.Refused("X", "other"), CFG) == 0


def test_wostore_records_through_its_write_engine():
    eng = FakeEngine()
    st = auto_acquire.WoStore(eng, FakeEngine(), {"licence_hold_observations_enabled": True})
    assert auto_acquire.record_licence_hold(st, _refused("Mystery Publisher Terms v3")) == 2 and len(eng.log) == 2
    assert auto_acquire.record_licence_hold(A.Store(), _refused("Mystery Publisher Terms v3")) == 0  # no engine: no-op


def test_run_job_records_an_unknown_licence_hold():
    class RecStore(A.Store):
        def __init__(self):
            super().__init__()
            self.holds = []

        def record_licence_hold(self, exc):
            self.holds.append(exc.licence_obs)
            return 2

    web, _ = A.epi_setup(rec_kw={"licence": "Mystery Publisher Terms v3"})
    store = RecStore()
    d = A.deps([A.cand("10.9999/acq.epi4")], web, store=store)
    out = auto_acquire.run_job(A.job(A.Q_EPI), A.cfg(), d, clock=A.clock_at(), env=A.ENV_ON)
    assert out["stored"] == 0 and store.plans == []
    assert [h["licence"]["content"]["id"] for h in store.holds] == ["Mystery Publisher Terms v3"], out


def test_shipped_configs_enable_hold_recording():
    import json
    from pathlib import Path
    root = Path(fl.__file__).resolve().parent.parent
    assert json.loads((root / "config" / "auto_acquire.json").read_text())["licence_hold_observations_enabled"] is True
    assert fl.load_config()["licence_hold_observations_enabled"] is True
