"""WP-D-1113 (security review F1): the chunk_domain_write kill switch must be settable through the real config loader."""
from __future__ import annotations

import json

import pytest

from vera import chunk_taxo as core


def _load(tmp_path, job):
    p = tmp_path / "chunk_taxo.json"
    p.write_text(json.dumps({"enabled": False, "job": job}), encoding="utf-8")
    return core.load_cfg(p)


@pytest.mark.parametrize("val", [True, False])
def test_load_cfg_accepts_boolean_chunk_domain_write(tmp_path, val):
    assert _load(tmp_path, {"chunk_domain_write": val})["job"] == {"chunk_domain_write": val}


@pytest.mark.parametrize("val", ["false", 1, 0, None, "true", [], {}])
def test_load_cfg_rejects_non_boolean_chunk_domain_write(tmp_path, val):
    with pytest.raises(ValueError, match="chunk_domain_write must be a bool"):
        _load(tmp_path, {"chunk_domain_write": val})


def test_load_cfg_still_rejects_unknown_job_keys_and_absent_means_unset(tmp_path):
    with pytest.raises(ValueError, match="unknown key job.chunk_domain_writes"):
        _load(tmp_path, {"chunk_domain_writes": False})
    assert _load(tmp_path, {})["job"] == {}   # absent: the job treats it as on (only a literal false disables)


def test_job_reads_the_validated_key(tmp_path):
    from tests.test_chunk_taxo_job_1109 import FakeDB, make, spans
    db = FakeDB(spans())
    db.domain_table = True
    j = make(db, **_load(tmp_path, {"chunk_domain_write": False})["job"])
    assert j.run()["complete"] and not db.active("chunk_domain") and "probe_chunk_domain" not in db.names()
