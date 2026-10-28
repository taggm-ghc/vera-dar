"""FIXR-7DA-1028 (VER-REL7DA-1-1028 finding 1): rows stored by the PRE-PROVFIX code still serve after PROVFIX-1028.

The stored rows are written by the frozen 7da574a vera/fulltext_loader.py (tests/fixtures/legacy_7da574a/, the exact
`git show 7da574a:vera/fulltext_loader.py` bytes, sha256 below) and by an acquisition item without the PROVFIX item
labels, as the production rows were; they are then read by the CURRENT code. Tampered licence id/url still fails
R_DBROW. SYNTHETIC bundle / stub web, no DB, no network, no env files."""
import copy
import hashlib
import json
import types
from pathlib import Path

import pytest

from tests import acq_support as A
from tests import fulldoc_bundle_synth as synth
from tests import fulltext_synth
from tests.test_fulldoc_db import _db, _script
from vera import auto_acquire, fulldoc_corpus as fc, fulldoc_db, fulltext_loader as fl

LEGACY_SRC = Path(__file__).resolve().parent / "fixtures" / "legacy_7da574a" / "fulltext_loader.py.txt"
LEGACY_SHA256 = "9087d695b6103f9dca95293788b3c6e6284ea4287766c720e1fda96d61ef9a9a"
DOI, PMCID = "10.9999/acq.epi4", "PMC1234567"


def _legacy_module():
    raw = LEGACY_SRC.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == LEGACY_SHA256  # the frozen 7da574a writer, unchanged
    m = types.ModuleType("vera._fulltext_loader_7da574a")
    m.__file__ = fl.__file__  # same ROOT as the live module
    exec(compile(raw.decode("utf-8"), str(LEGACY_SRC), "exec"), m.__dict__)  # noqa: S102  frozen in-repo source
    return m


def _old_prepare(monkeypatch_ctx):
    """Patch vera.fulltext_loader.prepare_from_parts with the 7da574a writer; the 7da574a acquisition item carried no
    licence_source / discovered_via, so those PROVFIX item labels are removed in place (stored item = old item)."""
    old = _legacy_module()

    def prep(item, *a, **k):
        item.pop("licence_source", None)
        item.pop("discovered_via", None)
        return old.prepare_from_parts(item, *a, **k)
    monkeypatch_ctx.setattr(fl, "prepare_from_parts", prep)


# ---------------------------------------------------------------- base corpus path (pinned, "CC BY 4.0" ids)
def _base_rows(monkeypatch, tmp_path, legacy):
    fc.reset_for_tests()
    root = synth.write_bundle(tmp_path / "b")
    cfg = synth.synth_cfg()
    with monkeypatch.context() as m:
        if legacy:
            _old_prepare(m)
        rows, report = _script().build_plan(root, cfg, target=700)
    sources, spans = _db(rows)
    docs = {s["provenance"]["document_id"]: {"text_sha256": s["provenance"]["text_sha256"],
                                             "licence_id": s["provenance"]["item"]["licence"],
                                             "licence_decision": "allow"} for s in sources}
    pinned = copy.deepcopy(cfg)
    pinned["pins"] = {**report["pins"], "documents": docs}
    return cfg, pinned, sources, spans


def test_base_rows_written_by_7da574a_lack_the_labels_new_rows_carry(monkeypatch, tmp_path):
    _, _, old_src, _ = _base_rows(monkeypatch, tmp_path / "o", legacy=True)
    _, _, new_src, _ = _base_rows(monkeypatch, tmp_path / "n", legacy=False)
    for s in old_src:
        assert s["provenance"]["item"]["licence"] == "CC BY 4.0"
        assert "version" not in s["provenance"]["licence"]["content"]  # the old shape the read path must accept
    assert all(s["provenance"]["licence"]["content"].get("version") == "4.0" for s in new_src)


@pytest.mark.parametrize("pinned_run", [False, True])
def test_base_rows_written_by_7da574a_still_serve(monkeypatch, tmp_path, pinned_run):
    cfg, pinned, sources, spans = _base_rows(monkeypatch, tmp_path, legacy=True)
    c = fulldoc_db.load_from_rows(sources, spans, pinned if pinned_run else cfg)
    assert c.status == "ok" and not c.unusable and len(c.acquired) == 3
    assert {a["licence"] for a in c.acquired} == {"CC BY 4.0"} and {a["licence_decision"] for a in c.acquired} == {"allow"}


@pytest.mark.parametrize("legacy", [True, False])
@pytest.mark.parametrize("field,value", [("id", "CC BY-NC-ND 4.0"),
                                         ("url", "https://creativecommons.org/licenses/by-nc-nd/4.0/")])
def test_tampered_stored_licence_id_or_url_still_fails_dbrow(monkeypatch, tmp_path, legacy, field, value):
    cfg, pinned, sources, spans = _base_rows(monkeypatch, tmp_path, legacy=legacy)
    sources[0]["provenance"]["licence"]["content"][field] = value
    c = fulldoc_db.load_from_rows(copy.deepcopy(sources), copy.deepcopy(spans), cfg)
    assert [u["code"] for u in c.unusable] == [fulldoc_db.R_DBROW] and len(c.acquired) == 2
    bad = fulldoc_db.load_from_rows(sources, spans, pinned)  # pinned base run: one failure -> whole run unusable
    assert bad.status == "corpus_unusable" and not bad.acquired


@pytest.mark.parametrize("legacy", [True, False])
def test_tampered_stored_licence_decision_or_raw_hash_still_fails_dbrow(monkeypatch, tmp_path, legacy):
    cfg, _, sources, spans = _base_rows(monkeypatch, tmp_path, legacy=legacy)
    for key, value in (("licence_decision", "hold"), ("raw_sha256", "0" * 64)):
        s = copy.deepcopy(sources)
        s[0]["provenance"][key] = value
        c = fulldoc_db.load_from_rows(s, copy.deepcopy(spans), cfg)
        assert [u["code"] for u in c.unusable] == [fulldoc_db.R_DBROW], key


def test_half_labelled_row_is_not_accepted(monkeypatch, tmp_path):
    """All-or-nothing: a row holding a PROVFIX label with a wrong value matches neither shape."""
    cfg, _, sources, spans = _base_rows(monkeypatch, tmp_path, legacy=True)
    sources[0]["provenance"]["licence"]["content"]["version"] = "3.0"
    c = fulldoc_db.load_from_rows(sources, spans, cfg)
    assert [u["code"] for u in c.unusable] == [fulldoc_db.R_DBROW]


# ---------------------------------------------------------------- acquired path (jats_licences, no converter licence)
def _jats_only_web():
    raw = fulltext_synth.jats("acq_epi", "epi", DOI).replace(
        b'<license xlink:href="https://creativecommons.org/licenses/by/4.0/"><license-p>SYNTHETIC licence line.',
        b'<license><license-p>This is an open access article distributed under the terms of the Creative Commons '
        b'Attribution License (CC BY 4.0).')
    assert b"xlink:href=\"https://creativecommons.org" not in raw
    return A.Web({DOI: A.epmc_record(DOI, pmcid=PMCID)}, {PMCID: raw}, None)


def _acquired_plans(monkeypatch, legacy):
    auto_acquire.reset_for_tests()
    store = A.Store()
    with monkeypatch.context() as m:
        if legacy:
            _old_prepare(m)
        auto_acquire.run_job(A.job(), A.cfg(), A.deps([A.cand(DOI)], _jats_only_web(), store=store),
                             clock=A.clock_at(), env=A.ENV_ON)
    assert len(store.plans) == 1
    return store.plans


def test_acquired_row_written_by_7da574a_with_jats_licences_only_still_serves(monkeypatch):
    plans = _acquired_plans(monkeypatch, legacy=True)
    prov = plans[0]["source"]["provenance"]
    assert prov["meta"]["jats_licences"] and not (prov["meta"]["meta"].get("licence") or {}).get("url")
    assert "embedded" not in prov["licence"] and "discovered_via" not in prov  # the old stored shape
    assert prov["licence"]["content"]["source"] == "manifest declared"
    src, spans = A.rows_from_plans(plans)
    c = fulldoc_db.load_acquired_rows(src, spans, fc.load_config(), A.cfg())
    assert not c.unusable and [a["doi"] for a in c.acquired] == [DOI]


def test_acquired_row_written_by_new_code_serves_and_carries_the_labels(monkeypatch):
    plans = _acquired_plans(monkeypatch, legacy=False)
    prov = plans[0]["source"]["provenance"]
    assert prov["licence"]["embedded"]["statement"].startswith("This is an open access") and prov["discovered_via"]
    src, spans = A.rows_from_plans(plans)
    c = fulldoc_db.load_acquired_rows(src, spans, fc.load_config(), A.cfg())
    assert not c.unusable and [a["doi"] for a in c.acquired] == [DOI]


@pytest.mark.parametrize("field,value", [("id", "cc by-nd"), ("url", "https://creativecommons.org/licenses/by-nd/4.0/")])
def test_acquired_old_row_with_tampered_licence_fails_dbrow(monkeypatch, field, value):
    plans = _acquired_plans(monkeypatch, legacy=True)
    plans[0]["source"]["provenance"]["licence"]["content"][field] = value
    src, spans = A.rows_from_plans(plans)
    c = fulldoc_db.load_acquired_rows(src, spans, fc.load_config(), A.cfg())
    assert not c.acquired and [u["code"] for u in c.unusable] == [fulldoc_db.R_DBROW]


def test_legacy_label_view_keeps_decision_fields():
    meta = {"meta": {"licence": {"id": None, "url": None}}, "jats_licences": [{"text": "t"}]}
    prov = {"licence": {"content": {"id": "CC BY 4.0", "url": "u", "source": "s", "version": "4.0"},
                        "embedded": {"statement": "t"}}, "licence_decision": "allow", "raw_sha256": "h",
            "discovered_via": "europepmc"}
    v = fl.legacy_label_view(prov, meta)
    assert v == {"licence": {"content": {"id": "CC BY 4.0", "url": "u", "source": "s"}}, "licence_decision": "allow",
                 "raw_sha256": "h"}
    assert prov["licence"]["content"]["version"] == "4.0" and "embedded" in prov["licence"]  # input not mutated
    conv = {"meta": {"licence": {"id": "x", "url": "https://creativecommons.org/licenses/by/4.0/"}}}
    assert fl.legacy_label_view(prov, conv)["licence"]["embedded"] == {"statement": "t"}  # converter embedded kept
    assert json.dumps(v)  # plain data
