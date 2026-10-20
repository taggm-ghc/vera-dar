"""AE-1020/DBC-1: DB-backed full-document corpus. SYNTHETIC bundle -> row plan -> in-memory fake DB -> reader.
No DB, no network, no env files: the connection is a fake DB-API object."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from tests import fulldoc_bundle_synth as synth
from vera import dar_store, fulldoc_corpus as fc, fulldoc_db, fulltext_loader

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "ingest_fulldoc_chunks.py"
Q = "transmission incidence outbreak infection"
QS = [Q, "temperature forcing ensemble precipitation", "reproducibility software provenance workflow"]
RUN = "7"


def _script():
    spec = importlib.util.spec_from_file_location("ingest_fulldoc_chunks", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture()
def world(tmp_path):
    fc.reset_for_tests()
    root = synth.write_bundle(tmp_path / "b")
    cfg = synth.synth_cfg()
    rows, report = _script().build_plan(root, cfg, target=700)  # small target: many partitions per document
    return root, cfg, rows, report


def _db(rows, run="7"):
    """(sources, spans) as the two SELECTs would return them."""
    sources, spans = [], []
    for i, r in enumerate(rows, start=1):
        s = r["source"]
        sources.append({"source_id": i, "canonical_url": s["canonical_url"], "content": s["content"],
                        "content_hash": s["content_hash"], "provenance": json.loads(json.dumps(s["provenance"]))})
        spans += [{"source_id": i, "start_index": x["start_index"], "end_index": x["end_index"], "text": x["text"]}
                  for x in r["spans"]]
    return sources, spans


class FakeConn:
    def __init__(self, sources, spans, log):
        self.sources, self.spans, self.log = sources, spans, log

    def cursor(self):
        return self

    def execute(self, sql, params):
        self.log.append((sql, dict(params)))
        self._sql, self._p = sql, params

    def fetchall(self):
        if self._sql == fulldoc_db.SQL_SOURCES:
            assert self._p["kind"] == dar_store.FULLDOC_CHUNKS_KIND
            return [(s["source_id"], s["canonical_url"], s["content"], s["content_hash"], s["provenance"]) for s in self.sources]
        if self._sql == fulldoc_db.SQL_SPANS:
            ids = set(self._p["ids"])
            return [(x["source_id"], x["start_index"], x["end_index"], x["text"]) for x in self.spans if x["source_id"] in ids]
        raise AssertionError("unexpected SQL")

    def close(self):
        pass


def _factory(sources, spans, log=None):
    log = log if log is not None else []
    return lambda: FakeConn(sources, spans, log)


def _chunks(dar):
    return [(c["source_key"], c["start"], c["end"], c["text"], dict(c["tf"]), c["len"]) for c in dar._chunks]


def test_partition_is_contiguous_and_nonoverlapping(world):
    _, _, rows, report = world
    for r in rows:
        prev = 0
        for sp in r["spans"]:
            assert sp["start_index"] == prev and sp["end_index"] > prev
            prev = sp["end_index"]
        assert prev == r["source"]["provenance"]["chars"]
        assert max(len(sp["text"]) for sp in r["spans"]) <= 900
        assert r["source"]["content"] == "" and r["source"]["content_hash"] == fulldoc_db.EMPTY_SHA256
        assert "domain_hint" not in r["source"]["provenance"]["item"]
    assert report["n_spans"] > report["n_documents"] and "text" not in json.dumps(report["documents"][0]["spans"][0])


def test_reader_corpus_equals_bundle_corpus(world):
    root, cfg, rows, _ = world
    bundle = fc.load_bundle(root, cfg)
    c = fulldoc_db.load_db(_factory(*_db(rows)), RUN, cfg)
    assert c.status == bundle.status == "ok", c.unusable
    assert not c.unusable, c.unusable
    assert c.acquired == bundle.acquired and c.unusable == bundle.unusable == []
    assert c._key_to_doc == bundle._key_to_doc and c._texts == bundle._texts
    assert _chunks(c._dar) == _chunks(bundle._dar)
    assert _chunks(c._evidence_dar()) == _chunks(bundle._evidence_dar())
    assert c.inspection() == bundle.inspection()
    for q in QS:
        for ev in (False, True):
            assert c.retrieve(q, 12, evidence_only=ev) == bundle.retrieve(q, 12, evidence_only=ev)
    assert any(c.retrieve(Q, 12, evidence_only=False))


def test_reader_issues_only_the_two_selects(world):
    _, cfg, rows, _ = world
    log = []
    fulldoc_db.load_db(_factory(*_db(rows), log), RUN, cfg)
    assert [s for s, _ in log] == [fulldoc_db.SQL_SOURCES, fulldoc_db.SQL_SPANS]
    assert all(s.lstrip().upper().startswith("SELECT") for s, _ in log)


def test_recorded_pins_checked_against_config(world):
    root, cfg, rows, report = world
    cfg2 = copy.deepcopy(cfg)
    cfg2["pins"] = dict(report["pins"])
    assert fulldoc_db.load_db(_factory(*_db(rows)), RUN, cfg2).status == "ok"
    cfg2["pins"]["sums_sha256"] = "0" * 64
    bad = fulldoc_db.load_db(_factory(*_db(rows)), RUN, cfg2)
    assert bad.status == "corpus_unusable" and {u["code"] for u in bad.unusable} == {fc.R_PIN}


@pytest.mark.parametrize("how", ["char", "missing", "gap", "hash", "kind", "content", "pins", "licence", "chunking",
                                 "canonical", "extra_prov"])
def test_tampered_rows_are_refused(world, how):
    _, cfg, rows, _ = world
    sources, spans = _db(rows)
    first = [x for x in spans if x["source_id"] == 1]
    if how == "char":
        first[1]["text"] = "X" + first[1]["text"][1:]
    elif how == "missing":
        spans.remove(first[1])
    elif how == "gap":
        first[1]["start_index"] += 1
    elif how == "hash":
        sources[0]["provenance"]["text_sha256"] = "0" * 64
    elif how == "kind":
        sources[0]["provenance"]["stored_text_kind"] = "full_text"
    elif how == "content":
        sources[0]["content"] = "x"
    elif how == "pins":
        sources[0]["provenance"]["bundle"]["sums_sha256"] = "1" * 64
    elif how == "licence":
        it = sources[0]["provenance"]["item"]
        it["licence"], it["licence_url"] = "CC BY-ND 4.0", "https://creativecommons.org/licenses/by-nd/4.0/"
    elif how == "chunking":
        sources[0]["provenance"]["chunking"]["chunk_chars"] += 1
    elif how == "canonical":
        sources[0]["canonical_url"] = "https://doi.org/10.9999/other"
    elif how == "extra_prov":
        sources[0]["provenance"]["title"] = "Changed title"
    c = fulldoc_db.load_from_rows(sources, spans, cfg)
    if how == "pins":
        assert c.status == "corpus_unusable" and not c.acquired
    else:
        assert "syn_epi" in {u["document_id"] for u in c.unusable} and "syn_epi" not in c._texts
        assert {a["document_id"] for a in c.acquired} == {"syn_clim", "syn_repro"} and c.status == "ok"
    for u in c.unusable:
        assert "text" not in u and u["code"].startswith("R-")


def test_licence_refusal_code_matches_bundle_path(world, tmp_path):
    root, cfg, rows, _ = world
    nd = {"licence": "CC BY-ND 4.0", "licence_url": "https://creativecommons.org/licenses/by-nd/4.0/"}
    bundle = fc.load_bundle(synth.write_bundle(tmp_path / "nd", items_over={"syn_epi": nd}), cfg)
    sources, spans = _db(rows)
    sources[0]["provenance"]["item"].update(nd)
    c = fulldoc_db.load_from_rows(sources, spans, cfg)
    assert {u["document_id"]: u["code"] for u in c.unusable} == {u["document_id"]: u["code"] for u in bundle.unusable}


def test_empty_bad_run_and_connection_errors(world):
    _, cfg, rows, _ = world
    assert {u["code"] for u in fulldoc_db.load_db(_factory([], []), RUN, cfg).unusable} == {fulldoc_db.R_DBEMPTY}
    assert fulldoc_db.load_db(_factory(*_db(rows)), "7; DROP", cfg).unusable[0]["code"] == fulldoc_db.R_DBRUN

    def boom():
        raise OSError("postgresql://secret-host/db")
    c = fulldoc_db.load_db(boom, RUN, cfg)
    assert c.unusable[0]["code"] == fulldoc_db.R_DBCONN and "secret-host" not in json.dumps(c.unusable)


def test_kind_constant_reused_and_legacy_paths_exclude_it(world):
    """The rebuilt text loads through the reader although the row marker kind is listed as non-document for the legacy
    readers (discovery_contract), which therefore exclude it."""
    from vera import answer_contract, corpus_description, memory_store
    _, cfg, rows, _ = world
    assert fulldoc_db.FULLDOC_CHUNKS_KIND == dar_store.FULLDOC_CHUNKS_KIND == memory_store.FULLDOC_CHUNKS_KIND \
        == corpus_description.FULLDOC_CHUNKS_KIND
    assert rows[0]["source"]["provenance"]["stored_text_kind"] == dar_store.FULLDOC_CHUNKS_KIND
    assert fulldoc_db.load_db(_factory(*_db(rows)), RUN, cfg).status == "ok"
    assert answer_contract.acquisition_state({"stored_text_kind": dar_store.FULLDOC_CHUNKS_KIND, "admission_passed": True}) \
        != answer_contract.ACQ_ACQUIRED
    assert dar_store.FULLDOC_CHUNKS_KIND in fulltext_loader._non_document_kinds()


# ----------------------------------------------------------------------------------------------- seam + readiness
def _patch_connect(monkeypatch, rows, log=None):
    srcs = _db(rows)
    monkeypatch.setattr(fulldoc_db, "default_connect", lambda cfg, env: _factory(*srcs, log))


def test_seam_db_source_and_bundle_source(world, monkeypatch):
    root, cfg, rows, _ = world
    _patch_connect(monkeypatch, rows)
    env_db = {"VERA_FULLDOC_DB_RUN_ID": RUN}
    c = fc.get_corpus(cfg, env_db)
    assert c.status == "ok" and len(c.acquired) == 3
    assert fc.get_corpus(cfg, env_db) is c  # cached
    b = fc.get_corpus(cfg, {cfg["bundle_dir_env"]: str(root)})
    assert b.acquired == c.acquired
    assert fc.get_corpus(cfg, {}) is None
    off = copy.deepcopy(cfg)
    off["fulldoc"]["enabled"] = False
    assert fc.get_corpus(off, env_db) is None


def test_both_sources_refused_with_code(world, monkeypatch):
    root, cfg, rows, _ = world
    _patch_connect(monkeypatch, rows, log := [])
    env = {cfg["bundle_dir_env"]: str(root), "VERA_FULLDOC_DB_RUN_ID": RUN}
    c = fc.get_corpus(cfg, env)
    assert c.status == "corpus_unusable" and not c.acquired and c.retrieve(Q) == []
    assert {u["code"] for u in c.unusable} == {fc.R_SOURCE} and log == []  # neither source was read
    r = fc.readiness(cfg, env)
    assert r["status"] == "unusable" and r["codes"] == [fc.R_SOURCE] and r["source"] == "conflict"


def test_db_without_ro_connection_is_unusable(world, monkeypatch):
    _, cfg, _, _ = world
    monkeypatch.setattr(fulldoc_db, "default_connect", lambda cfg, env: None)
    c = fc.get_corpus(cfg, {"VERA_FULLDOC_DB_RUN_ID": RUN})
    assert c.status == "corpus_unusable" and c.unusable[0]["code"] == fulldoc_db.R_DBCONN


def test_readiness_db_shape_codes_and_counts_only(world, monkeypatch):
    _, cfg, rows, report = world
    _patch_connect(monkeypatch, rows)
    r = fc.readiness(cfg, {"VERA_FULLDOC_DB_RUN_ID": RUN})
    assert r["source"] == "db" and r["status"] == "ok" and r["n_acquired"] == 3 and r["n_unusable"] == 0
    assert r["db_configured"] is True and r["bundle_configured"] is False and r["pins_ok"] is True and r["codes"] == []
    blob = json.dumps(r)
    assert RUN not in r.values() and "Synthetic" not in blob and "SYNTHETIC" not in blob and "http" not in blob
    assert set(r) == {"enabled", "bundle_configured", "db_configured", "status", "n_acquired", "n_unusable", "pins_ok",
                      "source", "codes"}
    assert fc.readiness(cfg, {})["status"] == "not_configured"
    off = copy.deepcopy(cfg)
    off["fulldoc"]["enabled"] = False
    assert fc.readiness(off, {"VERA_FULLDOC_DB_RUN_ID": RUN})["status"] == "disabled"
    bad = copy.deepcopy(cfg)
    bad["pins"] = {"manifest_sha256": "0" * 64, "sums_sha256": report["pins"]["sums_sha256"]}
    fc.reset_for_tests()
    r2 = fc.readiness(bad, {"VERA_FULLDOC_DB_RUN_ID": RUN})
    assert r2["status"] == "unusable" and r2["pins_ok"] is False and r2["codes"] == [fc.R_PIN]


def test_shipped_flag_still_false():
    assert json.loads(Path(fc.CONFIG_PATH).read_text(encoding="utf-8"))["fulldoc"]["enabled"] is False


# ----------------------------------------------------------------------------------------------- ingest script
def test_script_dry_run_prints_counts_only_and_writes_plan(world, tmp_path, capsys):
    root, _, _, _ = world
    s = _script()
    plan = tmp_path / "plan"
    assert s.main(["--bundle", str(root), "--plan-dir", str(plan)], env={}, engine_factory=lambda *a: 1 / 0, cfg=synth.synth_cfg()) == 0
    out = capsys.readouterr().out
    assert "documents=3" in out and "mode=dry-run" in out and "SYNTHETIC" not in out
    data = json.loads((plan / "ingest-plan.json").read_text())
    assert data["n_documents"] == 3 and "SYNTHETIC" not in json.dumps(data)


@pytest.mark.parametrize("extra,env", [
    ([], {}), (["--i-have-owner-authorization", "tok", "--new-run"], {}),
    (["--new-run"], {"VERA_DB_URL_WO": "x", "VERA_DB_URL_RO": "y"}),
    (["--i-have-owner-authorization", "tok"], {"VERA_DB_URL_WO": "x", "VERA_DB_URL_RO": "y"})])
def test_script_write_refused_without_authorization_and_env(world, tmp_path, capsys, extra, env):
    root, _, _, _ = world
    calls = []
    rc = _script().main(["--bundle", str(root), "--plan-dir", str(tmp_path / "p"), "--write", *extra], env=env,
                        engine_factory=lambda *a: calls.append(a), cfg=synth.synth_cfg())
    assert rc == 4 and calls == [] and "REFUSED" in capsys.readouterr().out
    assert not (tmp_path / "p").exists()


def test_script_refuses_partial_bundle(tmp_path, capsys):
    nd = {"licence": "CC BY-ND 4.0", "licence_url": "https://creativecommons.org/licenses/by-nd/4.0/"}
    root = synth.write_bundle(tmp_path / "b", items_over={"syn_epi": nd})
    assert _script().main(["--bundle", str(root), "--plan-dir", str(tmp_path / "p")], env={}, cfg=synth.synth_cfg()) == 3
    assert not (tmp_path / "p").exists()


def test_script_static_scan_no_mutation_beyond_inserts():
    src = SCRIPT.read_text(encoding="utf-8").upper()
    for bad in ("DELETE FROM", "TRUNCATE", "DROP ", "ALTER ", "CREATE TABLE", "UPDATE {S}", "GRANT "):
        assert bad not in src


# ----------------------------------------------------------------------------------------------- INT-V3 revisions
def test_partial_run_is_refused(world):
    _, cfg, rows, _ = world
    sources, spans = _db(rows[:2])  # 2 of the 3 recorded documents
    c = fulldoc_db.load_from_rows(sources, spans, cfg)
    assert c.status == "corpus_unusable" and not c.acquired
    assert {u["code"] for u in c.unusable} == {fulldoc_db.R_DBPARTIAL}
    s2, p2 = _db(rows)
    s2[0]["provenance"]["n_documents"] = 9
    assert fulldoc_db.load_from_rows(s2, p2, cfg).unusable[0]["code"] == fulldoc_db.R_DBPARTIAL


def test_rebuilt_text_size_limit_applies(world):
    _, cfg, rows, _ = world
    icfg = copy.deepcopy(fulltext_loader.load_config())
    icfg["max_text_bytes"] = 1000
    c = fulldoc_db.load_from_rows(*_db(rows), cfg, icfg)
    assert not c.acquired and {u["code"] for u in c.unusable} == {icfg["reason_codes"]["too_large"]}


def test_script_rejects_abbreviated_write_flag(world, tmp_path):
    root, _, _, _ = world
    for flag in ("--writ", "--wri"):
        with pytest.raises(SystemExit) as e:
            _script().main(["--bundle", str(root), "--plan-dir", str(tmp_path / "p"), flag], env={}, cfg=synth.synth_cfg())
        assert e.value.code == 2
    assert not (tmp_path / "p").exists()


def _write_args(root, tmp_path, extra=()):
    return ["--bundle", str(root), "--plan-dir", str(tmp_path / "p"), "--write", "--new-run",
            "--i-have-owner-authorization", "tok", *extra]


ENVS = {"VERA_DB_URL_WO": "x", "VERA_DB_URL_RO": "y"}


def test_write_mode_always_runs_the_guard(world, tmp_path):
    root, _, _, report = world
    s = _script()
    seen = []
    calls = []

    def stop():
        seen.append(1)
        raise SystemExit(3)
    with pytest.raises(SystemExit):
        s.main(_write_args(root, tmp_path, ["--expect-spans", "5"]), env=ENVS, cfg=synth.synth_cfg(),
               engine_factory=lambda *a: calls.append(a), guard=stop)
    assert seen == [1] and calls == []
    with pytest.raises(SystemExit) as e:  # the default guard refuses under a test runner
        s.main(_write_args(root, tmp_path, ["--expect-spans", "5"]), env=ENVS, cfg=synth.synth_cfg(),
               engine_factory=lambda *a: calls.append(a))
    assert e.value.code == 3 and calls == []
    ran = []  # dry-run never runs it
    assert s.main(["--bundle", str(root), "--plan-dir", str(tmp_path / "q")], env={}, cfg=synth.synth_cfg(),
                  guard=lambda: ran.append(1)) == 0 and ran == []


def test_row_caps_enforced_before_any_engine(world, tmp_path, capsys):
    root, _, rows, report = world
    s = _script()
    calls = []
    n = report["n_spans"]
    rc = s.main(_write_args(root, tmp_path, ["--expect-spans", "4"]), env=ENVS, cfg=synth.synth_cfg(),
                engine_factory=lambda *a: calls.append(a), guard=lambda: None)
    assert rc == 4 and calls == [] and "exceed the cap" in capsys.readouterr().out
    assert s.main(_write_args(root, tmp_path), env=ENVS, cfg=synth.synth_cfg(), guard=lambda: None) == 4  # no --expect-spans
    s.check_caps(rows, n)
    s.check_caps(rows, int(n / 1.05) + 1)
    s.check_caps(rows * 3, n * 3)  # 9 sources: at the cap
    with pytest.raises(fulltext_loader.IngestError):
        s.check_caps(rows * 4, n * 4)  # 12 sources
