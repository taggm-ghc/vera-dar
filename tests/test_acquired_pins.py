"""H1B-PROMOTE-1107: repository-pinned manifest for stored acquired documents (owner H1(b), 2026-11-07).
SYNTHETIC rows only (tests/acq_support.py): no DB, no network, no env files.

Covers: flag off = today's behaviour; pinned served; quarantine when on; coordinated tamper (scenario G), forged licence
and forged addition refused; in-memory inline rows unaffected; malformed manifest fails closed; shadow reports; the
promotion tool (export -> proposal -> manifest -> served) and its TAMPER-SUSPECT refusal; the shipped manifest is
consistent with the CTL-PIN E2 exception; the in-memory tag is set only at its two construction sites."""
import ast
import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from tests import acq_support as A
from vera import acquired_pins as ap, auto_acquire, fulldoc_corpus as fc, fulldoc_db

REPO = Path(__file__).resolve().parent.parent
DOIS = (("10.9999/acq.epi4", "PMC1234567", 1), ("10.9999/acq.epi5", "PMC7654321", 7))


def _store(doi, pmcid, seed):
    from tests import test_fulldoc_e2e as e2e
    web, _ = A.epi_setup(doi=doi, pmcid=pmcid, jats_kw={"seed": seed})
    store = A.Store()
    auto_acquire.run_job(A.job(e2e.Q_EPI), A.cfg(), A.deps([A.cand(doi)], web, store=store),
                         clock=A.clock_at(), env=A.ENV_ON)
    assert store.plans, "setup: no acquired plan was stored"
    return store.plans


@pytest.fixture(scope="module")
def plans():
    p = _store(*DOIS[0]) + _store(*DOIS[1])
    assert len({x["source"]["provenance"]["document_id"] for x in p}) == 2, "setup: two distinct documents"
    assert len({x["source"]["provenance"]["text_sha256"] for x in p}) == 2, "setup: two distinct texts"
    return p


def _rows(plans, n=None):
    src, spans = A.rows_from_plans(plans[:n] if n else plans)
    return copy.deepcopy(src), copy.deepcopy(spans)


def _manifest(tmp_path, monkeypatch, mode, documents=None, raw=None):
    p = tmp_path / "acquired_pins.json"
    batch = [{"id": "B1", "export_sha256": "0" * 64}]
    docs = {did: {**e, "batch": "B1"} for did, e in (documents or {}).items()}
    p.write_text(raw if raw is not None else json.dumps({"schema_version": 1, "mode": mode, "batches": batch,
                                                         "documents": docs}), encoding="utf-8")
    monkeypatch.setattr(ap, "CONFIG_PATH", p)
    ap._CACHE.clear()
    return p


def _load(src, spans):
    return fulldoc_db.load_acquired_rows(src, spans, fc.load_config(), A.cfg())


def _pins_of(src, spans):
    rec = ap.export_records(copy.deepcopy(src), copy.deepcopy(spans), fc.load_config(), A.cfg())
    return {d["document_id"]: {k: d[k] for k in ap.PIN_FIELDS} for d in rec["documents"]}


def _served(c):
    return sorted(a["document_id"] for a in c.acquired)


def _codes(c):
    return sorted((u.get("document_id"), u.get("code")) for u in c.unusable)


def _tamper_text(src, spans, i=0):
    """Scenario G: one span character AND both recorded text hashes rewritten together (self-consistent rows)."""
    sid = src[i]["source_id"]
    first = next(x for x in sorted(spans, key=lambda x: x["start_index"]) if x["source_id"] == sid and re.search("[a-z]", x["text"]))
    j = re.search("[a-z]", first["text"]).start()
    first["text"] = first["text"][:j] + ("z" if first["text"][j] != "z" else "y") + first["text"][j + 1:]
    text = "".join(x["text"] for x in sorted((x for x in spans if x["source_id"] == sid), key=lambda x: x["start_index"]))
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    prov = json.loads(src[i]["provenance"])
    prov["text_sha256"] = h
    prov["meta"]["text_sha256"] = h
    src[i]["provenance"] = json.dumps(prov)


# ------------------------------------------------------------------------------------------------ flag off = today
def test_flag_off_is_todays_behaviour(plans, tmp_path, monkeypatch):
    src, spans = _rows(plans)
    monkeypatch.setattr(ap, "CONFIG_PATH", tmp_path / "absent.json")
    ap._CACHE.clear()
    base = _load(*_rows(plans))
    assert len(base.acquired) == 2 and not hasattr(base, "pin_report")
    # mode off with WRONG pins for both documents, and a tampered row: still served exactly as today
    wrong = {did: {**e, "text_sha256": c * 64} for (did, e), c in zip(sorted(_pins_of(src, spans).items()), "ef")}
    _manifest(tmp_path, monkeypatch, "off", wrong)
    c = _load(*_rows(plans))
    assert _served(c) == _served(base) and _codes(c) == _codes(base) and not hasattr(c, "pin_report")
    _tamper_text(src, spans)
    t = _load(src, spans)
    assert _served(t) == _served(base), ("off: the coordinated tamper is served (the accepted risk through 2026-11-21)", _codes(t))


def test_shipped_manifest_is_valid_and_off():
    m = json.loads((REPO / "config" / "acquired_pins.json").read_text(encoding="utf-8"))
    assert ap.validate(m) is None
    assert m["mode"] in ap.MODES


def test_shipped_mode_matches_controls_register():
    """Mode on <=> CTL-PIN E2 is covered (no exception); mode off/shadow <=> E2 is still a dated exception."""
    m = json.loads((REPO / "config" / "acquired_pins.json").read_text(encoding="utf-8"))
    pin = next(c for c in json.loads((REPO / "config" / "controls.json").read_text(encoding="utf-8"))["controls"]
               if c["id"] == "CTL-PIN")
    e2_exception = any(e.get("path") == "E2" for e in pin.get("exceptions", []))
    if m["mode"] == "on":
        assert "E2" in pin["covered"] and not e2_exception and "_constraint" not in m
        assert m["documents"] and m["batches"], "mode on ships only with a promotion batch (else every stored doc is quarantined)"
    else:
        assert "E2" not in pin["covered"] and e2_exception and m["_constraint"]["kind"] == "temporary"


# ------------------------------------------------------------------------------------------------ mode on
def test_pinned_served_unpinned_quarantined(plans, tmp_path, monkeypatch):
    src, spans = _rows(plans)
    pins = _pins_of(src, spans)
    first = json.loads(src[0]["provenance"])["document_id"]
    second = json.loads(src[1]["provenance"])["document_id"]
    _manifest(tmp_path, monkeypatch, "on", {first: pins[first]})
    c = _load(src, spans)
    assert _served(c) == [first]
    assert (second, ap.R_QUARANTINE) in _codes(c)
    assert c.pin_report["pinned"] == 1 and c.pin_report["unpinned"] == 1
    assert set(c.pin_candidates) == {first, second}  # hash-only candidates for the next promotion batch


def test_scenario_g_coordinated_tamper_refused(plans, tmp_path, monkeypatch):
    src, spans = _rows(plans, 1)
    pins = _pins_of(src, spans)
    did = json.loads(src[0]["provenance"])["document_id"]
    for mode in ("on", "shadow"):
        _manifest(tmp_path, monkeypatch, mode, pins)
        s2, sp2 = copy.deepcopy(src), copy.deepcopy(spans)
        assert _served(_load(copy.deepcopy(s2), copy.deepcopy(sp2))) == [did], "setup: clean pinned row is served"
        _tamper_text(s2, sp2)
        c = _load(s2, sp2)
        assert not c.acquired and (did, ap.R_PINMISMATCH) in _codes(c), mode


def test_forged_provenance_and_licence_pin_refused(plans, tmp_path, monkeypatch):
    """Rewriting any stored provenance field (the licence record included) or the canonical URL changes a pinned value."""
    src, spans = _rows(plans, 1)
    pins = _pins_of(src, spans)
    did = json.loads(src[0]["provenance"])["document_id"]
    _manifest(tmp_path, monkeypatch, "on", pins)
    s2 = copy.deepcopy(src)
    prov = json.loads(s2[0]["provenance"])
    prov["acquisition"]["acquired_at"] = "2001-01-01T00:00:00Z"  # an rw-writable field the loader does not re-derive
    s2[0]["provenance"] = json.dumps(prov)
    c = _load(s2, copy.deepcopy(spans))
    assert not c.acquired and (did, ap.R_PINMISMATCH) in _codes(c), "only the pin catches a rewritten acquisition record"
    forged = {did: {**pins[did], "licence_id": "CC BY-NC-ND 4.0"}}
    _manifest(tmp_path, monkeypatch, "on", forged)
    c = _load(copy.deepcopy(src), copy.deepcopy(spans))
    assert not c.acquired and (did, ap.R_PINMISMATCH) in _codes(c)


def test_forged_addition_quarantined(plans, tmp_path, monkeypatch):
    """An rw holder inserts a complete, self-consistent new document (new id): it has no pin, so it is not served."""
    src, spans = _rows(plans)
    pins = _pins_of(src, spans)
    first = json.loads(src[0]["provenance"])["document_id"]
    _manifest(tmp_path, monkeypatch, "on", {first: pins[first]})
    c = _load(src, spans)
    assert _served(c) == [first]


def test_inline_ephemeral_rows_unaffected(plans, tmp_path, monkeypatch):
    src, spans = _rows(plans, 1)
    src[0][ap.EPHEMERAL_KEY] = True
    _manifest(tmp_path, monkeypatch, "on", {})
    c = _load(src, spans)
    assert len(c.acquired) == 1 and c.pin_report["ephemeral"] == 1 and c.pin_candidates == {}


def test_inline_ephemeral_corpus_strips_tag_from_stored_rows(plans, tmp_path, monkeypatch):
    from vera import acq_config, fulltext_loader, inline_acquire
    src, spans = _rows(plans, 1)
    src[0][ap.EPHEMERAL_KEY] = True  # a stored row must never keep the in-memory tag
    _manifest(tmp_path, monkeypatch, "on", {})
    acq = A.cfg()
    c = inline_acquire.ephemeral_corpus([], acq, fc.load_config(), acq_config.ingest_cfg(fulltext_loader.load_config(), acq),
                                        stored=(src, spans))
    assert c is None, "a stored unpinned row reused inline is quarantined under mode on"


def test_stored_row_sharing_an_inline_run_id_is_not_ephemeral(plans, tmp_path, monkeypatch):
    """Review M2: a stored row with the SAME run_id and document_id as an in-memory row stays subject to the pins."""
    src, spans = _rows(plans, 1)
    src[0]["run_id"] = -1
    eph = {**copy.deepcopy(src[0]), "source_id": -1, ap.EPHEMERAL_KEY: True}
    spans += [{**sp, "source_id": -1} for sp in spans if sp["source_id"] == src[0]["source_id"]]
    did = json.loads(src[0]["provenance"])["document_id"]
    _manifest(tmp_path, monkeypatch, "on", {})
    c = _load(src + [eph], spans)
    assert (did, ap.R_QUARANTINE) in _codes(c), "the stored copy must meet the quarantine"
    assert c.pin_report["unpinned"] == 1 and c.pin_report["ephemeral"] == 1
    assert _served(c) == [did]  # served once, from the in-memory copy


def test_promotion_refuses_truncated_or_non_ro_export(plans):
    src, spans = _rows(plans, 1)
    tool = _tool()
    empty = {"schema_version": 1, "mode": "off", "batches": [], "documents": {}}
    for bad in ({"account": "vera_rw"}, {"rows_read": 5000}, {"row_limit": None}):
        with pytest.raises(tool.Refused):
            tool.propose(empty, {**_export(src, spans), **bad}, "2" * 64, "B1", "2026-11-08")


def test_malformed_manifest_fails_closed(plans, tmp_path, monkeypatch):
    src, spans = _rows(plans, 1)
    _manifest(tmp_path, monkeypatch, "on", raw="{not json")
    c = _load(src, spans)
    assert not c.acquired and any(code == ap.R_PINCFG for _, code in _codes(c))
    _manifest(tmp_path, monkeypatch, "off", raw=json.dumps({"schema_version": 1, "mode": "off", "batches": [],
                                                          "documents": {"acq-0000000000000000": {"licence_decision": "deny"}}}))
    assert ap.load()["mode"] == "on" and ap.load()["error"], "an invalid manifest is read as on with no documents"


def test_shadow_serves_and_reports(plans, tmp_path, monkeypatch):
    src, spans = _rows(plans)
    _manifest(tmp_path, monkeypatch, "shadow", {})
    c = _load(src, spans)
    assert len(c.acquired) == 2 and c.pin_report["unpinned"] == 2 and c.pin_report["mismatch"] == 0


def test_fetch_side_preselection_keeps_only_pinned(monkeypatch):
    rows = [(5, 2, {"document_id": "acq-aaaaaaaaaaaaaaaa"}), (4, 3, {"document_id": "acq-bbbbbbbbbbbbbbbb"})]
    monkeypatch.setattr(fulldoc_db, "_select", lambda cur, sql, params: rows if "LIMIT" in sql and "scan" in str(params) else [])
    out = fulldoc_db.acquired_ids_per_topic(None, None, 1, "m", None, 10, 10, {}, allow_ids=frozenset({"acq-bbbbbbbbbbbbbbbb"}))
    assert out == [4]
    assert sorted(fulldoc_db.acquired_ids_per_topic(None, None, 1, "m", None, 10, 10, {})) == [4, 5]


# ------------------------------------------------------------------------------------------------ promotion tool
def _tool():
    import importlib.util
    spec = importlib.util.spec_from_file_location("promote_probe", REPO / "scripts" / "promote_acquired_pins.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _export(src, spans):
    rec = ap.export_records(copy.deepcopy(src), copy.deepcopy(spans), fc.load_config(), A.cfg())
    return {"kind": "vera-acquired-pin-export", "schema_version": 1, "repo_head": "a" * 40, "account": "vera_eval_ro",
            "rows_read": len(src), "row_limit": 5000,
            "exported_at": "2026-11-08T00:00:00Z", "documents": rec["documents"], "unusable": rec["unusable"]}


def test_promotion_end_to_end(plans, tmp_path, monkeypatch):
    src, spans = _rows(plans)
    exp = tmp_path / "export.json"
    exp.write_text(json.dumps(_export(src, spans)), encoding="utf-8")
    sha = hashlib.sha256(exp.read_bytes()).hexdigest()
    man = tmp_path / "pins.json"
    man.write_text(json.dumps({"_doc": "x", "schema_version": 1, "mode": "on", "batches": [], "documents": {}}), encoding="utf-8")
    tool = _tool()
    assert tool.main(["--export", str(exp), "--export-sha256", "0" * 64, "--batch-id", "B1", "--manifest", str(man)]) == 3
    assert tool.main(["--export", str(exp), "--export-sha256", sha, "--batch-id", "B1", "--manifest", str(man)]) == 0
    m = json.loads(man.read_text(encoding="utf-8"))
    assert ap.validate(m) is None and len(m["documents"]) == 2 and m["batches"][0]["export_sha256"] == sha
    assert m["_doc"] == "x"
    monkeypatch.setattr(ap, "CONFIG_PATH", man)
    ap._CACHE.clear()
    assert len(_load(*_rows(plans)).acquired) == 2, "promoted documents are served under mode on"
    # the same export again: refused (already promoted)
    assert tool.main(["--export", str(exp), "--export-sha256", sha, "--batch-id", "B2", "--manifest", str(man)]) == 3
    # a later export of a tampered row: TAMPER-SUSPECT, nothing written, and the loader refuses the row anyway
    s2, sp2 = _rows(plans)
    _tamper_text(s2, sp2)
    exp2 = tmp_path / "export2.json"
    exp2.write_text(json.dumps(_export(s2, sp2)), encoding="utf-8")
    before = man.read_bytes()
    sha2 = hashlib.sha256(exp2.read_bytes()).hexdigest()
    assert tool.main(["--export", str(exp2), "--export-sha256", sha2, "--batch-id", "B3", "--manifest", str(man)]) == 3
    assert man.read_bytes() == before
    ap._CACHE.clear()
    assert len(_load(s2, sp2).acquired) == 1


def test_promotion_skips_unverified_and_never_pins_text(plans, tmp_path):
    src, spans = _rows(plans, 1)
    e = _export(src, spans)
    e["documents"].append({**e["documents"][0], "document_id": "acq-1111111111111111", "verdict": "R-ACQ-INJECT",
                           "text_sha256": "1" * 64})
    tool = _tool()
    new, rep = tool.propose({"schema_version": 1, "mode": "off", "batches": [], "documents": {}}, e, "2" * 64, "B1", "2026-11-08")
    assert rep["added"] == [e["documents"][0]["document_id"]] and "acq-1111111111111111" in rep["skipped"]
    assert set(next(iter(new["documents"].values()))) == set(ap.PIN_FIELDS) | {"batch"}


# ------------------------------------------------------------------------------------------------ tag hygiene
def test_ephemeral_tag_set_only_at_in_memory_construction_sites():
    sites = set()
    for f in sorted((REPO / "vera").rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        rel = f.relative_to(REPO).as_posix()

        def walk(node, fn):
            for ch in ast.iter_child_nodes(node):
                name = ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
                if isinstance(ch, ast.Dict):
                    for k, v in zip(ch.keys, ch.values):
                        if (isinstance(k, ast.Attribute) and k.attr == "EPHEMERAL_KEY") or \
                                (isinstance(k, ast.Constant) and k.value == ap.EPHEMERAL_KEY):
                            sites.add(f"{rel}:{name}")
                if isinstance(ch, ast.Subscript) and isinstance(ch.ctx, ast.Store):
                    sl = ch.slice
                    if (isinstance(sl, ast.Attribute) and sl.attr == "EPHEMERAL_KEY") or \
                            (isinstance(sl, ast.Constant) and sl.value == ap.EPHEMERAL_KEY):
                        sites.add(f"{rel}:{name}")
                walk(ch, name)
        walk(tree, None)
    assert sites == {"vera/auto_acquire.py:relevance_gate", "vera/inline_acquire.py:ephemeral_corpus"}, sorted(sites)
