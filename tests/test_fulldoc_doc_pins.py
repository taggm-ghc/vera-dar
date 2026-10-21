"""AE-1020/DBC-M3 (security finding E3): per-document identity pins in config/fulldoc_answer.json.
A coordinated tamper (span text + provenance.text_sha256 + meta.text_sha256 rewritten together) is self-consistent, so the
recorded hashes alone cannot catch it; the config pins can. SYNTHETIC bundle + fake rows: no DB, no network, no env files."""
import copy
import json
import re
import hashlib
from pathlib import Path

import pytest

from tests import fulldoc_bundle_synth as synth
from tests.test_fulldoc_db import _db, _script
from vera import fulldoc_corpus as fc, fulldoc_db

REPO = Path(__file__).resolve().parent.parent
SHIPPED = REPO / "config" / "fulldoc_answer.json"
REAL_BUNDLE = REPO.parent / "work" / "fd-1020" / "bundle"
HEX64 = re.compile(r"[0-9a-f]{64}")
CODE = fulldoc_db.R_DBDOCPIN


@pytest.fixture()
def world(tmp_path):
    fc.reset_for_tests()
    root = synth.write_bundle(tmp_path / "b")
    cfg = synth.synth_cfg()
    rows, report = _script().build_plan(root, cfg, target=700)
    sources, spans = _db(rows)
    docs = {}
    for s in sources:
        p = s["provenance"]
        docs[p["document_id"]] = {"text_sha256": p["text_sha256"], "licence_id": p["item"]["licence"],
                                  "licence_decision": "allow"}
    pinned = copy.deepcopy(cfg)
    pinned["pins"] = {**report["pins"], "documents": copy.deepcopy(docs)}
    return cfg, pinned, sources, spans, docs


def _coordinated_tamper(sources, spans):
    sp = next(x for x in spans if x["source_id"] == 1 and re.search("[a-z]", x["text"]))
    i = re.search("[a-z]", sp["text"]).start()
    sp["text"] = sp["text"][:i] + ("z" if sp["text"][i] != "z" else "y") + sp["text"][i + 1:]
    text = "".join(x["text"] for x in sorted((x for x in spans if x["source_id"] == 1), key=lambda x: x["start_index"]))
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    sources[0]["provenance"]["text_sha256"] = h
    sources[0]["provenance"]["meta"]["text_sha256"] = h


def test_shipped_config_has_nine_document_pins():
    pins = json.loads(SHIPPED.read_text(encoding="utf-8"))["pins"]
    docs = pins["documents"]
    assert len(docs) == 9 and isinstance(pins.get("_documents_note"), str) and pins["_documents_note"]
    for did, d in docs.items():
        assert HEX64.fullmatch(d["text_sha256"]), did
        assert d["licence_decision"] == "allow" and d["licence_id"], did
    if not (REAL_BUNDLE / "manifest-filled.https.json").is_file():
        pytest.skip("UNVERIFIED: work/fd-1020/bundle is absent; manifest id set not compared")
    manifest = json.loads((REAL_BUNDLE / "manifest-filled.https.json").read_text(encoding="utf-8"))
    assert set(docs) == {s["document_id"] for s in manifest["sources"]}
    assert hashlib.sha256((REAL_BUNDLE / "manifest-filled.https.json").read_bytes()).hexdigest() == pins["manifest_sha256"]


def test_matching_doc_pins_load_ok(world):
    _, pinned, sources, spans, _ = world
    c = fulldoc_db.load_from_rows(sources, spans, pinned)
    assert c.status == "ok" and not c.unusable and len(c.acquired) == 3


def test_coordinated_tamper_is_accepted_without_doc_pins_and_rejected_with_them(world):
    cfg, pinned, sources, spans, _ = world
    plain = copy.deepcopy(cfg)

    plain["pins"] = {"manifest_sha256": None, "sums_sha256": None}  # fully unpinned: E3 demonstrated
    pristine = "".join(x["text"] for x in sorted((x for x in spans if x["source_id"] == 1), key=lambda x: x["start_index"]))
    _coordinated_tamper(sources, spans)
    ok = fulldoc_db.load_from_rows(copy.deepcopy(sources), copy.deepcopy(spans), plain)
    assert ok.status == "ok" and not ok.unusable and ok._texts["syn_epi"] != pristine
    bad = fulldoc_db.load_from_rows(sources, spans, pinned)
    assert bad.status == "corpus_unusable" and bad._texts == {} and not bad.acquired
    assert {u["code"] for u in bad.unusable} == {CODE}


def test_other_allowed_licence_with_recomputed_row_is_rejected(world):
    _, pinned, sources, spans, _ = world
    it = sources[0]["provenance"]["item"]
    it["licence"], it["licence_url"] = "CC BY 3.0", "https://creativecommons.org/licenses/by/3.0/"
    prov = sources[0]["provenance"]  # keep the row self-consistent: stored licence, embedded licence and conversion record
    for d in (prov["licence"]["content"], prov["licence"].get("embedded") or {}, prov["meta"]["meta"]["licence"]):
        d["id"], d["url"] = it["licence"], it["licence_url"]
    c = fulldoc_db.load_from_rows(sources, spans, pinned)
    assert c.status == "corpus_unusable" and not c.acquired and {u["code"] for u in c.unusable} == {CODE}


@pytest.mark.parametrize("how", ["missing", "extra", "manifest_pin_without_docs", "empty", "wrong_decision"])
def test_missing_extra_or_absent_pins_reject_the_run(world, how):
    _, pinned, sources, spans, docs = world
    p = pinned["pins"]
    if how == "missing":
        p["documents"].pop(next(iter(docs)))
    elif how == "extra":
        p["documents"]["not_in_run"] = dict(next(iter(docs.values())))
    elif how == "manifest_pin_without_docs":
        del p["documents"]
    elif how == "empty":
        p["documents"] = {}
    else:
        p["documents"][next(iter(docs))]["licence_decision"] = "hold"
    c = fulldoc_db.load_from_rows(sources, spans, pinned)
    assert c.status == "corpus_unusable" and not c.acquired and c._texts == {}
    assert {u["code"] for u in c.unusable} == {CODE}


def test_wrong_text_pin_rejects_whole_run(world):
    _, pinned, sources, spans, docs = world
    pinned["pins"]["documents"][next(iter(docs))]["text_sha256"] = "0" * 64
    c = fulldoc_db.load_from_rows(sources, spans, pinned)
    assert c.status == "corpus_unusable" and not c.acquired and {u["code"] for u in c.unusable} == {CODE}


def test_unpinned_config_skips_doc_pins(world):
    cfg, _, sources, spans, _ = world
    assert fulldoc_db.load_from_rows(sources, spans, cfg).status == "ok"


def test_uncoordinated_span_tamper_with_pins_rejects_whole_run(world):
    _, pinned, sources, spans, _ = world
    sp = next(x for x in spans if x["source_id"] == 1 and re.search("[a-z]", x["text"]))
    i = re.search("[a-z]", sp["text"]).start()
    sp["text"] = sp["text"][:i] + ("z" if sp["text"][i] != "z" else "y") + sp["text"][i + 1:]  # row hashes NOT recomputed
    c = fulldoc_db.load_from_rows(sources, spans, pinned)
    assert c.status == "corpus_unusable" and not c.acquired and c._texts == {}
    assert {u["code"] for u in c.unusable} == {CODE}


def test_non_allow_licence_row_with_pins_rejects_whole_run(world):
    _, pinned, sources, spans, _ = world
    it = sources[0]["provenance"]["item"]
    it["licence"], it["licence_url"] = "All rights reserved", "https://example.org/arr"
    prov = sources[0]["provenance"]
    for d in (prov["licence"]["content"], prov["licence"].get("embedded") or {}, prov["meta"]["meta"]["licence"]):
        d["id"], d["url"] = it["licence"], it["licence_url"]
    prov["licence_decision"] = "hold"
    c = fulldoc_db.load_from_rows(sources, spans, pinned)
    assert c.status == "corpus_unusable" and not c.acquired and {u["code"] for u in c.unusable} == {CODE}


def test_sums_only_pin_without_doc_pins_rejects_run(world):
    _, pinned, sources, spans, _ = world
    pinned["pins"]["manifest_sha256"] = None
    del pinned["pins"]["documents"]
    assert pinned["pins"].get("sums_sha256")
    c = fulldoc_db.load_from_rows(sources, spans, pinned)
    assert c.status == "corpus_unusable" and not c.acquired and {u["code"] for u in c.unusable} == {CODE}


def test_pinned_dar_ingest_refusal_rejects_whole_run(world, monkeypatch):
    _, pinned, sources, spans, _ = world
    monkeypatch.setattr(fulldoc_db.DarCorpus, "ingest_document",
                        lambda self, d: {"status": "refused", "reason": "synthetic_refusal"})
    c = fulldoc_db.load_from_rows(sources, spans, pinned)
    assert c.status == "corpus_unusable" and not c.acquired and c._texts == {}
    assert {u["code"] for u in c.unusable} == {CODE}


def test_unpinned_dar_ingest_refusal_still_drops_only_that_doc(world, monkeypatch):
    cfg, _, sources, spans, _ = world
    monkeypatch.setattr(fulldoc_db.DarCorpus, "ingest_document",
                        lambda self, d: {"status": "refused", "reason": "synthetic_refusal"})
    c = fulldoc_db.load_from_rows(sources, spans, cfg)
    assert c.status == "ok" and not c.acquired and c.unusable


def test_pinned_generic_exception_in_document_loop_rejects_whole_run(world, monkeypatch):
    _, pinned, sources, spans, _ = world

    def boom(*a, **k):
        raise RuntimeError("synthetic")
    monkeypatch.setattr(fulldoc_db.fulltext_loader, "prepare_from_parts", boom)
    c = fulldoc_db.load_from_rows(sources, spans, pinned)
    assert c.status == "corpus_unusable" and not c.acquired and c._texts == {}
    assert {u["code"] for u in c.unusable} == {CODE}
