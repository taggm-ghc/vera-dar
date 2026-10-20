"""D-055 ACQ: vera.fulldoc_corpus on synthetic bundles (tests/fulldoc_bundle_synth.py) + the real DAR-9 bundle if present."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from tests import fulltext_synth
from tests.fulldoc_bundle_synth import MANIFEST_NAME, SUMS_NAME, sha_file, synth_cfg, write_bundle
from vera import fulldoc_corpus as fc

Q = "transmission incidence outbreak infection"


def _unusable_ids(c):
    return {u["document_id"]: u["code"] for u in c.unusable}


def test_happy_path(tmp_path):
    b = write_bundle(tmp_path)
    c = fc.load_bundle(b, synth_cfg())
    assert c.status == "ok" and len(c.acquired) == 3 and not c.unusable
    a = c.acquired[0]
    for k in ("document_id", "canonical_source_id", "title", "url", "doi", "published", "authors", "licence",
              "licence_url", "licence_decision", "attribution", "text_sha256", "chars"):
        assert k in a
    assert a["canonical_source_id"].startswith("https://doi.org/10.9999/")
    assert a["licence_decision"] == "allow" and a["licence"] == "CC BY 4.0"
    spans = c.retrieve(Q, 12)
    assert spans
    for s in spans:
        assert s["text"] == c.text(s["document_id"])[s["start"]:s["end"]]
        assert s["span_id"] == f"{s['document_id']}:{s['start']}-{s['end']}"
    assert len({s["span_id"] for s in spans}) == len(spans)
    assert spans[0]["document_id"] == "syn_epi"
    insp = c.inspection()
    assert set(insp) == {"syn_epi", "syn_clim", "syn_repro"} and all(v["chunks_inspected"] > 1 for v in insp.values())


def test_pins_match_ok_and_mismatch(tmp_path):
    b = write_bundle(tmp_path)
    ok = synth_cfg({"manifest_sha256": sha_file(b / MANIFEST_NAME), "sums_sha256": sha_file(b / SUMS_NAME)})
    assert fc.load_bundle(b, ok).status == "ok"
    bad = synth_cfg({"manifest_sha256": "0" * 64})
    c = fc.load_bundle(b, bad)
    assert c.status == "corpus_unusable" and not c.acquired and c.retrieve(Q) == []
    assert {u["code"] for u in c.unusable} == {"R-PIN"} and len(c.unusable) == 3
    c = fc.load_bundle(b, synth_cfg({"sums_sha256": "f" * 64}))
    assert c.status == "corpus_unusable" and not c.acquired


def test_tampered_text_unusable_never_retrieved(tmp_path):
    b = write_bundle(tmp_path, tamper={"syn_clim": " TAMPERED"})
    c = fc.load_bundle(b, synth_cfg())
    assert c.status == "ok" and _unusable_ids(c) == {"syn_clim": "R-SUMS"}
    assert {a["document_id"] for a in c.acquired} == {"syn_epi", "syn_repro"}
    assert all(s["document_id"] != "syn_clim" for s in c.retrieve("temperature radiative forcing ensemble", 12))
    with pytest.raises(KeyError):
        c.text("syn_clim")


def test_missing_sums_line(tmp_path):
    b = write_bundle(tmp_path)
    p = b / SUMS_NAME
    p.write_text("".join(ln for ln in p.read_text().splitlines(True) if "syn_repro.meta.json" not in ln))
    c = fc.load_bundle(b, synth_cfg())
    assert _unusable_ids(c) == {"syn_repro": "R-SUMS"} and len(c.acquired) == 2


def test_missing_bundle_files_and_garbage(tmp_path):
    assert fc.load_bundle(tmp_path / "nope", synth_cfg()).status == "corpus_unusable"
    b = write_bundle(tmp_path / "b")
    (b / MANIFEST_NAME).write_text("not json")
    assert fc.load_bundle(b, synth_cfg()).status == "corpus_unusable"


def test_nd_licence_unusable(tmp_path):
    nd = {"licence": "CC BY-ND 4.0", "licence_url": "https://creativecommons.org/licenses/by-nd/4.0/"}
    b = write_bundle(tmp_path, items_over={"syn_epi": nd})
    c = fc.load_bundle(b, synth_cfg())
    assert list(_unusable_ids(c)) == ["syn_epi"] and _unusable_ids(c)["syn_epi"].startswith("R-LIC")
    assert len(c.acquired) == 2


def test_abstract_only_unusable(tmp_path):
    b = write_bundle(tmp_path, jats_over={"syn_clim": {"with_body": False}})
    c = fc.load_bundle(b, synth_cfg())
    assert _unusable_ids(c) == {"syn_clim": "R-ABSTRACT"} or list(_unusable_ids(c)) == ["syn_clim"]
    assert len(c.acquired) == 2


def test_traversal_text_file_unusable(tmp_path):
    (tmp_path / "outside.txt").write_text("x")
    b = write_bundle(tmp_path / "b", items_over={"syn_epi": {"text_file": "../outside.txt"}})
    c = fc.load_bundle(b, synth_cfg())
    assert list(_unusable_ids(c)) == ["syn_epi"] and len(c.acquired) == 2


def test_env_unset_returns_none_and_cache(tmp_path, monkeypatch):
    fc.reset_for_tests()
    cfg = synth_cfg()
    assert fc.get_corpus(cfg, env={}) is None
    assert fc.get_corpus(cfg, env={cfg["bundle_dir_env"]: ""}) is None
    b = write_bundle(tmp_path)
    env = {cfg["bundle_dir_env"]: str(b)}
    c1 = fc.get_corpus(cfg, env=env)
    assert c1.status == "ok" and fc.get_corpus(cfg, env=env) is c1
    fc.reset_for_tests()
    assert fc.get_corpus(cfg, env=env) is not c1


def test_routing_config_not_modified(tmp_path):
    p = fc.ROUTING_PATH
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    fc.load_bundle(write_bundle(tmp_path), synth_cfg())
    assert hashlib.sha256(p.read_bytes()).hexdigest() == before


REAL = os.environ.get("VERA_REAL_DAR9_BUNDLE") or ""  # no default path: set the env var to run this test


@pytest.mark.skipif(not REAL or not (Path(REAL) / MANIFEST_NAME).is_file(), reason="real DAR-9 bundle not present")
def test_real_bundle_nine_acquired():
    c = fc.load_bundle(Path(REAL), fc.load_config())  # shipped pins
    assert c.status == "ok", c.unusable
    assert len(c.acquired) == 9, c.unusable


def test_same_text_under_another_canonical_url_is_a_duplicate(tmp_path):  # D-057 parity (TREE-STABLE-5 dedupe)
    import hashlib
    import shutil
    from tests import fulltext_synth
    from tests.fulldoc_bundle_synth import synth_cfg, write_bundle
    from vera import fulldoc_corpus
    docs = fulltext_synth.DEFAULT_DOCS[:1] + (("syn_twin", "epi", "10.9999/syn.twin"),)
    root = write_bundle(tmp_path, docs=docs)
    t = root / "texts"
    for ext in (".txt", ".meta.json"):  # the twin carries byte-identical text under another DOI / canonical URL
        shutil.copyfile(t / f"syn_epi{ext}", t / f"syn_twin{ext}")
    sums = root / "texts.SHA256SUMS"
    sums.write_text("".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n" for f in sorted(t.iterdir())))
    c = fulldoc_corpus.load_bundle(root, synth_cfg())
    assert [d["document_id"] for d in c.acquired] == ["syn_epi"]
    assert [(u["document_id"], u["code"]) for u in c.unusable] == [("syn_twin", "R-DUP")]
