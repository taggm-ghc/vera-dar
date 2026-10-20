"""Item #91 mechanics tests on synthetic vectors. These prove the code works; they do NOT calibrate any threshold."""
import json

import numpy as np
import pytest

from vera import semantic_space as ss


@pytest.fixture
def cfg():
    return ss.load_config()


def _doc(rng, n=12, d=20):
    axis = np.zeros(d); axis[0] = 1.0
    return axis + 0.15 * rng.normal(size=(n, d))


def test_shipped_config_is_disabled_and_uncalibrated(cfg):
    assert cfg["enabled"] is False and cfg["calibration_status"] == "UNCALIBRATED"


def test_enable_refused_without_calibration(tmp_path, cfg):
    p = tmp_path / "c.json"; p.write_text(json.dumps({**cfg, "enabled": True}))
    with pytest.raises(ss.SemanticSpaceConfigError, match="calibration"):
        ss.load_config(p)
    p.write_text(json.dumps({**cfg, "enabled": True, "calibration_status": "CALIBRATED", "calibration_note": "n"}))
    assert ss.load_config(p)["enabled"] is True


def test_missing_key_fails_verbosely(tmp_path, cfg):
    bad = dict(cfg); bad.pop("fence_multiplier")
    p = tmp_path / "c.json"; p.write_text(json.dumps(bad))
    with pytest.raises(ss.SemanticSpaceConfigError, match="fence_multiplier"):
        ss.load_config(p)


def test_too_few_chunks_gives_none_and_undefined(cfg):
    X = _doc(np.random.default_rng(0), n=cfg["min_chunks_per_doc"] - 1)
    assert ss.fit_doc_space(X, cfg) is None
    assert ss.grade_vectors(X[:1], None, cfg)["grade"] == "undefined"


def test_inside_versus_outside(cfg):
    rng = np.random.default_rng(1)
    sp = ss.fit_doc_space(_doc(rng), cfg)
    assert sp is not None and 0 <= sp.weight <= 1 and sp.n_chunks == 12
    near = _doc(rng, n=1)
    far = np.zeros((1, 20)); far[0, 7] = 1.0
    assert ss.grade_vectors(far, sp, cfg)["grade"] == "outside"
    assert ss.grade_vectors(near, sp, cfg)["grade"] in ("inside", "bordering")


def test_deterministic(cfg):
    X = _doc(np.random.default_rng(2))
    a, b = ss.fit_doc_space(X, cfg), ss.fit_doc_space(X, cfg)
    assert a.stability == b.stability and a.maha_fences == b.maha_fences


def test_stability_of_identical_bases_is_one():
    e = np.eye(5)[:3]
    assert ss.principal_angle_stability(e, e) == pytest.approx(1.0)
    assert ss.principal_angle_stability(e, np.eye(5)[3:5]) == pytest.approx(0.0)


def test_hybrid_blend_endpoints(cfg):
    assert ss.hybrid_score(0.0, "inside", 0.3, cfg) == pytest.approx(0.3)   # thin doc: status quo only
    assert ss.hybrid_score(1.0, "inside", 0.3, cfg) == pytest.approx(cfg["grade_scores"]["inside"])
    with pytest.raises(ValueError):
        ss.hybrid_score(1.5, "inside", 0.3, cfg)


def test_rrf_weights_shift_ranking(cfg):
    fw, sq = ["a", "b"], ["b", "a"]
    assert ss.rrf_fuse(fw, sq, {"a": 1, "b": 1}, cfg)[0] == "a"
    assert ss.rrf_fuse(fw, sq, {"a": 0, "b": 0}, cfg)[0] == "b"


def test_chunking_floor_and_exclusion(cfg):
    n, mn = cfg["min_chunks_per_doc"], cfg["min_chunk_chars"]
    assert ss.chunk_text("short text", cfg) is None
    text = " ".join(f"Sentence number {i} says something about topic." for i in range(12))
    ch = ss.chunk_text(text, cfg)
    assert ch is not None and len(ch) >= n and all(len(c) >= mn for c in ch)
    assert " ".join(ch).split() == text.split()           # non-overlapping, lossless
