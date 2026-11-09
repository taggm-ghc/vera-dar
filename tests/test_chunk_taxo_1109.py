"""CHUNK-TAXO-CORE-1109: vera/chunk_taxo.py engine. Offline; synthetic invented words and a deterministic hashed fake embedder."""
from __future__ import annotations

import hashlib
import json
import random
import re

import pytest

from vera import chunk_taxo as ct

DIM = 128
SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def vocab(tag, n=6):
    return ["%s%s" % (tag, w) for w in ("kelv", "brin", "quos", "mard", "felp", "ozru", "tavi", "nimb")[:n]]


def embed(text):
    v = [0.0] * DIM
    for tok in text.split():
        h = hashlib.sha256(tok.encode()).digest()
        v[int.from_bytes(h[:2], "big") % DIM] += 1.0 if h[2] % 2 else -1.0
    return v


def corpus(per=12, seed=5):
    """2 domains x 2 topics x 2 subtopics; each text mixes domain, topic and subtopic words."""
    rng = random.Random(seed)
    chunks, sid = [], 1000
    for d in range(2):
        for t in range(2):
            for s in range(2):
                for _ in range(per):
                    words = (rng.sample(vocab("d%d" % d), 3) + rng.sample(vocab("t%d%d" % (d, t)), 3)
                             + rng.sample(vocab("s%d%d%d" % (d, t, s), 8), 6))
                    chunks.append((sid, 1 + (sid % 7), " ".join(words)))
                    sid += 1
    return chunks


@pytest.fixture(scope="module")
def cfg():
    c = ct.load_cfg()
    c["params"]["k_max"] = 14
    c["params"]["topic_range"]["min_count"] = 4  # the largest gap alone would merge straight to 2 domains
    return c


@pytest.fixture(scope="module")
def data():
    ch = corpus()
    return ch, [embed(c[2]) for c in ch]


@pytest.fixture(scope="module")
def model(cfg, data):
    return ct.build(data[0], data[1], cfg["params"])


def test_config_default_off_and_validation(tmp_path):
    c = ct.load_cfg()
    assert c["enabled"] is True and "params" in c and "display_gate" in c  # FLAGON-TAXO-1109: shipped on
    bad = tmp_path / "b.json"
    bad.write_text(json.dumps({"params": {"nope": 1}}))
    with pytest.raises(ValueError):
        ct.load_cfg(bad)
    bad.write_text(json.dumps({"params": {"null_q_dist": 3}}))
    with pytest.raises(ValueError):
        ct.load_cfg(bad)
    bad.write_text(json.dumps({"enabled": 1}))
    with pytest.raises(ValueError):
        ct.load_cfg(bad)


def test_determinism_and_input_order(cfg, data, model):
    ch, v = data
    again = ct.build(ch, v, cfg["params"])
    assert again.dumps() == model.dumps()
    idx = list(range(len(ch)))
    random.Random(1).shuffle(idx)
    shuf = ct.build([ch[i] for i in idx], [v[i] for i in idx], cfg["params"])
    assert shuf.dumps() == model.dumps()
    assert len(model.version) == 12


def test_hierarchy_containment_and_structure(model):
    by = {n["key"]: n for n in model.nodes}
    levels = {lv: [n for n in model.nodes if n["level"] == lv] for lv in ct.LEVELS}
    assert len(levels["subtopic"]) >= 2
    assert len(levels["topic"]) < len(levels["subtopic"]) and len(levels["domain"]) < len(levels["topic"])
    for n in model.nodes:
        if n["level"] == "domain":
            assert n["parent_key"] is None
            continue
        par = by[n["parent_key"]]
        assert par["level"] == ct.LEVELS[ct.LEVELS.index(n["level"]) + 1]
        assert set(n["members"]) <= set(par["members"])
    for lv in ct.LEVELS:
        allm = sorted(s for n in levels[lv] for s in n["members"])
        assert allm == sorted(model.span_order)
    # the planted structure is recovered: each subtopic node is pure in the planted subtopic
    planted = {c[0]: c[2].split()[-1][:3] for c in corpus()}
    for n in levels["subtopic"]:
        assert len({planted[s] for s in n["members"]}) == 1
    assert len(levels["domain"]) == 2


def test_slug_and_label_validity(model):
    keys = [n["key"] for n in model.nodes]
    assert len(set(keys)) == len(keys)
    for n in model.nodes:
        assert SLUG.match(n["key"]) and len(n["key"]) <= 64
        assert n["label"] and len(n["label"]) <= 120 and all(32 <= ord(c) < 127 for c in n["label"])
        assert n["terms"]
    for lv in ct.LEVELS:
        folded = [" ".join(n["label"].lower().split()) for n in model.nodes if n["level"] == lv]
        assert len(set(folded)) == len(folded)


def test_abstention_out_of_cluster(cfg, data, model):
    rng = random.Random(9)
    junk = [embed(" ".join("zz%dq%d" % (rng.randrange(10 ** 6), j) for j in range(8))) for _ in range(10)]
    res = [ct.associate(model, j) for j in junk]
    assert sum(1 for r in res if r["level"] is None) >= 8
    assert all(r["reason"] == "NO_FIT" for r in res if r["level"] is None)
    # in-cluster new chunks associate at the deepest level
    ch = corpus(per=2, seed=77)
    ok = [ct.associate(model, embed(c[2])) for c in ch]
    assert sum(1 for r in ok if r["level"] == "subtopic") >= len(ok) * 0.8
    for r in ok:
        if r["level"] == "subtopic":
            assert r["topic"] and r["domain"] and r["reason"] is None
    # build-time chunks mostly keep their cluster
    deep = sum(1 for a in model.assoc.values() if a["level"] == "subtopic")
    assert deep / len(model.assoc) > 0.8


def test_abstention_inside_build(cfg, data):
    ch, v = data
    rng = random.Random(3)
    extra = [(5000 + i, 99, " ".join("zq%d" % rng.randrange(10 ** 6) for _ in range(9))) for i in range(4)]
    m = ct.build(ch + extra, v + [embed(e[2]) for e in extra], cfg["params"])
    assert sum(1 for i in range(4) if m.assoc[5000 + i]["level"] is None) >= 3
    assert all(m.assoc[5000 + i]["reason"] for i in range(4) if m.assoc[5000 + i]["level"] is None)


def test_identity_reuse_and_retire(cfg, data, model):
    ch, v = data
    keep = [i for i in range(len(ch)) if i % 10 != 0]
    m2 = ct.build([ch[i] for i in keep] + [(9000, 1, ch[0][2])], [v[i] for i in keep] + [v[0]], cfg["params"],
                  prior=model)
    reused = [n for n in m2.nodes if n["reused_from"]]
    assert len(reused) >= len(m2.nodes) - 2
    assert all(n["key"] == n["reused_from"] for n in reused)
    assert len({n["key"] for n in m2.nodes}) == len(m2.nodes)
    assert set(m2.retired_keys).isdisjoint({n["key"] for n in m2.nodes})
    # dict prior works, and a prior with an unrelated node retires it
    pd = model.to_dict()
    pd["nodes"] = pd["nodes"] + [{"key": "st-gone-abcdef", "level": "subtopic", "members": [1, 2, 3]}]
    m3 = ct.build(ch, v, cfg["params"], prior=pd)
    assert "st-gone-abcdef" in m3.retired_keys and all(n["reused_from"] == n["key"] for n in m3.nodes)
    # no prior overlap: nothing reused
    m4 = ct.build(ch, v, cfg["params"], prior={"nodes": [{"key": "x1", "level": "topic", "members": [1]}]})
    assert not any(n["reused_from"] for n in m4.nodes) and m4.retired_keys == ["x1"]


def test_union_vs_vote(model):
    subs = [n for n in model.nodes if n["level"] == "subtopic"]
    topics = {n["key"]: n for n in model.nodes if n["level"] == "topic"}
    # one source with chunks from two different topics, one pure source
    t_keys = list(topics)
    a = [s for s in topics[t_keys[0]]["members"] if model.assoc[s]["level"] == "subtopic"]
    b = [s for s in topics[t_keys[1]]["members"] if model.assoc[s]["level"] == "subtopic"]
    mixed = {s: 1 for s in a[:2] + b[:2]}
    mixed.update({s: 2 for s in a[2:6]})
    r = ct.doc_subjects(model, mixed)
    assert set(r) == {1, 2}
    assert t_keys[0] in r[1]["union"] and t_keys[1] in r[1]["union"]  # union carries both
    assert r[1]["vote"]["topic"] is None  # 50/50 -> no lead
    assert r[2]["vote"]["topic"] == t_keys[0] and r[2]["vote"]["share"] == 1.0
    assert r[2]["vote"]["domain"] == topics[t_keys[0]]["parent_key"]
    assert sum(r[2]["union"][k] for k in r[2]["union"] if k in topics) == 4
    assert subs  # sanity


def test_quality_and_gate(model, data, cfg):
    q = ct.quality(model, data[1])
    assert set(q) >= {"silhouette", "stability", "null_floor_spread", "coverage", "abstention"}
    assert q["silhouette"]["subtopic"] > 0.1
    assert 0 <= q["coverage"]["domain"] <= 1
    assert q["stability"]["ari"]["subtopic"] is not None and q["stability"]["ari"]["subtopic"] > 0.6
    ok, why = ct.display_gate(q, {"min_silhouette_subtopic": 0.0, "max_abstain_share": 1.0})
    assert ok and why == []
    ok, why = ct.display_gate(q, {"min_silhouette_subtopic": 0.999})
    assert not ok and "silhouette_subtopic" in why[0]
    ok, why = ct.display_gate({}, {"min_coverage_subtopic": 0.5, "min_stability_ari": None})
    assert not ok and why == ["coverage_subtopic unavailable"]
    assert ct.display_gate(q, {})[0] is True
    assert q == ct.quality(model, data[1])  # deterministic


def test_small_inputs_do_not_crash(cfg):
    ch = [(1, 1, "alpha beta gamma"), (2, 1, "alpha beta delta")]
    m = ct.build(ch, [[1.0, 0.0], [0.9, 0.1]], cfg["params"])
    assert m.meta["levels"] == {"subtopic": 1, "topic": 1, "domain": 1}
    assert ct.quality(m, [[1.0, 0.0], [0.9, 0.1]])["silhouette"]["subtopic"] is None
    with pytest.raises(ValueError):
        ct.build(ch, [[0.0, 0.0], [1.0, 0.0]], cfg["params"])
    with pytest.raises(ValueError):
        ct.build(ch + [(1, 1, "x")], [[1.0, 0.0]] * 3, cfg["params"])


@pytest.mark.skipif(not ct.has_numpy(), reason="numpy not installed")
def test_numpy_fallback_parity(cfg, data):
    ch, v = data
    try:
        ct.set_backend(True)
        a = ct.build(ch, v, cfg["params"])
        ct.set_backend(False)
        b = ct.build(ch, v, cfg["params"])
    finally:
        ct.set_backend(None)
    da, db = a.to_dict(), b.to_dict()
    assert da["meta"].pop("numpy") is True and db["meta"].pop("numpy") is False
    for x, y in zip(da["nodes"], db["nodes"]):
        for c1, c2 in zip(x.pop("center"), y.pop("center")):
            assert abs(c1 - c2) < 1e-7
    assert da == db


@pytest.mark.skipif(not ct.has_numpy(), reason="numpy not installed")
def test_ndarray_input_matches_list_input(cfg, data):
    import numpy as np
    ch, v = data
    ref = ct.build(ch, v, cfg["params"])
    for dt in (np.float64, np.float32):
        arr = np.array(v, dtype=dt)
        lst = arr.astype(np.float64).tolist()  # what a float32 array holds, as exact doubles
        a, b = ct.build(ch, arr, cfg["params"]), ct.build(ch, lst, cfg["params"])
        assert a.dumps() == b.dumps()
        assert ct.quality(a, arr) == ct.quality(b, lst)
    assert ct.build(ch, np.array(v, dtype=np.float64), cfg["params"]).dumps() == ref.dumps()
    idx = list(range(len(ch)))[::-1]
    assert ct.build([ch[i] for i in idx], np.array(v)[idx], cfg["params"]).dumps() == ref.dumps()
    with pytest.raises(ValueError):
        ct.build(ch, np.zeros((len(ch), DIM)), cfg["params"])
