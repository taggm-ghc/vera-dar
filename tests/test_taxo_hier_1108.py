"""TAXO-HIER-1108: hierarchical matcher (vera/taxo_hier.py) and its inert shadow seam. Offline; synthetic slugs and words only."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import pytest

from vera import corpus_description as cd, micro_topic, taxo_hier as th

ROOT = Path(__file__).resolve().parent.parent
W = {  # synthetic topic vocabularies (invented words)
    "t1": "kelvor brindle quosh mardan felpit ozrune tavik", "t2": "ploxa drenth yumari sovik caldra wenfo hurzit",
    "t3": "nimbat orquel zestra bolvik tamuri geshon lopaz", "t4": "ruvix hallop senrod cuthra dimbek foravi yelnor",
    "s1": "wibrel tonkas parvud ekrol sundiv jothem ilquar", "x": "quarbo lenzik moffat varnis detroz abkil sipran"}
ENTRIES = [
    {"id": "syn-t1", "label": "Synthetic one", "field": "dom-a", "anchor_text": W["t1"]},
    {"id": "syn-t2", "label": "Synthetic two", "field": "dom-a", "anchor_text": W["t2"]},
    {"id": "syn-t3", "label": "Synthetic three", "field": "dom-a", "anchor_text": W["t3"]},
    {"id": "syn-t4", "label": "Synthetic four", "field": "dom-b", "anchor_text": W["t4"]}]
SUB = [{"id": "syn-s1", "label": "Synthetic sub", "field": "dom-a", "parent_topic_key": "syn-t1", "anchor_text": W["s1"]}]
V = ["w%04d" % i for i in range(2500)]
ZIPF = [1.0 / (i + 1) for i in range(2500)]


def chunk(rng, topic=None, share=0.15, n=150, extra=None):
    tw = (W[topic].split() if topic else []) + (W[extra].split() if extra else [])
    return " ".join(rng.choice(tw) if tw and rng.random() < share else rng.choices(V, ZIPF)[0] for _ in range(n))


def make_corpus(seed=1, sub_share=0.0):
    rng = random.Random(seed)
    docs = {}
    for t in ("t1", "t2", "t3", "t4", "x"):
        for j in range(8):
            docs[f"{t}-{j}"] = [chunk(rng, t) for _ in range(8)]
    for j in range(8):
        docs[f"g-{j}"] = [chunk(rng) for _ in range(8)]
    for j in range(4):
        docs[f"sub-{j}"] = [chunk(rng, "s1", 0.2) for _ in range(8)]
    docs["split-0"] = [chunk(rng, t) for t in ("t1", "t2", "t3") for _ in range(3)]
    return docs


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


@pytest.fixture(scope="module")
def model(corpus):
    return th.Model(ENTRIES, [c for cs in corpus.values() for c in cs])


@pytest.fixture(scope="module")
def model_sub(corpus):
    return th.Model(ENTRIES + SUB, [c for cs in corpus.values() for c in cs])


# --- config loader --------------------------------------------------------------------------------------------------
def test_shipped_config_is_off_and_uncalibrated():
    raw = json.loads((ROOT / "config" / "taxo_hier.json").read_text())
    assert raw["mode"] == "off" and raw["calibration_status"] == "UNCALIBRATED"
    assert raw["params"] == th.DEFAULT_PARAMS
    assert th.load_config()["mode"] == "off"


def test_missing_file_is_off(tmp_path):
    assert th.load_config(tmp_path / "nope.json")["mode"] == "off"


def _cfg(tmp_path, **kw):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(kw))
    return p


def test_bad_mode_and_unreadable_fail_closed(tmp_path):
    with pytest.raises(th.ConfigError):
        th.load_config(_cfg(tmp_path, mode="turbo"))
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(th.ConfigError):
        th.load_config(p)


def test_active_refused_unless_calibrated_with_note(tmp_path):
    for kw in ({}, {"calibration_status": "UNCALIBRATED", "calibration_note": "x"},
               {"calibration_status": "CALIBRATED"}, {"calibration_status": "CALIBRATED", "calibration_note": "  "}):
        with pytest.raises(th.ConfigError):
            th.load_config(_cfg(tmp_path, mode="active", **kw))
    ok = th.load_config(_cfg(tmp_path, mode="active", calibration_status="CALIBRATED", calibration_note="kit run out/"))
    assert ok["mode"] == "active"


@pytest.mark.parametrize("bad", [{"fit_quantile": 1.5}, {"fit_quantile": 0.1}, {"min_top_share": -0.1}, {"min_agree": 2},
                                 {"null_replicates": 0}, {"null_chunks": 10}, {"seed": True}, {"semantic_weight": 3},
                                 {"fit_quantile": "0.9"}, {"min_token_len": 1.5}])
def test_param_ranges(bad, tmp_path):
    with pytest.raises(th.ConfigError):
        th.load_config(_cfg(tmp_path, mode="shadow", params=bad))


def test_classify_override_rejects_non_threshold_keys(model, corpus):
    with pytest.raises(th.ConfigError):
        model.classify(corpus["t1-0"], {"seed": 5})
    with pytest.raises(th.ConfigError):
        model.classify(corpus["t1-0"], {"fit_quantile": 7})
    assert model.classify(corpus["t1-0"], {"fit_quantile": 0.9})["stats"]


# --- taxonomy -------------------------------------------------------------------------------------------------------
def test_taxonomy_levels_path_and_subtopic_depth():
    t = th.Taxonomy(ENTRIES + SUB)
    assert t.level_keys("domain") == ["domain:dom-a", "domain:dom-b"]
    assert t.level_keys("topic") == ["syn-t1", "syn-t2", "syn-t3", "syn-t4"]
    assert t.level_keys("subtopic") == ["syn-s1"]
    assert t.path("syn-s1") == {"domain": "domain:dom-a", "topic": "syn-t1", "subtopic": "syn-s1"}
    assert t.path("syn-t4") == {"domain": "domain:dom-b", "topic": "syn-t4", "subtopic": None}
    assert t.descendants_and_self("domain:dom-a") >= {"syn-t1", "syn-s1"}
    deep = th.Taxonomy(ENTRIES + SUB + [{"id": "syn-ss", "label": "x", "field": "dom-a", "parent_topic_key": "syn-s1", "anchor_text": "q"}])
    assert "syn-ss" not in deep.nodes  # depth <= 2
    assert th.Taxonomy([]).order == [] and th.Taxonomy(None).order == []


# --- corpus IDF -----------------------------------------------------------------------------------------------------
def test_corpus_idf_downweights_corpus_common_word():
    chunks = [("everywhere rare%d" % (i % 3)) for i in range(30)] + ["everywhere"] * 30
    entries = [{"id": "syn-a", "label": "n", "field": "f", "anchor_text": "everywhere rare0"}]
    m = th.Model(entries, chunks)
    assert m.idf["everywhere"] < m.idf["rare0"]
    v = m.node_lex["syn-a"]
    assert v["everywhere"] < v["rare0"]


# --- classification -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("t,dom,top", [("t1", "domain:dom-a", "syn-t1"), ("t2", "domain:dom-a", "syn-t2"),
                                       ("t3", "domain:dom-a", "syn-t3"), ("t4", "domain:dom-b", "syn-t4")])
def test_in_taxonomy_docs_reach_topic(model, corpus, t, dom, top):
    hits = [model.classify(corpus[f"{t}-{j}"]) for j in range(8)]
    assert sum(r["key"] == top for r in hits) >= 7
    r = hits[0]
    assert r["path"]["domain"] == dom and r["reason"] == "NO_CHILD_NODES" and r["level"] == "topic"


def test_out_of_taxonomy_docs_abstain_no_fit(model, corpus):
    rs = [model.classify(corpus[k]) for k in corpus if k.startswith(("x-", "g-"))]
    assert sum(r["level"] is None and r["reason"] == "NO_FIT" for r in rs) >= 14 and len(rs) == 16
    assert all(r["path"] == {"domain": None, "topic": None, "subtopic": None} for r in rs if r["level"] is None)


def test_split_document_backs_off_to_domain(model, corpus):
    r = model.classify(corpus["split-0"])
    assert r["level"] == "domain" and r["key"] == "domain:dom-a"
    assert r["reason"] in ("SPLIT", "LOW_SUPPORT") and r["path"]["topic"] is None
    assert r["stats"]["domain"]["top"] == "domain:dom-a"


def test_no_text(model):
    for c in (None, [], ["", "   "]):
        r = model.classify(c)
        assert r["reason"] == "NO_TEXT" and r["level"] is None


def test_no_child_nodes_then_subtopic_reached(model, model_sub, corpus):
    r = model.classify(corpus["t1-0"])
    assert r["level"] == "topic" and r["key"] == "syn-t1" and r["reason"] == "NO_CHILD_NODES"
    rs = [model_sub.classify(corpus[f"sub-{j}"]) for j in range(4)]
    assert sum(x["level"] == "subtopic" and x["key"] == "syn-s1" for x in rs) >= 3
    x = next(x for x in rs if x["level"] == "subtopic")
    assert x["path"] == {"domain": "domain:dom-a", "topic": "syn-t1", "subtopic": "syn-s1"}


def test_determinism(corpus):
    flat = [c for cs in corpus.values() for c in cs]
    a, b = th.Model(ENTRIES + SUB, flat), th.Model(ENTRIES + SUB, flat)
    assert a.signature == b.signature and a.provenance() == b.provenance()
    for k in ("t1-0", "split-0", "g-0", "sub-1"):
        assert a.classify(corpus[k]) == b.classify(corpus[k]) == a.classify(corpus[k])
    assert th.Model(ENTRIES, flat).signature != a.signature


def test_semantic_channel_optional_and_changes_fits(corpus):
    flat = [c for cs in corpus.values() for c in cs]

    def wv(w):
        r = random.Random(w)
        return [r.gauss(0, 1) for _ in range(24)]

    def embed(texts):  # fake deterministic embedding: sum of per-word pseudo-random vectors of content words
        out = []
        for t in texts:
            v = [0.0] * 24
            for w in t.split():
                if w.startswith("w") and w[1:].isdigit():  # filler words carry no meaning for this fake model
                    continue
                v = [a + b for a, b in zip(v, wv(w))]
            out.append(v)
        return out
    plain = th.Model(ENTRIES, flat)
    sem = th.Model(ENTRIES, flat, embed=embed)
    assert sem.signature != plain.signature and sem.nulls[None]["sem"]
    assert plain.nulls[None]["sem"] is None
    c = corpus["t1-0"][0]
    assert plain.fits(c, None) != sem.fits(c, None)
    assert sem.classify(corpus["t2-0"])["path"]["topic"] == "syn-t2"
    assert plain.classify(corpus["t2-0"])["path"]["topic"] == "syn-t2"


def test_summarize_counts(model, corpus):
    rs = [model.classify(corpus[k]) for k in ("t1-0", "t4-0", "g-0", "split-0")] + [model.classify([])]
    s = th.summarize(rs)
    assert s["documents"] == 5 and sum(s["deepest_level"].values()) == 5
    assert s["reasons"].get("NO_TEXT") == 1 and s["any_domain"] == 5 - s["deepest_level"].get("none", 0)
    assert th.summarize([]) == {"documents": 0, "deepest_level": {}, "reasons": {}, "any_domain": 0}


def test_no_real_taxonomy_names_in_module():
    src = (ROOT / "vera" / "taxo_hier.py").read_text().lower()
    reg = json.loads((ROOT / "config" / "micro_topic_anchors.json").read_text())
    for e in reg["entries"]:
        for s in (e["id"], e["label"], e["field"]):
            assert str(s).lower() not in src, s


# --- shadow seam ----------------------------------------------------------------------------------------------------
class _Conn:
    def __init__(self, rows): self.rows = rows
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, q, p=None):
        rows = self.rows

        class R:
            def all(_s): return rows
        return R()


class _Eng:
    def __init__(self, rows): self.rows = rows; self.calls = 0
    def connect(self): self.calls += 1; return _Conn(self.rows)


def _run(monkeypatch, mode, caplog, chunks=None, boom=False):
    rows = [(1, "t", "", None), (2, "t", "", None)]
    ch = {1: ["alpha beta gamma delta"] * 3, 2: ["epsilon zeta eta theta"] * 3} if chunks is None else chunks
    monkeypatch.setattr(cd, "document_chunks", lambda c, ids, s: ch)
    built = []
    real = th.Model

    def spy(*a, **k):
        built.append(1)
        if boom:
            raise RuntimeError("secret detail")
        return real(*a, **k)
    monkeypatch.setattr(th, "Model", spy)
    monkeypatch.setattr(th, "load_config", lambda path=None: {"mode": mode, "params": th.params_from(None)})
    reg = micro_topic.Registry({"entries": ENTRIES})
    eng = _Eng(rows)
    with caplog.at_level(logging.INFO, logger="vera"):
        out = cd.document_primary_topics(eng, reg, {}, 10, 5000, {"chunk_votes": True})
    return out, built, [r for r in caplog.records if "taxo_hier" in r.getMessage()], eng


def test_seam_mode_off_builds_no_model(monkeypatch, caplog):
    out, built, logs, _ = _run(monkeypatch, "off", caplog)
    assert built == [] and logs == []
    assert set(out) == {1, 2}


def test_seam_shadow_same_result_one_aggregate_line(monkeypatch, caplog):
    off, _, _, _ = _run(monkeypatch, "off", caplog)
    caplog.clear()
    on, built, logs, eng = _run(monkeypatch, "shadow", caplog)
    assert on == off and len(built) == 1 and eng.calls == 1  # no extra connection or SQL
    assert len(logs) == 1 and logs[0].levelno == logging.INFO
    assert json.loads(logs[0].getMessage().split("shadow ", 1)[1])["documents"] == 2


def test_seam_shadow_without_chunks_does_nothing(monkeypatch, caplog):
    out, built, logs, _ = _run(monkeypatch, "shadow", caplog, chunks={})
    assert built == [] and logs == []


def test_seam_failure_logs_class_only_and_page_unaffected(monkeypatch, caplog):
    off, _, _, _ = _run(monkeypatch, "off", caplog)
    caplog.clear()
    out, built, logs, _ = _run(monkeypatch, "shadow", caplog, boom=True)
    assert out == off and len(logs) == 1 and "RuntimeError" in logs[0].getMessage() and "secret" not in logs[0].getMessage()


def test_seam_config_error_is_skipped(monkeypatch, caplog):
    def bad(path=None):
        raise th.ConfigError("x")
    monkeypatch.setattr(th, "load_config", bad)
    monkeypatch.setattr(cd, "document_chunks", lambda c, ids, s: {1: ["alpha beta"]})
    reg = micro_topic.Registry({"entries": ENTRIES})
    with caplog.at_level(logging.INFO, logger="vera"):
        out = cd.document_primary_topics(_Eng([(1, "t", "", None)]), reg, {}, 10, 5000, {"chunk_votes": True})
    assert set(out) == {1}
    assert any("ConfigError" in r.getMessage() for r in caplog.records)
