"""PLAN-FDA CAL-impl: IDF-weighted admission signal (V2), metadata selectors, V3 salient check, POL hook,
excluded_section reason. Fake corpora and synthetic bundles only; no held-out data, no network, no DB."""
import copy

import pytest

from tests.fulldoc_bundle_synth import sec, synth_cfg, write_bundle
from vera import fulldoc_admission as fa
from vera import fulldoc_corpus as fc
from tests.test_fulldoc_admission import ROUTING, doc

WEIGHTED = {"signal": "idf_weighted", "min_relevance": 0.4, "min_matched_stems": 2, "max_spans_total": 6,
            "max_spans_per_doc": 3,
            "idf_signal": {"min_weighted_coverage": 0.60, "salient_k": 2, "min_relevance": 0.30},
            "metadata_selectors": {"enabled": True, "restrict_candidates": True},
            "pointer_filter": {"enabled": False}}


def cfg_w(**adm):
    a = copy.deepcopy(WEIGHTED)
    for k, v in adm.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k].update(v)
        else:
            a[k] = v
    return {"routing": ROUTING, "retrieval": {"candidate_k": 12}, "admission": a,
            "answer": {"attribution_changes_text": "Short excerpt; converted from {format}."}}


class Corp:
    """Duck-typed corpus with evidence chunks (the IDF universe)."""
    def __init__(self, texts, cands, chunks=(), authors=None, ranges=None, excluded=None):
        self.status, self.texts, self.cands, self.chunks = "ok", texts, cands, list(chunks)
        self.acquired = []
        for i in texts:
            d = doc(i)
            if authors and i in authors:
                d["authors"] = authors[i]
            if ranges and i in ranges:
                d["evidence_ranges"] = ranges[i]
            if excluded and i in excluded:
                d["excluded_ranges"] = excluded[i]
            self.acquired.append(d)
        self.unusable = []

    def text(self, i):
        return self.texts[i]

    def retrieve(self, q, k):
        return list(self.cands)

    def evidence_chunk_texts(self):
        return list(self.chunks)

    def inspection(self):
        return {i: {"chunks_inspected": 1} for i in self.texts}


def sp(i, sid, text, full):
    st = full.index(text)
    return {"span_id": sid, "document_id": i, "start": st, "end": st + len(text), "text": text, "score": 1.0}


def one(text, chunks=(), authors=None, **kw):
    full = "xx " + text + " yy"
    return Corp({"d1": full}, [sp("d1", "d1:a", text, full)], chunks, {"d1": authors} if authors else None, **kw), full


# --- V2 IDF -----------------------------------------------------------------------------------------------------------
def test_idf_formula_and_oov_gets_max_idf():
    w = fa.idf_weights({"seen", "rare", "oov"}, {"seen": 100, "rare": 1}, 100)
    import math
    assert w["seen"] == pytest.approx(math.log(101 / 101) + 1) and w["rare"] == pytest.approx(math.log(101 / 2) + 1)
    assert w["oov"] == pytest.approx(math.log(101) + 1) and w["oov"] == max(w.values())  # NOT 0 (idf.get(s, 0) bug)


def test_oov_question_stem_stays_in_denominator_and_blocks_admission():
    # three common stems cover the span; the fourth question stem is unseen in the corpus. Weighted coverage must be
    # (3*1.0)/(3*1.0 + ln(101)+1) ~ 0.35 < 0.60. An implementation using idf.get(s, 0) would give coverage 1.0 and admit.
    text = "forecasting rainfall accuracy improved."
    chunks = ["forecasting rainfall accuracy"] * 100
    c, _ = one(text, chunks)
    q = "forecasting rainfall accuracy zeppelin"
    b = fa.admit(q, c, cfg_w(idf_signal={"salient_k": 0}))
    assert b.admitted == () and [r["reason"] for r in b.rejected] == ["question_coverage"]
    # control: the same question without the OOV word is admitted
    assert fa.admit("forecasting rainfall accuracy", c, cfg_w()).status == "admitted"


def test_weighted_span_relevance_replaces_raw_share():
    # raw share would be 2/4 = 0.5 (>= legacy 0.4) but the two matched stems are the lowest-weight ones
    text = "forecasting rainfall improved."
    chunks = ["forecasting rainfall"] * 50 + ["glacier moraine"] * 2
    c, _ = one(text, chunks)
    q = "forecasting rainfall glacier moraine"
    b = fa.admit(q, c, cfg_w(idf_signal={"min_relevance": 0.6, "min_weighted_coverage": 0.0, "salient_k": 0}))
    assert [r["reason"] for r in b.rejected] == ["low_relevance"]
    b = fa.admit(q, c, cfg_w(idf_signal={"min_relevance": 0.2, "min_weighted_coverage": 0.0, "salient_k": 0}))
    assert b.status == "admitted"


def test_legacy_signal_is_default_when_key_absent():
    text = "forecasting rainfall accuracy improved."
    c, _ = one(text, ["forecasting rainfall accuracy"] * 100)
    cfg = cfg_w()
    del cfg["admission"]["signal"]
    cfg["admission"]["min_question_coverage"] = 0.8
    assert "signal" not in fa.admit("forecasting rainfall accuracy zeppelin", c, cfg).trace
    assert fa.admit("forecasting rainfall accuracy zeppelin", c, cfg).admitted == ()  # 3/4 < 0.8


# --- selectors ----------------------------------------------------------------------------------------------------------
def test_author_surname_selector_removed_from_denominator():
    text = "Forecasting rainfall accuracy improved."
    chunks = ["forecasting rainfall accuracy"] * 20
    c, _ = one(text, chunks, authors="Zorbek et al.")
    q = "Zorbek et al. forecasting rainfall accuracy"
    b = fa.admit(q, c, cfg_w(idf_signal={"salient_k": 0}))
    assert b.status == "admitted" and b.trace["selector_documents"] == ["d1"] and b.trace["et_al"] is True
    # without the selector feature the surname (never in its own text) stays in the denominator
    b2 = fa.admit(q, c, cfg_w(metadata_selectors={"enabled": False}, idf_signal={"salient_k": 0}))
    assert b2.admitted == ()


def test_selector_restricts_candidates_to_named_document():
    t = "Forecasting rainfall accuracy improved."
    full = "xx " + t + " yy"
    c = Corp({"d1": full, "d2": full}, [sp("d1", "d1:a", t, full), sp("d2", "d2:a", t, full)],
             ["forecasting rainfall accuracy"] * 20, authors={"d1": "Zorbek et al.", "d2": "Other et al."})
    b = fa.admit("Zorbek forecasting rainfall accuracy", c, cfg_w(idf_signal={"salient_k": 0}))
    assert [a["document_id"] for a in b.admitted] == ["d1"]
    assert [(r["document_id"], r["reason"]) for r in b.rejected] == [("d2", "selector_mismatch")]


def test_selector_list_authors_and_given_surname_forms():
    icfg, n = ROUTING["identification"], 5
    assert fa._surname_stems("Gostic et al.", icfg, n) == {"gosti"}
    assert fa._surname_stems(["Jane Gostic", "Bob Smith"], icfg, n) == {"gosti", "smith"}
    assert fa._surname_stems("Bracher, Wolffram", icfg, n) >= {"brach", "wolff"}


# --- B2 floor invariant -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("floor", [0, 1, 2])
def test_min_matched_stems_floor_two_on_non_selector_stems(floor):
    text = "Forecasting improved."  # only ONE non-selector question stem is present; the surname is a selector
    c, _ = one(text, ["forecasting"] * 20, authors="Zorbek et al.")
    cfg = cfg_w(min_matched_stems=floor, idf_signal={"min_relevance": 0.0, "min_weighted_coverage": 0.0, "salient_k": 0})
    b = fa.admit("Zorbek forecasting rainfall", c, cfg)
    assert b.admitted == () and [r["reason"] for r in b.rejected] == ["low_relevance"]


def test_selector_stem_never_counts_as_a_hit():
    text = "Zorbek forecasting improved."  # text contains the surname + one other stem: still only 1 non-selector hit
    c, _ = one(text, ["forecasting"] * 20, authors="Zorbek et al.")
    cfg = cfg_w(idf_signal={"min_relevance": 0.0, "min_weighted_coverage": 0.0, "salient_k": 0})
    assert fa.admit("Zorbek forecasting rainfall", c, cfg).admitted == ()


# --- V3 salient ---------------------------------------------------------------------------------------------------------
def test_salient_check_requires_top_idf_stems_in_admitted_spans():
    text = "forecasting rainfall accuracy improved."
    chunks = ["forecasting rainfall accuracy"] * 30 + ["glacier"]
    c, _ = one(text, chunks)
    q = "forecasting rainfall accuracy glacier"  # glacier is the highest-IDF stem and absent from the span
    loose = cfg_w(idf_signal={"min_weighted_coverage": 0.3, "salient_k": 0, "min_relevance": 0.3})
    assert fa.admit(q, c, loose).status == "admitted"
    strict = cfg_w(idf_signal={"min_weighted_coverage": 0.3, "salient_k": 1, "min_relevance": 0.3})
    b = fa.admit(q, c, strict)
    assert b.admitted == () and [r["reason"] for r in b.rejected] == ["salient_stem"]


# --- POL hook -----------------------------------------------------------------------------------------------------------
def test_pointer_only_hook_called_with_span_text_and_cfg(monkeypatch):
    from vera import fulldoc_policy
    calls = []

    def fake(span_text, cfg):
        calls.append((span_text, cfg))
        return True
    monkeypatch.setattr(fulldoc_policy, "pointer_only", fake)
    text = "Forecasting rainfall accuracy improved."
    c, _ = one(text, ["forecasting rainfall accuracy"] * 20)
    cfg = cfg_w(pointer_filter={"enabled": True, "min_content_tokens": 9})
    b = fa.admit("forecasting rainfall accuracy", c, cfg)
    assert calls == [(text, cfg["admission"]["pointer_filter"])] and b.admitted == ()
    assert [r["reason"] for r in b.rejected] == ["pointer_only"]
    cfg2 = cfg_w()  # filter disabled: hook not called, span admitted
    calls.clear()
    assert fa.admit("forecasting rainfall accuracy", c, cfg2).status == "admitted" and calls == []


def test_pointer_only_real_policy_rejects_cross_reference_span():
    text = "Forecasting rainfall accuracy: see Table 2."
    c, _ = one(text, ["forecasting rainfall accuracy"] * 20)
    on = fa.admit("forecasting rainfall accuracy", c, cfg_w(pointer_filter={"enabled": True}))
    assert on.admitted == () and [r["reason"] for r in on.rejected] == ["pointer_only"]
    assert fa.admit("forecasting rainfall accuracy", c, cfg_w()).status == "admitted"


def test_pointer_hook_failure_fails_closed(monkeypatch):
    from vera import fulldoc_policy

    def boom(*a):
        raise RuntimeError("x")
    monkeypatch.setattr(fulldoc_policy, "pointer_only", boom)
    c, _ = one("Forecasting rainfall accuracy improved.", ["forecasting rainfall accuracy"] * 20)
    b = fa.admit("forecasting rainfall accuracy", c, cfg_w(pointer_filter={"enabled": True}))
    assert b.admitted == ()


# --- excluded_section reason --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["legacy", "idf_weighted"])
def test_excluded_section_reason(mode):
    t = "Forecasting rainfall accuracy improved."
    full = "xx " + t + " yy ZZ " + t
    s1, s2 = full.index(t), full.rindex(t)
    cands = [{"span_id": "in", "document_id": "d1", "start": s1, "end": s1 + len(t), "text": t, "score": 1.0},
             {"span_id": "out", "document_id": "d1", "start": s2, "end": s2 + len(t), "text": t, "score": 0.9}]
    c = Corp({"d1": full}, cands, ["forecasting rainfall accuracy"] * 20,
             ranges={"d1": [[0, s2 - 1]]}, excluded={"d1": [{"start": s2 - 1, "end": len(full), "reason": "excluded_section"}]})
    cfg = cfg_w(signal=mode, min_question_coverage=0.8)
    b = fa.admit("forecasting rainfall accuracy", c, cfg)
    assert [a["span_id"] for a in b.admitted] == ["in"]
    assert {r["span_id"]: r["reason"] for r in b.rejected} == {"out": "excluded_section"}
    assert "excluded_section" in fa.REASONS
    c.acquired[0]["excluded_ranges"] = []  # no record => the old generic reason
    assert {r["reason"] for r in fa.admit("forecasting rainfall accuracy", c, cfg).rejected} == {"not_body_section"}


# --- synthetic lexical-absent negatives (deterministic, no LLM) -----------------------------------------------------------
OFF_CORPUS = ["zeppelin", "marzipan", "obsidian", "quokka", "saffron", "tundra", "velcro", "walrus", "xylophone", "yoghurt",
              "zucchini", "abacus", "bagpipe", "cobblestone", "dirigible", "espresso", "flamingo", "gargoyle", "harpsichord",
              "igloo", "jamboree", "kaleidoscope", "lighthouse", "marmalade", "nutmeg", "origami", "pomegranate", "quicksand",
              "rhubarb", "sundial", "trombone", "umbrella", "vermicelli", "windmill", "yardstick", "zigzag", "avocado",
              "buttercup", "cinnamon", "dragonfly"]
FRAMES = ["{a} {b} {c} {d}", "{a} of {b} compared with {c} under {d}"]  # no generic framing words (CAL owns those)


def _bundle_corpus(tmp_path):
    c = fc.load_bundle(write_bundle(tmp_path), synth_cfg())
    assert c.status == "ok"
    return c


def _cfg_prior(corpus_cfg):
    cfg = copy.deepcopy(corpus_cfg)
    cfg["retrieval"] = {"candidate_k": 20, "evidence_only": True}
    cfg["admission"] = {**cfg["admission"], **copy.deepcopy(WEIGHTED), "max_spans_per_doc": 3}
    cfg["admission"]["pointer_filter"] = {"enabled": True}
    return cfg


def test_synthetic_lexical_absent_negatives_rejected_and_positive_controls_admitted(tmp_path):
    from tests import fulltext_synth
    corpus = _bundle_corpus(tmp_path)
    cfg = _cfg_prior(synth_cfg())
    icfg, n = fa._routing(cfg)["identification"], 5
    df, _ = fa._df_table(corpus, corpus.acquired, icfg, n)
    assert all(w[:n] not in df for w in OFF_CORPUS), "off-corpus list must be OOV in the bundle"
    positives = negatives = admitted_pos = admitted_neg = 0
    for topic, words in fulltext_synth.TOPICS.items():
        for i in range(0, 12):
            quad = [words[(i + j * 3) % len(words)] for j in range(4)]
            if len(set(quad)) < 4:
                continue
            frame = FRAMES[i % len(FRAMES)]
            pos = frame.format(a=quad[0], b=quad[1], c=quad[2], d=quad[3])
            positives += 1
            admitted_pos += fa.admit(pos, corpus, cfg).status == "admitted"
            repl = list(quad)
            repl[i % 4] = OFF_CORPUS[(i * 7 + len(topic)) % len(OFF_CORPUS)]  # replace one content stem with an OOV term
            neg = frame.format(a=repl[0], b=repl[1], c=repl[2], d=repl[3])
            negatives += 1
            admitted_neg += fa.admit(neg, corpus, cfg).status == "admitted"
    assert negatives >= 30 and admitted_neg == 0
    assert admitted_pos >= positives // 2  # positive control: the signal is not simply rejecting everything


def test_integration_c1_excluded_candidates_get_excluded_section_reason(tmp_path):
    secs = sec("Code availability", ["transmission incidence outbreak infection QQQ " * 6])
    corpus = fc.load_bundle(write_bundle(tmp_path, body_sections={"syn_epi": secs}), _c1_cfg())
    cfg = _cfg_prior(_c1_cfg())
    cfg["retrieval"]["evidence_only"] = False  # let the excluded chunks reach admission
    b = fa.admit("transmission incidence outbreak infection QQQ", corpus, cfg)
    assert "excluded_section" in {r["reason"] for r in b.rejected}
    assert all("QQQ" not in a["text"] for a in b.admitted)


def _c1_cfg():
    c = synth_cfg()
    c["admission"]["excluded_section_headings"] = ["How to cite", "Code availability", "Supporting information"]
    c["admission"]["excluded_paragraph_labels"] = ["How to cite this article:", "How to cite:"]
    return c
