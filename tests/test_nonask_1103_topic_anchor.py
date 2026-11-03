"""NONASK-1103 WP-G1: topic-anchor precision rule (flag off by default). Offline; fake corpus."""
import copy
import json
import pathlib

from vera.fulldoc_admission import admit, topic_anchor_stems

ROUTING = {"identification": {"token_pattern": r"[a-z0-9]+", "min_token_len": 3, "stopwords": ["the", "and", "for"]},
           "corpus": {"relevance_stem_len": 5}}
BASE = {"routing": ROUTING, "retrieval": {"candidate_k": 12},
        "admission": {"min_relevance": 0.5, "min_matched_stems": 2, "max_spans_total": 6, "max_spans_per_doc": 6,
                      "min_question_coverage": 0.5},
        "answer": {"attribution_changes_text": "x"}}
Q = "multi agent orchestration"
ON_TOPIC = "Multi agent orchestration patterns coordinate agents."
MARL = "Multi agent reinforcement learning trains many agents."
FILLER = "\n".join(["multi agent systems and agent studies"] * 6)


def doc(i):
    return {"document_id": i, "canonical_source_id": f"https://x/{i}", "title": i, "url": f"https://x/{i}",
            "doi": None, "published": "2020", "authors": ["A"], "licence": "CC-BY",
            "licence_url": "https://creativecommons.org/licenses/by/4.0/", "licence_decision": "allow",
            "attribution": "a", "text_sha256": "h" + i, "chars": 100, "evidence_ranges": [[0, 10 ** 9]]}


class Fake:
    status = "ok"

    def __init__(self):
        self.texts = {"d1": ON_TOPIC + "\n" + FILLER, "d2": MARL + "\n" + FILLER}
        self.acquired = [doc("d1"), doc("d2")]
        self.unusable = []

    def text(self, i):
        return self.texts[i]

    def retrieve(self, q, k):
        return [{"span_id": "s1", "document_id": "d1", "start": 0, "end": len(ON_TOPIC), "text": ON_TOPIC, "score": 1.0},
                {"span_id": "s2", "document_id": "d2", "start": 0, "end": len(MARL), "text": MARL, "score": 0.9}]

    def inspection(self):
        return {}


def cfg(**ta):
    c = copy.deepcopy(BASE)
    if ta is not None and ta != {"_absent": True}:
        c["admission"]["topic_anchor"] = ta
    return c


def test_flag_off_identity_absent_vs_disabled_vs_shipped():
    b0 = admit(Q, Fake(), BASE)
    b1 = admit(Q, Fake(), cfg(enabled=False, max_df_ratio=0.2))
    assert b0 == b1 and b0.trace == b1.trace
    assert {a["document_id"] for a in b0.admitted} == {"d1", "d2"}  # current behaviour: the MARL span passes


def test_shipped_config_flag_is_off():
    p = pathlib.Path(__file__).resolve().parent.parent / "config" / "fulldoc_answer.json"
    ta = json.loads(p.read_text())["admission"]["topic_anchor"]
    assert ta["enabled"] is False and "_calibration_note" in ta and "feedback_threshold_calibration_rule" in ta["_calibration_note"]


def test_flag_on_drops_offtopic_keeps_ontopic():
    b = admit(Q, Fake(), cfg(enabled=True, max_df_ratio=0.2))
    assert [a["document_id"] for a in b.admitted] == ["d1"]
    assert any(r["document_id"] == "d2" and r["reason"] == "low_relevance" for r in b.rejected)


def test_flag_on_only_removes():
    off = admit(Q, Fake(), BASE)
    on = admit(Q, Fake(), cfg(enabled=True, max_df_ratio=0.2))
    assert {a["span_id"] for a in on.admitted} <= {a["span_id"] for a in off.admitted}


def test_flag_on_without_threshold_or_anchor_is_inactive():
    assert admit(Q, Fake(), cfg(enabled=True)) == admit(Q, Fake(), BASE)
    assert admit(Q, Fake(), cfg(enabled=True, max_df_ratio=-1.0)).trace["admitted"] == admit(Q, Fake(), BASE).trace["admitted"]


def test_anchor_function_df_derived():
    df = {"multi": 9, "agent": 10, "orche": 1}
    assert topic_anchor_stems({"multi", "agent", "orche", "zzzzz"}, df, 10, 0.2) == {"orche", "zzzzz"}
    assert topic_anchor_stems({"a"}, {}, 0, 0.2) == set()
