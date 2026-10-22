"""INT-1022-B: fix2 verifier must-fixes 1 and 2 (offline; shipped config/fulldoc_answer.json admission settings)."""
import copy
import json
from pathlib import Path

from vera import fulldoc_admission as fa

CFG = json.loads((Path(__file__).resolve().parent.parent / "config" / "fulldoc_answer.json").read_text())


def _icfg_n():
    r = fa._routing(CFG)
    return r["identification"], int(r["corpus"]["relevance_stem_len"])


def test_generic_terms_are_dropped_as_whole_tokens_not_by_stem_prefix():
    icfg, n = _icfg_n()
    q = fa._q_stems("How do imports affect commodity prices?", icfg, n)
    assert "imports"[:n] in q and "commodity"[:n] in q
    for word in ("majority", "description", "differential"):
        assert word[:n] in fa._q_stems(f"What is the {word} here?", icfg, n), word


def _doc(i, title):
    return {"document_id": i, "canonical_source_id": f"https://x/{i}", "title": title, "url": f"https://x/{i}",
            "doi": None, "published": "2020", "authors": ["A"], "licence": "CC-BY",
            "licence_url": "https://creativecommons.org/licenses/by/4.0/", "licence_decision": "allow",
            "attribution": "a", "text_sha256": "h", "chars": 100, "evidence_ranges": [[0, 10 ** 9]]}


class _Corpus:
    status = "ok"

    def __init__(self, title, spans):
        self.acquired = [_doc("d1", title)]
        self.spans = spans

    def text(self, i):
        return " ".join(self.spans)

    def retrieve(self, q, k):
        out, pos = [], 0
        for j, t in enumerate(self.spans):
            out.append({"span_id": f"d1:{j}", "document_id": "d1", "start": pos, "end": pos + len(t), "text": t,
                        "score": 1.0})
            pos += len(t) + 1
        return out

    def inspection(self):
        return {}


HORSES = "What are the different types of horses?"
INCIDENTAL = ("Participants were recruited at a riding school where one horse was kept; we measured blood pressure in "
              "office workers over twelve weeks of a diet intervention.")


def _admit(title, spans):
    return fa.admit(HORSES, _Corpus(title, spans), copy.deepcopy(CFG))


def test_one_stem_question_single_incidental_mention_is_not_admitted():
    b = _admit("Blood pressure in office workers", [INCIDENTAL])
    assert len(b.admitted) == 0 and [r["reason"] for r in b.rejected] == ["low_relevance"]


def test_one_stem_question_admitted_with_two_passing_spans_in_the_document():
    spans = ["Horse breeds differ in size: draft horses are heavy and ponies are small breeds of horse.",
             "Each horse breed was bred for work; the horses of riding schools are often warmbloods."]
    b = _admit("A survey", spans)
    assert len(b.admitted) >= 1


def test_one_stem_question_admitted_with_one_span_when_the_stem_is_in_the_title():
    b = _admit("Horse breeds of Europe", ["Draft horses are heavy and ponies are small breeds of horse."])
    assert len(b.admitted) == 1
