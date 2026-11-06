"""MIDTERMS-WP1-1106: request_intent split (diagnosis tests 1 and 9)."""
import pytest

from vera.request_intent import split_request

Q = "What are implications of the US 2026 Midterm elections?"


@pytest.mark.parametrize("tail", [" Keep searching until you find an answer.", " Please keep searching until you find an answer!",
                                  " Be thorough.", " Search harder, please."])
def test_tail_split(tail):
    out, fl = split_request(Q + tail)
    assert out == Q
    assert fl["persist"] is (tail.strip().lower().find("thorough") < 0)


def test_trailing_clause():
    out, fl = split_request("What happened in the 2026 midterms, and keep searching until you find an answer")
    assert out == "What happened in the 2026 midterms?" or out.startswith("What happened in the 2026 midterms")
    assert "search" not in out.lower() and fl["persist"]


def test_mentions_unchanged():
    for q in ("How do search engines keep searching for new pages?", "Why do detectives keep searching until they find an answer?", Q):
        assert split_request(q) == (q, {"persist": False})


def test_only_request_unchanged():
    assert split_request("Keep searching until you find an answer.") == ("Keep searching until you find an answer.", {"persist": False})


def test_admission_cleaned_vs_polluted():
    import json
    from pathlib import Path
    from vera import fulldoc_admission as fa
    cfg = json.loads((Path(fa.__file__).resolve().parent.parent / "config" / "fulldoc_answer.json").read_text())
    routing = fa._routing(cfg)
    icfg = routing["identification"]
    n = int(routing["corpus"]["relevance_stem_len"])
    polluted = Q + " Keep searching until you find an answer."
    out, _ = split_request(polluted)
    span = fa._stems("midterm elections", icfg, n)
    cov = lambda t: (lambda q: len(q & span) / max(1, len(q)))(fa._q_stems(t, icfg, n))
    min_cov = float(cfg["admission"]["min_question_coverage"])
    assert cov(out) >= min_cov
    assert cov(polluted) < min_cov
