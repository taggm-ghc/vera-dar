"""NONASK-1103 A6-1: vera.query_variants_acq. Offline, synthetic questions only."""
import json

from vera import query_variants_acq as qv
from vera import search_query

Q = ("We are building a multi-agent system where a supervisor delegates to worker agents. "
     "What are the best practices for multi-agent orchestration of LLM agents?")
FIRST = "multi-agent supervisor delegates worker"  # an already-run first search


def test_deterministic():
    assert qv.variants(Q) == qv.variants(Q) and qv.variants(Q)


def test_never_the_raw_sentence_and_bounded_terms():
    raw = qv.norm(Q)
    out = qv.variants(Q)
    assert all(qv.norm(v) != raw and len(v.split()) <= 20 for v in out)
    assert all("?" not in v and "." not in v for v in out)


def test_pii_removed_before_extraction():
    out = qv.variants(Q + " Contact bob@example.com or +1 555 123 4567 or https://example.invalid/x 123456789")
    blob = " ".join(out).lower()
    assert all(x not in blob for x in ("bob", "example", "555", "123456789", "http"))
    assert out == qv.variants(Q)


def test_dedup_against_tried_and_within():
    out = qv.variants(Q)
    assert len({qv.norm(v) for v in out}) == len(out)
    again = qv.variants(Q, tried=[out[0].upper(), "  " + out[1] + " "])
    assert qv.norm(out[0]) not in {qv.norm(v) for v in again} and qv.norm(out[1]) not in {qv.norm(v) for v in again}
    assert len(again) == len(out) - 2


def test_or_query_normalised_for_dedup():
    assert qv.norm("b OR a") == qv.norm("A or B")


def test_stage_order_drop_then_lexicon_then_entity_then_seeds_then_or():
    cfg = search_query._default_terms_cfg()
    groups = search_query._groups(None)
    q = "Which fighters from mixed martial arts moved into boxing coaching?"
    out = qv.variants(q, terms_cfg=cfg, seeds=("zzseed",))
    seed_i = next(i for i, v in enumerate(out) if v.endswith("zzseed"))
    assert seed_i > 0 and out[-1].count(" OR ") >= 1 or " OR " not in " ".join(out)
    if groups:  # the lexicon stage (if the shipped groups trigger) precedes the seeds
        lex = [i for i, v in enumerate(out) if v in ("ufc", "ultimate fighting championship", "mma")]
        assert not lex or max(lex) < seed_i
    # synthetic ordering check with a known base: drops first (fewer terms than base), OR last
    out = qv.variants("supervisor orchestration workflow delegation", seeds=("zzseed",), groups=())
    assert out[0].split() and len(out[0].split()) < 4
    assert out[-1].startswith("supervisor OR") and " OR " in out[-1]
    assert max(i for i, v in enumerate(out) if len(v.split()) == 3 and "zzseed" not in v) < \
        min(i for i, v in enumerate(out) if v.endswith("zzseed"))


def test_reported_question_gets_practitioner_variants():
    out = qv.variants(Q, tried=[FIRST])
    seeds = json.loads(qv.CONFIG_PATH.read_text())["practitioner_terms"]
    for s in ("best practices", "patterns", "architecture", "guide", "reference", "documentation", "design patterns",
              "framework"):
        assert s in seeds
        assert any(v.endswith(s) and "multi-agent" in v for v in out), s


def test_no_content_terms_gives_no_variants():
    assert qv.variants("What is the?") == [] and qv.variants("") == []


def test_config_has_calibration_note_and_fallback(tmp_path):
    d = json.loads(qv.CONFIG_PATH.read_text())
    assert "UNCALIBRATED" in d["_calibration_note"] and "seed/fallback" in d["_calibration_note"].lower()
    seeds, mt = qv.load_seeds(tmp_path / "missing.json")
    assert seeds == qv.FALLBACK_SEEDS and mt == qv.FALLBACK_MAX_TERMS
