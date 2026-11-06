"""MIDTERMS-WP6-1106 (RC4): route-aware variant seeds. Offline, synthetic questions only (no network, DB or env)."""
import json

from vera import query_variants_acq as qv
from vera import query_variants6

# The cleaned research text of the diagnosed question (the trailing request is WP1's job to split off).
NEWS_Q = "What are implications of the US 2026 Midterm elections?"
NEWS_RAW = NEWS_Q + " Keep searching until you find an answer"
TECH_Q = ("We are building a multi-agent system where a supervisor delegates to worker agents. "
          "What are the best practices for multi-agent orchestration of LLM agents?")
PRACTITIONER_ONLY = ("architecture", "design patterns", "framework", "best practices")
NEWS_SEEDS = ("results", "analysis", "aftermath", "outcome", "reaction")


def _cfg():
    return json.loads(qv.CONFIG_PATH.read_text())


def test_diagnosis_test5_news_question_gets_no_practitioner_terms():
    # Diagnosis test 5: variants for a news-intent question contain no architecture, design patterns, framework.
    assert qv.is_news_route(NEWS_Q)
    out = " | ".join(query_variants6.variants(NEWS_Q)).lower()
    for term in ("architecture", "design patterns", "framework"):
        assert term not in out, term
    for term in PRACTITIONER_ONLY:
        assert term not in out, term


def test_news_question_uses_news_terms_from_config():
    assert tuple(_cfg()["news_terms"]) == NEWS_SEEDS
    out = query_variants6.variants(NEWS_Q)
    for s in NEWS_SEEDS:
        assert any(v.endswith(s) for v in out), s


def test_raw_news_question_still_has_no_practitioner_terms():
    # The tail is not stripped here (WP1). Practitioner seeds must still stay out on the raw text.
    assert qv.is_news_route(NEWS_RAW)
    out = " | ".join(query_variants6.variants(NEWS_RAW)).lower()
    assert all(t not in out for t in ("architecture", "design patterns", "framework"))


def test_technical_question_still_gets_practitioner_terms():
    # Regression: a question without news intent is the technical route and keeps its practitioner seeds.
    assert not qv.is_news_route(TECH_Q)
    out = query_variants6.variants(TECH_Q, tried=["multi-agent supervisor worker"])
    seeds = tuple(_cfg()["practitioner_terms"])
    for s in ("best practices", "patterns", "architecture", "guide", "reference", "documentation",
              "design patterns", "framework"):
        assert s in seeds
        assert any(v.endswith(s) and "multi-agent" in v for v in out), s
    assert not any(v.endswith(n) for v in out for n in NEWS_SEEDS)


def test_load_seeds_routes_by_news_flag(tmp_path):
    p = tmp_path / "acq.json"
    p.write_text(json.dumps({"practitioner_terms": ["pt1"], "news_terms": ["nt1", " nt2 "],
                             "max_terms_per_query": 3}))
    assert qv.load_seeds(p) == (("pt1",), 3)
    assert qv.load_seeds(p, news=True) == (("nt1", "nt2"), 3)


def test_load_seeds_fallback_when_missing(tmp_path):
    seeds, mt = qv.load_seeds(tmp_path / "missing.json", news=True)
    assert seeds == qv.FALLBACK_NEWS_SEEDS and mt == qv.FALLBACK_MAX_TERMS
    seeds, _ = qv.load_seeds(tmp_path / "missing.json")
    assert seeds == qv.FALLBACK_SEEDS
