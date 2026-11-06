"""F1-QUERY-1106: a bare year is never central or a variant; the arXiv query ANDs the year with topic terms."""
from vera import query_variants_acq as qv, search_providers as sp, search_query as sq

Q = "What are implications of the US 2026 Midterm elections?"


def _cfg():
    return sq._default_terms_cfg()


def test_year_not_central_alone():
    kc, w = sq.key_concepts(Q, _cfg())
    assert w["2026"] < w["midterm"] and w["midterm"] == w["elections"]


def test_cmaes_title_not_relevant_midterm_title_is():
    kc = sq.key_concepts(Q, _cfg())
    bad, ok = sq.relevance_tiers(kc, ["Refined CMA-ES for CEC 2026 competition", "2026 midterm elections: what next"])
    assert bad != 0 and ok == 0


def test_no_bare_year_or_us_variants():
    vs = qv.variants(Q)
    assert vs
    for v in vs:
        assert not sq.is_degenerate_query(v), v
    assert "2026" not in vs and "us" not in vs
    assert any("midterm elections" in v for v in vs)


def test_degenerate():
    assert sq.is_degenerate_query("2026") and sq.is_degenerate_query("us") and sq.is_degenerate_query("U.S. OR 2026")
    assert not sq.is_degenerate_query("us 2026 midterm")


def test_arxiv_query_ands_year_with_topic():
    q = {"min_term_chars": 3, "max_terms": 8, "operator": "OR"}
    body, mode = sp.build_arxiv_query_info(Q, q, _cfg())
    assert "all:2026" in body and " AND " in body
    assert "OR all:2026" not in body and "all:2026 OR" not in body
    assert "all:us" not in body
    assert "all:midterm" in body and "all:elections" in body


def test_name_filter_still_applied():
    assert "midterm" in sq.build_query_terms(Q, _cfg()).extra  # WP2 name filter keeps topic terms
