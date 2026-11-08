"""QUERY-SHAPE-1108: definitional questions drop the OpenAlex recency filter and get one review call in the slot of the
first variant round; news questions keep recency; variants are topic-neutral and de-duplicated. Real code paths."""
from vera import query_shape, query_variants6, search_providers as sp

SOFT = {"architecture", "design patterns", "best practices", "framework", "documentation", "patterns"}
OA = {"endpoint": "https://example.invalid/works", "min_interval_s": 0.0, "max_response_bytes": 2097152,
      "daily_search_budget": 90, "snippet_max_chars": 100, "from_publication_date": "2023-01-01",
      "to_publication_date": "2026-12-31"}


def _params(shape):
    seen = []

    class R:
        text = '{"results": []}'

    def tx(url, params, headers, timeout):
        seen.append(params)
        return R()
    tok = sp.OPENALEX_SHAPE.set(shape)
    try:
        sp.OpenAlexProvider(OA, transport=tx).call("quarks", 5, 5.0)
    finally:
        sp.OPENALEX_SHAPE.reset(tok)
    return seen[0]


def test_shape_classification():
    assert query_shape.shape("what are quarks?") == {"recency": False, "review": True}
    assert query_shape.shape("Explain photosynthesis")["recency"] is False
    assert query_shape.shape("what is the latest news on quarks?")["recency"] is True
    assert query_shape.shape("Who won the midterm elections?")["recency"] is True
    assert query_shape.shape("what are the current rules for X")["review"] is False


def test_provider_filters():
    base = _params(None)
    assert "from_publication_date:2023-01-01" in base["filter"] and "sort" not in base
    nr = _params({"recency": False, "review": False})
    assert "filter" not in nr
    rv = _params({"recency": False, "review": True})
    assert rv["filter"] == "type:review" and rv["sort"] == "cited_by_count:desc"


def _search(question, rounds):
    calls = []

    def base(q, n, t):
        calls.append((q, dict(sp.OPENALEX_SHAPE.get() or {})))
        return []
    cfg = {"search": {"providers": ["openalex"]}, "bounds": {"max_search_calls_per_job": 7}}
    s = sp.acquisition_search(base, cfg, ddg_search=lambda *a: [])
    s(question, 5, 5.0)
    for v in rounds:
        tok = sp.SCHOLARLY_ONLY.set(True)
        try:
            s(v, 5, 5.0)
        finally:
            sp.SCHOLARLY_ONLY.reset(tok)
    return calls


def test_definitional_no_recency_one_review_in_first_variant_round():
    calls = _search("what are quarks?", ["quarks overview", "quarks review", "quarks introduction"])
    assert [c[1].get("recency") for c in calls] == [False] * 4
    assert [c[1].get("review") for c in calls] == [False, True, False, False]
    assert len(calls) == 4  # no call added beyond round 0 plus the variant rounds


def test_news_keeps_recency_no_review():
    calls = _search("Who won the midterm elections?", ["midterm elections results"])
    assert all(c[1].get("recency") is True and not c[1].get("review") for c in calls)


def test_variants_topic_neutral_and_unique():
    for q in ["what are quarks?", "Who won the midterm elections?", "how do bees make honey"]:
        v = query_variants6.variants(q)
        low = [x.lower() for x in v]
        assert not any(t in x for x in low for t in SOFT), (q, v)
        assert len(set(low)) == len(low)
    v = query_variants6.variants("what are quarks?")
    assert "quarks" not in v and v[0] == "quarks introduction"
    v = query_variants6.variants("what is the best way to design a REST API architecture?")
    assert any("architecture" in x for x in v)  # the question's own term stays
