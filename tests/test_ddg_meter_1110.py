"""DDG-REDUCE-1110 WP0: the per-call-site DuckDuckGo meter, and the CURRENT per-question POST counts pinned on offline
fixtures (plan work/DDG-REDUCE-1110/plan.md, section 1). The provider is the REAL DuckDuckGoLiteProvider with a counting
fake transport; nothing touches the network, a database or an env file. Counts only: the meter never holds a query,
URL, title or question text."""
from types import SimpleNamespace
from urllib.parse import quote

import pytest

import vera.search_providers as sp
from vera import ddg_meter, ddg_reduce, grounded_ask, official_lane
from vera import domain_routing as dr
from vera import request_trace as rt
from vera.eval_config import load_eval_config
from vera.search_providers import DuckDuckGoLiteProvider, RateGate

CFG = load_eval_config().search
DDG_CFG = {**CFG["duckduckgo_lite"], "daily_query_cap": 10_000, "min_interval_s": 0.0}
QT = CFG["query_terms"]

Q_PLAIN = "How reliable are published estimates of developer productivity gains from coding assistants?"
Q_ONE = "How do error correction codes affect hardware reliability estimates?"        # content-identified: photonics
Q_TWO = "How do error correction codes scale in sequencing pipelines for hardware?"   # photonics + genomics
DOMAINS = {Q_ONE: [1], Q_TWO: [1, 2]}
Q_NEWS = "What is the latest news about funding for coding assistants startups?"


def _page(group_hint: str, k: int) -> str:
    host = "reuters.com" if group_hint == "news" else "nature.com"
    rows = "".join(
        f"<tr><td>{i}.</td><td><a rel=\"nofollow\" href=\"//duckduckgo.com/l/?uddg={quote(f'https://{host}/a{k}-{i}', safe='')}"
        f"&amp;rut=x\" class='result-link'>Result {k}-{i}</a></td></tr>" for i in range(1, 4))
    return f"<html><body><table>{rows}</table></body></html>"


class Resp:
    def __init__(self, text):
        self.status_code, self.text, self.headers = 200, text, {}


class Rig:
    """A real DuckDuckGoLiteProvider + a fake scholarly provider, and the POST log of the fake transport."""

    def __init__(self, cfg_over=None):
        self.posts = []
        self.sleeps = []

        def tx(url, data, headers, timeout):
            hint = "news" if "reuters.com" in data["q"] else "portals"
            self.posts.append(data["q"])
            return Resp(_page(hint, len(self.posts)))
        self.ddg = DuckDuckGoLiteProvider({**DDG_CFG, **(cfg_over or {})}, transport=tx, query_terms=QT,
                                          sleep=self.sleeps.append)
        self.scholar_calls = []
        self.scholar_n = 1          # how many candidates the fake scholarly provider returns per call
        rig = self

        class Scholar:
            name, required_env, endpoint, gate = "openalex", (), "fake", RateGate(0.0)

            def call(self, q, n, t):
                rig.scholar_calls.append(q)
                return SimpleNamespace(status_code=200, payload=[f"https://doi.org/10.1/{len(rig.scholar_calls)}-{i}"
                                                                 for i in range(rig.scholar_n)], text="", headers={})

            def parse(self, resp, n):
                return [{"url": u, "title": "T", "snippet": "an abstract", "source_type": "journal_article"}
                        for u in resp.payload][:n]

            def cost_for(self, resp):
                return 0.0, True, "no_published_price_keyless"

            def failure_basis(self):
                return "no_published_price_keyless"
        self.chain = [self.ddg, Scholar()]


class FakeRegistry:
    def __init__(self, *names):
        self.domains = [SimpleNamespace(domain_id=i + 1, name=n, name_norm=n, description="", origin="declared")
                        for i, n in enumerate(names)]

    def list_domains(self):
        return list(self.domains)

    def identify_question(self, question):  # content-derived domains (DARINT P4): need not occur in the question text
        return DOMAINS.get(question, [])


GROUNDING = {"providers": ["duckduckgo_lite", "openalex"], "num_results": 5, "deadline_s": 30, "snippet_max_chars": 1500,
             "provider_timeout_s": 5, "provider_max_retries": 1, "resolve": {"enabled": False}}


def set_flags(monkeypatch, **flags):
    """config/ddg_reduce.json with the given switches; unnamed switches are OFF (the pre-DDG-REDUCE behaviour)."""
    monkeypatch.setattr(ddg_reduce, "load", lambda path=None: dict(flags))


@pytest.fixture(autouse=True)
def baseline_flags(monkeypatch):
    set_flags(monkeypatch)  # every test pins the BASELINE unless it calls set_flags itself


@pytest.fixture
def rig(monkeypatch):
    r = Rig()
    ddg_meter.reset()
    monkeypatch.setattr(grounded_ask, "_provider_chain", lambda names: [p for p in r.chain if p.name in set(names)])
    monkeypatch.setattr(grounded_ask, "_merge_stored", lambda results, *a, **k: results)
    monkeypatch.setattr(dr, "get_cfg", lambda *a, **k: {**dr.load_config(), "enabled": True, "max_domain_branches": 2})
    events = []
    monkeypatch.setattr(rt, "event", lambda stage, code, counts=None, detail=None: events.append((stage, code, counts)))
    r.events = events
    return r


def reading_list(rig, question):
    """The /ask reading-list search (retrieve_all) with domain routing ON and the two fake domains registered."""
    corpus = SimpleNamespace(registry=FakeRegistry("photonics", "genomics", "quantum"), cfg={})
    return grounded_ask.retrieve_all(question, GROUNDING, dar_corpus=corpus)


def posts_by_site():
    return {k: v["posts"] for k, v in ddg_meter.snapshot().items() if k != "total" and v["posts"]}


# ------------------------------------------------------------------------------------- the pinned current counts
def test_base_search_costs_two_posts_one_per_site_group(rig):
    reading_list(rig, Q_PLAIN)
    assert len(rig.posts) == 2
    assert posts_by_site() == {"base": 2}
    assert ddg_meter.snapshot()["total"] == {"calls": 1, "posts": 2, "cache_hits": 0}


@pytest.mark.parametrize("question,expect_posts,expect_sites", [
    (Q_PLAIN, 2, {"base": 2}),
    (Q_ONE, 4, {"base": 2, "domain": 2}),
    (Q_TWO, 6, {"base": 2, "domain": 4}),
    (Q_NEWS, 2, {"base": 2}),
])
def test_current_posts_per_question_reading_list(rig, question, expect_posts, expect_sites):
    reading_list(rig, question)
    assert len(rig.posts) == expect_posts
    assert posts_by_site() == expect_sites
    assert ddg_meter.snapshot()["total"]["posts"] == expect_posts


def test_acquisition_round0_with_the_same_question_is_served_from_the_cache(rig):
    reading_list(rig, Q_PLAIN)
    ddg_fn = lambda q, n, t: rig.ddg.parse(rig.ddg.call(q, n, t), n)  # noqa: E731
    cfg = {"search": {"ddg_first": {"enabled": True}, "europepmc": {"enabled": False}}, "bounds": {"max_search_calls_per_job": 6}}
    search = sp.acquisition_search(lambda q, n, t: [], cfg, ddg_search=ddg_fn)
    with ddg_meter.site("inline"), sp.portal_scope():
        search(Q_PLAIN, 5, 5.0)
    assert len(rig.posts) == 2                              # no new POST
    assert ddg_meter.snapshot()["inline"] == {"calls": 1, "posts": 0, "cache_hits": 2}


def _variant_round(rig, news: bool):
    no_ddg = ddg_reduce.NO_DDG.set(ddg_reduce.on("variant_rounds_no_ddg"))  # exactly what inline_acquire._extra_rounds does
    ddg_fn = lambda q, n, t: rig.ddg.parse(rig.ddg.call(q, n, t), n)  # noqa: E731
    cfg = {"search": {"ddg_first": {"enabled": True}, "europepmc": {"enabled": False}}, "bounds": {"max_search_calls_per_job": 6}}
    search = sp.acquisition_search(lambda q, n, t: [], cfg, ddg_search=ddg_fn)
    so = sp.SCHOLARLY_ONLY.set(not news)         # exactly what inline_acquire._extra_rounds does
    try:
        with ddg_meter.site("variant"), sp.portal_scope():
            search("coding assistants funding rounds 2026 reported", 5, 5.0)
    finally:
        sp.SCHOLARLY_ONLY.reset(so)
        ddg_reduce.NO_DDG.reset(no_ddg)


def test_news_variant_round_costs_two_posts_a_non_news_round_costs_none(rig):
    _variant_round(rig, news=False)
    assert rig.posts == []
    _variant_round(rig, news=True)
    assert len(rig.posts) == 2 and posts_by_site() == {"variant": 2}


def test_official_lane_site_query_is_one_post_labelled_official(rig, monkeypatch):
    import vera.search_and_fetch as sf
    monkeypatch.setattr(sf, "cached_chain", lambda *a, **k: (rig.chain, 0, "fan_out"))
    got = official_lane._ddg_site_search("nist.gov", ["ai", "risk", "framework"])
    assert got and len(rig.posts) == 1 and posts_by_site() == {"official": 1}


def test_trace_event_carries_counts_only(rig):
    reading_list(rig, Q_ONE)
    ev = [e for e in rig.events if e[0] == "search" and e[1].startswith("ddg_")]
    assert [(c, n) for _s, c, n in ev] == [("ddg_base", {"posts": 2, "cache_hits": 0}),
                                           ("ddg_domain", {"posts": 2, "cache_hits": 0})]
    assert all(set(n) == {"posts", "cache_hits"} for _s, _c, n in ev)


def test_labels_are_closed_tokens():
    with ddg_meter.site("Not A Token"):
        assert ddg_meter.current() == "base"
    with ddg_meter.site("domain"):
        assert ddg_meter.current() == "domain"


def test_domain_name_already_in_the_question_adds_no_post_the_branch_query_is_cached(rig):
    """Finding: the DuckDuckGo query is built from the question's core terms, so a domain word that is already a
    core term gives the same query string as the base search and the branch is served from the cache (0 POSTs)."""
    reading_list(rig, "How does quantum error correction affect hardware reliability estimates?")
    assert len(rig.posts) == 2
    assert ddg_meter.snapshot()["domain"] == {"calls": 1, "posts": 0, "cache_hits": 2}


# ------------------------------------------------------------------------------------------------------------- C1
@pytest.mark.parametrize("question,before,after", [(Q_PLAIN, 2, 2), (Q_ONE, 4, 2), (Q_TWO, 6, 2), (Q_NEWS, 2, 2)])
def test_c1_domain_branches_spend_no_duckduckgo(rig, monkeypatch, question, before, after):
    reading_list(rig, question)
    assert len(rig.posts) == before
    ddg_meter.reset(); rig.posts.clear(); rig.ddg._mem.clear()
    set_flags(monkeypatch, domain_branches_scholarly_only=True)
    reading_list(rig, question)
    assert len(rig.posts) == after and posts_by_site() == {"base": 2}


def test_c1_branches_still_search_the_scholarly_provider(rig, monkeypatch):
    set_flags(monkeypatch, domain_branches_scholarly_only=True)
    reading_list(rig, Q_TWO)
    assert len(rig.scholar_calls) == 3      # the base query and one call per branch (2 domains)


def test_c1_shipped_config_defaults_on():
    assert ddg_reduce.CONFIG_PATH.exists() and __import__("json").loads(ddg_reduce.CONFIG_PATH.read_text())["domain_branches_scholarly_only"] is True


# ------------------------------------------------------------------------------------------------------------- C2
def test_c2_news_variant_rounds_spend_no_duckduckgo(rig, monkeypatch):
    set_flags(monkeypatch, variant_rounds_no_ddg=True)
    for _ in range(2):                     # a news question with 2 variant rounds: 4 POSTs before, 0 now
        _variant_round(rig, news=True)
    assert rig.posts == [] and posts_by_site() == {}


def test_c2_switch_off_restores_two_posts_per_news_round(rig, monkeypatch):
    set_flags(monkeypatch, variant_rounds_no_ddg=False)
    for _ in range(2):
        rig.ddg._mem.clear()
        _variant_round(rig, news=True)
    assert len(rig.posts) == 4 and posts_by_site() == {"variant": 4}


def test_c2_extra_rounds_set_no_ddg_for_the_search_call(monkeypatch):
    """The real inline_acquire._extra_rounds loop sets NO_DDG (and the variant label) around deps.search."""
    import inspect
    from vera import inline_acquire
    src = inspect.getsource(inline_acquire._extra_rounds)
    assert 'ddg_reduce.NO_DDG.set(ddg_reduce.on("variant_rounds_no_ddg"))' in src and 'ddg_meter.SITE.set("variant")' in src


# ------------------------------------------------------------------------------------------------------------- C3
@pytest.mark.parametrize("question,group_site", [(Q_PLAIN, "nature.com"), (Q_ONE, "nature.com"), (Q_NEWS, "reuters.com")])
def test_c3_one_group_per_call_chosen_by_intent(rig, monkeypatch, question, group_site):
    set_flags(monkeypatch, one_group_per_call=True)
    reading_list(rig, question)
    base_posts = rig.posts[:1]
    assert len(base_posts) == 1 and group_site in base_posts[0]
    assert ("reuters.com" in base_posts[0]) == (group_site == "reuters.com")   # exactly one of the two groups
    assert posts_by_site()["base"] == 1


def test_c3_off_keeps_both_groups(rig, monkeypatch):
    set_flags(monkeypatch, one_group_per_call=False)
    reading_list(rig, Q_PLAIN)
    assert len(rig.posts) == 2


def test_c3_explicit_queries_are_not_reduced(rig, monkeypatch):
    set_flags(monkeypatch, one_group_per_call=True)
    rig.ddg.call("x", 5, 5.0, queries=[{"group": "official", "query": "a (site:nist.gov)"},
                                       {"group": "official", "query": "b (site:nist.gov)"}])
    assert len(rig.posts) == 2        # a caller's explicit list is its own decision (the official lane sends one)


def test_c3_gate_exempt_uses_the_same_single_query(rig, monkeypatch):
    set_flags(monkeypatch, one_group_per_call=True)
    reading_list(rig, Q_PLAIN)
    assert rig.ddg.gate_exempt(Q_PLAIN)


def test_c1_c2_c3_cumulative_posts_per_question(rig, monkeypatch):
    """The reading-list search with C1 + C3: one POST for every question shape (was 2 / 4 / 6 / 2)."""
    set_flags(monkeypatch, domain_branches_scholarly_only=True, one_group_per_call=True)
    got = {}
    for name, q in (("plain", Q_PLAIN), ("one_domain", Q_ONE), ("two_domains", Q_TWO), ("news", Q_NEWS)):
        rig.posts.clear(); rig.ddg._mem.clear()
        reading_list(rig, q)
        got[name] = len(rig.posts)
    assert got == {"plain": 1, "one_domain": 1, "two_domains": 1, "news": 1}


# ------------------------------------------------------------------------------------------------------------- C4
def _freshness_attempt(rig, monkeypatch, question):
    """ask_service._research_attempt (the OD-1109-SIX search on an answered question) with only the guards stubbed and
    inline_acquire.acquire replaced by a round-0 search through the REAL provider (acquisition_search)."""
    import time
    from vera import acq_discovered, ask_service, fulldoc_corpus, inline_acquire
    monkeypatch.setattr(ask_service, "_layer1_clean", lambda q: True)
    monkeypatch.setattr(ask_service, "inline_need_s", lambda acq, g: 1.0)
    monkeypatch.setattr(ask_service, "_doc_keys", lambda x: [])
    monkeypatch.setattr(fulldoc_corpus, "flag_enabled", lambda c: True)
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda: {})
    monkeypatch.setattr(inline_acquire, "icfg_of", lambda acq: {"in_scope": True})
    monkeypatch.setattr(inline_acquire, "inline_enabled", lambda acq: True)
    monkeypatch.setattr(acq_discovered, "use_discovered", lambda acq: True)
    ddg_fn = lambda q, n, t: rig.ddg.parse(rig.ddg.call(q, n, t), n)  # noqa: E731
    cfg = {"search": {"ddg_first": {"enabled": True}, "europepmc": {"enabled": False}}, "bounds": {"max_search_calls_per_job": 6}}

    def acquire(q, acq, **kw):
        search = sp.acquisition_search(lambda *a: [], cfg, ddg_search=ddg_fn)
        with ddg_meter.site("inline" if ddg_meter.current() == "base" else ddg_meter.current()), sp.portal_scope():
            search(q, 5, 5.0)
        return SimpleNamespace(status="ok", public=lambda: {"status": "ok"})
    monkeypatch.setattr(inline_acquire, "acquire", acquire)
    corpus = SimpleNamespace(registry=FakeRegistry("photonics", "genomics", "quantum"), cfg={})
    grounding = (lambda restricted=False: grounded_ask.retrieve_all(question, GROUNDING, dar_corpus=corpus)[0],)
    return ask_service._research_attempt(question, grounding, {"x": 1}, {}, {}, SimpleNamespace(
        inline_used=False, deadline=time.monotonic() + 100))


@pytest.mark.parametrize("question,before", [(Q_PLAIN, 2), (Q_ONE, 4), (Q_TWO, 6)])
def test_c4_answered_non_news_question_spends_no_duckduckgo(rig, monkeypatch, question, before):
    _freshness_attempt(rig, monkeypatch, question)
    assert len(rig.posts) == before                       # baseline: the discovery search (+ branches); round 0 is cached
    ddg_meter.reset(); rig.posts.clear(); rig.ddg._mem.clear()
    set_flags(monkeypatch, answered_question_no_ddg=True)
    _freshness_attempt(rig, monkeypatch, question)
    assert rig.posts == [] and posts_by_site() == {}


def test_c4_news_question_keeps_duckduckgo(rig, monkeypatch):
    set_flags(monkeypatch, answered_question_no_ddg=True)
    _freshness_attempt(rig, monkeypatch, Q_NEWS)
    assert len(rig.posts) == 2 and posts_by_site() == {"freshness": 2}


def test_c4_scholarly_providers_still_search_on_an_answered_question(rig, monkeypatch):
    set_flags(monkeypatch, answered_question_no_ddg=True)
    _freshness_attempt(rig, monkeypatch, Q_PLAIN)
    assert len(rig.scholar_calls) >= 1


def test_c4_the_flag_is_scoped_to_the_call(rig, monkeypatch):
    set_flags(monkeypatch, answered_question_no_ddg=True)
    _freshness_attempt(rig, monkeypatch, Q_PLAIN)
    assert ddg_reduce.NO_DDG.get() is False and ddg_meter.current() == "base"


# ------------------------------------------------------------------------------------------------------------- C5
class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def _clocked(rig):
    c = Clock()
    rig.ddg._wall = c
    return c


def test_c5_repeat_question_within_six_hours_is_served_from_cache(rig, monkeypatch):
    set_flags(monkeypatch, one_group_per_call=True, longer_normalized_cache=True,
              memory_cache_ttl_s_by_group={"portals": 21600, "news": 3600})
    c = _clocked(rig)
    reading_list(rig, Q_PLAIN)
    c.t += 5 * 3600
    reading_list(rig, Q_PLAIN)
    assert len(rig.posts) == 1
    c.t += 2 * 3600                                         # 7 h after the first call: expired
    reading_list(rig, Q_PLAIN)
    assert len(rig.posts) == 2


def test_c5_news_group_expires_after_one_hour(rig, monkeypatch):
    set_flags(monkeypatch, one_group_per_call=True, longer_normalized_cache=True,
              memory_cache_ttl_s_by_group={"portals": 21600, "news": 3600})
    c = _clocked(rig)
    reading_list(rig, Q_NEWS)
    c.t += 1800
    reading_list(rig, Q_NEWS)
    assert len(rig.posts) == 1
    c.t += 3600
    reading_list(rig, Q_NEWS)
    assert len(rig.posts) == 2


def test_c5_key_is_normalized_case_order_and_repeats(rig, monkeypatch):
    set_flags(monkeypatch, longer_normalized_cache=True, memory_cache_ttl_s_by_group={"portals": 21600})
    a = {"group": "portals", "query": "Coding assistants productivity (site:nature.com)"}
    b = {"group": "portals", "query": "productivity coding CODING assistants (site:nature.com)"}
    assert rig.ddg._mem_key(a) == rig.ddg._mem_key(b)
    assert rig.ddg._mem_key(a) != rig.ddg._mem_key({**a, "group": "news"})


def test_c5_off_keeps_the_exact_key_and_900_s(rig, monkeypatch):
    set_flags(monkeypatch, longer_normalized_cache=False)
    c = _clocked(rig)
    reading_list(rig, Q_PLAIN)
    c.t += 1000
    reading_list(rig, Q_PLAIN)
    assert len(rig.posts) == 4                              # both groups, twice: 900 s expired


def test_c5_cache_stays_bounded_and_holds_url_and_title_only(rig, monkeypatch):
    set_flags(monkeypatch, longer_normalized_cache=True, memory_cache_ttl_s_by_group={"portals": 21600})
    for i in range(300):
        rig.ddg._cache_put({"group": "portals", "query": f"term{i} (site:a.org)"}, [{"url": "https://a.org/x", "title": "T", "snippet": "S"}])
    assert len(rig.ddg._mem) <= 256
    assert all(set(it) == {"url", "title"} for _t, items in rig.ddg._mem.values() for it in items)


# ------------------------------------------------------------------------------------------------------------- C6
C6 = dict(one_group_per_call=True, ddg_fallback_only=True, ddg_fallback_min_results=3)


def test_c6_enough_scholarly_candidates_means_no_duckduckgo(rig, monkeypatch):
    set_flags(monkeypatch, **C6)
    rig.scholar_n = 3
    reading_list(rig, Q_PLAIN)
    assert rig.posts == [] and posts_by_site() == {}
    assert len(rig.scholar_calls) == 1


def test_c6_too_few_scholarly_candidates_falls_back_to_duckduckgo(rig, monkeypatch):
    set_flags(monkeypatch, **C6)
    rig.scholar_n = 2
    reading_list(rig, Q_PLAIN)
    assert len(rig.posts) == 1 and posts_by_site() == {"base": 1}


def test_c6_a_question_with_no_other_route_still_reaches_duckduckgo(rig, monkeypatch):
    set_flags(monkeypatch, **C6)
    monkeypatch.setattr(grounded_ask, "_provider_chain", lambda names: [rig.ddg])
    reading_list(rig, Q_PLAIN)
    assert len(rig.posts) == 1


def test_c6_fan_out_meta_records_duckduckgo_as_not_attempted(rig, monkeypatch):
    import vera.search_and_fetch as sf
    set_flags(monkeypatch, **C6)
    rig.scholar_n = 3
    res = sf.search_question(Q_PLAIN, num_results=5, providers=rig.chain, min_candidates=1, mode="fan_out",
                             deadline_s=30, timeout_s=5, max_retries=1)
    assert {"portal": "duckduckgo_lite", "status": "not_attempted", "n_results": 0} in res.meta["portals"]


def test_c6_seed_requests_never_reach_duckduckgo(rig, monkeypatch):
    from vera import request_origin
    set_flags(monkeypatch, **C6)
    rig.scholar_n = 0
    with request_origin.scope("seed"):
        reading_list(rig, Q_PLAIN)
    assert rig.posts == []


def test_c6_switch_off_runs_duckduckgo_first_and_concurrent_as_before(rig, monkeypatch):
    set_flags(monkeypatch, ddg_fallback_only=False)
    rig.scholar_n = 3
    reading_list(rig, Q_PLAIN)
    assert len(rig.posts) == 2


def _acq_search(rig, base_n):
    ddg_fn = lambda q, n, t: rig.ddg.parse(rig.ddg.call(q, n, t), n)  # noqa: E731
    cfg = {"search": {"ddg_first": {"enabled": True}, "europepmc": {"enabled": False}}, "bounds": {"max_search_calls_per_job": 6}}
    base = lambda q, n, t: [{"provider": "openalex", "doi": f"10.9/{i}", "url": f"https://doi.org/10.9/{i}"} for i in range(base_n)]  # noqa: E731
    with ddg_meter.site("inline"), sp.portal_scope() as recs:
        sp.acquisition_search(base, cfg, ddg_search=ddg_fn)(Q_PLAIN, 5, 5.0)
    return recs


def test_c6_acquisition_search_skips_duckduckgo_when_the_scholarly_routes_suffice(rig, monkeypatch):
    set_flags(monkeypatch, **C6)
    _acq_search(rig, 3)
    assert rig.posts == []


def test_c6_acquisition_search_falls_back_when_they_return_too_little(rig, monkeypatch):
    set_flags(monkeypatch, **C6)
    recs = _acq_search(rig, 1)
    assert len(rig.posts) == 1 and posts_by_site() == {"inline": 1}
    assert ("ddg", "completed") in [(r["portal"], r["status"]) for r in recs]


def test_c6_fixture_mix_measured_posts_per_question(rig, monkeypatch):
    """Per-question DuckDuckGo POSTs on the plan's question mix (plan section 1: 50% no domain, 30% one domain, 10% two,
    10% news), reading-list search only, same fixtures: baseline vs C1+C3 vs C1+C3+C6. The scholarly fixture returns >= 3
    candidates for 3 of every 4 questions (the fallback fires on 25%: a fixture assumption, not live data)."""
    mix = [(Q_PLAIN, .5), (Q_ONE, .3), (Q_TWO, .1), (Q_NEWS, .1)]

    def measure(flags, poor_every=None):
        out = 0.0
        for k, (q, w) in enumerate(mix):
            rig.posts.clear(); rig.ddg._mem.clear()
            set_flags(monkeypatch, **flags)
            rig.scholar_n = 0 if (poor_every is not None and k % poor_every == 0) else 3
            reading_list(rig, q)
            out += w * len(rig.posts)
        return out
    base = measure({})
    c13 = measure(dict(domain_branches_scholarly_only=True, one_group_per_call=True))
    c136 = measure(dict(domain_branches_scholarly_only=True, **C6))
    poor_all = measure(dict(domain_branches_scholarly_only=True, **C6), poor_every=1)   # every question falls back
    assert (round(base, 2), round(c13, 2), round(c136, 2), round(poor_all, 2)) == (3.0, 1.0, 0.0, 1.0)
    # a fallback rate of 25% of questions (one in four falls back) gives 0.25 POST per question
    assert abs(0.25 * poor_all + 0.75 * c136 - 0.25) < 1e-9
