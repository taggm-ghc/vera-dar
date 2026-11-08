"""OD-1106-ROBOTS-AI, as amended by OD-1108-DISPOSITIONS-2 H13 (OFFLINE): AI statements block VERA-DAR;
hyperscaler-only robots groups are advisory (recorded by robots_ai.advisory, never deny); fails closed."""
import pytest
import requests

from vera import robots_ai, selective_fetch as sf

LSE = "# We block all known AI crawlers and assistants.\nUser-agent: ClaudeBot\nDisallow: /\n"
GPT = "User-agent: GPTBot\nDisallow: /x\n"


class R:
    def __init__(self, status, body=b""):
        self.status_code, self.headers, self._b = status, {}, body

    def iter_content(self, n):
        yield self._b

    def close(self):
        pass


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    sf.robots_cache_clear()
    yield
    sf.robots_cache_clear()


def allowed(txt, url, status=200):
    sf.robots_cache_clear()
    body = txt.encode() if isinstance(txt, str) else txt
    ok = sf.robots_allowed(url, http_get=lambda u, **k: R(status, body), url_check=lambda u: (True, ""),
                           pacer=type("P", (), {"wait": lambda s, u: None})())
    return ok, sf.robots_ai_reason(url)


def test_lse_statement_and_group_block_host():
    ok, why = allowed(LSE, "https://a.example/any")
    assert not ok and why == "robots_ai_statement"


def test_agent_group_is_advisory_and_path_scoped():  # OD-1108 H13: hyperscaler-only groups never deny
    assert allowed(GPT, "https://a.example/x/1") == (True, None)
    assert allowed(GPT, "https://a.example/y") == (True, None)
    rp = robots_ai.annotate(_parsed(GPT), GPT)
    assert robots_ai.advisory(rp, "https://a.example/x/1") == ("GPTBot",)
    assert robots_ai.advisory(rp, "https://a.example/y") == ()


def test_group_without_statement_is_advisory_only():  # H13: was robots_ai_agent_block
    t = "User-agent: ClaudeBot\nDisallow: /\n"
    assert allowed(t, "https://a.example/p") == (True, None)
    assert robots_ai.advisory(robots_ai.annotate(_parsed(t), t), "https://a.example/p") == ("ClaudeBot",)
    # the * group and the AI statement still bind alongside a hyperscaler group
    assert allowed(t + "\nUser-agent: *\nDisallow: /p\n", "https://a.example/p")[0] is False
    assert allowed("# AI crawlers are blocked\n" + t, "https://a.example/q") == (False, "robots_ai_statement")


def test_generic_only_unchanged():
    assert allowed("User-agent: *\nDisallow: /priv\n", "https://a.example/ok")[0] is True
    assert allowed("User-agent: *\nDisallow: /priv\n", "https://a.example/priv/1")[0] is False
    assert allowed("", "https://a.example/ok", status=404)[0] is True


def test_malformed_fails_closed(monkeypatch):
    assert allowed("x", "https://a.example/", status=503)[0] is False
    monkeypatch.setattr(robots_ai, "CONFIG_PATH", robots_ai.CONFIG_PATH.with_name("missing.json"))
    monkeypatch.setattr(robots_ai, "_CFG", None)
    assert allowed(GPT, "https://a.example/y")[0] is False


def test_crawl_delay_largest_group():
    t = "User-agent: *\nCrawl-delay: 2\nUser-agent: GPTBot\nCrawl-delay: 7\n"
    assert sf.parse_crawl_delay(t) == 7.0
    assert sf.parse_crawl_delay("User-agent: *\nCrawl-delay: 2\n") == 2.0


def test_shared_crawl_delay_includes_ai_groups_and_arxiv_path_blocks():
    from urllib import robotparser
    t = "User-agent: *\nCrawl-delay: 2\nUser-agent: ClaudeBot\nCrawl-delay: 9\n"
    rp = robotparser.RobotFileParser()
    rp.parse(t.splitlines())
    robots_ai.annotate(rp, t)
    assert robots_ai.crawl_delay(rp, sf.USER_AGENT) == 9.0
    assert robots_ai.crawl_delay(True, sf.USER_AGENT) is None
    from vera import acq_discovered
    import time
    acq_discovered.reset_for_tests()
    acq_discovered._ROBOTS["a.example"] = (time.monotonic(), robots_ai.annotate(_parsed(LSE), LSE))
    assert acq_discovered._robots_allowed("https://a.example/abs/1", {"a.example"}, None, {}, time.monotonic, None) is False
    acq_discovered.reset_for_tests()


def _parsed(t):
    from urllib import robotparser
    rp = robotparser.RobotFileParser()
    rp.parse(t.splitlines())
    return rp


def _br(t, url="https://a/x"):
    return robots_ai.block_reason(robots_ai.annotate(_parsed(t), t), url)


# H13: the RFC 9309 matcher cases, formerly run through a GPTBot group (now advisory), re-pointed at an own-family
# group (User-agent: VERA-DAR), which binds every VERA+ route. want = True means allowed.
@pytest.mark.parametrize("t,url,want", [
    ("User-agent: VERA-DAR\nDisallow: /*\n", "https://a/x", False),
    ("User-agent: VERA-DAR\nDisallow: /*.pdf$\n", "https://a/f.pdf", False),
    ("User-agent: VERA-DAR\nDisallow: /*.pdf$\n", "https://a/f.pdf?x=1", True),
    ("User-agent: VERA-DAR\nDisallow: /docs*/private\n", "https://a/docs1/private", False),
    ("User-agent: VERA-DAR\nAllow: /\nDisallow: /p\n", "https://a/p/1", False),
    ("User-agent: VERA-DAR\nDisallow: /\nAllow: /pub\n", "https://a/pub/1", True),
    ("User-agent: VERA-DAR\nDisallow: /a\n\nUser-agent: VERA-DAR\nDisallow: /b\n", "https://a/b", False),
    ("User-agent: VERA-DAR/1.1\nDisallow: /\n", "https://a/x", False),
    ("User-agent: vera-dar\nDisallow: /\n", "https://a/x", False),
    ("User-agent: *\nUser-agent: VERA-DAR\nDisallow: /p\n", "https://a/p/1", False),
    ("User-agent: VERA-DAR\nDisallow:\n", "https://a/x", True),
    ("User-agent: VERA-DAR\nDisallow: /a%2Fb\n", "https://a/a/b", False),
    ("User-agent: VERA-DAR\nAllow: /x$\nDisallow: /x\n", "https://a/x", True),  # tie-free: longer Allow? equal length -> Allow
])
def test_rfc9309_own_family_groups(t, url, want):
    rp = robots_ai.annotate(_parsed(t), t)
    assert robots_ai.allowed(rp, sf.USER_AGENT, url) is want
    assert robots_ai.block_reason(rp, url) is None  # no statement: never an AI block reason
    g = t.replace("VERA-DAR", "GPTBot").replace("vera-dar", "gptbot")  # same file for a hyperscaler: advisory only
    rpg = robots_ai.annotate(_parsed(g), g)
    assert robots_ai.block_reason(rpg, url) is None
    assert bool(robots_ai.advisory(rpg, url)) is (not want)
    if "User-agent: *" not in g:
        assert robots_ai.allowed(rpg, sf.USER_AGENT, url) is True


def test_rfc9309_own_and_star_groups():
    ok = lambda t, u: robots_ai.allowed(robots_ai.annotate(_parsed(t), t), sf.USER_AGENT, u)
    assert not ok("User-agent: *\nDisallow: /*.pdf$\n", "https://a/f.pdf")
    assert not ok("User-agent: *\nAllow: /\nDisallow: /p\n", "https://a/p/1")
    assert ok("User-agent: *\nDisallow: /\nAllow: /pub\n", "https://a/pub/1")
    assert not ok("User-agent: vera-research-fetcher\nDisallow: /docs*/x\n", "https://a/docs9/x")


@pytest.mark.parametrize("t", [
    "# There are no AI crawlers restrictions on this site\nUser-agent: *\nAllow: /\n",
    "# We do not block all AI crawlers\n",
    "Sitemap: https://a/s.xml#block-all-ai-crawlers\n",
    "User-agent: *\nDisallow: /x # block all ai crawlers\n",
    "# we block all\n# ai crawlers\n",
])
def test_statement_false_positives(t):
    assert _br(t) is None


@pytest.mark.parametrize("t", ["# AI bots are blocked\n", "# Blocking AI crawlers\n", "# AI scrapers are not welcome\n",
                               "  # Block all known AI crawlers and assistants\n"])
def test_statement_matches_full_line_comments(t):
    assert _br(t) == "robots_ai_statement"


def test_r2_1_adversarial_wildcards_linear_time():
    """R2-1: a 32-star adversarial pattern (backtracking blow-up under the old regex) finishes in under 50 ms."""
    import time
    pat = "/" + "*a" * 32 + "b"
    t = f"User-agent: *\nDisallow: {pat}\n"
    rp = robots_ai.annotate(_parsed(t), t)
    url = "https://a/" + "a" * 4000
    t0 = time.perf_counter()
    for _ in range(5):
        assert robots_ai.allowed(rp, sf.USER_AGENT, url)
    assert time.perf_counter() - t0 < 0.05
    assert not robots_ai.allowed(rp, sf.USER_AGENT, url + "b")


@pytest.mark.parametrize("pat", ["/" + "*x" * 33, "/" + "x" * 600])
def test_r2_1_over_cap_disallow_blocks_allow_ignored(pat):
    ok = lambda t, u: robots_ai.allowed(robots_ai.annotate(_parsed(t), t), sf.USER_AGENT, u)
    assert not ok(f"User-agent: *\nDisallow: {pat}\n", "https://a/unrelated")
    assert not ok(f"User-agent: *\nDisallow: /\nAllow: {pat}\n", "https://a/" + pat)


@pytest.mark.parametrize("pat,path,want", [
    ("/a*b*c", "/aXbYc", True), ("/a*b*c", "/acb", False), ("/*.pdf$", "/x.pdf", True), ("/*.pdf$", "/x.pdfx", False),
    ("/a*a$", "/a", False), ("/a*a$", "/aa", True), ("/x$", "/x", True), ("/x$", "/x/", False), ("*", "/", True),
    ("/a**b", "/ab", True), ("/p", "/q/p", False),
])
def test_r2_1_matcher_semantics(pat, path, want):
    assert robots_ai._match(pat, path) is want


@pytest.mark.parametrize("t", ["﻿User-agent: *\nDisallow: /\n", "﻿User-agent: GPTBot\nDisallow: /\n"])
def test_r2_2_bom_is_stripped(t):
    rp = robots_ai.annotate(_parsed(t), t)
    blocked = not robots_ai.allowed(rp, sf.USER_AGENT, "https://a/x") or robots_ai.block_reason(rp, "https://a/x")
    # the BOM must not hide the first group: * binds; a GPTBot group is advisory (H13) but must still be seen
    assert blocked or robots_ai.advisory(rp, "https://a/x") == ("GPTBot",)
    assert robots_ai.parse_groups(t)[0]["agents"] in (["*"], ["gptbot"])


def test_r2_2_bom_crawl_delay_and_fetch_path():
    t = "﻿User-agent: *\nCrawl-delay: 7\nDisallow: /\n"
    assert robots_ai.crawl_delay(robots_ai.annotate(_parsed(t), t), sf.USER_AGENT) == 7.0
    ok, _ = allowed(t.encode("utf-8"), "https://a/x")
    assert not ok


def test_r2_3_own_ua_exact_product_token():
    ok = lambda t, u: robots_ai.allowed(robots_ai.annotate(_parsed(t), t), sf.USER_AGENT, u)
    assert not ok("User-agent: a\nAllow: /\n\nUser-agent: *\nDisallow: /\n", "https://a/x")
    assert not ok("User-agent: VERA\nAllow: /\n\nUser-agent: *\nDisallow: /\n", "https://a/x")
    assert ok("User-agent: VERA-Research-Fetcher/9\nAllow: /\n\nUser-agent: *\nDisallow: /\n", "https://a/x")
