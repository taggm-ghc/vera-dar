"""ROBOTS-POLICY-1108 (OFFLINE): owner dispositions OD-1108-DISPOSITIONS-2.

H13: hyperscaler-only robots groups are advisory (recorded, never deny); *, own-family groups, the AI statement,
     use signals (ai-input / ai-use), TDM and terms still bind; hyperscaler Crawl-delays still pace.
H11: Bing ai-grounding Disallow / DisallowExtended (Bingbot meta or X-Robots-Tag) bars answer-evidence use, like
     ai-input=no. Allow does not. Other Bingbot directives are not read (no over-binding).
Fake inputs only: no network, no DB."""
import json
from pathlib import Path

import pytest
import requests

from vera import public_fetch as pf
from vera import robots_ai, selective_fetch as sf, source_permissions as sp, web_route

ROOT = Path(__file__).resolve().parent.parent
U = "https://a.example/p/1"
HYPER = ["GPTBot", "ChatGPT-User", "ClaudeBot", "Claude-User", "anthropic-ai", "Google-Extended", "CCBot",
         "PerplexityBot", "Perplexity-User", "Applebot-Extended", "Meta-ExternalAgent"]


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


def fetch_ok(txt, ua=None, url=U):
    sf.robots_cache_clear()
    return sf.robots_allowed(url, http_get=lambda u, **k: R(200, txt.encode()), url_check=lambda u: (True, ""),
                             pacer=type("P", (), {"wait": lambda s, u: None})(), user_agent=ua)


def rp_of(txt):
    from urllib import robotparser
    rp = robotparser.RobotFileParser()
    rp.parse(txt.splitlines())
    return robots_ai.annotate(rp, txt)


# ------------------------------------------------------------------------------------------------ H13
@pytest.mark.parametrize("tok", HYPER)
@pytest.mark.parametrize("ua", [None, "VERA-DAR/1.0 (x)", "VERA-DAR-wiki-ref-portal/0.2 (x)",
                                "VERA-research-prototype/0.1 (x)", "VERA-DAR-reference-portal/0.1 (x)"])
def test_h13_hyperscaler_only_group_never_denies_any_route(tok, ua):
    t = f"User-agent: {tok}\nDisallow: /\n"
    assert fetch_ok(t, ua) is True
    assert robots_ai.block_reason(rp_of(t), U) is None
    assert robots_ai.advisory(rp_of(t), U) == (tok,)


def test_h13_all_hyperscalers_blocked_but_star_allows():
    t = "".join(f"User-agent: {x}\n" for x in HYPER) + "Disallow: /\n\nUser-agent: *\nAllow: /\n"
    assert fetch_ok(t) is True
    assert set(robots_ai.advisory(rp_of(t), U)) == set(HYPER)


@pytest.mark.parametrize("t", [
    "User-agent: GPTBot\nDisallow: /\n\nUser-agent: *\nDisallow: /p\n",  # * binds
    "User-agent: GPTBot\nAllow: /\n\nUser-agent: *\nDisallow: /\n",  # a hyperscaler Allow never admits us
    "User-agent: GPTBot\nDisallow: /\n\nUser-agent: VERA-DAR\nDisallow: /\n",  # own family binds
    "User-agent: GPTBot\nUser-agent: VERA-DAR\nDisallow: /\n",  # a shared group naming us binds
    "# We block all known AI crawlers and assistants.\nUser-agent: ClaudeBot\nDisallow: /\n",  # statement
])
def test_h13_star_own_family_and_statement_still_bind(t):
    assert fetch_ok(t) is False


def test_h13_use_signals_still_bind_alongside_hyperscaler_groups():
    t = "User-agent: GPTBot\nDisallow: /\n\nUser-agent: *\nAllow: /\nContent-Signal: ai-input=no\n"
    assert fetch_ok(t) is True and sf.robots_no_ai(U) is True
    t = "User-agent: ClaudeBot\nDisallow: /\nContent-Usage: ai-use=n\n\nUser-agent: *\nAllow: /\n"
    assert fetch_ok(t) is True and sf.robots_no_ai(U) is True  # a use line in ANY group still counts
    rec = sp.evaluate(U, headers={"TDM-Reservation": "1"}, robots_txt_entries=["Content-Signal: search=yes"])
    assert not rec.use_for_answer and rec.use_reason == "use_tdm_reserved"


def test_h13_hyperscaler_crawl_delay_still_paces():
    t = "User-agent: *\nCrawl-delay: 1\n\nUser-agent: ClaudeBot\nDisallow: /\nCrawl-delay: 9\n"
    assert sf.parse_crawl_delay(t) == 9.0
    assert robots_ai.crawl_delay(rp_of(t), sf.USER_AGENT) == 9.0


def test_h13_config_shape_fails_closed(tmp_path):
    c = json.loads((ROOT / "config" / "robots_ai.json").read_text())
    assert "ai_agent_tokens" not in c and "agent" not in c["reasons"]
    assert "H13" in c["_doc"]
    del c["advisory_agent_tokens"]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        robots_ai.load_config(p)


# ------------------------------------------------------------------------------------------------ H11
@pytest.mark.parametrize("v,want", [
    ("ai-grounding:disallow", ["ai-grounding:disallow"]),
    ("ai-grounding:DisallowExtended", ["ai-grounding:disallowextended"]),
    ("AI-Grounding: Disallow", ["ai-grounding:disallow"]),
    ('ai-grounding:"disallow"', ["ai-grounding:disallow"]),
    ("bingbot: ai-grounding:disallow", ["ai-grounding:disallow"]),
    ("noarchive, ai-grounding:allow", ["ai-grounding:allow"]),
    ("noindex", []), ("", []), ("ai-grounding:maybe", []),
])
def test_h11_parse(v, want):
    assert sp.ai_grounding_tokens(v) == want
    assert sp.ai_grounding_disallow(v) is any(w != "ai-grounding:allow" for w in want)


@pytest.mark.parametrize("head,ok", [
    ('<meta name="Bingbot" content="ai-grounding:disallow">', False),
    ('<meta name="bingbot" content="ai-grounding:DisallowExtended">', False),
    ('<meta name="Bingbot" content="ai-grounding:allow">', True),
    ('<meta name="bingbot" content="noai">', True),  # only ai-grounding is read from a Bingbot meta
    ('<meta name="bingbot" content="noindex, nofollow">', True),
])
def test_h11_meta_bars_answer_use_like_ai_input_no(head, ok):
    rec = sp.evaluate(U, html_head=head)
    assert rec.use_for_answer is ok
    if not ok:
        assert rec.use_reason == sp.evaluate(U, headers={"Content-Signal": "ai-input=no"}).use_reason \
            == "use_no_ai_signal"
        assert rec.ingest_reason == "ingest_not_usable"


@pytest.mark.parametrize("xrt,ok", [
    ("ai-grounding:disallow", False), ("bingbot: ai-grounding:DisallowExtended", False),
    ("ai-grounding: allow", True), ("noarchive", True),
])
def test_h11_x_robots_tag_header(xrt, ok):
    assert sp.evaluate(U, headers={"X-Robots-Tag": xrt}).use_for_answer is ok


def test_h11_public_fetch_to_web_route_path():
    html = ('<html><head><meta name="Bingbot" content="ai-grounding:disallow">'
            '<meta name="bingbot" content="noindex"></head><body>x</body></html>')
    mr = pf.signal_value(", ".join(pf._meta_robots(html)))
    assert mr == "ai-grounding:disallow"  # the bingbot noindex is not carried
    post = web_route.evaluate_fetched(U, {"text": "x", "meta_robots": mr})
    assert not post.use_for_answer and post.use_reason == "use_no_ai_signal"
    post = web_route.evaluate_fetched(U, {"text": "x", "x_robots_tag": "bingbot: ai-grounding:disallowextended"})
    assert not post.use_for_answer
    assert web_route.evaluate_fetched(U, {"text": "x", "meta_robots": "ai-grounding:allow"}).use_for_answer

