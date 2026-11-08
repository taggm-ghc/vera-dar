"""ROBOTS-TIGHTEN-1108 (OFFLINE): own-family robots groups bind every route; use signals still bind;
__usage_ai-training is informational; Content-Usage ai-use=n equals ai-input=no; UA honesty guard."""
import json
from pathlib import Path

import pytest
import requests

from vera import robots_ai, selective_fetch as sf, source_permissions as sp
from vera.search_providers import USER_AGENT as SEARCH_UA

ROOT = Path(__file__).resolve().parent.parent
U = "https://a.example/p/1"
OWN = ["VERA-DAR", "VERA-research-fetcher", "VERA-research-pipeline", "VERA-DAR-wiki-ref-portal", "VERA-research-prototype"]
UAS = {"VERA-research-fetcher": sf.USER_AGENT, "VERA-DAR-wiki-ref-portal/0.2": "VERA-DAR-wiki-ref-portal/0.2 (x)",
       "VERA-research-prototype": "VERA-research-prototype/0.1 (x)", "VERA-DAR": "VERA-DAR/1.0 (x)"}


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


# ---- S1b
@pytest.mark.parametrize("tok", OWN)
@pytest.mark.parametrize("ua", list(UAS.values()))
def test_any_own_token_group_binds_every_route(tok, ua):
    assert fetch_ok(f"User-agent: *\nAllow: /\nUser-agent: {tok}\nDisallow: /\n", ua) is False


def test_family_all_must_allow():
    t = "User-agent: VERA-research-fetcher\nAllow: /\nUser-agent: VERA-DAR\nDisallow: /\n"
    assert fetch_ok(t) is False
    assert fetch_ok("User-agent: VERA-research-fetcher\nAllow: /\nUser-agent: VERA-DAR\nAllow: /\n"
                    "User-agent: *\nDisallow: /\n") is True


def test_family_path_scoped_and_exact_token():
    t = "User-agent: VERA-DAR\nDisallow: /p\n"
    assert fetch_ok(t) is False and fetch_ok(t, url="https://a.example/q") is True
    assert fetch_ok("User-agent: VERA\nDisallow: /\n") is True  # exact token, no substring (R2-3)


def test_star_still_applies_and_hyperscaler_group_unchanged():
    assert fetch_ok("User-agent: *\nDisallow: /p\n") is False
    assert fetch_ok("User-agent: GPTBot\nDisallow: /\n") is False  # S1a not implemented: still denied


def test_own_tokens_missing_fails_closed(tmp_path):
    c = json.loads((ROOT / "config" / "robots_ai.json").read_text())
    del c["own_tokens"]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        robots_ai.load_config(p)
    assert sorted(robots_ai.load_config()["own_tokens"]) == sorted(t.lower() for t in OWN)


# ---- S2 use signals still bind
GPT_ONLY = "User-agent: GPTBot\nDisallow: /nothing\n"


def test_content_signal_robots_and_header_bind():
    assert fetch_ok("User-agent: *\nAllow: /\nContent-Signal: ai-input=no\n") is True  # access unchanged
    assert sf.robots_no_ai(U) is True
    r = sp.evaluate(U, headers={"Content-Signal": "ai-input=no"})
    assert r.use_reason == "use_no_ai_signal"


def test_tdm_noai_meta_bind():
    assert not sp.evaluate(U, headers={"TDM-Reservation": "1"}).use_for_answer
    assert not sp.evaluate(U, html_head='<meta name="robots" content="noai">').use_for_answer
    assert not sp.evaluate(U, html_head='<meta name="tdm-reservation" content="1">', headers={"TDM-Reservation": "1"}).use_for_answer
    assert sp.evaluate(U).use_for_answer


def test_statement_comment_still_denies():
    assert fetch_ok("# We block all known AI crawlers and assistants.\nUser-agent: *\nAllow: /\n") is False


def test_public_fetch_tdmrep_function_present():
    from vera import public_fetch
    assert callable(public_fetch.tdmrep_reserved)


# ---- A1
def test_usage_token_informational():
    t = "User-agent: *\nAllow: /\nUser-agent: __usage_ai-training\nDisallow: /a\n"
    assert fetch_ok(t, url="https://a.example/a/1") is True
    assert robots_ai.usage_signals(rp_of(t), "https://a.example/a/1") == "ai-train=no"
    assert robots_ai.usage_signals(rp_of(t), "https://a.example/b") == ""
    assert sp.evaluate(U, robots_txt_entries=sp.robots_entries(sf.robots_content_signal("https://a.example/a/1"))).use_for_answer
    assert "ai-train=no" in sf.robots_content_signal("https://a.example/a/1")


def test_usage_no_star_fallback_and_case():
    assert robots_ai.usage_signals(rp_of("User-agent: *\nDisallow: /\n"), U) == ""
    assert robots_ai.usage_signals(rp_of("User-agent: __USAGE_AI-Training/1\nDisallow: /\n"), U) == "ai-train=no"
    t = "User-agent: *\n__x\nUser-agent: __usage_ai-training\nDisallow: /\n"
    assert robots_ai.usage_signals(rp_of(t), U) == "ai-train=no"


def test_usage_tokens_malformed_fails_closed(tmp_path):
    c = json.loads((ROOT / "config" / "robots_ai.json").read_text())
    c["usage_tokens"] = ["x"]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        robots_ai.load_config(p)


# ---- A2
def test_content_usage_header_bars_use_and_ingest():
    r = sp.evaluate(U, headers={"Content-Usage": "ai-use=n"})
    assert not r.use_for_answer and not r.ingest
    assert sp.evaluate(U, headers={"Content-Usage": "train-ai=n, ai-use=n"}).use_for_answer is False


@pytest.mark.parametrize("v", ["ai-use=y", "train-ai=n", "search=n", "", "garbage"])
def test_content_usage_non_bar(v):
    assert sp.evaluate(U, headers={"Content-Usage": v}).use_for_answer


def test_content_usage_unparseable_recorded():
    assert sp.content_usage_signal("garbage") == "info:content-usage-unparsed"
    assert sp.content_usage_signal("ai-use=n") == "ai-input=no"


@pytest.mark.parametrize("grp", ["*", "GPTBot"])
def test_content_usage_robots_any_group(grp):
    t = f"User-agent: *\nAllow: /\nUser-agent: {grp}\nContent-Usage: ai-use=n\n"
    fetch_ok(t)
    assert sf.robots_no_ai(U) is True


def test_content_usage_mixed_with_signal():
    assert not sp.evaluate(U, headers={"Content-Signal": "ai-train=no", "Content-Usage": "ai-use=n"}).use_for_answer


# ---- S4
def test_no_ua_claims_hyperscaler_or_browser():
    cfg = json.loads((ROOT / "config" / "robots_ai.json").read_text())
    bad = [str(t).lower() for t in cfg["ai_agent_tokens"]] + ["mozilla", "compatible;", "chrome", "safari",
                                                              "applewebkit", "googlebot", "bingbot"]
    uas = [sf.USER_AGENT, SEARCH_UA,
           json.loads((ROOT / "config" / "public_fetch.json").read_text())["user_agent"],
           json.loads((ROOT / "config" / "reporting_lane.json").read_text())["user_agent"],
           json.loads((ROOT / "config" / "wiki_ref_portal.json").read_text())["user_agent"],
           json.loads((ROOT / "config" / "vera_eval_run.json").read_text())["search"]["duckduckgo_lite"]["user_agent"]]
    own = {t.lower() for t in robots_ai.load_config()["own_tokens"]}
    for ua in uas:
        assert ua.startswith("VERA") and ua.split("/")[0].lower() in own, ua
        assert not [b for b in bad if b in ua.lower()], ua
    assert "https://github.com/taggm-ghc/vera-dar" in sf.USER_AGENT and "https://github.com/taggm-ghc/vera-dar" in SEARCH_UA


# ---- FINAL-1108: RFC 9309 own-family semantics (owner ruling: deny-all + allow-list VERA admits VERA)
def test_star_disallow_own_family_allow_admits():
    t = "User-agent: *\nDisallow: /\nUser-agent: VERA-DAR\nAllow: /\n"
    assert fetch_ok(t) is True
    assert fetch_ok(t, "VERA-DAR/1.0 (x)") is True


def test_star_allow_own_family_disallow_refuses():
    t = "User-agent: *\nAllow: /\nUser-agent: VERA-DAR\nDisallow: /\n"
    assert fetch_ok(t) is False
    assert fetch_ok(t, "VERA-DAR/1.0 (x)") is False


def test_conflicting_own_family_groups_stricter_wins():
    t = "User-agent: VERA-DAR\nAllow: /\nUser-agent: vera-research-fetcher\nDisallow: /\n"
    assert fetch_ok(t) is False
    assert fetch_ok(t, "VERA-DAR/1.0 (x)") is False


@pytest.mark.parametrize("v", ["ai-use = n", 'ai-use="n"', "AI-USE=N", "ai-use=y, ai-use=n", "/p ai-use=n"])
def test_content_usage_lenient_forms_bar(v):
    assert sp.content_usage_signal(v) == "ai-input=no"


def test_content_usage_raw_robots_lines_bar():
    assert sp.robots_no_ai("User-agent: *\nContent-Usage: ai-use=n\n") is True
    assert sp.robots_no_ai("User-agent: *\nContent-Usage: train-ai=n\n") is False
