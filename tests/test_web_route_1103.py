"""NONASK-1103 I-A6 D5: A6-W web route, offline. Fetch is a recording stub; no socket, DB or env file."""
import json
import logging

import pytest

from tests import leak_sentinel as L
from vera import public_fetch as pf
from vera import source_permissions as sp
from vera import web_route as wr

REG = [
    {"entity": "AWS", "domains": ["aws.amazon.com", "docs.aws.amazon.com"], "state": "active",
     "access_verdict": "forbid", "derivative_verdict": "allow", "ingest_verdict": "forbid"},
    {"entity": "Nvidia", "domains": ["nvidia.com"], "state": "active", "access_verdict": "forbid",
     "derivative_verdict": "forbid", "ingest_verdict": "forbid"},
    {"entity": "Microsoft Learn", "domains": ["learn.microsoft.com"], "state": "active", "access_verdict": "forbid",
     "derivative_verdict": "forbid", "ingest_verdict": "forbid"},
]


def cand(url, title="A page", **kw):
    return {"url": url, "title": title, "snippet": "about things", "doi": None, "acq_source": "ddg", **kw}


class Fetch:
    def __init__(self, pages=None):
        self.calls, self.pages = [], pages or {}

    def __call__(self, url):
        self.calls.append(url)
        r = self.pages.get(url, {"text": "Readable body text about the topic."})
        if isinstance(r, Exception):
            raise r
        return {"url": url, "content_type": "text/html", "headers": {}, "meta_robots": [], **r}


def run(cands, f=None, **kw):
    f = f or Fetch()
    kw.setdefault("registry", REG)
    kw.setdefault("deny_list", [])
    return wr.read_web("q", cands, fetch=f, **kw), f


@pytest.mark.parametrize("url", ["https://docs.aws.amazon.com/x", "https://developer.nvidia.com/y",
                                 "https://learn.microsoft.com/en-us/azure/z"])
def test_registry_forbidden_hosts_never_fetched(url):
    r, f = run([cand(url)])
    assert f.calls == [] and r.pages == [] and len(r.listed) == 1
    assert r.listed[0]["phrase"].startswith("not read") and r.dev[0]["gate"]["read"] == "read_registry_forbid"
    assert r.listed[0]["reference"] and r.listed[0]["title"] == "A page"


def test_deny_list_host_never_fetched():
    r, f = run([cand("https://sub.blocked.example/p")], deny_list=["blocked.example"])
    assert f.calls == [] and r.pages == [] and r.dev[0]["gate"]["read"] == "read_deny_list"


def test_robots_disallow_no_text():
    r, f = run([cand("https://a.example/p")], Fetch({"https://a.example/p": pf.FetchRefused("robots")}))
    assert r.pages == [] and r.dev[0]["gate"]["read"] == "read_robots_disallow"
    assert r.listed[0]["phrase"].startswith("not read")


def test_other_fetch_refusal_listed():
    r, _ = run([cand("https://a.example/p")], Fetch({"https://a.example/p": pf.FetchRefused("status")}))
    assert r.pages == [] and r.dev[0]["gate"]["read"] == "fetch_status"


@pytest.mark.parametrize("body", ["Please sign in to continue to the article.", "Subscribe to read the rest."])
def test_login_paywall_listed_only(body):
    r, _ = run([cand("https://a.example/p")], Fetch({"https://a.example/p": {"text": body}}))
    assert r.pages == [] and r.dev[0]["gate"]["read"] == "read_login_paywall"
    assert "page requires" in r.listed[0]["phrase"]


@pytest.mark.parametrize("extra,code", [
    ({"meta_robots": ["noai, noimageai"]}, "use_no_ai_signal"),
    ({"meta_robots": ["index,noai"]}, "use_no_ai_signal"),
    ({"headers": {"x-robots-tag": "noai"}}, "use_no_ai_signal"),
    ({"headers": {"content-signal": "search=yes, ai-input=no"}}, "use_no_ai_signal"),
])
def test_no_ai_signals_contribute_no_text(extra, code):
    r, _ = run([cand("https://a.example/p")], Fetch({"https://a.example/p": extra}))
    assert r.pages == [] and r.dev[0]["gate"]["use"] == code and "does not permit reuse" in r.listed[0]["phrase"]


def test_nd_registry_derivative_forbid_after_redirect_lists_only():
    reg = REG + [{"entity": "X", "domains": ["nd.example"], "state": "active", "derivative_verdict": "forbid"}]
    r, _ = run([cand("https://a.example/p")], Fetch({"https://a.example/p": {"url": "https://nd.example/p"}}),
               registry=reg)
    assert r.pages == [] and r.dev[0]["gate"]["use"] == "use_registry_forbid"


def test_redirect_to_read_forbidden_host_discards_text():
    r, _ = run([cand("https://a.example/p")], Fetch({"https://a.example/p": {"url": "https://docs.aws.amazon.com/q"}}))
    assert r.pages == [] and r.dev[0]["gate"]["read"] == "read_registry_forbid"


def test_unknown_host_readable_page_fits_lane_contract():
    r, f = run([cand("https://blog.example/post", "Post title")])
    assert f.calls == ["https://blog.example/post"] and len(r.pages) == 1
    p = r.pages[0]
    assert (p.lane, p.title, p.url) == ("web", "Post title", "https://blog.example/post")
    assert p.answer_only is True and p.stored is False and p.epistemic == "web" and p.label == "web source"
    assert p.text and p.text not in repr(p) and r.dev[0]["ingest"] is False
    assert r.dev[0]["gate"]["ingest"] == "ingest_answer_only_wp"


def test_one_fetch_per_host_per_call():
    r, f = run([cand("https://a.example/1"), cand("https://a.example/2"), cand("https://b.example/1")])
    assert f.calls == ["https://a.example/1", "https://b.example/1"]
    assert len(r.pages) == 2 and len(r.listed) == 1 and r.dev[1]["gate"]["read"] == "read_host_round_cap"


def test_identifier_candidates_ignored():
    cs = [cand("https://doi.org/10.1/x", doi="10.1/x"), cand("https://x.example/a", doi="10.2/y"),
          cand("https://x.example/b", pmcid="PMC1"), cand("https://arxiv.org/abs/1"), cand("ftp://x.example/c"),
          cand("https://x.example/d", arxiv_id="1.2")]
    r, f = run(cs)
    assert f.calls == [] and r.dev == [] and r.pages == [] and r.listed == []


def test_public_phrases_carry_no_host():
    cs = [cand("https://docs.aws.amazon.com/x"), cand("https://a.example/p"), cand("https://a.example/q"),
          cand("https://b.example/p")]
    r, _ = run(cs, Fetch({"https://b.example/p": {"meta_robots": ["noai"]}}))
    for e in r.listed:
        assert e["phrase"] and not any(h in e["phrase"] for h in ("aws", "amazon", "a.example", "b.example", "http"))
    cfg = json.loads(wr.CONFIG_PATH.read_text())
    assert cfg["disclosure"] == "public web pages may be read to answer, and are stored only when their licence allows"


def test_quotes_and_characterisation_bounded():
    r, _ = run([cand("https://docs.aws.amazon.com/x", snippet="s" * 900)])
    assert len(r.listed[0]["characterisation"]) <= 200 + 60


def test_never_stored_sentinel():
    s = L.new()
    bad = cand("https://docs.aws.amazon.com/x")
    pages = {"https://a.example/p": {"text": f"Body {s} body"},
             "https://b.example/p": {"text": f"Body {s}", "meta_robots": ["noai"]},
             "https://c.example/p": {"text": f"Please sign in to continue {s}"},
             "https://d.example/p": pf.FetchRefused("robots")}
    cs = [cand(u) for u in pages] + [bad]
    with L.capture_all() as cap:
        logging.getLogger("x").info("run")
        r, f = run(cs, Fetch(pages))
        _ = repr(r) + repr(r.pages) + str(r.pages[0])
    assert len(r.pages) == 1 and r.pages[0].text.count(s) == 1
    L.assert_no_leak(s, logs=cap, listed=r.listed, dev=r.dev, result_repr=repr(r), page_repr=repr(r.pages[0]))


def test_route_imports_no_store_or_queue():
    src = open(wr.__file__, encoding="utf-8").read()
    for bad in ("enqueue", "run_job", "dar_corpus", "psycopg", "sqlite", "open(", "logging", "print("):
        assert bad not in src, bad


def test_fetch_error_is_contained():
    r, _ = run([cand("https://a.example/p"), cand("https://b.example/p")],
               Fetch({"https://a.example/p": RuntimeError("boom")}))
    assert len(r.pages) == 1 and r.dev[0]["gate"]["error"] == "RuntimeError"


def test_crawl_delay_refusal_listed_no_text():
    r, _ = run([cand("https://a.example/p")], Fetch({"https://a.example/p": pf.FetchRefused("crawl_delay")}))
    assert r.pages == [] and r.dev[0]["gate"]["read"] == "read_crawl_delay"
    assert r.listed[0]["phrase"].startswith("not read yet") and r.listed[0]["read"] is False
