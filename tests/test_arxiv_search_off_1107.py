"""CONTROL-GAPS-1107 ARXIV-OFF: the shipped configs never query export.arxiv.org or /api/query (robots.txt: Disallow /)."""
import json
import pathlib

import pytest
import requests

from vera import auto_acquire as aa
from vera import search_and_fetch
from vera import search_providers as sp

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _cfg(name):
    return json.loads((ROOT / "config" / name).read_text())


def _bad(url):
    u = str(url).lower()
    return "export.arxiv.org" in u or "/api/query" in u


class _Spy:
    def __init__(self):
        self.urls = []

    def __call__(self, url, *a, **kw):
        self.urls.append(str(url))
        if _bad(url):
            raise AssertionError(f"forbidden arXiv API call: {url}")
        raise requests.ConnectionError("offline test: no network")


def test_shipped_flags_are_off_and_fulltext_route_stays_on():
    assert _cfg("vera_eval_run.json")["search"]["arxiv"]["enabled"] is False
    assert "arxiv" not in _cfg("auto_acquire.json")["search"]["providers"]
    assert _cfg("auto_acquire.json")["arxiv_route"]["enabled"] is True


def test_shipped_chain_has_no_arxiv_provider():
    chain, _, _ = search_and_fetch.cached_chain(search_and_fetch._search_settings())
    assert "arxiv" not in [p.name for p in chain]


def test_shipped_chain_and_acquisition_search_never_call_the_arxiv_api(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(requests, "get", spy)
    monkeypatch.setattr(requests, "post", spy)
    monkeypatch.setattr(requests.Session, "request", lambda self, m, url, *a, **k: spy(url))
    monkeypatch.setattr(sp, "arxiv_default_transport", spy)
    settings = search_and_fetch._search_settings()
    chain = sp.build_chain(settings, repo_root=ROOT, transport=spy)
    assert "arxiv" not in [p.name for p in chain]
    for p in chain:  # run a real search through every shipped provider with the spy as transport
        try:
            p.call("tau protein alzheimer", 5, 2.0)
        except Exception:
            pass
    cfg = _cfg("auto_acquire.json")
    search = aa.make_search(cfg, chain_fn=lambda: chain)
    try:
        search("tau protein alzheimer", 5, 2.0)
    except Exception:
        pass
    assert spy.urls, "spy saw no request: the test exercised nothing"
    assert not [u for u in spy.urls if _bad(u)]
