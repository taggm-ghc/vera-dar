"""OD-1109-SIX (owner 2026-11-09): every question searches for >= 6 NEW documents (not already stored) before answering,
following portal links toward original hosts (bounded hops). SYNTHETIC fixtures only; no socket, DB or model."""
from __future__ import annotations

import json
import pathlib
import re
from types import SimpleNamespace

import pytest

from vera import inline_acquire as ia
from vera import reference_portal as rp
from vera import research_record as rr
from vera.ask_service import _six_line

ROOT = pathlib.Path(__file__).resolve().parents[1]
ACQ = json.loads((ROOT / "config" / "auto_acquire.json").read_text())


# ------------------------------------------------------------------------------------------------ shipped config
def test_shipped_config_six_rule():
    f = ACQ["freshness"]
    assert f["always_acquire_new"] is True and rr.always_new(f)
    assert ACQ["bounds"]["acq_round_target_new_docs"] == 6
    assert ACQ["bounds"]["acq_round_count_reused"] is False and not ia.count_reused(ACQ)
    assert "{n}" in f["texts"]["new_documents"] and "{m}" in f["texts"]["new_documents"]
    assert "min_distinct_documents" not in f["sufficiency"]  # the cited-evidence floor is dropped
    portals = json.loads((ROOT / "config" / "reference_portals.json").read_text())
    assert portals["caps"]["max_hops"] == 2


def test_reused_stored_documents_do_not_count_toward_six():
    reused = (["a", "b", "c"], [1, 2, 3])
    assert ia._rn(ACQ, reused) == 0
    assert ia._rn({"bounds": {"acq_round_count_reused": True}}, reused) == 3  # pre-SIX switch
    assert ia._rn({}, reused) == 0


def test_six_line_states_new_count_only_under_the_trigger():
    texts = ACQ["freshness"]["texts"]
    o = SimpleNamespace(admitted=2, web=SimpleNamespace(pages=[object()]))
    line = _six_line(texts, [rr.TRIGGER_NEW], o, ACQ)
    assert "3 new full documents" in line and "at least 6" in line
    one = _six_line(texts, [rr.TRIGGER_NEW], SimpleNamespace(admitted=1, web=None), ACQ)
    assert "1 new full document " in one
    assert _six_line(texts, [rr.TRIGGER_STALE], o, ACQ) == ""  # trigger off: nothing said
    assert _six_line(texts, [rr.TRIGGER_NEW], None, ACQ) == ""  # no search ran (guards): nothing claimed


def test_always_new_switch_is_exact_true():
    assert not rr.always_new({"always_acquire_new": "true"}) and not rr.always_new(None)


# ------------------------------------------------------------------------------------------------ portal hop 2
class Resp:
    def __init__(self, body):
        self.status_code, self._b = 200, body.encode()
        self.headers = {"Content-Type": "text/html; charset=utf-8"}

    def iter_content(self, n):
        for i in range(0, len(self._b), n):
            yield self._b[i:i + n]

    def close(self):
        pass


P1 = "https://en.wikipedia.org/wiki/Zorblats"
P2 = "https://portal-two.example.org/records/zorb"
P3 = "https://portal-three.example.org/records/zorb"
PAGE1 = (f'<a href="{P2}">a further portal</a><a href="https://unknown.example.net/x">unknown host</a>'
         '<a href="/wiki/Nav">same site</a>')
PAGE2 = f'<a href="https://doi.org/10.9999/synth.hop2">original</a><a href="{P3}">third portal</a>'
PAGE3 = '<a href="https://doi.org/10.9999/synth.hop3">too deep</a>'


def _hosts():
    rx = re.compile(r"/(wiki|records)/[A-Za-z_]+")
    return {h: {"host": h, "paths": rx, "rate_portal": "wikipedia"}
            for h in ("en.wikipedia.org", "portal-two.example.org", "portal-three.example.org")}


def _cfg(max_hops):
    c = json.loads((ROOT / "config" / "reference_portals.json").read_text())
    c["mode"] = "on"
    for a in c["adapters"].values():
        if isinstance(a, dict):
            a["enabled"] = False
    c["adapters"]["wikipedia_title"]["enabled"] = True
    c["caps"]["max_hops"] = max_hops
    return c


def _collect(max_hops, monkeypatch):
    calls = []
    pages = {P1: PAGE1, P2: PAGE2, P3: PAGE3}

    def get(url, **kw):
        calls.append(url)
        return Resp(pages[url]) if url in pages else SimpleNamespace(status_code=404, headers={}, close=lambda: None,
                                                                       iter_content=lambda n: iter(()))
    deps = rp.Deps(http_get=get, url_check=lambda u: (True, ""), robots_ok=lambda u, ua: True,
                   no_ai_input=lambda u, ua: False, tdmrep_ok=lambda u, p=None: True, crawl_ok=lambda u, ua: True,
                   limited=lambda portal, fn: fn, openalex_lookup=lambda s, n, t: [])
    monkeypatch.setattr(rp, "wikipedia_title_urls", lambda q, c: [P1])
    cands, counts = rp.collect("What are zorblats?", cfg=_cfg(max_hops), deps=deps, hosts=_hosts(),
                               clock=lambda: 0.0)
    return {c.get("doi") for c in cands}, counts, calls


@pytest.fixture(autouse=True)
def _reset():
    rp.reset_for_tests()
    yield
    rp.reset_for_tests()


def test_portal_hop_two_reaches_the_original_identifier(monkeypatch):
    dois, counts, calls = _collect(2, monkeypatch)
    assert "10.9999/synth.hop2" in dois
    assert "10.9999/synth.hop3" not in dois and P3 not in calls  # bounded: no third hop
    assert not any("unknown.example.net" in u for u in calls)  # default deny: unregistered host never fetched
    assert counts["hop_queued"] == 1


def test_single_hop_is_the_pre_six_behaviour(monkeypatch):
    dois, counts, calls = _collect(1, monkeypatch)
    assert calls == [P1] and "10.9999/synth.hop2" not in dois and "hop_queued" not in counts


def test_hops_respect_the_per_question_page_cap(monkeypatch):
    c = _cfg(3)
    c["caps"]["max_pages_per_question"] = 2
    calls = []
    pages = {P1: PAGE1, P2: PAGE2, P3: PAGE3}
    deps = rp.Deps(http_get=lambda u, **k: (calls.append(u), Resp(pages[u]))[1], url_check=lambda u: (True, ""),
                   robots_ok=lambda u, ua: True, no_ai_input=lambda u, ua: False, tdmrep_ok=lambda u, p=None: True,
                   crawl_ok=lambda u, ua: True, limited=lambda portal, fn: fn, openalex_lookup=lambda s, n, t: [])
    monkeypatch.setattr(rp, "wikipedia_title_urls", lambda q, cc: [P1])
    rp.collect("What are zorblats?", cfg=c, deps=deps, hosts=_hosts(), clock=lambda: 0.0)
    assert len([u for u in calls if u in pages]) <= 2
