"""WPC-HOOK-1112: the Ask-flow hook that matches the permitted-originals catalogue for a definitional question
(vera.inline_acquire._original_html_hook). Offline: sockets raise, snapshot/config monkeypatched. Shadow must be
answer-neutral (candidate list identical to off) and log counts only."""
from __future__ import annotations

import copy
import logging
import socket

import pytest

from tests import acq_support as A
from tests import test_original_html_wpg_1111 as W
from vera import acq_discovered, inline_acquire, original_html

Q = "What are quarks and gluons?"
URLS = [W.DOE, "https://www.energy.gov/science/doe-explainsquarks-and-gluons-2",
        "https://science.nasa.gov/dark-matter"]
SECRETS = ("energy.gov", "nasa.gov", "Test page", "quarks", "gluons", "http")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    original_html.reset_for_tests()
    acq_discovered.reset_for_tests()

    def boom(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(socket.socket, "connect", boom)
    yield
    original_html.reset_for_tests()


def setup(monkeypatch, mode, urls=None, **limits):
    c = W.set_mode(monkeypatch, mode, **limits)
    snap = W.index_snapshot(urls or URLS)
    for e in snap["entries"]:
        e["title"] = "Quarks and Gluons"
        e["slug_terms"] = ["quarks", "gluons"]
    snap["sha256"] = original_html.snapshot_digest(snap)
    monkeypatch.setattr(original_html, "load_snapshot", lambda cfg=None: snap)
    return c


def pool():
    return [{"url": "https://example.org/a", "title": "A"}, {"doi": "10.9999/x", "title": "B"}]


def logs(caplog):
    return [r.getMessage() for r in caplog.records if "original_html_match" in r.getMessage()]


def test_off_is_noop(monkeypatch, caplog):
    setup(monkeypatch, "off")
    called = []
    monkeypatch.setattr(original_html, "match_subject", lambda *a, **k: called.append(1) or [])
    cands = pool()
    with caplog.at_level(logging.INFO):
        out = inline_acquire._original_html_hook(Q, cands)
    assert out is cands and out == pool() and not called and not logs(caplog)


def test_shadow_identical_to_off_and_counts_only(monkeypatch, caplog):
    setup(monkeypatch, "off")
    off = inline_acquire._original_html_hook(Q, pool())
    setup(monkeypatch, "shadow")
    cands = pool()
    with caplog.at_level(logging.INFO):
        out = inline_acquire._original_html_hook(Q, cands)
    assert out is cands and out == off == pool()
    ev = logs(caplog)
    assert len(ev) == 1 and '"n_matches"' in ev[0] and '"subject_tokens": 2' in ev[0]
    assert "row_energy_gov_doe_explains" in ev[0]
    for s in SECRETS:
        assert s.lower() not in ev[0].lower().replace("energy_gov_doe_explains", "")
    for r in caplog.records:
        m = r.getMessage()
        assert "Quarks" not in m and "https://" not in m and "quarks" not in m.lower().replace("subject_tokens", "")


def test_on_appends_capped_at_two(monkeypatch):
    setup(monkeypatch, "on", max_candidates_per_host=3, max_candidates_per_question=3)
    out = inline_acquire._original_html_hook(Q, pool())
    assert out[:2] == pool() and 1 <= len(out) - 2 <= 2
    assert all(c.get("original_html") and c["source"] == "permitted_catalogue" for c in out[2:])


def test_on_cap_exactly_two(monkeypatch):
    setup(monkeypatch, "on", max_candidates_per_host=3, max_candidates_per_question=3)
    assert len(original_html.match_subject("quarks and gluons")) == 3
    assert len(inline_acquire._original_html_hook(Q, [])) == 2


def test_on_matches_survive_rank_candidates(monkeypatch):
    from vera import auto_acquire as aa
    c = setup(monkeypatch, "on")
    out = inline_acquire._original_html_hook(Q, [])
    assert out and all(original_html.claims_cand(x, c) for x in out)
    acq = A.cfg()
    ranked = aa.rank_candidates(Q, out, acq, set(), aa._fd_cfg(), lambda code: None, drops={})
    assert {r[2]["url"] for r in ranked} == {x["url"] for x in out}


def test_what_does_x_mean_subject(monkeypatch):
    setup(monkeypatch, "on", urls=["https://science.nasa.gov/dark-matter"])
    seen = []
    real = original_html.match_subject
    monkeypatch.setattr(original_html, "match_subject", lambda s, *a, **k: seen.append(s) or real(s, *a, **k))
    inline_acquire._original_html_hook("What does neutrino mean?", [])
    inline_acquire._original_html_hook("What do quarks and gluons mean?", [])
    assert seen == ["neutrino", "quarks and gluons"]


def test_non_definitional_nothing(monkeypatch, caplog):
    setup(monkeypatch, "on")
    cands = pool()
    with caplog.at_level(logging.INFO):
        out = inline_acquire._original_html_hook("How do quarks and gluons interact in 2025 experiments given X?", cands)
    assert out is cands and not logs(caplog)


@pytest.mark.parametrize("m", ["shadow", "on"])
def test_exception_is_count_only_and_ask_continues(monkeypatch, caplog, m):
    setup(monkeypatch, m)

    def bad(*a, **k):
        raise RuntimeError("https://secret.example/Quarks")
    monkeypatch.setattr(original_html, "match_subject", bad)
    cands = pool()
    with caplog.at_level(logging.INFO):
        out = inline_acquire._original_html_hook(Q, cands)
    assert out is cands
    ev = logs(caplog)
    assert len(ev) == 1 and "original_html_match_error" in ev[0] and "RuntimeError" in ev[0]
    assert "secret.example" not in ev[0] and "Quarks" not in ev[0]
