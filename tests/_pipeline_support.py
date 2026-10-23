"""Offline helpers for the DAR-PIPE-1022 acceptance tests, v2 (written against commit 7da574a, CMF-1022).

Nothing here opens a socket, a DB connection or reads env files. It reuses the 7da574a test conventions:
  tests/acq_support.py          A.Web / A.Store / A.Lookup / A.deps / A.epi_setup (fake Europe PMC + JATS, fake store)
  tests/test_acq_discovered.py  epmc_web / oa_src / arxiv_src / Router (exact-URL routes, call log)
  tests/test_dar_inline_acquire.py  the `rig` fixture (fake deps factory, fake scope decision, no DB, no network)
  tests/test_acq_serve.py       the `h` fixture (db_configured patched, enqueue/inline stubs, Harness)
  tests/conftest.py + vera_test_boundary  the kernel-level no-network / no-DB boundary
Config is changed ONLY on the rig's private deep copy (rig.cfg["inline"][...] = ...), never on a file."""
from __future__ import annotations

import json

from tests import acq_support as A
from tests import test_fulldoc_e2e as e2e
from tests.test_acq_discovered import (DOI, PMC, W1, Router, arxiv_src, epmc_web, oa_src)  # noqa: F401  (re-exported)
from tests.test_acq_serve import BASE_NO_EPI, _decide, h  # noqa: F401  (h is a fixture)
from tests.test_dar_inline_acquire import GROUND, rig  # noqa: F401  (rig is a fixture)
from vera import ask_service

Q_EPI = e2e.Q_EPI
PRICING = e2e.PRICING
Q_NONE = e2e.Q_NONE
ABSTRACT_SENTINEL = "SENTINEL-DISCOVERED-ABSTRACT"
PRIVATE_IP = "10.0.0.5"


class DiscSearch:
    """The request's reading-list search seam (grounding[0]): returns the given discovered sources and logs the calls."""

    def __init__(self, sources):
        self.sources, self.calls = list(sources), []

    def __call__(self, restricted=False):
        self.calls.append(restricted)
        return [dict(s) for s in self.sources]


def discovered(**over) -> dict:
    """An OpenAlex-style discovered record: a DOI, an abstract (never full text) and the index's metadata licence."""
    return oa_src(snippet=f"{ABSTRACT_SENTINEL} transmission incidence outbreak pathogen", **over)


def ask_offtopic(rig, sources, q=Q_EPI, *, use_discovered=True, cands=(), web=None, mutate=None, deps_mutate=None):
    """Corpus miss (the base corpus holds no epidemiology paper) through ask_service.answer_in_scope on the proven
    off-topic override path of tests/test_dar_inline_acquire.Rig. The acquisition's own search returns `cands`
    (default none), so only discovered records can reach acquisition. Returns (result, DiscSearch)."""
    rig.cfg["inline"]["use_discovered"] = bool(use_discovered)  # test-time override on the rig's deep copy only
    if mutate:
        mutate(rig.cfg)
    rig.setup(web=web, cands=list(cands))
    if deps_mutate:
        deps_mutate(rig.deps)  # e.g. replace url_check on the fake deps
    ds = DiscSearch(sources)
    _decide(rig.mp, False, "layer2_llm", "llm_out:x", "off_topic")
    r = ask_service.answer_in_scope(q, "m", e2e.PRICING, grounding=(ds, GROUND[1]))
    return r, ds


def disc_entries(r) -> list:
    """Per-candidate outcomes: contract.inline_acquisition.discovered (inline_acquire.py:246, 7da574a)."""
    return list(((r.contract or {}).get("inline_acquisition") or {}).get("discovered") or [])


def entry_for(r, record_id=W1) -> dict:
    for e in disc_entries(r):
        if e.get("record_id") == record_id:
            return e
    return {}


def reason_code(rig, key) -> str:
    from vera import acq_config
    return acq_config.reason(rig.cfg, key)


def stored_text(plans) -> str:
    return "".join(s["text"] for p in plans for s in p["spans"])


def passages_sent_to_model(rig) -> list:
    return [p["text"] for p in json.loads(rig.h.answer_calls[0][-1]["content"])["passages"]]
