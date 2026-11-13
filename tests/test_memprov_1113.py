"""MEMPROV-1113: portal providers leave the memory evidence list; discovery notes keep their own provider list.

Real shipped config file (config/ask-provider-chain.json, `memory` block), real candidate and SourceRef shapes, reserved values only
(example.org, 10.1000 DOIs, synthetic arXiv ids). Offline: no DB, no network."""
from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from vera import discovery_memory as dm
from vera import memory_gate as mg
from vera.discovery_memory import check_discovery_note, eligible_candidates
from vera.memory_gate import (MemoryGateConfigError, PORTAL_PROVIDERS, PORTAL_SOURCE, PROVIDER_NOT_ALLOWED, SourceRef,
                              check_memory_write, gate_config_from_dict, load_gate_config)

ROOT = Path(mg.__file__).resolve().parent.parent
BLOCK = json.loads((ROOT / "config" / "ask-provider-chain.json").read_text(encoding="utf-8"))["memory"]
CFG = load_gate_config()
GOOD = "Sparse attention reduces memory cost on long sequences while retaining accuracy on benchmarks."
Q = "How does sparse attention affect long documents?"
UNRELATED = "unrelated words here entirely"


def cand(provider="arxiv", n=1, **kw):
    d = {"title": "Efficient Int8 Weight Compression for Language Models", "url": f"https://arxiv.org/abs/2401.0000{n}",
         "provider": provider, "licence_decision": "allow", "identifier": f"arXiv:2401.0000{n}"}
    d.update(kw)
    return d


def evidence(src, cfg=CFG):
    return check_memory_write("cited_finding", GOOD, "supported", (src,), Q, cfg)


def note_result(c, cfg=CFG):
    note = dm.build_note([c], BLOCK["store"], gate_cfg=cfg)
    return check_discovery_note(note.split("\n")[0], [c], UNRELATED, cfg)


# ---- shipped config values (red on base) ----
def test_shipped_evidence_list_has_no_portal_provider():
    assert not set(CFG.allowed_providers) & PORTAL_PROVIDERS
    assert "openalex" not in CFG.allowed_providers


def test_shipped_discovery_list_is_today_list():
    assert CFG.discovery_allowed_providers == ("arxiv", "openalex")
    assert BLOCK["discovery_allowed_providers"] == ["arxiv", "openalex"]
    assert BLOCK["allowed_providers"] == []


def test_parity_evidence_subset_of_discovery_and_no_portal():
    assert set(CFG.allowed_providers) <= set(CFG.discovery_allowed_providers)
    assert not set(CFG.allowed_providers) & PORTAL_PROVIDERS


def test_missing_discovery_key_raises_verbosely():
    block = copy.deepcopy(BLOCK)
    del block["discovery_allowed_providers"]
    with pytest.raises(MemoryGateConfigError, match="discovery_allowed_providers"):
        gate_config_from_dict(block)


def test_block_with_key_loads():
    assert gate_config_from_dict(copy.deepcopy(BLOCK)) == CFG


# ---- discovery notes unchanged (green on base and after) ----
@pytest.mark.parametrize("provider", ["arxiv", "openalex", "arxiv+openalex"])
def test_discovery_eligible_and_note_allowed(provider):
    c = cand(provider)
    assert eligible_candidates([c], CFG) == [c]
    res = note_result(c)
    assert res.allowed, res.reasons


@pytest.mark.parametrize("provider", ["arxivx", "openalex+arxivx", "ARXIV", "stored_corpus", "", "duckduckgo"])
def test_discovery_lookalikes_ineligible(provider):
    c = cand(provider)
    assert eligible_candidates([c], CFG) == []
    res = note_result(c)
    assert not res.allowed and PROVIDER_NOT_ALLOWED in res.reasons


# ---- evidence writes ----
def test_evidence_arxiv_refused_on_list_and_floor():
    r = evidence(SourceRef("arxiv", "allow", "10.1000/example.1"))
    assert not r.allowed and PROVIDER_NOT_ALLOWED in r.reasons and PORTAL_SOURCE in r.reasons


def test_evidence_openalex_label_refused_even_with_journal_doi():
    r = evidence(SourceRef("openalex", "allow", "doi:10.1000/example.1", url="https://doi.org/10.1000/example.1"))
    assert not r.allowed and PROVIDER_NOT_ALLOWED in r.reasons and PORTAL_SOURCE not in r.reasons


def test_evidence_composite_arxiv_openalex_refused():
    r = evidence(SourceRef("arxiv+openalex", "allow", "doi:10.1000/example.1", url="https://doi.org/10.1000/example.1"))
    assert not r.allowed and PROVIDER_NOT_ALLOWED in r.reasons


def test_old_shape_config_still_floored_for_arxiv():
    old = dataclasses.replace(CFG, allowed_providers=("arxiv", "openalex"))
    r = evidence(SourceRef("arxiv", "allow", "10.1000/example.1"), old)
    assert PORTAL_SOURCE in r.reasons and PROVIDER_NOT_ALLOWED not in r.reasons


# ---- the split is real: swapping the two lists swaps behaviour (red on base) ----
def test_swapping_lists_swaps_behaviour():
    swapped = dataclasses.replace(CFG, allowed_providers=("openalex",), discovery_allowed_providers=())
    assert eligible_candidates([cand("arxiv"), cand("openalex", 2)], swapped) == []
    ok = evidence(SourceRef("openalex", "allow", "doi:10.1000/example.1", url="https://doi.org/10.1000/example.1"), swapped)
    assert ok.allowed
    only_disc = dataclasses.replace(CFG, allowed_providers=(), discovery_allowed_providers=("openalex",))
    assert eligible_candidates([cand("openalex", 2)], only_disc) != []


def test_provider_allowed_empty_list_is_honoured_not_falling_back():
    assert mg._provider_allowed("arxiv", CFG, ()) is False
    assert mg._provider_allowed("arxiv", CFG, ("arxiv",)) is True
    assert mg._provider_allowed("arxiv", CFG) is False  # None falls back to the (empty) evidence list
