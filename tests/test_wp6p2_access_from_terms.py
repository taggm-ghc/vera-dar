"""NONASK-1103 WP6-P2 (R-050): access is derived from terms_verdict when access_verdict is missing; a links_only row
(known-forbid terms: AWS, Nvidia, Microsoft Learn) is never fetched; unknown rows may be read."""
import json
from pathlib import Path

import pytest

from vera import source_permissions as sp

CFG = {"deny_hosts": []}


def _row(**kw):
    return {"entity": "X", "domains": ["docs.x.example"], "state": "active", **kw}


@pytest.mark.parametrize("row, want", [
    (_row(terms_verdict="links_only"), "forbid"),
    (_row(terms_verdict="allow"), "allow"),
    (_row(terms_verdict="unknown"), "unknown"),
    (_row(), "unknown"),
    (_row(terms_verdict="links_only", access_verdict="unknown"), "forbid"),  # links_only always wins
    (_row(terms_verdict="unknown", access_verdict="forbid"), "forbid"),
    (_row(terms_verdict="allow", access_verdict="forbid"), "forbid"),
])
def test_access_verdict_derivation(row, want):
    assert sp.access_verdict(row) == want


def test_links_only_row_never_read_unknown_row_readable():
    u = "https://docs.x.example/start"
    assert not sp.read_gate(u, registry=[_row(terms_verdict="links_only")], config=CFG).read
    assert sp.read_gate(u, registry=[_row(terms_verdict="links_only")], config=CFG).read_reason == "read_registry_forbid"
    assert sp.read_gate(u, registry=[_row(terms_verdict="unknown")], config=CFG).read
    assert sp.read_gate(u, registry=[_row(terms_verdict="allow")], config=CFG).read


def test_shipped_links_only_rows_are_forbid():
    reg = json.loads((Path(__file__).resolve().parent.parent / "config" / "entity_registry.json").read_text("utf-8"))
    lo = [e for e in reg["entities"] if e.get("terms_verdict") == "links_only"]
    assert {e["entity"] for e in lo} >= {"AWS", "Nvidia", "Azure"}
    assert all(sp.access_verdict(e) == "forbid" for e in lo)
