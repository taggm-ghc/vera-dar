"""ROUTES-1108 INSPIRE batch: vera/portal_inspire.py (OD-1108-SRC item 2). SYNTHETIC JSON (invented DOIs 10.9999/...,
invented arXiv ids and an invented email). The API call is a fake: no socket, no DB, no model."""
from __future__ import annotations

import copy
import json
import pathlib

import pytest

from vera import portal_inspire as pi
from vera import reference_portal as rp

ROOT = pathlib.Path(__file__).resolve().parents[1]
EMAIL = "synthetic.author@example.org"
CANARY = "SYNTHETIC-INSPIRE-CANARY-3301"

DOC = {"hits": {"total": 2, "hits": [
    {"metadata": {"titles": [{"title": CANARY}], "abstracts": [{"value": CANARY}],
                  "authors": [{"full_name": "A", "emails": [EMAIL]}],
                  "dois": [{"value": "10.9999/synth.hep.1"}], "arxiv_eprints": [{"value": "2101.05555"}],
                  "references": [{"reference": {"dois": ["10.9999/synth.hep.ref1"], "title": {"title": CANARY}}},
                                 {"reference": {"arxiv_eprint": "hep-ph/9901001"}},
                                 {"reference": {"misc": [CANARY, EMAIL]}},
                                 {"record": {"$ref": "https://inspirehep.net/api/literature/1"}}]}},
    {"metadata": {"arxiv_eprints": [{"value": "2202.06666"}], "references": "not a list"}},
]}}


_REAL_PERMITTED = rp.inspire_permitted


class Resp:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code, self._b, self.headers = status, body, headers or {}

    def iter_content(self, n):
        for i in range(0, len(self._b), n):
            yield self._b[i:i + n]

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _reset():
    pi.reset_for_tests()
    rp.reset_for_tests()
    yield
    pi.reset_for_tests()


def cfg():
    c = copy.deepcopy(json.loads((ROOT / "config" / "reference_portals.json").read_text(encoding="utf-8")))
    c["adapters"]["inspire_refs"]["enabled"] = True
    return c


def call(get, q="What are quarks?", clock=lambda: 0.0, c=None):
    counts = {}
    out = pi.adapter(q, None, c or cfg(), None, counts, clock=clock, get=get)
    return out, counts


def test_one_metadata_call_with_identifier_fields_only_and_candidates():
    sent = []

    def get(url, params, t):
        sent.append((url, params))
        return Resp(200, json.dumps(DOC).encode())
    out, counts = call(get)
    assert len(sent) == 1 and sent[0][0] == "https://inspirehep.net/api/literature"
    p = sent[0][1]
    assert p["q"] == "quarks" and p["size"] == 3
    assert not any(f in p["fields"] for f in ("email", "authors", "abstract", "title"))
    ids = [(c.get("doi"), c.get("arxiv")) for c in out]
    assert ("10.9999/synth.hep.1", None) in ids and ("10.9999/synth.hep.ref1", None) in ids
    assert any(c.get("arxiv") == "hep-ph/9901001" for c in out) and any(c.get("arxiv") == "2202.06666" for c in out)
    blob = json.dumps(out)
    assert EMAIL not in blob and CANARY not in blob
    assert all(c["acq_source"] == "reference_portal" and c["portal"] == "inspire_refs" and c["snippet"] == ""
               for c in out)
    assert counts["inspire_calls"] == 1


def test_cache_and_429_cooldown_without_retry():
    n = []

    def ok(url, params, t):
        n.append(1)
        return Resp(200, json.dumps(DOC).encode())
    call(ok)
    _o, counts = call(ok, clock=lambda: 5.0)
    assert len(n) == 1 and counts["inspire_cache_hit"] == 1

    def busy(url, params, t):
        n.append(1)
        return Resp(429, b"", {"Retry-After": "2"})
    _o, counts = call(busy, q="What are gluons?", clock=lambda: 10.0)
    assert counts["inspire_429"] == 1 and len(n) == 2
    _o, counts = call(busy, q="What are gluons?", clock=lambda: 14.0)  # cooldown max(5, Retry-After 2) = 5 s
    assert counts["inspire_cooldown"] == 1 and len(n) == 2
    _o, counts = call(busy, q="What are gluons?", clock=lambda: 15.5)
    assert len(n) == 3


@pytest.mark.parametrize("body", [b"{bad", b"[]", json.dumps({"hits": {"hits": "x"}}).encode()])
def test_any_other_shape_yields_nothing(body):
    out, _ = call(lambda u, p, t: Resp(200, body))
    assert out == []


def test_size_cap_and_errors_never_raise():
    c = cfg()
    c["adapters"]["inspire_refs"]["max_response_bytes"] = 10
    out, counts = call(lambda u, p, t: Resp(200, json.dumps(DOC).encode()), c=c)
    assert out == [] and counts["inspire_too_large"] == 1

    class Denied(Exception):
        code = "R-API-DENIED"

    def deny(u, p, t):
        raise Denied()
    out, counts = call(deny, q="What are leptons?")
    assert out == [] and counts["inspire_r_api_denied"] == 1


def test_without_the_api_allowlist_module_nothing_is_sent(monkeypatch):
    import builtins
    import vera.selective_fetch  # noqa: F401  import it before the import hook is installed
    real = builtins.__import__

    def imp(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vera" and fromlist and "api_permissions" in fromlist:
            raise ImportError("not shipped")
        return real(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", imp)
    sent = []
    import vera.selective_fetch as sf
    monkeypatch.setattr(sf, "pinned_get", lambda *a, **k: sent.append(1))
    out, counts = call(None)
    assert out == [] and sent == [] and counts["inspire_api_gate_missing"] == 1


def test_stop_word_question_sends_nothing():
    out, counts = call(lambda *a: pytest.fail("sent"), q="what is the")
    assert out == [] and counts == {}


def test_loaded_through_the_generic_portal_when_enabled():
    c = cfg()
    for a in ("wikipedia_title", "generic_page", "openalex_rw"):
        c["adapters"][a]["enabled"] = False
    c["mode"] = "on"
    import vera.portal_inspire as mod
    rp.inspire_permitted = lambda: True  # restored by the monkeypatch-free finally below
    got = {}

    def fake(question, base, cfg_, deps, counts, clock, deadline):
        got["q"] = question
        return [{"doi": "10.9999/synth.hep.9", "url": "https://doi.org/10.9999/synth.hep.9",
                 "acq_source": "reference_portal"}]
    orig = mod.adapter
    mod.adapter = fake
    try:
        cands, counts = rp.collect("What are quarks?", cfg=c, deps=rp.Deps(), hosts={}, clock=lambda: 0.0)
    finally:
        mod.adapter = orig
        del rp.inspire_permitted
        rp.inspire_permitted = _REAL_PERMITTED
    assert got["q"] == "What are quarks?" and [x["doi"] for x in cands] == ["10.9999/synth.hep.9"]


def test_default_get_passes_the_transport_regate_with_purpose_metadata(monkeypatch):
    """ROUTES-1108-VERIFY: once the owner permits the entry, pinned_get's own per-hop gate must check the SAME purpose
    (metadata), not its default fulltext; a public-address stub refusal proves the gates passed and nothing was sent."""
    import vera.api_permissions as ap
    import vera.selective_fetch as sf
    raw = json.loads((ROOT / "config" / "api_permissions.json").read_text(encoding="utf-8"))
    e = raw["entries"]["inspire_literature"]
    assert e["owner_switch"] == "deny"  # shipped denied
    e["owner_switch"], e["owner_decision"] = "permit", "OD-1108-SRC"
    ap.set_policy_for_tests(ap._validate(raw))
    monkeypatch.setattr(sf, "is_public_url", lambda u: (False, "stub"))
    try:
        with pytest.raises(sf.PinnedRefused):
            pi._default_get("https://inspirehep.net/api/literature", {"q": "quark"}, 2.0)
        with pytest.raises(sf.ApiRefused):  # a fulltext transport hop stays refused (scope)
            sf.pinned_get("https://inspirehep.net/api/literature", timeout=2.0)
    finally:
        ap.set_policy_for_tests(None)


@pytest.mark.parametrize("mode,enabled,switch,calls", [
    ("on", True, "deny", 0), ("off", True, "permit", 0), ("shadow", True, "permit", 0),
    ("on", False, "permit", 0), ("on", True, "permit", 1)])
def test_inspire_called_only_when_mode_on_and_enabled_and_owner_permit(monkeypatch, mode, enabled, switch, calls):
    from vera import api_permissions as ap
    import vera.portal_inspire as mod
    raw = json.loads((ROOT / "config" / "api_permissions.json").read_text(encoding="utf-8"))
    raw["entries"]["inspire_literature"]["owner_switch"] = switch
    raw["entries"]["inspire_literature"]["owner_decision"] = "OD-1108-SRC"
    monkeypatch.setattr(ap, "get_policy", lambda: ap._validate(raw))
    c = cfg()
    for a in ("wikipedia_title", "generic_page", "openalex_rw"):
        c["adapters"][a]["enabled"] = False
    c["mode"], c["adapters"]["inspire_refs"]["enabled"] = mode, enabled
    n = []
    monkeypatch.setattr(mod, "adapter", lambda *a, **k: n.append(1) or [])
    rp.collect("What are quarks?", cfg=c, deps=rp.Deps(), hosts={}, clock=lambda: 0.0)
    assert len(n) == calls


def test_shipped_inspire_is_off_and_denied():
    c = json.loads((ROOT / "config" / "reference_portals.json").read_text(encoding="utf-8"))
    assert c["mode"] == "off" and c["adapters"]["inspire_refs"]["enabled"] is False
    assert rp.inspire_permitted() is False
