"""NONASK-1103 I-A6 D4: shared per-source permission record. Offline, pure."""
import json
from pathlib import Path

from vera import source_permissions as sp

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config" / "source_permissions.json").read_text(encoding="utf-8"))
REG = json.loads((ROOT / "config" / "entity_registry.json").read_text(encoding="utf-8"))
CC_BY = {"id": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}
GEN = "https://example.org/page"


def ev(**kw):
    kw.setdefault("licence", CC_BY)
    kw.setdefault("registry", REG)
    kw.setdefault("config", CFG)
    return sp.evaluate(kw.pop("url", GEN), **kw)


def test_unknown_allowed_read_and_use_ingest_needs_licence():
    r = ev(licence=None)
    assert r.read and r.use_for_answer and not r.ingest
    assert r.ingest_reason == "ingest_no_storage_licence"
    g = sp.read_gate(GEN, registry=REG, config=CFG)
    assert g.read and g.use_for_answer and g.read_reason == "read_ok"


def test_ingest_requires_licence_and_terms():
    assert ev().ingest
    assert ev(url="https://docs.cloud.google.com/x").ingest  # allow row + CC BY
    assert not ev(url="https://docs.aws.amazon.com/x").ingest  # ingest forbid
    r = ev(url="https://docs.cloud.google.com/x", licence=None)
    assert not r.ingest and r.ingest_reason == "ingest_no_storage_licence"


def test_robots_disallow():
    r = sp.read_gate(GEN, robots_allowed=False, config=CFG)
    assert not r.read and r.read_reason == "read_robots_disallow"
    assert not ev(robots_allowed=False).ingest


def test_registry_hosts_read_forbidden_incl_subdomain():
    for u in ("https://docs.aws.amazon.com/a", "https://docs.nvidia.com/a", "https://learn.microsoft.com/a",
              "https://x.docs.nvidia.com/a"):
        r = sp.read_gate(u, registry=REG, config=CFG)
        assert not r.read and r.read_reason == "read_registry_forbid", u
    assert sp.read_gate("https://docs.cloud.google.com/a", registry=REG, config=CFG).read
    assert sp.read_gate("https://notaws.amazon.com.evil.org/a", registry=REG, config=CFG).read


def test_derivative_forbid_registry():
    r = sp.read_gate("https://docs.nvidia.com/a", registry=REG, config=CFG)
    assert not r.use_for_answer and r.use_reason == "use_registry_forbid"
    r2 = sp.read_gate("https://docs.aws.amazon.com/a", registry=REG, config=CFG)
    assert r2.use_for_answer  # AWS derivative allow (CC BY-SA)


def test_deny_list_host():
    assert CFG["deny_hosts"] == []
    r = sp.read_gate("https://sub.blocked.example/a", deny_list=["blocked.example"], config=CFG)
    assert not r.read and r.read_reason == "read_deny_list"
    assert sp.read_gate("https://blocked.example.org/a", deny_list=["blocked.example"], config=CFG).read
    r = sp.read_gate("https://b.example/a", config={"deny_hosts": [{"host": "b.example"}]})
    assert r.read_reason == "read_deny_list"


def test_login_and_paywall():
    assert ev(page_signals={"login_required": True}).read_reason == "read_login_paywall"
    assert ev(page_signals={"paywall": True}).read_reason == "read_login_paywall"
    assert ev(html_head='<input type="password">').read_reason == "read_login_paywall"
    assert ev(html_head='<script>{"isAccessibleForFree": "False"}</script>'.replace("False", "false")
              ).read_reason == "read_login_paywall"


def test_no_crawl_signal():
    assert ev(page_signals={"no_crawl": True}).read_reason == "read_no_crawl"
    assert ev(headers={"X-Robots-Tag": "nocrawl"}).read_reason == "read_no_crawl"


def test_noai_meta_and_headers():
    r = ev(html_head='<head><meta name="robots" content="noindex, NoAI"></head>')
    assert r.read and not r.use_for_answer and r.use_reason == "use_no_ai_signal" and not r.ingest
    assert ev(headers={"x-robots-tag": "noimageai"}).use_reason == "use_no_ai_signal"
    assert ev(headers={"X-Robots-Tag": "noai"}).use_reason == "use_no_ai_signal"
    assert ev(html_head="<meta content='noai' name='robots'>").use_reason == "use_no_ai_signal"
    assert ev(headers={"X-Robots-Tag": "noindex"}).use_for_answer


def test_content_signal():
    assert ev(headers={"Content-Signal": "ai-train=no, ai-input=no"}).use_reason == "use_no_ai_signal"
    assert ev(headers={"Content-Signal": "ai-train=no, ai-input=yes"}).use_for_answer
    robots = "User-agent: *\nContent-Signal: search=yes, ai-input=no\nAllow: /"
    assert ev(robots_txt_entries=robots).use_reason == "use_no_ai_signal"
    assert ev(robots_txt_entries=["Content-Signal: ai-input=no"]).use_reason == "use_no_ai_signal"
    assert ev(robots_txt_entries=["Content-Signal: ai-input=yes"]).use_for_answer


def test_nd_licences():
    for lic in ({"id": "CC BY-ND 4.0"}, {"url": "https://creativecommons.org/licenses/by-nd/4.0/"},
                {"id": "CC-BY-NC-ND-4.0"}, "https://creativecommons.org/licenses/by-nc-nd/4.0/"):
        r = ev(licence=lic)
        assert r.read and not r.use_for_answer and r.use_reason == "use_nd_licence" and not r.ingest, lic
    assert ev(licence={"id": "CC-BY-NC-4.0"}).use_for_answer  # plain NC accepted


def test_read_restricted_never_ingested():
    r = ev(url="https://docs.aws.amazon.com/x")
    assert not r.read and not r.ingest and r.ingest_reason == "ingest_not_readable"


def test_public_phrases_have_no_host_or_url():
    cases = [ev(url="https://docs.nvidia.com/a"), ev(robots_allowed=False), ev(licence={"id": "CC-BY-ND-4.0"}),
             ev(licence=None), ev(page_signals={"paywall": True}),
             sp.read_gate("https://q.blocked.example/a", deny_list=["blocked.example"], config=CFG)]
    for r in cases:
        for p in (r.read_phrase, r.use_phrase, r.ingest_phrase):
            low = p.lower()
            assert "http" not in low and "nvidia" not in low and "blocked.example" not in low and "." not in low.rstrip(".")
    for gate in CFG["public_phrases"].values():
        for p in gate.values():
            assert "http" not in p and "://" not in p
    assert ev(url="https://docs.nvidia.com/a").read_phrase == "not read: the site does not permit automated access"
    assert ev(licence={"id": "CC-BY-ND-4.0"}).use_phrase == "not used: the source does not permit reuse"
    assert ev().read_phrase == ""


def test_reason_codes_declared_in_config():
    codes = CFG["reason_codes"]
    r = ev(licence={"id": "CC-BY-ND-4.0"})
    assert r.read_reason in codes["read"] and r.use_reason in codes["use_for_answer"]
    assert r.ingest_reason in codes["ingest"]


def test_record_frozen():
    import dataclasses
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev().read = False


def test_registry_seed_values():
    rows = {e["entity"]: e for e in REG["entities"]}
    exp = {"AWS": ("forbid", "allow", "forbid"), "Nvidia": ("forbid", "forbid", "forbid"),
           "Azure": ("forbid", "forbid", "forbid"), "Google Cloud": ("allow", "allow", "allow")}
    for name, (a, d, i) in exp.items():
        e = rows[name]
        assert (e["access_verdict"], e["derivative_verdict"], e["ingest_verdict"]) == (a, d, i)
        assert all(e[k + "_evidence"] and "entity-hosts-evidence.md" in e[k + "_evidence"]
                   for k in ("access", "derivative", "ingest"))
    for name, e in rows.items():
        for k in ("access", "derivative", "ingest"):
            assert k + "_verdict" in e and k + "_evidence" in e
            if name not in exp:
                assert e[k + "_verdict"] == "unknown" and e[k + "_evidence"] is None
    assert rows["AWS"]["terms_verdict"] == "links_only" and rows["Google Cloud"]["terms_verdict"] == "allow"
    assert "_permissions_note" in REG
