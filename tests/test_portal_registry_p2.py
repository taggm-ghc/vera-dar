"""PORT-P2: portal registry, discovery attribution, versioned metrics and portal selection (offline)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from vera import portal_recorder as dr
from vera import portal_store as ds
from vera import portal_registry as pr
from vera import portal_select as psel

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config" / "portal_registry.json").read_text(encoding="utf-8"))
IMPORT = json.loads((ROOT / "config" / "portal_sources" / "authoritative_sources.json").read_text(encoding="utf-8"))
ALL_ADAPTERS = lambda a: bool(a)  # noqa: E731  adapter present in config = supported (unit tests only)


def _reg():
    return pr.load_registry(CFG)


# ------------------------------------------------------------------------------------------- registry
def test_seed_portals_and_real_adapters_supported():
    reg = _reg()
    for k in ("ddg", "openalex", "europepmc", "doaj", "arxiv", "wikipedia", "pubmed"):
        assert k in reg.portals
    st = pr.statuses(reg)
    for k in ("ddg", "openalex", "europepmc", "doaj", "arxiv", "wikipedia"):
        assert st[k]["supported"], k  # the adapter really exists in code
    assert st["pubmed"]["level"] == "registered"  # no adapter
    assert reg.get("duckduckgo_lite").key == "ddg" and reg.get("epmc").key == "europepmc"
    assert reg.resolve_key("wikipedia_portal") == "wikipedia"


def test_registration_dedup_by_key_and_url():
    reg = _reg()
    n = len(reg.portals)
    p, created = reg.register(pr.Portal(key="web:x", name="Dup", canonical_url="https://DOAJ.org/", category="other",
                                        provenance=("test",)))
    assert not created and p.key == "doaj" and len(reg.portals) == n and "test" in p.provenance


def test_import_owner_sources_counts_and_levels():
    reg = _reg()
    entries = IMPORT["entries"]
    imported = [k for k, p in reg.portals.items() if any(x.startswith("import:") for x in p.provenance)]
    merged = [e for e in entries if e.get("existing_vera_portal")]
    assert {e["existing_vera_portal"] for e in merged} <= set(reg.portals)
    assert "journals" in reg.portals["doaj"].domains  # owner page section recorded as a domain
    assert any(x.startswith("import:") for x in reg.portals["arxiv"].provenance)
    assert len(imported) >= len({pr.portal_key_for_url(e["url"]) for e in entries if not e.get("existing_vera_portal")})
    st = pr.statuses(reg)
    for k in imported:
        p = reg.portals[k]
        if k not in ("doaj", "arxiv"):
            assert st[k]["level"] == "registered" and not p.enabled and p.adapter is None, k
        assert p.category in pr.CATEGORIES
    nonsearch = [p for p in reg.portals.values() if p.category == "reference_resource_not_searchable"]
    assert nonsearch and all(st[p.key]["level"] == "registered" for p in nonsearch)


def test_listed_url_alone_never_operational():
    reg = _reg()
    health = {k: {"last_success_at": 1e9, "failures_since_success": 0} for k in reg.portals}
    st = pr.statuses(reg, health, now_epoch=1e9 + 60)
    for k, p in reg.portals.items():
        if p.adapter is None:
            assert st[k]["level"] == "registered"


def test_status_levels_configured_and_operational():
    reg = _reg()
    p = reg.portals["openalex"]
    cfg_yes = lambda rel: {"search": {"providers": ["openalex"]}}  # noqa: E731
    cfg_no = lambda rel: {"search": {"providers": []}}  # noqa: E731
    assert pr.status(p, cfg=CFG, read_json=cfg_no)["level"] == "supported"
    assert pr.status(p, cfg=CFG, read_json=cfg_yes)["level"] == "configured"  # no health yet
    ok = {"last_success_at": 1000.0, "failures_since_success": 0}
    assert pr.status(p, cfg=CFG, read_json=cfg_yes, health=ok, now_epoch=1100.0)["level"] == "operational"
    bad = {"last_success_at": 1000.0, "failures_since_success": CFG["health"]["max_consecutive_failures"]}
    assert pr.status(p, cfg=CFG, read_json=cfg_yes, health=bad, now_epoch=1100.0)["level"] == "configured"
    stale = CFG["health"]["max_success_age_hours"] * 3600 + 1001.0
    assert pr.status(p, cfg=CFG, read_json=cfg_yes, health=ok, now_epoch=stale)["level"] == "configured"


def test_multiple_portals_per_domain_and_many_domains_per_portal():
    reg = _reg()
    bio = {p.key for p in reg.by_domain("Biomedicine")}
    assert {"europepmc", "pubmed", "ddg", "openalex"} <= bio
    assert len(reg.portals["europepmc"].domains) >= 2
    pairs = set(pr.domain_pairs(reg))
    assert ("europepmc", "biomedicine") in pairs and ("pubmed", "biomedicine") in pairs
    assert any(r["key"] == "europepmc" for r in reg.portals["pubmed"].related)


# ------------------------------------------------------------------------------------------- attribution
def _rec(portals, candidates, *, documents=()):
    return {"triggered": True, "portals": portals, "candidates": candidates,
            "freshness": {"before": None, "after": {"documents": list(documents)}}}


def _record(store, rec, **kw):
    return dr.record(store, rec, registry=_reg(), question="Q?", **kw)


def test_one_document_many_portals_unique_vs_credited():
    store = ds.SQLiteDiscoveryStore()
    rec = _rec([{"portal": "ddg", "status": "completed", "results": 5},
                {"portal": "openalex", "status": "completed", "results": 7},
                {"portal": "europepmc", "status": "completed", "results": 3}],
               [{"portal": "ddg", "found_by": ["ddg", "openalex", "epmc"], "doi": "10.1/ABC", "outcome": "admitted",
                 "terminal_reason": "admitted", "n_chunks": 4, "canonical_ref": ds.canonical_ref("https://x/1")},
                {"portal": "openalex", "doi": "10.1/def", "outcome": "rejected",
                 "terminal_reason": "relevance_rejected"}],
               documents=[{"document_id": "d1", "freshness": "fresh", "source_date": "2026-10-01",
                           "acquired_at": "2026-10-23T00:00:00"}])
    sources = [{"n": 1, "document_id": "d1", "canonical_source_id": "https://x/1", "url": "https://europepmc.org/x"}]
    claims = [{"id": 1, "citations": [1], "verdict": "supported"}]
    rid = _record(store, rec, sources=sources, claims=claims, accepted=True)
    t = store.portal_tallies()
    assert t["ddg"]["deduplicated_documents"] == 1 and t["openalex"]["deduplicated_documents"] == 2
    assert t["europepmc"]["valuable"] == 1 and t["ddg"]["valuable"] == 1 and t["openalex"]["valuable"] == 1
    assert sum(x["valuable"] for x in t.values()) == 3  # credited to each portal ...
    u = store.unique_totals()
    assert u["documents"] == 2 and u["valuable"] == 1 and u["cited"] == 1  # ... but one unique publication
    assert u["admitted_chunks"] == 4 and u["freshness_qualified"] == 1 and u["rejected"] == 1
    hist = store.attribution_history("doi:10.1/abc")
    assert [h["portal_key"] for h in hist] == ["ddg", "openalex", "europepmc"] and {h["request_id"] for h in hist} == {rid}
    o = store.rows("SELECT * FROM document_outcome WHERE doc_key='doi:10.1/abc'")[0]
    assert o["acquired_at"] and o["source_date"] == "2026-10-01" and o["definitions_version"] == 1
    # a second request finding the same document adds history, not a new publication
    _record(store, rec, sources=sources, claims=claims, accepted=True)
    assert len(store.attribution_history("doi:10.1/abc")) == 6
    assert store.rows("SELECT COUNT(*) n FROM document_identity")[0]["n"] == 2


def test_valuable_is_not_availability_relevance_or_citation_alone():
    store = ds.SQLiteDiscoveryStore()
    ref = ds.canonical_ref("https://x/2")
    base = {"portal": "openalex", "doi": "10.2/a", "outcome": "admitted", "terminal_reason": "admitted",
            "n_chunks": 2, "canonical_ref": ref}
    src = [{"n": 1, "document_id": "d", "canonical_source_id": "https://x/2"}]
    sup = [{"citations": [1], "verdict": "supported"}]
    cases = [
        (dict(base), src, [], True),  # available + relevant, no claim check -> not valuable
        (dict(base), src, [{"citations": [1], "verdict": "partial"}], True),  # cited, only partially supported
        (dict(base), src, sup + [{"citations": [1], "verdict": "unsupported"}], True),  # contradicted
        (dict(base), src, sup, False),  # answer not accepted
        (dict(base, outcome="rejected", terminal_reason="relevance_rejected"), src, sup, True),  # not relevant
        (dict(base, canonical_ref=None), [], sup, True),  # relevant + acquired but not among the answer sources
    ]
    for c, s_, cl, accepted in cases:
        assert dr.stage_outcome(c, s_, cl, {}, accepted=accepted)["valuable"] is False, (c, cl, accepted)
    good = dr.stage_outcome(dict(base), src, [{"citations": [1], "verdict": "supported"}], {}, accepted=True)
    assert good["valuable"] is True and good["cited"] is True and good["contradictory"] is False
    contra = dr.stage_outcome(dict(base), src, [{"citations": [1], "verdict": "unsupported"}], {}, accepted=True)
    assert contra["contradictory"] is True and contra["valuable"] is False
    assert "never availability" in CFG["metrics"]["definitions"]["valuable"].lower() \
        and CFG["metrics"]["definitions"]["calibrated"] is False


def test_failure_and_quota_accounting():
    store = ds.SQLiteDiscoveryStore(clock=iter(range(100, 1000)).__next__)
    rec = _rec([{"portal": "openalex", "status": "skipped_quota", "results": 0},
                {"portal": "ddg", "status": "timeout", "results": 0},
                {"portal": "europepmc", "status": "failed", "results": 0},
                {"portal": "wikipedia", "status": "skipped_budget", "results": 0},
                {"portal": "doaj", "status": "quota"}], [])
    _record(store, rec)
    t = store.portal_tallies()
    assert t["openalex"]["events"] == {"quota": 1} and t["doaj"]["events"] == {"quota": 1}
    assert t["ddg"]["events"] == {"timeout": 1} and t["europepmc"]["events"] == {"error": 1}
    assert t["wikipedia"]["events"] == {"skipped": 1}
    h = store.health()
    assert "wikipedia" not in h and h["ddg"]["failures_since_success"] == 1 and h["ddg"]["last_success_at"] is None
    assert ds.rates(t["ddg"])["success_rate"] == 0.0
    events = store.rows("SELECT query_hash FROM discovery_event")
    assert all(e["query_hash"] == dr.query_hash("q?") for e in events)  # hash only (no question text)


def test_recorder_noop_when_disabled_and_never_raises(monkeypatch):
    assert ds.open_store(CFG) is None  # shipped disabled
    assert dr.record(None, {"triggered": True}) is None
    assert dr.record(ds.SQLiteDiscoveryStore(), {"triggered": False, "portals": [{"portal": "ddg"}]}) is None
    monkeypatch.setattr(dr, "_registry_cfg", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    dr.record_ask({"triggered": True})  # swallowed


def test_research_record_and_merge_feed_attribution():
    """End to end over the real FRESH-1023 record builder and the real merge: duplicates keep every portal."""
    from vera import research_record as rr
    from vera import search_providers as sp
    merged = sp.merge_sources([{"doi": "10.9/z", "acq_source": "ddg", "provider": "ddg"}],
                              [{"doi": "10.9/Z", "provider": "openalex"}],
                              [{"doi": "10.9/z", "provider": "europepmc"}])
    assert len(merged) == 1 and merged[0]["found_by"] == ["ddg", "openalex", "epmc"]
    assert sp.acq_source(merged[0]) == "ddg"  # first occurrence keeps its place and source
    cand = {"portal": "ddg", "doi": "10.9/z", "pmcid": None, "openalex_id": None, "arxiv_id": None,
            "url": "https://doi.org/10.9/z",
            "published": None, "outcome": "admitted", "reason": None, "found_by": merged[0]["found_by"],
            "n_chunks": 3, "canonical_ref": ds.canonical_ref("https://c/z")}
    inline = {"status": "ok", "portals": [{"portal": "ddg", "status": "completed", "results": 1},
                                          {"portal": "openalex", "status": "completed", "results": 1},
                                          {"portal": "europepmc", "status": "completed", "results": 1}],
              "candidates": [cand]}
    rec = rr.summarize(triggers=["insufficient"], inline=inline, used_new=True, older_available=False, fr_before=None)
    store = ds.SQLiteDiscoveryStore()
    dr.record(store, rec, registry=_reg(), sources=[{"n": 1, "document_id": "d", "canonical_source_id": "https://c/z"}],
              claims=[{"citations": [1], "verdict": "supported"}], accepted=True)
    t = store.portal_tallies()
    assert all(t[k]["valuable"] == 1 and t[k]["acq_succeeded"] == 1 for k in ("ddg", "openalex", "europepmc"))
    assert store.unique_totals()["valuable"] == 1
    pub = rr.public_fields(rec)["research_outcome"]["candidates"][0]
    # additive only (URLPUBLIC-1023 / PMCIDPUBLIC-1023): the URL and ids are kept; found_by and canonical_ref are added
    assert pub["url"] == "https://doi.org/10.9/z" and pub["ids"]["doi"] == "10.9/z"
    assert pub["found_by"] == ["ddg", "openalex", "epmc"] and pub["canonical_ref"] == ds.canonical_ref("https://c/z")


def test_domain_tallies():
    store = ds.SQLiteDiscoveryStore()
    rec = _rec([{"portal": "europepmc", "status": "completed", "results": 1}],
               [{"portal": "epmc", "doi": "10.3/q", "outcome": "reused", "terminal_reason": "admitted"}])
    dr.record(store, rec, registry=_reg(), domain="biomedicine")
    d = store.domain_tallies()
    assert d["biomedicine"]["unique"]["documents"] == 1
    assert d["biomedicine"]["by_portal"]["europepmc"]["relevance_qualified"] == 1


# ------------------------------------------------------------------------------------------- selection
def test_ddg_first_and_arxiv_never_default():
    reg = _reg()
    for seed in range(30):
        out = psel.select_portals(None, reg, {}, rng=random.Random(seed), adapter_check=ALL_ADAPTERS)
        assert out[0] == "ddg" and "arxiv" not in out and len(out) > 1
    phys = psel.select_portals("physics", reg, {}, rng=random.Random(1), adapter_check=ALL_ADAPTERS)
    assert phys[0] == "ddg" and "arxiv" in phys  # only on an explicit domain match


def test_non_arxiv_portal_selected_second():
    reg = _reg()
    stats = {k: {"events": {"completed": 50}, "deduplicated_documents": 50, "relevance_qualified": 20,
                 "valuable": v} for k, v in (("openalex", 10), ("europepmc", 2), ("doaj", 1), ("wikipedia", 0))}
    out = psel.select_portals(None, reg, stats, rng=random.Random(3), cfg={"selection": {**CFG["selection"],
                                                                                        "epsilon": 0.0}},
                              adapter_check=ALL_ADAPTERS)
    assert out[:2] == ["ddg", "openalex"]


def test_exploration_keeps_low_history_portal_selectable():
    reg = _reg()
    stats = {k: {"events": {"completed": 200}, "deduplicated_documents": 200, "valuable": 150}
             for k in ("openalex", "europepmc", "wikipedia")}
    stats["doaj"] = {"events": {"completed": 1}, "deduplicated_documents": 1, "valuable": 0}  # below min_trials
    seconds = [psel.select_portals(None, reg, stats, rng=random.Random(s), adapter_check=ALL_ADAPTERS)[1]
               for s in range(50)]
    assert seconds.count("doaj") >= 25
    stats["doaj"] = {"events": {"completed": 100}, "deduplicated_documents": 100, "valuable": 0}
    seconds = [psel.select_portals(None, reg, stats, rng=random.Random(s), adapter_check=ALL_ADAPTERS)[1]
               for s in range(400)]
    assert 0 < seconds.count("doaj") < 400 and seconds.count("openalex") < 400  # epsilon: no monopoly


def test_max_share_demotes_monopolist():
    reg = _reg()
    cfg = {"selection": {**CFG["selection"], "epsilon": 0.0, "min_trials": 0}}
    stats = {"openalex": {"events": {"completed": 10}, "deduplicated_documents": 10, "valuable": 10}}
    out = psel.select_portals(None, reg, stats, rng=random.Random(0), cfg=cfg, recent_first=["openalex"] * 10,
                              adapter_check=ALL_ADAPTERS)
    assert out[0] == "ddg" and out[1] != "openalex" and out[2] == "openalex"


# ------------------------------------------------------------------------------------------- no new network route
def test_registry_creates_no_fetchable_host():
    from vera import acq_config, acq_discovered
    acq = acq_config.load_config()
    wiki = json.loads((ROOT / "config" / "wiki_ref_portal.json").read_text(encoding="utf-8"))
    rl = json.loads((ROOT / "config" / "rate_limits.json").read_text(encoding="utf-8"))
    before = (acq_discovered.landing_hosts(acq), acq_discovered.fulltext_hosts(acq), set(wiki["allowed_hosts"]),
              set(rl["portals"]))
    reg = _reg()
    pr.statuses(reg)
    acq2 = acq_config.load_config()
    after = (acq_discovered.landing_hosts(acq2), acq_discovered.fulltext_hosts(acq2), set(wiki["allowed_hosts"]),
             set(json.loads((ROOT / "config" / "rate_limits.json").read_text(encoding="utf-8"))["portals"]))
    assert before == after
    imported_hosts = {p.host for p in reg.portals.values() if p.adapter is None}
    assert not imported_hosts & (after[0] | after[1])
    for mod in ("portal_registry", "portal_store", "portal_recorder", "portal_select"):
        src = (ROOT / "vera" / f"{mod}.py").read_text(encoding="utf-8")
        for bad in ("import requests", "urllib.request", "import socket", "http.client", "httpx"):
            assert bad not in src, (mod, bad)


def test_migration_draft_is_not_runnable_and_insert_only():
    mig = (ROOT / "db" / "migrations" / "pending" / "009-DRAFT-portal-discovery.sql.draft").read_text(encoding="utf-8")
    assert "NOT EXECUTED" in mig and "GRANT UPDATE" not in mig and "GRANT DELETE" not in mig
    for t in ("portal_domain", "discovery_event", "candidate_attribution", "document_outcome", "document_identity"):
        assert f"vera_vjay.{t}" in mig
    assert "content" not in mig.split("GRANT")[0].lower().replace("no document content", "")
    assert not list((ROOT / "db" / "migrations").glob("009_vera_*.sql"))


# ------------------------------------------------------------------------------------------- verify follow-ups F2, F3
def test_configured_reads_live_config_and_reports_source():
    reg = _reg()
    st = pr.statuses(reg)
    live = json.loads((ROOT / "config" / "auto_acquire.json").read_text(encoding="utf-8"))["search"]["providers"]
    for k in ("doaj", "arxiv"):
        assert st[k]["configured"] is (k in live), k  # eval-run config alone never makes a portal configured
        assert st[k]["eval_configured"] is True and st[k]["eval_configured_from"] == ["config/vera_eval_run.json"]
    assert st["ddg"]["configured_from"] == ["config/auto_acquire.json"]
    assert st["wikipedia"]["configured_from"] == ["config/wiki_ref_portal.json"]
    for k, p in reg.portals.items():
        assert all(pr_["file"] != "config/vera_eval_run.json" for pr_ in p.config_probes), k


def test_record_ask_is_queued_bounded_and_swallows(monkeypatch, tmp_path):
    import threading
    cfg = {**CFG, "store": {"enabled": True, "backend": "sqlite", "sqlite_path": ":memory:", "busy_timeout_s": 0.05,
                            "max_pending": 1}}
    monkeypatch.setattr(dr, "_registry_cfg", lambda: cfg)
    monkeypatch.setattr(dr, "_STORE_CFG", None)
    rec = _rec([{"portal": "ddg", "status": "completed", "results": 1}], [{"portal": "ddg", "doi": "10.5/a"}])
    dr.record_ask(rec, question="q")
    assert dr.drain(5.0)
    store = dr._STORE
    assert store.rows("SELECT COUNT(*) n FROM discovery_event")[0]["n"] == 1
    gate = threading.Event()
    assert dr._submit(gate.wait, 1)  # occupies the worker
    import time
    time.sleep(0.05)
    assert dr._submit(lambda: None, 1)  # fills the 1-slot queue
    t0 = time.monotonic()
    assert dr._submit(lambda: None, 1) is False  # full: dropped, not awaited
    assert time.monotonic() - t0 < 0.5
    gate.set()
    assert dr.drain(5.0)
    # a locked database fails fast (short busy timeout) instead of the sqlite 5 s default
    import sqlite3
    path = str(tmp_path / "locked.db")
    s1 = ds.SQLiteDiscoveryStore(path, busy_timeout_s=0.05)
    lock = sqlite3.connect(path, isolation_level=None)
    lock.execute("BEGIN EXCLUSIVE")
    t0 = time.monotonic()
    with pytest.raises(sqlite3.OperationalError):
        s1.add_event(request_id="r", portal_key="ddg", status="completed")
    assert time.monotonic() - t0 < 1.0
    lock.execute("ROLLBACK")
