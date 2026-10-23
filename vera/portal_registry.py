"""PORT-P2: research portal registry (owner requirement: portal registry and discovery performance).

Reads config/portal_registry.json (seed portals: the existing VERA-DAR providers) plus its `imports` (the owner's
authoritative-sources list, PORT-P1). Pure metadata: this module performs NO network I/O and never adds a fetchable
host; fetch allowlists stay where they are (config/auto_acquire.json landing_route, config/wiki_ref_portal.json).

Separate concepts kept separate: research subject domain (Portal.domains, normalized with
vera.domain_registry.normalize_domain_name), portal/website hostname (Portal.host), discovery portal (Portal.key),
acquisition endpoint (Portal.acquisition), source publisher and document identity (vera.portal_store).

Status levels (computed, never stored; a listed URL alone never makes a portal operational):
  registered  - listed in the seed or an import
  supported   - its adapter (module + attribute) exists in code
  configured  - supported AND every config probe passes AND enabled
  operational - configured AND health: a completed search within health.max_success_age_hours and fewer than
                health.max_consecutive_failures non-completed searches since that success
"""
from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "portal_registry.json"
LEVELS = ("registered", "supported", "configured", "operational")
CATEGORIES = ("search_discovery_portal", "scholarly_index", "institutional_repository", "publisher_journal_platform",
              "government_research_institution", "reference_resource_not_searchable", "other")


def _norm_domain(name: str) -> str:
    from vera.domain_registry import normalize_domain_name
    return normalize_domain_name(name)


def normalize_url(url: str) -> str:
    """Identity form of a portal URL: lower-case scheme/host, no default port, no trailing slash, query kept."""
    sp = urlsplit(str(url or "").strip())
    host = (sp.hostname or "").lower()
    path = (sp.path or "").rstrip("/")
    return f"{(sp.scheme or 'https').lower()}://{host}{path}" + (f"?{sp.query}" if sp.query else "")


def portal_key_for_url(url: str) -> str:
    """Stable key for an imported portal: 'web:' + host + path, non-alphanumerics folded to '-'."""
    n = normalize_url(url).split("://", 1)[-1]
    return "web:" + re.sub(r"[^a-z0-9.]+", "-", n.lower()).strip("-")


@dataclass
class Portal:
    key: str
    name: str
    canonical_url: str
    category: str
    domains: tuple = ()
    general: bool = False
    enabled: bool = True
    default_routing: bool = False
    aliases: tuple = ()
    adapter: dict | None = None
    acquisition: str | None = None
    formats: tuple = ()
    identifiers: tuple = ()
    access: str = "unknown"
    auth: str = "unknown"
    licence_notes: str = ""
    security_notes: str = ""
    cost_basis: str | None = None
    config_probes: tuple = ()
    eval_probes: tuple = ()  # evaluation-run config only (reported as eval_configured; never makes a portal configured)
    related: tuple = ()
    provenance: tuple = ()  # every registry source that asserted this portal (seed and/or owner import)
    verified_at: str | None = None
    owner_entries: tuple = ()  # raw owner-page rows merged into this portal (section, position, confidence)

    @property
    def host(self) -> str:
        return (urlsplit(self.canonical_url).hostname or "").lower()


@dataclass
class Registry:
    portals: dict = field(default_factory=dict)  # key -> Portal
    alias: dict = field(default_factory=dict)  # alias/key -> key
    cfg: dict = field(default_factory=dict)

    def get(self, name: str) -> Portal | None:
        k = self.alias.get(str(name or "").strip().lower())
        return self.portals.get(k) if k else None

    def resolve_key(self, name: str) -> str:
        """Canonical portal key for a provider/record name; unknown names are kept verbatim (never dropped)."""
        p = self.get(name)
        return p.key if p else str(name or "unknown")

    def register(self, p: Portal) -> tuple[Portal, bool]:
        """Add or merge (idempotent): a portal with the same key or the same canonical URL is merged, not duplicated.
        Returns (portal, created)."""
        existing = self.portals.get(p.key) or next(
            (q for q in self.portals.values() if normalize_url(q.canonical_url) == normalize_url(p.canonical_url)), None)
        if existing is None:
            self.portals[p.key] = p
            for a in (p.key, *p.aliases):
                self.alias.setdefault(str(a).lower(), p.key)
            return p, True
        existing.domains = tuple(dict.fromkeys(existing.domains + p.domains))
        existing.provenance = tuple(dict.fromkeys(existing.provenance + p.provenance))
        existing.owner_entries = existing.owner_entries + p.owner_entries
        existing.aliases = tuple(dict.fromkeys(existing.aliases + p.aliases))
        for a in p.aliases:
            self.alias.setdefault(str(a).lower(), existing.key)
        return existing, False

    def by_domain(self, domain: str | None) -> list:
        """Portals serving a domain: declared domain match, or general portals (any domain)."""
        d = _norm_domain(domain) if domain else None
        return [p for p in self.portals.values() if p.general or (d is not None and d in p.domains)]


def _portal_from_seed(e: Mapping, provenance: str, verified_at: str | None) -> Portal:
    cat = e.get("category") if e.get("category") in CATEGORIES else "other"
    return Portal(key=str(e["key"]).lower(), name=str(e["name"]), canonical_url=str(e["canonical_url"]), category=cat,
                  domains=tuple(_norm_domain(d) for d in e.get("domains") or ()), general=e.get("general") is True,
                  enabled=e.get("enabled") is True, default_routing=e.get("default_routing") is True,
                  aliases=tuple(str(a).lower() for a in e.get("aliases") or ()),
                  adapter=dict(e["adapter"]) if isinstance(e.get("adapter"), dict) else None,
                  acquisition=e.get("acquisition"), formats=tuple(e.get("formats") or ()),
                  identifiers=tuple(e.get("identifiers") or ()), access=str(e.get("access") or "unknown"),
                  auth=str(e.get("auth") or "unknown"), licence_notes=str(e.get("licence_notes") or ""),
                  security_notes=str(e.get("security_notes") or ""), cost_basis=e.get("cost_basis"),
                  config_probes=tuple(dict(p) for p in e.get("config_probes") or ()),
                  eval_probes=tuple(dict(p) for p in e.get("eval_probes") or ()),
                  related=tuple(dict(r) for r in e.get("related") or ()), provenance=(provenance,),
                  verified_at=verified_at)


def import_owner_sources(reg: Registry, data: Mapping, provenance: str) -> dict:
    """PORT-P1 import (schema: {provenance, entries:[{name,url,host,page_section,page_position,category,confidence,
    rationale,existing_vera_portal}]}). An entry naming an existing VERA portal is merged into it (owner provenance
    and domain added); every other entry is REGISTERED only: no adapter, no config probes, enabled false, so it can
    never reach configured/operational. page_section is the owner's subject grouping, recorded as its domain.
    Returns counts {merged, created, duplicates, by_category}."""
    meta = data.get("provenance") or {}
    when = meta.get("extracted_at")
    stats = {"merged": 0, "created": 0, "duplicates": 0, "by_category": {}}
    for e in data.get("entries") or ():
        if not isinstance(e, Mapping) or not e.get("url"):
            continue
        cat = e.get("category") if e.get("category") in CATEGORIES else "other"
        stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
        dom = (_norm_domain(e["page_section"]),) if e.get("page_section") else ()
        row = ({"section": e.get("page_section"), "position": e.get("page_position"), "category": cat,
                "confidence": e.get("confidence"), "name": e.get("name"), "url": e.get("url")},)
        target = reg.portals.get(str(e.get("existing_vera_portal") or "").lower())
        if target is not None:
            target.domains = tuple(dict.fromkeys(target.domains + dom))
            target.provenance = tuple(dict.fromkeys(target.provenance + (provenance,)))
            target.owner_entries = target.owner_entries + row
            stats["merged"] += 1
            continue
        p = Portal(key=portal_key_for_url(e["url"]), name=str(e.get("name") or e["url"]), canonical_url=str(e["url"]),
                   category=cat, domains=dom, enabled=False, default_routing=False, adapter=None,
                   access="unknown", auth="unknown",
                   licence_notes="not assessed" if cat != "reference_resource_not_searchable" else "reference only",
                   provenance=(provenance,), verified_at=when, owner_entries=row)
        _, created = reg.register(p)
        stats["created" if created else "duplicates"] += 1
    return stats


def load_registry(cfg: Mapping | None = None, root: Path | None = None) -> Registry:
    """Seed portals plus every configured import. Never raises for a missing import file (it is skipped)."""
    root = root or ROOT
    if cfg is None:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    reg = Registry(cfg=dict(cfg))
    prov = cfg.get("provenance") or {}
    for e in cfg.get("portals") or ():
        reg.register(_portal_from_seed(e, "seed:" + str(prov.get("seed") or "vera-dar"), prov.get("verified_at")))
    for imp in cfg.get("imports") or ():
        path = root / str(imp.get("path"))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        import_owner_sources(reg, data, "import:" + str(imp.get("provenance") or imp.get("path")))
    return reg


# ---------------------------------------------------------------------------------------------- status levels
def _dig(obj, path: str):
    for part in path.split("."):
        if not isinstance(obj, Mapping) or part not in obj:
            return None
        obj = obj[part]
    return obj


def probe_passes(probe: Mapping, read_json: Callable[[str], dict]) -> bool:
    try:
        v = _dig(read_json(str(probe["file"])), str(probe["path"]))
    except Exception:  # noqa: BLE001  unreadable config = not configured (fail closed)
        return False
    if "equals" in probe:
        return v == probe["equals"]
    if "contains" in probe:
        return isinstance(v, (list, tuple)) and probe["contains"] in v
    return False


def adapter_exists(adapter: Mapping | None) -> bool:
    if not adapter:
        return False
    try:
        return callable(getattr(importlib.import_module(str(adapter["module"])), str(adapter["attr"]), None))
    except Exception:  # noqa: BLE001
        return False


def _default_read_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def status(p: Portal, *, health: Mapping | None = None, cfg: Mapping | None = None,
           read_json: Callable[[str], dict] | None = None, adapter_check: Callable | None = None,
           now_epoch: float | None = None) -> dict:
    """{"level", "registered", "supported", "configured", "operational", "reasons"}. health is the per-portal row of
    DiscoveryStore.health(): {last_success_at, last_attempt_at, failures_since_success} (epoch seconds)."""
    import time
    hc = (cfg or {}).get("health") or {}
    reasons = []
    supported = (adapter_check or adapter_exists)(p.adapter)
    if not supported:
        reasons.append("no adapter in code" if p.adapter is None else "adapter not found")
    configured = supported and p.enabled and bool(p.config_probes) and all(
        probe_passes(pr, read_json or _default_read_json) for pr in p.config_probes)
    if supported and not configured:
        reasons.append("disabled or config probe failed")
    operational = False
    if configured:
        h = health or {}
        last = h.get("last_success_at")
        age_ok = last is not None and ((now_epoch if now_epoch is not None else time.time()) - float(last)) <= \
            float(hc.get("max_success_age_hours", 0)) * 3600
        fail_ok = int(h.get("failures_since_success") or 0) < int(hc.get("max_consecutive_failures", 1))
        operational = age_ok and fail_ok
        if not operational:
            reasons.append("no recent completed search" if not age_ok else "recent consecutive failures")
    rj = read_json or _default_read_json
    eval_configured = bool(p.eval_probes) and all(probe_passes(pr, rj) for pr in p.eval_probes)
    level = "operational" if operational else "configured" if configured else "supported" if supported else "registered"
    return {"level": level, "registered": True, "supported": supported, "configured": configured,
            "operational": operational, "reasons": reasons,
            "configured_from": sorted({str(pr.get("file")) for pr in p.config_probes}) if configured else [],
            "eval_configured": eval_configured,
            "eval_configured_from": sorted({str(pr.get("file")) for pr in p.eval_probes}) if eval_configured else []}


def statuses(reg: Registry, health_by_key: Mapping | None = None, **kw) -> dict:
    return {k: status(p, health=(health_by_key or {}).get(k), cfg=reg.cfg, **kw) for k, p in reg.portals.items()}


def summary(reg: Registry, st: Mapping) -> dict:
    by_level = {lv: 0 for lv in LEVELS}
    by_cat: dict = {}
    for k, p in reg.portals.items():
        by_level[st[k]["level"]] += 1
        by_cat[p.category] = by_cat.get(p.category, 0) + 1
    return {"total": len(reg.portals), "by_level": by_level, "by_category": by_cat}


def domain_pairs(reg: Registry) -> Iterable[tuple]:
    """(portal_key, domain_name_norm) rows for the many-to-many portal_domain table."""
    for p in reg.portals.values():
        for d in p.domains:
            yield p.key, d
