"""PORT-P2: portal selection input for routing (pure function; NOT wired into the live chain - the routing owner
integrates it at vera.search_providers.acquisition_search, where the portal set is decided).

select_portals(domain, registry, stats, *, rng) -> ordered portal keys:
  1. DuckDuckGo first, always (owner 2026-10-22), unless administratively disabled.
  2. Eligible others: enabled, supported adapter, default_routing true (arXiv is never default) or an explicit
     declared-domain match, and serving the domain (declared domain or general).
  3. Ranked by weighted history (valuable_rate, relevant_rate, success_rate; config selection.weights) plus a UCB
     exploration bonus; portals with fewer than min_trials attempts are explored first (random order); with
     probability epsilon a random eligible portal is promoted to the first slot after DDG; a portal holding that slot
     in more than max_share of recent selections is moved down one place. All parameters are config (uncalibrated).
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from vera.portal_store import rates
from vera.portal_registry import Registry, _norm_domain, adapter_exists


def eligible(domain: str | None, registry: Registry, *, adapter_check=adapter_exists) -> list:
    d = _norm_domain(domain) if domain else None
    out = []
    for p in registry.portals.values():
        if not p.enabled or not adapter_check(p.adapter):
            continue
        domain_match = d is not None and d in p.domains
        if not (p.default_routing or domain_match):
            continue  # e.g. arXiv: only on an explicit domain match, never by default
        if p.general or domain_match:
            out.append(p.key)
    return out


def select_portals(domain: str | None, registry: Registry, stats: Mapping | None, *, rng,
                   recent_first: Sequence[str] = (), cfg: Mapping | None = None, adapter_check=adapter_exists) -> list:
    """stats: DiscoveryStore.portal_tallies() (per portal). recent_first: the portal placed right after DDG in recent
    selections (for the max_share cap). rng: a random.Random (injected for determinism)."""
    sc = dict((cfg or registry.cfg).get("selection") or {})
    first = str(sc.get("first") or "ddg")
    keys = eligible(domain, registry, adapter_check=adapter_check)
    head = [first] if first in registry.portals and registry.portals[first].enabled else []
    others = [k for k in keys if k != first]
    if not others:
        return head
    w = sc.get("weights") or {}
    r = {k: rates((stats or {}).get(k) or {}) for k in others}
    total = sum(x["attempts"] for x in r.values()) or 1
    c = float(sc.get("ucb_c", 0.0))
    min_trials = int(sc.get("min_trials", 0))

    def score(k):
        x = r[k]
        base = sum(float(w.get(m, 0.0)) * x[m] for m in ("valuable_rate", "relevant_rate", "success_rate"))
        return base + c * math.sqrt(math.log(total + 1) / (x["attempts"] + 1))

    fresh = [k for k in others if r[k]["attempts"] < min_trials]
    rng.shuffle(fresh)
    seasoned = sorted((k for k in others if k not in fresh), key=lambda k: (-score(k), k))
    ranked = fresh + seasoned
    if rng.random() < float(sc.get("epsilon", 0.0)):
        pick = rng.choice(ranked)
        ranked.remove(pick)
        ranked.insert(0, pick)
    if recent_first and len(ranked) > 1:
        share = sum(1 for x in recent_first if x == ranked[0]) / len(recent_first)
        if share > float(sc.get("max_share", 1.0)):
            ranked[0], ranked[1] = ranked[1], ranked[0]
    return head + ranked
