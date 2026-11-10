"""OA-4 (OA37-1108, owner OD-1108-HITL D3): the full-text acquisition route registry.

One FulltextRoute per way the acquisition worker can turn a candidate into stored full text. A new source (OA-5: PLOS
JATS) adds one route here instead of copying the Europe PMC branch of auto_acquire.evaluate_candidate.

This first slice is a behaviour-preserving refactor (characterised by tests/test_oa4_fulltext_routes_1108.py): the two
existing routes, arXiv (vera/arxiv_fulltext.py) and Europe PMC (auto_acquire._evaluate_europepmc), keep their code,
order, configuration keys and reason codes. Nothing here fetches: every request still goes through
auto_acquire._guarded_get (robots/TDM gate, host allowlist, rate limit), called by the route bodies.

- `api_permission_id` names the API-permission entry the route needs (P0-2, API-ALLOW-1107, not yet shipped). It is
  declared only; enforcement (skip with R-API-DENIED) arrives with that allowlist. None = the route reads website
  content (arXiv abs/html/pdf pages), which the robots/TDM gate governs, not an API entry.
- Route bodies are looked up at call time (module attributes), so tests that monkeypatch
  arxiv_fulltext.evaluate or auto_acquire.resolve_oa keep working.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class FulltextRoute:
    name: str
    api_permission_id: str | None
    prepare: Callable  # (cand, doi, cfg) -> (cand, doi): metadata-only normalisation before the claim test
    claims: Callable  # (cand, cfg) -> bool: this route takes the (prepared) candidate
    evaluate: Callable  # (job, score, doi, cand, isbn, cfg, deps, icfg, fdcfg, clock, deadline, dur, min_spans)
    routable: Callable  # (cand, cfg) -> bool: metadata OA signal for ranking; never a request


# ------------------------------------------------------------------------------------------------ arXiv
def _arxiv_prepare(cand: dict, doi, cfg: dict):
    """INTEG-1023: an arXiv hit from the acquisition search takes the arXiv route (search_candidate); the candidate's own
    DOI is adopted, as before, whichever route then claims it."""
    from vera import arxiv_fulltext
    if not isinstance(cand.get("arxiv"), dict):
        cand = arxiv_fulltext.search_candidate(cand, doi, cfg) or cand
        doi = cand.get("doi") or doi
    return cand, doi


def _arxiv_claims(cand: dict, cfg: dict) -> bool:
    return isinstance(cand.get("arxiv"), dict)  # ARX-A2


def _arxiv_evaluate(*args):
    from vera import arxiv_fulltext
    return arxiv_fulltext.evaluate(*args)


def _arxiv_routable(c: dict, cfg: dict) -> bool:
    try:
        from vera import arxiv_fulltext
        from vera.auto_acquire import _doi
        return isinstance(c.get("arxiv"), dict) or arxiv_fulltext.search_candidate(c, _doi(c), cfg) is not None
    except Exception:  # noqa: BLE001  (unchanged: any error means "not routable")
        return False


# ------------------------------------------------------------------------------------------------ Europe PMC
def _same(cand: dict, doi, cfg: dict):
    return cand, doi


def _always(cand: dict, cfg: dict) -> bool:
    return True  # the default route: every candidate no other route claims (unchanged behaviour)


def _europepmc_evaluate(*args):
    from vera import auto_acquire
    return auto_acquire._evaluate_europepmc(*args)


def _europepmc_routable(c: dict, cfg: dict) -> bool:
    """Europe PMC full text (oa.epmc_ft) or a PMCID, or the OA metadata's is_oa / has_oa_location True (a DOI the
    Europe PMC resolver may find open)."""
    from vera.auto_acquire import _pmcid
    oa = c.get("oa") if isinstance(c.get("oa"), dict) else {}
    return bool(oa.get("epmc_ft") or _pmcid(c) or oa.get("is_oa") is True or oa.get("has_oa_location") is True)


ARXIV = FulltextRoute("arxiv", None, _arxiv_prepare, _arxiv_claims, _arxiv_evaluate, _arxiv_routable)
EUROPEPMC = FulltextRoute("europepmc", "europepmc_rest", _same, _always, _europepmc_evaluate, _europepmc_routable)

ROUTES: tuple = (ARXIV, EUROPEPMC)  # dispatch order; the last route is the default and claims everything
ROUTABLE_ORDER: tuple = (EUROPEPMC, ARXIV)  # routable_fulltext's historical check order (short-circuit)


def by_name(name: str) -> FulltextRoute | None:
    return next((r for r in ROUTES if r.name == name), None)


def select(cand: dict, doi, cfg: dict):
    """(route, cand, doi) for a candidate already past the known check. Metadata only; makes no request."""
    for r in ROUTES:
        cand, doi = r.prepare(cand, doi, cfg)
        if r.claims(cand, cfg):
            return r, cand, doi
    raise LookupError("no full-text route claims the candidate")  # unreachable: the default route claims all


def routable(c: dict, cfg: dict) -> bool:
    """True when any route's metadata signal says it could fetch full text for c (already URL-followed)."""
    return any(r.routable(c, cfg) for r in ROUTABLE_ORDER)
