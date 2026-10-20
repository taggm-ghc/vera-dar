"""VERA-DAR T2: domain-aware discovery routing (pure functions, DRAFT pre-stage).

Builds domain-shaped query branches PLUS the unchanged unrestricted fallback, fuses them rank-only (RRF, k read
from config), dedupes by DOI / normalized URL (same key as grounded_ask.retrieve_all), and returns a trace.

Hard rules (vet-B RC-1..RC-4, RC-8):
  * no SQL, no DB, no network imports; the search callable is INJECTED (this module never imports
    vera.search_and_fetch, which imports `requests`). vera.domain_registry (stdlib-only, B2) supplies routing_enabled/load_config/load_registry.
  * never filters cross-domain items, never touches admission or evidence gates.
  * flag OFF (config `enabled` is not True) -> the baseline results are returned by identity.
  * no module-level caches: config and registry are loaded per call or injected.
"""
import logging
import math
import re
import unicodedata
from pathlib import Path
from typing import Callable

from vera.domain_registry import CONFIG_PATH, load_config, load_registry, routing_enabled  # stdlib-only (B2)

logger = logging.getLogger("vera")

FALLBACK = "unrestricted"
LIMITATION = "discovery only; no stored corpus/chunk coverage"


# --- config / flag (loaded per call; RC-8). routing_enabled/load_config are B2's (stdlib-only), not duplicated ---
def flag_enabled(path: Path = CONFIG_PATH) -> bool:
    """Cheapest check, run BEFORE anything else is loaded. Missing/unreadable config means OFF."""
    return routing_enabled(load_config(path))


def get_cfg(path: Path = CONFIG_PATH) -> dict:
    return load_config(path)


def get_registry(cfg: dict | None = None):
    return load_registry(cfg if cfg is not None else load_config())


def make_search_fn(search_question: Callable, providers, cfg: dict) -> Callable[[str], list]:
    """Adapter for the hook: the same kwargs retrieve_all passes for the baseline call (cfg = grounding block).
    search_question is injected by the caller; nothing is imported here."""
    return lambda q: search_question(
        q, num_results=int(cfg["num_results"]), providers=providers, min_candidates=1, mode="fan_out",
        deadline_s=float(cfg["deadline_s"]), timeout_s=float(cfg.get("provider_timeout_s", 5)),
        max_retries=int(cfg.get("provider_max_retries", 1)))


# --- domain identification (deterministic term matching; no LLM) -----------------------------------------------
def _norm(text) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text or "")).casefold()).strip()


def identify_domains(question: str, registry, cfg: dict) -> dict:
    """Multi-label: EVERY registered domain whose name or configured match terms occur in the question.
    cfg["domains"] = [{"name": str, "terms": [str, ...]}, ...] (owner-supplied; empty by default)."""
    q = _norm(question)
    terms_by_name = {}
    for entry in cfg.get("domains") or []:
        if isinstance(entry, dict) and entry.get("name"):  # str entries (registry-only seeds) carry no extra terms
            terms_by_name[_norm(entry["name"])] = [t for t in entry.get("terms") or [] if _norm(t)]
    selected = []
    for d in registry.list_domains():
        cands = [d.name, *terms_by_name.get(_norm(d.name), [])]
        if any(re.search(r"(?<!\w)" + re.escape(_norm(c)) + r"(?!\w)", q) for c in cands if _norm(c)):
            selected.append(d)
    method = "terms" if selected else "none"
    # content-derived domains (DARINT P4): a registry that exposes identify_question(question) -> [domain_id]
    # (vera.dar_corpus sets it) adds domains identified from the stored content; additive, never replaces term matches.
    fn = getattr(registry, "identify_question", None)
    if callable(fn):
        have = {d.domain_id for d in selected}
        try:
            wanted = list(fn(question))
        except Exception as exc:  # noqa: BLE001
            logger.warning("DAR content identification skipped (%s)", type(exc).__name__)
            wanted = []
        by_id = {d.domain_id: d for d in registry.list_domains()}
        al = getattr(registry, "aliases", None)  # optional alias -> canonical mapping: an alias never makes a branch
        al = al if isinstance(al, dict) else {}
        wanted = [al.get(i, i) for i in wanted]
        added, seen = [], set(have)
        for i in wanted:
            if i in by_id and i not in seen and al.get(i, i) == i:
                seen.add(i)
                added.append(by_id[i])
        if added:
            selected += added
            method = "terms+content" if method == "terms" else "content"
    return {"selected": selected, "ambiguous": len(selected) > 1, "method": method}


def domain_query(question: str, domain, cfg: dict, registry=None) -> str:
    """Question plus the domain's query text. For a DERIVED domain whose registry exposes
    `domain_terms(domain_id, n)` (n = identification.name_terms) the query text is the domain's top profile terms (the
    derived name is only those terms in a fixed order, so this is the same vocabulary, content-chosen); otherwise (or on
    any failure / empty result) the domain NAME, as before."""
    template = cfg.get("query_template") or "{question} {domain}"
    text = domain.name
    fn = getattr(registry, "domain_terms", None)
    if callable(fn) and getattr(domain, "origin", None) == "derived":
        try:
            n = int((cfg.get("identification") or {}).get("name_terms") or 3)
            terms = [str(t) for t in (fn(domain.domain_id, n) or []) if str(t).strip()]
            if terms:
                text = " ".join(terms)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DAR domain terms skipped (%s)", type(exc).__name__)
    return template.format(question=question, domain=text).strip()


# --- fusion ----------------------------------------------------------------------------------------------------
def _key(it: dict):
    """Same key as grounded_ask.retrieve_all: DOI lowercased, else URL lowercased without trailing slash."""
    return str(it.get("doi") or "").lower() or str(it.get("url") or "").lower().rstrip("/") or None


def _rrf(branches: list[tuple[str, list[dict]]], k: int, key_fn: Callable[[dict], object] | None = None) -> tuple[list[dict], int]:
    """Rank-only RRF (score = sum 1/(k+rank)); scores of different branches are never compared.
    Returns (groups, dedup_count). A group = {"rep","score","best","branches","key"}; branches[0] is the unrestricted
    branch when present, so on a collision its dict is kept (RC-3), else the best-ranked branch's dict.
    key_fn (optional, default None = the DOI/URL _key, behaviour unchanged) lets chunk-level callers key on span_id (DARINT R1)."""
    keyf = key_fn or _key
    groups, index, dups = [], {}, 0
    for order, (name, items) in enumerate(branches):
        seen_in_branch = set()
        for rank, it in enumerate(items, start=1):
            key = keyf(it)
            if key is not None:
                if key in seen_in_branch:
                    continue  # a repeat inside one branch scores once (first, best-ranked occurrence)
                seen_in_branch.add(key)
            gi = index.get(key) if key is not None else None
            if gi is None:
                gi = len(groups)
                groups.append({"rep": it, "score": 0.0, "best": (rank, order), "branches": [], "key": key})
                if key is not None:
                    index[key] = gi
            else:
                dups += 1  # first occurrence in this branch of an item already seen in an earlier branch
            g = groups[gi]
            g["score"] += 1.0 / (k + rank)
            if name not in g["branches"]:
                g["branches"].append(name)
            if name != FALLBACK and FALLBACK not in g["branches"] and (rank, order) < g["best"]:
                g["rep"], g["best"] = it, (rank, order)  # best-ranked domain dict, only until the fallback claims it
    return groups, dups


def _reserve_count(cfg: dict, base_n: int, num_results: int | None) -> int:
    """Slots reserved for the unrestricted fallback. Configured value (experimental; calibration note required before
    enabling), else max(2, ceil(n/2)). Clamped so that at least one slot remains for domain items when n >= 2, and
    never below 1 (domain results cannot displace every unrestricted result)."""
    n = num_results or base_n or 1
    want = int(cfg["min_fallback_slots"]) if cfg.get("min_fallback_slots") is not None else max(2, math.ceil(n / 2))
    return max(1, min(want, n - 1 if n >= 2 else 1))


def route_search(question: str, base_results: list[dict], search_fn: Callable[[str], list[dict]], registry, cfg: dict,
                 num_results: int | None = None) -> tuple[list[dict], dict]:
    """Fail-safe wrapper: any routing error returns base_results unchanged (class name logged only)."""
    if not routing_enabled(cfg):
        return base_results, {"routing": "off"}
    try:
        return _route(question, base_results, search_fn, registry, cfg, num_results)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DAR routing error (%s)", type(exc).__name__)
        return base_results, {"routing": "error", "error_class": type(exc).__name__}


def _route(question: str, base_results: list[dict], search_fn: Callable[[str], list[dict]], registry, cfg: dict,
           num_results: int | None = None) -> tuple[list[dict], dict]:
    """(results, trace). base_results is the unchanged unrestricted call, already run by the caller (budgeted
    independently, includes unclassified assets). OFF / no selected domain / no usable domain branch -> base_results
    returned by identity. ON -> reserved fallback items first (RC-1), then the rest by RRF."""
    if not routing_enabled(cfg):
        return base_results, {"routing": "off"}
    try:
        k = int(cfg["rrf_k"])  # same meaning as search.rrf_k; read from config, never hard-coded
        if k < 1:
            raise ValueError("rrf_k must be >= 1")
    except (KeyError, TypeError, ValueError):
        return base_results, {"routing": "config_error", "detail": "rrf_k missing or invalid"}
    ident = identify_domains(question, registry, cfg)
    selected = list(ident["selected"])
    if not selected:
        return base_results, {"routing": "no_domain", "limitation": LIMITATION}
    cap = cfg.get("max_domain_branches")
    searched = selected[: int(cap)] if cap is not None else selected
    branches: list[tuple[str, list[dict]]] = [(FALLBACK, list(base_results))]
    counts, failed = {FALLBACK: len(base_results)}, []
    for d in searched:
        name = f"domain:{d.domain_id}"
        try:
            items = [r for r in (search_fn(domain_query(question, d, cfg, registry)) or []) if isinstance(r, dict)]
        except Exception as exc:  # noqa: BLE001  (a failed branch is dropped; the fallback always stays)
            logger.warning("DAR branch %s failed (%s)", name, type(exc).__name__)
            failed.append(name)
            continue
        counts[name] = len(items)
        branches.append((name, items))
    trace = {"routing": "on", "selected": [d.domain_id for d in selected],
             "searched": [d.domain_id for d in searched], "branch_counts": counts, "failed_branches": failed,
             "limitation": LIMITATION}
    if len(branches) == 1 or all(counts[n] == 0 for n, _ in branches[1:]):
        trace.update(fallback_contribution=len(base_results), fallback_only=len(base_results), dedup_count=0,
                     reserved=0)
        return base_results, trace  # nothing from any domain branch: exactly the baseline
    groups, dups = _rrf(branches, k)
    reserve_n = _reserve_count(cfg, len(base_results), num_results)
    base_groups = sorted((g for g in groups if FALLBACK in g["branches"]), key=lambda g: g["best"])
    reserved = base_groups[:reserve_n]
    reserved_ids = {id(g) for g in reserved}
    rest = sorted((g for g in groups if id(g) not in reserved_ids), key=lambda g: (-g["score"], g["best"]))
    out = []
    for g in reserved + rest:
        rep = dict(g["rep"])  # shallow copy: the source dict is never mutated, fields never merged (RC-3)
        rep["_domain_branches"] = list(g["branches"])
        out.append(rep)
    sch = cfg.get("scholarly_source_types")
    if sch:
        trace["all_non_scholarly_branches"] = [
            n for n, its in branches[1:] if its and not any(i.get("source_type") in sch for i in its)]
    trace.update(fallback_contribution=sum(1 for g in groups if FALLBACK in g["branches"]),
                 fallback_only=sum(1 for g in groups if g["branches"] == [FALLBACK]),
                 dedup_count=dups, reserved=len(reserved))
    return out, trace
