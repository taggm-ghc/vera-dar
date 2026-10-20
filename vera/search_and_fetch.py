"""M2 search: question -> ranked, de-duplicated candidate URLs, through a provider chain.

Providers live in vera.search_providers (arXiv first, a hash-frozen seed list as a DISCLOSED fallback). The
chain and its limits come from config/vera_eval_run.json `search`; nothing here needs an API key.

Contract kept for run_m2 / run_mvp: `search_question(question, num_results, *, ledger)` returns
list[dict] with url, title, snippet, rank, provider (plus optional doi, published, licence, ...). On failure
it raises SearchError carrying every attempt's outcome: "search failed" is never returned as "nothing found".
The returned list is a `SearchResults` (a list) whose `.meta` records the route taken.

Routing: a provider that SUCCEEDS with at least `min_candidates` results wins. A failure, or fewer results,
advances to the next provider. If no provider reaches the minimum, the largest result is returned with
meta["below_minimum"]=True (disclosed, never silent). A seed-file provider is used only when the earlier
routes failed or fell short; its use is recorded in meta["fallback_used"].
"""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import os

import requests

from vera.cost_ledger import CostLedger
from vera.search_providers import (ProviderResponseError, ProviderUnavailable, SearchConfigError as _SCE,
                                    build_chain)

# Why: discovery providers (DuckDuckGo Lite) must not crowd out scholarly results in the final top-n. Used when the
# config has no search.max_share and for injected providers.
DEFAULT_MAX_SHARE = {"duckduckgo_lite": 2}
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_ROOT = Path(__file__).resolve().parent.parent


class SearchError(RuntimeError):
    pass


class SearchConfigError(_SCE, SearchError):
    """Chain config invalid/unsafe; still a SearchError so existing callers keep working."""


class SearchResults(list):
    """list[dict] plus `.meta` (route record for the run report)."""
    meta: dict

    def __init__(self, items=(), meta=None):
        super().__init__(items)
        self.meta = meta or {}


class _ProviderFailed(Exception):
    def __init__(self, notes):
        super().__init__("; ".join(notes))
        self.notes = notes


def normalize_url(url: str) -> str:
    """Canonical form for de-duplication: lowercase host, no fragment, no trailing slash."""
    p = urlsplit(url.strip())
    path = p.path.rstrip("/") or "/"
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), path, p.query, ""))


def _attempts(provider, question, n, **kw):
    """One provider call with bounded retries; R72-e: if an arXiv/OpenAlex AND query returns ZERO results, retry
    ONCE with the core phrases ORed, through the same gate and deadline. `provider.last_relaxed` records it.
    A failed retry is not a failure of the provider (the first call succeeded): it is recorded and [] returned."""
    provider.last_relaxed = False
    provider.last_relax_error = None
    items = _attempts_once(provider, question, n, **kw)
    can = getattr(provider, "can_relax", None)
    if items or not (can and can(question)):
        return items
    provider.relaxed = True
    provider.last_relaxed = True
    try:
        return _attempts_once(provider, question, n, **kw)
    except _ProviderFailed as f:
        provider.last_relax_error = "; ".join(f.notes)
        return []
    finally:
        provider.relaxed = False


def _attempts_once(provider, question, n, *, ledger, max_retries, backoff_s, deadline_at, timeout_s, sleep, clock):
    notes: list[str] = []
    basis = provider.failure_basis()

    def rec(**kw):
        if ledger:
            ledger.record(kind="search", provider=provider.name, cost_basis=basis, **kw)

    for attempt in range(1, max_retries + 1):
        with provider.gate.lock:  # single connection
            # a provider may say a call will be served from its own cache (no request): it then skips the gate
            exempt = getattr(provider, "gate_exempt", None)
            wait = 0.0 if (exempt and exempt(question)) else provider.gate.wait_needed(clock())
            if clock() + wait > deadline_at:
                notes.append(f"attempt {attempt}: deadline exceeded before call")
                break
            if wait > 0:
                sleep(wait)
            try:
                resp = provider.call(question, n, timeout_s)
            except _SCE:
                raise
            except ProviderUnavailable as exc:  # non-retryable (e.g. daily budget spent): recorded, chain skips
                notes.append(f"attempt {attempt}: provider unavailable: {exc}")
                rec(cost_usd=0.0, ok=False, detail=notes[-1], units={"attempt": attempt})
                raise _ProviderFailed(notes) from None
            except requests.RequestException as exc:
                provider.gate.stamp(clock())
                notes.append(f"attempt {attempt}: network error {type(exc).__name__}: {exc}")
                rec(cost_usd=0.0, ok=False, detail=notes[-1], units={"attempt": attempt})
                resp, retry_after = None, None
            else:
                if not getattr(provider, "last_call_cached", False):  # cache-only calls leave the gate untouched
                    provider.gate.stamp(clock())
        if resp is not None:
            if resp.status_code == 200:
                try:
                    items = provider.parse(resp, n)
                except ProviderResponseError as e:
                    notes.append(f"attempt {attempt}: unusable response: {e}")
                    rec(cost_usd=0.0, ok=False, detail=notes[-1], units={"attempt": attempt})
                    break  # a malformed/error document is not retried
                usd, known, cb = provider.cost_for(resp)
                if ledger:
                    ledger.record(kind="search", provider=provider.name, cost_usd=usd, cost_known=known,
                                  cost_basis=cb, units={"attempt": attempt, "requested": n})
                return items
            notes.append(f"attempt {attempt}: HTTP {resp.status_code} {resp.text[:200]!r}")
            rec(cost_usd=0.0, ok=False, detail=notes[-1], units={"attempt": attempt})
            if resp.status_code not in RETRYABLE_STATUS:
                break
            try:
                retry_after = float(resp.headers.get("Retry-After", ""))
            except ValueError:
                retry_after = None
        wait_s = retry_after if retry_after is not None else backoff_s * 2 ** (attempt - 1)
        if attempt < max_retries:
            sleep(min(wait_s, max(0.0, deadline_at - clock())))
    raise _ProviderFailed(notes)


_ARXIV_ID = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|10\.48550/arxiv\.)((?:\d{4}\.\d{4,5})|(?:[a-z\-]+(?:\.[a-z]{2})?/\d{7}))(?:v\d+)?", re.I)
# Why: a one-letter or generic title must never merge two different records; real paper titles are longer.
MIN_TITLE_KEY_CHARS = 12


def _arxiv_id(it) -> str | None:
    for field in ("url", "doi"):
        m = _ARXIV_ID.search(str(it.get(field) or ""))
        if m:
            return m.group(1).lower()
    return None


def _norm_doi(doi) -> str | None:
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", (doi or "").strip().lower())
    return d or None


def _norm_title(title) -> str:
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def _year(it) -> int | None:
    y = it.get("year")
    if isinstance(y, int):
        return y
    m = re.match(r"(\d{4})", str(it.get("published") or ""))
    return int(m.group(1)) if m else None


def _first_surname(it) -> str | None:
    a = it.get("authors")
    first = (a[0] if isinstance(a, (list, tuple)) and a else a if isinstance(a, str) else "") or ""
    first = str(first).split(";")[0].strip()
    if not first:
        return None
    sur = first.split(",")[0] if "," in first else first.split()[-1]
    return re.sub(r"[^a-z0-9]", "", sur.lower()) or None


def _title_family(title) -> str | None:
    """Normalised title up to the first ':' (R72-d). Only a long-enough prefix counts, and only when there IS a
    subtitle: a generic or one-letter prefix must never merge two different records."""
    t = (title or "")
    if ":" not in t:
        return None
    base = _norm_title(t.split(":", 1)[0])
    return base if len(base) >= MIN_FAMILY_CHARS else None


# Why: R72-d. A family key (title before ':' + same year) is fixed here BEFORE testing; 25 normalised characters is
# roughly four words, so short generic prefixes ("Code review", "A survey") never form a family.
MIN_FAMILY_CHARS = 25


def _dedupe_keys(it) -> list[str]:
    """Hard identifiers only (url, doi, arXiv id). Title-based matching lives in `_Deduper` because it depends on
    year and author."""
    keys = []
    url = (it.get("url") or "").strip()
    if url:
        keys.append("url:" + normalize_url(url))
    d = _norm_doi(it.get("doi"))
    if d:
        keys.append("doi:" + d)
    aid = _arxiv_id(it)
    if aid:
        keys.append("arxiv:" + aid)
    return keys


class _Deduper:
    """Equal-record finder for _finalize and rrf_merge. Hard keys (url, DOI, arXiv id) merge unconditionally.
    Title matches need a compatible year (equal, or unknown on either side) and no conflicting first-author
    surname; the family key (title before ':') additionally needs BOTH years known and equal. `find` returns
    (index, reason) so a merge is always attributable (R72-d). No false-merge rate is claimed (unverified)."""

    def __init__(self):
        self.hard, self.titles, self.family = {}, {}, {}

    @staticmethod
    def _authors_ok(a, b) -> bool:
        return a is None or b is None or a == b

    def find(self, it):
        for k in _dedupe_keys(it):
            if k in self.hard:
                return self.hard[k], k.split(":", 1)[0]
        year, sur = _year(it), _first_surname(it)
        t = _norm_title(it.get("title"))
        if len(t) >= MIN_TITLE_KEY_CHARS:
            for idx, y, s in self.titles.get(t, []):
                if (year is None or y is None or year == y) and self._authors_ok(sur, s):
                    return idx, "title+year" if year is not None and y is not None else "title(year unknown)"
        fam = _title_family(it.get("title"))
        if fam and year is not None:
            for idx, s in self.family.get((fam, year), []):
                if self._authors_ok(sur, s):
                    return idx, "title_family+year"
        return None

    def add(self, it, idx) -> None:
        for k in _dedupe_keys(it):
            self.hard.setdefault(k, idx)
        year, sur = _year(it), _first_surname(it)
        t = _norm_title(it.get("title"))
        if len(t) >= MIN_TITLE_KEY_CHARS:
            self.titles.setdefault(t, []).append((idx, year, sur))
        fam = _title_family(it.get("title"))
        if fam and year is not None:
            self.family.setdefault((fam, year), []).append((idx, sur))


def _merge_note(it, reason, prov) -> dict:
    return {"reason": reason, "provider": prov, "title": (it.get("title") or "")[:160],
            "url": it.get("url"), "year": _year(it)}


def _finalize(items, provider_name, n):
    """Validate, de-duplicate (url, doi, arXiv id, normalised title+year, title family+year, first-author guard:
    G6, R72-d), rank. The higher-ranked (earlier) item is kept; the providers of dropped duplicates go to
    `found_by_providers` and each merge is recorded in the kept item's `merged_from` (reason, provider, title,
    url) so a wrong merge is visible."""
    out, dd = [], _Deduper()
    now = datetime.now(timezone.utc).isoformat()
    for it in items:
        url = (it.get("url") or "").strip()
        sp = urlsplit(url)
        if sp.scheme not in ("http", "https") or not sp.netloc:
            continue
        prov = it.get("provider") or provider_name
        cand = {**it, "url": url}
        hit = dd.find(cand)
        if hit is not None:
            idx, reason = hit
            kept = out[idx]
            for pn in (it.get("found_by_providers") or [prov]):
                if pn not in kept["found_by_providers"]:
                    kept["found_by_providers"].append(pn)
            kept["duplicates_merged"] = kept.get("duplicates_merged", 0) + 1
            kept.setdefault("merged_from", []).extend(it.get("merged_from") or [])
            kept["merged_from"].append(_merge_note(cand, reason, prov))
            dd.add(cand, idx)
            continue
        dd.add(cand, len(out))
        out.append({**it, "url": url, "title": (it.get("title") or "").strip(),
                    "snippet": (it.get("snippet") or "").strip(), "rank": len(out) + 1,
                    "provider": prov, "found_by_providers": list(it.get("found_by_providers") or [prov]),
                    "retrieved_at": now})
        if len(out) >= n:
            break
    return out


def apply_max_share(items: list, max_share: dict) -> tuple[list, int]:
    """Keep at most max_share[p] items whose providers are ALL capped, counted against the item's first provider
    (an item also found by an uncapped provider is not capped). Order is kept. -> (kept, dropped_count)."""
    used, out, dropped = {}, [], 0
    for it in items:
        found = it.get("found_by_providers") or [it.get("provider")]
        if all(p in max_share for p in found):
            first = found[0]
            if used.get(first, 0) >= int(max_share[first]):
                dropped += 1
                continue
            used[first] = used.get(first, 0) + 1
        out.append(it)
    return out, dropped


def rrf_merge(per_provider: dict, k: int, n: int):
    """Reciprocal rank fusion over per-provider ranked lists (ranks only: scores of different engines are never
    compared). score = sum over providers of 1/(k + rank). Equal items (any dedupe key) are one group; the
    representative is its best-ranked member."""
    groups, dd = [], _Deduper()
    for order, (pname, items) in enumerate(per_provider.items()):
        for rank, it in enumerate(items, start=1):
            hit = dd.find(it)
            if hit is None:
                gi = len(groups)
                groups.append({"item": it, "score": 0.0, "best": (rank, order), "providers": [], "merges": []})
            else:
                gi = hit[0]
                groups[gi]["merges"] += list(it.get("merged_from") or []) + [_merge_note(it, hit[1], pname)]
            g = groups[gi]
            dd.add(it, gi)
            g["score"] += 1.0 / (k + rank)
            if pname not in g["providers"]:
                g["providers"].append(pname)
            if (rank, order) < g["best"]:
                g["item"], g["best"] = it, (rank, order)
    groups.sort(key=lambda g: (-g["score"], g["best"]))
    out = []
    for g in groups[:n]:
        rec = {**g["item"], "found_by_providers": g["providers"], "rrf_score": round(g["score"], 6)}
        merges = list(g["item"].get("merged_from") or []) + g["merges"]
        if merges:
            rec["merged_from"] = merges
        out.append(rec)
    return out


# G2: ONE provider chain (and so one RateGate per provider) per process and config. Keyed by the config's
# search block, so a changed config gets a new chain.
_CHAIN_CACHE: dict = {}
_CHAIN_LOCK = threading.Lock()


def _search_settings():
    from vera.eval_config import load_eval_config
    return load_eval_config().search


def cached_chain(settings=None):
    """-> (chain, min_candidates, mode), cached per process and per search-config content."""
    s = settings if settings is not None else _search_settings()
    key = json.dumps(s, sort_keys=True, default=str)
    with _CHAIN_LOCK:
        hit = _CHAIN_CACHE.get(key)
        if hit is None:
            hit = (build_chain(s, repo_root=_ROOT), int(s["min_candidates"]), s.get("mode", "first_sufficient"))
            _CHAIN_CACHE[key] = hit
    return hit


def default_chain_and_min():
    chain, minimum, _mode = cached_chain()
    return chain, minimum


def _relaxed_flags(chain):
    """R72-e: which providers retried with OR-ed core phrases (per provider, latest call)."""
    return {p.name: bool(getattr(p, "last_relaxed", False)) for p in chain if hasattr(p, "can_relax")}


def _dedupe_merges(items):
    return [{"kept_url": it.get("url"), **m} for it in items for m in (it.get("merged_from") or [])]


def _route_queries(chain):
    q = {p.name: p.last_query for p in chain if getattr(p, "last_query", None)}
    return q


def search_question(question: str, num_results: int = 10, *, ledger: CostLedger | None = None, providers=None,
                    min_candidates: int | None = None, max_retries: int = 3, backoff_s: float = 1.0,
                    deadline_s: float = 60.0, timeout_s: float = 20.0, sleep=time.sleep,
                    clock=time.monotonic, mode: str | None = None, rrf_k: int | None = None,
                    max_share: dict | None = None) -> SearchResults:
    if not question or not question.strip():
        raise ValueError("question must be non-empty")
    if num_results < 1:
        raise ValueError("num_results must be >= 1")
    cfg_mode, cfg_k, cfg_share = "first_sufficient", 60, dict(DEFAULT_MAX_SHARE)
    if providers is None:
        # G2 (fixed, item #71): the chain is cached per process and config, so each provider's RateGate persists
        # across calls and `min_interval_s` holds across questions. `providers=` injection still works.
        try:
            s = _search_settings()
            chain, cfg_min, cfg_mode = cached_chain(s)
            cfg_k = int(s.get("rrf_k", 60))
            cfg_share = dict(s["max_share"]) if s.get("max_share") is not None else dict(DEFAULT_MAX_SHARE)
        except SearchConfigError:
            raise
        except _SCE as e:
            raise SearchConfigError(str(e)) from None
        except Exception as e:  # noqa: BLE001  (config errors are reported verbosely, never defaulted)
            raise SearchConfigError(f"search configuration unavailable: {type(e).__name__}: {e}") from None
        minimum = cfg_min if min_candidates is None else min_candidates
    else:
        chain, minimum = list(providers), (1 if min_candidates is None else min_candidates)
    mode = mode or cfg_mode
    if mode not in ("fan_out", "first_sufficient"):
        raise SearchConfigError(f"unknown search mode {mode!r}")
    rrf_k = cfg_k if rrf_k is None else rrf_k
    minimum = min(minimum, num_results)
    if not chain:
        raise SearchConfigError("no search provider configured")
    missing = [e for p in chain for e in p.required_env if not os.getenv(e)]
    if missing:
        raise SearchConfigError(f"provider(s) need env {missing}")
    deadline_at = clock() + deadline_s  # ONE deadline for the whole chain
    kw = dict(ledger=ledger, max_retries=max_retries, backoff_s=backoff_s, deadline_at=deadline_at,
              timeout_s=timeout_s, sleep=sleep, clock=clock)
    if mode == "fan_out":
        return _fan_out(chain, question, num_results, minimum, rrf_k, kw,
                        cfg_share if max_share is None else max_share)
    notes, tried, best = [], [], None
    for i, p in enumerate(chain):
        try:
            raw = _attempts(p, question, num_results, **kw)
        except _SCE as e:  # unusable config of a route we need: never skipped silently
            raise SearchConfigError(f"{p.name}: {e}") from None
        except _ProviderFailed as f:
            notes += [f"{p.name}: {x}" for x in f.notes]
            tried.append({"provider": p.name, "outcome": "failed"})
            continue
        items = _finalize(raw, p.name, num_results)
        tried.append({"provider": p.name, "outcome": "ok", "count": len(items)})
        if best is None or len(items) > len(best[1]):
            best = (p, items)
        if len(items) >= minimum:
            break
    if best is None:
        raise SearchError("search failed after bounded retries across providers: " + " | ".join(notes))
    p, items = best
    meta = {"mode": "first_sufficient", "route_tried": tried, "route_used": p.name, "min_candidates": minimum,
            "below_minimum": len(items) < minimum, "fallback_used": p.name.startswith("seed:"),
            "provider_endpoint": p.endpoint, "queries": _route_queries(chain),
            "relaxed": _relaxed_flags(chain), "dedupe_merges": _dedupe_merges(items)}
    if notes:
        meta["notes"] = notes
    if meta["fallback_used"]:
        meta["seed_sha256"] = p.expected
    return SearchResults(items, meta)


def _fan_out(chain, question, n, minimum, rrf_k, kw, max_share=None):
    """Call every live provider (each under its own gate and the shared deadline), tolerate individual failures
    (recorded), merge by reciprocal rank fusion. The seed provider is used only if no live provider returned
    anything."""
    live = [p for p in chain if not p.name.startswith("seed:")]
    seeds = [p for p in chain if p.name.startswith("seed:")]
    notes, tried, per = [], [], {}
    for p in live:
        try:
            raw = _attempts(p, question, n, **kw)
        except _SCE as e:
            raise SearchConfigError(f"{p.name}: {e}") from None
        except _ProviderFailed as f:
            notes += [f"{p.name}: {x}" for x in f.notes]
            tried.append({"provider": p.name, "outcome": "failed"})
            continue
        items = _finalize(raw, p.name, n)
        per[p.name] = items
        tried.append({"provider": p.name, "outcome": "ok", "count": len(items)})
    merged = rrf_merge(per, rrf_k, max(n * len(per), n)) if per else []
    share_dropped = 0
    if merged and max_share:  # after RRF, before the top-n cut: discovery cannot crowd out scholarly results
        merged, share_dropped = apply_max_share(merged, max_share)
    used = [pn for pn, its in per.items() if its]
    fallback = False
    if not merged and seeds:
        s = seeds[0]
        try:
            raw = _attempts(s, question, n, **kw)
        except _SCE as e:
            raise SearchConfigError(f"{s.name}: {e}") from None
        except _ProviderFailed as f:
            notes += [f"{s.name}: {x}" for x in f.notes]
            tried.append({"provider": s.name, "outcome": "failed"})
        else:
            merged = _finalize(raw, s.name, n)
            tried.append({"provider": s.name, "outcome": "ok", "count": len(merged)})
            used, fallback = [s.name], True
    elif not per:
        raise SearchError("search failed after bounded retries across providers: " + " | ".join(notes))
    if not merged and not per:
        raise SearchError("search failed after bounded retries across providers: " + " | ".join(notes))
    items = _finalize(merged, "fan_out", n)
    meta = {"mode": "fan_out", "route_tried": tried, "route_used": "+".join(used) or None,
            "min_candidates": minimum, "below_minimum": len(items) < minimum, "fallback_used": fallback,
            "provider_endpoint": ",".join(p.endpoint for p in chain if p.name in used),
            "provider_counts": {pn: len(its) for pn, its in per.items()},
            "provider_failures": [t["provider"] for t in tried if t["outcome"] == "failed"],
            "rrf_k": rrf_k, "queries": _route_queries(chain), "relaxed": _relaxed_flags(chain),
            "dedupe_merges": _dedupe_merges(items),
            "max_share": dict(max_share or {}), "share_dropped": share_dropped}
    if notes:
        meta["notes"] = notes
    if fallback:
        meta["seed_sha256"] = seeds[0].expected
    return SearchResults(items, meta)
