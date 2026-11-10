"""Search provider interface and the two providers of the M2 redo (plan vera-plan-m2-keyless-search.md).

  * `SearchProvider` is the seam `vera.search_and_fetch.search_question` drives. A provider knows how to make
    ONE request (`call`), turn the response into candidate dicts (`parse`), say what the call cost
    (`cost_for`) and how politely it must be called (`gate`).
  * `ArxivProvider` (first route): keyless, abstract pages only (see ARXIV_* constants for why).
  * `SeedFileProvider` (DISCLOSED FALLBACK): an agent-prepared candidate list read from a file that is frozen
    by sha256 (a changed file is a different run input and is refused).

Real network access exists in exactly one place per networked provider: `arxiv_default_transport`. Tests patch
it to raise. The seed provider never touches the network.

EXTENSION SEAM (R1 D5, last resort only): a provider for a hosted web-search tool would implement the same
`SearchProvider` protocol and be registered in `PROVIDER_FACTORIES`. NONE exists: no code path here spends
money, uses a key, or can be selected by config (an unknown provider name is a SearchConfigError).
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
from vera import strict_json
import re
import threading
from datetime import datetime, timezone
from html.parser import HTMLParser
import xml.etree.ElementTree as ET

import defusedxml
import defusedxml.ElementTree as DET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol
from urllib.parse import urlsplit

from vera import request_trace as rt
from vera.search_query import build_query_terms

# Why: names the cost basis recorded on every ledger record of a keyless provider. Research found no
# published price on the pages read, but also no terms text stating $0, so the basis says exactly that
# (plan section 12, SC4 "qualified"). The cost is recorded as KNOWN $0: an unknown cost would stop every
# scored run before M6 with exit 4 (run_mvp.stop_if_cost_unknown).
COST_BASIS_KEYLESS = "no_published_price_keyless"
# Why: a seed list is prepared out of band; inside the run its cost is truly 0, the discovery cost is not
# metered by VERA-DAR and is disclosed in run_mvp.DIVERGENCES.
COST_BASIS_SEED = "out_of_band_seed"

# Why: arXiv's API terms put DESCRIPTIVE METADATA under CC0 1.0 but forbid storing and serving e-prints
# without the copyright holder's permission or a permissive licence (plan 12.3 SC3, evidence arxiv-terms).
# CC0 is therefore recorded as a METADATA licence only and is never read as permission to store a paper.
ARXIV_METADATA_LICENCE = {"id": "CC0-1.0", "url": "https://creativecommons.org/publicdomain/zero/1.0/",
                          "source": "arxiv_api_terms"}
# Why: R1 approval: only the abstract page (https://arxiv.org/abs/<id>) is ever a candidate. HTML, PDF and
# e-print URLs are never emitted, because fetching them would store the paper itself.
ARXIV_ABSTRACT_PATH_PREFIX = "/abs/"
ARXIV_HOST = "arxiv.org"
# Why: a response over this size is not an Atom page of <=100 entries; refuse before parsing (the real value
# is config search.arxiv.max_response_bytes; this is only the validator's absolute ceiling).
MAX_RESPONSE_BYTES_CEILING = 8 * 1024 * 1024
# Why: no contact e-mail is sent to any API (R1 has not approved one), so the identifier names only the tool.
USER_AGENT = "VERA-research-pipeline/1.0 (+https://github.com/taggm-ghc/vera-dar; offline-reviewed)"
# Why: the frozen question is prose; arXiv searches terms. These words carry no topic signal, so they are
# dropped before the term query is built. Deliberately small and plain: the effect on recall is UNMEASURED
# (plan L1 measures it).
QUERY_STOPWORDS = frozenset(
    "a an and are as at be been but by can did do does for from has have how in into is it its of on or "
    "that the their them these this those to was were what when which who why with would your".split())
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
SEED_SCHEMA = "vera-seed-v1"
SEED_ITEM_REQUIRED = ("url", "title", "snippet", "found_at", "found_by", "reliability_tier")


class SearchConfigError(RuntimeError):
    """Chain/provider configuration is invalid or unsafe (declared in vera.search_and_fetch as a SearchError
    subclass; defined here so providers can raise it without an import cycle)."""


class ProviderResponseError(RuntimeError):
    """The provider answered HTTP 200 but the body is not a usable result (an error document, malformed XML,
    an oversized body). A failure, NEVER an empty result."""


class ProviderUnavailable(RuntimeError):
    """A provider refuses to be called right now and retrying cannot help (e.g. its per-process daily budget is
    spent). Non-retryable: search_and_fetch records it in the notes and the chain skips the provider."""


class HttpLike(Protocol):
    status_code: int
    text: str
    headers: Mapping[str, str]


@dataclass
class RateGate:
    """Per-provider politeness: minimum interval between calls and ONE connection at a time (lock)."""
    min_interval_s: float
    last_end: float | None = None

    def __post_init__(self):
        self.lock = threading.Lock()

    def wait_needed(self, now: float) -> float:
        if self.last_end is None:
            return 0.0
        return max(0.0, self.last_end + self.min_interval_s - now)

    def stamp(self, now: float) -> None:
        self.last_end = now


class SearchProvider(Protocol):
    name: str
    endpoint: str
    required_env: tuple[str, ...]
    gate: RateGate

    def call(self, query: str, n: int, timeout_s: float) -> HttpLike: ...
    def parse(self, resp: HttpLike, n: int) -> list[dict]: ...
    def cost_for(self, resp: HttpLike) -> tuple[float, bool, str]: ...
    def failure_basis(self) -> str: ...


# ----------------------------------------------------------------------------------------- licence records
def licence_record(metadata: dict | None = None, content: dict | None = None) -> dict:
    none = {"id": None, "url": None, "source": "none"}
    return {"metadata": dict(metadata) if metadata else dict(none),
            "content": dict(content) if content else dict(none)}


# ----------------------------------------------------------------------------------------- arXiv
def arxiv_default_transport(url: str, params: dict, headers: dict, timeout: float):
    """THE one real network call for arXiv. Tests patch this to raise; nothing else may import requests."""
    import requests
    from vera import api_permissions  # API-ALLOW-1107: permission gate + the entry's rate portal (RATE-1022 A1)
    return api_permissions.call(requests.get, url, params=params, headers=headers, timeout=timeout)


def _stamp(date_iso: str, end: bool) -> str:
    y, m, d = date_iso.split("-")
    return f"{y}{m}{d}{'2359' if end else '0000'}"


def _date_clause(q: Mapping) -> str:
    if q.get("submitted_from") and q.get("submitted_to"):
        return f" AND submittedDate:[{_stamp(q['submitted_from'], False)} TO {_stamp(q['submitted_to'], True)}]"
    return ""


def _legacy_arxiv_body(question: str, q: Mapping) -> str:
    seen, terms = set(), []
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]*", question.lower()):
        if len(tok) < q["min_term_chars"] or tok in QUERY_STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        terms.append(tok)
        if len(terms) >= q["max_terms"]:
            break
    if not terms:
        raise ValueError("question has no searchable term after stopword removal")
    return "(" + f" {q['operator']} ".join(f"all:{t}" for t in terms) + ")"


def build_arxiv_query_info(question: str, q: Mapping, terms_cfg: Mapping | None = None, *,
                           relaxed: bool = False) -> tuple[str, str]:
    """Prose -> (arXiv `search_query`, mode). With `terms_cfg` (config search.query_terms) the query is
    AND of the core concept phrases, AND (OR of the other content terms), within the submittedDate range
    (item #71 G1; fixes the meta-word defect measured in docs/trace_eval/retrieval_diagnosis_2026-10-04.md).
    With no usable term, or no `terms_cfg`, the previous behaviour (first `max_terms` non-stopword tokens) is
    used and the mode says so ("fallback_legacy" / "legacy_no_terms_config"); it raises ValueError only when
    even that has no term.

    FIXED 2026-10-04 by p3m3 item #71 (G1). It was a KNOWN WEAKNESS (docs/trace_eval/retrieval_diagnosis_2026-10-04.md):
    meta-words (published, evidence, show, about, such, studies) OR-ed across all fields matched almost any
    paper (mean 0.82 of 5 results on topic over 22 traced questions). The re-measurement is item #71's job."""
    mode = "legacy_no_terms_config"
    if terms_cfg is not None:
        qt = build_query_terms(question, terms_cfg)
        mode = qt.mode
        if not qt.empty:
            parts = [f'all:"{c}"' for c in qt.core]
            if relaxed and len(parts) > 1:  # R72-e: core phrases ORed (one retry after a zero-hit AND query)
                parts = ["(" + " OR ".join(parts) + ")"]
            extra = list(qt.extra)
            years = [t for t in extra if re.fullmatch(r"(?:19|20)\d{2}", t)]
            topic = [t for t in extra if t not in years]
            if years and topic:  # F1-QUERY-1106: a year never matches alone; it is ANDed with the topic terms
                from vera.search_query import stoplike_tokens
                topic = [t for t in topic if t not in stoplike_tokens()] or topic
                parts.append("(" + " OR ".join(f"all:{t}" for t in topic) + ")")
                parts.extend(f"all:{y}" for y in years)
            elif extra:
                # R3 (RELEVANCE-R1R4-1106): configured compounds ("multi-agent", config compound_concepts) are ANDed as phrases; a bare component
                # ("agent") is dropped; the other distinctive terms are ORed in one group. No compound: unchanged.
                from vera.search_query import is_compound
                comps = [t for t in extra if is_compound(t)]
                if comps:
                    rest = [t for t in extra if t not in comps
                            and not any(t in c.split("-") for c in comps)]
                    parts.extend(f'all:"{c}"' for c in comps)
                    if rest:
                        parts.append("(" + " OR ".join(f"all:{t}" for t in rest) + ")")
                else:
                    parts.append("(" + " OR ".join(f"all:{t}" for t in extra) + ")")
            body = "(" + " AND ".join(parts) + ")" if (qt.core or len(parts) > 1) else parts[0]
            return body + _date_clause(q), mode
        mode = "fallback_legacy"
    return _legacy_arxiv_body(question, q) + _date_clause(q), mode


def build_arxiv_query(question: str, q: Mapping, terms_cfg: Mapping | None = None) -> str:
    return build_arxiv_query_info(question, q, terms_cfg)[0]


def can_relax(question: str, terms_cfg: Mapping | None) -> bool:
    """R72-e: relaxation applies only when the config flag is on (default true) and the AND query had at least
    two core phrases to OR; otherwise the retry would repeat the same query."""
    if terms_cfg is None or not terms_cfg.get("relax_on_zero", True):
        return False
    return len(build_query_terms(question, terms_cfg).core) > 1


class ArxivProvider:
    """arXiv Atom API, abstract pages only. `settings` is config search.arxiv (validated by eval_config)."""
    name = "arxiv"
    required_env: tuple[str, ...] = ()

    CACHE_TTL_S = 86400.0  # COMP-1103: the arXiv API manual asks callers to cache results; in-process, 24 h
    CACHE_MAX = 128

    def __init__(self, settings: Mapping, *, transport: Callable | None = None, query_terms: Mapping | None = None,
                 wall_clock: Callable[[], float] | None = None):
        import time
        self.s = settings
        self.endpoint = settings["endpoint"]
        self.max_bytes = int(settings["max_response_bytes"])
        self.gate = RateGate(float(settings["min_interval_s"]))
        self._transport = transport
        self._wall = wall_clock or time.time
        self._cache: dict = {}  # (endpoint, search_query, n) -> (stored_at, response); key = the query terms only
        self.query_terms = query_terms
        self.last_query: dict | None = None  # {"query": ..., "mode": ...} of the latest call (route record)
        self.relaxed = False       # set by search_and_fetch for the single zero-hit retry (R72-e)
        self.last_relaxed = False

    def can_relax(self, query: str) -> bool:
        return can_relax(query, self.query_terms)

    def call(self, query: str, n: int, timeout_s: float):
        sq, mode = build_arxiv_query_info(query, self.s["query"], self.query_terms, relaxed=self.relaxed)
        self.last_query = {"query": sq, "mode": mode, **({"relaxed": True} if self.relaxed else {})}
        params = {"search_query": sq, "start": 0, "max_results": n,
                  "sortBy": "relevance", "sortOrder": "descending"}
        # Resolved at call time so a test patching arxiv_default_transport is honoured.
        tx = self._transport or arxiv_default_transport
        key = (self.endpoint, sq, n)
        hit = self._cache.get(key)
        if hit is not None and self._wall() - hit[0] < self.CACHE_TTL_S:
            self.last_query["cache_hit"] = True
            return hit[1]
        resp = tx(self.endpoint, params, {"User-Agent": USER_AGENT}, timeout_s)
        if getattr(resp, "status_code", None) == 200:
            if len(self._cache) >= self.CACHE_MAX:
                self._cache.pop(next(iter(self._cache)), None)
            self._cache[key] = (self._wall(), resp)
        return resp

    def parse(self, resp, n: int) -> list[dict]:
        body = resp.text or ""
        if len(body.encode("utf-8", "ignore")) > self.max_bytes:
            raise ProviderResponseError(f"arxiv response over {self.max_bytes} bytes; refused before parsing")
        low = body.lower()
        if "<!doctype" in low or "<!entity" in low:
            raise ProviderResponseError("arxiv response carries a DOCTYPE/ENTITY declaration; refused (XML bomb guard)")
        try:
            root = DET.fromstring(body)
        except ET.ParseError as e:
            raise ProviderResponseError(f"arxiv response is not well-formed XML: {e}") from None
        except defusedxml.DefusedXmlException as e:  # fixed reason: the exception may carry untrusted ids
            raise ProviderResponseError(f"arxiv response refused by defusedxml ({type(e).__name__})") from None
        if root.tag != ATOM + "feed":
            raise ProviderResponseError(f"arxiv response root is {root.tag!r}, expected an Atom feed")
        out = []
        for entry in root.findall(ATOM + "entry"):
            eid = (entry.findtext(ATOM + "id") or "").strip()
            if "/api/errors" in eid:  # arXiv reports errors as a one-entry feed: a FAILURE, not "no results"
                raise ProviderResponseError("arxiv returned an error entry: "
                                            + " ".join((entry.findtext(ATOM + "summary") or "").split())[:200])
            sp = urlsplit(eid)
            if sp.netloc.lower() not in (ARXIV_HOST, "www." + ARXIV_HOST) or not sp.path.startswith(ARXIV_ABSTRACT_PATH_PREFIX):
                continue  # not an abstract page: never emitted (R1: abstract page only)
            doi = (entry.findtext(ARXIV_NS + "doi") or "").strip() or None
            authors = [" ".join((a.findtext(ATOM + "name") or "").split()) for a in entry.findall(ATOM + "author")]
            out.append({
                "authors": [a for a in authors if a],
                "url": f"https://{ARXIV_HOST}{sp.path}",
                "title": " ".join((entry.findtext(ATOM + "title") or "").split()),
                "snippet": " ".join((entry.findtext(ATOM + "summary") or "").split()),
                "doi": doi,
                "published": (entry.findtext(ATOM + "published") or "").strip() or None,
                "licence": licence_record(metadata=ARXIV_METADATA_LICENCE),
                # Why: what fetch stores for this URL is the abstract page, i.e. metadata (title, abstract).
                "stored_text_kind": "abstract_metadata",
                "provider_endpoint": self.endpoint,
            })
            if len(out) >= n:
                break
        return out

    def cost_for(self, resp):
        return 0.0, True, COST_BASIS_KEYLESS

    def failure_basis(self) -> str:
        return COST_BASIS_KEYLESS


# ----------------------------------------------------------------------------------------- OpenAlex
# Why: OpenAlex metadata is CC0 (item #71 plan R2-a). Recorded as a METADATA licence only; the publisher's full
# text is never fetched or stored by this provider (stored_text_kind stays "abstract_metadata").
OPENALEX_METADATA_LICENCE = {"id": "CC0-1.0", "url": "https://creativecommons.org/publicdomain/zero/1.0/",
                             "source": "openalex_data_licence"}
COST_BASIS_KEYLESS_FREE = "keyless_free_allowance"
OPENALEX_SELECT = "id,doi,title,publication_year,publication_date,primary_location,abstract_inverted_index,type,authorships"


def openalex_default_transport(url: str, params: dict, headers: dict, timeout: float):
    """THE one real network call for OpenAlex. Tests patch this (and requests.get) to raise."""
    import requests
    from vera import api_permissions  # API-ALLOW-1107: permission gate + the entry's rate portal (RATE-1022 A2)
    return api_permissions.call(requests.get, url, params=params, headers=headers, timeout=timeout)


def openalex_key_env_name(settings: Mapping | None = None) -> str:
    """D4: ONE source of truth for the env var NAME: config/auto_acquire.json keys.openalex_api_key_env; fallback
    search.openalex.api_key_env (settings api_key_env); fallback OPENALEX_API_KEY. A name must be [A-Z][A-Z0-9_]*."""
    cand = []
    try:
        from vera import acq_config
        cand.append(((acq_config.load_config() or {}).get("keys") or {}).get("openalex_api_key_env"))
    except Exception:  # noqa: BLE001
        pass
    cand.append((settings or {}).get("api_key_env"))
    for c in cand:
        if isinstance(c, str) and re.fullmatch(r"[A-Z][A-Z0-9_]*", c.strip()):
            return c.strip()
    return "OPENALEX_API_KEY"


def openalex_api_key(settings: Mapping | None = None, env: Mapping | None = None) -> str | None:
    """YIELD-1021: OPTIONAL OpenAlex API key from the env var named by openalex_key_env_name().
    Absent or blank = the keyless free tier, unchanged. The value is never logged or stored."""
    import os
    name = openalex_key_env_name(settings)
    v = str((os.environ if env is None else env).get(name) or "").strip()
    return v if re.fullmatch(r"[A-Za-z0-9_\-]{8,200}", v) else None


def _scrub_resp_url(resp, key: str) -> None:
    """ACQ-1027 K3: a response's .url may carry the key; overwrite it with the scrubbed form when settable."""
    from vera import acq_urls
    u = getattr(resp, "url", None)
    if isinstance(u, str):
        try:
            resp.url = acq_urls.scrub(u, [key])
        except Exception:  # noqa: BLE001
            pass


def reconstruct_abstract(inverted: Mapping | None, max_chars: int) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text in position order and truncate."""
    if not isinstance(inverted, Mapping) or not inverted:
        return ""
    slots: dict[int, str] = {}
    for word, positions in inverted.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int) and 0 <= pos < 100000:
                slots[pos] = str(word)
    text = " ".join(slots[i] for i in sorted(slots))
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."


# ACQ-SEARCH-1021: per-call search profile for the P1.7 acquisition job only (set by vera.auto_acquire.make_search around
# its own search; default None = the unchanged /ask behaviour). Keys: "filter" (extra OpenAlex filter clause, e.g. works
# held by PubMed Central, which is where Europe PMC full text comes from), "and_extra_max" (int), "generic_terms" (list).
OPENALEX_PROFILE: contextvars.ContextVar = contextvars.ContextVar("openalex_search_profile", default=None)
# YIELD-1021: acquisition-only extra OpenAlex fields (OA status, OA/PMC locations with licence, ids.pmcid) for the
# licence/OA pre-filter in vera.auto_acquire.rank_candidates. Set ONLY by acquisition_search; /ask never sets it, so the
# /ask select and candidate dicts are unchanged. Separate from OPENALEX_PROFILE so the OR retry (profile None) keeps it.
OPENALEX_OA_FIELDS: contextvars.ContextVar = contextvars.ContextVar("openalex_oa_fields", default=False)
# QUERY-SHAPE-1108: per-call discovery shape set by acquisition_search: {"recency": bool, "review": bool}; None = unchanged.
OPENALEX_SHAPE: contextvars.ContextVar = contextvars.ContextVar("openalex_query_shape", default=None)
OPENALEX_OA_SELECT_EXTRA = "open_access,best_oa_location,locations,ids"
OPENALEX_PMC_SOURCE_ID = "S2764455111"  # PubMed Central (where Europe PMC full text comes from)


def _profile_extra_and(extra, profile: Mapping, qt=None) -> str:
    """Acquisition: a question with no core phrase gives an all-OR query that ranks any one common token (the
    ambiguous 'mma' matched alpha-gal papers). Instead AND the distinctive terms: drop the profile's generic terms
    (unless that leaves none) and keep at most `and_extra_max`. Deterministic, no model call.
    DDFIX-1023/REL D1: with `qt`, one term per concept run first (search_query.and_terms: anchor run first, filler words
    such as "affect" never), so no concept of the question is lost to the first words of the question."""
    generic = {str(g).lower() for g in profile.get("generic_terms") or ()}
    k = max(1, int(profile.get("and_extra_max") or 4))
    if qt is not None:
        from vera.search_query import and_terms
        picked = and_terms(qt, k, exclude=generic)
        if picked:
            return " AND ".join(picked)
    terms = [t for t in extra if str(t).lower() not in generic] or list(extra)
    return " AND ".join(terms[:k])


def build_openalex_search(question: str, terms_cfg: Mapping | None, *, relaxed: bool = False,
                          profile: Mapping | None = None) -> tuple[str, str]:
    """-> (OpenAlex `search` string, mode): quoted core phrases AND-ed, AND (OR of the other terms)."""
    if terms_cfg is not None:
        qt = build_query_terms(question, terms_cfg)
        if not qt.empty:
            if profile and not qt.core and qt.extra and profile.get("and_extra_max"):
                return _profile_extra_and(qt.extra, profile, qt), qt.mode + "+and"
            parts = [f'"{c}"' for c in qt.core]
            if relaxed and len(parts) > 1:  # R72-e: core phrases ORed (one retry after a zero-hit AND query)
                parts = ["(" + " OR ".join(parts) + ")"]
            if qt.extra:
                parts.append("(" + " OR ".join(qt.extra) + ")")
            return " AND ".join(parts), qt.mode
        mode = "fallback_legacy"
    else:
        mode = "legacy_no_terms_config"
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]*", question.lower()) if w not in QUERY_STOPWORDS and len(w) > 1]
    if not words:
        raise ValueError("question has no searchable term after stopword removal")
    return " ".join(words[:8]), mode


class OpenAlexProvider:
    """OpenAlex works search, keyless, metadata + reconstructed abstract only. `settings` is config
    search.openalex; `query_terms` is config search.query_terms. No `mailto` is sent (none is approved)."""
    name = "openalex"
    required_env: tuple[str, ...] = ()

    def __init__(self, settings: Mapping, *, transport: Callable | None = None, query_terms: Mapping | None = None,
                 today: Callable[[], str] | None = None):
        self.s = settings
        self.endpoint = settings["endpoint"]
        self.max_bytes = int(settings["max_response_bytes"])
        self.snippet_max = int(settings["snippet_max_chars"])
        self.gate = RateGate(float(settings["min_interval_s"]))
        self._transport = transport
        self.query_terms = query_terms
        self.last_query: dict | None = None
        self.relaxed = False       # set by search_and_fetch for the single zero-hit retry (R72-e)
        self.last_relaxed = False
        self._budget = int(settings["daily_search_budget"])
        self._today = today or (lambda: datetime.now(timezone.utc).date().isoformat())
        self._day, self._used = None, 0

    def _spend(self) -> None:
        day = self._today()
        if day != self._day:
            self._day, self._used = day, 0
        if self._used >= self._budget:
            raise ProviderUnavailable(f"openalex daily search budget of {self._budget} spent for {day} "
                                      "(per-process guard; non-retryable)")
        self._used += 1

    def can_relax(self, query: str) -> bool:
        return can_relax(query, self.query_terms)

    def call(self, query: str, n: int, timeout_s: float):
        profile = OPENALEX_PROFILE.get()
        search, mode = build_openalex_search(query, self.query_terms, relaxed=self.relaxed, profile=profile)
        self.last_query = {"query": search, "mode": mode, **({"relaxed": True} if self.relaxed else {})}
        self._spend()  # every HTTP attempt counts, retries included
        shp = OPENALEX_SHAPE.get() or {}
        flt = (f"from_publication_date:{self.s['from_publication_date']},"
               f"to_publication_date:{self.s['to_publication_date']}")
        if shp.get("recency") is False:  # QUERY-SHAPE-1108: definitional question, no date filter
            flt = ""
        if profile and profile.get("filter"):
            flt = (flt + "," if flt else "") + str(profile["filter"])
        sort = None
        if shp.get("review"):  # QUERY-SHAPE-1108: the one review-filtered call per question
            from vera import query_shape
            qc = query_shape.load_config()
            flt = (flt + "," if flt else "") + str(qc.get("review_filter") or "type:review")
            sort = str(qc.get("review_sort") or "cited_by_count:desc")
        select = OPENALEX_SELECT + ("," + OPENALEX_OA_SELECT_EXTRA if OPENALEX_OA_FIELDS.get() else "")
        params = {"search": search, "per_page": n, "select": select}
        if flt:
            params["filter"] = flt
        if sort:
            params["sort"] = sort
        tx = self._transport or openalex_default_transport
        key = openalex_api_key(self.s)
        if not key:
            return tx(self.endpoint, params, {"User-Agent": USER_AGENT}, timeout_s)
        from vera import acq_urls
        try:  # YIELD-1021: OPTIONAL key (env only); never logged: any error text is redacted before it propagates
            with acq_urls.quiet_urllib3([key]):  # ACQ-1027 K3: no DEBUG log of the raw keyed URL
                resp = tx(self.endpoint, {**params, "api_key": key}, {"User-Agent": USER_AGENT}, timeout_s)
            _scrub_resp_url(resp, key)
            return resp
        except Exception as e:  # noqa: BLE001
            msg = acq_urls.redact_text(str(e), [key])
            try:
                err = type(e)(msg)
            except Exception:  # noqa: BLE001
                err = ProviderUnavailable(msg)
            raise err from None

    def metadata_get(self, params: dict, timeout_s: float):
        """ARXIV-PORTAL-1110 (OD-1110-ARXIV-OPENALEX): ONE targeted metadata call (filter + select, no search text)
        through the same daily budget and visitor/seed split (_spend), transport (API allowlist + openalex rate portal)
        and optional OPENALEX_API_KEY as call(). The key is never logged."""
        self._spend()
        tx = self._transport or openalex_default_transport
        key = openalex_api_key(self.s)
        if not key:
            return tx(self.endpoint, dict(params), {"User-Agent": USER_AGENT}, timeout_s)
        from vera import acq_urls
        try:
            with acq_urls.quiet_urllib3([key]):
                resp = tx(self.endpoint, {**params, "api_key": key}, {"User-Agent": USER_AGENT}, timeout_s)
            _scrub_resp_url(resp, key)
            return resp
        except Exception as e:  # noqa: BLE001
            msg = acq_urls.redact_text(str(e), [key])
            try:
                err = type(e)(msg)
            except Exception:  # noqa: BLE001
                err = ProviderUnavailable(msg)
            raise err from None

    def parse(self, resp, n: int) -> list[dict]:
        body = resp.text or ""
        if len(body.encode("utf-8", "ignore")) > self.max_bytes:
            raise ProviderResponseError(f"openalex response over {self.max_bytes} bytes; refused before parsing")
        try:
            doc = strict_json.loads(body, keys="exact")  # abstract_inverted_index keys are words
        except ValueError as e:
            raise ProviderResponseError(f"openalex response is not JSON: {e}") from None
        if not isinstance(doc, dict) or not isinstance(doc.get("results"), list):
            raise ProviderResponseError("openalex response has no results list: "
                                        + " ".join(str(doc.get("error") or doc.get("message") or "")[:200].split())
                                        if isinstance(doc, dict) else "openalex response is not an object")
        out = []
        for w in doc["results"]:
            if not isinstance(w, dict):
                continue
            doi_raw = (w.get("doi") or "").strip()
            doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi_raw, flags=re.I) or None
            loc = w.get("primary_location") if isinstance(w.get("primary_location"), dict) else {}
            src = loc.get("source") if isinstance(loc.get("source"), dict) else {}
            landing = (loc.get("landing_page_url") or "").strip()
            url = f"https://doi.org/{doi}" if doi else (landing or (w.get("id") or "").strip())
            if not url:
                continue
            names = [" ".join(str((a.get("author") or {}).get("display_name") or "").split())
                     for a in (w.get("authorships") or []) if isinstance(a, dict)]
            out.append({
                "authors": [x for x in names if x],
                "url": url,
                "title": " ".join((w.get("title") or "").split()),
                "snippet": reconstruct_abstract(w.get("abstract_inverted_index"), self.snippet_max),
                "doi": doi,
                "published": (w.get("publication_date") or "").strip() or None,
                "year": w.get("publication_year") if isinstance(w.get("publication_year"), int) else None,
                "publisher": (src.get("display_name") or "").strip() or None,
                "source_type": (w.get("type") or "").strip() or None,
                "openalex_id": (w.get("id") or "").strip() or None,
                "licence": licence_record(metadata=OPENALEX_METADATA_LICENCE),
                "stored_text_kind": "abstract_metadata",
                "provider_endpoint": self.endpoint,
                **({"oa": oa} if (oa := openalex_oa_info(w)) is not None else {}),
            })
            if len(out) >= n:
                break
        return out

    def cost_for(self, resp):
        return 0.0, True, COST_BASIS_KEYLESS_FREE

    def failure_basis(self) -> str:
        return COST_BASIS_KEYLESS_FREE



# ----------------------------------------------------------------------------------------- YIELD-1021 acquisition
# Acquisition only (vera.auto_acquire / inline_acquire). Nothing here is used by the /ask reading list.
_OA_KEYS = ("open_access", "best_oa_location", "locations", "ids")


def _pmcid_of(v) -> str | None:
    m = re.search(r"PMC\d{1,10}", str(v or ""), flags=re.I)
    return m.group(0).upper() if m else None


def openalex_oa_info(w: Mapping) -> dict | None:
    """OA metadata the OpenAlex work already carries (no extra call), or None when the work has none of the fields
    (the /ask select): {"is_oa", "has_oa_location", "licence", "pmcid"}; each None when not stated."""
    if not isinstance(w, Mapping) or not any(k in w for k in _OA_KEYS):
        return None
    oa = w.get("open_access") if isinstance(w.get("open_access"), Mapping) else {}
    best = w.get("best_oa_location") if isinstance(w.get("best_oa_location"), Mapping) else None
    locs = [x for x in (w.get("locations") or ()) if isinstance(x, Mapping)] if isinstance(w.get("locations"), list) else []

    def src_id(x):
        return str(((x.get("source") or {}) if isinstance(x.get("source"), Mapping) else {}).get("id") or "")
    pmc = [x for x in locs if src_id(x).rstrip("/").endswith(OPENALEX_PMC_SOURCE_ID)]
    if "best_oa_location" in w or "locations" in w:
        has_oa = best is not None or any(x.get("is_oa") is True for x in locs)
    else:
        has_oa = None
    # The PMC location's licence is what Europe PMC full text reports; else the best OA location's.
    lic = next((str(x["license"]) for x in pmc if x.get("license")), None) or (
        str(best["license"]) if best is not None and best.get("license") else None)
    ids = w.get("ids") if isinstance(w.get("ids"), Mapping) else {}
    pmcid = _pmcid_of(ids.get("pmcid")) or next((p for p in (_pmcid_of(x.get("landing_page_url")) for x in pmc) if p),
                                                 None)
    return {"is_oa": oa.get("is_oa") if isinstance(oa.get("is_oa"), bool) else None, "has_oa_location": has_oa,
            "licence": lic, "pmcid": pmcid}


def _licence_is_nd(lic) -> bool:
    s = re.sub(r"[\s_]+", "-", str(lic or "").strip().lower())
    return bool(re.search(r"(^|-)nd($|-|\d)", s)) or "noderiv" in s


def acq_prefilter(cand: Mapping) -> str | None:
    """YIELD-1021 pre-filter BEFORE the top-N cut: "nd" (stated ND licence) or "not_oa" (metadata states no OA location
    at all), else None. Missing metadata or an unknown licence keeps the candidate; the licence gate on the fetched text
    is unchanged and still decides."""
    oa = cand.get("oa") if isinstance(cand, Mapping) else None
    if not isinstance(oa, Mapping):
        return None
    if _licence_is_nd(oa.get("licence")):
        return "nd"
    if oa.get("epmc_ft"):
        return None
    if oa.get("has_oa_location") is False and oa.get("is_oa") is not True:
        return "not_oa"
    return None


def europepmc_default_transport(url: str, params: dict, headers: dict, timeout: float):
    """THE one real network call for the acquisition Europe PMC search. Tests patch this (or requests.get)."""
    import requests
    from vera import api_permissions  # API-ALLOW-1107: permission gate + rate portal (RATE-1022 A3: search, id lookup)
    return api_permissions.call(requests.get, url, params=params, headers=headers, timeout=timeout,
                                allow_redirects=False)


def build_europepmc_query(question: str, terms_cfg: Mapping | None, profile: Mapping | None, suffix: str) -> str:
    search, _mode = build_openalex_search(question, terms_cfg, profile=profile)
    return f"({search}) AND {suffix}" if suffix else search


def europepmc_search(question: str, n: int, timeout_s: float, ecfg: Mapping, *, terms_cfg: Mapping | None = None,
                     profile: Mapping | None = None, transport: Callable | None = None) -> list[dict]:
    """ONE Europe PMC REST search (resultType lite) for open-access works with full text in Europe PMC. Hits without a
    PMCID, not open access, or from a refused source are dropped; hits without a DOI stay (counted R-ACQ-NODOI by the
    ranking, which resolves by DOI). Raises on transport or response errors."""
    params = {"query": build_europepmc_query(question, terms_cfg, profile, str(ecfg.get("query_suffix") or "")),
              "resultType": "lite", "format": "json", "pageSize": int(ecfg.get("page_size", 10))}
    resp = (transport or europepmc_default_transport)(str(ecfg["search_url"]), params, {"User-Agent": USER_AGENT},
                                                      float(timeout_s))
    if int(getattr(resp, "status_code", 200) or 200) != 200:
        raise ProviderResponseError(f"europepmc search HTTP {getattr(resp, 'status_code', '?')}")
    body = resp.text or ""
    if len(body.encode("utf-8", "ignore")) > int(ecfg.get("max_response_bytes", 2_000_000)):
        raise ProviderResponseError("europepmc response too large; refused before parsing")
    try:
        doc = strict_json.loads(body)
        rows = (doc.get("resultList") or {}).get("result") or []
    except (ValueError, AttributeError):
        raise ProviderResponseError("europepmc response is not the expected JSON") from None
    refuse = {str(x).upper() for x in ecfg.get("refuse_sources") or ()}
    out = []
    for r in rows if isinstance(rows, list) else ():
        if not isinstance(r, dict):
            continue
        pmcid = _pmcid_of(r.get("pmcid"))
        if (not pmcid or str(r.get("isOpenAccess") or "").upper() != "Y"
                or str(r.get("source") or "").upper() in refuse):
            continue
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", str(r.get("doi") or "").strip(), flags=re.I) or None
        out.append({"url": f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/PMC/{pmcid}",
                    "title": " ".join(str(r.get("title") or "").split()), "snippet": "", "doi": doi,
                    "year": int(r["pubYear"]) if str(r.get("pubYear") or "").isdigit() else None,
                    "provider": "europepmc", "pmcid": pmcid,
                    "oa": {"is_oa": True, "has_oa_location": True, "licence": None, "pmcid": pmcid, "epmc_ft": True}})
        if len(out) >= n:
            break
    return out


def _or_differs(question: str, terms_cfg, profile, suffix: str) -> bool:
    """DAR-TRACE-1022 review M3: True only when the OR (profile None) Europe PMC query differs from the AND one."""
    if not profile:
        return False
    try:
        return build_europepmc_query(question, terms_cfg, None, suffix) != build_europepmc_query(question, terms_cfg,
                                                                                                  profile, suffix)
    except Exception:  # noqa: BLE001
        return False


def _is_timeout(e: BaseException) -> bool:
    """DDGFIX-1023 DF-5: a portal call that timed out (builtin/futures TimeoutError or requests' Timeout family:
    ReadTimeout, ConnectTimeout)."""
    import concurrent.futures as cf
    if isinstance(e, (cf.TimeoutError, TimeoutError)):
        return True
    try:
        import requests
        return isinstance(e, requests.exceptions.Timeout)
    except Exception:  # noqa: BLE001
        return False


# NONASK-1103 F0 follow-up: a record also de-duplicates by normalised title against the PMC/OpenAlex record of the same
# work (an arXiv preprint and its journal version have different identifiers). A title shorter than MIN_TITLE_WORDS
# words is never a key ("Introduction" must not merge unrelated works), and two records whose identifiers CONFLICT
# (different PMCIDs, or different DOIs when neither is an arXiv DataCite DOI) are different works whatever their titles.
# CALIBRATION NOTE: MIN_TITLE_WORDS is a planner prior, UNCALIBRATED; raise it if distinct works merge, lower it if
# duplicates of short titles survive (review the found_by lists of merged records).
MIN_TITLE_WORDS = 4
_ARXIV_DOI = "10.48550/arxiv."


def norm_title(t) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(t or "").lower()))


def title_key(c: Mapping) -> str | None:
    nt = norm_title(c.get("title"))
    return "title:" + nt if len(nt.split()) >= MIN_TITLE_WORDS else None


def _same_work_ids(a: Mapping, b: Mapping) -> bool:
    """No identifier conflict between two records that share a title."""
    ia, ib = _cand_ids(a), _cand_ids(b)
    pa = {i for i in ia if i.startswith("pmc:")}
    pb = {i for i in ib if i.startswith("pmc:")}
    if pa and pb and pa != pb:
        return False
    da = {i for i in ia if i.startswith("doi:")}
    db = {i for i in ib if i.startswith("doi:")}
    if da and db and da != db and not any(i[4:].startswith(_ARXIV_DOI) for i in da | db):
        return False
    return True


def _cand_ids(c: Mapping) -> set:
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", str(c.get("doi") or "").strip().lower())
    pm = _pmcid_of(c.get("pmcid") or ((c.get("oa") or {}) if isinstance(c.get("oa"), Mapping) else {}).get("pmcid"))
    return ({"doi:" + doi} if doi else set()) | ({"pmc:" + pm} if pm else set())


def _default_terms_and_profile(cfg: Mapping):
    from vera import search_and_fetch
    chain = search_and_fetch.cached_chain()[0]
    oa = next((p for p in chain if getattr(p, "name", None) == "openalex"), None)
    return getattr(oa, "query_terms", None), ((cfg.get("search") or {}).get("openalex_profile") or None)


# ---- DuckDuckGo results -> acquirable candidates (owner 2026-10-21: "ddg should be the first source for prompts/queries
# that are not in the corpus"). Only results that carry or resolve to a scholarly identifier become candidates; they go
# through the UNCHANGED resolve (Europe PMC by DOI), fetch and licence gates. Any other page is HELD (never ingested,
# never evidence): the acquisition fetch accepts only Europe PMC JATS XML, so licensed HTML pages are not candidates.
_DOI_IN_URL = re.compile(r"(10\.\d{4,9}/[^\s?#&\"'<>]+)", re.I)
_ARXIV_URL = re.compile(r"^https?://(?:www\.|export\.)?arxiv\.org/(?:abs|pdf|html)/([a-z\-]+/\d{7}|\d{4}\.\d{4,5})(v\d+)?"
                        r"(?:\.pdf)?/?$", re.I)  # DDGFIX-1023 DF-2: the version is captured (group 2), never dropped
_PMC_URL = re.compile(r"^https?://(?:www\.)?(?:ncbi\.nlm\.nih\.gov/pmc/articles|pmc\.ncbi\.nlm\.nih\.gov/articles|"
                      r"europepmc\.org/(?:article|abstract)/PMC)/(PMC\d{1,10})\b", re.I)
_PMID_URL = re.compile(r"^https?://(?:pubmed\.ncbi\.nlm\.nih\.gov/|(?:www\.)?ncbi\.nlm\.nih\.gov/pubmed/|"
                       r"europepmc\.org/(?:article|abstract)/MED/)(\d{1,9})\b", re.I)
_META_DOI = re.compile(r"<meta\s[^>]*?name=[\"'](?:citation_doi|dc\.identifier|prism\.doi)[\"'][^>]*?>", re.I)
_CONTENT = re.compile(r"content=[\"']\s*(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[^\"'\s]+)", re.I)


_DOI_TAIL = re.compile(r"(?:\.full|\.pdf|\.abstract|\.html?|/full|/abstract|/epdf|/pdf|/meta|/|[.,;:])$", re.I)
_CLOSERS = {")": "(", "]": "[", "}": "{"}


def normalise_doi(s) -> str | None:
    """DDGFIX-1023 DF-1: a DOI taken from a URL or page text -> its canonical form, or None. URL-decoded, query/fragment
    removed, a resolver prefix ("doi:", doi.org) removed, trailing "/", ".full", ".pdf", ".abstract", "/full",
    "/abstract", "/pdf", "/epdf" and trailing punctuation stripped, a closing ")", "]" or "}" only when UNBALANCED
    (verifier D3: "...U2(1)" stays intact; SICI DOIs with "<...>" are kept whole), lower-cased (case-insensitive)."""
    from urllib.parse import unquote
    d = unquote(str(s or "")).strip()  # ONE decode: "%252F" stays "%2F" (verifier D3)
    d = re.sub(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)", "", d, flags=re.I)
    d = re.split(r"[?#\s\"']", d, maxsplit=1)[0]
    while d:
        t = _DOI_TAIL.sub("", d)
        if t == d and d[-1] in _CLOSERS and d.count(_CLOSERS[d[-1]]) < d.count(d[-1]):
            t = d[:-1]
        if t == d:
            break
        d = t
    d = d.lower()
    return d if re.fullmatch(r"10\.\d{4,9}/\S+", d) else None


# NONASK-1103 WP-D: Handle System identifiers (stable PIDs). prefix = digits and dots and NOT "10." (a "10.<digits>" prefix
# is a DOI, which stays a DOI even inside hdl.handle.net/10.x/y); suffix = any non-space text.
_HANDLE_URL = re.compile(r"^https?://(?:hdl\.)?handle\.net/(?:api/handles/)?((?!10\.)\d+(?:\.\d+)*)/([^\s?#]+)", re.I)
_HANDLE_SCHEME = re.compile(r"^hdl:\s*(?:/{0,2})((?!10\.)\d+(?:\.\d+)*)/([^\s?#]+)$", re.I)


def ddg_identifier(url: str) -> dict | None:
    """{"kind": "doi"|"pmcid"|"pmid"|"arxiv"|"handle", "value"[, "version"]} from the URL alone (no request), or None. An
    arXiv URL keeps its version ("v2") when it names one (DDGFIX-1023 DF-2); "value" stays the bare id. A Handle
    (hdl.handle.net/<prefix>/<suffix> or hdl:<prefix>/<suffix>) is kind "handle", value "<prefix>/<suffix>"; it is a
    persistent identifier with no automatic route (WP-D); a malformed handle is None."""
    u = str(url or "").strip()
    from urllib.parse import unquote as _unq
    m = _HANDLE_URL.match(u) or _HANDLE_SCHEME.match(u)
    if m:
        suffix = _unq(m.group(2)).strip("/")
        return {"kind": "handle", "value": f"{m.group(1)}/{suffix}"} if suffix else None
    m = _ARXIV_URL.match(u)
    if m:
        return {"kind": "arxiv", "value": m.group(1).lower(), "version": (m.group(2) or "").lower() or None}
    m = _PMC_URL.match(u)
    if m:
        return {"kind": "pmcid", "value": m.group(1).upper()}
    m = _PMID_URL.match(u)
    if m:
        return {"kind": "pmid", "value": m.group(1)}
    from urllib.parse import unquote
    m = _DOI_IN_URL.search(unquote(u.split("#", 1)[0].split("?", 1)[0]))  # DF-1: decoded path, no query/fragment
    if m:
        doi = normalise_doi(m.group(1))
        if doi:
            return {"kind": "doi", "value": doi}
    return None


_WIKI_HOST = re.compile(r"(?:^|\.)(?:wikipedia|wikimedia|wikidata)\.org$", re.I)


def _is_wikipedia(url: str) -> bool:
    """Wikipedia is a portal only (owner 2026-10-22): its pages are never fetched, followed or used as evidence here."""
    try:
        return bool(_WIKI_HOST.search(urlsplit(str(url or "")).hostname or ""))
    except ValueError:
        return False


def _landing_url_ok(u: str) -> bool:
    """DF-8: https only, no userinfo, default port: the same URL shape rule as every other acquisition fetch."""
    try:
        sp = urlsplit(u)
        port = sp.port
    except ValueError:
        return False
    return sp.scheme == "https" and bool(sp.hostname) and sp.username is None and sp.password is None \
        and port in (None, 443)


def _landing_robots(acq: Mapping | None, get: Callable, url_check: Callable, deadline: float | None,
                    clock: Callable) -> Callable:
    """DF-8: robots.txt for each landing hop through the EXISTING checker (acq_discovered._robots_allowed: cached per
    host, 4xx = allowed per RFC 9309, 429/5xx/error/timeout = NOT allowed, fail closed), using the same pinned,
    public-address-checked GET. The host set is the hop's own host only (no allowlist entry is added anywhere)."""
    from types import SimpleNamespace

    from vera import acq_discovered
    from vera import selective_fetch as _sf

    class _Tee:  # ASK-PARALLEL-FIX-1105: keep the raw robots.txt text (robotparser drops non-integer Crawl-delay)
        def __init__(self, resp, host):  # explicit fields only (no run-time-named getattr: test_semantic_similarity)
            self._r, self._h = resp, host
            self.status_code, self.headers = resp.status_code, resp.headers

        def close(self):
            close = getattr(self._r, "close", None)
            if close is not None:
                close()

        def iter_content(self, n):
            buf = []
            for chunk in self._r.iter_content(n):
                buf.append(chunk)
                yield chunk
            _sf.remember_robots_text(self._h, b"".join(buf).decode("utf-8", "ignore"))

    def tee_get(u, *a, **k):
        r = get(u, *a, **k)
        sp_ = urlsplit(u)
        if sp_.path == "/robots.txt":
            if getattr(r, "status_code", None) == 200:
                return _Tee(r, (sp_.hostname or "").lower())
            if getattr(r, "status_code", None) not in (301, 302, 303, 307, 308):
                _sf.remember_robots_text(sp_.hostname or "", None)
        return r
    deps = SimpleNamespace(url_check=url_check, http_get=tee_get)

    def ok(u: str) -> bool:
        try:
            host = (urlsplit(u).hostname or "").lower()
            return bool(acq_discovered._robots_allowed(u, {host}, deps, dict(acq or {}), clock, deadline))
        except Exception:  # noqa: BLE001  fail closed
            return False
    return ok


# ASK-PARALLEL-FIX-1105 (owner rule feedback_respect_access_and_compliance_terms): at most ONE landing request in flight
# per actual request host, on every hop (redirects included), for every host (portal or not), process-wide. Entries are
# refcounted and dropped when no landing holds or waits on them, so the table stays bounded by in-flight landings.
_HOST_SLOTS: dict = {}
_HOST_SLOTS_LOCK = threading.Lock()


@contextlib.contextmanager
def _host_slot(host: str, timeout: float):
    """Hold the host's in-flight slot (yields True), or yield False if it is not free within `timeout` seconds."""
    host = (host or "").lower()
    with _HOST_SLOTS_LOCK:
        ent = _HOST_SLOTS.get(host)
        if ent is None:
            ent = _HOST_SLOTS[host] = [threading.Lock(), 0]
        ent[1] += 1
    got = False
    try:
        got = ent[0].acquire(timeout=max(0.0, timeout))
        yield got
    finally:
        if got:
            ent[0].release()
        with _HOST_SLOTS_LOCK:
            ent[1] -= 1
            if ent[1] <= 0 and _HOST_SLOTS.get(host) is ent:
                del _HOST_SLOTS[host]


def _landing_crawl_delay(url: str) -> float:
    """robots.txt Crawl-delay (seconds) for this URL's host, from the robots.txt already parsed by the landing robots
    check (acq_discovered cache) or by selective_fetch.robots_allowed; 0 when none is known. Never fetches or raises."""
    try:
        from vera import acq_discovered as _ad
        from vera import selective_fetch as _sf
        host = (urlsplit(url).hostname or "").lower()
        t = _sf.robots_text_crawl_delay(host)  # ASK-PARALLEL-FIX-1105: fractions honoured; inf = malformed
        if t is not None:
            return float(t)
        with _ad._ROBOTS_LOCK:
            hit = _ad._ROBOTS.get(host)
        rp = hit[1] if hit is not None else None
        d = 0.0
        if rp not in (None, True, False):
            v = rp.crawl_delay(_sf.USER_AGENT)
            d = float(v) if v else 0.0
        return max(d, _sf.robots_crawl_delay(url))
    except Exception:  # noqa: BLE001
        return 0.0


def _crawl_gate(url: str, budget: float, clock: Callable, sleep: Callable) -> bool:
    """Honour the host's Crawl-delay before a landing request: wait (inside `budget` seconds, keeping 0.2 s for the
    request) until selective_fetch.crawl_reserve (the shared per-host gate) admits it. False = skip, never send early.
    Bounded: at most 3 waits (the caller holds the host slot, so only other code paths can race for the host)."""
    from vera import selective_fetch as _sf
    host = (urlsplit(url).hostname or "").lower()
    delay = _landing_crawl_delay(url)
    if not delay:
        return True
    if delay == float("inf"):
        return False  # malformed Crawl-delay: skip this host rather than fetch without a delay
    end = clock() + budget
    for _ in range(3):
        if _sf.crawl_reserve(host, delay, clock):
            return True
        w = _sf.crawl_wait_s(host, delay, clock)
        if w <= 0:
            continue
        if clock() + w > end - 0.2:
            return False  # the delay does not fit the budget: skip this landing rather than wait past it
        try:
            from vera import progress as _prog  # NOISY-UI: politeness (crawl-delay) wait is visible
            ctx = _prog.waiting(_prog.ACT_POLITE, w)
        except Exception:  # noqa: BLE001
            ctx = contextlib.nullcontext()
        with ctx:
            sleep(w)
    return False


def landing_doi(url: str, timeout_s: float, max_bytes: int, *, get: Callable | None = None,
                url_check: Callable | None = None, deadline: float | None = None,
                clock: Callable | None = None, acq: Mapping | None = None,
                robots_ok: Callable | None = None) -> str | None:
    """ONE cheap GET of a journal landing page -> its citation_doi meta tag, or None. Never raises. The page is read
    ONLY for the identifier: its text is never stored, returned or used as evidence. DDGFIX-1023 DF-8, on every hop
    (at most 2 redirects): https/no userinfo/port 443, the public-address check, robots.txt through the existing
    checker (fail closed), the pinned transport (connects only to the address just validated) wrapped in the
    per-host rate limiter (config/rate_limits.json; hosts in no portal pass unlimited, as for every caller),
    Accept-Encoding identity, an HTML content type, the size cap and the hop/overall deadline. Owner P0 (2026-10-23):
    previously unknown public websites are allowed through this existing safe transport; no host list is added."""
    if get is None:  # CMF-1022-I2 (E1/A4): connect only to the address validated for this hop
        from vera import api_permissions
        from vera.selective_fetch import pinned_get
        get = api_permissions.limited_transport(pinned_get)  # 1108b (4): an API hop uses its own entry's portal
    if url_check is None:
        from vera.selective_fetch import is_public_url as url_check
    import time as _t
    clock = clock or _t.monotonic
    if robots_ok is None:
        robots_ok = _landing_robots(acq, get, url_check,
                                    clock() + timeout_s if deadline is None else min(deadline, clock() + timeout_s),
                                    clock)
    cur = url
    try:
        for _hop in range(3):
            left = timeout_s if deadline is None else min(timeout_s, deadline - clock())
            if left <= 0.2:
                return None
            if not _landing_url_ok(cur) or _is_wikipedia(cur):
                return None
            from vera import selective_fetch as _sf_api
            try:  # API-URL-GATE-1108: an API URL (first or redirect hop) needs its api_permissions entry, any transport
                _sf_api.api_gate(cur, "fulltext")
            except _sf_api.ApiRefused:
                return None
            ok = url_check(cur)
            if not (ok[0] if isinstance(ok, tuple) else ok):
                return None
            # ASK-PARALLEL-FIX-1105: this hop's host slot is held from its robots.txt check through the page GET,
            # body read and close, so no two landings (first hop or redirect) are ever in flight to one host.
            with _host_slot(urlsplit(cur).hostname or "", left - 0.2) as got:
                if not got:
                    return None  # the host stayed busy for the whole hop budget: skip, never overlap
                if not robots_ok(cur):
                    return None
                left = timeout_s if deadline is None else min(timeout_s, deadline - clock())
                if left <= 0.2:
                    return None
                if not _crawl_gate(cur, left, clock, _t.sleep):
                    return None  # robots.txt Crawl-delay does not fit the hop budget: skip rather than send early
                left = timeout_s if deadline is None else min(timeout_s, deadline - clock())
                if left <= 0.2:
                    return None
                from vera.selective_fetch import USER_AGENT as _FETCH_UA  # verifier D1: the UA robots.txt was checked for
                _purpose = {"api_purpose": "fulltext"} if _sf_api.api_permissions.accepts_purpose(get) else {}  # 1108b (6)
                r = get(cur, headers={"User-Agent": _FETCH_UA, "Accept-Encoding": "identity"}, timeout=left,
                        allow_redirects=False, stream=True, **_purpose)
                stop_at = clock() + left  # a trickling server cannot hold the thread past the hop budget or the deadline
                try:
                    if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
                        from urllib.parse import urljoin
                        cur = urljoin(cur, r.headers["Location"])
                        continue
                    if r.status_code != 200:
                        return None
                    ctype = str((getattr(r, "headers", None) or {}).get("Content-Type") or "").split(";")[0].strip().lower()
                    if ctype and ctype not in ("text/html", "application/xhtml+xml"):
                        return None  # DF-8: only an HTML landing page is read for its meta tags
                    buf = b""
                    for chunk in r.iter_content(65536):
                        buf += chunk
                        if len(buf) >= max_bytes or clock() > stop_at:
                            break
                finally:
                    try:
                        r.close()
                    except Exception:  # noqa: BLE001
                        pass
            text = buf[:max_bytes].decode("utf-8", "ignore")
            for tag in _META_DOI.findall(text):
                m = _CONTENT.search(tag)
                if m:
                    return normalise_doi(m.group(1))  # DF-1
            return None
    except Exception:  # noqa: BLE001
        return None
    return None


def europepmc_ids_lookup(ids: list, timeout_s: float, ecfg: Mapping, *, transport: Callable | None = None) -> dict:
    """ONE Europe PMC lite search mapping PMCIDs / PMIDs to DOIs: {"PMC..."|"pmid:N": {"doi", "pmcid"}}. Raises on error."""
    terms = [f"PMCID:{v}" if k == "pmcid" else f"(EXT_ID:{v} AND SRC:MED)" for k, v in ids]
    if not terms:
        return {}
    params = {"query": " OR ".join(terms), "resultType": "lite", "format": "json", "pageSize": len(terms)}
    resp = (transport or europepmc_default_transport)(str(ecfg["search_url"]), params, {"User-Agent": USER_AGENT},
                                                      float(timeout_s))
    if int(getattr(resp, "status_code", 200) or 200) != 200:
        raise ProviderResponseError(f"europepmc lookup HTTP {getattr(resp, 'status_code', '?')}")
    rows = ((strict_json.loads(resp.text or "{}").get("resultList") or {}).get("result") or [])
    out = {}
    for r in rows if isinstance(rows, list) else ():
        if not isinstance(r, dict) or not r.get("doi"):
            continue
        rec = {"doi": str(r["doi"]).strip().lower(), "pmcid": _pmcid_of(r.get("pmcid"))}
        if rec["pmcid"]:
            out[rec["pmcid"]] = rec
        if r.get("pmid"):
            out["pmid:" + str(r["pmid"]).strip()] = rec
    return out


def _landing_group(url: str) -> str:
    """The rate-limit portal a landing URL's first hop goes through (config/rate_limits.json), else its host."""
    host = (urlsplit(url).hostname or "").lower()
    try:
        from vera import rate_limit
        a = rate_limit.get_registry().portal_for_url(url)
        return "portal:" + a.name if a is not None else "host:" + host
    except Exception:  # noqa: BLE001  no limiter config: group by host (still one call at a time per host)
        return "host:" + host


def _landing_dois_concurrently(urls: list, lt: float, lb: int, **kw) -> list:
    """ASK-PARALLEL-1105 (owner rule feedback_nonblocking_spawned_steps): the DDG landing-page DOI lookups (each its own
    robots.txt check and ONE GET, through landing_doi unchanged) are independent, so they run concurrently instead of one
    after another. Landings whose FIRST hop shares a rate-limit portal (or, outside every portal, a host) run in ONE
    worker in result order, rechecking the serial loop's `deadline - clock() > landing_timeout_s` before each later
    landing (a skipped one is returned in `skipped` and not counted). ASK-PARALLEL-FIX-1105: on top of that grouping,
    landing_doi holds a per-host in-flight slot (_host_slot) on EVERY hop, redirects included, for every host (portal
    or not), and honours robots.txt Crawl-delay per host (_crawl_gate -> selective_fetch.crawl_reserve), so landings
    that converge on one host after redirects run one at a time; different hosts still run in parallel. The same
    deadline bounds every call. Workers
    run in a copy of this context (the inline rate-limit deadline). Returns (DOIs or None in the order of `urls`,
    set of skipped indexes)."""
    out: list = [None] * len(urls)
    skipped: set = set()
    if not urls:
        return out, skipped
    deadline, clock = kw.get("deadline"), kw.get("clock")
    groups: dict = {}
    for i, u in enumerate(urls):
        groups.setdefault(_landing_group(u), []).append(i)

    errs: list = [None] * len(urls)

    def run(idxs):
        for j, i in enumerate(idxs):
            if j and deadline is not None and clock is not None and not deadline - clock() > lt:
                skipped.add(i)  # as the serial loop: no landing GET started without landing_timeout_s left
                continue
            try:
                out[i] = landing_doi(urls[i], lt, lb, **kw)
            except Exception as e:  # noqa: BLE001  landing_doi never raises; anything else propagates as before
                errs[i] = e
                return
    if len(groups) == 1:
        run(next(iter(groups.values())))
    else:
        threads = []
        for idxs in groups.values():
            t = threading.Thread(target=contextvars.copy_context().run, args=(run, idxs), name="vera-landing",
                                 daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()  # each landing_doi call is bounded by landing_timeout_s and the caller's deadline
    for e in errs:
        if e is not None:
            raise e
    return out, skipped


def ddg_to_candidates(results: list, ecfg: Mapping, dcfg: Mapping, *, deadline: float, clock: Callable,
                      calls_left: int, transport: Callable | None = None, get: Callable | None = None,
                      url_check: Callable | None = None, acq: Mapping | None = None,
                      robots_ok: Callable | None = None) -> tuple[list, dict]:
    """DDG results -> (acquirable candidates, counts). DOI in the URL and arXiv ids map directly; PMC/PubMed links map
    through ONE Europe PMC id lookup (only when calls_left >= 1); a portal landing page with no identifier gets ONE
    cheap GET for its citation_doi meta tag (at most dcfg.max_landing_fetches, inside the deadline). Everything else is
    held (counted, dropped from acquisition; it stays a reading-list link on /ask).
    DDGFIX-1023 DF-3 (URLPUBLIC-1023: "a candidate URL may indicate that it may need to be followed to find the actual
    source"): a public https result with no identifier becomes a URL-ONLY candidate (no DOI/PMCID): acquisition drops
    it R-ACQ-NODOI and the CONT-1023 rules queue it to be followed (route follow_url: identifier extraction from the
    URL only, no request). Wikipedia pages stay held (portal only, never evidence or a candidate)."""
    counts = {"ddg_results": 0, "ddg_mapped": 0, "ddg_held": 0, "ddg_landing_get": 0, "ddg_url_only": 0}
    out, pending, landing = [], [], 0
    max_landing = int(dcfg.get("max_landing_fetches", 3))
    lt = float(dcfg.get("landing_timeout_s", 5))
    lb = int(dcfg.get("landing_max_bytes", 524288))
    lgroups = tuple(dcfg.get("landing_groups") or ("portals",))
    rows, jobs = [], []  # ASK-PARALLEL-1105: (result, identifier, landing job index or None) in result order
    for r in results or ():
        if not isinstance(r, Mapping) or not r.get("url"):
            continue
        counts["ddg_results"] += 1
        ident = ddg_identifier(r["url"])
        if ident is None and _is_wikipedia(r["url"]):
            counts["ddg_held"] += 1  # portal only: never fetched, never a candidate
            continue
        job = None
        if ident is None and landing < max_landing and str(r.get("discovery_group") or "") in set(lgroups) \
                and deadline - clock() > lt:
            landing += 1
            counts["ddg_landing_get"] += 1
            job = len(jobs)
            jobs.append(str(r["url"]))
        rows.append((r, ident, job))
    dois, skipped = _landing_dois_concurrently(jobs, lt, lb, get=get, url_check=url_check, deadline=deadline,
                                               clock=clock, acq=acq, robots_ok=robots_ok)
    counts["ddg_landing_get"] -= len(skipped)
    for r, ident, job in rows:
        if job is not None:
            d = dois[job]
            ident = {"kind": "doi", "value": d} if d else None
        if ident is None:
            if _landing_url_ok(str(r["url"])):  # DF-3: followable URL-only candidate (no request is made for it)
                counts["ddg_url_only"] += 1
                out.append(_ddg_url_cand(r))
            else:
                counts["ddg_held"] += 1
            continue
        if ident["kind"] == "handle":  # WP-D: no DOI/PMCID to map; a followable URL-only candidate, no request is made
            if _landing_url_ok(str(r["url"])):
                counts["ddg_url_only"] += 1
                out.append(_ddg_url_cand(r))
            else:
                counts["ddg_held"] += 1
            continue
        if ident["kind"] in ("pmcid", "pmid"):
            pending.append((ident, r))
            continue
        doi = ident["value"] if ident["kind"] == "doi" else f"10.48550/arxiv.{ident['value']}"
        out.append(_ddg_cand(r, doi, None, ident["kind"], arxiv_version=ident.get("version")))
    if pending:
        found = {}
        if calls_left >= 1 and deadline - clock() > 1:
            try:
                found = europepmc_ids_lookup([(i["kind"], i["value"]) for i, _ in pending],
                                             max(1.0, min(lt, deadline - clock())), ecfg, transport=transport)
            except Exception:  # noqa: BLE001
                found = {}
        for ident, r in pending:
            rec = found.get(ident["value"] if ident["kind"] == "pmcid" else "pmid:" + ident["value"])
            if rec is None:
                counts["ddg_held"] += 1
                continue
            out.append(_ddg_cand(r, rec["doi"], rec.get("pmcid"), ident["kind"]))
    counts["ddg_mapped"] = len(out) - counts["ddg_url_only"]
    return out, counts


def _ddg_cand(r: Mapping, doi: str, pmcid, kind: str, arxiv_version: str | None = None) -> dict:
    c = {"url": f"https://doi.org/{doi}", "title": " ".join(str(r.get("title") or "").split()), "snippet": "",
         "doi": doi, "provider": "ddg", "acq_source": "ddg", "ddg_id_kind": kind, "discovered_url": r.get("url"),
         **({"pmcid": pmcid, "oa": {"pmcid": pmcid}} if pmcid else {})}
    if kind == "arxiv":  # DF-2: the exact arXiv id+version the result named, handed to the arXiv route
        aid = doi.split("arxiv.", 1)[1]
        c["arxiv_id"] = aid + (arxiv_version or "")
        c["identifier"] = "arXiv:" + c["arxiv_id"]  # read by arxiv_fulltext.arxiv_ident (version-bound route)
    return c


def _ddg_url_cand(r: Mapping) -> dict:
    """DF-3: a URL-only DDG candidate (no identifier yet): never fetched here; followed only via follow_url."""
    return {"url": str(r.get("url")), "title": " ".join(str(r.get("title") or "").split()), "snippet": "",
            "doi": None, "provider": "ddg", "acq_source": "ddg", "ddg_id_kind": "url", "discovered_url": r.get("url")}


def acq_source(c: Mapping) -> str:
    if c.get("acq_source") in ("ddg", "wikipedia_portal", "discovered", "reference_portal"):  # WIKIREF-1022; P0-5; ROUTES-1108
        return c["acq_source"]
    return "epmc" if c.get("provider") == "europepmc" else "openalex"


def acq_rank_tier(c: Mapping, arxiv_route: bool = False) -> int:
    """Ranking tier (lower first): DuckDuckGo-mapped candidates first (owner 2026-10-21), everything else in between.
    arXiv-mapped ones are last (tier 2) unless `arxiv_route` is True (config arxiv_route.enabled) AND the candidate is one
    the route can take (an arXiv id, or an arxiv.org / 10.48550/arXiv DOI): then tier 1 (NONASK-1103 WP-F0). The licence
    is still decided at fetch by the arXiv route (non-CC is held). arxiv_route False = the previous tiers exactly."""
    if c.get("acq_source") == "ddg":
        if c.get("ddg_id_kind") == "arxiv":
            return 1 if arxiv_route else 2
        return 0
    if c.get("acq_source") == "wikipedia_portal" or (c.get("acq_source") == "reference_portal" and c.get("ref_id_kind")):
        # WIKIREF-1022 edit 2 / ROUTES-1108 (page-extracted refs): arXiv refs last, others scholarly tier; OpenAlex
        # reference-list records (no ref_id_kind) take the generic arXiv rule below (SECREV L6)
        return (1 if arxiv_route else 2) if c.get("ref_id_kind") == "arxiv" else 1
    if _arxiv_identified(c) and not _cand_pmcid(c):  # ROUTE-R3 (D3): no full-text route unless the arXiv route is on
        return 1 if (arxiv_route and _arxiv_routable(c)) else 2
    return 1


_ARXIV_MARK = re.compile(r"arxiv\.org/(?:abs|pdf)/|10\.48550/arxiv\.|\barxiv:", re.I)


def _arxiv_routable(c: Mapping) -> bool:
    """True when the arXiv route can take the candidate: an arXiv id, or an arxiv.org URL / 10.48550/arXiv DOI."""
    d = c.get("discovery") if isinstance(c.get("discovery"), Mapping) else {}
    return bool(c.get("arxiv_id") or _ARXIV_MARK.search(" ".join(str(x or "") for x in (
        c.get("doi"), c.get("url"), c.get("discovered_url"), d.get("url"), d.get("record_id")))))


def _arxiv_identified(c: Mapping) -> bool:
    d = c.get("discovery") if isinstance(c.get("discovery"), Mapping) else {}
    if c.get("arxiv_id") or "arxiv" in (str(c.get("provider") or "").lower(), str(d.get("service") or "").lower()):
        return True
    return bool(_ARXIV_MARK.search(" ".join(str(x or "") for x in (c.get("doi"), c.get("url"), d.get("url"),
                                                                    d.get("record_id")))))


def _cand_pmcid(c: Mapping):
    oa = c.get("oa") if isinstance(c.get("oa"), Mapping) else {}
    return _pmcid_of(c.get("pmcid") or oa.get("pmcid"))


def acq_portal(c: Mapping) -> str:
    """ROUTE-R3: the portal that found one candidate (acq_source, but arXiv/DOAJ-listed records keep their own name)."""
    prov = str(c.get("provider") or "").lower()
    return prov if prov in ("arxiv", "doaj") and acq_source(c) == "openalex" else acq_source(c)


def acq_portals(c: Mapping) -> list:
    """ROUTE-R3: every portal that found this candidate, first finder first (merge_sources keeps the list)."""
    return list(c.get("portals") or [acq_portal(c)])


def merge_sources(*lists) -> list:
    """Merge candidate lists in order, de-duplicated by DOI or PMCID; the first occurrence keeps its place and source,
    and absorbs a later duplicate's OA metadata (Europe PMC full text marks epmc_ft) and empty snippet/title."""
    from vera import work_key  # OA-3b: canonical work keys (config/work_key.json mode off|shadow|on)
    wk_mode, would, wk_index = work_key.mode(), 0, {}
    out, index = [], {}
    for lst in lists:
        for c in lst or ():
            if not isinstance(c, dict):
                continue
            ids = _cand_ids(c)
            wk = work_key.extra_keys(c, ids) if wk_mode != "off" else set()
            if wk_mode == "on":
                ids = ids | {"wk:" + k for k in wk}
            hit = next((index[k] for k in ids if k in index), None)
            if wk_mode == "shadow":
                if hit is None and any(k in wk_index for k in wk):
                    would += 1
                for k in wk:
                    wk_index.setdefault(k, True)
            tk = title_key(c)
            if hit is None and tk and tk in index and _same_work_ids(index[tk], c):
                hit = index[tk]  # same normalised title, no conflicting identifier: the same work (F0 follow-up)
            if hit is None:
                if wk_mode == "on":  # OA-3c: the first sighting is the first discovery event
                    work_key.note_discovery(c, c)
                out.append(c)
                for k in ids:
                    index[k] = c
                if tk:
                    index.setdefault(tk, c)
                continue
            oa = dict(hit.get("oa") or {}) if isinstance(hit.get("oa"), Mapping) else {}
            other = c.get("oa") if isinstance(c.get("oa"), Mapping) else {}
            for k, v in other.items():
                if oa.get(k) is None:
                    oa[k] = v
            if other.get("epmc_ft"):
                oa["epmc_ft"] = True
            if oa:
                hit["oa"] = oa
            try:  # PORT-P2: keep every portal that found this document (attribution only; order/source unchanged)
                hit["found_by"] = list(dict.fromkeys([*(hit.get("found_by") or [acq_source(hit)]), acq_source(c)]))
            except Exception:  # noqa: BLE001
                pass
            for k in ("snippet", "title", "doi", "pmcid", "openalex_id", "arxiv_id", "doaj_id"):  # ROUTE-R3: ids kept
                if not hit.get(k) and c.get(k):
                    hit[k] = c[k]
            hit["portals"] = list(dict.fromkeys(acq_portals(hit) + acq_portals(c)))  # ROUTE-R3: provenance kept
            if c.get("part") is not None:  # NONASK-1103 S3: every sub-query that found this work
                hit["parts"] = sorted(set(hit.get("parts") or ([hit["part"]] if hit.get("part") is not None else []))
                                      | {c["part"]})
            if wk_mode == "on":  # OA-3c: the losing duplicate's discovery event is kept, not dropped
                work_key.note_discovery(hit, c)
            for k in ids:
                index.setdefault(k, hit)
    if wk_mode == "shadow":
        work_key.log_shadow("merge_sources", would)
    return out


def merge_candidates(primary: list, extra: list) -> list:
    return merge_sources(primary, extra)


class _QuotaGate:
    """Process-wide: after an OpenAlex 429/quota error the acquisition search treats OpenAlex as unavailable until the
    cooldown ends (search.openalex_quota_cooldown_s). Nothing waits on it; DDG and Europe PMC still run."""
    def __init__(self):
        self.until, self.lock = 0.0, threading.Lock()

    def blocked(self, now: float) -> bool:
        with self.lock:
            return now < self.until

    def block(self, now: float, s: float) -> None:
        with self.lock:
            self.until = max(self.until, now + max(0.0, s))

    def reset(self) -> None:
        with self.lock:
            self.until = 0.0


_QUOTA = _QuotaGate()


def _is_rate_skip(e: Exception) -> bool:
    """RATE-1022: a portal skip (vera.rate_limit.RateLimited, or the search chain's 'rate_limited (...; not an
    attempt)' note): nothing was sent, so it is never a 429 and never starts the fallback quota cooldown."""
    from vera import rate_limit
    return isinstance(e, rate_limit.RateLimited) or "not an attempt" in str(e).lower()


_NOT_ATTEMPT = re.compile(r"rate_limited \((\w+); not an attempt\)")
_OWN_CAP = re.compile(r"daily (?:query cap|search budget).{0,40}\bspent\b")


def _own_decision(e: Exception):
    """OUTCOME-LABELS-1108: which of VERA-DAR's OWN decisions (not a provider failure) ended a call, else None:
    "paced" (our limiter's pacing, concurrency or deadline skip), "own_cap" (our daily allowance for the service was
    used up) or "robots" (our robots gate refused). The skip reaches here as a RateLimited or, through the search
    chain, as the note "rate_limited (<reason>; not an attempt)" or "provider unavailable: ... spent ...". A skip
    caused by the provider (cooldown, open breaker) is None: it follows a real refusal."""
    from vera import rate_limit
    t = str(e)
    tl = t.lower()
    if "r-acq-robots" in tl:
        return "robots"
    if isinstance(e, rate_limit.RateLimited):
        reason = e.reason
    else:
        m = _NOT_ATTEMPT.search(t)
        reason = m.group(1) if m else None
    if reason == "daily_cap":
        return "own_cap"
    if reason is not None:
        return None if reason in rate_limit.PROVIDER_BUSY_REASONS else "paced"
    return "own_cap" if _OWN_CAP.search(tl) else None


def _own_pacing(e: Exception) -> bool:
    """F4a: a skip by VERA-DAR's own limiter (pacing, concurrency, deadline), not a provider refusal or cooldown
    after one. Reasons that rate_limit reports as the provider being busy are excluded; nothing was sent either way."""
    return _own_decision(e) == "paced"


def _is_quota_error(e: Exception) -> bool:
    t = str(e).lower()
    return bool(re.search(r"\b429\b", t)) or "rate limit" in t or "budget" in t or "quota" in t


def _default_ddg_search(cfg: Mapping) -> Callable:
    from vera import auto_acquire
    s = dict(cfg.get("search") or {})
    s.update({"providers": [DDG_NAME], "openalex_profile": None})
    return auto_acquire.make_search({**cfg, "search": s})  # the shared /ask DDG instance: its cache, cap and gate


def _wiki_cfg() -> dict:
    """config/wiki_ref_portal.json (fails closed to {"enabled": False}). One seam so the offline suite can keep the
    portal off unless a test opts in."""
    from vera import wiki_ref_portal
    return wiki_ref_portal.load_config()


ARXIV_ALLOWED: contextvars.ContextVar = contextvars.ContextVar("vera_acq_arxiv_allowed", default=True)  # WP-F0: budget slot
ARXIV_STATUS: contextvars.ContextVar = contextvars.ContextVar("vera_acq_arxiv_status", default=None)  # [status, n] handoff
PORTALS: contextvars.ContextVar = contextvars.ContextVar("vera_acq_portals", default=None)
# NONASK-1103 S3: the request's sub-queries [(query text, entities)], set by inline_acquire; None/one part = unchanged.
# NONASK-1103 A6 (review N3): set by the inline round loop for variant rounds 2+: DuckDuckGo (and its id lookup) and
# Wikipedia are skipped, so DDG runs ONCE per question (its daily cap unchanged); variant rounds use OpenAlex, Europe PMC
# and arXiv only. Default False = unchanged.
SCHOLARLY_ONLY: contextvars.ContextVar = contextvars.ContextVar("vera_acq_scholarly_only", default=False)
SUBQUERIES: contextvars.ContextVar = contextvars.ContextVar("vera_acq_subqueries", default=None)
# NONASK-1103 S3: None = today's OpenAlex call (AND profile, then OR retry); "and_only" = the AND call only (one per
# sub-query); "or_only" = the plain OR query only (the question-level retry).
OA_MODE: contextvars.ContextVar = contextvars.ContextVar("vera_acq_oa_mode", default=None)


def subquery_budget(cfg: Mapping, n_parts: int) -> int:
    """NONASK-1103 S3: bounds.max_search_calls_per_job scaled for n sub-queries: cap + per_part*(n-1), where per_part is
    the number of per-part providers (OpenAlex 1, Europe PMC 1 if enabled, arXiv 1 if in search.providers; 3 shipped), so
    7 -> 16 for n = 4. A RECORDED LOOSENING of an existing budget (OD-1027 p3), not a new cap; the rate gates are
    unchanged and the question still makes one job reservation and one DDG call."""
    s = cfg.get("search") or {}
    cap = int((cfg.get("bounds") or {}).get("max_search_calls_per_job", 0) or 0)
    if n_parts <= 1 or cap <= 0 or (s.get("subquery") or {}).get("enabled") is not True:
        return cap
    per = 1 + (1 if (s.get("europepmc") or {}).get("enabled") is True else 0) \
        + (1 if "arxiv" in set(s.get("providers") or ()) else 0)
    return cap + per * (n_parts - 1)


def record_portal(name: str, status: str, results: int = 0) -> None:
    """FRESH-1023 C3: append one portal attempt {"portal","status","results"} to the active portal_scope() list.
    No-op without a scope; never raises."""
    try:
        lst = PORTALS.get()
        if lst is not None:
            lst.append({"portal": str(name), "status": str(status), "results": int(results or 0)})
        rt.event("search", "portal", counts={"results": int(results or 0)},
                 detail={"provider": str(name), "status": str(status)})
    except Exception:  # noqa: BLE001
        pass


@contextlib.contextmanager
def portal_scope():
    """Set a fresh portal-attempt list for this context and yield it."""
    lst: list = []
    tok = PORTALS.set(lst)
    try:
        yield lst
    finally:
        PORTALS.reset(tok)


def _submit_in_order(pool, fn, *a, **kw):
    """ROUTE-R3: submit fn in a copy of this context and wait (<= 1 s) until its worker has STARTED, so the portals
    start in submission order (DDG first, owner rule YIELD-1021) and still run concurrently."""
    ev, ctx = threading.Event(), contextvars.copy_context()

    def run():
        ev.set()
        return ctx.run(fn, *a, **kw)
    f = pool.submit(run)
    ev.wait(1.0)
    return f


def _search_parts(base_search: Callable, parts: list, B: Mapping, question: str, n: int, t_end: float,
                  clock: Callable, arx_hand: list) -> list:
    """NONASK-1103 S3: the OpenAlex/arXiv side for several sub-queries. One OpenAlex AND call per part (the first
    B["n_oa"] parts), arXiv for the first B["n_arx"] parts, each result tagged part=k; then ONE question-level OpenAlex
    OR retry (plain OR query, whole question) only when no part found anything. Raises the first error only when every
    call failed and nothing was found (the caller's all-providers-failed rule then applies)."""
    out, errs, calls_ok = [], [], 0
    arx_n, arx_status = 0, []
    for k, (ptxt, _ents) in enumerate(parts[:max(1, B["n_oa"])]):
        ta = ARXIV_ALLOWED.set(k < B["n_arx"])
        ts = ARXIV_STATUS.set(hand := [])
        tm = OA_MODE.set("and_only")
        try:
            got = list(base_search(ptxt, n, max(0.1, t_end - clock())) or [])
            calls_ok += 1
        except Exception as e:  # noqa: BLE001
            errs.append(e)
            got = []
        finally:
            OA_MODE.reset(tm)
            ARXIV_ALLOWED.reset(ta)
            ARXIV_STATUS.reset(ts)
        if hand:
            arx_status.append(hand[0])
            arx_n += int(hand[1])
        out.extend({**c, "part": k} for c in got if isinstance(c, dict))
    if not out and B.get("or_retry") and t_end - clock() > 0.1:
        ta = ARXIV_ALLOWED.set(False)
        tm = OA_MODE.set("or_only")
        try:
            out = list(base_search(question, n, max(0.1, t_end - clock())) or [])
            calls_ok += 1
        except Exception as e:  # noqa: BLE001
            errs.append(e)
        finally:
            OA_MODE.reset(tm)
            ARXIV_ALLOWED.reset(ta)
    if arx_status:
        arx_hand[:] = ["completed" if "completed" in arx_status else arx_status[0], arx_n]
    if errs and not out and not calls_ok:
        raise errs[0]
    return out


def acquisition_search(base_search: Callable, cfg: Mapping, *, transport: Callable | None = None,
                       terms_fn: Callable | None = None, ddg_search: Callable | None = None,
                       get: Callable | None = None, url_check: Callable | None = None,
                       clock: Callable | None = None, wiki_get: Callable | None = None) -> Callable:
    """Wrap the acquisition job's search (vera.auto_acquire.make_search): DuckDuckGo Lite (config search.ddg_first),
    OpenAlex and Europe PMC run CONCURRENTLY within the same timeout, merged DDG first. Search calls are counted
    against bounds.max_search_calls_per_job: OpenAlex 2 (AND profile + OR retry; else 1), then DuckDuckGo 1, Europe
    PMC 1, Wikipedia 1, arXiv 1 (when in search.providers), the DDG id lookup 1, each only while the cap leaves a call. Any provider failing returns the others' results;
    all failing re-raises the OpenAlex error. Landing-page GETs are fetches (dcfg.max_landing_fetches), not searches."""
    import time as _time
    clock = clock or _time.monotonic
    s = cfg.get("search") or {}
    ecfg = s.get("europepmc") or {}
    dcfg = s.get("ddg_first") or {}
    from vera import wiki_ref_portal  # WIKIREF-1022 edit 1: Wikipedia as a reference portal (never evidence)
    wcfg = _wiki_cfg()
    wiki_on = wiki_ref_portal.enabled(wcfg)
    prof_on = bool(s.get("openalex_profile"))

    def budget(cap: int, n_parts: int = 1) -> dict:
        b = _budget(cap, n_parts)
        if SCHOLARLY_ONLY.get():  # NONASK-1103 N3: variant rounds never call DDG, its id lookup or Wikipedia
            b = dict(b, ddg=False, wiki=False, lookup=0)
        return b

    def _budget(cap: int, n_parts: int = 1) -> dict:
        """The slots for `cap` search calls. One part: the legacy order (OpenAlex 2, DDG, Europe PMC, Wikipedia, arXiv,
        then the id lookup). NONASK-1103 S3, n parts: the QUESTION-level slots first (OpenAlex OR retry, id lookup,
        Wikipedia, DDG, one each), then one OpenAlex call per part, then Europe PMC per part, then arXiv per part
        (round-robin: every part gets its OpenAlex call before any gets a second provider)."""
        if n_parts <= 1:
            left = cap - (2 if prof_on else 1)
            u_ddg = dcfg.get("enabled") is True and left >= 1
            left -= 1 if u_ddg else 0
            u_epmc = ecfg.get("enabled") is True and left >= 1
            left -= 1 if u_epmc else 0
            u_wiki = wiki_on and left >= 1
            left -= 1 if u_wiki else 0
            u_arx = "arxiv" in set(s.get("providers") or ()) and left >= 1  # WP-F0: arXiv counted, after DDG/EPMC/Wikipedia
            left -= 1 if u_arx else 0
            return {"ddg": u_ddg, "epmc": u_epmc, "wiki": u_wiki, "arxiv": u_arx, "lookup": left,
                    "n_oa": 1, "n_epmc": 1 if u_epmc else 0, "n_arx": 1 if u_arx else 0, "or_retry": prof_on}
        left = cap
        or_retry = prof_on and left >= 1
        left -= 1 if or_retry else 0
        lookup = 1 if left >= 1 else 0
        left -= lookup
        u_wiki = wiki_on and left >= 1
        left -= 1 if u_wiki else 0
        u_ddg = dcfg.get("enabled") is True and left >= 1
        left -= 1 if u_ddg else 0
        n_oa = min(n_parts, max(0, left))
        left -= n_oa
        n_epmc = min(n_parts, max(0, left)) if ecfg.get("enabled") is True else 0
        left -= n_epmc
        n_arx = min(n_parts, max(0, left)) if "arxiv" in set(s.get("providers") or ()) else 0
        left -= n_arx
        return {"ddg": u_ddg, "epmc": n_epmc > 0, "wiki": u_wiki, "arxiv": n_arx > 0, "lookup": lookup + max(0, left),
                "n_oa": n_oa, "n_epmc": n_epmc, "n_arx": n_arx, "or_retry": or_retry}

    cap0 = int((cfg.get("bounds") or {}).get("max_search_calls_per_job", 0) or 0)
    B0 = budget(cap0)
    use_wiki = B0["wiki"]
    if wiki_on and wiki_get is None:
        from vera import rate_limit, selective_fetch
        # WIKI-REDESIGN-1107: article pages only (robots-checked in the portal); pinned GET, never a redirect
        from vera import api_permissions as _ap_wiki  # 1108b (4): fixed portal for pages; an API hop uses its own
        wiki_get = _ap_wiki.limited_transport(selective_fetch.pinned_get, fixed_portal="wikipedia")

    shape_state = {"recency": True, "review": False, "pending": False}  # QUERY-SHAPE-1108: set in round 0, review spent by the next round

    def search(question, n, timeout_s):
        import concurrent.futures as cf
        t_end = clock() + float(timeout_s)
        futs = {}
        # NONASK-1103 S3: with several sub-queries the budget is re-planned for this call (question-level slots first)
        parts = list(SUBQUERIES.get() or ())
        multi = len(parts) > 1 and (s.get("subquery") or {}).get("enabled") is True
        B = budget(subquery_budget(cfg, len(parts)), len(parts)) if multi else (budget(cap0) if SCHOLARLY_ONLY.get() else B0)
        use_ddg, use_epmc, use_wiki, use_arxiv, lookup_left = B["ddg"], B["epmc"], B["wiki"], B["arxiv"], B["lookup"]
        terms, profile = None, None
        pool = cf.ThreadPoolExecutor(max_workers=3, thread_name_prefix="acq-search")
        try:
            if use_ddg:  # RATE-1022 C2: each future runs in a copy of this context (the inline rate-limit deadline)
                futs["ddg"] = _submit_in_order(pool, ddg_search or _default_ddg_search(cfg), question, n, timeout_s)
            if use_epmc:
                try:
                    terms, profile = (terms_fn or (lambda: _default_terms_and_profile(cfg)))()
                except Exception:  # noqa: BLE001
                    terms, profile = None, None
                if multi:  # one Europe PMC call per part (the first n_epmc parts), the entity central in each query
                    def epmc_parts():
                        res, errs = [], []
                        for k, (ptxt, _ents) in enumerate(parts[:B["n_epmc"]]):
                            try:
                                got_k = list(europepmc_search(ptxt, int(ecfg.get("page_size", n)), float(timeout_s), ecfg,
                                                              terms_cfg=terms, profile=profile, transport=transport) or [])
                            except Exception as e:  # noqa: BLE001  one part failing never loses the others
                                errs.append(e)
                                continue
                            res.extend({**c, "part": k} for c in got_k if isinstance(c, dict))
                        if errs and not res:
                            raise errs[0]
                        return res
                    futs["epmc"] = _submit_in_order(pool, epmc_parts)
                else:
                    futs["epmc"] = _submit_in_order(pool, europepmc_search, question,
                                               int(ecfg.get("page_size", n)), float(timeout_s),
                                               ecfg, terms_cfg=terms, profile=profile, transport=transport)
            if use_wiki:
                futs["wiki"] = _submit_in_order(pool, wiki_ref_portal.make_portal_search(wcfg, http_get=wiki_get),
                                           question, n, timeout_s)
        finally:
            pool.shutdown(wait=False)
        counts = {}
        base, base_err = [], None
        futs_order = list(futs)  # ROUTE-R3: actual start order (dict insertion = submission = start order)
        st_oa = "completed"
        arx_hand: list = []
        if _QUOTA.blocked(clock()):  # a recent OpenAlex 429/quota error: unavailable, no call (no retry storm)
            counts["openalex_skipped_quota"] = 1
            st_oa = "skipped_quota"
            base_err = ProviderUnavailable("openalex quota cooldown")
        else:
            tok = OPENALEX_OA_FIELDS.set(True)
            futs_order.append("openalex")  # ROUTE-R3: called after the concurrent portals started
            try:
                if multi:
                    base = _search_parts(base_search, parts, B, question, n, t_end, clock, arx_hand)
                else:
                    from vera import query_shape
                    use_review = False
                    if not SCHOLARLY_ONLY.get():  # round 0: the question itself decides the shape for this question
                        shape_state.update(query_shape.shape(question))
                        shape_state["pending"] = bool(shape_state["review"])
                    elif shape_state["pending"]:
                        # QUERY-SHAPE-1108: the first variant round (scholarly-only) carries the review filter, in the
                        # slot of the duplicate first variant the generator no longer emits: the OpenAlex call total
                        # per question is unchanged. Variant rounds receive the variant string, not the question.
                        use_review, shape_state["pending"] = True, False
                    shape_tok = OPENALEX_SHAPE.set({"recency": shape_state["recency"], "review": use_review})
                    atok = ARXIV_ALLOWED.set(use_arxiv)  # make_search calls arXiv only when the budget left a slot
                    stok = ARXIV_STATUS.set(arx_hand)  # make_search reports the arXiv call here: [status, n]
                    try:
                        base = list(base_search(question, n, max(0.1, t_end - clock())) or [])  # P0DTA-PRIO D3: capped at t_end
                    finally:
                        ARXIV_ALLOWED.reset(atok)
                        ARXIV_STATUS.reset(stok)
                        OPENALEX_SHAPE.reset(shape_tok)
            except Exception as e:  # noqa: BLE001
                base_err = e
                st_oa = "timeout" if _is_timeout(e) else "failed"
                own = _own_decision(e)
                if own in ("own_cap", "robots"):  # OUTCOME-LABELS-1108: our own decision, never "a search service failed"
                    st_oa = own
                elif _is_rate_skip(e):  # RATE-1022 D1: the openalex adapter skipped (nothing sent); it owns the cooldown
                    counts["openalex_skipped_quota"] = 1
                    st_oa = "paced" if _own_pacing(e) else "skipped_quota"
                elif _is_quota_error(e):
                    counts["openalex_429"] = 1
                    st_oa = "quota"
                    # INT-1022-B: kept as a FALLBACK (spec D1 option 1) for injected searches that bypass the adapter;
                    # on the live path the adapter cools down from Retry-After/headers as well.
                    _QUOTA.block(clock(), float(s.get("openalex_quota_cooldown_s", 900)))
            finally:
                OPENALEX_OA_FIELDS.reset(tok)
        got = {}
        epmc_ok = False
        st_f = {}
        for k, f in futs.items():
            try:
                got[k] = list(f.result(timeout=max(0.1, t_end - clock())) or [])
                epmc_ok = epmc_ok or k == "epmc"
                st_f[k] = "completed"
            except Exception as e:  # noqa: BLE001  (timeout, transport or parse error: the other sources stand)
                got[k] = []
                own = _own_decision(e)
                if own == "paced":  # F4a: our own limiter skipped the call; the provider refused nothing
                    counts[f"{k}_paced"] = 1
                    st_f[k] = "paced"
                elif own:  # OUTCOME-LABELS-1108: own_cap (our daily allowance) | robots (our compliance gate)
                    st_f[k] = own
                elif _is_quota_error(e):
                    counts[f"{k}_429"] = 1
                    st_f[k] = "quota"
                elif _is_timeout(e):  # DDGFIX-1023 DF-5: requests ReadTimeout/ConnectTimeout too
                    st_f[k] = "timeout"
                else:
                    st_f[k] = "failed"
        sfx = str(ecfg.get("query_suffix") or "")
        if (use_epmc and epmc_ok and not got.get("epmc") and not got.get("ddg") and lookup_left >= 1
                and t_end - clock() > 1 and _or_differs(question, terms, profile, sfx)):
            # DAR-TRACE-1022: the AND query SUCCEEDED with 0 hits -> ONE Europe PMC retry with the plain OR query
            # (profile None), in the id-lookup slot, used only when DDG returned nothing (so no lookup can need it).
            # Review M3: never after a failed first call (429/error/timeout), never when the OR query is identical.
            try:
                got["epmc"] = list(europepmc_search(question, int(ecfg.get("page_size", n)),
                                                    max(1.0, t_end - clock()), ecfg, terms_cfg=terms, profile=None,
                                                    transport=transport) or [])
                counts["epmc_or_retry"] = 1
            except Exception as e:  # noqa: BLE001
                if _is_quota_error(e):
                    counts["epmc_429"] = 1
        ddg = []
        if got.get("ddg"):
            ddg, dcounts = ddg_to_candidates(got["ddg"], ecfg, dcfg, deadline=t_end, clock=clock,
                                            calls_left=lookup_left, transport=transport, get=get, url_check=url_check,
                                            acq=cfg)
            counts.update(dcounts)
        try:  # FRESH-1023 C3: each configured portal exactly once, from this (the main) thread. ROUTE-R3: in actual
            # start order; an enabled portal left without a search-call slot is not_attempted (never "completed")
            names = {"ddg": "ddg", "epmc": "europepmc", "wiki": "wikipedia", "openalex": "openalex", "arxiv": "arxiv"}
            stat = {**st_f, "openalex": st_oa}
            nres = {**{k: len(got.get(k) or ()) for k in futs}, "openalex": len(base)}
            if arx_hand:  # WP-F0: arXiv was called (after OpenAlex, inside make_search): recorded in call order
                futs_order.append("arxiv")
                stat["arxiv"], nres["arxiv"] = arx_hand[0], int(arx_hand[1])
            for key in futs_order + ([] if "openalex" in futs_order else ["openalex"]):
                record_portal(names[key], stat.get(key, "failed"), nres.get(key, 0))
            for key, enabled_cfg, used in (("ddg", dcfg.get("enabled") is True, use_ddg),
                                           ("epmc", ecfg.get("enabled") is True, use_epmc),
                                           ("wiki", wiki_ref_portal.enabled(wcfg), use_wiki),
                                           ("arxiv", "arxiv" in set(s.get("providers") or ()), use_arxiv)):
                if enabled_cfg and not used and not (key in ("ddg", "wiki") and SCHOLARLY_ONLY.get()):
                    # NOANSWER-PAGE-1108 D8: a variant round skips DDG and Wikipedia BY DESIGN (SCHOLARLY_ONLY); that
                    # is not a portal left without a search-call slot, so it is not recorded as not_attempted (which
                    # research_record reads as "the search budget ran out").
                    record_portal(names[key], "not_attempted", 0)
        except Exception:  # noqa: BLE001
            pass
        if base_err is not None and not ddg and not got.get("epmc") and not got.get("wiki"):
            raise base_err
        merged = merge_sources(ddg, base, got.get("epmc"), got.get("wiki"))  # DDG first; portal refs last
        if not SCHOLARLY_ONLY.get():  # ROUTES-1108: generic reference portal, once per question (first round only);
            from vera import reference_portal  # off = no work; shadow = background, logs only; on = merged last
            merged = reference_portal.extend(question, merged, base=base, ddg_raw=got.get("ddg"), clock=clock,
                                             deadline=t_end)
        from vera import arxiv_portal  # ARXIV-PORTAL-1110: arXiv seeds -> published version + references (metadata only)
        merged = arxiv_portal.extend(question, merged, clock=clock, deadline=t_end)
        by = {}
        for c in merged:
            by[acq_source(c)] = by.get(acq_source(c), 0) + 1
        import logging
        logging.getLogger("vera.auto_acquire").info(
            "autoacq sources %s", " ".join(f"{k}={v}" for k, v in sorted({**counts, **{f"cand_{k}": v for k, v in by.items()},
                                                                         "openalex_raw": len(base),
                                                                         "epmc_raw": len(got.get("epmc") or ()),
                                                                         "wiki_raw": len(got.get("wiki") or ())}.items())))
        try:  # RATE-1022 E2: portal names and counters only
            from vera import rate_limit
            logging.getLogger("vera.auto_acquire").info("autoacq %s", rate_limit.get_registry().log_line())
        except Exception:  # noqa: BLE001  telemetry never breaks the search
            pass
        return merged
    return search


# ----------------------------------------------------------------------------------------- DOAJ
# item #71 plan section 10 (R10-a). Keyless. DOAJ's API docs state a limit of two requests per second; the gate is
# 0.6 s. DOAJ's terms put article metadata under a CC0 waiver (recorded as a METADATA licence only). The query is
# plain terms (core + extra joined by spaces): a quoted phrase with AND returned 0 hits in the plan's probe.
DOAJ_NAME = "doaj"
DOAJ_METADATA_LICENCE = {"id": "CC0-1.0", "url": "https://creativecommons.org/publicdomain/zero/1.0/",
                         "source": "doaj_metadata_cc0_waiver"}
COST_BASIS_DOAJ = "keyless_doaj"


def doaj_default_transport(url: str, params: dict, headers: dict, timeout: float):
    """THE one real network call for DOAJ. Tests patch this (and requests.get) to raise."""
    import requests
    from vera import api_permissions  # API-ALLOW-1107: permission gate + the entry's rate portal (RATE-1022 A4)
    return api_permissions.call(requests.get, url, params=params, headers=headers, timeout=timeout)


def _doaj_words(question: str, terms_cfg: Mapping | None) -> tuple[list[str], list[str], str]:
    """-> (core words, extra words, mode); falls back to the stopword-filtered question, never empty."""
    if terms_cfg is not None:
        qt = build_query_terms(question, terms_cfg)
        if not qt.empty:
            return list(qt.core), list(qt.extra), qt.mode
        mode = "fallback_legacy"
    else:
        mode = "legacy_no_terms_config"
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]*", question.lower()) if w not in QUERY_STOPWORDS and len(w) > 1][:8]
    if not words:
        raise ValueError("question has no searchable term after stopword removal")
    return [], words, mode


class DOAJProvider:
    """DOAJ article search, keyless, metadata + abstract only. `settings` is config search.doaj."""
    name = DOAJ_NAME
    required_env: tuple[str, ...] = ()

    def __init__(self, settings: Mapping, *, transport: Callable | None = None, query_terms: Mapping | None = None,
                 today: Callable[[], str] | None = None, sleep: Callable[[float], None] | None = None,
                 clock: Callable[[], float] | None = None):
        import time
        self.s = settings
        self.endpoint = settings["endpoint"].rstrip("/")
        self.max_bytes = int(settings["max_response_bytes"])
        self.snippet_max = int(settings["snippet_max_chars"])
        self.year_from, self.year_to = int(settings["year_from"]), int(settings["year_to"])
        self.gate = RateGate(float(settings["min_interval_s"]))
        self._transport, self.query_terms = transport, query_terms
        self._sleep, self._clock = sleep or time.sleep, clock or time.monotonic
        self._budget = int(settings["daily_search_budget"])
        self._today = today or (lambda: datetime.now(timezone.utc).date().isoformat())
        self._day, self._used = None, 0
        self.last_query: dict | None = None

    def _spend(self) -> None:
        day = self._today()
        if day != self._day:
            self._day, self._used = day, 0
        if self._used >= self._budget:
            raise ProviderUnavailable(f"doaj daily search budget of {self._budget} spent for {day} "
                                      "(per-process guard; non-retryable)")
        self._used += 1

    def _get(self, words: list[str], n: int, timeout_s: float):
        from urllib.parse import quote
        self._spend()  # every HTTP attempt counts
        url = f"{self.endpoint}/{quote(' '.join(words), safe='')}"
        tx = self._transport or doaj_default_transport
        return tx(url, {"pageSize": n}, {"User-Agent": USER_AGENT}, timeout_s)

    @staticmethod
    def _zero_hits(resp) -> bool:
        if getattr(resp, "status_code", None) != 200:
            return False
        try:
            doc = strict_json.loads(resp.text or "")
        except ValueError:
            return False
        return isinstance(doc, dict) and isinstance(doc.get("results"), list) and not doc["results"]

    def call(self, query: str, n: int, timeout_s: float):
        core, extra, mode = _doaj_words(query, self.query_terms)
        self.last_query = {"query": " ".join(core + extra), "mode": mode}
        resp = self._get(core + extra, n, timeout_s)
        if core and extra and self._zero_hits(resp):
            # R10-a: one retry with the core terms only, recorded; the second request keeps the gate spacing
            self.gate.stamp(self._clock())
            wait = self.gate.wait_needed(self._clock())
            if wait > 0:
                from vera import progress as _prog  # NOISY-UI
                with _prog.waiting(_prog.ACT_RATE, wait):
                    self._sleep(wait)
            self.last_query["note"] = "0 hits for core+extra terms; retried once with core terms only"
            self.last_query["retry_query"] = " ".join(core)
            resp = self._get(core, n, timeout_s)
        return resp

    def parse(self, resp, n: int) -> list[dict]:
        body = resp.text or ""
        if len(body.encode("utf-8", "ignore")) > self.max_bytes:
            raise ProviderResponseError(f"doaj response over {self.max_bytes} bytes; refused before parsing")
        try:
            doc = strict_json.loads(body)
        except ValueError as e:
            raise ProviderResponseError(f"doaj response is not JSON: {e}") from None
        if not isinstance(doc, dict) or not isinstance(doc.get("results"), list):
            raise ProviderResponseError("doaj response has no results list: "
                                        + (" ".join(str(doc.get("error") or doc.get("message") or "")[:200].split())
                                           if isinstance(doc, dict) else "not an object"))
        out = []
        for rec in doc["results"]:
            bib = rec.get("bibjson") if isinstance(rec, dict) else None
            if not isinstance(bib, dict):
                continue
            try:
                year = int(str(bib.get("year") or "").strip())
            except ValueError:
                continue  # no usable year: cannot be shown to be inside the configured range, so dropped
            if not self.year_from <= year <= self.year_to:
                continue
            month = str(bib.get("month") or "").strip()
            published = f"{year:04d}-{int(month):02d}" if month.isdigit() and 1 <= int(month) <= 12 else f"{year:04d}"
            ids = [i for i in (bib.get("identifier") or []) if isinstance(i, dict)]
            doi = next((str(i.get("id")).strip() for i in ids if str(i.get("type")).lower() == "doi" and i.get("id")), None)
            fulltext = next((str(link.get("url")).strip() for link in (bib.get("link") or [])
                             if isinstance(link, dict) and link.get("type") == "fulltext" and link.get("url")), None)
            url = f"https://doi.org/{doi}" if doi else fulltext
            if not url:
                continue
            journal = bib.get("journal") if isinstance(bib.get("journal"), dict) else {}
            abstract = " ".join(str(bib.get("abstract") or "").split())
            if len(abstract) > self.snippet_max:
                abstract = abstract[:self.snippet_max].rstrip() + "..."
            lang = journal.get("language")
            out.append({
                "url": url,
                "title": " ".join(str(bib.get("title") or "").split()),
                "snippet": abstract,
                "doi": doi,
                "published": published,
                "year": year,
                "publisher": (str(journal.get("title") or "").strip() or None),
                "source_type": "journal-article",
                "language": lang if lang else None,
                "doaj_id": (str(rec.get("id") or "").strip() or None),
                "licence": licence_record(metadata=DOAJ_METADATA_LICENCE),
                "stored_text_kind": "abstract_metadata",
                "provider_endpoint": self.endpoint,
            })
            if len(out) >= n:
                break
        return out

    def cost_for(self, resp):
        return 0.0, True, COST_BASIS_DOAJ

    def failure_basis(self) -> str:
        return COST_BASIS_DOAJ


# ----------------------------------------------------------------------------------------- DuckDuckGo Lite
# DISCOVERY ONLY (item #71 plan section 7a, R1 D-71-1b): DuckDuckGo results are used to FIND source URLs (news
# articles, journal portals). Only the real target URL, the target page's own headline and its domain are kept;
# DuckDuckGo snippets, pages and ranking are never kept or shown. Its params page says the parameters are
# "intended for individual use", so volume is low (5 s gate, daily cap) and a challenge page is never evaded.
DDG_NAME = "duckduckgo_lite"
DDG_HOST = "duckduckgo.com"
COST_BASIS_DDG = "keyless_discovery_no_published_price"
# Why: structural markers of DuckDuckGo's bot-challenge ("anomaly") page. Deliberately specific so a headline
# that merely contains the word "challenge" is not mistaken for one.
DDG_CHALLENGE_MARKERS = ("anomaly-modal", "anomaly.js", "g-recaptcha", "h-captcha", "cf-challenge", "challenge-form",
                         "unusual traffic", "select all squares", "are you a robot", "are you a human")
DDG_CHALLENGE_NOTE = "duckduckgo challenge page; not evaded"
_IP_LITERAL = re.compile(r"^(\d{1,3}(\.\d{1,3}){3}|\[.*\]|.*:.*)$")


def ddg_default_transport(url: str, data: dict, headers: dict, timeout: float):
    """THE one real network call for DuckDuckGo Lite. Tests patch this (and requests.post) to raise."""
    import requests
    from vera import rate_limit  # RATE-1022 A5: a 200 challenge page is a throttle; 202 via throttle_statuses
    return rate_limit.limited("duckduckgo", requests.post,
                              classify=lambda r: DuckDuckGoLiteProvider._is_challenge(r.text or ""))(
        url, data=data, headers=headers, timeout=timeout)


def _domain_of(host: str) -> str:
    host = host.lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def _domain_matches(domain: str, listed: str) -> bool:
    return domain == listed or domain.endswith("." + listed)


def safe_target_url(url: str) -> str | None:
    """https-only, no credentials, no IP literal, no odd port, no single-label or localhost host. A static
    check (no DNS at parse time); fetch re-checks with selective_fetch.is_public_url and discovered URLs are not
    auto-fetched anyway (licence gate defers licence=None)."""
    sp = urlsplit(url.strip())
    host = (sp.hostname or "").lower()
    try:
        port = sp.port
    except ValueError:
        return None
    if (sp.scheme != "https" or not host or sp.username or sp.password or port not in (None, 443)
            or _IP_LITERAL.match(host) or "." not in host or host.endswith((".local", ".internal", ".localhost"))
            or host == "localhost"):
        return None
    return url.strip()


def unwrap_ddg_link(href: str) -> str | None:
    """-> the real target URL, or None for ads, DuckDuckGo's own URLs and anything unsafe."""
    from urllib.parse import parse_qs
    h = (href or "").strip()
    if h.startswith("//"):
        h = "https:" + h
    sp = urlsplit(h)
    host = (sp.hostname or "").lower()
    if host == DDG_HOST or host.endswith("." + DDG_HOST):
        if sp.path.startswith("/y.js") or "ad_domain" in sp.query or "ad_provider" in sp.query or "ad_type" in sp.query:
            return None  # ad
        if not sp.path.startswith("/l/"):
            return None
        target = (parse_qs(sp.query).get("uddg") or [""])[0]
        if not target:
            return None
        h = target
        host = (urlsplit(h).hostname or "").lower()
        if host == DDG_HOST or host.endswith("." + DDG_HOST):
            return None
    return safe_target_url(h)


class _LiteLinks(HTMLParser):
    """Collects (href, text) of <a class="...result-link..."> anchors."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href, self._text = None, []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        a = dict(attrs)
        if "result-link" in (a.get("class") or "").split():
            self._href, self._text = a.get("href") or "", []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join("".join(self._text).split())))
            self._href = None


def build_ddg_queries(question: str, terms_cfg: Mapping | None, s: Mapping) -> list[dict]:
    """-> [{"group", "query"}], at most `max_queries_per_question`, each <= `max_query_chars`. Unquoted core+extra
    terms plus a (site:a OR site:b) group; a group whose sites do not fit is split into chunks (first chunk of
    every group is served before any second chunk). Raises ValueError when nothing searchable remains."""
    if terms_cfg is not None:
        qt = build_query_terms(question, terms_cfg)
        words = list(qt.core) + list(qt.extra)
    else:
        words = []
    if not words:
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]*", question.lower())
                 if w not in QUERY_STOPWORDS and len(w) > 1][:8]
    if not words:
        raise ValueError("question has no searchable term after stopword removal")
    cap, groups = int(s["max_query_chars"]), {g: list(v) for g, v in s["site_groups"].items() if v}
    if not groups:
        raise ValueError("duckduckgo_lite: no site group has any site")
    longest = max(len(x) for v in groups.values() for x in v)
    # drop trailing terms until one site fits beside them ("(site:" + site + ")" and a space)
    while len(" ".join(words)) + 1 + len(f"(site:{'x' * longest})") > cap and len(words) > 1:
        words.pop()
    base = " ".join(words)
    chunks: dict[str, list[list[str]]] = {}
    for g, sites in groups.items():
        cur, out = [], []
        for site in sites:
            trial = cur + [site]
            if cur and len(base + " " + "(" + " OR ".join("site:" + x for x in trial) + ")") > cap:
                out.append(cur)
                cur = [site]
            else:
                cur = trial
        if cur:
            out.append(cur)
        chunks[g] = out
    ordered = []
    for i in range(max(len(v) for v in chunks.values())):
        for g, v in chunks.items():
            if i < len(v):
                ordered.append({"group": g, "query": base + " (" + " OR ".join("site:" + x for x in v[i]) + ")"})
    return ordered[:int(s["max_queries_per_question"])]


class _CachedPart:
    """Stands in for a response when a site-group query is served from the local cache (no request made)."""
    status_code, headers = 200, {}

    def __init__(self, items):
        self.items, self.text = items, ""


def _ddg_cache_dir(settings: Mapping) -> Path | None:
    """Repo-relative cache directory from config (R11-b); None when the config switches the cache off."""
    rel = settings.get("cache_dir")
    return (Path(__file__).resolve().parent.parent / rel) if rel else None


class _DdgParts:
    """Composite response of one DuckDuckGo call (up to max_queries_per_question POSTs)."""

    def __init__(self, parts):
        self.status_code, self.text, self.headers, self.parts = 200, "", {}, parts


class DuckDuckGoLiteProvider:
    """DuckDuckGo Lite as a DISCOVERY provider. `settings` is config search.duckduckgo_lite."""
    name = DDG_NAME
    required_env: tuple[str, ...] = ()

    def __init__(self, settings: Mapping, *, transport: Callable | None = None, query_terms: Mapping | None = None,
                 today: Callable[[], str] | None = None, sleep: Callable[[float], None] | None = None,
                 clock: Callable[[], float] | None = None, cache_dir: Path | None = None,
                 wall_clock: Callable[[], float] | None = None, robots_check: Callable[[str, str], bool] | None = None):
        import time
        self.s = settings
        self.endpoint = settings["endpoint"]
        self.max_bytes = int(settings["max_response_bytes"])
        self.gate = RateGate(float(settings["min_interval_s"]))
        # COMP-1103 (R-050): robots.txt of the host actually queried is checked before each live query (cached per host
        # in selective_fetch). Injected transports (tests) skip the default network check unless a check is injected.
        self._robots_check = robots_check if robots_check is not None else (None if transport is not None else self._default_robots)
        self.cache_dir = Path(cache_dir) if cache_dir else None  # None = no cache (tests inject a tmp dir)
        # ROUTE-R3 (D2): with no cache_dir, an in-process cache (url + title only) for memory_cache_ttl_s (0/absent = off)
        self._mem_ttl_s = float(settings.get("memory_cache_ttl_s") or 0) if self.cache_dir is None else 0.0
        self._mem: dict = {}
        self._ttl_s = float(settings.get("cache_ttl_hours", 72)) * 3600.0
        self._wall = wall_clock or time.time
        self.last_call_cached = False
        self._transport, self.query_terms = transport, query_terms
        self._sleep, self._clock = sleep or time.sleep, clock or time.monotonic
        self._cap = int(settings["daily_query_cap"])
        self._today = today or (lambda: datetime.now(timezone.utc).date().isoformat())
        self._day, self._used = None, 0
        self.last_query: dict | None = None
        self.news_domains = [d for d in settings["site_groups"].get("news", [])]
        self.source_class_by_domain = dict(settings.get("source_class_by_domain") or {})

    def limiter_cooling_down(self) -> bool:
        """NT-2: True while the DuckDuckGo adapter's throttle cooldown is active (the call would be refused unsent)."""
        try:
            from vera import rate_limit
            return bool(rate_limit.get_registry().portal("duckduckgo").cooling_down())
        except Exception:  # noqa: BLE001  unknown state: keep the normal spacing
            return False

    def _spend(self) -> None:
        day = self._today()
        if day != self._day:
            self._day, self._used = day, 0
        if self._used >= self._cap:
            raise ProviderUnavailable(f"duckduckgo_lite daily query cap (quota) of {self._cap} spent for {day} "
                                      "(per-process guard; non-retryable)")
        self._used += 1

    # ---- per-question, per-site-group cache (item #71 R11-b). Key = sha256 of the query string; value = a list of
    # {url, title} only (R11-c: no snippet is ever stored). Corrupt, expired or malformed entries are ignored.
    def _cache_path(self, q: Mapping) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / (hashlib.sha256(q["query"].encode("utf-8")).hexdigest() + ".json")

    def _cache_get(self, q: Mapping) -> list[dict] | None:
        path = self._cache_path(q)
        if path is None:
            hit = self._mem.get(q["query"]) if self._mem_ttl_s > 0 else None
            if hit is None or not (0 <= self._wall() - hit[0] <= self._mem_ttl_s):
                return None
            return [dict(i) for i in hit[1]]
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            stored_at, items = float(doc["stored_at"]), doc["items"]
            if not isinstance(items, list) or self._wall() - stored_at > self._ttl_s or stored_at > self._wall() + 60:
                return None
            out = []
            for it in items:
                if not (isinstance(it, dict) and isinstance(it.get("url"), str) and isinstance(it.get("title"), str)):
                    return None
                url = safe_target_url(it["url"])
                if not url:
                    return None
                out.append({"url": url, "title": it["title"]})
            return out
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _cache_put(self, q: Mapping, items: list[dict]) -> None:
        path = self._cache_path(q)
        if path is None:
            if self._mem_ttl_s > 0:
                if len(self._mem) >= 256:  # bounded: drop the oldest entry
                    self._mem.pop(next(iter(self._mem)), None)
                self._mem[q["query"]] = (self._wall(), [{"url": i["url"], "title": i["title"]} for i in items])
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"stored_at": self._wall(), "query": q["query"],
                                       "items": [{"url": i["url"], "title": i["title"]} for i in items]}),
                           encoding="utf-8")
            tmp.replace(path)
        except OSError as e:  # a cache that cannot be written is recorded, never fatal
            if self.last_query is not None:
                self.last_query.setdefault("errors", []).append(f"cache write failed: {type(e).__name__}: {e}")

    def gate_exempt(self, question: str) -> bool:
        """True when every site-group query for this question is cached, so no request (and no gate wait) is needed."""
        try:
            queries = build_ddg_queries(question, self.query_terms, self.s)
        except ValueError:
            return False
        return bool(queries) and all(self._cache_get(q) is not None for q in queries)

    @staticmethod
    def _default_robots(url: str, user_agent: str) -> bool:
        from vera import selective_fetch as sf
        return sf.robots_allowed(url, http_get=sf.pinned_get, url_check=sf.is_public_url, user_agent=user_agent)

    def call(self, query: str, n: int, timeout_s: float, *, queries: list | None = None):
        # NONASK-1103 S4 (plan 6.3): `queries` = explicit [{"group", "query"}] (the official lane's one site: query);
        # it goes through the same cache, daily cap, rate gate and transport as every other call.
        queries = list(queries) if queries else build_ddg_queries(query, self.query_terms, self.s)
        self.last_query = {"query": [q["query"] for q in queries], "mode": "site_groups"}
        tx = self._transport or ddg_default_transport
        headers = {"User-Agent": self.s["user_agent"]}  # never altered, even after a challenge
        if self._robots_check is not None and any(self._cache_get(q) is None for q in queries):
            try:
                allowed = bool(self._robots_check(self.endpoint, self.s["user_agent"]))
            except Exception:  # noqa: BLE001  fail closed
                allowed = False
            if not allowed:
                self.last_query["cause"] = "R-ACQ-ROBOTS"
                raise ProviderUnavailable("R-ACQ-ROBOTS: robots.txt of the DuckDuckGo host disallows the query path "
                                          "or could not be read; DuckDuckGo skipped")
        parts, live = [], 0
        self.last_call_cached = False
        for i, q in enumerate(queries):
            hit = self._cache_get(q)
            if hit is not None:  # served locally: no request, no cap spend, no gate wait
                parts.append((q, _CachedPart(hit)))
                self.last_query["cache_hits"] = self.last_query.get("cache_hits", 0) + 1
                continue
            if live:  # the caller's gate covers the first POST; later POSTs keep the same spacing
                self.gate.stamp(self._clock())
                wait = self.gate.wait_needed(self._clock())
                if wait > 0:
                    from vera import progress as _prog  # NOISY-UI
                    with _prog.waiting(_prog.ACT_RATE, wait):
                        self._sleep(wait)
            try:
                self._spend()
            except ProviderUnavailable:
                if not live and not parts:
                    raise
                self.last_query["note"] = "daily cap reached before the later queries"
                break
            try:
                r = tx(self.endpoint, {"q": q["query"], "kl": self.s["region"]}, headers, timeout_s)
            except requests_exceptions() as e:
                if not live and not parts:
                    raise
                self.last_query.setdefault("errors", []).append(f"{q['group']}: {type(e).__name__}: {e}")
                break
            live += 1
            parts.append((q, r))
            if r.status_code != 200 or self._is_challenge(r.text or ""):
                break  # a challenge or error stops further queries; it is never retried or evaded
        self.last_call_cached = bool(parts) and not live
        return _DdgParts(parts)

    @staticmethod
    def _is_challenge(body: str) -> bool:
        low = body.lower()
        return any(m in low for m in DDG_CHALLENGE_MARKERS)

    def _source_class(self, domain: str, news: bool) -> str:
        """item #71 section 13.2: config label by domain (longest match wins), else news / web_portal."""
        best = None
        for d, cls in self.source_class_by_domain.items():
            if _domain_matches(domain, d) and (best is None or len(d) > len(best[0])):
                best = (d, cls)
        return best[1] if best else ("news" if news else "web_portal")

    def _record(self, url: str, title: str, group: str) -> dict:
        domain = _domain_of(urlsplit(url).hostname or "")
        news = any(_domain_matches(domain, d) for d in self.news_domains)
        return {
            "url": url, "title": title,
            "snippet": "",  # DuckDuckGo snippets are never kept
            "doi": None, "published": None, "publisher": domain, "domain": domain,
            "source_type": "news" if news else "web_portal",
            "source_class": self._source_class(domain, news),
            "evidence_class": "grey" if news else None,
            "licence": None,  # unknown: the licence gate holds these for human review
            "stored_text_kind": "link_metadata",
            "discovery": DDG_NAME, "discovery_group": group,
            "provider_endpoint": self.endpoint,
        }

    def _parse_one(self, group: str, r) -> list[dict]:
        if isinstance(r, _CachedPart):
            return [self._record(i["url"], i["title"], group) for i in r.items]
        if r.status_code == 202:
            raise ProviderResponseError(f"{DDG_CHALLENGE_NOTE} (HTTP 202)")
        if r.status_code != 200:
            raise ProviderResponseError(f"duckduckgo HTTP {r.status_code}; not retried (low-volume discovery)")
        body = r.text or ""
        if len(body.encode("utf-8", "ignore")) > self.max_bytes:
            raise ProviderResponseError(f"duckduckgo response over {self.max_bytes} bytes; refused before parsing")
        if self._is_challenge(body):
            raise ProviderResponseError(DDG_CHALLENGE_NOTE)
        p = _LiteLinks()
        p.feed(body)
        if not p.links:
            if "no results" in " ".join(body.lower().split()):
                return []  # a genuine empty result page
            raise ProviderResponseError(f"{DDG_CHALLENGE_NOTE} (200 page without a result table)")
        out = []
        for href, title in p.links:
            url = unwrap_ddg_link(href)
            if not url or not title:
                continue
            out.append(self._record(url, title, group))
        return out

    def parse(self, resp, n: int) -> list[dict]:
        per, errors = [], []
        for q, r in resp.parts:
            try:
                items = self._parse_one(q["group"], r)
                if not isinstance(r, _CachedPart):
                    self._cache_put(q, items)  # only url + title are written (R11-c)
                per.append(items)
            except ProviderResponseError as e:
                errors.append(str(e))
        if errors and not any(per):
            raise ProviderResponseError("; ".join(errors))
        if errors and self.last_query is not None:
            self.last_query.setdefault("errors", []).extend(errors)
        out, seen = [], set()
        for rank in range(max((len(x) for x in per), default=0)):  # round-robin across the groups
            for items in per:
                if rank < len(items) and items[rank]["url"] not in seen:
                    seen.add(items[rank]["url"])
                    out.append(items[rank])
        return out[:n]

    def cost_for(self, resp):
        return 0.0, True, COST_BASIS_DDG

    def failure_basis(self) -> str:
        return COST_BASIS_DDG


def requests_exceptions():
    import requests
    return requests.RequestException


# ----------------------------------------------------------------------------------------- seed file
def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _Resp:
    def __init__(self, payload):
        self.status_code, self.payload, self.headers, self.text = 200, payload, {}, ""


def validate_seed(doc, query: str) -> list[dict]:
    """Return the validated items or raise SearchConfigError listing EVERY problem."""
    problems = []
    if not isinstance(doc, dict) or doc.get("schema") != SEED_SCHEMA:
        raise SearchConfigError(f"seed file: schema must be {SEED_SCHEMA!r}")
    strat = doc.get("search_strategy")
    if not isinstance(strat, dict) or not all(strat.get(k) for k in ("queries", "tools", "found_at", "inclusion_rule", "agent", "model")):
        problems.append("search_strategy needs non-empty queries, tools, found_at, inclusion_rule, agent, model")
    elif not doc.get("query_agnostic") and query not in strat["queries"]:
        problems.append("the run query is not among search_strategy.queries and the seed is not declared "
                        "query_agnostic: a seed for a different query is a different run input")
    items = doc.get("items")
    if not isinstance(items, list) or not items:
        problems.append("items: missing or empty")
        items = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            problems.append(f"items[{i}]: not an object")
            continue
        for k in SEED_ITEM_REQUIRED:
            if not it.get(k):
                problems.append(f"items[{i}].{k}: missing or empty")
        lic = it.get("licence")
        if lic is not None and not (isinstance(lic, dict) and set(lic) <= {"metadata", "content"}):
            problems.append(f"items[{i}].licence: must be an object with optional metadata/content records")
    if problems:
        raise SearchConfigError("seed file invalid: " + "; ".join(problems[:20]))
    return items


class SeedFileProvider:
    """DISCLOSED FALLBACK: reads an agent-prepared candidate list, verifies its sha256 first."""
    required_env: tuple[str, ...] = ()
    endpoint = "file"

    def __init__(self, path: Path, expected_sha256: str):
        self.path, self.expected = Path(path), expected_sha256
        self.name = f"seed:{expected_sha256[:8]}"
        self.gate = RateGate(0.0)

    def call(self, query: str, n: int, timeout_s: float):
        try:
            data = self.path.read_bytes()
        except OSError as e:
            raise SearchConfigError(f"seed file {self.path} unreadable: {type(e).__name__}: {e}") from None
        got = hashlib.sha256(data).hexdigest()
        if got != self.expected:
            raise SearchConfigError(f"seed file sha256 {got[:16]}... != frozen {self.expected[:16]}...: a changed "
                                    "seed file is a different run input (re-freeze it in the run config)")
        try:
            doc = json.loads(data)
        except ValueError as e:
            raise SearchConfigError(f"seed file is not valid JSON: {e}") from None
        validate_seed(doc, query)
        return _Resp({"doc": doc, "sha256": got})

    def parse(self, resp, n: int) -> list[dict]:
        out = []
        for it in resp.payload["doc"]["items"]:
            lic = it.get("licence") or {}
            out.append({"url": it["url"], "title": it["title"], "snippet": it["snippet"], "doi": it.get("doi"),
                        "published": it.get("published"),
                        "licence": licence_record(lic.get("metadata"), lic.get("content")),
                        "stored_text_kind": "full_content", "provider_endpoint": "file:" + self.expected,
                        "reliability_tier": it["reliability_tier"], "found_by": it["found_by"]})
        return out

    def cost_for(self, resp):
        return 0.0, True, COST_BASIS_SEED

    def failure_basis(self) -> str:
        return COST_BASIS_SEED


# ----------------------------------------------------------------------------------------- registry
PROVIDER_FACTORIES: dict[str, Callable] = {
    "arxiv": lambda settings, **kw: ArxivProvider(settings["arxiv"], query_terms=settings.get("query_terms"), **kw),
    "openalex": lambda settings, **kw: OpenAlexProvider(settings["openalex"], query_terms=settings.get("query_terms"), **kw),
    DOAJ_NAME: lambda settings, **kw: DOAJProvider(settings[DOAJ_NAME], query_terms=settings.get("query_terms"), **kw),
    DDG_NAME: lambda settings, **kw: DuckDuckGoLiteProvider(settings[DDG_NAME], query_terms=settings.get("query_terms"),
                                                            cache_dir=_ddg_cache_dir(settings[DDG_NAME]), **kw),
}
# Why: names the extension seam only; there is deliberately no factory for a spending provider (R1 D5).
KNOWN_PROVIDER_NAMES = tuple(PROVIDER_FACTORIES)


def build_chain(settings: Mapping, *, repo_root: Path, transport: Callable | None = None) -> list:
    """Providers in route order from config `search`; the seed fallback (when configured) is LAST."""
    chain = []
    for name in settings["providers"]:
        factory = PROVIDER_FACTORIES.get(name)
        if factory is None:
            raise SearchConfigError(f"unknown search provider {name!r}; known: {list(KNOWN_PROVIDER_NAMES)}")
        if isinstance(settings.get(name), Mapping) and settings[name].get("enabled") is False:
            continue  # switched off in config (provider block `enabled: false`)
        chain.append(factory(settings, **({"transport": transport} if transport else {})))
    fb = settings.get("fallback_seed") or {}
    if fb.get("path") or fb.get("sha256"):
        if not (fb.get("path") and fb.get("sha256")):
            raise SearchConfigError("search.fallback_seed needs BOTH path and sha256 (frozen by hash)")
        chain.append(SeedFileProvider(Path(repo_root) / fb["path"], fb["sha256"]))
    return chain
