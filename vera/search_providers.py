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

import contextvars
import hashlib
import json
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
USER_AGENT = "VERA-research-pipeline/1.0 (offline-reviewed; no contact address configured)"
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
    from vera import rate_limit  # RATE-1022 A1
    return rate_limit.limited("arxiv", requests.get)(url, params=params, headers=headers, timeout=timeout)


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
            if qt.extra:
                parts.append("(" + " OR ".join(f"all:{t}" for t in qt.extra) + ")")
            body = "(" + " AND ".join(parts) + ")" if qt.core else parts[0]
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

    def __init__(self, settings: Mapping, *, transport: Callable | None = None, query_terms: Mapping | None = None):
        self.s = settings
        self.endpoint = settings["endpoint"]
        self.max_bytes = int(settings["max_response_bytes"])
        self.gate = RateGate(float(settings["min_interval_s"]))
        self._transport = transport
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
        return tx(self.endpoint, params, {"User-Agent": USER_AGENT}, timeout_s)

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
    from vera import rate_limit  # RATE-1022 A2
    return rate_limit.limited("openalex", requests.get)(url, params=params, headers=headers, timeout=timeout)


def openalex_api_key(settings: Mapping | None = None, env: Mapping | None = None) -> str | None:
    """YIELD-1021: OPTIONAL OpenAlex API key from the env var named by settings api_key_env (default OPENALEX_API_KEY).
    Absent or blank = the keyless free tier, unchanged. The value is never logged or stored."""
    import os
    name = str((settings or {}).get("api_key_env") or "OPENALEX_API_KEY")
    v = str((os.environ if env is None else env).get(name) or "").strip()
    return v if re.fullmatch(r"[A-Za-z0-9_\-]{8,200}", v) else None


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
OPENALEX_OA_SELECT_EXTRA = "open_access,best_oa_location,locations,ids"
OPENALEX_PMC_SOURCE_ID = "S2764455111"  # PubMed Central (where Europe PMC full text comes from)


def _profile_extra_and(extra, profile: Mapping) -> str:
    """Acquisition: a question with no core phrase gives an all-OR query that ranks any one common token (the
    ambiguous 'mma' matched alpha-gal papers). Instead AND the distinctive terms: drop the profile's generic terms
    (unless that leaves none) and keep at most `and_extra_max` in question order. Deterministic, no model call."""
    generic = {str(g).lower() for g in profile.get("generic_terms") or ()}
    terms = [t for t in extra if str(t).lower() not in generic] or list(extra)
    return " AND ".join(terms[: max(1, int(profile.get("and_extra_max") or 4))])


def build_openalex_search(question: str, terms_cfg: Mapping | None, *, relaxed: bool = False,
                          profile: Mapping | None = None) -> tuple[str, str]:
    """-> (OpenAlex `search` string, mode): quoted core phrases AND-ed, AND (OR of the other terms)."""
    if terms_cfg is not None:
        qt = build_query_terms(question, terms_cfg)
        if not qt.empty:
            if profile and not qt.core and qt.extra and profile.get("and_extra_max"):
                return _profile_extra_and(qt.extra, profile), qt.mode + "+and"
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
        flt = (f"from_publication_date:{self.s['from_publication_date']},"
               f"to_publication_date:{self.s['to_publication_date']}")
        if profile and profile.get("filter"):
            flt += "," + str(profile["filter"])
        select = OPENALEX_SELECT + ("," + OPENALEX_OA_SELECT_EXTRA if OPENALEX_OA_FIELDS.get() else "")
        params = {"search": search, "filter": flt, "per_page": n, "select": select}
        tx = self._transport or openalex_default_transport
        key = openalex_api_key(self.s)
        if not key:
            return tx(self.endpoint, params, {"User-Agent": USER_AGENT}, timeout_s)
        try:  # YIELD-1021: OPTIONAL key (env only); never logged: any error text is redacted before it propagates
            return tx(self.endpoint, {**params, "api_key": key}, {"User-Agent": USER_AGENT}, timeout_s)
        except Exception as e:  # noqa: BLE001
            msg = str(e).replace(key, "[redacted]")
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
            doc = json.loads(body)
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
    from vera import rate_limit  # RATE-1022 A3 (search and id lookup)
    return rate_limit.limited("europepmc", requests.get)(url, params=params, headers=headers, timeout=timeout,
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
        doc = json.loads(body)
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
_ARXIV_URL = re.compile(r"^https?://(?:www\.|export\.)?arxiv\.org/(?:abs|pdf)/([a-z\-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?/?$",
                        re.I)
_PMC_URL = re.compile(r"^https?://(?:www\.)?(?:ncbi\.nlm\.nih\.gov/pmc/articles|pmc\.ncbi\.nlm\.nih\.gov/articles|"
                      r"europepmc\.org/(?:article|abstract)/PMC)/(PMC\d{1,10})\b", re.I)
_PMID_URL = re.compile(r"^https?://(?:pubmed\.ncbi\.nlm\.nih\.gov/|(?:www\.)?ncbi\.nlm\.nih\.gov/pubmed/|"
                       r"europepmc\.org/(?:article|abstract)/MED/)(\d{1,9})\b", re.I)
_META_DOI = re.compile(r"<meta\s[^>]*?name=[\"'](?:citation_doi|dc\.identifier|prism\.doi)[\"'][^>]*?>", re.I)
_CONTENT = re.compile(r"content=[\"']\s*(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[^\"'\s]+)", re.I)


def ddg_identifier(url: str) -> dict | None:
    """{"kind": "doi"|"pmcid"|"pmid"|"arxiv", "value"} from the URL alone (no request), or None."""
    u = str(url or "").strip()
    m = _ARXIV_URL.match(u)
    if m:
        return {"kind": "arxiv", "value": m.group(1).lower()}
    m = _PMC_URL.match(u)
    if m:
        return {"kind": "pmcid", "value": m.group(1).upper()}
    m = _PMID_URL.match(u)
    if m:
        return {"kind": "pmid", "value": m.group(1)}
    m = _DOI_IN_URL.search(u)
    if m:
        doi = re.sub(r"(\.pdf|/full|/abstract|/epdf|/pdf|[.,;)])+$", "", m.group(1), flags=re.I).lower()
        if re.fullmatch(r"10\.\d{4,9}/\S+", doi):
            return {"kind": "doi", "value": doi}
    return None


def landing_doi(url: str, timeout_s: float, max_bytes: int, *, get: Callable | None = None,
                url_check: Callable | None = None, deadline: float | None = None,
                clock: Callable | None = None) -> str | None:
    """ONE cheap GET of a journal landing page (public-address check on every hop, at most 2 redirects, size-capped,
    short timeout) -> its citation_doi meta tag, or None. Never raises."""
    if get is None:
        import requests
        get = requests.get
    if url_check is None:
        from vera.selective_fetch import is_public_url as url_check
    import time as _t
    clock = clock or _t.monotonic
    cur = url
    try:
        for _hop in range(3):
            left = timeout_s if deadline is None else min(timeout_s, deadline - clock())
            if left <= 0.2:
                return None
            ok = url_check(cur)
            if not (ok[0] if isinstance(ok, tuple) else ok):
                return None
            r = get(cur, headers={"User-Agent": USER_AGENT}, timeout=left, allow_redirects=False, stream=True)
            stop_at = clock() + left  # a trickling server cannot hold the thread past the hop budget or the deadline
            try:
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
                    from urllib.parse import urljoin
                    cur = urljoin(cur, r.headers["Location"])
                    continue
                if r.status_code != 200:
                    return None
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
                    return m.group(1).rstrip(".").lower()
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
    rows = ((json.loads(resp.text or "{}").get("resultList") or {}).get("result") or [])
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


def ddg_to_candidates(results: list, ecfg: Mapping, dcfg: Mapping, *, deadline: float, clock: Callable,
                      calls_left: int, transport: Callable | None = None, get: Callable | None = None,
                      url_check: Callable | None = None) -> tuple[list, dict]:
    """DDG results -> (acquirable candidates, counts). DOI in the URL and arXiv ids map directly; PMC/PubMed links map
    through ONE Europe PMC id lookup (only when calls_left >= 1); a portal landing page with no identifier gets ONE
    cheap GET for its citation_doi meta tag (at most dcfg.max_landing_fetches, inside the deadline). Everything else is
    held (counted, dropped from acquisition; it stays a reading-list link on /ask)."""
    counts = {"ddg_results": 0, "ddg_mapped": 0, "ddg_held": 0, "ddg_landing_get": 0}
    out, pending, landing = [], [], 0
    max_landing = int(dcfg.get("max_landing_fetches", 3))
    lt = float(dcfg.get("landing_timeout_s", 5))
    lb = int(dcfg.get("landing_max_bytes", 524288))
    for r in results or ():
        if not isinstance(r, Mapping) or not r.get("url"):
            continue
        counts["ddg_results"] += 1
        ident = ddg_identifier(r["url"])
        if ident is None and landing < max_landing and str(r.get("discovery_group") or "") in set(
                dcfg.get("landing_groups") or ("portals",)) and deadline - clock() > lt:
            landing += 1
            counts["ddg_landing_get"] += 1
            d = landing_doi(r["url"], lt, lb, get=get, url_check=url_check, deadline=deadline, clock=clock)
            ident = {"kind": "doi", "value": d} if d else None
        if ident is None:
            counts["ddg_held"] += 1
            continue
        if ident["kind"] in ("pmcid", "pmid"):
            pending.append((ident, r))
            continue
        doi = ident["value"] if ident["kind"] == "doi" else f"10.48550/arxiv.{ident['value']}"
        out.append(_ddg_cand(r, doi, None, ident["kind"]))
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
    counts["ddg_mapped"] = len(out)
    return out, counts


def _ddg_cand(r: Mapping, doi: str, pmcid, kind: str) -> dict:
    return {"url": f"https://doi.org/{doi}", "title": " ".join(str(r.get("title") or "").split()), "snippet": "",
            "doi": doi, "provider": "ddg", "acq_source": "ddg", "ddg_id_kind": kind, "discovered_url": r.get("url"),
            **({"pmcid": pmcid, "oa": {"pmcid": pmcid}} if pmcid else {})}


def acq_source(c: Mapping) -> str:
    if c.get("acq_source") in ("ddg", "wikipedia_portal", "discovered"):  # WIKIREF-1022 edit 2; P0-5 (flag on only)
        return c["acq_source"]
    return "epmc" if c.get("provider") == "europepmc" else "openalex"


def acq_rank_tier(c: Mapping) -> int:
    """Ranking tier (lower first): DuckDuckGo-mapped candidates first (owner 2026-10-21), arXiv-mapped ones last
    (Europe PMC refuses preprints, so the unchanged resolver cannot yield them), everything else in between."""
    if c.get("acq_source") == "ddg":
        return 2 if c.get("ddg_id_kind") == "arxiv" else 0
    if c.get("acq_source") == "wikipedia_portal":  # WIKIREF-1022 edit 2: arXiv refs last, others scholarly tier
        return 2 if c.get("ref_id_kind") == "arxiv" else 1
    return 1


def merge_sources(*lists) -> list:
    """Merge candidate lists in order, de-duplicated by DOI or PMCID; the first occurrence keeps its place and source,
    and absorbs a later duplicate's OA metadata (Europe PMC full text marks epmc_ft) and empty snippet/title."""
    out, index = [], {}
    for lst in lists:
        for c in lst or ():
            if not isinstance(c, dict):
                continue
            ids = _cand_ids(c)
            hit = next((index[k] for k in ids if k in index), None)
            if hit is None:
                out.append(c)
                for k in ids:
                    index[k] = c
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
            for k in ("snippet", "title"):
                if not hit.get(k) and c.get(k):
                    hit[k] = c[k]
            for k in ids:
                index.setdefault(k, hit)
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


def acquisition_search(base_search: Callable, cfg: Mapping, *, transport: Callable | None = None,
                       terms_fn: Callable | None = None, ddg_search: Callable | None = None,
                       get: Callable | None = None, url_check: Callable | None = None,
                       clock: Callable | None = None, wiki_get: Callable | None = None) -> Callable:
    """Wrap the acquisition job's search (vera.auto_acquire.make_search): DuckDuckGo Lite (config search.ddg_first),
    OpenAlex and Europe PMC run CONCURRENTLY within the same timeout, merged DDG first. Search calls are counted
    against bounds.max_search_calls_per_job: OpenAlex 2 (AND profile + OR retry; else 1), then DuckDuckGo 1, Europe
    PMC 1, the DDG id lookup 1, each only while the cap leaves a call. Any provider failing returns the others' results;
    all failing re-raises the OpenAlex error. Landing-page GETs are fetches (dcfg.max_landing_fetches), not searches."""
    import time as _time
    clock = clock or _time.monotonic
    s = cfg.get("search") or {}
    ecfg = s.get("europepmc") or {}
    dcfg = s.get("ddg_first") or {}
    cap = int((cfg.get("bounds") or {}).get("max_search_calls_per_job", 0) or 0)
    left = cap - (2 if s.get("openalex_profile") else 1)
    use_ddg = dcfg.get("enabled") is True and left >= 1
    left -= 1 if use_ddg else 0
    use_epmc = ecfg.get("enabled") is True and left >= 1
    left -= 1 if use_epmc else 0
    from vera import wiki_ref_portal  # WIKIREF-1022 edit 1: Wikipedia as a reference portal (never evidence)
    wcfg = _wiki_cfg()
    use_wiki = wiki_ref_portal.enabled(wcfg) and left >= 1
    left -= 1 if use_wiki else 0
    lookup_left = left
    if use_wiki and wiki_get is None:
        import requests
        from vera import rate_limit
        wiki_get = rate_limit.limited("wikipedia", requests.get)

    def search(question, n, timeout_s):
        import concurrent.futures as cf
        t_end = clock() + float(timeout_s)
        futs = {}
        terms, profile = None, None
        pool = cf.ThreadPoolExecutor(max_workers=3, thread_name_prefix="acq-search")
        try:
            if use_ddg:  # RATE-1022 C2: each future runs in a copy of this context (the inline rate-limit deadline)
                futs["ddg"] = pool.submit(contextvars.copy_context().run, ddg_search or _default_ddg_search(cfg),
                                          question, n, timeout_s)
            if use_epmc:
                try:
                    terms, profile = (terms_fn or (lambda: _default_terms_and_profile(cfg)))()
                except Exception:  # noqa: BLE001
                    terms, profile = None, None
                futs["epmc"] = pool.submit(contextvars.copy_context().run, europepmc_search, question,
                                           int(ecfg.get("page_size", n)), float(timeout_s),
                                           ecfg, terms_cfg=terms, profile=profile, transport=transport)
            if use_wiki:
                futs["wiki"] = pool.submit(contextvars.copy_context().run,
                                           wiki_ref_portal.make_portal_search(wcfg, http_get=wiki_get),
                                           question, n, timeout_s)
        finally:
            pool.shutdown(wait=False)
        counts = {}
        base, base_err = [], None
        if _QUOTA.blocked(clock()):  # a recent OpenAlex 429/quota error: unavailable, no call (no retry storm)
            counts["openalex_skipped_quota"] = 1
            base_err = ProviderUnavailable("openalex quota cooldown")
        else:
            tok = OPENALEX_OA_FIELDS.set(True)
            try:
                base = list(base_search(question, n, timeout_s) or [])
            except Exception as e:  # noqa: BLE001
                base_err = e
                if _is_rate_skip(e):  # RATE-1022 D1: the openalex adapter skipped (nothing sent); it owns the cooldown
                    counts["openalex_skipped_quota"] = 1
                elif _is_quota_error(e):
                    counts["openalex_429"] = 1
                    # INT-1022-B: kept as a FALLBACK (spec D1 option 1) for injected searches that bypass the adapter;
                    # on the live path the adapter cools down from Retry-After/headers as well.
                    _QUOTA.block(clock(), float(s.get("openalex_quota_cooldown_s", 900)))
            finally:
                OPENALEX_OA_FIELDS.reset(tok)
        got = {}
        epmc_ok = False
        for k, f in futs.items():
            try:
                got[k] = list(f.result(timeout=max(0.1, t_end - clock())) or [])
                epmc_ok = epmc_ok or k == "epmc"
            except Exception as e:  # noqa: BLE001  (timeout, transport or parse error: the other sources stand)
                got[k] = []
                if _is_quota_error(e):
                    counts[f"{k}_429"] = 1
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
                                            calls_left=lookup_left, transport=transport, get=get, url_check=url_check)
            counts.update(dcounts)
        if base_err is not None and not ddg and not got.get("epmc") and not got.get("wiki"):
            raise base_err
        merged = merge_sources(ddg, base, got.get("epmc"), got.get("wiki"))  # DDG first; portal refs last
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
    from vera import rate_limit  # RATE-1022 A4
    return rate_limit.limited("doaj", requests.get)(url, params=params, headers=headers, timeout=timeout)


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
            doc = json.loads(resp.text or "")
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
            doc = json.loads(body)
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
                 wall_clock: Callable[[], float] | None = None):
        import time
        self.s = settings
        self.endpoint = settings["endpoint"]
        self.max_bytes = int(settings["max_response_bytes"])
        self.gate = RateGate(float(settings["min_interval_s"]))
        self.cache_dir = Path(cache_dir) if cache_dir else None  # None = no cache (tests inject a tmp dir)
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

    def _spend(self) -> None:
        day = self._today()
        if day != self._day:
            self._day, self._used = day, 0
        if self._used >= self._cap:
            raise ProviderUnavailable(f"duckduckgo_lite daily query cap of {self._cap} spent for {day} "
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
            return None
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

    def call(self, query: str, n: int, timeout_s: float):
        queries = build_ddg_queries(query, self.query_terms, self.s)
        self.last_query = {"query": [q["query"] for q in queries], "mode": "site_groups"}
        tx = self._transport or ddg_default_transport
        headers = {"User-Agent": self.s["user_agent"]}  # never altered, even after a challenge
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
