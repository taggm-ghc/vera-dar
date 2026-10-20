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
    return requests.get(url, params=params, headers=headers, timeout=timeout)


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
    return requests.get(url, params=params, headers=headers, timeout=timeout)


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


def build_openalex_search(question: str, terms_cfg: Mapping | None, *, relaxed: bool = False) -> tuple[str, str]:
    """-> (OpenAlex `search` string, mode): quoted core phrases AND-ed, AND (OR of the other terms)."""
    if terms_cfg is not None:
        qt = build_query_terms(question, terms_cfg)
        if not qt.empty:
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
        search, mode = build_openalex_search(query, self.query_terms, relaxed=self.relaxed)
        self.last_query = {"query": search, "mode": mode, **({"relaxed": True} if self.relaxed else {})}
        self._spend()  # every HTTP attempt counts, retries included
        params = {"search": search,
                  "filter": f"from_publication_date:{self.s['from_publication_date']},"
                            f"to_publication_date:{self.s['to_publication_date']}",
                  "per_page": n, "select": OPENALEX_SELECT}
        tx = self._transport or openalex_default_transport
        return tx(self.endpoint, params, {"User-Agent": USER_AGENT}, timeout_s)

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
            })
            if len(out) >= n:
                break
        return out

    def cost_for(self, resp):
        return 0.0, True, COST_BASIS_KEYLESS_FREE

    def failure_basis(self) -> str:
        return COST_BASIS_KEYLESS_FREE



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
    return requests.get(url, params=params, headers=headers, timeout=timeout)


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
    return requests.post(url, data=data, headers=headers, timeout=timeout)


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
