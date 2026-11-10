"""Item #79: grounded /ask. Live scholarly retrieval once per accepted question, the measured grounded-v1
prompt, and a citation check on the answer.

Settings come from the `grounding` block of config/ask-provider-chain.json. Retrieval uses VERA-DAR's own
keyless search providers; results with an abstract are used directly and link-only results are followed to
their original sources (vera/source_resolver.py), so every numbered source gives the model text to ground
on. Nothing is stored: no database, no corpus file.

PORTAL-WIRE-B-1110 (owner 2026-11-10, OD-1110-PORTALS-ALL; owner: "That sample illustrates a result from a portal through
which we can links to original sources"): search providers and indexes (OpenAlex, DuckDuckGo, arXiv, Semantic Scholar,
INSPIRE, Wikipedia, Europe PMC search) are PORTALS. Their abstracts and snippets are NEVER evidence: every search
record is listed as a discovery entry (title, link, identifier; its snippet stays internal, for relevance judgement and
ranking only) with evidence_eligible False. Only a stored full-document chunk (dar_span) on a non-portal host is
evidence_eligible; grounding_sources() is the one filter every grounding prompt goes through.
"""
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Callable

from vera.trace_eval.capture import grounded_v1_messages

logger = logging.getLogger("vera")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "ask-provider-chain.json"
_CITE_GROUP = re.compile(r"\[(\s*\d+\s*(?:[,;–-]\s*\d+\s*)*)\]")


def load_grounding(path: Path = CONFIG_PATH) -> dict | None:
    """The grounding block, or None when it is missing, unreadable or disabled (then /ask is ungrounded)."""
    try:
        g = json.loads(Path(path).read_text()).get("grounding") or {}
    except (OSError, ValueError):
        return None
    return g if g.get("enabled") else None


def load_corpus_cfg(path: Path = CONFIG_PATH) -> dict | None:
    """The corpus block (phase C), or None when missing, unreadable or disabled."""
    try:
        c = json.loads(Path(path).read_text()).get("corpus") or {}
    except (OSError, ValueError):
        return None
    return c if c.get("enabled") else None


def clean_text(text: str, max_chars: int) -> str:
    """Drop invisible/format Unicode (prompt-injection hiding place) and cap length."""
    s = "".join(ch for ch in str(text or "") if unicodedata.category(ch) != "Cf")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_chars]


def _provider_chain(names: list[str]):
    from vera.search_and_fetch import _search_settings, cached_chain

    chain, _, _ = cached_chain(_search_settings())  # cached per process: provider rate gates persist
    return [p for p in chain if p.name in set(names)]


def _branch_providers(providers):
    """DDG-REDUCE-1110 C1: a domain-routing branch searches with the providers other than DuckDuckGo (the base query keeps
    it); config/ddg_reduce.json domain_branches_scholarly_only false restores the full chain."""
    from vera import ddg_reduce
    from vera.search_providers import DDG_NAME
    if not ddg_reduce.on("domain_branches_scholarly_only"):
        return providers
    return [p for p in providers if getattr(p, "name", None) != DDG_NAME]


def _domain_search_fn(fn: Callable) -> Callable:
    """DDG-REDUCE-1110 WP0: DuckDuckGo calls made by a domain-routing branch are counted as the "domain" site."""
    from vera import ddg_meter

    def run(q):
        with ddg_meter.site("domain"):
            return fn(q)
    return run


def retrieve(question: str, cfg: dict, search: Callable | None = None, resolver: Callable | None = None) -> list[dict]:
    """Numbered sources with abstracts (the prompt's sources)."""
    return retrieve_all(question, cfg, search, resolver)[0]


def _trace_counts(trace: dict) -> dict:
    """Counts and codes only (sidebar rule: never an individual source, title, URL or text)."""
    keep = ("routing", "error_class", "fallback_contribution", "fallback_only", "dedup_count", "reserved")
    out = {k: trace[k] for k in keep if k in trace and isinstance(trace[k], (str, int))}
    for k in ("selected", "searched", "failed_branches"):
        if isinstance(trace.get(k), list):
            out[f"{k}_n"] = len(trace[k])
    if isinstance(trace.get("branch_counts"), dict):
        out["branch_counts"] = {str(k): int(v) for k, v in trace["branch_counts"].items() if isinstance(v, int)}
    return out


def _acquired_keys() -> frozenset:
    """ARCH-FIX-1110 A: keys of the acquired full documents (the set R-ACQ-KNOWN is decided from); empty when none or
    on any error. Never waits on the DB (get_acquired_corpus probes in the background)."""
    try:
        from vera import fulldoc_corpus
        dar = getattr(fulldoc_corpus.get_acquired_corpus(), "_dar", None)
        return frozenset(dar.doc_keys()) if dar is not None and hasattr(dar, "doc_keys") else frozenset()
    except Exception:  # noqa: BLE001
        return frozenset()


def _mark_in_corpus(results: list, dar_corpus, extra_keys=None) -> list:
    """Copy of `results` with in_corpus=True on every result whose DOI or URL (normalised, either direction) is that of
    any stored document (DarCorpus.doc_keys). The input list is returned unchanged when nothing matches."""
    from vera.dar_corpus import doc_key_set

    keys_fn = getattr(dar_corpus, "doc_keys", None)  # duck-typed corpora without it: no corpus-wide check
    held = frozenset(keys_fn() if callable(keys_fn) else ()) | frozenset(_acquired_keys() if extra_keys is None else extra_keys)
    if not held:
        return results
    out, hit = [], False
    for r in results:
        if isinstance(r, dict) and doc_key_set(r.get("doi"), r.get("url")) & held:
            r, hit = {**r, "in_corpus": True}, True
        out.append(r)
    return out if hit else results


def _merge_stored(results: list, question: str, dar_corpus, cfg: dict, info: dict) -> list:
    """DAR stored-corpus branch (flag-gated; DEC-FDA1: stored FULL documents only, the corpus refuses abstracts).
    The ordering policy lives in DarCorpus.rank_merge: the first head external results are protected; stored chunks
    below corpus.stored_min_relevance are not merged next to external evidence; the remaining external results and the
    qualifying stored chunks are ordered by ONE common relevance (query-term coverage of title + abstract/chunk), so a
    stored chunk can outrank a weaker external result, a stronger external result is never displaced by a weaker or
    irrelevant stored chunk, and insertion position plays no role. Truncation to num_results happens later, in
    retrieve_all, after this comparable ordering. With the flag off, no corpus, or any error the input list is
    returned by identity."""
    if dar_corpus is None:
        return results
    try:
        ccfg = getattr(dar_corpus, "cfg", None) or {}
        if ccfg.get("enabled") is not True:
            return results
        results = _mark_in_corpus(results, dar_corpus)  # WED-ACQ A1: corpus-wide, not only the retrieved chunks
        if (ccfg.get("corpus") or {}).get("stored_branch_enabled") is not True:
            return results
        stored, tr = dar_corpus.as_sources(question)
        info["stored"] = {k: v for k, v in (tr or {}).items() if isinstance(v, (int, str))}
        if not stored:
            floor = getattr(dar_corpus, "floor_external", None)  # WP-H reading-list floor (flag off = identity)
            return floor(question, results) if callable(floor) else results
        return dar_corpus.rank_merge(question, list(results), list(stored), int(cfg["num_results"]))
    except Exception as exc:  # noqa: BLE001  (the stored branch never fails the request)
        logger.warning("DAR stored branch skipped (%s)", type(exc).__name__)
        info["stored"] = {"error_class": type(exc).__name__}
        return results


def _content_matcher(question: str):
    """NINE-DOCS-RC4-1108: a predicate text -> True when `text` shares >= 1 CONTENT stem with the question: the
    admission relevance tokenizer (config/fulldoc_answer.json admission.relevance: stopwords, question_generic_terms,
    stem_len), i.e. after stopword and generic-word removal. A result that matches the question only on function or
    generic words ("Come, Go" for "from where did vera come?") is not relevant and is never listed. A non-numeric
    rule: no threshold. None (no filtering) when the question has no content stem of >= min_token_len letters (for
    example "What is AI?": nothing to judge on) or the settings cannot be read."""
    try:
        from vera import fulldoc_admission as fa
        from vera.fulldoc_corpus import load_config

        rel = fa._routing(load_config())
        icfg = rel.get("identification") or {}
        n = int((rel.get("corpus") or {})["relevance_stem_len"])
        q = fa._q_stems(question, icfg, n)
        if not q or n <= 0:
            return None
        return lambda text: bool(q & fa._stems(str(text or ""), icfg, n))
    except Exception:  # noqa: BLE001  (unreadable settings: the list is left as it was)
        return None


_NO_EVIDENCE_ROLES = ("PORTAL", "EXCLUDED")


def evidence_eligible(r) -> bool:
    """PORTAL-WIRE-B-1110: may this record's text ground an answer? Only a stored full-document chunk (dar_span) whose
    record role (DOI prefix, then URL host; vera/host_role.py) is not PORTAL or EXCLUDED, and which no producer marked
    evidence_eligible False. Every search or index record (abstract or snippet) is False. Never raises (False)."""
    try:
        if not isinstance(r, dict) or r.get("evidence_eligible") is False or not isinstance(r.get("dar_span"), dict):
            return False
        from vera import host_role
        return host_role.role_for_record({"doi": r.get("doi"), "url": r.get("url")}).role not in _NO_EVIDENCE_ROLES
    except Exception:  # noqa: BLE001
        return False


def grounding_sources(sources) -> list[dict]:
    """The sources a grounding prompt may use: evidence_eligible ones only, renumbered 1..n (copies). A source already
    filtered here keeps its eligibility flag, so the call is idempotent."""
    out = []
    for s in sources or ():
        if isinstance(s, dict) and s.get("evidence_eligible") is True:
            out.append({**s, "n": len(out) + 1})
    return out


def retrieve_all(question: str, cfg: dict, search: Callable | None = None,
                 resolver: Callable | None = None, trace_out: dict | None = None,
                 dar_corpus=None) -> tuple[list[dict], list[dict]]:
    """(sources, unused): numbered listed sources [{n, title, url, published, snippet, source_type, evidence_eligible}]
    and traced originals (source_resolver, metadata only: never an abstract) and arXiv-only works, listed and handed to
    acquisition only. ([], []) on failure. PORTAL-WIRE-B-1110: a search/index record's snippet is internal (relevance,
    ranking) and evidence_eligible False.
    Optional, additive: trace_out (a dict filled with COUNTS-ONLY routing/stored-branch aggregates) and dar_corpus
    (a vera.dar_corpus.DarCorpus; None = no stored branch). The return contract is unchanged."""
    info: dict = {}
    try:
        if search is None:
            from vera.search_and_fetch import search_question

            providers = _provider_chain(cfg["providers"])
            from vera import ddg_reduce
            if ddg_reduce.NO_DDG.get():  # DDG-REDUCE-1110 C4: this call site spends no DuckDuckGo
                from vera.search_providers import DDG_NAME
                providers = [p for p in providers if getattr(p, "name", None) != DDG_NAME]
            # per-provider timeout below the overall deadline, so one slow provider cannot starve the others
            results = search_question(question, num_results=int(cfg["num_results"]), providers=providers,
                                      min_candidates=1, mode="fan_out", deadline_s=float(cfg["deadline_s"]),
                                      timeout_s=float(cfg.get("provider_timeout_s", 5)),
                                      max_retries=int(cfg.get("provider_max_retries", 1)))
            try:
                from vera import domain_routing as _dr
                if _dr.flag_enabled():
                    results, rtrace = _dr.route_search(question, results, _domain_search_fn(_dr.make_search_fn(search_question, _branch_providers(providers), cfg)), (dar_corpus.registry if dar_corpus is not None else _dr.get_registry()), _dr.get_cfg(), num_results=int(cfg["num_results"]))
                    info["routing"] = _trace_counts(rtrace)
                    logger.info("DAR routing: %s", info["routing"])
            except Exception as exc:  # noqa: BLE001  (routing never fails the request; baseline results are kept)
                logger.warning("DAR routing skipped (%s)", type(exc).__name__)
        else:
            results = search(question)
    except Exception as exc:  # noqa: BLE001  (search failure -> no-sources message, logged by class only)
        logger.warning("/ask grounding search failed (%s)", type(exc).__name__)
        return [], []
    results = _merge_stored(results, question, dar_corpus, cfg, info)
    if info.get("stored"):
        logger.info("DAR stored branch: %s", info["stored"])
    if trace_out is not None:
        trace_out.update(info)
    direct = [r for r in results if clean_text(r.get("snippet"), 1)]
    linked = [r for r in results if not clean_text(r.get("snippet"), 1)]
    # shortcut judged on the unrestricted (base) items only: domain-branch or stored items never stand in for them
    base_direct = [r for r in direct if "unrestricted" in r.get("_domain_branches", ["unrestricted"])]
    if len(base_direct) >= int(cfg["num_results"]):
        linked = []  # enough sources with abstracts already; skip the slower link following
    rcfg = cfg.get("resolve") or {}
    if linked and rcfg.get("enabled"):
        direct += _resolve_links(linked, rcfg, resolver)
    out, seen, unused = [], set(), []
    on_topic = _content_matcher(question)  # NINE-DOCS-RC4-1108: function-word-only matches are never listed
    n_off = 0
    for r in direct:
        snippet = clean_text(r.get("snippet"), int(cfg["snippet_max_chars"]))
        url = str(r.get("url") or "")
        key = str(r.get("doi") or "").lower() or url.lower().rstrip("/")
        if not url.lower().startswith(("https://", "http://")) or key in seen:
            continue
        # judged after link resolution, on the title plus the abstract (or the title alone for a traced original)
        if on_topic is not None and not on_topic(f"{r.get('title') or ''} {snippet or ''}"):
            n_off += 1
            continue
        from vera import arxiv_portal
        arxiv_only = not isinstance(r.get("dar_span"), dict) and arxiv_portal.is_arxiv_only_work(r)
        # OD-1110-ARXIV-ABSTRACT: an arXiv-only work's abstract is never grounding (arXiv is a portal to sources)
        if not snippet or arxiv_only:
            if r.get("discovered_via") or arxiv_only:  # handed to acquisition; a traced original is also listed (below)
                seen.add(key)
                unused.append({"title": clean_text(r.get("title"), 300) or url, "url": url,
                               "published": (str(r.get("published"))[:10] if r.get("published") else None),
                               "discovered_via": str(r.get("discovered_via") or "arXiv portal"),
                               **{k: r[k] for k in ("doi", "pmcid", "openalex_id", "doaj_id", "provider") if r.get(k)}})  # identifiers for acquisition only; traced_sources shows title/url/published
            # PORTAL-WIRE-B-V5 (review F1): a resolved original (metadata only, empty snippet) is ALSO listed in the reading
            # list, evidence_eligible False (never answer evidence); an arXiv-only work stays unlisted (portal)
            if not r.get("discovered_via") or arxiv_only:
                continue
        seen.add(key)
        src = {"n": len(out) + 1, "title": clean_text(r.get("title"), 300) or url, "url": url,
               "published": (str(r.get("published"))[:10] if r.get("published") else None),
               "snippet": snippet, "source_type": r.get("source_type"),
               "evidence_eligible": evidence_eligible(r)}  # PORTAL-WIRE-B-1110: portal text is never evidence
        if r.get("discovered_via"):
            src["discovered_via"] = str(r["discovered_via"])
        for k in ("licence", "stored_text_kind", "doi", "pmcid", "openalex_id", "doaj_id", "provider", "admission_passed", "canonical_source_id", "source_id", "disposition", "in_corpus", "authors", "publisher", "container_title", "journal"):  # internal, for the corpus gauntlet (pmcid is published only via public_sources/_pmcid_of, PMCID-PUBLIC-1023)
            if r.get(k):
                src[k] = r[k]
        src.update(provenance(r))
        if isinstance(r.get("dar_span"), dict):  # stored-corpus provenance: the exact character range of the quoted chunk
            src["span"] = f"chars={int(r['dar_span']['start'])}-{int(r['dar_span']['end'])}"
            if r["dar_span"].get("span_id"):
                src["span_id"] = r["dar_span"]["span_id"]
        out.append(src)
        if len(out) >= int(cfg["num_results"]):
            break
    if n_off:
        logger.info("grounding: %d result(s) shared no content term with the question and were not listed", n_off)
    return out, unused[: int(cfg["num_results"])]


def _resolve_links(linked: list[dict], rcfg: dict, resolver=None) -> list[dict]:
    """Follow up to max_results_to_resolve link-only results within one deadline; log what was skipped."""
    import time

    from vera.source_resolver import resolve

    resolver = resolver or resolve
    deadline_at = time.monotonic() + float(rcfg["deadline_s"])
    found, tried = [], 0
    for r in linked[: int(rcfg["max_results_to_resolve"])]:
        if time.monotonic() >= deadline_at:
            break
        tried += 1
        try:
            found += resolver(r, rcfg, deadline_at)
        except Exception as exc:  # noqa: BLE001  (one bad page never fails the request)
            logger.warning("/ask link resolution failed (%s)", type(exc).__name__)
    skipped = len(linked) - tried
    if skipped:
        logger.info("/ask link resolution: %d followed, %d skipped (limit or deadline), %d originals found",
                    tried, skipped, len(found))
    return found


_ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", re.I)


def _public_pmcid(r: dict) -> str | None:
    """PMCID-PUBLIC-1023: the record's PMCID in validated form (search_providers._pmcid_of), else None (omitted)."""
    from vera.search_providers import _pmcid_of
    return _pmcid_of(r.get("pmcid")) if r.get("pmcid") else None


def provenance(r: dict) -> dict:
    """Per-citation provenance (R1 criterion): provider, identifier, retrieval time, licence decision."""
    from datetime import datetime, timezone

    from vera.licence_gate import licence_decision

    url = str(r.get("url") or "")
    m = _ARXIV_ID.search(url)
    pmc = _public_pmcid(r)  # PMCID-PUBLIC-1023: a public identifier (validated format), after the DOI
    ident = (f"doi:{str(r['doi']).lower()}" if r.get("doi") else f"pmcid:{pmc}" if pmc else f"arXiv:{m.group(1)}" if m
             else url)
    provider = r.get("provider") or ("openalex" if r.get("discovered_via") else None)
    if r.get("abstract_source"):
        provider = f"{provider or 'openalex'}+{r['abstract_source']}"
    decision, _why = licence_decision({"url": url, "licence": r.get("licence"),
                                       "stored_text_kind": r.get("stored_text_kind")})
    stored = isinstance(r.get("dar_span"), dict) or bool(r.get("span"))
    if stored:  # stored source: the stored retrieval time only, never now() (absent stays absent)
        retrieved = r.get("retrieved_at")
    else:
        retrieved = r.get("retrieved_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {"provider": provider or "unknown", "identifier": ident,
            "retrieved_at": retrieved,
            "licence_decision": decision}


def messages_for(question: str, sources: list[dict]) -> list[dict]:
    """The grounded-v1 prompt over grounding_sources(sources) only (PORTAL-WIRE-B-1110: a portal abstract or snippet
    never reaches the model as a source)."""
    return grounded_v1_messages(question, grounding_sources(sources))


def _expand(group: str) -> list[int]:
    nums = []
    for part in re.split(r"[,;]", group):
        bounds = [int(x) for x in re.findall(r"\d+", part)]
        if len(bounds) == 2 and bounds[0] <= bounds[1] and bounds[1] - bounds[0] < 50:
            nums.extend(range(bounds[0], bounds[1] + 1))
        else:
            nums.extend(bounds)
    return nums


def fix_citations(answer: str, n_sources: int, note: str) -> tuple[str, int]:
    """Remove citation numbers outside 1..n_sources; return (text, number removed). A note is appended if any."""
    removed = 0

    def repl(m):
        nonlocal removed
        nums = _expand(m.group(1))
        keep = [n for n in nums if 1 <= n <= n_sources]
        removed += len(nums) - len(keep)
        return f"[{', '.join(str(n) for n in dict.fromkeys(keep))}]" if keep else ""

    text = _CITE_GROUP.sub(repl, answer or "")
    if removed:
        text = text.rstrip() + "\n\n" + note
    return text, removed


PUBLIC_FIELDS = ("n", "title", "url", "published", "source_type", "provider", "identifier", "retrieved_at",
                 "licence_decision", "reference", "in_text", "characterisation")  # NONASK-1103 C2: reference/in_text = formatted APA 7 strings only


# present only for stored-corpus (DAR) sources: the exact character range of the quoted chunk, the stored identity and
# whether the stored text passed admission (full document); web sources never carry them
STORED_PUBLIC_FIELDS = ("span", "span_id", "disposition", "canonical_source_id", "source_id", "admission_passed", "acquisition")


def _licence_id(s: dict):
    """The declared licence id of a stored source (content licence preferred, else metadata licence); None if unknown.
    Taken from the stored provenance record, never inferred."""
    lic = s.get("licence")
    if isinstance(lic, dict):
        for scope in ("content", "metadata"):
            rec = lic.get(scope)
            if isinstance(rec, dict) and rec.get("id"):
                return str(rec["id"])
    return None


def _citation_strings(sources: list[dict]) -> list[dict]:
    """NONASK-1103 C2: the only citation additions to public output ({reference, in_text}, aligned to `sources`).
    Raw ids, authors, source type and access dates stay dev-only; style flag 'legacy' gives None values."""
    from vera import references_ui

    return references_ui.citation_strings(sources)


def public_sources(sources: list[dict]) -> list[dict]:
    """What the response shows: no abstracts (short-quote rule); id, title, link, date, type and provenance."""
    from vera import answer_contract as contract

    cit = _citation_strings(sources)

    def one(s, i):
        out = {k: s.get(k) for k in PUBLIC_FIELDS}
        out.update(cit[i])
        if _public_pmcid(s):  # PMCID-PUBLIC-1023 (owner 2026-10-23): public identifier, validated, shown when known
            out["pmcid"] = _public_pmcid(s)
        if s.get("span"):  # stored-corpus source: provenance of the quoted chunk and its acquisition state (from stored provenance)
            s = {**s, "acquisition": contract.acquisition_state(s)}
            out.update({k: s[k] for k in STORED_PUBLIC_FIELDS if s.get(k) is not None})
            if _licence_id(s):
                out["licence"] = _licence_id(s)
        return out
    return [one(s, i) for i, s in enumerate(sources)]


def count_valid_citations(answer: str, n_sources: int) -> int:
    return sum(1 for g in _CITE_GROUP.findall(answer or "") for n in _expand(g) if 1 <= n <= n_sources)


STORED_CANDIDATE_FIELDS = ("span", "span_id", "disposition", "canonical_source_id", "source_id", "admission_passed")
def _keys_of(r: dict) -> set:
    """URL (trailing slash dropped) and lower-case DOI identity keys of a record or an inline-acquisition entry."""
    out = set()
    ids = r.get("ids") if isinstance(r.get("ids"), dict) else {}
    for u in (r.get("url"), r.get("discovered_url"), ids.get("url")):
        if isinstance(u, str) and u.strip():
            out.add(("url", u.strip().rstrip("/")))
    for d in (r.get("doi"), ids.get("doi")):
        if isinstance(d, str) and d.strip():
            out.add(("doi", d.strip().lower()))
    return out


def irrelevant_marks(inline, code: str = "R-ACQ-IRRELEVANT") -> set:
    """Identity keys of every inline-acquisition entry (discovered + candidates) whose reason is R-ACQ-IRRELEVANT."""
    out: set = set()
    if not isinstance(inline, dict):
        return out
    for e in [*(inline.get("discovered") or ()), *(inline.get("candidates") or ())]:
        if isinstance(e, dict) and str(e.get("reason") or "") == code:
            out |= _keys_of(e)
    return out


def drop_irrelevant(sources, inline=None) -> list:
    """DDFIX-1023/INTEG item 6 (owner objected to unrelated records in the reading list): remove every record this
    request's inline acquisition marked R-ACQ-IRRELEVANT (pre-fetch relevance drop or full-text admission), matched by
    URL or DOI, but ONLY when at least one other record remains; otherwise the list is returned unchanged. Stored-corpus
    items are never removed. Keys on the recorded status only (no separate judgement here). Never raises (fails open to
    the input list: the previous behaviour)."""
    srcs = [s for s in sources or () if isinstance(s, dict)]
    try:
        marks = irrelevant_marks(inline)
        if not marks:
            return srcs
        keep = [s for s in srcs if s.get("span") or s.get("in_corpus") or not (_keys_of(s) & marks)]
        return keep if keep else srcs
    except Exception:  # noqa: BLE001
        return srcs


PUBLIC_CANDIDATE_FIELDS = ("n", "title", "url", "published", "identifier", "provider", "licence_decision",
                           "reference", "in_text", "characterisation")  # NONASK-1103 C2: formatted strings only


def public_candidates(sources: list[dict]) -> tuple:
    """K2 L0 candidates for the response: the allowlisted fields only, never abstract/snippet text (twin of
    public_sources). `acquisition` and `representation` are derived per candidate from the acquisition state (D2),
    never from fetch_status."""
    from vera import answer_contract as contract

    out = []
    cit = _citation_strings(sources)
    for i, s in enumerate(sources):
        c = {k: s.get(k) for k in PUBLIC_CANDIDATE_FIELDS}
        c.update(cit[i])
        if _public_pmcid(s):  # PMCID-PUBLIC-1023 (owner 2026-10-23): public identifier, validated, shown when known
            c["pmcid"] = _public_pmcid(s)
        c["acquisition"] = contract.acquisition_state(s)
        if c["acquisition"] == contract.ACQ_NOT_ACQUIRED and (s.get("span") or s.get("in_corpus")):
            c["acquisition"] = contract.ACQ_IN_CORPUS  # WED-ACQ A1: stored in the collection (not "not acquired")
        c["representation"] = contract.representation(c["acquisition"])
        c["retrieved_at"] = s.get("retrieved_at")
        if s.get("span"):  # stored candidates only; web candidates keep their exact shape
            c.update({k: s[k] for k in STORED_CANDIDATE_FIELDS if s.get(k) is not None})
            if _licence_id(s):
                c["licence"] = _licence_id(s)
        else:  # P0-5B: the index's METADATA licence id, when the record declares one (never an article licence)
            meta = (s.get("licence") or {}).get("metadata") if isinstance(s.get("licence"), dict) else None
            if isinstance(meta, dict) and meta.get("id"):
                c["metadata_licence"] = str(meta["id"])[:64]
        out.append(c)
    return tuple(out)
