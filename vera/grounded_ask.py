"""Item #79: grounded /ask. Live scholarly retrieval once per accepted question, the measured grounded-v1
prompt, and a citation check on the answer.

Settings come from the `grounding` block of config/ask-provider-chain.json. Retrieval uses VERA-DAR's own
keyless search providers; results with an abstract are used directly and link-only results are followed to
their original sources (vera/source_resolver.py), so every numbered source gives the model text to ground
on. Nothing is stored: no database, no corpus file.
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


def _mark_in_corpus(results: list, dar_corpus) -> list:
    """Copy of `results` with in_corpus=True on every result whose DOI or URL (normalised, either direction) is that of
    any stored document (DarCorpus.doc_keys). The input list is returned unchanged when nothing matches."""
    from vera.dar_corpus import doc_key_set

    keys_fn = getattr(dar_corpus, "doc_keys", None)  # duck-typed corpora without it: no corpus-wide check
    held = keys_fn() if callable(keys_fn) else ()
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
            return results
        return dar_corpus.rank_merge(question, list(results), list(stored), int(cfg["num_results"]))
    except Exception as exc:  # noqa: BLE001  (the stored branch never fails the request)
        logger.warning("DAR stored branch skipped (%s)", type(exc).__name__)
        info["stored"] = {"error_class": type(exc).__name__}
        return results


def retrieve_all(question: str, cfg: dict, search: Callable | None = None,
                 resolver: Callable | None = None, trace_out: dict | None = None,
                 dar_corpus=None) -> tuple[list[dict], list[dict]]:
    """(sources, unused): numbered sources with abstracts [{n, title, url, published, snippet, source_type}],
    and original sources that were traced but have no free abstract (listed, never used). ([], []) on failure.
    Optional, additive: trace_out (a dict filled with COUNTS-ONLY routing/stored-branch aggregates) and dar_corpus
    (a vera.dar_corpus.DarCorpus; None = no stored branch). The return contract is unchanged."""
    info: dict = {}
    try:
        if search is None:
            from vera.search_and_fetch import search_question

            providers = _provider_chain(cfg["providers"])
            # per-provider timeout below the overall deadline, so one slow provider cannot starve the others
            results = search_question(question, num_results=int(cfg["num_results"]), providers=providers,
                                      min_candidates=1, mode="fan_out", deadline_s=float(cfg["deadline_s"]),
                                      timeout_s=float(cfg.get("provider_timeout_s", 5)),
                                      max_retries=int(cfg.get("provider_max_retries", 1)))
            try:
                from vera import domain_routing as _dr
                if _dr.flag_enabled():
                    results, rtrace = _dr.route_search(question, results, _dr.make_search_fn(search_question, providers, cfg), (dar_corpus.registry if dar_corpus is not None else _dr.get_registry()), _dr.get_cfg(), num_results=int(cfg["num_results"]))
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
    for r in direct:
        snippet = clean_text(r.get("snippet"), int(cfg["snippet_max_chars"]))
        url = str(r.get("url") or "")
        key = str(r.get("doi") or "").lower() or url.lower().rstrip("/")
        if not url.lower().startswith(("https://", "http://")) or key in seen:
            continue
        if not snippet:
            if r.get("discovered_via"):  # a traced original without a free abstract: show it, never ground on it
                seen.add(key)
                unused.append({"title": clean_text(r.get("title"), 300) or url, "url": url,
                               "published": (str(r.get("published"))[:10] if r.get("published") else None),
                               "discovered_via": str(r["discovered_via"]),
                               **{k: r[k] for k in ("doi", "pmcid", "openalex_id", "doaj_id", "provider") if r.get(k)}})  # identifiers for acquisition only; traced_sources shows title/url/published
            continue
        seen.add(key)
        src = {"n": len(out) + 1, "title": clean_text(r.get("title"), 300) or url, "url": url,
               "published": (str(r.get("published"))[:10] if r.get("published") else None),
               "snippet": snippet, "source_type": r.get("source_type")}
        if r.get("discovered_via"):
            src["discovered_via"] = str(r["discovered_via"])
        for k in ("licence", "stored_text_kind", "doi", "pmcid", "openalex_id", "doaj_id", "provider", "admission_passed", "canonical_source_id", "source_id", "disposition", "in_corpus"):  # internal, for the corpus gauntlet (pmcid is published only via public_sources/_pmcid_of, PMCID-PUBLIC-1023)
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
    return grounded_v1_messages(question, sources)


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
                 "licence_decision")


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


def public_sources(sources: list[dict]) -> list[dict]:
    """What the response shows: no abstracts (short-quote rule); id, title, link, date, type and provenance."""
    from vera import answer_contract as contract

    def one(s):
        out = {k: s.get(k) for k in PUBLIC_FIELDS}
        if _public_pmcid(s):  # PMCID-PUBLIC-1023 (owner 2026-10-23): public identifier, validated, shown when known
            out["pmcid"] = _public_pmcid(s)
        if s.get("span"):  # stored-corpus source: provenance of the quoted chunk and its acquisition state (from stored provenance)
            s = {**s, "acquisition": contract.acquisition_state(s)}
            out.update({k: s[k] for k in STORED_PUBLIC_FIELDS if s.get(k) is not None})
            if _licence_id(s):
                out["licence"] = _licence_id(s)
        return out
    return [one(s) for s in sources]


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


PUBLIC_CANDIDATE_FIELDS = ("n", "title", "url", "published", "identifier", "provider", "licence_decision")


def public_candidates(sources: list[dict]) -> tuple:
    """K2 L0 candidates for the response: the allowlisted fields only, never abstract/snippet text (twin of
    public_sources). `acquisition` and `representation` are derived per candidate from the acquisition state (D2),
    never from fetch_status."""
    from vera import answer_contract as contract

    out = []
    for s in sources:
        c = {k: s.get(k) for k in PUBLIC_CANDIDATE_FIELDS}
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
