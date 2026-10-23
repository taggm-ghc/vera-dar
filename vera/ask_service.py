"""OpenAI request handling for /ask: the completion call, token-usage
extraction, and cost calculation. OpenAI SDK exceptions intentionally
propagate to the routing layer, where main.py translates provider failures
into HTTP responses — this module has no knowledge of HTTP.
"""
import json
import logging
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

from vera.pricing.config import PricingRecord


@dataclass(frozen=True)
class AskResult:
    answer: str
    tokens_used: int
    cost_usd: float
    sources: tuple = ()  # item #79: numbered sources the answer was grounded in (empty when ungrounded)
    claim_check: dict | None = None  # item #79: per-claim check summary (grounded path only)
    accepted: bool = False  # item #84: the scope router accepted the question (memory recall/write only then)
    claims: tuple = ()  # item #84: internal per-claim records [{id, sentence, citations, verdict}]
    # K2 answer contract (plan-K2-contract.md r2). The default is a non-answer (V2): "answer" is reachable only through
    # the L3 seam (vera.answer_contract.examined_full_documents), which returns [] today.
    result_kind: str = "unspecified"  # answer | discovery | reading_list | decline | unspecified
    candidates: tuple = ()  # public, allowlisted candidate dicts (no abstract text); sources stays the internal list (V1)
    contract: dict | None = None  # {"layers": [...], "fallback": None | token}
    # D-055 Option C (full-document-grounded answers): the vera.fulldoc_evidence.EvidenceBundle for this question (None
    # when the full-document path is not configured) and the answer's citations bound to admitted spans.
    evidence: object | None = None
    citations: tuple = ()
    # WED-ACQ2 (P1.7): None, or "decline" / "insufficient" when this result should start ONE background acquisition.
    # Set only while the feature is effectively on; main.py enqueues once, after the provider chain succeeded.
    acquire_trigger: str | None = None


_CHAIN_CONFIG = Path(__file__).resolve().parent.parent / "config" / "ask-provider-chain.json"


def _chain_timeout_s() -> float | None:
    """The /ask chain's per-call timeout (config/ask-provider-chain.json `timeout_s`); None if unreadable."""
    try:
        return float(json.loads(_CHAIN_CONFIG.read_text())["timeout_s"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


class OpenAIFallbackDisabled(RuntimeError):
    """The default (OpenAI) client was requested while the OpenAI fallback is switched off for this process."""


_DEFAULT_CLIENT_FORBIDDEN = False  # process latch; set only by the API process (main.py) when the fallback is off


def forbid_default_client() -> None:
    global _DEFAULT_CLIENT_FORBIDDEN
    _DEFAULT_CLIENT_FORBIDDEN = True


def reset_default_client_latch() -> None:
    """Clear the latch (tests only; production sets it once at startup and never clears it)."""
    global _DEFAULT_CLIENT_FORBIDDEN
    _DEFAULT_CLIENT_FORBIDDEN = False


def _get_client() -> OpenAI:
    """Latch check first (never reaches the constructor, so it cannot be bypassed by the cache), then the cached
    constructor."""
    if _DEFAULT_CLIENT_FORBIDDEN:
        raise OpenAIFallbackDisabled("the OpenAI fallback is disabled (config/ask-provider-chain.json)")
    return _build_client()


@lru_cache(maxsize=1)
def _build_client() -> OpenAI:
    """Construct the OpenAI client on first use, not at module import time. S4: the last-resort client honours the
    chain timeout and makes no SDK-internal retries (the SDK default would be 600 s and 2 retries)."""
    t = _chain_timeout_s()
    return OpenAI(max_retries=0, **({"timeout": t} if t else {}))


_get_client.cache_clear = _build_client.cache_clear  # existing callers/tests clear the cache through the wrapper


def answer_question(question: str, model: str, pricing: PricingRecord, client: OpenAI | None = None,
                    extra_body: dict | None = None) -> AskResult:
    """client/extra_body are set only by the /ask provider chain (item #77); every other caller gets the
    default OpenAI client, so the evaluation paths are unchanged."""
    return complete([{"role": "user", "content": question}], model, pricing, client=client, extra_body=extra_body)


def complete(messages: list[dict], model: str, pricing: PricingRecord, client: OpenAI | None = None,
             extra_body: dict | None = None) -> AskResult:
    """One chat completion with token and cost accounting (shared by the bare and grounded /ask paths)."""
    completion = (client or _get_client()).chat.completions.create(
        model=model,
        messages=messages,
        **({"extra_body": extra_body} if extra_body else {}),
    )

    usage = completion.usage
    input_cost = (usage.prompt_tokens / 1_000_000) * pricing.input
    output_cost = (usage.completion_tokens / 1_000_000) * pricing.output

    return AskResult(
        answer=completion.choices[0].message.content,
        tokens_used=usage.total_tokens,
        cost_usd=round(input_cost + output_cost, 6),
    )


def answer_in_scope(question: str, model: str, pricing: PricingRecord, client: OpenAI | None = None,
                    extra_body: dict | None = None, grounding=None) -> AskResult:
    """The /ask entry point (item #72 W1): the scope router runs first; a declined question never reaches the
    answering model and gets the fixed decline template. answer_question stays the bare model call (the trace
    harness's baseline shape)."""
    from vera.scope_router import decline_text, route

    from vera.scope_router import default_chat

    def chat(messages, model_, l2):
        return default_chat(messages, model_, l2, client=client, extra_body=extra_body)

    decision = route(question, model, pricing, chat=chat if (client or extra_body) else None)
    acq = _acq_cfg_if_on()  # WED-ACQ2: None unless automatic acquisition is effectively on (byte-identical when off)
    chained = {k: v for k, v in (("client", client), ("extra_body", extra_body)) if v}  # only from the #77 chain
    if not decision.accepted:
        trig = None
        inline_note = None  # DAR-INLINE-1021: set only when an inline search was made for this request
        ia: dict = {"status": "off"} if acq is None else {}  # P0-B1b: inline outcome for a declined non-answer (Report)
        if acq is not None:
            from vera import acq_config
            if (acq_config.trigger_for_decline(acq, decision.layer, decision.reason) and _layer1_clean(question)
                    and acq_config.layer2_category_ok(acq, decision)):  # DAR-1021 category gate (fails closed)
                trig = "decline"
                if decision.layer != "layer1" and acq_config.override_layer2_declines(acq):  # ships OFF (plan V1)
                    used: dict = {}  # R2: the served acquired corpus this attempt actually tried (None = not loaded yet)
                    out = _fulldoc(question, model, pricing, decision, chained, grounding, acquired_only=True, used=used)
                    if isinstance(out, AskResult):  # DAR-1021 SEC M1: scope notice first; no memory (accepted=False)
                        pre = acq_config.override_answer_prefix(acq)
                        return replace(out, answer=f"{pre} {out.answer}" if pre else out.answer, accepted=False,
                                       contract={**(out.contract or {}), "offtopic": True, "acquired_only": True})
                    info: dict = {}
                    inl = _inline(question, model, pricing, decision, chained, grounding, acq,
                                  tried=_doc_keys(used.get("acq")), info=info)
                    if isinstance(inl, AskResult):
                        return inl
                    inline_note = inl
                    ia = info.get("outcome") or {}
        iac = {"inline_acquisition": ia} if ia else {}
        # DAR-1021 (FIX1 + P1.7): a qualifying layer-2 out-of-scope verdict shows the FIX1 notice + scholarly reading
        # list AND keeps P1.7's background acquisition trigger (same trig as the refusal would carry). The list itself
        # never admits anything (sources=()); acquired documents never answer a declined question unless
        # override_layer2_declines is true (ships false).
        off = _offtopic_reading_list(decision, grounding)  # FIX1-1021 (config-gated); None = keep the refusal
        if off is not None:
            if not trig:
                return replace(off, contract={**(off.contract or {}), **iac}) if iac else off
            from vera import acq_config
            note = acq_config.reading_list_notice(acq) if inline_note is None else inline_note
            return replace(off, answer=f"{off.answer} {note}".strip() if note else off.answer,
                           contract={**(off.contract or {}), **_inline_trace(inline_note), **iac} if (_inline_trace(inline_note) or iac) else off.contract,
                           acquire_trigger=trig if inline_note is None else None,  # inline searched: no 2nd job
                           **_spent(inline_note))
        return replace(AskResult(answer=decline_text(), tokens_used=decision.tokens_used, cost_usd=decision.cost_usd,
                                 result_kind="decline", acquire_trigger=trig if inline_note is None else None,
                                 **({"contract": iac} if iac and acq is not None else {})),  # feature off: the bare refusal stays byte-identical
                       **_spent(inline_note))
    from vera import answer_contract as contract

    used_in: dict = {}
    fd = _fulldoc(question, model, pricing, decision, chained, grounding, used=used_in)  # D-055; None = not configured
    if isinstance(fd, AskResult):  # grounded in admitted passages of inspected, verified full documents
        return fd
    if fd is None:
        return _reading_list(question, model, pricing, decision, chained, grounding, contract)
    ev, token, tokens, cost, text = fd  # fail closed: no generated answer, never an abstract fallback
    inline_note = None
    ia: dict = {}  # P0-B1: the inline outcome for a non-answer, so the Report can say whether a search was made
    if acq is None:
        ia = {"status": "off"}  # automatic acquisition effectively off (flag, kill switch or DB env): not attempted
    elif _layer1_clean(question):  # DAR-INLINE-1021-R2 must-have 1: in-scope, nothing answered
        from vera import inline_acquire
        from vera import fulldoc_corpus
        ft = fulldoc_corpus.load_config()["fallback_tokens"]
        # INT-1022-B (orchestrator-approved): a base-corpus 'nonresponsive' result (RESP-1022: every verified claim was
        # off-question) also starts inline acquisition. _inline is called at most ONCE per request: if its own answer
        # is nonresponsive too, it returns a notice and the request falls through to the reading list (no loop).
        retry_tokens = (ft["insufficient"], ft["corpus_unusable"], ft.get("nonresponsive"))
        if inline_acquire.icfg_of(acq).get("in_scope") is True and token and token in retry_tokens:
            info: dict = {}
            inl = _inline(question, model, pricing, decision, chained, grounding, acq, tried=_doc_keys(used_in.get("acq")),
                          in_scope=True, info=info)
            if isinstance(inl, AskResult):
                return inl
            inline_note = inl
            ia = info.get("outcome") or {}
    res = _reading_list(question, model, pricing, decision, chained, grounding, contract, tokens=tokens, cost=cost)
    trig = {"acquire_trigger": "insufficient"} if acq is not None and _acq_on_insufficient(acq) and inline_note is None else {}
    return replace(res, answer=f"{text} {inline_note}" if inline_note else text, evidence=ev,
                   contract={**(res.contract or {"layers": contract.layers()}), "fallback": token,
                             **_inline_trace(inline_note), **({"inline_acquisition": ia} if ia else {})}, **trig,
                   **_spent(inline_note, res.tokens_used, res.cost_usd, decision.tokens_used, decision.cost_usd))


class _InlineNotice(str):
    """Honest inline-search notice text that also carries the specific fallback token (None for non-fallback notices)."""
    token = None


class _Note(_InlineNotice):
    """DAR-TRACE-1022: an inline notice that also carries the INLINE attempt's tokens/cost (classifier + inline answer
    writer + claim check) when the writer ran but no checked answer came out, so tokens_used no longer hides that spend.
    Review S6 limit: on the off-topic path a writer run on the earlier STORED acquired-corpus attempt (rare: it needs
    an admitting stored corpus whose answer then failed the check) is still not added."""

    def __new__(cls, text, tokens, cost):
        o = super().__new__(cls, text)
        o.tokens, o.cost = int(tokens or 0), float(cost or 0.0)
        return o


def _spent(note, base_tokens=None, base_cost=None, classifier_tokens=0, classifier_cost=0.0) -> dict:
    """tokens_used / cost_usd overrides for a result carrying an inline _Note. With base_* (in-scope path, review M1):
    the base attempt's total PLUS the inline spend, with the classifier counted once."""
    if not isinstance(note, _Note):
        return {}
    if base_tokens is None:
        return {"tokens_used": note.tokens, "cost_usd": round(note.cost, 6)}
    return {"tokens_used": int(base_tokens) + note.tokens - int(classifier_tokens or 0),
            "cost_usd": round(float(base_cost or 0.0) + note.cost - float(classifier_cost or 0.0), 6)}


def _doc_keys(corpus) -> set:
    """doc_key_set keys of a served corpus (empty for None): what the stored-corpus attempt already tried."""
    try:
        dar = getattr(corpus, "_dar", None)
        return set(dar.doc_keys()) if dar is not None and hasattr(dar, "doc_keys") else set()
    except Exception:  # noqa: BLE001
        return set()


def _inline_trace(note) -> dict:
    tok = getattr(note, "token", None)
    return {"inline_fallback": tok} if tok else {}


def _inline(question, model, pricing, decision, chained, grounding, acq, tried=None, in_scope=False, info=None):
    """DAR-INLINE-1021: acquire inline (vera.inline_acquire) and answer from an EPHEMERAL corpus of exactly the
    just-admitted documents through the unchanged admission, answer writer and claim check. Returns an AskResult (an
    answer), else the honest reading-list notice (str) when a search was made, else None (no search: today's path).
    P0-B1: `info` (a dict) receives "outcome", the public inline outcome ({"status": ...}), for the non-answer contract."""
    from vera import acq_config, fulldoc_corpus, inline_acquire

    if not fulldoc_corpus.flag_enabled(fulldoc_corpus.load_config()):  # VER L6: no answer path -> never acquire inline
        if info is not None:
            info["outcome"] = {"status": "off"}
        return None
    from vera import acq_discovered
    kw = {}  # P0-5 (inline.use_discovered, default off): the request's own cached discovery results; off = unchanged call
    if acq_discovered.use_discovered(acq):  # the off-topic path lists the restricted search, in-scope the full one
        kw["discovered"] = acq_discovered.discovery_results(grounding, restricted=not in_scope)
    o = inline_acquire.acquire(question, acq, served=tried, **kw)  # R2: only documents actually tried are skipped
    if info is not None:
        st = str(getattr(o, "status", "error"))  # "off" carries no counts: identical whether the flag or the run said off
        info["outcome"] = o.public() if st != "off" and hasattr(o, "public") else {"status": st}
    if o.corpus is None:
        return inline_acquire.notice(acq, o.status)
    out = _fulldoc(question, model, pricing, decision, chained, grounding, acquired_only=True, ephemeral=o.corpus)
    if not isinstance(out, AskResult):
        # post-writer fallback: keep the specific token (response telemetry + log); user text stays the generic
        # "unanswered" notice except for premise_unverified, which has its own fixed text. INT-1022-B merge: the
        # DAR-TRACE-1022 _Note (an _InlineNotice subclass) also carries the inline writer/check spend.
        tok = out[1] if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], str) else None
        logging.getLogger("vera").info("inline answer fallback token=%s", tok)
        from vera.fulldoc_answer import PREMISE_FALLBACK, PREMISE_UNVERIFIED_TEXT
        text = PREMISE_UNVERIFIED_TEXT if tok == PREMISE_FALLBACK else inline_acquire.notice(acq, "unanswered")
        if text is None:
            return None
        note = (_Note(text, out[2], out[3]) if isinstance(out, tuple) and len(out) == 5  # DAR-TRACE-1022: spend kept
                else _InlineNotice(text))
        note.token = tok
        return note
    if in_scope:  # R2 must-have 1: the normal in-scope wording (no off-topic prefix); memory rules as any in-scope answer
        note = inline_acquire.answer_note(acq, o)
        return replace(out, answer=f"{note} {out.answer}" if note else out.answer,
                       contract={**(out.contract or {}), "acquired_only": True, "inline_acquisition": o.public()})
    lead = " ".join(t for t in (acq_config.override_answer_prefix(acq), inline_acquire.answer_note(acq, o)) if t)
    return replace(out, answer=f"{lead} {out.answer}" if lead else out.answer, accepted=False,
                   contract={**(out.contract or {}), "offtopic": True, "acquired_only": True,
                             "inline_acquisition": o.public()})


def _base_warming() -> bool:
    """True while the base-corpus startup warm-up is still running (never raises; False when unknown)."""
    try:
        from vera import fulldoc_warm
        return fulldoc_warm.warming()
    except Exception:  # noqa: BLE001
        return False


def _acq_cfg_if_on():
    """The auto_acquire config when the feature is effectively on (flag, kill switch, DB env), else None. Never raises."""
    import os

    try:
        from vera import acq_config
        cfg = acq_config.load_config()
        return cfg if acq_config.enabled(cfg, os.environ) else None
    except Exception:  # noqa: BLE001
        return None


def _acq_on_insufficient(acq: dict) -> bool:
    return ((acq.get("triggers") or {}).get("on_insufficient")) is True


def _layer1_clean(question: str) -> bool:
    """plan s11 V1: re-run the layer-1 injection and deny patterns before any acquisition (never trigger on a hit).
    SCOPE-1022: the shared intent rule (scope_router.intent_clean), so routing, discovery and acquisition agree."""
    try:
        from vera import scope_router as sr
        return sr.intent_clean(question)
    except Exception:  # noqa: BLE001  fail closed: no acquisition
        return False


try:  # SCOPE-1022: one closed set of refusal categories (scope_router.REFUSE_CATEGORIES)
    from vera.scope_router import REFUSE_CATEGORIES as _NO_LIST_CATEGORIES
except Exception:  # noqa: BLE001
    _NO_LIST_CATEGORIES = frozenset({"advice", "harmful", "programming_task", "rule_change"})


def _offtopic_reading_list(decision, grounding):
    """FIX1-1021: a well-formed layer-2 out-of-scope verdict (layer 1 matched nothing) returns a fixed notice plus a
    scholarly-only reading list instead of the refusal. No model call, no full-document path, no answer. The result
    carries sources=() and accepted=False so main.py neither admits anything to the corpus nor touches memory.
    Returns None (caller refuses as before) when the flag is off, the verdict does not qualify or no search seam exists."""
    from vera import answer_contract as contract
    from vera import scope_router as sr

    if grounding is None or not sr.offtopic_applies(decision):
        return None
    if getattr(decision, "category", "") in _NO_LIST_CATEGORIES:  # DAR-1021 SEC M2: classifier said advice/harm/...
        return None
    sources_fn, _cfg = grounding
    cands = ()
    tok = None
    try:
        from vera import search_providers
        from vera.grounded_ask import public_candidates

        profile = sr.offtopic_openalex_profile()
        tok = search_providers.OPENALEX_PROFILE.set(profile)  # RL-REL-1021 (None = unchanged)
        cands = public_candidates(sources_fn(restricted=True) or [])
        if not cands and profile is not None:  # RL-REL-1021-B: AND query empty -> ONE more call, old OR query, off-topic only
            search_providers.OPENALEX_PROFILE.reset(tok)
            tok = search_providers.OPENALEX_PROFILE.set(None)
            cands = public_candidates(sources_fn(restricted=True) or [])
    except Exception as exc:  # noqa: BLE001  (fail closed: the fixed refusal; class only in the log)
        import logging

        logging.getLogger("vera").warning("off-topic reading list unavailable (%s)", type(exc).__name__)
        return None
    finally:
        if tok is not None:
            search_providers.OPENALEX_PROFILE.reset(tok)
    notice = sr.offtopic_notice()
    res = contract.reading_list_result(cands, decision.tokens_used, decision.cost_usd, accepted=False)
    return replace(res, answer=notice if cands else f"{notice} {res.answer}", sources=(),
                   contract={**(res.contract or {}), "offtopic": True})


def _reading_list(question, model, pricing, decision, chained, grounding, contract, tokens=None,
                  cost=None) -> AskResult:
    """The L0 reading-list result (unchanged K2 behaviour). tokens/cost default to the router's."""
    tokens = decision.tokens_used if tokens is None else tokens
    cost = decision.cost_usd if cost is None else cost
    if grounding is None:  # V2: never answer from the bare model; the L0 no-sources result instead
        return contract.reading_list_result((), tokens, cost)
    sources_fn, cfg = grounding  # item #79: sources_fn caches one search per request
    sources = sources_fn()
    if not sources:  # never answer from memory
        return contract.reading_list_result((), tokens, cost, empty_text=cfg.get("no_sources_text"))
    from vera.grounded_ask import public_candidates

    candidates = public_candidates(sources)
    if contract.result_kind_for(candidates) == "answer":  # unreachable: dict candidates never satisfy the L3 seam
        return _answer_grounded(question, model, pricing, sources, cfg, decision, chained)  # (only a grounded
    #                                                       EvidenceBundle does; tests monkeypatch the seam to reach it)
    _log_acquisition_counts(candidates)
    # L0 reading list: no model call after the router, no fix_citations, no claim check, no claims (C-1)
    return contract.reading_list_result(candidates, tokens, cost, sources=sources)


def _log_acquisition_counts(candidates) -> None:
    """WED-ACQ A2 (owner decision D2, counts only): one log line with how many listed candidates are not yet in the
    collection, in total and per search provider. No question, title, URL, identifier or snippet is logged, so the
    R-028 disclosure ("does not save ... from your questions") stays true. Never fails the request."""
    import logging
    from collections import Counter

    from vera import answer_contract as contract

    try:
        out = [c for c in candidates if c.get("acquisition") == contract.ACQ_NOT_ACQUIRED]
        allowed = contract.acquisition_log_providers()
        by = Counter(p if p in allowed else "other" for p in (str(c.get("provider") or "").lower() for c in out))
        logging.getLogger(__name__).info("acquisition candidates: %s", json.dumps(
            {"listed": len(candidates), "not_in_collection": len(out), "by_provider": dict(sorted(by.items()))}))
    except Exception:  # noqa: BLE001  (a counts log never fails the request)
        pass


def _fulldoc(question, model, pricing, decision, chained, grounding, acquired_only: bool = False, ephemeral=None,
             used: dict | None = None):
    """D-055 Option C: full-document acquisition/verification -> span retrieval -> admission -> generation -> claim
    check. None when the path is not configured (today's behaviour exactly); an AskResult (result_kind "answer") when
    grounded; else (evidence, fallback token, tokens, cost, text) for an explicit insufficient-evidence result.
    Provider errors from the answering call propagate (the #77 chain tries the next provider)."""
    import logging

    from vera import answer_contract as contract
    from vera import fulldoc_corpus

    try:
        cfg = fulldoc_corpus.load_config()
        # DAR-1021-ASKDIAG: the acquired-only branch never reads the base corpus, so it must not wait on its load
        # (startup warm-up follower wait up to db_bounds.single_flight_wait_s, or a synchronous DB read + CPU build).
        # DAR-INLINE-1021-R2 must-have 2: while the base warm-up is still running, never wait on it either: the stored
        # acquired documents answer, else the caller tries inline acquisition.
        warming = not acquired_only and _base_warming()
        corpus = None if (acquired_only or warming) else fulldoc_corpus.get_corpus(cfg)
        if (not fulldoc_corpus.flag_enabled(cfg)) if (acquired_only or warming) else corpus is None:
            return None
        if warming:
            acquired_only = True  # answer only from acquired documents (their own claim-check reference)
        from vera.fulldoc_admission import admit

        if acquired_only:  # WED-ACQ2 scope override (ships off): only acquired documents may answer a declined question
            # DAR-INLINE-1021: `ephemeral` = the in-memory corpus of the documents admitted inline for this request
            acq_corpus = ephemeral if ephemeral is not None else fulldoc_corpus.get_acquired_corpus(cfg)
            if used is not None:
                used["acq"] = acq_corpus
            ev = admit(question, acq_corpus, cfg) if acq_corpus is not None else None
            if ev is None or contract.result_kind_for((ev,)) != "answer":
                if warming:  # in-scope while warming: an explicit insufficient result (the caller may go inline)
                    return (ev, cfg["fallback_tokens"]["insufficient"], decision.tokens_used, decision.cost_usd,
                            _with_count(cfg["answer"]["insufficient_text"], ev))
                return None
        else:
            ev = admit(question, corpus, cfg)
            if contract.result_kind_for((ev,)) != "answer" and ev.status != "corpus_unusable":
                # WED-ACQ2 (plan V6): base corpus first, exactly as before; the separate acquired corpus is consulted
                # only when the base corpus cannot answer, so base answers never change.
                acq_corpus = fulldoc_corpus.get_acquired_corpus(cfg)
                if used is not None:
                    used["acq"] = acq_corpus
                if acq_corpus is not None:
                    ev2 = admit(question, acq_corpus, cfg)
                    if contract.result_kind_for((ev2,)) == "answer":
                        ev = ev2
    except Exception as exc:  # noqa: BLE001  (fail closed: unusable corpus, no answer; class only in the log)
        logging.getLogger("vera").warning("full-document path unavailable (%s)", type(exc).__name__)
        try:
            cfg
        except NameError:
            return None  # no config at all: the path is not configured
        return (None, cfg["fallback_tokens"]["corpus_unusable"], decision.tokens_used, decision.cost_usd,
                cfg["answer"]["corpus_unusable_text"])
    texts, tokens_ = cfg["answer"], cfg["fallback_tokens"]
    if contract.result_kind_for((ev,)) != "answer":
        unusable = ev.status == "corpus_unusable"
        return (ev, tokens_["corpus_unusable" if unusable else "insufficient"], decision.tokens_used,
                decision.cost_usd, texts["corpus_unusable_text"] if unusable else _with_count(texts["insufficient_text"], ev))
    from vera import fulldoc_answer

    out = fulldoc_answer.answer(question, ev, model, pricing, chained=chained,
                                grounding_cfg=grounding[1] if grounding else None, cfg=cfg,
                                base_tokens=decision.tokens_used, base_cost=decision.cost_usd,
                                # FIX-CCWAIT-1021: the claim check's verbatim reference is the acquired corpus itself
                                reference_corpus=acq_corpus if acquired_only else None)
    if out.result is not None and out.result.result_kind == "answer":
        return replace(out.result, evidence=ev, accepted=True,
                       contract={"layers": contract.layers(), "fallback": None})
    if getattr(out, "premise", None) and out.fallback:  # DAR-PREMISE-1022: flagged premise, correction unverifiable
        from vera.fulldoc_answer import PREMISE_UNVERIFIED_TEXT

        return (ev, out.fallback, out.tokens_used, out.cost_usd, PREMISE_UNVERIFIED_TEXT)
    return (ev, out.fallback or tokens_["insufficient"], out.tokens_used, out.cost_usd,
            _with_count(texts["insufficient_text"], ev))


def _with_count(text: str, ev) -> str:
    """C-4: substitute {n_inspected} (a count only, never titles) with the number of examined full documents. Plain
    str.replace, so braces elsewhere are safe and a text without the placeholder is returned unchanged."""
    try:
        n = len(getattr(ev, "examined_full_documents", ()) or ())
    except TypeError:
        n = 0
    return text.replace("{n_inspected}", str(n))


def _answer_grounded(question, model, pricing, sources, cfg, decision, chained) -> AskResult:
    """State A (evidentiary answer). The ONLY caller is the L3 branch of answer_in_scope, and only when full documents
    were examined; the abstract-grounded behaviour it used to implement for every question is retired (DEC-FDA1)."""
    from vera import answer_contract as contract
    from vera.grounded_ask import count_valid_citations, fix_citations, messages_for, public_candidates

    kind = "answer"
    result = complete(messages_for(question, sources), model, pricing, **chained)
    text, _ = fix_citations(result.answer or "", len(sources), cfg["removed_citation_note"])
    tokens = result.tokens_used + decision.tokens_used
    cost = round(result.cost_usd + decision.cost_usd, 6)
    if count_valid_citations(text, len(sources)) == 0:  # R1 criterion: no valid citation -> no answer
        return contract.reading_list_result(public_candidates(sources), tokens, cost, sources=sources,
                                            fallback="no_valid_citations")  # C-F1: a non-answer, no claim_check
    cc = cfg.get("claim_check") or {}
    if cc.get("enabled"):  # R1 criterion: each cited claim checked against the cited text; fail closed
        from vera.claim_check import RECORDS, check_answer

        records: list = []
        token = RECORDS.set(records)
        try:
            checked, summary = check_answer(text, sources, cc)
        finally:
            RECORDS.reset(token)
        if checked is None:
            token = "checker_unavailable" if summary["outcome"] == "checker_unavailable" else "claim_check_insufficient"
            return contract.reading_list_result(public_candidates(sources), tokens, cost, sources=sources,
                                                fallback=token)  # C-F1
        return AskResult(answer=checked, tokens_used=tokens, cost_usd=cost, sources=tuple(sources),
                         claim_check=summary, accepted=True, claims=tuple(records), result_kind=kind)
    return AskResult(answer=text, tokens_used=tokens, cost_usd=cost, sources=tuple(sources), accepted=True,
                     result_kind=kind)
