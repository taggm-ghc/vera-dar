"""OpenAI request handling for /ask: the completion call, token-usage
extraction, and cost calculation. OpenAI SDK exceptions intentionally
propagate to the routing layer, where main.py translates provider failures
into HTTP responses — this module has no knowledge of HTTP.
"""
import json
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
        if acq is not None:
            from vera import acq_config
            if (acq_config.trigger_for_decline(acq, decision.layer, decision.reason) and _layer1_clean(question)
                    and acq_config.layer2_category_ok(acq, decision)):  # DAR-1021 category gate (fails closed)
                trig = "decline"
                if decision.layer != "layer1" and acq_config.override_layer2_declines(acq):  # ships OFF (plan V1)
                    out = _fulldoc(question, model, pricing, decision, chained, grounding, acquired_only=True)
                    if isinstance(out, AskResult):  # DAR-1021 SEC M1: scope notice first; no memory (accepted=False)
                        pre = acq_config.override_answer_prefix(acq)
                        return replace(out, answer=f"{pre} {out.answer}" if pre else out.answer, accepted=False,
                                       contract={**(out.contract or {}), "offtopic": True, "acquired_only": True})
        # DAR-1021 (FIX1 + P1.7): a qualifying layer-2 out-of-scope verdict shows the FIX1 notice + scholarly reading
        # list AND keeps P1.7's background acquisition trigger (same trig as the refusal would carry). The list itself
        # never admits anything (sources=()); acquired documents never answer a declined question unless
        # override_layer2_declines is true (ships false).
        off = _offtopic_reading_list(decision, grounding)  # FIX1-1021 (config-gated); None = keep the refusal
        if off is not None:
            if not trig:
                return off
            from vera import acq_config
            note = acq_config.reading_list_notice(acq)
            return replace(off, answer=f"{off.answer} {note}".strip() if note else off.answer, acquire_trigger=trig)
        return AskResult(answer=decline_text(), tokens_used=decision.tokens_used, cost_usd=decision.cost_usd,
                         result_kind="decline", acquire_trigger=trig)
    from vera import answer_contract as contract

    fd = _fulldoc(question, model, pricing, decision, chained, grounding)  # D-055; None = path not configured
    if isinstance(fd, AskResult):  # grounded in admitted passages of inspected, verified full documents
        return fd
    if fd is None:
        return _reading_list(question, model, pricing, decision, chained, grounding, contract)
    ev, token, tokens, cost, text = fd  # fail closed: no generated answer, never an abstract fallback
    res = _reading_list(question, model, pricing, decision, chained, grounding, contract, tokens=tokens, cost=cost)
    return replace(res, answer=text, evidence=ev,
                   contract={**(res.contract or {"layers": contract.layers()}), "fallback": token},
                   **({"acquire_trigger": "insufficient"} if acq is not None and _acq_on_insufficient(acq) else {}))


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
    """plan s11 V1: re-run the layer-1 injection and deny patterns before any acquisition (never trigger on a hit)."""
    try:
        from vera import scope_router as sr
        cfg = sr.load_config()
        verdict, why = sr.layer1(question, cfg)
        return not (verdict == sr.DECLINE)
    except Exception:  # noqa: BLE001  fail closed: no acquisition
        return False


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
    try:
        from vera.grounded_ask import public_candidates

        cands = public_candidates(sources_fn(restricted=True) or [])
    except Exception as exc:  # noqa: BLE001  (fail closed: the fixed refusal; class only in the log)
        import logging

        logging.getLogger("vera").warning("off-topic reading list unavailable (%s)", type(exc).__name__)
        return None
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


def _fulldoc(question, model, pricing, decision, chained, grounding, acquired_only: bool = False):
    """D-055 Option C: full-document acquisition/verification -> span retrieval -> admission -> generation -> claim
    check. None when the path is not configured (today's behaviour exactly); an AskResult (result_kind "answer") when
    grounded; else (evidence, fallback token, tokens, cost, text) for an explicit insufficient-evidence result.
    Provider errors from the answering call propagate (the #77 chain tries the next provider)."""
    import logging

    from vera import answer_contract as contract
    from vera import fulldoc_corpus

    try:
        cfg = fulldoc_corpus.load_config()
        corpus = fulldoc_corpus.get_corpus(cfg)
        if corpus is None:
            return None
        from vera.fulldoc_admission import admit

        if acquired_only:  # WED-ACQ2 scope override (ships off): only acquired documents may answer a declined question
            acq_corpus = fulldoc_corpus.get_acquired_corpus(cfg)
            ev = admit(question, acq_corpus, cfg) if acq_corpus is not None else None
            if ev is None or contract.result_kind_for((ev,)) != "answer":
                return None
        else:
            ev = admit(question, corpus, cfg)
            if contract.result_kind_for((ev,)) != "answer" and ev.status != "corpus_unusable":
                # WED-ACQ2 (plan V6): base corpus first, exactly as before; the separate acquired corpus is consulted
                # only when the base corpus cannot answer, so base answers never change.
                acq_corpus = fulldoc_corpus.get_acquired_corpus(cfg)
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
                                base_tokens=decision.tokens_used, base_cost=decision.cost_usd)
    if out.result is not None and out.result.result_kind == "answer":
        return replace(out.result, evidence=ev, accepted=True,
                       contract={"layers": contract.layers(), "fallback": None})
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
