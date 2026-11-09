"""OpenAI request handling for /ask: the completion call, token-usage
extraction, and cost calculation. OpenAI SDK exceptions intentionally
propagate to the routing layer, where main.py translates provider failures
into HTTP responses — this module has no knowledge of HTTP.
"""
import asyncio
import contextvars
import json
import logging
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

from vera import progress as _progress
from vera import stream_ctx as _stream_ctx
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
    # ANS-LAYERS-1101: ({"kind", "text"}, ...) process/limitation notes kept out of `answer` (answer_contract.add_note)
    notes: tuple = ()


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
    from vera import provider_exhaustion as _px
    g = _px.GRACE.get()
    if g is not None and g.get("client") is not None:  # OD-1031-EXHAUST-GRACE: scoped completion client (not the latch)
        return g["client"]
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


@dataclass
class RequestBudget:
    """CMF-1022: ONE request-level deadline and inline-use flag shared by the whole provider chain (never restarted per
    provider). `deadline` is time.monotonic() at the /ask handler entry + public_mode.ASK_TIMEOUT_S (the client's wait).
    The server does NOT terminate a request at the deadline (client-side enforcement only; reported residual gap)."""
    deadline: float
    inline_used: bool = False


def request_budget() -> RequestBudget:
    if _stream_ctx.active():  # STREAM-1031 / OD-1031-NODEADLINE: the streamed path has no request deadline (no 125 s guard)
        return RequestBudget(float("inf"))
    from vera.public_mode import ASK_TIMEOUT_S
    return RequestBudget(time.monotonic() + ASK_TIMEOUT_S)


def _pre_llm(writer: bool = False) -> None:
    """STREAM-1031: before a writer / claim-check step. Raises StreamCancelled when the client is gone (no-op off the
    stream path). With writer=True also the provider-exhaustion guard (vera.provider_exhaustion, written concurrently:
    tolerated absent)."""
    _stream_ctx.check_cancelled()
    if writer:
        try:
            from vera import provider_exhaustion
        except ImportError:
            return
        provider_exhaustion.guard()


# The budget travels with the request in a context variable (set once by main.ask, around answer_via_chain), so the
# chain loop and every provider attempt see the SAME deadline and inline flag without changing any signature.
REQUEST_BUDGET: contextvars.ContextVar = contextvars.ContextVar("cmf_request_budget", default=None)
_QUESTION: contextvars.ContextVar = contextvars.ContextVar("port_p2_question", default=None)  # PORT-P2: query hash only
# DDFIX-1023/INTEG item 6: this request's inline-acquisition outcome (its R-ACQ-IRRELEVANT marks), reset per request
_REQUEST_PERSIST: contextvars.ContextVar = contextvars.ContextVar("midterms_wp1_persist", default=False)  # MIDTERMS-WP1-1106
_INLINE_OUT: contextvars.ContextVar = contextvars.ContextVar("ddfix_inline_outcome", default=None)
# MIDTERMS-WP10-1106: the answer-time evidence bundle (ephemeral corpus admission + lane pages) of the inline attempt in
# progress, for the examined-source ledger (vera.evidence_ledger). Set only by _fulldoc's acquired-only branch.
_LEDGER_EV: contextvars.ContextVar = contextvars.ContextVar("midterms_wp10_ledger_ev", default=None)
# NONASK-1103 E4m: the public evidence trace of the latest full-document attempt that produced no answer (response-only,
# never logged or stored). _fulldoc sets it; the no-answer contracts merge it. Reset at the start of every attempt.
_FD_TRACE: contextvars.ContextVar = contextvars.ContextVar("nonask_fd_trace", default=None)
# SELFQ-RCA-1107 D2: True when that trace came from an attempt on documents fetched for THIS request (the inline round's
# ephemeral corpus or lane pages), whose passages are not the response's admitted_evidence (the held attempt's).
_FD_TRACE_FETCHED: contextvars.ContextVar = contextvars.ContextVar("selfq_fd_trace_fetched", default=False)
_INLINE_COUNTS: contextvars.ContextVar = contextvars.ContextVar("vera_inline_counts", default=None)  # NONASK-1103 A6-4: {stored, fetched} of the inline attempt in progress
# MULTIQ-1108: set only while answer_in_scope answers ONE question of a multi-question prompt; holds the whole prompt's
# scope decision (routed once, never per part) with its classifier cost zeroed (counted once on the combined result).
_MQ_PART: contextvars.ContextVar = contextvars.ContextVar("multiq_part_decision", default=None)
_MQ_ROUNDS: contextvars.ContextVar = contextvars.ContextVar("multiq_part_rounds", default=None)  # MULTIQ-1109 F3: max acquisition rounds of this question (no-deadline stream path); None = unbounded
_MQ_CUT: contextvars.ContextVar = contextvars.ContextVar("multiq_part_cut", default=None)  # MULTIQ-1109 F3: [bool] set when that bound ended this question's search early
_MQ_DOCS: contextvars.ContextVar = contextvars.ContextVar("multiq_part_docs_share", default=None)  # MULTIQ-1109: this question's share of max_docs_per_prompt


def _fd_trace_contract() -> dict:
    t = _FD_TRACE.get()
    if not t:
        return {}
    return {"trace": t, "trace_attempt": "fetched" if _FD_TRACE_FETCHED.get() else "stored"}  # SELFQ-RCA-1107 D2


# Post-inline LLM calls that can follow an inline search, per the auto_acquire.json _calibration_note: one answer writer,
# one claim check, and a second claim-check call (the note reserves room for it). Each is bounded by the chain timeout_s.
POST_INLINE_LLM_CALLS = 3
# The reading-list fallback after a failed retry runs the grounding search twice at most (rl3 AND, then OR), each bounded
# by grounding.deadline_s (calibration note: "<= 2 x 10 s").
READING_LIST_SEARCHES = 2


def inline_need_s(acq: dict, grounding) -> float | None:
    """Seconds one inline retry may need: inline deadline_s (+ the arXiv cold-tdmrep window, arxiv_fulltext.
    deadline_extension, CONTROL-GAPS-1107) + POST_INLINE_LLM_CALLS x chain timeout_s + the reading-list
    tail (READING_LIST_SEARCHES x grounding deadline_s; 0 when ungrounded). Every term is an existing config value.
    None when any term cannot be read (the caller then skips the retry: fail closed)."""
    ct = _chain_timeout_s()
    from vera import inline_acquire
    dl = inline_acquire._b(acq, "deadline_s", 45)  # the value inline_acquire itself applies (its own default 45 when absent)
    if ct is None or isinstance(dl, bool) or not isinstance(dl, (int, float)):
        return None
    try:  # CONTROL-GAPS-1107 follow-up: the arXiv cold tdmrep.json window (<= one Crawl-delay) extends the inline pass
        from vera import arxiv_fulltext
        ext = float(arxiv_fulltext.deadline_extension(acq))
    except Exception:  # noqa: BLE001  unreadable: skip the retry (fail closed, like every other term)
        return None
    tail = 0.0
    if grounding is not None:
        from vera.grounded_ask import load_grounding  # the shipped grounding block (config/ask-provider-chain.json)
        gd = (load_grounding() or {}).get("deadline_s")
        if isinstance(gd, bool) or not isinstance(gd, (int, float)):
            return None
        tail = READING_LIST_SEARCHES * float(gd)
    return float(dl) + ext + POST_INLINE_LLM_CALLS * ct + tail


def answer_in_scope(question: str, model: str, pricing: PricingRecord, client: OpenAI | None = None,
                    extra_body: dict | None = None, grounding=None, budget: RequestBudget | None = None) -> AskResult:
    """The /ask entry point (item #72 W1): the scope router runs first; a declined question never reaches the
    answering model and gets the fixed decline template. answer_question stays the bare model call (the trace
    harness's baseline shape)."""
    from vera.scope_router import decline_text, route

    from vera.scope_router import default_chat, nullify_mode_claims

    original = question  # REL-FIX3-1028 D1: injection and markers are judged on the un-stripped text (route, below)
    from vera.scope_router import strip_instructions
    question, _removed = strip_instructions(question)  # INJ-SEP-1028 (owner 2026-10-28, OD-1028(21) c3): a reliably
    if _removed:  # separated injection sentence is discarded; the rest goes through every gate. Category-only log
        from vera.scope_router import INSTRUCTION_REMOVED_TOKEN
        logging.getLogger("vera").info("scope %s rid=%s", INSTRUCTION_REMOVED_TOKEN, _rid())
    from vera.request_intent import split_request  # MIDTERMS-WP1-1106 RC1: a trailing request is not research content
    question, _intent = split_request(question)  # cleaned text flows on; `original` (un-split) is judged by route only
    _REQUEST_PERSIST.set(bool(_intent.get("persist")))  # WP8 reads this later; acquisition behaviour is unchanged
    if _intent.get("persist"):
        _rt_event("scope", "request_intent_split", counts={"persist": 1})
    question, _claims = nullify_mode_claims(question)  # SEC-V3-1028 / owner 2026-10-28: a mode-switch claim is
    if _claims:  # stripped (never obeyed) and the rest is routed, answered and searched as usual; token-only log
        from vera.scope_router import MODE_CLAIM_TOKEN
        logging.getLogger("vera").info("scope %s rid=%s", MODE_CLAIM_TOKEN, _rid())

    def chat(messages, model_, l2):
        return default_chat(messages, model_, l2, client=client, extra_body=extra_body)

    _INLINE_OUT.set(None)
    _QUESTION.set(question)  # PORT-P2: the discovery recorder hashes it (text never stored)
    from vera import scope_router as _sr_self
    _pd = _MQ_PART.get()  # MULTIQ-1108: the whole prompt's decision when this call answers one of its questions
    if _pd is None and not _sr_self.injection_unstripped(original):  # SELFQ-RELEVANCE-1108: before routing, help scoring or any search
        _pre = _clarify_or_help(question)
        if _pre is not None:
            return _pre
    budget = budget or REQUEST_BUDGET.get() or request_budget()  # CMF-1022: one deadline per request (main.py passes the handler-entry one)
    if _stream_ctx.active() and budget.deadline != float("inf"):
        budget.deadline = float("inf")  # STREAM-1031: whichever budget was passed, the streamed path has no deadline
    if _pd is not None:  # MULTIQ-1108: the whole prompt was routed once (a part is never re-classified on its own)
        decision = _pd
    else:
        decision = route(original, model, pricing, chat=chat if (client or extra_body) else None)  # D1: route nullifies
        decision = _catstep(decision, question, model, pricing, chat if (client or extra_body) else None)  # CATSTEP-1028
    _mq = _multiq(question, original) if _pd is None else None  # MULTIQ-1108: off/shadow -> None (trace counts only)
    from vera import provider_exhaustion as _pxw
    _pxw.note_work(decision=decision)  # OD-1031-EXHAUST-GRACE: work has started; the decision is what a completion reuses
    acq = _acq_cfg_if_on()  # WED-ACQ2: None unless automatic acquisition is effectively on (byte-identical when off)
    acq = _mq_share_acq(acq)  # MULTIQ-1109: a question of a multi-question prompt gets its share of the per-prompt ceiling
    _log_basis(decision)  # SCOPE-GATE-1104 S2: basis=safe|none (class tokens only, never reason text)
    chained = {k: v for k, v in (("client", client), ("extra_body", extra_body)) if v}  # only from the #77 chain
    if not decision.accepted:
        # SCOPE-GATE-1104 S2 (owner amendment 1, 2026-11-04): only a safety refusal (advice, harmful, rule_change), an
        # unsupported-use request, a layer-1 deny/injection/marker hit or a fail-closed verdict (malformed after the
        # retry, classifier unavailable, layer2_mode none, a reply naming no category: D2a) reaches this branch. Subject
        # matter never does: every question that passes safety takes the normal path below, and whether the corpus
        # can answer is decided after retrieval. The decline branch itself is unchanged (decline_text, or the fixed
        # unsupported-use sentence plus the scholarly reading list); it never acquires.
        from vera import scope_router as _sru
        ia: dict = {"status": "off"} if acq is None else {}  # P0-B1b: inline outcome for a declined non-answer (Report)
        iac = {"inline_acquisition": ia} if ia else {}
        _INLINE_OUT.set(ia)
        off = _offtopic_reading_list(decision, grounding) if _sru.unsupported_use(decision) else None
        if off is not None:
            if acq is None:  # PATHA-1028 CT-2: say plainly that automatic search is off (neutral, no topic text)
                note = _sru.acquisition_off_note()
                # non-answer notices ARE the answer (ANS-LAYERS-1101 B): kept in `answer` whatever the flag
                off = replace(off, answer=f"{off.answer} {note}".strip() if note else off.answer)
            _rt_event("outcome", "unsupported_use_reading_list")
            return replace(off, contract={**(off.contract or {}), **iac}) if iac else off
        _rt_event("outcome", "decline", detail={"category": _rt_token(decision.category)})
        return AskResult(answer=decline_text(), tokens_used=decision.tokens_used, cost_usd=decision.cost_usd,
                         result_kind="decline")
    if _mq is not None:  # MULTIQ-1108 (mode on): each question routed and answered, or refused, on its own
        return _answer_parts(_mq, question, model, pricing, client, extra_body, grounding, budget, decision, acq)
    if not _sr_self.injection_unstripped(original):  # SELFQ-RELEVANCE-1108: a routed, accepted question about VERA-DAR
        _h = _help_answer(question, covered_only=False, tokens=decision.tokens_used, cost=decision.cost_usd)
        if _h is not None:
            return _h
    from vera import answer_contract as contract

    used_in: dict = {}
    fd = _fulldoc(question, model, pricing, decision, chained, grounding, used=used_in)  # D-055; None = not configured
    if isinstance(fd, AskResult):  # grounded in admitted passages of inspected, verified full documents
        fcfg = _fresh_cfg(acq)  # FRESH-1023: a retrieved, grounded answer is not by itself a reason to stop researching
        if fcfg is None or fd.evidence is None:
            return fd
        return _freshness_research(question, model, pricing, decision, chained, grounding, acq, fcfg, fd, used_in,
                                   budget)
    not_configured = fd is None  # the full-document path is not configured (or its base corpus is not loaded)
    if not_configured:
        ev, token, tokens, cost, text = None, None, decision.tokens_used, decision.cost_usd, _NOT_CONFIGURED_TEXT
        _rt_event("outcome", "fulldoc_not_configured")
    else:
        ev, token, tokens, cost, text = fd  # fail closed: no generated answer, never an abstract fallback
        _rt_event("outcome", "fulldoc_fallback", counts={"tokens": int(tokens or 0)}, detail={"fallback": _rt_token(token)})
    inline_note = None
    ia: dict = {}  # P0-B1: the inline outcome for a non-answer, so the Report can say whether a search was made
    if acq is None:
        ia = {"status": "off"}  # automatic acquisition effectively off (flag, kill switch or DB env): not attempted
    elif _layer1_clean(question):  # DAR-INLINE-1021-R2 must-have 1: nothing answered from the held corpus
        res_inl = _acquire_rounds(question, model, pricing, decision, chained, grounding, acq, budget, used_in, token,
                                  not_configured)
        if isinstance(res_inl, AskResult):
            return _research_on_insufficient(res_inl, ev, acq, (res_inl.contract or {}).get("inline_acquisition"),
                                             answered=True,
                                             held_ev=used_in.get("ev_acq"))
        inline_note, ia = res_inl
    _INLINE_OUT.set(ia)  # DDFIX-1023/INTEG item 6: the reading list drops records this attempt marked irrelevant
    res = _reading_list(question, model, pricing, decision, chained, grounding, contract, tokens=tokens, cost=cost)
    if not_configured and inline_note is None:  # unchanged pre-S2 result when nothing was searched inline
        # the corpus was not consulted, so the acquisition-off note ("looked only in its documents") is not added
        out = _with_cause(res, "reading_list", _NOT_CONFIGURED_TEXT)
        _rt_event("outcome", "reading_list", detail={"branch": "fulldoc_not_configured"})
        return out
    trig = {"acquire_trigger": "insufficient"} if acq is not None and _acq_on_insufficient(acq) and inline_note is None else {}
    text = _with_fetched(text, ev, ia, token, used_in)  # SCOPE-COUNT-1106; SELFQ-RCA-1107: same headline as the UI
    _a, _n = _nonanswer_note(text, res.notes, inline_note, strip=False)
    out = replace(res, answer=_a, notes=_n, evidence=ev,
                  contract={**(res.contract or {"layers": contract.layers()}), "fallback": token,
                            **_inline_trace(inline_note), **({"inline_acquisition": ia} if ia else {}),
                            **_fd_trace_contract(),
                            **({"search_counts": _sc} if (_sc := _search_counts(used_in)) is not None else {})}, **trig,
                  **_spent(inline_note, res.tokens_used, res.cost_usd, decision.tokens_used, decision.cost_usd))
    out = _research_on_insufficient(out, ev, acq, ia or None, answered=False, extra_ev=used_in.get("ev_acq"))
    cause = "reading_list" if not_configured else _CAUSE_CODE.get(token, "no_supported_answer")
    _rt_event("outcome", "no_answer", detail={"cause": _rt_token(cause), "fallback": _rt_token(token),
                                              "inline_status": _rt_token((ia or {}).get("status"))})
    return _with_off_note(_with_cause(out, cause, text), acq)


_NOT_CONFIGURED_TEXT = ("VERA-DAR could not write an answer to this question. The sources below may still help; read them "
                        "yourself.")



def _clarify_or_help(question: str):
    """SELFQ-RELEVANCE-1108 (owner 2026-11-08: no hard-coded self-question routing), BEFORE routing. Only for a question
    layer 1 does not decline and that holds no classifier marker: (1) a content word known to neither the stored
    documents, the help nor the lay-term vocabulary pauses the request with a clarification round (vera.spell_clarify);
    (2) a question whose every content word the published help covers, and which is about VERA-DAR by the shared
    relevance code (vera.help_evidence), is answered by quoting the help. A question with words the help does not cover
    goes to the scope classifier first (_help_after_route). No LLM, search, storage or memory. None = carry on."""
    from vera import scope_router as _sr, spell_clarify
    try:
        if _sr.has_marker(question) or _sr.layer1(question, _sr.load_config())[0] == _sr.DECLINE:
            return None
    except Exception:  # noqa: BLE001  fail closed: the normal route judges it
        return None
    from vera import answer_contract as contract
    c = spell_clarify.check(question)
    if c is not None:
        c = _multiq_spell(question, c)  # MULTIQ-1108 (mode on): one round covering every question's unknown word
        logging.getLogger("vera").info("scope basis=none layer=layer1 class=spell_clarify rid=%s", _rid())
        _rt_event("outcome", "clarification", counts={"suggestions": len(c["suggestions"])})
        return AskResult(answer=spell_clarify.reply_text(c), tokens_used=0, cost_usd=0.0, accepted=False,
                         result_kind="clarification", contract={"layers": contract.layers(), "fallback": None,
                                                                "clarification": c})
    return _help_answer(question, covered_only=True)


def _help_answer(question: str, covered_only: bool, tokens: int = 0, cost: float = 0.0):
    """The help reply when help is the best evidence (vera.help_evidence), else None."""
    from vera import help_evidence, spell_clarify
    ev = help_evidence.evaluate(question, corpus_best=_corpus_best_relevance,
                                known_to_corpus=spell_clarify.known_to_corpus, covered_only=covered_only)
    if ev is None:
        return None
    logging.getLogger("vera").info("scope basis=none layer=%s class=help_evidence rid=%s",
                                   "layer1" if covered_only else "routed", _rid())
    _rt_event("outcome", "self_description", counts={"passages": len(ev["passages"]), "uncovered": len(ev["uncovered"])})
    return AskResult(answer=help_evidence.reply_text(ev), tokens_used=tokens, cost_usd=cost, accepted=False,
                     result_kind="self_description")


def _corpus_best_relevance(question: str):
    """SELFQ-RELEVANCE-1108: the relevance of the best passage the stored collection (base, then stored acquired)
    admits for the question, by the same admission code as the normal path; 0.0 when none is admitted, None when
    the collection is off, still warming or unreadable. Local and deterministic (no LLM, search or storage)."""
    try:
        from vera import fulldoc_corpus
        from vera.fulldoc_admission import admit
        cfg = fulldoc_corpus.load_config()
        if not fulldoc_corpus.flag_enabled(cfg) or _base_warming():
            return None
        corpora = [c for c in (fulldoc_corpus.get_corpus(cfg), fulldoc_corpus.get_acquired_corpus(cfg)) if c is not None]
        if not corpora:
            return None
        best = 0.0
        for c in corpora:
            for a in getattr(admit(question, c, cfg), "admitted", ()) or ():
                best = max(best, float(a.get("relevance") or 0.0))
        return best
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------ MULTIQ-1108
def _multiq(question: str, original: str):
    """MULTIQ-1108: the split of a routed prompt into separate questions (vera.multi_question). off: nothing runs.
    shadow: counts and the shape token go to the request trace only; None (nothing user-visible changes). on: the
    Split when the prompt holds 2+ questions, else None. A prompt with an unstripped injection is never split (the
    normal route judges it whole). Never raises."""
    try:
        from vera import multi_question as mq, scope_router as _sr
        cfg = mq.load_config()
        mode = mq.mode_of(cfg)
        if mode == "off":
            return None
        sp = mq.detect(question, cfg)
        _rt_event("multiq", "split" if sp is not None else "single",
                  counts={"parts": len(sp.parts) if sp else 1, "over_cap": len(sp.over_cap) if sp else 0,
                          "pronoun_carry": sp.pronoun_carry if sp else 0},
                  detail={"shape": _rt_token(mq.shape(question, sp)), "mode": _rt_token(mode)})
        if mode != "on" or sp is None or _sr.injection_unstripped(original):
            return None
        return sp
    except Exception:  # noqa: BLE001
        return None


def _mq_rewrite(part, q: str) -> str:
    """A part's text after a clarification choice: the chosen question for the part, in the user's words (the carried
    subject words that were appended for answering are not shown back)."""
    q = str(q or "")
    return part.text + q[len(part.query):] if part.query != part.text and q.startswith(part.query) else q


def _mq_combined(question: str, split, items: list) -> dict:
    """ONE clarification block for the whole prompt from the per-question blocks `items` [(Part, block)], in the
    spell_clarify shape. Suggestion k rewrites every unclear question with its k-th option (its last when it has
    fewer), so one click answers the whole round. final=True: whatever the user picks is sent with the per-request
    spell check off, so neither a typo nor a sense round is asked again for this prompt (owner: at most one round)."""
    many = len(items) > 1
    lines, n = [], max(len(b.get("suggestions") or []) for _, b in items)
    for p, b in items:
        opts = ", ".join(str(s.get("word")) for s in b.get("suggestions") or [])
        lines.append(f"{f'({p.index}) ' if many else ''}{b.get('text', '')} {opts}?".strip())
    sugs = []
    for k in range(n):
        texts, words = [], []
        by_idx = {p.index: b for p, b in items}
        for p in split.parts:
            b = by_idx.get(p.index)
            opts = (b or {}).get("suggestions") or []
            if not opts:
                texts.append(p.text)
                continue
            o = opts[min(k, len(opts) - 1)]
            texts.append(_mq_rewrite(p, o.get("question")))
            words.append(str(o.get("word")))
        sugs.append({"word": " / ".join(words), "question": " ".join(texts + list(split.over_cap))})
    first = items[0][1]
    return {"word": first.get("word"), "text": " ".join(lines), "as_typed": question,
            "as_typed_label": first.get("as_typed_label") or "search as typed", "suggestions": sugs, "final": True,
            "questions": len(items)}


def _multiq_spell(question: str, c: dict) -> dict:
    """MULTIQ-1108 (mode on): when the prompt holds several questions, the typo round covers every question's first
    unknown word at once and is final. Otherwise (or on any failure) the unchanged block."""
    try:
        from vera import multi_question as mq, spell_clarify
        cfg = mq.load_config()
        if mq.mode_of(cfg) != "on":
            return c
        sp = mq.detect(question, cfg)
        if sp is None:
            return c
        items = [(p, b) for p in sp.parts if (b := spell_clarify.check(p.text)) is not None]
        return _mq_combined(question, sp, items) if items else dict(c, final=True)
    except Exception:  # noqa: BLE001
        return c


def _mq_share_acq(acq):
    """MULTIQ-1109: while one question of a multi-question prompt is answered, bounds.max_docs_per_prompt (the per-prompt
    new-document ceiling, OD-1031-CEIL31) is replaced by this question's share, so the prompt as a whole stays within
    the ceiling. Outside a part (or with acquisition off) acq is returned unchanged. Never raises."""
    try:
        share = _MQ_DOCS.get()
        if share is None or not isinstance(acq, dict):
            return acq
        b = dict(acq.get("bounds") or {})
        cap = b.get("max_docs_per_prompt")
        if isinstance(cap, int) and not isinstance(cap, bool):
            b["max_docs_per_prompt"] = min(cap, int(share))
        out = dict(acq)
        out["bounds"] = b
        return out
    except Exception:  # noqa: BLE001
        return acq


def _mq_part_decline(text: str) -> bool:
    """Layer 1 (deterministic, free) on one question: True when it declines that question."""
    try:
        from vera import scope_router as _sr
        return _sr.layer1(text, _sr.load_config())[0] == _sr.DECLINE
    except Exception:  # noqa: BLE001  fail closed: the part is not answered
        return True


def _answer_parts(split, question, model, pricing, client, extra_body, grounding, budget, decision, acq) -> AskResult:
    """MULTIQ-1108 (mode on). The whole prompt was routed once (a safety decline already declined everything). Each
    question is then answered on its own, sequentially and cheapest first: questions the published help fully covers
    (no cost), then the others through the unchanged single-question path (help-as-evidence, the stored documents,
    inline acquisition within the remaining request budget, its own no-answer and reading list). Layer 1 also judges
    each question; a declined question gets the decline line for that question only. A clarification from any
    question (a sense round) becomes ONE combined round for the prompt."""
    from vera.scope_router import decline_text
    n = len(split.parts)
    part_dec = replace(decision, tokens_used=0, cost_usd=0.0, llm_calls=0)
    results: dict = {}
    from vera import multi_question as _mq
    _mcfg = _mq.load_config()
    rc = _mcfg.get("stream_rounds_per_question")
    rounds_cap = rc if isinstance(rc, int) and not isinstance(rc, bool) and rc >= 1 else 2
    cut_text = str(_mcfg.get("cut_text") or "The search for new documents for question {n} was stopped early so that every question gets a fair share of the time.")
    ceiling = ((acq or {}).get("bounds") or {}).get("max_docs_per_prompt")
    share = ceiling // max(1, n) if isinstance(ceiling, int) and not isinstance(ceiling, bool) else 1
    try:  # F3: a question's share of the ceiling stays >= the round target (6): more questions than that are listed, not answered
        from vera import acq_rounds
        tgt = int(acq_rounds.target_of(acq)) if acq else 0
        if tgt > 0 and isinstance(ceiling, int) and not isinstance(ceiling, bool) and share < tgt:
            keep = max(1, ceiling // tgt)
            if keep < n:
                split = replace(split, parts=split.parts[:keep], over_cap=tuple(p.text for p in split.parts[keep:]) + tuple(split.over_cap))
                n = keep
                share = ceiling // n
    except Exception:  # noqa: BLE001
        pass
    for p in split.parts:  # help first: zero cost, never lost to a late deadline
        if not _mq_part_decline(p.query):
            h = _help_answer(p.query, covered_only=True)
            if h is not None:
                results[p.index] = h
    for p in split.parts:
        if p.index in results:
            continue
        _rt_event("multiq", "part_start", counts={"part": p.index, "parts": n})
        if _mq_part_decline(p.query):
            results[p.index] = AskResult(answer=decline_text(), tokens_used=0, cost_usd=0.0, result_kind="decline")
            continue
        # MULTIQ-1109 (reconciles SIX-1109): every question is its own research unit: its own new-document search
        # (budget.inline_used is per question, so one job per question against jobs_per_hour/jobs_per_day), under the
        # ONE shared request deadline, and an equal share of the per-prompt ceiling (max_docs_per_prompt // n).
        budget.inline_used = False
        tok = _MQ_PART.set(part_dec)
        tok2 = _MQ_DOCS.set(max(1, share))
        cut: list = []
        tok3 = _MQ_CUT.set(cut)
        tok4 = _MQ_ROUNDS.set(rounds_cap if budget.deadline == float("inf") else None)
        try:
            results[p.index] = answer_in_scope(p.query, model, pricing, client=client, extra_body=extra_body,
                                               grounding=grounding, budget=budget)
        finally:
            _MQ_PART.reset(tok)
            _MQ_DOCS.reset(tok2)
            _MQ_CUT.reset(tok3)
            _MQ_ROUNDS.reset(tok4)
        if cut and results[p.index].result_kind == "answer":  # F3: say so when this question's search was cut short
            results[p.index] = replace(results[p.index], answer=f"{results[p.index].answer}\n\n{cut_text.replace('{n}', str(p.index))}".strip())
    for p in split.parts:
        _rt_event("multiq", "part_done", counts={"part": p.index},
                  detail={"result_kind": _rt_token(results[p.index].result_kind)})
    tokens = int(decision.tokens_used or 0) + sum(int(r.tokens_used or 0) for r in results.values())
    cost = round(float(decision.cost_usd or 0.0) + sum(float(r.cost_usd or 0.0) for r in results.values()), 6)
    clar = [(p, (results[p.index].contract or {}).get("clarification")) for p in split.parts
            if results[p.index].result_kind == "clarification"]
    clar = [(p, b) for p, b in clar if isinstance(b, dict) and b.get("suggestions")]
    if clar:
        from vera import answer_contract as contract, spell_clarify
        c = _mq_combined(question, split, clar)
        _rt_event("outcome", "clarification", counts={"suggestions": len(c["suggestions"]), "questions": len(clar)})
        return AskResult(answer=spell_clarify.reply_text(c), tokens_used=tokens, cost_usd=cost, accepted=False,
                         result_kind="clarification", contract={"layers": contract.layers(), "fallback": None,
                                                                "clarification": c})
    return _mq_compose(split, results, decision, acq, tokens, cost)


def _mq_public_sources(r) -> list:
    return [{k: s.get(k) for k in ("n", "title", "url", "published")} for s in r.sources or ()
            if isinstance(s, dict)] if r.result_kind == "answer" else []


def _mq_compose(split, results: dict, decision, acq, tokens: int, cost: float) -> AskResult:
    """One page for the prompt: a section per question (the user's own words as its heading), each with its own
    answer or honest no-answer, sources and counts; notes shared by every question (the acquisition-off note, limitation
    notes) once; questions beyond the cap listed once to ask separately. No sentence appears twice: a question whose
    whole text an earlier one already said gets the same-reason line instead. Fail closed for older clients: the
    top-level kind is "answer" only when every question was answered; no top-level sources, evidence or claims (so
    nothing is written to memory from a combined result)."""
    from vera import answer_contract as contract, multi_question as mq, page_text
    cfg = mq.load_config()
    off = ""
    if acq is None:
        try:
            from vera import scope_router as _sr
            off = _sr.acquisition_off_note() or ""
        except Exception:  # noqa: BLE001
            off = ""
    said: dict = {}  # sentence key -> question index that said it first
    sections, parts_pub, notes, seen_notes = [], [], [], set()
    for p in split.parts:
        r = results[p.index]
        text = (r.answer or "").replace(off, "").strip() if off else (r.answer or "").strip()
        sents = page_text.sentences(text)
        keys = [page_text._key(s) for s in sents]
        fresh = [s for s, k in zip(sents, keys) if k not in said]
        if sents and not fresh:
            if r.result_kind != "answer":  # MULTIQ-1109 (verifier): an answered question is never relabelled "not answered"
                text = str(cfg.get("same_reason_text") or "").replace("{n}", str(said[keys[0]])) or text
        elif len(fresh) < len(sents):
            text = " ".join(fresh)
        for k in keys:
            said.setdefault(k, p.index)
        c = r.contract or {}
        parts_pub.append({"index": p.index, "question": p.text, "result_kind": r.result_kind,
                          "outcome": contract.outcome_for(r.result_kind, c.get("fallback"), len(r.candidates or ())),
                          "answer": text, "fallback": c.get("fallback"), "sources": _mq_public_sources(r),
                          "candidates": [] if r.result_kind == "answer" else list(r.candidates or ()),
                          **{k: c[k] for k in ("inline_acquisition", "search_counts") if c.get(k) is not None}})
        sections.append(f"**{p.index}. {p.text}**\n\n{text}")
        for note in r.notes or ():
            key = (note.get("kind"), note.get("text")) if isinstance(note, dict) else (None, str(note))
            if key not in seen_notes:
                seen_notes.add(key)
                notes.append(note)
    if split.over_cap:
        sections.append(str(cfg.get("over_cap_text") or "") + "\n" + "\n".join(f"- {q}" for q in split.over_cap))
    if off:
        sections.append(off)
    kinds = [results[p.index].result_kind for p in split.parts]
    top = "answer" if all(k == "answer" for k in kinds) else next(k for k in kinds if k != "answer")
    fallback = next((results[p.index].contract or {}).get("fallback") for p in split.parts
                    if results[p.index].result_kind == top) if top != "answer" else None
    _rt_event("outcome", "multi_question", counts={"parts": len(split.parts), "over_cap": len(split.over_cap),
                                                   "answered": sum(1 for k in kinds if k == "answer")})
    return AskResult(answer="\n\n".join(sections), tokens_used=tokens, cost_usd=cost, accepted=decision.accepted,
                     result_kind=top, notes=tuple(notes),
                     contract={"layers": contract.layers(), "fallback": fallback, "parts": parts_pub,
                               "over_cap": list(split.over_cap), "shared_notes": [off] if off else [],
                               "multi_question": {"parts": len(split.parts), "over_cap": len(split.over_cap)}})


def _with_off_note(res, acq):
    """SCOPE-GATE-1104 S2-T2: with automatic acquisition effectively off, a non-answer is the corpus attempt plus the
    normal reading list AND the fixed acquisition_off_note (never decline_template). Unchanged when acquisition is on,
    for an answer, or when the note is not configured. Non-answer notices ARE the answer (ANS-LAYERS-1101 B)."""
    if acq is not None or getattr(res, "result_kind", "answer") == "answer":
        return res
    try:
        from vera import scope_router as _sr
        note = _sr.acquisition_off_note()
    except Exception:  # noqa: BLE001
        note = ""
    if not note or note in (res.answer or ""):
        return res
    return replace(res, answer=f"{res.answer} {note}".strip())


def _retry_tokens(question, token, not_configured: bool) -> bool:
    """Whether a non-answer of the held corpus starts inline acquisition. Owner rules: an insufficient corpus goes to
    acquisition and the other lanes (project_ingest_six_then_answer; SCOPE-GATE-1104 amendment 1). Every evidence-miss
    token does (insufficient, corpus_unusable, nonresponsive (INT-1022-B), no_valid_citations, claim_check_insufficient,
    model_insufficient, premise_unverified (OD-1104-TRACE)); so does a full-document path that is not configured or
    whose base corpus is not loaded (fd None; _inline itself still never acquires when the flag is off, VER L6).
    claim_check_disabled and checker_unavailable never do (OD-CMF-1022g: the checker, not the evidence, failed).
    NEWS-LANE-1104 W1: with the reporting lane on, a news-intent question retries on any other non-answer token."""
    from vera import fulldoc_corpus
    from vera.fulldoc_answer import PREMISE_FALLBACK
    ft = fulldoc_corpus.load_config()["fallback_tokens"]
    excluded = tuple(t for t in (ft.get("claim_check_disabled"), ft.get("checker_unavailable"),
                                 "claim_check_disabled", "checker_unavailable") if t)
    if not_configured:
        return True
    if not token or token in excluded:
        return False
    retry = (ft["insufficient"], ft["corpus_unusable"], ft.get("nonresponsive"), ft.get("no_valid_citations"),
             ft.get("claim_check_insufficient"), ft.get("model_insufficient"), PREMISE_FALLBACK)
    if token in retry:
        return True
    try:
        from vera import news_intent, reporting_lane  # NEWS-LANE-1104 W1
        return bool(reporting_lane.lane_enabled() and news_intent.news_intent(question))
    except Exception:  # noqa: BLE001
        return False


def _round_acq(acq: dict, seconds: float) -> dict:
    """A copy of the acquisition config whose inline deadline_s is the time this round may use (never more than the
    configured deadline_s). The request budget (RequestBudget.deadline) is the only bound applied; nothing new."""
    inl = dict(acq.get("inline") or {})
    from vera import inline_acquire
    inl["deadline_s"] = min(float(inline_acquire._b(acq, "deadline_s", 45)), float(seconds))
    return {**acq, "inline": inl}


def persist_acq(acq: dict, persist: bool) -> dict:
    """RETRIEVAL-1109 H-4 (owner OD-1109-RETRIEVAL; switch config/auto_acquire.json inline.persist_extends_deadline,
    exactly true = on): a keep-searching request's inline deadline_s becomes inline.persist_deadline_max_s (never
    lower than the configured deadline_s). Still bounded by the request budget (_acquire_rounds: the remaining request
    time minus the post-inline tail), the job caps, the finite variant list and ~31 documents. Off, not persist, or an
    unreadable value: `acq` unchanged (the same object). The streamed path has no inline deadline, so it is unaffected."""
    inl = (acq or {}).get("inline") or {}
    if not persist or inl.get("persist_extends_deadline") is not True:
        return acq
    mx = inl.get("persist_deadline_max_s")
    if isinstance(mx, bool) or not isinstance(mx, (int, float)) or not mx > 0:
        return acq
    from vera import inline_acquire
    dl = float(inline_acquire._b(acq, "deadline_s", 45))
    return {**acq, "inline": {**inl, "deadline_s": max(dl, float(mx))}}


def _acquire_rounds(question, model, pricing, decision, chained, grounding, acq, budget, used_in, token,
                    not_configured):
    """INTEGRATE-1104 step 2 (owner: ingest >= 6 full texts, then answer; acquisition until the target is met or the
    request time budget runs out). Repeated inline acquisition rounds after a held-corpus non-answer. Returns an
    AskResult (an answer), else (inline_note, inline outcome dict).

    Stops when: an answer is produced; the round target (acq_rounds.target_of, 6) of answer-eligible documents is met;
    the acquisition reports anything other than a deadline stop (met, resource_limited: its own variant rounds ran out
    of routes; off, busy, error, capped); a round adds no new document (no progress, so another identical round cannot
    help); or the request budget cannot hold another round plus the post-inline tail (inline_need_s minus the inline
    deadline: writer, claim checks, reading-list fallback). Each round's inline deadline is the configured deadline_s
    or, when less time is left, the remaining time minus that tail. No new cap: the request deadline, the configured
    deadline_s, the job caps inside inline_acquire and the finite candidate/variant lists bound the loop.
    CMF-1022 CR-3 is kept across the provider chain: budget.inline_used marks that this request's acquisition ran,
    so a chained provider retry never starts a second series.
    MIDTERMS-WP8-1106: a keep-searching request (_REQUEST_PERSIST) lifts only the inner no-progress heuristic
    (inline_acquire._extra_rounds); this loop's conditions are unchanged. MIDTERMS-WP10-1106: one examined-source
    ledger is kept across rounds (same dedup), and the round target is met on its counted sources."""
    from vera import acq_rounds, inline_acquire
    ia: dict = {}
    acq = persist_acq(acq, bool(_REQUEST_PERSIST.get()))  # RETRIEVAL-1109 H-4 (switch; bounded below by the budget)
    if inline_acquire.icfg_of(acq).get("in_scope") is not True or not _retry_tokens(question, token, not_configured):
        return None, ia
    if budget.inline_used:
        _rt_event("acquire", "budget_used")
        return None, ia
    target = acq_rounds.target_of(acq)
    from vera import query_norm  # A-P1-NORM-1107: shadow logs only; 'on' adds one vouched-terms retry round
    qn, q_cur, qn_retried = query_norm.start(question), question, False
    tried = set(_doc_keys(used_in.get("acq")))
    eligible = 0
    ledger = None  # WP10: examined-source ledger merged across rounds
    note = None
    spent_t, spent_c = 0, 0.0
    rounds = 0
    while True:
        need = inline_need_s(acq, grounding)
        remaining = budget.deadline - time.monotonic()
        if remaining == float("inf"):  # streamed path (OD-1031-NODEADLINE): no request deadline
            secs = float(inline_acquire._b(acq, "deadline_s", 45))
        elif need is None:  # fail closed: a budget term is unreadable, so the request deadline cannot be honoured
            if rounds == 0:
                ia = {"status": "budget_unknown"}
            break
        else:
            secs = remaining - (need - float(inline_acquire._b(acq, "deadline_s", 45)))
            if secs <= 0:
                if rounds == 0:
                    ia = {"status": "deadline"}
                break
        budget.inline_used = True
        info: dict = {}
        t0 = time.monotonic()
        inl = _inline(q_cur, model, pricing, decision, chained, grounding, _round_acq(acq, secs),
                      tried=set(tried), in_scope=True, info=info)
        rounds += 1
        out = info.get("outcome") or {}
        _rt_event("acquire", "round", counts={"round": rounds, "latency_ms": int((time.monotonic() - t0) * 1000)},
                  detail={"status": _rt_token(out.get("status"))})
        if isinstance(note, _Note):  # the previous round's writer/check spend is kept (DAR-TRACE-1022); its total
            # includes the classifier spend (base_tokens), which the final result already counts once
            spent_t += max(note.tokens - int(decision.tokens_used or 0), 0)
            spent_c += max(note.cost - float(decision.cost_usd or 0.0), 0.0)
            note = _InlineNotice(str(note))
        rl = info.get("_ledger")
        if rl is not None:
            ledger = rl if ledger is None else ledger.merge(rl)
        if isinstance(inl, AskResult):
            if ledger is not None and rounds > 1:
                c = dict(inl.contract or {})
                if isinstance(c.get("inline_acquisition"), dict):
                    c["inline_acquisition"] = _with_ledger(c["inline_acquisition"], ledger, acq)
                    inl = replace(inl, contract=c)
            if spent_t or spent_c:
                inl = replace(inl, tokens_used=int(inl.tokens_used or 0) + spent_t,
                              cost_usd=round(float(inl.cost_usd or 0.0) + spent_c, 6))
            return inl
        note, ia = inl, out
        new_keys = set(info.get("tried_keys") or ()) - tried
        tried |= new_keys
        if ledger is not None:
            eligible = ledger.counted()  # WP10: deduplicated across rounds; lane and web pages included
        else:
            eligible += int(out.get("admitted", 0) or 0) + int(out.get("reused", 0) or 0)
        stop = str(((out.get("acquisition") or {}).get("status")) or "")
        if qn is not None and not qn_retried and stop not in ("busy", "off", "capped", "error"):
            _q2, _code = query_norm.retry_for(question, qn, eligible, qn_retried)
            if _code != "none":
                _rt_event("query_norm", _code)
            if _q2 is not None:  # mode 'on' only; the same time check at the top of the loop applies
                q_cur, qn_retried = _q2, True
                continue
        if inl is None or eligible >= target or stop != "incomplete_deadline" or not new_keys:
            break
        _mr = _MQ_ROUNDS.get()
        if _mr is not None and rounds >= _mr:  # MULTIQ-1109 F3: a fair share of rounds per question on the no-deadline stream
            _cut = _MQ_CUT.get()
            if _cut is not None:
                _cut.append(True)
            _rt_event("multiq", "rounds_cut", counts={"rounds": rounds})
            break
    if (spent_t or spent_c) and isinstance(note, str):
        t, c = (note.tokens, note.cost) if isinstance(note, _Note) else (int(decision.tokens_used or 0),
                                                                          float(decision.cost_usd or 0.0))
        merged = _Note(str(note), t + spent_t, c + spent_c)
        merged.token = getattr(note, "token", None)
        if getattr(note, "bg_note", None):
            merged.bg_note = note.bg_note
        note = merged
    if ia and rounds > 1:
        ia = {**ia, "rounds": rounds}
    if ia and ledger is not None and rounds > 1:
        ia = _with_ledger(ia, ledger, acq)
    if ia and _REQUEST_PERSIST.get():  # WP8: the contract carries the keep-searching flag (a boolean, never text)
        ia = {**ia, "persist": True}
    if not rounds and ia:
        logging.getLogger("vera").info("inline retry skipped: %s rid=%s", ia["status"], _rid())
        _rt_event("acquire", "retry_skipped", detail={"reason": _rt_token(ia["status"])})
    return note, ia


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


def _nonanswer_note(answer: str, notes: tuple, note, strip: bool = True) -> tuple:
    """(answer, notes) for a non-answer notice. The notice IS the user's answer, so it stays in `answer` whatever the
    flag (legacy strings exactly, including trailing spaces); only a separate background note is filed as notes."""
    n = str(note or "")
    if strip:
        a = f"{answer} {n}".strip() if n else answer
    else:
        a = f"{answer} {n}" if n else answer
    from vera import answer_contract as _ac
    return _ac.add_note(a, notes, "background", getattr(note, "bg_note", "") or "", "suffix")


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
    from vera import fulldoc_corpus, inline_acquire

    if not fulldoc_corpus.flag_enabled(fulldoc_corpus.load_config()):  # VER L6: no answer path -> never acquire inline
        _rt_event("acquire", "off", detail={"reason": "fulldoc_flag_off"})
        if info is not None:
            info["outcome"] = {"status": "off"}
        return None
    from vera import acq_discovered
    kw = {}  # P0-5 (inline.use_discovered, default off): the request's own cached discovery results; off = unchanged call
    if acq_discovered.use_discovered(acq) and inline_acquire.inline_enabled(acq):  # E5: no discovery search when acquire() is off; the off-topic path lists the restricted search, in-scope the full one
        kw["discovered"] = acq_discovered.discovery_results(grounding, restricted=False)  # S2: never topic-restricted
    if getattr(decision, "topic_key", ""):  # ROUNDS-1028: catstep's key, used only to label the acquisition round
        kw["topic_key"] = decision.topic_key
    # PERF-MISS-1105-FIX2 F2: the lane pass needs only the question, so it starts NOW and runs concurrently with the
    # acquisition rounds (it used to start only after acquire() returned); joined below, with a visible activity.
    persist = bool(_REQUEST_PERSIST.get())  # MIDTERMS-WP8-1106: keep-searching overrides the no-progress heuristic
    if persist:
        kw["persist"] = True  # absent otherwise, so the unchanged call keeps its signature
    lane_task = inline_acquire.start_lane_pass(question, acq)
    o = inline_acquire.acquire(question, acq, served=tried, **kw)  # R2: only documents actually tried are skipped
    lanes = inline_acquire.with_web(o, lane_task.result(), question)  # NONASK-1103 (+A6-W web pages): request-local only
    from vera import provider_exhaustion as _pxw
    _pxw.note_work(ephemeral=o.corpus, inline=(o.public() if hasattr(o, "public") else None))  # OD-1031-EXHAUST-GRACE
    try:  # PERF-MISS-1104-TRACE: per inline attempt (counts, status, reason codes; never DOIs, URLs or titles)
        _rt_event("acquire", "outcome",
                  counts={"admitted": int(getattr(o, "admitted", 0) or 0), "stored": int(getattr(o, "stored", 0) or 0),
                          "reused": int(getattr(o, "reused", 0) or 0),
                          "candidates": len(getattr(o, "candidates", None) or []),
                          **({"persist": 1} if persist else {})},
                  detail={"status": _rt_token(getattr(o, "status", "")),
                          "acquisition": {k: v for k, v in (getattr(o, "acquisition", None) or {}).items()
                                          if k in ("status", "stored", "target", "rounds", "stop")},
                          "lanes": lanes is not None})
        for c in list(getattr(o, "candidates", None) or [])[:20]:
            if isinstance(c, dict):
                _rt_event("acquire", "candidate", detail={"provider": _rt_token(c.get("portal")),
                                                          "outcome": _rt_token(c.get("outcome")),
                                                          "reason": _rt_token(str(c.get("reason") or "").lower())})
    except Exception:  # noqa: BLE001
        pass
    if info is not None:
        info["tried_keys"] = _doc_keys(o.corpus)  # INTEGRATE-1104 step 2: the next round skips these documents
        st = str(getattr(o, "status", "error"))  # "off" carries no counts: identical whether the flag or the run said off
        info["outcome"] = o.public() if st != "off" and hasattr(o, "public") else {"status": st}
        if lanes is not None:
            info["outcome"] = dict(info["outcome"], lanes=lanes.public())  # links and counts only (need-to-know)
    from vera import answer_contract as contract
    bg = inline_acquire.background_note(acq, o)  # CONT-1023: "" unless candidates were queued for background acquisition
    if o.corpus is None and not (lanes is not None and lanes.pages):
        _ledger_out(o, None, lanes, acq, info)  # WP10: candidates, discovered records and links only
        t = _inline_notice(acq, o)
        if t is not None and bg:
            if contract.notes_separate():  # ANS-LAYERS-1101 A: the background note travels separately (kind "background")
                t = t if isinstance(t, _InlineNotice) else _InlineNotice(t)
                t.bg_note = bg
                return t
            return f"{t} {bg}".strip()
        return t
    from vera import fulldoc_corpus as _fdc
    _cfg_h = None
    try:
        _cfg_h = _fdc.load_config()
    except Exception:  # noqa: BLE001
        pass
    hon = _honest_counts_on(_cfg_h)  # A6-4 (D3): answer.inline_honest_counts (ships true)
    n_st, n_re, n_ad = (int(getattr(o, "stored", 0) or 0), int(getattr(o, "reused", 0) or 0), int(getattr(o, "admitted", 0) or 0))
    # NOANSWER-1109: "stored" = held in the collection BEFORE this request (reused); "fetched" = fresh documents read
    # for this request (o.admitted; disjoint from reused), whether or not they were then saved.
    counts = {"stored": n_re, "fetched": n_ad} if hon else None  # admitted = fresh documents only (disjoint from reused)
    tok_c = _INLINE_COUNTS.set(counts)
    tok_l = _LEDGER_EV.set(None)
    try:
        out = _fulldoc(question, model, pricing, decision, chained, grounding, acquired_only=True, ephemeral=o.corpus,
                       **({"lanes": lanes} if lanes is not None else {}))
        led_ev = _LEDGER_EV.get()
    finally:
        _INLINE_COUNTS.reset(tok_c)
        _LEDGER_EV.reset(tok_l)
    ledger = _ledger_out(o, led_ev, lanes, acq, info)  # WP10: before the page text is dropped
    led_ev = None
    lanes = None  # page text is dropped with the request (R-037)
    if not isinstance(out, AskResult):
        # post-writer fallback: keep the specific token (response telemetry + log); user text is the generic
        # "unanswered" notice (OD-1104-TRACE: premise_unverified is an evidence miss too, no premise refusal text).
        # INT-1022-B merge: the DAR-TRACE-1022 _Note (an _InlineNotice subclass) also carries the inline writer/check spend.
        tok = out[1] if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], str) else None
        logging.getLogger("vera").info("inline answer fallback token=%s rid=%s", tok, _rid())
        _rt_event("acquire", "answer_fallback", detail={"fallback": _rt_token(tok)})
        text = inline_acquire.notice(acq, "unanswered")
        if text:  # MIDTERMS-WP8-1106 review F2: the stop disclosure on this non-answer path too
            text = f"{text}{inline_acquire.stop_clause(getattr(o, 'acquisition', None))}"
        if hon and counts is not None and isinstance(out, tuple) and len(out) == 5 \
                and tok not in tuple((_cfg_h or {}).get("fallback_tokens", {}).get(x) for x in ("checker_unavailable", "nonresponsive")):
            text = " ".join(x for x in (text, _honest_text(_cfg_h, counts)) if x)  # A6-4: existing wording kept, split counts appended; never "tried"
        if text is None:
            return None
        note = (_Note(text, out[2], out[3]) if isinstance(out, tuple) and len(out) == 5  # DAR-TRACE-1022: spend kept
                else _InlineNotice(text))
        note.token = tok
        return note
    # R2 must-have 1 / SCOPE-GATE-1104 S2: the normal wording (there is no off-topic prefix: a question that passed
    # safety is never warned or downgraded for its subject); memory rules as any answer
    _rt_event("outcome", "inline_answer")
    ans, nts = out.answer, out.notes
    items = (("acquisition", inline_acquire.answer_note(acq, o)), ("background", bg))
    for kind, t in (items if contract.notes_separate() else reversed(items)):  # legacy text order ACQ, BG, answer
        ans, nts = contract.add_note(ans, nts, kind, t, "prefix")
    return replace(out, answer=ans, notes=nts,
                   contract={**(out.contract or {}), "acquired_only": True,
                             "inline_acquisition": _with_ledger({**o.public(), **({"counts": counts} if hon else {})},
                                                                ledger, acq)})


def _with_ledger(pub: dict, ledger, acq) -> dict:
    """MIDTERMS-WP10-1106 (owner D3): a COPY of a public inline outcome with the ledger summary (counts only) and
    acquisition.counted; status becomes "met" when the counted examined sources reach the target, so the status and the
    "Read N of the T sources targeted" line agree. The (possibly cached) Outcome is never mutated."""
    if ledger is None or not isinstance(pub, dict):
        return pub
    summ = ledger.summary()
    out = {**pub, "ledger": summ}
    aq = pub.get("acquisition")
    if isinstance(aq, dict):
        aq = {**aq, "counted": summ["counted"]}
        t = aq.get("target")
        if isinstance(t, int) and not isinstance(t, bool) and summ["counted"] >= t > 0:
            aq["status"] = "met"
        out["acquisition"] = aq
    return out


def _ledger_out(o, ev, lanes, acq, info):
    """WP10: build this attempt's examined-source ledger, put its summary on info["outcome"] (and the object on
    info["_ledger"] for _acquire_rounds), and trace it (counts only). Never raises; None on failure."""
    try:
        from vera import evidence_ledger
        ledger = evidence_ledger.from_inline(o, ev=ev, lanes=lanes, acq=acq)
    except Exception as exc:  # noqa: BLE001  the ledger never breaks the answer path
        logging.getLogger("vera").warning("evidence ledger unavailable (%s)", type(exc).__name__)
        return None
    if info is not None:
        info["_ledger"] = ledger
        if isinstance(info.get("outcome"), dict) and info["outcome"].get("status") != "off":
            info["outcome"] = _with_ledger(info["outcome"], ledger, acq)
    try:
        _rt_event("acquire", "ledger", counts=ledger.summary())
    except Exception:  # noqa: BLE001
        pass
    try:  # the streamed progress bar "Read n of 6" (counts only; an existing lifecycle state, no new contract state)
        aq = (getattr(o, "acquisition", None) or {})
        if isinstance(aq.get("target"), int) and _stream_ctx.active():
            from vera import progress
            progress.emit("acquire", "update", counted=int(ledger.counted()), target=int(aq["target"]))
    except Exception:  # noqa: BLE001
        pass
    return ledger


_PREFILTER_LICENCE = ("ineligible_not_open_access", "ineligible_licence_prefilter")
_UNCHECKED_REASONS = ("ranked_below_cap",)  # discovered_not_selected cut by the candidate cap (R-ACQ-DEFER_CAP)


def _notice_class(r) -> str:
    """DDFIX-1023/INTEG item 9: what happened to one candidate, for the notice. checked = retrieved/read and rejected;
    licence = dropped before retrieval by the licence/open-access prefilter (judged on its listed licence);
    irrelevant = judged not relevant before any fetch (R-ACQ-IRRELEVANT pre-fetch drop; a full-text relevance rejection
    is "checked"); unchecked = only cap, deadline, quota
    or rate skips (never judged); other = excluded for another recorded reason; unknown = no recorded status."""
    from vera.research_outcome import DIAG_STATUSES
    if not isinstance(r, dict) or r.get("status") not in DIAG_STATUSES:
        return "unknown"
    st, why = r["status"], r.get("status_reason")
    code = str(r.get("reason") or "").strip().upper()
    if st == "admitted":
        return "admitted"
    if st == "discovered_not_selected":
        if code == "R-ACQ-IRRELEVANT":  # judged not relevant on its listed title/abstract before any fetch
            return "irrelevant"
        if code in ("R-ACQ-NOTOA", "R-ACQ-NOLIC") or why in _PREFILTER_LICENCE:
            return "licence"
        if why in _UNCHECKED_REASONS or code == "R-ACQ-DEFER_CAP" or (why is None and not code):
            return "unchecked"  # cut by the cap (or not selected with no recorded reason): never judged
        return "other"
    if st == "skipped":
        return "unchecked"
    if st in ("fetched_rejected", "acquired_not_admitted"):
        return "checked"
    return "other"  # selected_not_fetched: no route, already held, crawl delay


# NOANSWER-PAGE-1108 D9: plain words per reason class of a candidate found but not retrieved (selected_not_fetched and
# the other "other" statuses); keys are research_outcome reason classes.
_OTHER_REASON_TEXT = {"already_in_corpus": "already held in VERA-DAR's collection",
                      "no_route": "with no identifier or retrieval route VERA-DAR can use",
                      "crawl_delay": "because the source's required pause between requests did not fit in the time left",
                      "ineligible_no_identifier": "with no identifier or retrieval route VERA-DAR can use",
                      "duplicate_or_in_corpus": "already held in VERA-DAR's collection",
                      "duplicate": "found twice in this search"}


def _docs(n: int) -> str:
    return f"{n} document{'' if n == 1 else 's'}"


def _inline_notice(acq, o):
    """The inline non-answer notice (_inline_notice_body) plus, MIDTERMS-WP8-1106 review F2, the need-to-know stop
    clause (vera.inline_acquire.stop_clause): a heuristic stop is disclosed on the non-answer path too."""
    t = _inline_notice_body(acq, o)
    if not t:
        return t
    try:
        from vera import inline_acquire
        sc = inline_acquire.stop_clause(getattr(o, "acquisition", None))
    except Exception:  # noqa: BLE001
        sc = ""
    if not sc:
        return t
    out = f"{t}{sc}"
    return _InlineNotice(out) if isinstance(t, _InlineNotice) else out


def _inline_notice_body(acq, o):
    """DDFIX-1023/SCOPE (b) + INTEG item 9: the inline "none" notice ("none passed its licence, content and relevance
    checks") is used only when every candidate was actually checked. Otherwise the text follows the per-candidate
    diagnostic statuses (P0DTA-DIAG): licence/open-access prefilter drops are reported as judged on their listed
    licence, R-ACQ-IRRELEVANT ones as judged not relevant, and only cap, deadline, quota or rate skips as not checked.
    Other statuses, or no per-candidate records, keep the configured notice unchanged. Aggregates only (no titles)."""
    from vera import inline_acquire
    t = inline_acquire.notice(acq, o.status)
    recs = getattr(o, "candidates", None)
    if o.status != "none" or not t or recs is None:
        return t
    try:
        from collections import Counter

        from vera.research_outcome import DIAG_LABELS
        recs = list(recs)
        cls = [_notice_class(r) for r in recs]
        n = Counter(cls)
    except Exception:  # noqa: BLE001
        return t
    lead = "VERA-DAR searched for open-access full-text documents on this question while answering"
    total = len(recs)
    if total == 0:
        return f"{lead}, but found no candidate documents to check."
    if n["unknown"] == total:
        return f"{lead}. It found {_docs(total)}, but no per-document outcome was recorded."
    un = Counter(r.get("status") for r, c in zip(recs, cls) if c == "unchecked")
    un_parts = "; ".join(f"{v} {DIAG_LABELS.get(k, k)}" for k, v in un.items())
    n_un = n["unchecked"]
    if n["checked"] == total or n["admitted"]:  # all checked (or a reuse record): the configured text stays true
        return t
    if n["checked"] and n["checked"] + n_un == total:  # the configured sentence is true of the checked ones
        return f"{t} {n_un} other document{'' if n_un == 1 else 's'} found {'was' if n_un == 1 else 'were'} not checked ({un_parts})."
    if n_un == total:
        return (f"{lead}. None of the {total} documents it found was checked in this request ({un_parts}), "
                "so no document was judged on its licence, content or relevance.")
    chk = Counter(r.get("status") for r, c in zip(recs, cls) if c == "checked")
    parts = [f"{v} {DIAG_LABELS.get(k, k)}" for k, v in chk.items()]
    if n["licence"]:
        parts.append(f"{n['licence']} excluded before retrieval because the listed licence or open-access status does "
                     "not permit use")
    if n["irrelevant"]:
        parts.append(f"{n['irrelevant']} judged not relevant to the question")
    if n_un:
        parts.append(f"{n_un} not checked in this request ({un_parts})")
    if n["other"]:
        # NOANSWER-PAGE-1108 D9: the reasons listed are the records' own reason classes (research_outcome.diag_status
        # status_reason: the codes the Research outcome line also uses), counted; never a fixed list that omits one.
        why = Counter(r.get("status_reason") for r, c in zip(recs, cls) if c == "other")
        rs = "; ".join(f"{v} {_OTHER_REASON_TEXT.get(k, 'for another recorded reason')}"
                       for k, v in sorted(why.items(), key=lambda kv: (-kv[1], str(kv[0]))))
        parts.append(f"{n['other']} not retrieved ({rs})")
    if n["unknown"]:
        parts.append(f"{n['unknown']} with no recorded outcome")
    return f"{lead}. It found {_docs(total)} and used none: {'; '.join(parts)}."
def _catstep(decision, question: str, model: str, pricing, chat):
    """CATSTEP-1028 (plan s27; owner 2026-10-28): a well-formed in_scope:false verdict with a missing or unrecognised
    category (D-P10b, after the single retry) is categorised instead of declined. An existing topic or no topic both
    become a clean coverage verdict (category off_topic, fixed reason llm_out:catstep, topic_key set), which then takes
    the unchanged Path A gates (CT-1 / CT-2) exactly as a clean off_topic verdict. Only a model-named refusal category
    refuses (deny:catstep:*; proximity alone never does); a retrieval-only outcome keeps category "" so the record-only
    reading list follows (no answer, no acquisition); invalid optional-LLM output falls back to the previous safe
    decline (classifier_malformed:catstep:*). Any other verdict is returned unchanged. A fault returns the decision unchanged
    (the previous decline path)."""
    try:
        from vera import catstep
        if not catstep.applies(decision, question):
            return decision
        out = catstep.categorise(question, chat=chat, model=model, pricing=pricing)
        spend = {"tokens_used": decision.tokens_used + out.tokens, "cost_usd": round(decision.cost_usd + out.cost, 8),
                 "llm_calls": decision.llm_calls + out.calls}
        if out.status == "refuse":
            return replace(decision, reason=catstep.decision_reason(out), category=out.refuse_category, **spend)
        if out.status == "retrieval_only":  # scrutiny re-check unavailable/malformed or a fault: list only, no answer
            return replace(decision, reason=catstep.RETRIEVAL_ONLY_REASON, category="", **spend)
        # SCOPE-GATE-1104 S2 (D2a): no topical relabel. A reply that named no category is not a safety pass; with no
        # refusal category found it stays the neutral decline (category "", fail closed), never an off-topic path.
        return replace(decision, reason=catstep.REASON_OK, topic_key=out.key, **spend)
    except Exception:  # noqa: BLE001
        return decision


# INTEGRATE-1104 round 2 (safety fix): only an ACCEPTED decision passes safety. A declined decision is never re-read
# as a pass from its label: the legacy labels in_scope / off_topic are off-schema in scope_router (retry, then the
# neutral decline). Kept as a constant for the record; nothing routes on it.
_LEGACY_LABELS = ("in_scope", "off_topic")


_REFUSAL_VERDICTS = ("classifier_malformed:refusal_token", "classifier_malformed:category_conflict")


def _log_basis(decision) -> None:
    """PATHA-1028 CT-4: log the routing basis as class tokens only (feeds B-SCOPE-2 counts). Never raises."""
    try:
        cat = getattr(decision, "category", "")
        from vera.scope_router import SAFE_CATEGORY
        # SCOPE-GATE-1104 S2: basis=safe (passed the safety check) | none; never a topic label
        basis = "safe" if getattr(decision, "accepted", False) is True and getattr(decision, "layer", "") == "layer2_llm" \
            and cat == SAFE_CATEGORY else "none"
        reason = str(getattr(decision, "reason", ""))
        cls = reason.split(":", 1)[0] if reason.startswith(("llm_in:", "llm_out:")) else reason[:48]
        # REFUSE-FIX-1028 V5 (OD-1028(22)): a layer-2 refusal (a clean refusal-category verdict, or a refusal_token /
        # category_conflict verdict) adds refusal=<category>, taken from the closed REFUSE_CATEGORIES enum (the
        # constant itself, never raw or model text). A nullified verdict (no usable category) never carries it.
        ref = _REFUSAL_LOG_TOKENS.get(cat) if getattr(decision, "layer", "") == "layer2_llm" and isinstance(cat, str) \
            else None
        if ref and not (reason.startswith("llm_out:") or reason in _REFUSAL_VERDICTS):
            ref = None
        logging.getLogger("vera").info("scope basis=%s layer=%s class=%s%s rid=%s", basis, getattr(decision, "layer", ""),
                                       cls, f" refusal={ref}" if ref else "", _rid())
        _rt_event("scope", "basis", detail={"basis": basis, "layer": _rt_token(getattr(decision, "layer", "")),
                                            "class": _rt_token(cls), "accepted": bool(getattr(decision, "accepted", False)),
                                            **({"refusal": ref} if ref else {})})
    except Exception:  # noqa: BLE001
        pass


def _rt_event(stage, code, counts=None, detail=None) -> None:
    """PERF-MISS-1104-TRACE: one request-trace event (a no-op outside a traced request; never raises). detail carries
    only class tokens, counts, scores and latency, never question, passage, URL, host or model text."""
    try:
        from vera import request_trace
        request_trace.event(stage, code, counts=counts, detail=detail)
    except Exception:  # noqa: BLE001
        pass


def _rid() -> str:
    """PERF-MISS-1104-TRACE: this request's trace ID for log lines ("-" outside a traced request; never raises)."""
    try:
        from vera import request_trace
        return request_trace.current_id() or "-"
    except Exception:  # noqa: BLE001
        return "-"


def _rt_token(v) -> str:
    """A closed-enum-shaped token for trace detail ("" / "other" for anything that is not label-shaped)."""
    import re as _re
    v = str(v or "")
    return v if _re.fullmatch(r"[a-z][a-z0-9_:]{0,47}", v) else ("" if not v else "other")


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
_REFUSAL_LOG_TOKENS = {c: c for c in sorted(_NO_LIST_CATEGORIES)}  # REFUSE-FIX-1028 V5: closed enum -> log token


def _offtopic_reading_list(decision, grounding):
    """FIX1-1021: a well-formed layer-2 out-of-scope verdict (layer 1 matched nothing) returns a fixed notice plus a
    scholarly-only reading list instead of the refusal. No model call, no full-document path, no answer. The result
    carries sources=() and accepted=False so main.py neither admits anything to the corpus nor touches memory.
    Returns None (caller refuses as before) when the flag is off, the verdict does not qualify or no search seam exists."""
    from vera import answer_contract as contract
    from vera import scope_router as sr

    if grounding is None or not sr.offtopic_applies(decision):
        return None
    unsupported = sr.unsupported_use(decision)  # REFUSE-FIX-1028 V1: code request -> fixed sentence + this list
    if getattr(decision, "category", "") in _NO_LIST_CATEGORIES and not unsupported:  # DAR-1021 SEC M2: advice/harm/...
        return None
    sources_fn, _cfg = grounding
    cands = ()
    tok = None
    try:
        from vera import search_providers
        from vera.grounded_ask import public_candidates

        profile = sr.offtopic_openalex_profile()
        tok = search_providers.OPENALEX_PROFILE.set(profile)  # RL-REL-1021 (None = unchanged)
        cands = public_candidates(_relevant(sources_fn(restricted=True) or [], _QUESTION.get()))
        if not cands and profile is not None:  # RL-REL-1021-B: AND query empty -> ONE more call, old OR query, off-topic only
            search_providers.OPENALEX_PROFILE.reset(tok)
            tok = search_providers.OPENALEX_PROFILE.set(None)
            cands = public_candidates(_relevant(sources_fn(restricted=True) or [], _QUESTION.get()))
    except Exception as exc:  # noqa: BLE001  (fail closed: the fixed refusal; class only in the log)
        import logging

        logging.getLogger("vera").warning("off-topic reading list unavailable (%s)", type(exc).__name__)
        return None
    finally:
        if tok is not None:
            search_providers.OPENALEX_PROFILE.reset(tok)
    notice = sr.offtopic_notice()
    if unsupported:  # fixed config text, never model text; the list's own not-read notice follows
        notice = f"{sr.unsupported_use_text()} {notice}"
    res = contract.reading_list_result(cands, decision.tokens_used, decision.cost_usd, accepted=False)
    return replace(res, answer=notice if cands else f"{notice} {res.answer}", sources=(),
                   contract={**(res.contract or {}), "offtopic": True})


def _relevant(sources, question=None) -> list:
    """DDFIX-1023/INTEG item 6: the reading list without the records this request's inline acquisition marked
    R-ACQ-IRRELEVANT (grounded_ask.drop_irrelevant). Unchanged when none would remain or nothing was marked."""
    from vera.grounded_ask import drop_irrelevant
    return drop_irrelevant(sources, _INLINE_OUT.get())


def _reading_list(question, model, pricing, decision, chained, grounding, contract, tokens=None,
                  cost=None) -> AskResult:
    """PERF-MISS-1105-FIX2 F4: the reading-list fallback was a silent phase (its grounding search emits no stage
    event); it now shows the reading-list activity, re-set every stream.activity_tick_s, plus trace events."""
    from vera import progress as _prog
    _rt_event("reading_list", "start")
    with _prog.ticking(_prog.ACT_READING_LIST):
        res = _reading_list_body(question, model, pricing, decision, chained, grounding, contract, tokens=tokens,
                                 cost=cost)
    _rt_event("reading_list", "done", counts={"candidates": len(getattr(res, "candidates", None) or ())})
    return res


def _reading_list_body(question, model, pricing, decision, chained, grounding, contract, tokens=None,
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

    sources = _relevant(sources, question)  # DDFIX-1023/INTEG item 6 (never empties a non-empty list)
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
        logging.getLogger(__name__).info("acquisition candidates: %s rid=%s", json.dumps(
            {"listed": len(candidates), "not_in_collection": len(out), "by_provider": dict(sorted(by.items()))}), _rid())
    except Exception:  # noqa: BLE001  (a counts log never fails the request)
        pass


def _honest_counts_on(cfg) -> bool:
    """NONASK-1103 A6-4 (D3): answer.inline_honest_counts, exactly true (ships true; absent/false = pre-change)."""
    return isinstance(cfg, dict) and (cfg.get("answer") or {}).get("inline_honest_counts") is True


def _honest_text(cfg, counts: dict) -> str:
    """The honest insufficient notice with SEPARATE stored and fetched counts (never "tried"; counts only)."""
    ns, nf = int(counts.get("stored") or 0), int(counts.get("fetched") or 0)
    t = (cfg.get("answer") or {}).get("inline_honest_text") or "VERA-DAR used {n_stored} stored and {n_fetched} fetched {n_word}."
    return t.replace("{n_stored}", str(ns)).replace("{n_fetched}", str(nf)).replace("{n_word}", "source" if ns + nf == 1 else "sources")


def _fulldoc(question, model, pricing, decision, chained, grounding, acquired_only: bool = False, ephemeral=None,
             used: dict | None = None, lanes=None):
    """D-055 Option C: full-document acquisition/verification -> span retrieval -> admission -> generation -> claim
    check. None when the path is not configured (today's behaviour exactly); an AskResult (result_kind "answer") when
    grounded; else (evidence, fallback token, tokens, cost, text) for an explicit insufficient-evidence result.
    Provider errors from the answering call propagate (the #77 chain tries the next provider)."""
    import logging

    from vera import answer_contract as contract
    from vera import fulldoc_corpus

    counts = _INLINE_COUNTS.get()
    qualified = False  # C6: True when the writer runs only because answer.always_attempt promoted an "insufficient" bundle
    _FD_TRACE.set(None)  # NONASK-1103 E4m: a stale trace of an earlier attempt never reaches this attempt's body
    _FD_TRACE_FETCHED.set(bool(acquired_only and (ephemeral is not None or lanes is not None)))  # SELFQ-RCA-1107 D2
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
        from vera.fulldoc_evidence import attempt_anyway

        if acquired_only:  # WED-ACQ2 scope override (ships off): only acquired documents may answer a declined question
            # DAR-INLINE-1021: `ephemeral` = the in-memory corpus of the documents admitted inline for this request
            acq_corpus = ephemeral if ephemeral is not None else fulldoc_corpus.get_acquired_corpus(cfg)
            if used is not None:
                used["acq"] = acq_corpus
            ev = admit(question, acq_corpus, cfg) if acq_corpus is not None else None
            if lanes is not None:  # NONASK-1103 N2: lane spans join this answer only (answer_only, never stored)
                from vera import lane_evidence
                ev = lane_evidence.merge(ev, lanes, question, cfg)
            if ephemeral is not None or lanes is not None:  # MIDTERMS-WP10-1106: what this inline attempt examined
                _LEDGER_EV.set(ev)
            ev, qualified = attempt_anyway(ev, cfg)  # C6 (R-045): >= 1 admitted passage -> the writer runs
            if ev is None or contract.result_kind_for((ev,)) != "answer":
                if warming:  # in-scope while warming: an explicit insufficient result (the caller may go inline)
                    return (ev, cfg["fallback_tokens"]["insufficient"], decision.tokens_used, decision.cost_usd,
                            _insufficient(cfg["answer"], ev))
                if counts is not None and ephemeral is not None and _honest_counts_on(cfg):  # A6-4: never silent
                    return (ev, cfg["fallback_tokens"]["insufficient"], decision.tokens_used, decision.cost_usd,
                            _honest_text(cfg, counts))
                return None
        else:
            ev = admit(question, corpus, cfg)
            ev, qualified = attempt_anyway(ev, cfg)  # C6 (R-045)
            if used is not None:
                used["ev_base"] = ev  # NOANSWER-RC-1107 r2: the base corpus search, counted with the acquired one
            if contract.result_kind_for((ev,)) != "answer" and ev.status != "corpus_unusable":
                # WED-ACQ2 (plan V6): base corpus first, exactly as before; the separate acquired corpus is consulted
                # only when the base corpus cannot answer, so base answers never change.
                acq_corpus = fulldoc_corpus.get_acquired_corpus(cfg)
                if used is not None:
                    used["acq"] = acq_corpus
                if acq_corpus is not None:
                    ev2 = admit(question, acq_corpus, cfg)
                    ev2, q2 = attempt_anyway(ev2, cfg)
                    if used is not None:
                        used["ev_acq"] = ev2  # FRESH-1023: the stored acquired documents' retrieval, for freshness
                    if contract.result_kind_for((ev2,)) == "answer":
                        ev, qualified = ev2, q2
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
        from vera import fulldoc_answer as _fda

        _FD_TRACE.set(_fda._trace(ev, None, None, None, _fda.sources_from(ev) if getattr(ev, "admitted", None) else None))
        return (ev, tokens_["corpus_unusable" if unusable else "insufficient"], decision.tokens_used,
                decision.cost_usd, texts["corpus_unusable_text"] if unusable else _insufficient(texts, ev, used))
    from vera import fulldoc_answer

    _pre_llm(writer=True)  # STREAM-1031 (no-op off the stream path unless providers are exhausted)
    out = fulldoc_answer.answer(question, ev, model, pricing, chained=chained,
                                grounding_cfg=grounding[1] if grounding else None, cfg=cfg,
                                base_tokens=decision.tokens_used, base_cost=decision.cost_usd,
                                # FIX-CCWAIT-1021: the claim check's verbatim reference is the acquired corpus itself
                                reference_corpus=acq_corpus if acquired_only else None)
    if out.result is not None and out.result.result_kind == "answer":
        if qualified:  # C6: coverage shortfall shown as a qualification, never a reason to skip generation
            _q = texts.get("coverage_qualification_text")
            _a, _n = contract.add_note(out.result.answer, out.result.notes, "coverage", _q)
            out = replace(out, result=replace(out.result, answer=_a, notes=_n))
        return replace(out.result, evidence=ev, accepted=True,
                       contract={**(out.result.contract or {}), "layers": contract.layers(), "fallback": None})  # NONASK-1103: MERGE; only layers+fallback are overridden (all other keys, e.g. trace, kept)
    _FD_TRACE.set(getattr(out, "trace", None))  # NONASK-1103 E4m: no answer, so the body carries the reasons
    # OD-1104-TRACE: a flagged but unverifiable premise (DAR-PREMISE-1022) is an evidence miss: it keeps its
    # premise_unverified token (telemetry, retry gate) but gets the same "could not write a supported answer" text.
    tok = out.fallback or tokens_["insufficient"]
    # NONASK-1103 WP-C: passages WERE selected and the writer or checker declined, so never "found no passage"
    key = "verification_unavailable_text" if tok == tokens_.get("checker_unavailable") else "unsupported_text"
    return (ev, tok, out.tokens_used, out.cost_usd, _unsupported(texts, key, ev))


def _plural_docs(n: int, m: int = 0) -> str:
    """SCOPE-COUNT-1106: with m > 0 documents fetched for this question, "N full documents from VERA-DAR's collection and
    M fetched for this question"; m == 0 keeps the old wording."""
    base = f"{n} full document" if n == 1 else f"{n} full documents"
    return base if m <= 0 else f"{base} from VERA-DAR's collection and {m} fetched for this question"


_SAME_AS_UI = ("insufficient_full_document_evidence", "premise_unverified", "model_insufficient")


def _ui_headline(text: str, ev, ia, token, used) -> str:
    """SELFQ-RCA-1107 D2: with answer.count_matching_documents on, the server no-answer sentence for the count-worded
    tokens is exactly the headline vera.ui_evidence.render_fallback_reason shows for the same response (same token,
    same fetched / matched / selected counts), so the page never carries two different no-answer reasons. Other tokens,
    the flag off, or any error: `text` unchanged."""
    from vera import ui_evidence
    cfg = ui_evidence._cfg()
    if token not in _SAME_AS_UI or not ui_evidence.count_matching_on(cfg):
        return text
    adm = [a for a in (getattr(ev, "admitted", ()) or ()) if isinstance(a, dict)]
    sc = _search_counts(used) if isinstance(used, dict) and used.get("ev_base") is not None else None
    data = {"corpus_source": {"kind": getattr(ev, "corpus_source", None)}, "admitted_evidence": adm,
            "examined_full_documents": _rows(ev) or [],
            "contract": {"fallback": token, "inline_acquisition": ia if isinstance(ia, dict) else None,
                         **({"search_counts": sc} if sc is not None else {})}}
    return ui_evidence.fallback_text(token, cfg, n_inspected=len(data["examined_full_documents"]),
                                     n_selected=ui_evidence._distinct_docs(adm) if adm else 0,
                                     n_fetched=ui_evidence.fetched_for_request(data), n_matched=ui_evidence.matched_for(data))


def _with_fetched(text: str, ev, ia, token=None, used=None) -> str:
    """SCOPE-COUNT-1106: re-word a "searched N full documents" headline once the inline round's fetched documents are
    known (they are checked after the held-collection text was built). Unchanged when none, or when text lacks the phrase.
    SELFQ-RCA-1107 D2: with a fallback `token` and count_matching_documents on, the UI headline for the same response."""
    try:
        from vera import ui_evidence
        if token is not None and ev is not None:
            new = _ui_headline(text, ev, ia, token, used)
            if new != text:
                return new
        m = ui_evidence.fetched_for_request({"corpus_source": {"kind": getattr(ev, "corpus_source", None)},
                                             "contract": {"inline_acquisition": ia}})
        if m <= 0:
            return text
        n = len(getattr(ev, "examined_full_documents", ()) or ())
        for verb in ("searched", "checked"):  # R8-HEADLINE-1106: the non-answer headline says "checked"
            if f"{verb} {_plural_docs(n)}" in text:
                return text.replace(f"{verb} {_plural_docs(n)}", f"{verb} {_plural_docs(n, m)}", 1)
        return text
    except Exception:  # headline wording must never break an answer
        return text


def _rows(ev):
    return None if ev is None else list(getattr(ev, "examined_full_documents", ()) or ())


def _search_counts(used) -> dict | None:
    """NOANSWER-RC-1107 r2: contract.search_counts {"matched": documents of the base AND the stored acquired corpus
    (when searched) with relevance-cleared passages (relevant_spans) for this question}; None without a base attempt
    or an exact count."""
    if not isinstance(used, dict) or used.get("ev_base") is None:
        return None
    try:
        from vera import ui_evidence
        return ui_evidence.search_counts(_rows(used["ev_base"]), _rows(used.get("ev_acq")))
    except Exception:  # noqa: BLE001  (counts must never break the request)
        return None


def _insufficient(texts: dict, ev, used=None) -> str:
    """NOANSWER-RC-1107: the no-passage headline. answer.count_matching_documents on: only documents with candidate
    passages for this question, over every stored corpus searched (base and acquired); no count when unknown; never
    the corpus size. Off: the previous insufficient_text (the only way back to it)."""
    try:
        from vera import ui_evidence
        if isinstance(used, dict) and used.get("ev_base") is not None:
            sc = _search_counts(used)
            n = sc["matched"] if sc is not None else None
        else:
            n = ui_evidence.matched_documents(_rows(ev))
        alt = ui_evidence.insufficient_matching_text(texts, n)
    except Exception:  # noqa: BLE001  (headline wording must never break the request)
        alt = None
    if alt is None and _count_on(texts):
        alt = ("VERA-DAR found no stored passage that answers this question well enough, so it has not written an "
               "answer or answered from memory.")
    return alt or _with_count(texts["insufficient_text"], ev)


def _count_on(texts) -> bool:
    return isinstance(texts, dict) and texts.get("count_matching_documents") is True


def _unsupported(texts: dict, key: str, ev) -> str:
    """NOANSWER-RC-1107 r2: passages selected, no supported answer. Flag on and the unsupported text: count only the
    documents the selected passages came from, never every document searched."""
    if key == "unsupported_text":
        try:
            from vera import ui_evidence
            k = len({a.get("document_id") for a in (getattr(ev, "admitted", ()) or ()) if isinstance(a, dict)})
            alt = ui_evidence.unsupported_matching_text(texts, k)
            if alt:
                return alt
        except Exception:  # noqa: BLE001
            pass
    return _with_count(texts.get(key) or texts["insufficient_text"], ev, selected=True)


def _with_count(text: str, ev, selected: bool = False) -> str:
    """C-4: substitute {n_docs_phrase} ("1 full document" / "9 full documents") and the legacy {n_inspected} (a bare
    number) with a count only, never titles. selected=True counts the documents the admitted passages came from. Plain
    str.replace, so braces elsewhere are safe and a text without a placeholder is returned unchanged."""
    try:
        n = len(getattr(ev, "examined_full_documents", ()) or ())
        ids = {a.get("document_id") for a in (getattr(ev, "admitted", ()) or ()) if isinstance(a, dict)}
        k = len(ids)
        if selected:  # R8-HEADLINE-1106: inspected count and passage-supplying count are separate
            n = max(n, k)
    except TypeError:
        n, k = 0, 0
    return (text.replace("{n_docs_phrase}", _plural_docs(n)).replace("{n_selected}", str(k))
            .replace("{n_inspected}", str(n)))


# NONASK-1103 WP-C (stream B contract E5): fallback token -> no_answer_cause code
_CAUSE_CODE = {"insufficient_full_document_evidence": "no_passages", "full_document_corpus_unusable": "corpus_unusable",
               "checker_unavailable": "verification_unavailable"}  # OD-1104-TRACE: premise_unverified -> default no_supported_answer


def _portal_counts(contract_: dict) -> tuple[int, int]:
    ps = (contract_.get("inline_acquisition") or {}).get("portals")
    ps = [p for p in ps if isinstance(p, dict)] if isinstance(ps, list) else []
    return (sum(1 for p in ps if p.get("status") != "skipped"), sum(1 for p in ps if p.get("status") == "completed"))


def _with_cause(res, code: str, text: str):
    """E5: contract.no_answer_cause {code, text<=300} and contract.research.portals_attempted/completed ints, on a
    non-answer only (an answer is returned unchanged)."""
    if getattr(res, "result_kind", "answer") == "answer":
        return res
    ctr = dict(res.contract or {})
    ctr["no_answer_cause"] = {"code": code, "text": str(text or "")[:300]}
    att, comp = _portal_counts(ctr)
    ctr["research"] = {**(ctr.get("research") if isinstance(ctr.get("research"), dict) else {}),
                       "portals_attempted": att, "portals_completed": comp}
    return replace(res, contract=ctr)


def _answer_grounded(question, model, pricing, sources, cfg, decision, chained) -> AskResult:
    """State A (evidentiary answer). The ONLY caller is the L3 branch of answer_in_scope, and only when full documents
    were examined; the abstract-grounded behaviour it used to implement for every question is retired (DEC-FDA1)."""
    from vera import answer_contract as contract
    from vera.grounded_ask import count_valid_citations, fix_citations, messages_for, public_candidates

    kind = "answer"
    _pre_llm(writer=True)  # STREAM-1031
    _progress.emit("write", "start")  # PROG-1027 W2a (legacy path; test-only today)
    result = complete(messages_for(question, sources), model, pricing, **chained)
    _progress.emit("write", "done")
    text, _ = fix_citations(result.answer or "", len(sources), cfg["removed_citation_note"])
    notes: list = []  # ANS-LAYERS-1101: (kind, text) kept out of the answer when answer_notes_separate is on
    if contract.notes_separate() and cfg["removed_citation_note"] and cfg["removed_citation_note"] in text:
        i = text.rfind(cfg["removed_citation_note"])
        text = text[:i].rstrip()
        notes.append(("removed_citation", cfg["removed_citation_note"]))
    tokens = result.tokens_used + decision.tokens_used
    cost = round(result.cost_usd + decision.cost_usd, 6)
    if count_valid_citations(text, len(sources)) == 0:  # R1 criterion: no valid citation -> no answer
        return contract.reading_list_result(public_candidates(sources), tokens, cost, sources=sources,
                                            fallback="no_valid_citations")  # C-F1: a non-answer, no claim_check
    cc = cfg.get("claim_check") or {}
    if cc.get("enabled"):  # R1 criterion: each cited claim checked against the cited text; fail closed
        from vera.claim_check import RECORDS, check_answer

        records: list = []
        _pre_llm()  # STREAM-1031: no claim check for a prompt whose client is gone
        token = RECORDS.set(records)
        _progress.emit("check", "start")  # PROG-1027 W2a
        try:
            checked, summary = check_answer(text, sources, cc)
        finally:
            RECORDS.reset(token)
        _progress.emit("check", "done")
        if checked is None:
            token = "checker_unavailable" if summary["outcome"] == "checker_unavailable" else "claim_check_insufficient"
            return contract.reading_list_result(public_candidates(sources), tokens, cost, sources=sources,
                                                fallback=token)  # C-F1
        if contract.notes_separate():  # ANS-LAYERS-1101-FC: the checker leaves its removed notes out; build them from the counts
            for k, t, n in (("removed", (cc.get("responsiveness") or {}).get("removed_note"), summary.get("nonresponsive_removed")),
                            ("removed", cc.get("removed_note"), summary.get("removed"))):
                if t and checked.endswith(t):  # a checker that still appended its note: strip it as before
                    checked = checked[: -len(t)].rstrip()
                    notes.append((k, t))
                elif t and n:
                    notes.append((k, t))
        return AskResult(answer=checked, tokens_used=tokens, cost_usd=cost, sources=tuple(sources),
                         claim_check=summary, accepted=True, claims=tuple(records), result_kind=kind,
                         notes=tuple({"kind": k, "text": t} for k, t in notes))
    return AskResult(answer=text, tokens_used=tokens, cost_usd=cost, sources=tuple(sources), accepted=True,
                     result_kind=kind, notes=tuple({"kind": k, "text": t} for k, t in notes))


# ---------------------------------------------------------------------------------------------- FRESH-1023
def _fresh_cfg(acq):
    """config/auto_acquire.json freshness block when enabled (also when acquisition itself is off: the freshness
    limitation must still be disclosed), else None (feature off: today's behaviour exactly). Never raises."""
    from vera import research_record as rr
    try:
        if acq is not None:
            return rr.cfg_of(acq)
        from vera import acq_config
        return rr.cfg_of(acq_config.load_config())
    except Exception:  # noqa: BLE001
        return None


def _source_freshness(sources, fr: dict) -> list:
    """Per cited source number: the freshness of its document and whether it was found by this request's search."""
    by = {str(d.get("document_id")): d for d in (fr or {}).get("documents") or ()}
    out = []
    for s in sources or ():
        if not isinstance(s, dict):
            continue
        d = by.get(str(s.get("document_id")), {})
        out.append({"n": s.get("n"), "document_id": s.get("document_id"), "freshness": d.get("freshness", "unknown"),
                    "new_this_request": bool(d.get("new_this_request")),
                    "evidence_recency": d.get("evidence_recency", "unknown"),
                    **{k: d.get(k) for k in _DATE_KEYS}})
    return out


def _record_discovery(rec, res, question=None) -> None:
    """PORT-P2: discovery events + portal attributions for this request (no-op unless enabled; never raises)."""
    try:
        from vera import portal_recorder
        portal_recorder.record_ask(rec, question=question or "", result=res)
    except Exception:  # noqa: BLE001
        pass


def _with_research(res, rec: dict, suffix: str | None = None, kind: str | None = None):
    from vera import answer_contract as contract
    if kind is None:  # "older evidence only" qualifies the answer the user sees: a limitation (need-to-know), not a process note
        kind = "coverage" if rec.get("kind") == "qualified_older_only" else "research_outcome"
    _record_discovery(rec, res, _QUESTION.get())
    text, notes = contract.add_note(res.answer, res.notes, kind, suffix or "", "suffix", "\n\n")
    return replace(res, answer=text, notes=notes, contract={**(res.contract or {}), "research": rec})


def _backfill(fcfg, at, acq, *frs):
    """Owner refinement 2026-10-23: revise missing/invalid corpus_added_at stamps of the chunks used here to now()."""
    from vera import research_record as rr
    stamps = [x for fr in frs for x in ((fr or {}).get("_backfill") or [])]
    return rr.apply_backfill({"_backfill": stamps}, fcfg, at, acq) if stamps else None


def _merge_bf(a, b):
    if not b:
        return a
    if not a:
        return b
    pa, pb = a.get("persist") or {}, b.get("persist") or {}
    return {"count": a["count"] + b["count"], "chunk_ids": list(a["chunk_ids"]) + list(b["chunk_ids"]), "at": a["at"],
            "persist": {"status": pa.get("status") if pa.get("status") == pb.get("status") else "mixed",
                        "updated": int(pa.get("updated") or 0) + int(pb.get("updated") or 0),
                        "not_persistable": int(pa.get("not_persistable") or 0) + int(pb.get("not_persistable") or 0),
                        "reason": "; ".join(x for x in (pa.get("reason"), pb.get("reason")) if x)}}


def _doc_key(a: dict) -> str:
    """Document identity used for the merge de-duplication (_merge_evidence): canonical source id, else document id."""
    return str(a.get("canonical_source_id") or a.get("document_id"))


def _held_doc_ids(*evs) -> set:
    """Keys of documents already held before this request: admitted passages of the given evidence bundles (by
    canonical_source_id or document_id, plus the raw document_id)."""
    out = set()
    for ev in evs:
        for a in (getattr(ev, "admitted", ()) or ()):
            if isinstance(a, dict):
                out.add(_doc_key(a))
                out.add(str(a.get("document_id")))
    return out


def _acq_held_ids() -> set:
    """Keys of every document in the stored acquired corpus (cached snapshot; empty when unavailable)."""
    try:
        from vera import fulldoc_corpus
        c = fulldoc_corpus.get_acquired_corpus(fulldoc_corpus.load_config())
        return {k for a in (getattr(c, "acquired", ()) or ()) if isinstance(a, dict)
                for k in (_doc_key(a), str(a.get("document_id")))}
    except Exception:  # noqa: BLE001
        return set()


def _reused_keys(pub) -> set:
    """NOANSWER-1109: identity keys of the documents this request's inline round REUSED from the collection (held
    before the request). The cached acquired-corpus snapshot can lag a document stored for an earlier question, which
    made a reused document count as "newly found" beside the note that said it was already held."""
    try:
        from vera import evidence_ledger as el
        recs = (pub or {}).get("candidates") if isinstance(pub, dict) else None
        return {k for r in (recs or ()) if isinstance(r, dict) and r.get("outcome") == "reused"
                for k in el.keys_of(el._cand_ids(r))}
    except Exception:  # noqa: BLE001
        return set()


def _nothing_fresh(pub) -> bool:
    """NOANSWER-1109: the inline round admitted no fresh document, so no cited document can be newly found."""
    a = pub.get("admitted") if isinstance(pub, dict) else None
    return isinstance(a, int) and not isinstance(a, bool) and a == 0


def _new_doc_ids(admitted, *held_evs, held_keys=(), reused_keys=(), none_fresh=False) -> set:
    """Documents among the final admitted passages that were not held before this request. A new passage of an
    already-held document, a de-duplicated passage, a held document re-acquired under a new id and a reused stored
    acquired document are not new. Returns document ids."""
    held = _held_doc_ids(*held_evs) | {str(k) for k in held_keys}
    if none_fresh:
        return set()
    reused = set(reused_keys)
    if reused:
        from vera import evidence_ledger as el
    return {str(a.get("document_id")) for a in (admitted or ())
            if _doc_key(a) not in held and str(a.get("document_id")) not in held
            and not (reused and el.keys_of({k: a.get(k) for k in ("doi", "url", "canonical_url", "canonical_source_id", "pmcid")}) & reused)}


def _fresh_line(texts: dict, fr2: dict, n_new: int) -> str:
    """Publication wording: counts by source_authored_at (authored_counts), never by corpus_added_at."""
    a = fr2.get("authored_counts") or {}
    c = {"fresh": a.get("recent", 0), "stale": a.get("older", 0), "unknown": a.get("unknown", 0)}
    t = texts.get("fresh_new") or ""
    try:
        return t.format(n_new=n_new, n_fresh=c["fresh"], n_stale=c["stale"], n_unknown=c["unknown"])
    except (KeyError, IndexError, ValueError):
        return ""


def _merge_evidence(base, new, cfg):
    """Combined admitted evidence: the base bundle's admitted passages, then the new bundle's (each admitted by the
    unchanged vera.fulldoc_admission.admit), de-duplicated by (document, text hash), within the existing admission caps
    max_spans_total / max_admitted_chars. When both cannot fit, the new evidence keeps at least half of the slots (or
    every free slot), so recent evidence is not crowded out and older evidence is not discarded wholesale."""
    from vera.fulldoc_evidence import EvidenceBundle
    adm = (cfg.get("admission") or {}) if isinstance(cfg, dict) else {}
    max_total = int(adm.get("max_spans_total") or 6)
    max_chars = adm.get("max_admitted_chars")
    seen, b, n = set(), [], []
    for src, dst in ((base.admitted, b), (new.admitted, n)):
        for a in src or ():
            k = (str(a.get("canonical_source_id") or a.get("document_id")), a.get("text_sha256"))
            if k not in seen:
                seen.add(k)
                dst.append(a)
    n_new = min(len(n), max(max_total - len(b), max_total // 2))
    picked = b[: max_total - n_new] + n[:n_new]
    if max_chars is not None:
        kept, tot = [], 0
        for a in picked:
            if tot + len(a.get("text") or "") <= int(max_chars):
                kept.append(a)
                tot += len(a.get("text") or "")
        picked = kept
    ids = {str(a.get("document_id")) for a in picked}
    examined = tuple(d for d in tuple(base.examined_full_documents) + tuple(new.examined_full_documents))
    return EvidenceBundle(question=base.question, examined_full_documents=examined, admitted=tuple(picked),
                          unusable=tuple(base.unusable) + tuple(new.unusable),
                          rejected=tuple(base.rejected) + tuple(new.rejected),
                          status="admitted" if picked else "insufficient", corpus_source=base.corpus_source,
                          trace={**dict(base.trace or {}), "combined": {"base": sum(1 for a in picked if a in b),
                                                                        "new": sum(1 for a in picked if a in n),
                                                                        "documents": len(ids)}})


def _reference_texts(ev, corpora) -> dict | None:
    """Full text of every document in the combined evidence (claim-check verbatim reference), from whichever corpus
    holds it. None (the weaker admitted-passage fallback) when any document cannot be read."""
    out = {}
    for a in ev.admitted:
        did = a.get("document_id")
        if str(did) in out:
            continue
        txt = None
        for c in corpora:
            if c is None:
                continue
            try:
                t = c.text(did)
            except Exception:  # noqa: BLE001
                t = None
            if isinstance(t, str) and t:
                txt = t
                break
        if txt is None:
            return None
        out[str(did)] = txt
    return out


def _freshness_research(question, model, pricing, decision, chained, grounding, acq, fcfg, fd, used_in, budget):
    """FRESH-1023 requirement 1 on the ANSWER path: assess sufficiency and the 24-hour source-content freshness of the
    admitted passages (vera.research_record -> vera.freshness); on either trigger run ONE inline acquisition (existing
    caps, kill switch, request-deadline guard and every gate unchanged), admit the new full-text passages, combine them
    with the base passages and write ONE new claim-checked answer in this request. Without usable new evidence the
    grounded base answer is returned, qualified by the research outcome (vera.research_outcome)."""
    from vera import research_record as rr
    at = rr.now()
    texts = fcfg.get("texts") or {}
    fr = rr.assess(fd.evidence.admitted, fcfg, at)
    trig = rr.triggers(rr.sufficient(fd, fcfg), fr)
    if rr.always_new(fcfg) and rr.TRIGGER_NEW not in trig:  # OD-1109-SIX: search for new documents on every answer
        trig.append(rr.TRIGGER_NEW)
    bf = _backfill(fcfg, at, acq, fr)
    if not trig:
        return _with_research(_annotate(fd, fr), rr.summarize(triggers=[], inline=None, used_new=False,
                                                              older_available=True, fr_before=fr,
                                                              citations=_source_freshness(fd.sources, fr),
                                                              answer_basis="generated_existing_only"))
    from vera import inline_acquire as _ia
    # PERF-MISS-1105-FIX2 F2: start the lane pass concurrently with the research attempt when that attempt can run
    # (its first guards); the result is used only when the attempt ran (o is not None), as before.
    lane_task = _ia.start_lane_pass(question, acq) if _research_may_run(question, acq, budget, grounding) else None
    pub, o = _research_attempt(question, grounding, acq, fcfg, used_in, budget)
    lane_res = lane_task.result() if lane_task is not None and o is not None else None  # unused: not joined
    if o is not None and lane_task is None:
        lane_res = _ia.lane_pass(question, acq)
    lanes = _ia.with_web(o, lane_res, question) if o is not None else None  # NONASK-1103 (+A6-W): request-local
    if lanes is not None and isinstance(pub, dict):
        pub = dict(pub, lanes=lanes.public())  # links and counts only (need-to-know)
    new_res, failed = None, False
    if o is not None and (getattr(o, "corpus", None) is not None or (lanes is not None and lanes.pages)):
        new_res, failed = _combined_answer(question, model, pricing, chained, grounding, acq, fd, o,
                                           **({"lanes": lanes} if lanes is not None else {}))
    lanes = None  # page text is dropped with the request (R-037)
    if new_res is not None:
        new_ids = _new_doc_ids([a for a in new_res.evidence.admitted if not (isinstance(a, dict) and a.get("answer_only"))],
                               fd.evidence, held_keys=_acq_held_ids(), reused_keys=_reused_keys(pub), none_fresh=_nothing_fresh(pub))  # lane pages are read, not found or held
        fr2 = rr.assess(new_res.evidence.admitted, fcfg, at, new_ids=new_ids)
        bf = _merge_bf(bf, _backfill(fcfg, at, acq, fr2))
        found = bool(new_ids)  # newly found = a cited passage from a document not held before this request
        rec = rr.summarize(triggers=trig, inline=pub, used_new=found, older_available=True, fr_before=fr, fr_after=fr2,
                           citations=_source_freshness(new_res.sources, fr2), backfill=bf,
                           answer_basis="generated_new_evidence" if found else "generated_existing_only")
        suffix = " ".join(t for t in ((_fresh_line(texts, fr2, len(new_ids)) if found else ""),
                                      _mixed_line(texts, fr2), rec["user_text"]) if t)
        return _with_six(_with_research(_annotate(new_res, fr2), rec, suffix or None), texts, trig, o, acq, pub)
    rec = rr.summarize(triggers=trig, inline=pub, used_new=False, older_available=True, fr_before=fr,
                       citations=_source_freshness(fd.sources, fr), answer_failed=bool(failed),
                       failed_text=texts.get("new_evidence_unanswered"), backfill=bf,
                       answer_basis="generated_existing_only")
    base = _annotate(fd, fr)
    if failed:  # the failed combined writer/claim-check spend is still reported
        base = replace(base, tokens_used=failed.tokens, cost_usd=round(failed.cost, 6))
    if pub is not None and pub.get("status") not in (None, "off"):
        base = replace(base, contract={**(base.contract or {}), "inline_acquisition": pub})
    return _with_six(_with_research(base, rec, rec["user_text"] or None), texts, trig, o, acq, pub)


def _with_six(res, texts, trig, o, acq, pub=None):
    """OD-1109-SIX: the new-document outcome as its own process note (kind research_outcome), never merged into another."""
    t = _six_line(texts, trig, o, acq, (pub or {}).get("status") if isinstance(pub, dict) else None)
    if not t:
        return res
    from vera import answer_contract as contract
    a, n = contract.add_note(res.answer, res.notes, "research_outcome", t, "suffix", "\n\n")
    return replace(res, answer=a, notes=n)


def _six_line(texts: dict, trig, o, acq, status=None) -> str:
    """OD-1109-SIX: one accurate sentence about the search for new documents. The count sentence
    (freshness.texts.new_documents) ONLY when a search really ran (the outcome status is in inline_acquire.SEARCHED);
    otherwise the sentence for why no search ran (freshness.texts.new_documents_not_run[status], else its "error"):
    hourly quota ("this hour"), daily cap ("today"), busy, off, no time left, error. "" when the trigger is off."""
    from vera import research_record as rr
    if rr.TRIGGER_NEW not in (trig or ()) or not isinstance(texts, dict):
        return ""
    from vera import inline_acquire
    st = str(getattr(o, "status", "") or "") if o is not None else str(status or "")
    if o is not None and st in inline_acquire.SEARCHED:
        t = texts.get("new_documents")
        if not isinstance(t, str) or not t:
            return ""
        try:
            from vera import acq_rounds
            n, m = inline_acquire.new_documents(o), acq_rounds.target_of(acq)
            return t.format(n=n, m=m, s="" if n == 1 else "s")
        except Exception:  # noqa: BLE001
            return ""
    nr = texts.get("new_documents_not_run")
    if not isinstance(nr, dict):
        return ""
    key = {"rate": "rate", "daily_cap": "daily_cap", "busy": "busy", "off": "off", "budget_unknown": "no_time",
           "deadline": "no_time"}.get(st, "error")
    if key == "off" and o is not None and getattr(o, "guard", False) is True and isinstance(nr.get("guard"), str):
        key = "guard"  # BATCH3-1109: the intent guard refused this question; searching itself is not switched off
    t = nr.get(key) or nr.get("error")
    return t if isinstance(t, str) else ""


def _research_may_run(question, acq, budget, grounding=None) -> bool:
    """PERF-MISS-1105-FIX2 F2: the cheap first guards of _research_attempt (no search, no network), so the lane pass
    is started concurrently only for an attempt that can run. Never raises."""
    try:
        if acq is None or not _layer1_clean(question) or getattr(budget, "inline_used", False):
            return False
        from vera import fulldoc_corpus, inline_acquire
        if not (inline_acquire.icfg_of(acq).get("in_scope") is True
                and fulldoc_corpus.flag_enabled(fulldoc_corpus.load_config())):
            return False
        need = inline_need_s(acq, grounding)  # the attempt's own deadline guard (no network)
        return need is not None and budget.deadline - time.monotonic() >= need
    except Exception:  # noqa: BLE001
        return False


def _research_attempt(question, grounding, acq, fcfg, used_in, budget):
    """(public inline outcome dict, Outcome or None): at most ONE inline acquisition per request, under the same
    guards as the in-scope inline retry. A refusal is reported (status), never treated as 'nothing found'."""
    if acq is None:
        return {"status": "off"}, None
    if not _layer1_clean(question):
        return {"status": "off"}, None
    from vera import fulldoc_corpus, inline_acquire
    if inline_acquire.icfg_of(acq).get("in_scope") is not True \
            or not fulldoc_corpus.flag_enabled(fulldoc_corpus.load_config()):
        return {"status": "off"}, None
    if budget.inline_used:
        return {"status": "busy"}, None
    need = inline_need_s(acq, grounding)
    remaining = budget.deadline - time.monotonic()
    if need is None or remaining < need:
        return {"status": "budget_unknown" if need is None else "deadline"}, None
    budget.inline_used = True
    from vera import acq_discovered
    kw = {}
    if acq_discovered.use_discovered(acq) and inline_acquire.inline_enabled(acq):
        kw["discovered"] = acq_discovered.discovery_results(grounding, restricted=False)
    if fcfg.get("rank_prefer_recent") is True:
        kw["prefer_recent"] = True
    o = inline_acquire.acquire(question, acq, served=_doc_keys(used_in.get("acq")), **kw)
    st = str(getattr(o, "status", "error"))
    return (o.public() if st != "off" and hasattr(o, "public") else {"status": st}), o


class _Spend:
    """A truthy 'combined answer failed' marker that carries the spend of the failed writer/claim-check run."""

    def __init__(self, tokens, cost):
        self.tokens, self.cost = int(tokens or 0), float(cost or 0.0)


def _combined_answer(question, model, pricing, chained, grounding, acq, fd, o, lanes=None):
    """(AskResult or None, failure marker or False): new passages admitted from the inline corpus by the unchanged
    admission, merged with the base passages, then ONE answer through the unchanged writer and claim check."""
    from vera import answer_contract as contract
    from vera import fulldoc_answer, fulldoc_corpus, inline_acquire
    from vera.fulldoc_admission import admit
    try:
        cfg = fulldoc_corpus.load_config()
        ev_new = admit(question, o.corpus, cfg) if getattr(o, "corpus", None) is not None else None
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("vera").warning("combined admission failed (%s)", type(exc).__name__)
        return None, False
    has_lane = lanes is not None and bool(lanes.pages)
    if not (ev_new is not None and ev_new.admitted) and not has_lane:
        return None, False
    merged = _merge_evidence(fd.evidence, ev_new, cfg) if ev_new is not None and ev_new.admitted else fd.evidence
    if has_lane:  # NONASK-1103 N2: lane spans in their own slots after the research merge (answer only, never stored)
        from vera import lane_evidence
        with_lanes = lane_evidence.merge(merged, lanes, question, cfg)
        if with_lanes is merged and not (ev_new is not None and ev_new.admitted):
            return None, False  # no lane span admitted and nothing new: unchanged behaviour
        merged = with_lanes
    base_corpus = None
    if not _base_warming():
        try:
            base_corpus = fulldoc_corpus.get_corpus(cfg)
        except Exception:  # noqa: BLE001
            base_corpus = None
    try:
        acq_corpus = fulldoc_corpus.get_acquired_corpus(cfg)
    except Exception:  # noqa: BLE001
        acq_corpus = None
    ref = _reference_texts(merged, (o.corpus, base_corpus, acq_corpus))
    _pre_llm(writer=True)  # STREAM-1031
    out = fulldoc_answer.answer(question, merged, model, pricing, chained=chained,
                                grounding_cfg=grounding[1] if grounding else None, cfg=cfg,
                                base_tokens=fd.tokens_used, base_cost=fd.cost_usd, reference_texts=ref)
    if out.result is None or out.result.result_kind != "answer":
        logging.getLogger("vera").info("combined answer fallback token=%s", out.fallback)
        return None, _Spend(out.tokens_used, out.cost_usd)
    ans, nts = contract.add_note(out.result.answer, out.result.notes, "acquisition",
                                 inline_acquire.answer_note(acq, o), "prefix")
    return replace(out.result, answer=ans, notes=nts, evidence=merged,
                   accepted=True, contract={**(out.result.contract or {}), "layers": contract.layers(), "fallback": None,
                                            "inline_acquisition": o.public()}), False  # NONASK-1103: MERGE; overrides: layers, fallback, inline_acquisition


_DATE_KEYS = ("corpus_added_at", "corpus_added_basis", "source_authored_at", "source_authored_precision")


def _annotate(res, fr: dict):
    """Per cited source (FRESH-AMEND-1023): evidence_recency (recent | older | unknown) from its document's SOURCE
    AUTHORING date, plus both per-document dates: corpus_added_at (the freshness trigger basis) and source_authored_at
    (with precision). A newly added old publication is therefore labelled older, never recent."""
    by = {str(d.get("document_id")): d for d in (fr or {}).get("documents") or ()}
    srcs = []
    for s in res.sources or ():
        if not isinstance(s, dict):
            srcs.append(s)
            continue
        d = by.get(str(s.get("document_id"))) or {}
        srcs.append(dict(s, evidence_recency=d.get("evidence_recency") or "unknown", **{k: d.get(k) for k in _DATE_KEYS}))
    return replace(res, sources=tuple(srcs))


def _mixed_line(texts: dict, fr2: dict) -> str:
    docs = (fr2 or {}).get("documents") or ()
    if any(d.get("new_this_request") for d in docs) and any(not d.get("new_this_request") for d in docs):
        return texts.get("mixed_ages") or ""
    return ""


def _relevant_docs(*evs) -> list:
    """Retrieved documents with at least one candidate passage for this question (freshness of the RETRIEVED relevant
    chunks when nothing was admitted), as admitted-like rows carrying the document's own date fields."""
    out = []
    for ev in evs:
        for d in (getattr(ev, "examined_full_documents", ()) or ()):
            # NINE-DOCS-RC4-1108: deliberately raw candidates here (freshness-trigger input, not a displayed count;
            # tests/test_p0dta_live_regression.py relies on near-miss documents triggering the freshness check)
            if isinstance(d, dict) and (d.get("candidate_spans") or 0) > 0:
                out.append(d)
    return out


def _research_on_insufficient(res, ev, acq, inline_pub, answered: bool, extra_ev=None, held_ev=None):
    """FRESH-1023 on the NON-answer path (the existing in-scope inline retry is the insufficiency trigger): attach the
    research record and the concise outcome. No freshness block = unchanged result."""
    # acquisition off, or no inline attempt possible (status off): non-answers stay exactly as before
    off = acq is None or not isinstance(inline_pub, dict) or inline_pub.get("status") in (None, "off")
    fcfg = None if off and not answered else _fresh_cfg(acq)
    if fcfg is None:
        return res
    from vera import research_record as rr
    at = rr.now()
    texts = fcfg.get("texts") or {}
    fr = rr.assess(getattr(ev, "admitted", ()) or _relevant_docs(ev, extra_ev), fcfg, at)
    trig = rr.triggers(False, fr)
    bf = _backfill(fcfg, at, acq, fr)
    if answered:
        adm = res.evidence.admitted if res.evidence is not None else ()
        new_ids = _new_doc_ids(adm, ev, extra_ev, held_ev, held_keys=_acq_held_ids(), reused_keys=_reused_keys(inline_pub), none_fresh=_nothing_fresh(inline_pub))
        fr2 = rr.assess(adm, fcfg, at, new_ids=new_ids)
        bf = _merge_bf(bf, _backfill(fcfg, at, acq, fr2))
        found = bool(new_ids)  # newly found = a cited passage from a document not held before this request
        rec = rr.summarize(triggers=trig, inline=inline_pub, used_new=found, older_available=bool(adm and not found),
                           fr_before=fr, fr_after=fr2, citations=_source_freshness(res.sources, fr2), backfill=bf,
                           answer_basis="generated_new_evidence" if found else "generated_existing_only")
        suffix = " ".join(t for t in ((_fresh_line(texts, fr2, len(new_ids)) if found else ""),
                                      rec["user_text"]) if t)
        return _with_research(_annotate(res, fr2), rec, suffix or None)
    rec = rr.summarize(triggers=trig, inline=inline_pub, used_new=False, older_available=False, fr_before=fr,
                       backfill=bf, answer_basis="abstained")
    _record_discovery(rec, res, _QUESTION.get())
    return replace(res, contract={**(res.contract or {}), "research": rec})


# ---------------------------------------------------------------------------------------------- EXHAUST-GRACE-1031
def complete_after_exhaustion(question: str, grounding, state: dict) -> AskResult:
    """OD-1031-EXHAUST-GRACE: every free provider is exhausted AFTER work started on this Ask. No further free-provider
    call; no new acquisition. Completes the response with what exists, best first:
      1. OpenAI (scoped, config exhaustion.grace.openai_completion, OPENAI_API_KEY present): the unchanged full-document
         writer and claim check over the admitted evidence (including this request's inline-admitted documents); a
         verified answer is returned normally;
      2. else the existing non-LLM reading list (or the correct no-answer text when nothing was found) via _reading_list.
    Never an unverified claim; evidence and licence standards are untouched. The result carries
    contract.exhaustion_grace = {mode, stage} and a plain-text notice."""
    import os

    from vera import answer_contract as contract
    from vera import progress as _prog
    from vera import provider_exhaustion as _px
    from vera.scope_router import decline_text

    decision = state.get("decision")
    gcfg = _px.grace_cfg()
    stage = None
    try:
        rec = _prog.SINK.get()
        stage = (rec._events[-1]["stage"] if rec is not None and rec._events else None)
    except Exception:  # noqa: BLE001
        stage = None
    budget = REQUEST_BUDGET.get()
    if budget is not None:
        budget.inline_used = True  # no new acquisition in the completion
    _prog.set_activity(_prog.ACT_BUSY_FALLBACK)  # NOISY-UI: "free AI providers are busy; finishing with backup"
    info = {"mode": "reading_list", "stage": stage}
    if state.get("inline"):
        info["inline_acquisition"] = state["inline"]

    def finish(res: AskResult, mode: str, notice_key: str) -> AskResult:
        info["mode"] = mode
        note = (gcfg.get(notice_key) or "").strip()
        ctr = {**(res.contract or {}), "exhaustion_grace": dict(info)}
        if info.get("inline_acquisition") and "inline_acquisition" not in ctr:
            ctr["inline_acquisition"] = info["inline_acquisition"]
        if res.result_kind == "answer" and contract.notes_separate():  # a real answer: the grace notice is a process note, not answer text
            _a, _n = contract.add_note(res.answer, res.notes, "research_outcome", note, "suffix")
            return replace(res, answer=_a.strip(), notes=_n, contract=ctr)
        return replace(res, answer=f"{res.answer} {note}".strip() if note else res.answer, contract=ctr)

    if decision is None or not decision.accepted:  # nothing LLM-dependent was pending; the unchanged decline text
        return finish(AskResult(answer=decline_text(), tokens_used=getattr(decision, "tokens_used", 0) or 0,
                                cost_usd=getattr(decision, "cost_usd", 0.0) or 0.0, result_kind="decline"),
                      "decline", "notice_fallback")

    fd = None
    gp = _px.grace_openai()
    key_ok = bool((os.getenv("OPENAI_API_KEY") or "").strip())
    if gp is not None and key_ok and _px.grace_openai_enabled():
        model, pricing = gp
        tok = None
        try:
            from openai import OpenAI
            client = OpenAI(max_retries=0, **({"timeout": t} if (t := _chain_timeout_s()) else {}))
            tok = _px.GRACE.set({"client": client})
            logging.getLogger("vera").info("exhaustion grace: OpenAI completion (writer and claim check only)")
            fd = _fulldoc(question, model, pricing, decision, {"client": client}, grounding,
                          acquired_only=state.get("ephemeral") is not None, ephemeral=state.get("ephemeral"), used={})
        except (asyncio.CancelledError, _stream_ctx.StreamCancelled):
            raise  # EXG-FIX-1031 D3: a client disconnect must stop the paid completion, never fall through to the reading list
        except Exception as exc:  # noqa: BLE001  (OpenAI absent/failed/rate-limited: the non-LLM completion below)
            logging.getLogger("vera").warning("exhaustion grace: OpenAI completion failed (%s)", type(exc).__name__)
            fd = None
        finally:
            if tok is not None:
                _px.GRACE.reset(tok)
        if isinstance(fd, AskResult) and fd.result_kind == "answer":
            return finish(replace(fd, accepted=True), "openai_completed", "notice_openai")
    # non-LLM completion: the existing reading list / no-answer path, no model call
    tokens = int(fd[2] or 0) if isinstance(fd, tuple) and len(fd) == 5 else decision.tokens_used  # fd spend includes the base
    cost = decision.cost_usd
    if isinstance(fd, tuple) and len(fd) == 5:
        cost = round(float(fd[3] or decision.cost_usd), 6)  # keep the spend of the attempted writer/check
    try:
        res = _reading_list(question, "", None, decision, {}, grounding, contract, tokens=tokens, cost=cost)
    except Exception as exc:  # noqa: BLE001  (search seam unavailable: the correct no-answer result)
        logging.getLogger("vera").warning("exhaustion grace: reading list unavailable (%s)", type(exc).__name__)
        res = contract.reading_list_result((), tokens, cost)
    _t = _fd_trace_contract() if isinstance(fd, tuple) else {}
    if _t:  # NONASK-1103 E4m: merge, never replace
        res = replace(res, contract={**(res.contract or {}), **_t})
    return finish(res, "reading_list" if res.candidates else "no_answer", "notice_fallback")
