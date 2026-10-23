"""OpenAI request handling for /ask: the completion call, token-usage
extraction, and cost calculation. OpenAI SDK exceptions intentionally
propagate to the routing layer, where main.py translates provider failures
into HTTP responses — this module has no knowledge of HTTP.
"""
import contextvars
import json
import logging
import time
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


@dataclass
class RequestBudget:
    """CMF-1022: ONE request-level deadline and inline-use flag shared by the whole provider chain (never restarted per
    provider). `deadline` is time.monotonic() at the /ask handler entry + public_mode.ASK_TIMEOUT_S (the client's wait).
    The server does NOT terminate a request at the deadline (client-side enforcement only; reported residual gap)."""
    deadline: float
    inline_used: bool = False


def request_budget() -> RequestBudget:
    from vera.public_mode import ASK_TIMEOUT_S
    return RequestBudget(time.monotonic() + ASK_TIMEOUT_S)


# The budget travels with the request in a context variable (set once by main.ask, around answer_via_chain), so the
# chain loop and every provider attempt see the SAME deadline and inline flag without changing any signature.
REQUEST_BUDGET: contextvars.ContextVar = contextvars.ContextVar("cmf_request_budget", default=None)
_QUESTION: contextvars.ContextVar = contextvars.ContextVar("port_p2_question", default=None)  # PORT-P2: query hash only
# DDFIX-1023/INTEG item 6: this request's inline-acquisition outcome (its R-ACQ-IRRELEVANT marks), reset per request
_INLINE_OUT: contextvars.ContextVar = contextvars.ContextVar("ddfix_inline_outcome", default=None)


# Post-inline LLM calls that can follow an inline search, per the auto_acquire.json _calibration_note: one answer writer,
# one claim check, and a second claim-check call (the note reserves room for it). Each is bounded by the chain timeout_s.
POST_INLINE_LLM_CALLS = 3
# The reading-list fallback after a failed retry runs the grounding search twice at most (rl3 AND, then OR), each bounded
# by grounding.deadline_s (calibration note: "<= 2 x 10 s").
READING_LIST_SEARCHES = 2


def inline_need_s(acq: dict, grounding) -> float | None:
    """Seconds one inline retry may need: inline deadline_s + POST_INLINE_LLM_CALLS x chain timeout_s + the reading-list
    tail (READING_LIST_SEARCHES x grounding deadline_s; 0 when ungrounded). Every term is an existing config value.
    None when any term cannot be read (the caller then skips the retry: fail closed)."""
    ct = _chain_timeout_s()
    from vera import inline_acquire
    dl = inline_acquire._b(acq, "deadline_s", 45)  # the value inline_acquire itself applies (its own default 45 when absent)
    if ct is None or isinstance(dl, bool) or not isinstance(dl, (int, float)):
        return None
    tail = 0.0
    if grounding is not None:
        from vera.grounded_ask import load_grounding  # the shipped grounding block (config/ask-provider-chain.json)
        gd = (load_grounding() or {}).get("deadline_s")
        if isinstance(gd, bool) or not isinstance(gd, (int, float)):
            return None
        tail = READING_LIST_SEARCHES * float(gd)
    return float(dl) + POST_INLINE_LLM_CALLS * ct + tail


def answer_in_scope(question: str, model: str, pricing: PricingRecord, client: OpenAI | None = None,
                    extra_body: dict | None = None, grounding=None, budget: RequestBudget | None = None) -> AskResult:
    """The /ask entry point (item #72 W1): the scope router runs first; a declined question never reaches the
    answering model and gets the fixed decline template. answer_question stays the bare model call (the trace
    harness's baseline shape)."""
    from vera.scope_router import decline_text, route

    from vera.scope_router import default_chat

    def chat(messages, model_, l2):
        return default_chat(messages, model_, l2, client=client, extra_body=extra_body)

    _INLINE_OUT.set(None)
    _QUESTION.set(question)  # PORT-P2: the discovery recorder hashes it (text never stored)
    budget = budget or REQUEST_BUDGET.get() or request_budget()  # CMF-1022: one deadline per request (main.py passes the handler-entry one)
    decision = route(question, model, pricing, chat=chat if (client or extra_body) else None)
    acq = _acq_cfg_if_on()  # WED-ACQ2: None unless automatic acquisition is effectively on (byte-identical when off)
    if _research_eligible(decision, question, acq):  # DDFIX-1023/SCOPE: corpus coverage is not research eligibility
        decision = replace(decision, accepted=True)
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
                    inl = None
                    if not budget.inline_used:  # CMF-1022 CR-3: at most one _inline per request across the chain
                        budget.inline_used = True
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
        _INLINE_OUT.set(ia)  # DDFIX-1023/INTEG item 6: the reading list drops records this attempt marked irrelevant
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
        fcfg = _fresh_cfg(acq)  # FRESH-1023: a retrieved, grounded answer is not by itself a reason to stop researching
        if fcfg is None or fd.evidence is None:
            return fd
        return _freshness_research(question, model, pricing, decision, chained, grounding, acq, fcfg, fd, used_in,
                                   budget)
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
        # CMF-1022 R1: also three post-writer failure tokens (premise, claim_check_disabled and checker_unavailable stay out: HITL-CMF-6, OD-CMF-1022g).
        retry_tokens = (ft["insufficient"], ft["corpus_unusable"], ft.get("nonresponsive"), ft.get("no_valid_citations"),
                        ft.get("claim_check_insufficient"), ft.get("model_insufficient"))
        if inline_acquire.icfg_of(acq).get("in_scope") is True and token and token in retry_tokens \
                and not budget.inline_used:  # CMF-1022 CR-3: one _inline per request
            need = inline_need_s(acq, grounding)
            remaining = budget.deadline - time.monotonic()
            if need is None or remaining < need:  # fail closed: unknown budget or too little time -> skip the retry
                ia = {"status": "budget_unknown" if need is None else "deadline"}
                logging.getLogger("vera").info("inline retry skipped: %s", ia["status"])
            else:
                budget.inline_used = True
                info: dict = {}
                inl = _inline(question, model, pricing, decision, chained, grounding, acq,
                              tried=_doc_keys(used_in.get("acq")), in_scope=True, info=info)
                if isinstance(inl, AskResult):
                    return _research_on_insufficient(inl, ev, acq, info.get("outcome"), answered=True)
                inline_note = inl
                ia = info.get("outcome") or {}
    _INLINE_OUT.set(ia)  # DDFIX-1023/INTEG item 6: the reading list drops records this attempt marked irrelevant
    res = _reading_list(question, model, pricing, decision, chained, grounding, contract, tokens=tokens, cost=cost)
    trig = {"acquire_trigger": "insufficient"} if acq is not None and _acq_on_insufficient(acq) and inline_note is None else {}
    out = replace(res, answer=f"{text} {inline_note}" if inline_note else text, evidence=ev,
                  contract={**(res.contract or {"layers": contract.layers()}), "fallback": token,
                            **_inline_trace(inline_note), **({"inline_acquisition": ia} if ia else {})}, **trig,
                  **_spent(inline_note, res.tokens_used, res.cost_usd, decision.tokens_used, decision.cost_usd))
    return _research_on_insufficient(out, ev, acq, ia or None, answered=False, extra_ev=used_in.get("ev_acq"))


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
    if acq_discovered.use_discovered(acq) and inline_acquire.inline_enabled(acq):  # E5: no discovery search when acquire() is off; the off-topic path lists the restricted search, in-scope the full one
        kw["discovered"] = acq_discovered.discovery_results(grounding, restricted=not in_scope)
    o = inline_acquire.acquire(question, acq, served=tried, **kw)  # R2: only documents actually tried are skipped
    if info is not None:
        st = str(getattr(o, "status", "error"))  # "off" carries no counts: identical whether the flag or the run said off
        info["outcome"] = o.public() if st != "off" and hasattr(o, "public") else {"status": st}
    bg = inline_acquire.background_note(acq, o)  # CONT-1023: "" unless candidates were queued for background acquisition
    if o.corpus is None:
        t = _inline_notice(acq, o)
        return f"{t} {bg}".strip() if t is not None and bg else t
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
        note = " ".join(t for t in (inline_acquire.answer_note(acq, o), bg) if t)
        return replace(out, answer=f"{note} {out.answer}" if note else out.answer,
                       contract={**(out.contract or {}), "acquired_only": True, "inline_acquisition": o.public()})
    lead = " ".join(t for t in (acq_config.override_answer_prefix(acq), inline_acquire.answer_note(acq, o), bg) if t)
    return replace(out, answer=f"{lead} {out.answer}" if lead else out.answer, accepted=False,
                   contract={**(out.contract or {}), "offtopic": True, "acquired_only": True,
                             "inline_acquisition": o.public()})


def _research_eligible(decision, question: str, acq) -> bool:
    """DDFIX-1023/SCOPE (owner directive 2026-10-23): curated corpus coverage is not research eligibility. A WELL-FORMED
    layer-2 verdict that a question is only outside the curated subject domains (category "off_topic": harmless) takes
    the normal research path: corpus attempt, unrestricted discovery, inline full-text acquisition and the unchanged
    admission, answer writer and claim check (no answer without admitted evidence). Refusal categories (advice, harmful,
    programming_task, rule_change), layer-1 deny/injection hits, data markers, malformed or contradictory verdicts, a
    missing category and layer2_mode none keep the decline path. It applies only where the shipped switches already
    authorise researching AND answering such a question (acquisition effectively on, scope_expansion
    override_layer2_declines, decline trigger and layer-2 category gate), so every existing kill switch still applies;
    what changes is that the research runs as the normal path, not as a refusal with a reading list. Fails closed."""
    try:
        from vera import acq_config
        return (acq is not None and decision.accepted is False and decision.layer == "layer2_llm"
                and str(decision.reason).startswith("llm_out:") and decision.category == "off_topic"
                and decision.category not in _NO_LIST_CATEGORIES
                and acq_config.override_layer2_declines(acq) and acq_config.layer2_category_ok(acq, decision)
                and acq_config.trigger_for_decline(acq, decision.layer, decision.reason) and _layer1_clean(question))
    except Exception:  # noqa: BLE001
        return False




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


def _docs(n: int) -> str:
    return f"{n} document{'' if n == 1 else 's'}"


def _inline_notice(acq, o):
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
        parts.append(f"{n['other']} not retrieved for another recorded reason (already held, no identifier or no "
                     "retrieval route)")
    if n["unknown"]:
        parts.append(f"{n['unknown']} with no recorded outcome")
    return f"{lead}. It found {_docs(total)} and used none: {'; '.join(parts)}."


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
                    if used is not None:
                        used["ev_acq"] = ev2  # FRESH-1023: the stored acquired documents' retrieval, for freshness
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


def _with_research(res, rec: dict, suffix: str | None = None):
    _record_discovery(rec, res, _QUESTION.get())
    text = res.answer
    if suffix:
        text = f"{text}\n\n{suffix}" if text else suffix
    return replace(res, answer=text, contract={**(res.contract or {}), "research": rec})


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
    bf = _backfill(fcfg, at, acq, fr)
    if not trig:
        return _with_research(_annotate(fd, fr), rr.summarize(triggers=[], inline=None, used_new=False,
                                                              older_available=True, fr_before=fr,
                                                              citations=_source_freshness(fd.sources, fr),
                                                              answer_basis="generated_existing_only"))
    pub, o = _research_attempt(question, grounding, acq, fcfg, used_in, budget)
    new_res, failed = None, False
    if o is not None and getattr(o, "corpus", None) is not None:
        new_res, failed = _combined_answer(question, model, pricing, chained, grounding, acq, fd, o)
    if new_res is not None:
        base_keys = {(a.get("document_id"), a.get("text_sha256")) for a in fd.evidence.admitted}
        new_ids = {str(a.get("document_id")) for a in new_res.evidence.admitted
                   if (a.get("document_id"), a.get("text_sha256")) not in base_keys}
        fr2 = rr.assess(new_res.evidence.admitted, fcfg, at, new_ids=new_ids)
        bf = _merge_bf(bf, _backfill(fcfg, at, acq, fr2))
        rec = rr.summarize(triggers=trig, inline=pub, used_new=True, older_available=True, fr_before=fr, fr_after=fr2,
                           citations=_source_freshness(new_res.sources, fr2), backfill=bf,
                           answer_basis="generated_new_evidence")
        suffix = " ".join(t for t in (_fresh_line(texts, fr2, len(new_ids)), _mixed_line(texts, fr2), rec["user_text"])
                          if t)
        return _with_research(_annotate(new_res, fr2), rec, suffix or None)
    rec = rr.summarize(triggers=trig, inline=pub, used_new=False, older_available=True, fr_before=fr,
                       citations=_source_freshness(fd.sources, fr), answer_failed=bool(failed),
                       failed_text=texts.get("new_evidence_unanswered"), backfill=bf,
                       answer_basis="generated_existing_only")
    base = _annotate(fd, fr)
    if failed:  # the failed combined writer/claim-check spend is still reported
        base = replace(base, tokens_used=failed.tokens, cost_usd=round(failed.cost, 6))
    if pub is not None and pub.get("status") not in (None, "off"):
        base = replace(base, contract={**(base.contract or {}), "inline_acquisition": pub})
    return _with_research(base, rec, rec["user_text"] or None)


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


def _combined_answer(question, model, pricing, chained, grounding, acq, fd, o):
    """(AskResult or None, failure marker or False): new passages admitted from the inline corpus by the unchanged
    admission, merged with the base passages, then ONE answer through the unchanged writer and claim check."""
    from vera import answer_contract as contract
    from vera import fulldoc_answer, fulldoc_corpus, inline_acquire
    from vera.fulldoc_admission import admit
    try:
        cfg = fulldoc_corpus.load_config()
        ev_new = admit(question, o.corpus, cfg)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("vera").warning("combined admission failed (%s)", type(exc).__name__)
        return None, False
    if not ev_new.admitted:
        return None, False
    merged = _merge_evidence(fd.evidence, ev_new, cfg)
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
    out = fulldoc_answer.answer(question, merged, model, pricing, chained=chained,
                                grounding_cfg=grounding[1] if grounding else None, cfg=cfg,
                                base_tokens=fd.tokens_used, base_cost=fd.cost_usd, reference_texts=ref)
    if out.result is None or out.result.result_kind != "answer":
        logging.getLogger("vera").info("combined answer fallback token=%s", out.fallback)
        return None, _Spend(out.tokens_used, out.cost_usd)
    note = inline_acquire.answer_note(acq, o)
    return replace(out.result, answer=f"{note} {out.result.answer}" if note else out.result.answer, evidence=merged,
                   accepted=True, contract={"layers": contract.layers(), "fallback": None,
                                            "inline_acquisition": o.public()}), False


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
            if isinstance(d, dict) and (d.get("candidate_spans") or 0) > 0:
                out.append(d)
    return out


def _research_on_insufficient(res, ev, acq, inline_pub, answered: bool, extra_ev=None):
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
        new_ids = {str(a.get("document_id")) for a in adm}
        fr2 = rr.assess(adm, fcfg, at, new_ids=new_ids)
        bf = _merge_bf(bf, _backfill(fcfg, at, acq, fr2))
        rec = rr.summarize(triggers=trig, inline=inline_pub, used_new=True, older_available=False, fr_before=fr,
                           fr_after=fr2, citations=_source_freshness(res.sources, fr2), backfill=bf,
                           answer_basis="generated_new_evidence")
        suffix = " ".join(t for t in (_fresh_line(texts, fr2, len(new_ids)), rec["user_text"]) if t)
        return _with_research(_annotate(res, fr2), rec, suffix or None)
    rec = rr.summarize(triggers=trig, inline=inline_pub, used_new=False, older_available=False, fr_before=fr,
                       backfill=bf, answer_basis="abstained")
    _record_discovery(rec, res, _QUESTION.get())
    return replace(res, contract={**(res.contract or {}), "research": rec})
