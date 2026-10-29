import logging
import os
from enum import Enum

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from openai import AuthenticationError, OpenAIError, RateLimitError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_serializer

from vera import answer_contract
from vera.corpus_admission import admit as admit_to_corpus, llm_call_for, store_from_env
from vera.corpus_description import CorpusDescriber, generate_with_chain, load_cfg as load_corpus_description_cfg
from vera.grounded_ask import load_corpus_cfg, load_grounding, public_sources, retrieve_all
from vera.memory_gate import VERDICT_SUPPORTED
from vera.memory_store import CONFIRMED, NOT_FOUND, REJECTED, MemoryUnavailable, load_memory_block, memory_from_env
from vera.memory_operator import make_verify_owner_key
from vera.ask_service import OpenAIFallbackDisabled, forbid_default_client
from vera.inference_chain import answer_via_chain, load_chain, openai_fallback_enabled
from vera.auth import (
    OPENAI_KEY_ERROR_DETAIL,
    classify_openai_api_key,
    classify_vera_api_key,
    verify_api_key,
)
from vera.version_info import deployed_commit, short_commit
from vera.public_mode import enforce_caps, record_request_cost
from vera.pricing.config import latest_pricing_for, load_model_pricing, load_model_selection
from vera.env_files import load_vera_env
from vera import trace_viz
from vera import progress

logging.basicConfig(level=logging.INFO)  # before load_vera_env so its names-only audit line is emitted
load_vera_env()  # PLAN-ENV: R/config/env/.env-vera (explicit, preflighted); the file wins over inherited values

logger = logging.getLogger("vera")


class _NoExcText(logging.Filter):
    """R-028: an exception message or traceback can quote user input (e.g. an HTTP client error quotes the request URL with the
    query words). Keep the exception type only, for records on the server-error path (uvicorn re-logs unhandled errors)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info or record.exc_text:
            et = record.exc_info[0].__name__ if record.exc_info and record.exc_info[0] else "Exception"
            record.msg = f"{str(record.msg).rstrip()} ({et})"
            record.args, record.exc_info, record.exc_text, record.stack_info = (), None, None, None
        return True


def _harden_server_logging() -> None:
    """R-028: no per-request access log (it carries the caller's ip:port) and no exception text on the server-error path.
    Done in-app so it holds however the host launches uvicorn (uvicorn configures logging before importing this module)."""
    access = logging.getLogger("uvicorn.access")
    access.disabled, access.propagate = True, False
    access.handlers.clear()
    access.addFilter(lambda record: False)  # belt and braces if something re-enables the logger
    for name in ("uvicorn.error", "uvicorn", "starlette", "fastapi"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, _NoExcText) for f in lg.filters):
            lg.addFilter(_NoExcText())


_harden_server_logging()

app = FastAPI()


# --- Model selection and pricing config -------------------------------------
# Schema, loaders, and the append-only writer live in vera/pricing/config.py,
# shared with scripts/append_model_pricing.py and
# scripts/refresh_openai_pricing.py — main.py just coordinates: load config,
# pick the active record, hand it to the service layer. See the Resource
# invariant in docs/vera-design.md: config is version-controlled JSON, not a
# database, because Render's filesystem is ephemeral and would not preserve a
# runtime database file across a redeploy.

_model_selection = load_model_selection()
_model_pricing = load_model_pricing()

MODEL = _model_selection.selected_model if _model_selection else "gpt-4o-mini"
CURRENT_PRICING = latest_pricing_for(MODEL, _model_pricing)
OPENAI_FALLBACK = openai_fallback_enabled()  # config/ask-provider-chain.json openai_fallback.enabled (shipped false)
ASK_CHAIN = load_chain(MODEL, CURRENT_PRICING, openai_enabled=OPENAI_FALLBACK)  # item #77: free providers first; OpenAI last only when enabled
GROUNDING = load_grounding()  # item #79: live scholarly retrieval for /ask (None = ungrounded)
CORPUS = load_corpus_cfg()  # item #79 phase C: corpus admission settings (None = disabled)
_DESC_CFG = load_corpus_description_cfg()  # item #83
MEMORY_BLOCK = load_memory_block()  # item #84
CORPUS_DESCRIBER = CorpusDescriber(_DESC_CFG) if _DESC_CFG else None

if OPENAI_FALLBACK and CURRENT_PRICING is None:
    logger.warning(
        "No usable pricing record for model %s at startup — /ask will reject requests "
        "until config/model-pricing.json has a record for this model", MODEL,
    )


# --- API key startup checks --------------------------------------------------
# Classification policy lives in vera/auth.py. main.py coordinates startup
# checks so configuration problems surface before the affected request path.

def startup_checks() -> None:
    """Key checks and the active-provider log line; called once at startup (below) and directly by tests. With the
    OpenAI fallback off, OPENAI_API_KEY is not classified or required, and the default-client latch is set so no
    code path in this process can construct an OpenAI client with no base_url."""
    checks = [(classify_vera_api_key(os.getenv("VERA_API_KEY")), "VERA_API_KEY")]
    if OPENAI_FALLBACK:
        checks.insert(0, (classify_openai_api_key(os.getenv("OPENAI_API_KEY")), "OPENAI_API_KEY"))
    else:
        forbid_default_client()
    for _status, _var in checks:
        if _status != "ok":
            logger.warning("%s looks %s at startup — requests that need it will be rejected until it's fixed in .env", _var, _status)
    checkers = [f"{c['provider']}:{c['model']}" for c in ((GROUNDING or {}).get("claim_check") or {}).get("checkers", [])
                if OPENAI_FALLBACK or c.get("base_url") is not None]
    logger.info("providers: answer=%s, checkers=%s, openai_fallback=%s, OPENAI_API_KEY=%s", [e.label for e in ASK_CHAIN],
                checkers, "on" if OPENAI_FALLBACK else "off",
                "set" if (os.getenv("OPENAI_API_KEY") or "").strip() else "unset" + ("" if OPENAI_FALLBACK else " (ignored)"))
    if not OPENAI_FALLBACK and (os.getenv("OPENAI_API_KEY") or "").strip():
        logger.info("OPENAI_API_KEY is set but ignored: openai_fallback is off")


startup_checks()

# TOPICREG-1028 (fail loud): a key-like value in config/auto_acquire.json raises acq_config.ConfigError here (paths logged at ERROR, never
# values). HARDEN-FIX2-011: the service DEGRADES instead of crashing: the refusal is logged at ERROR by load_config (paths only), acquisition
# is marked off (every request-path caller already catches ConfigError and treats the feature as OFF), /health reports it (no values),
# and answering stays up. test_acq_urls.py remains the CI gate for the tracked config.
from vera import acq_config as _acq_config  # noqa: E402
ACQ_CONFIG_REFUSED = False
try:
    _acq_config.load_config()
except _acq_config.ConfigError:
    ACQ_CONFIG_REFUSED = True
    logger.error("acquisition config refused at startup: acquisition is OFF; answering stays up (see the config-refused log line for paths)")

# OOE-APRIME-C-1028 D2: import the scope router at process start so the owner-curated classifier category table
# (config/classifier_categories.json) is loaded and validated now; an invalid table raises ScopeConfigError and the
# service does not start (fail closed). The lazy imports in the request paths stay as a second line.
import vera.scope_router as _scope_router  # noqa: E402,F401
_scope_router.load_config()  # FINAL-REVIEW-1028 D3: scope_router.json + rendered category slots fail closed at start

# DAR readiness (codes only, read-only; the probe is bounded by config/dar_health.json timeouts). The startup line is
# logged from a daemon thread so import/startup is never delayed by the DB probe. Never raises.
from vera import dar_health  # noqa: E402
dar_health.startup_log_async()

# WED-P161: warm the DB-backed full-document corpus and its evidence index in ONE daemon thread (only when fulldoc is
# enabled, a DB source is configured and fulldoc.warm_at_startup is not false). Never raises; no thread otherwise.
from vera import fulldoc_warm  # noqa: E402
fulldoc_warm.start()


class APIModel(BaseModel):
    """Shared base: unknown fields 422, they are never silently dropped."""

    model_config = ConfigDict(extra="forbid")


class DemoMode(str, Enum):
    normal = "normal"
    force_bad = "force_bad"


class AskRequest(APIModel):
    question: str = Field(..., max_length=4000)
    mode: DemoMode = DemoMode.normal


class AskSource(BaseModel):
    n: int
    title: str
    url: str
    published: str | None = None
    source_type: str | None = None
    provider: str | None = None  # item #79: per-citation provenance (R1 criterion)
    identifier: str | None = None
    retrieved_at: str | None = None
    licence_decision: str | None = None
    span: str | None = None  # DAR stored-corpus provenance: 'chars=<start>-<end>' of the quoted chunk (absent for web sources)
    canonical_source_id: int | None = None  # DAR stored-corpus identity (absent for web sources)
    source_id: int | None = None
    admission_passed: bool | None = None  # stored text passed admission as a full document
    acquisition: str | None = None  # 'full_document_acquired' for stored full documents
    span_id: str | None = None  # DAR stored-corpus chunk id (absent for web sources)
    disposition: str | None = None  # evidence disposition of a stored source (e.g. "proposed")
    licence: str | None = None  # declared licence id of a stored source (from its stored provenance)
    # FRESH-1023 / FRESH-AMEND-1023 (additive; set only when the freshness step ran, absent otherwise):
    # evidence_recency "recent" | "older" | "unknown" by the SOURCE AUTHORING date (source_authored_at, precision
    # year|month|day; a partial date is never "recent"); corpus_added_at = when the chunk was added to the corpus (the
    # 24-hour freshness trigger basis), corpus_added_basis chunk|document|unknown.
    evidence_recency: str | None = None
    corpus_added_at: str | None = None
    corpus_added_basis: str | None = None
    source_authored_at: str | None = None
    source_authored_precision: str | None = None

    @model_serializer(mode="wrap")
    def _drop_absent_recency(self, handler):
        d = handler(self)
        for k in _FRESH_SOURCE_KEYS:
            if d.get(k) is None:
                d.pop(k, None)
        return d


_FRESH_SOURCE_KEYS = ("evidence_recency", "corpus_added_at", "corpus_added_basis", "source_authored_at",
                      "source_authored_precision")


class AskCandidate(BaseModel):
    """K2 reading-list candidate: allowlist only (extras such as snippet or abstract are dropped)."""
    n: int
    title: str
    url: str
    published: str | None = None
    identifier: str | None = None
    provider: str | None = None
    licence_decision: str | None = None
    acquisition: str | None = None
    representation: str | None = None
    retrieved_at: str | None = None
    span: str | None = None  # stored candidates only (absent for web candidates)
    span_id: str | None = None
    disposition: str | None = None
    canonical_source_id: int | None = None
    source_id: int | None = None
    admission_passed: bool | None = None
    licence: str | None = None  # declared licence id of a stored candidate
    metadata_licence: str | None = Field(default=None, max_length=64)  # P0-5B: index (metadata) licence id, never an article licence

    @model_serializer(mode="wrap")
    def _drop_absent_stored_fields(self, handler):
        d = handler(self)  # web candidates keep their exact shape: stored-only fields are omitted when absent
        for k in ("span", "span_id", "disposition", "canonical_source_id", "source_id", "admission_passed", "licence",
                  "metadata_licence"):
            if d.get(k) is None:
                d.pop(k, None)
        return d


class ExaminedDocument(BaseModel):
    """D-055: a full document VERA-DAR acquired, verified and inspected for this question (not all of it is evidence)."""
    document_id: str
    canonical_source_id: str | None = None
    title: str | None = None
    url: str | None = None
    doi: str | None = None
    published: str | None = None
    licence: str | None = None
    licence_decision: str | None = None
    text_sha256: str | None = None
    chars: int | None = None
    chunks_inspected: int | None = None
    candidate_spans: int | None = None
    admitted_spans: int | None = None


class AdmittedEvidence(BaseModel):
    """D-055: a passage of an examined document admitted into reasoning context (no passage text in the response)."""
    span_id: str
    document_id: str
    canonical_source_id: str | None = None
    title: str | None = None
    url: str | None = None
    licence: str | None = None
    span: str | None = None  # 'chars=<start>-<end>' in the verified full text
    relevance: float | None = None
    text_sha256: str | None = None


class CitationAttribution(BaseModel):
    """CC BY attribution from VERA-DAR's verified acquisition record (never generated)."""
    creator: str
    title: str
    source: str  # DOI URL, else the landing URL
    licence: str
    licence_url: str
    changes: str


class EvidenceCitation(BaseModel):
    """D-055: one claim -> admitted span binding, with a short verbatim quote only."""
    claim_id: int
    n: int
    span_id: str
    document_id: str
    canonical_source_id: str | None = None
    licence: str | None = None
    span: str | None = None
    verdict: str
    quote: str | None = None  # None when the per-response quote budget is spent (the span reference stays)
    quote_span: str | None = None
    attribution: CitationAttribution


class RecalledNote(BaseModel):
    kind: str = "discovery_note"
    note: str
    urls: list[str] = []
    saved_at: str | None = None


def _research_fields(result) -> dict:
    """FRESH-1023: additive top-level fields from contract.research (never raises; {} = fields absent)."""
    try:
        from vera import research_record
        return research_record.public_fields((result.contract or {}).get("research"))
    except Exception:  # noqa: BLE001
        return {}


class AskResponse(BaseModel):
    answer: str
    model: str
    tokens_used: int
    cost_usd: float
    sources: list[AskSource] = []  # item #79: what the answer was grounded in (no abstracts)
    traced_sources: list[AskSource] = []  # item #79: originals traced from links but with no free abstract (not used)
    recalled: list[dict] = []  # item #84: operator-confirmed findings relevant to this question (never the question)
    memory_written: int = 0  # deprecated constant 0 (writes are background); use memory_submitted
    memory_submitted: int = 0  # K1-M1: upper bound of candidates handed to the background writer for this response; not stored rows
    claim_check: dict | None = None  # item #79: how many cited claims were checked, kept, marked or removed
    result_kind: str = "unspecified"  # K2: only "answer" is an evidentiary answer
    # C-2: response-level disposition (answer | insufficient_evidence | corpus_unusable | reading_list | no_sources |
    # decline), derived by answer_contract.outcome_for. Differs from claim_check.outcome (the checker's internal
    # summary). Always serialized; additive.
    outcome: str = "insufficient_evidence"
    candidates: list[AskCandidate] = []  # K2: reading-list candidates (non-answer kinds only)
    contract: dict | None = None  # K2: {"layers": [...], "fallback": None | token}
    recalled_discovery: list[RecalledNote] = []  # K2 L2: earlier discovery notes (never evidence)
    discovery_note_submitted: int = 0  # K2 L2: 0 or 1
    # D-055 Option C (additive; absent from web-only behaviour): examined full documents are distinct from the admitted
    # passages that entered reasoning context; citations bind each kept claim to an admitted span.
    evidence_status: str | None = None
    corpus_source: dict | None = None  # {"kind": "db"|"bundle"}; only when the corpus loaded OK; code only
    examined_full_documents: list[ExaminedDocument] = []
    admitted_evidence: list[AdmittedEvidence] = []
    citations: list[EvidenceCitation] = []
    # FRESH-1023 (additive; absent unless the freshness/research step ran): which research triggers fired
    # ("insufficient" | "stale" | "freshness_unknown"), source-content freshness of the cited evidence, and the
    # research outcome (kind, concise user_text, complete, per-portal and per-candidate records).
    triggers: list[str] | None = None
    freshness: dict | None = None
    research_outcome: dict | None = None

    @model_serializer(mode="wrap")
    def _drop_absent_evidence_fields(self, handler):
        d = handler(self)  # web-only responses keep their exact K2 key set: the D-055 keys appear only with evidence
        for k in ("triggers", "freshness", "research_outcome"):
            if d.get(k) is None:
                d.pop(k, None)
        if d.get("evidence_status") is None:
            for k in ("evidence_status", "examined_full_documents", "admitted_evidence", "citations"):
                d.pop(k, None)
        if d.get("corpus_source") is None:
            d.pop("corpus_source", None)
        return d


class SyntheticFault(str, Enum):
    """Deliberately invalid AskResponse payloads, one violation each, used by the
    force_bad demo mode to prove the real response schema rejects malformed output."""

    non_numeric_tokens = "non_numeric_tokens"
    non_numeric_cost = "non_numeric_cost"
    null_answer = "null_answer"


_SYNTHETIC_FAULT_BASE = {"answer": "placeholder", "model": MODEL, "tokens_used": 0, "cost_usd": 0.0}

SYNTHETIC_FAULT_PAYLOADS: dict[SyntheticFault, dict] = {
    SyntheticFault.non_numeric_tokens: {**_SYNTHETIC_FAULT_BASE, "tokens_used": "not-a-number"},
    SyntheticFault.non_numeric_cost: {**_SYNTHETIC_FAULT_BASE, "cost_usd": "not-a-float"},
    SyntheticFault.null_answer: {**_SYNTHETIC_FAULT_BASE, "answer": None},
}

# force_bad always injects this one fault; not selectable via AskRequest — see
# ask-guardrail-design-prv.md FMEA #9 on not expanding the public request contract casually.
FORCE_BAD_FAULT = SyntheticFault.non_numeric_tokens


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # R-028: type only; a traceback/message can carry user input (e.g. httpx errors quote the request URL with the query words)
    logger.error("Unhandled exception on %s (%s)", request.url.path, type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


class CorpusDescription(BaseModel):
    source_count: int | None = None
    corpus_updated_at: str | None = None
    summary: str | None = None
    topics: list[str] = []
    summary_source: str
    summary_generated_at: str | None = None
    summary_model: str | None = None


@app.get("/corpus-summary", response_model=CorpusDescription, dependencies=[Depends(verify_api_key)])
def corpus_summary():
    """Item #83: description of VERA-DAR's own corpus, regenerated only when it changes. Behind VERA_API_KEY like /ask
    (the UI calls it server-side); no titles, URLs or abstracts in the response."""
    if CORPUS_DESCRIBER is None:
        return CorpusDescription(summary_source="disabled")
    try:
        return CorpusDescription(**CORPUS_DESCRIBER.current(generate_with_chain(ASK_CHAIN)))
    except Exception as exc:  # noqa: BLE001  (DB unreachable etc.: a generic state, never the error text)
        logger.warning("corpus description unavailable (%s)", type(exc).__name__)
        return CorpusDescription(summary_source="unavailable")


# --- Item #84 D6: operator-only review of memory writes ---------------------------
# Every route declares the owner-key dependency itself (per-call authorisation). The public VERA_API_KEY is not
# accepted; an unset, weak or reused owner key disables these routes (503). Responses carry no host or error text.
verify_owner_key = make_verify_owner_key(MEMORY_BLOCK)


class PendingSource(BaseModel):
    title: str = ""
    url: str = ""


class PendingClaim(BaseModel):
    claim_id: int
    claim_text: str
    created_at: str = ""
    state: str | None = None
    sources: list[PendingSource] = []


class PendingList(BaseModel):
    pending: list[PendingClaim]


class MemoryDecision(BaseModel):
    claim_id: int
    outcome: str


def _operator_memory():
    try:
        memory = memory_from_env(MEMORY_BLOCK)
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory unavailable for operator (%s)", type(exc).__name__)
        memory = None
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory store unavailable")
    return memory


def _operator_call(fn, *args):
    try:
        return fn(*args)
    except MemoryUnavailable as exc:
        raise HTTPException(status_code=503, detail="Memory store unavailable") from exc
    except Exception as exc:  # noqa: BLE001  (DB errors: generic detail, type only in the log)
        logger.warning("memory operator action failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Memory store unavailable") from exc


@app.get("/memory/pending", response_model=PendingList, dependencies=[Depends(verify_owner_key)])
def memory_pending():
    """Operator: memory rows awaiting review (pending, and legacy auto-written rows that are no longer recalled)."""
    memory = _operator_memory()
    return PendingList(pending=_operator_call(memory.list_pending))


def _decide(claim_id: int, action) -> MemoryDecision:
    outcome = _operator_call(action, claim_id)
    if outcome == NOT_FOUND:
        raise HTTPException(status_code=404, detail="No memory claim with that id")
    if outcome not in (CONFIRMED, REJECTED):
        raise HTTPException(status_code=409, detail="That claim is not in a state that allows this action")
    return MemoryDecision(claim_id=claim_id, outcome=outcome)


@app.post("/memory/{claim_id}/confirm", response_model=MemoryDecision, dependencies=[Depends(verify_owner_key)])
def memory_confirm(claim_id: int):
    """Operator: a pending (or legacy) claim becomes recallable."""
    return _decide(claim_id, _operator_memory().confirm)


@app.post("/memory/{claim_id}/reject", response_model=MemoryDecision, dependencies=[Depends(verify_owner_key)])
def memory_reject(claim_id: int):
    """Operator: a pending, legacy or confirmed claim is never recalled and never re-queued."""
    return _decide(claim_id, _operator_memory().reject)


_DAR_PUBLIC_STATES = frozenset({"ready", "degraded", "unavailable", "disabled", "unknown"})


def _dar_public_state(readiness) -> str:
    """Coarse public enum from the readiness dict. Anything unrecognised, null or uncached maps to "unknown"."""
    try:
        state = readiness.get("dar") if isinstance(readiness, dict) else None
    except Exception:  # noqa: BLE001
        state = None
    return state if isinstance(state, str) and state in _DAR_PUBLIC_STATES else "unknown"


@app.get("/health")
def health():
    """Public API liveness. "status" never depends on the DB; "dar" is a coarse enum only (no ids, codes or counts)."""
    try:
        readiness = dar_health.cached_readiness()
    except Exception:  # noqa: BLE001
        readiness = None
    out = {"status": "ok", "dar": _dar_public_state(readiness)}
    if ACQ_CONFIG_REFUSED:
        out["acquisition"] = "off_config_refused"  # no paths or values
    return out


@app.get("/health/dar", dependencies=[Depends(verify_api_key)])
def health_dar():
    """Full DAR readiness (migrations, codes, corpus counts). Behind the same API key as /corpus-summary."""
    try:
        from vera import rate_limit  # RATE-1022 E1: authenticated route only; portal names and counters, no hosts/keys
        return {**dar_health.cached_readiness(), "fulldoc": dar_health._fulldoc_status(),
                "rate_limits": rate_limit.health_aggregates()}
    except Exception:  # noqa: BLE001
        return {"dar": None, "reason_code": "health_error"}


@app.get("/trace/public")
def trace_public():
    """TRACEVIZ-1022: public Trace Eval aggregates (no key). Allowlisted fields only; see vera/trace_viz.py."""
    try:
        return trace_viz.public_view()
    except Exception as exc:  # noqa: BLE001  (generic state; the type only in the log)
        logger.warning("trace public view failed (%s)", type(exc).__name__)
        return {"schema": trace_viz.SCHEMA, "status": "error", "stages": [], "run": None, "check_validation": None}


@app.get("/trace/private", dependencies=[Depends(trace_viz.verify_viz_key)])
def trace_private():
    """TRACEVIZ-1022: per-trace detail. Needs X-Viz-Key == VERA_TRACE_VIZ_KEY; unset key disables the route (403)."""
    try:
        return trace_viz.private_view()
    except Exception as exc:  # noqa: BLE001
        logger.warning("trace private view failed (%s)", type(exc).__name__)
        return {"available": False, "traces": []}


@app.get("/version")
def version():
    """Deployed commit (Render's RENDER_GIT_COMMIT) so the live build is verifiable. No host, no secrets."""
    commit = deployed_commit()
    return {"commit": commit, "short": short_commit(commit)}


def _memory_submitted(memory, result) -> int:
    """K1-M1: upper bound of supported claims handed to the background writer. Never raises; 0 when no write is possible."""
    try:
        if result.result_kind != "answer" or not result.accepted or memory is None or memory.reader is None or memory.store is None \
                or result.evidence is not None:  # D-055: the memory gate is not built for full-document passages
            return 0
        n = sum(1 for r in result.claims if r.get("verdict") == VERDICT_SUPPORTED)
        return max(0, min(n, int(memory.cfg["max_writes_per_answer"])))
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory_submitted failed (%s)", type(exc).__name__)
        return 0


def _discovery_note_wanted(memory, result) -> bool:
    """K2.I6 item 5: all conditions for submitting one discovery note. Never raises."""
    try:
        return bool(result.accepted and result.result_kind in ("reading_list", "discovery") and result.candidates
                    and "L2" in answer_contract.layers() and answer_contract.write_on_l0()
                    and memory is not None and memory.reader is not None and memory.store is not None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("discovery note check failed (%s)", type(exc).__name__)
        return False


def _acquire(question: str, trigger: str) -> None:
    """WED-ACQ2 (P1.7): hand the question to the bounded background acquisition queue. O(1), never blocks and never
    fails /ask; the question is held in memory only (vera/auto_acquire.py). Logs a class name at most."""
    try:
        from vera import auto_acquire
        auto_acquire.enqueue(question, trigger)
    except Exception as exc:  # noqa: BLE001
        logger.warning("acquisition enqueue failed (%s)", type(exc).__name__)


def _after_response(admit_args, memory, result, question: str, discovery=None) -> None:
    """Background: corpus admission first, then memory writes (never raises, never affects the response)."""
    try:
        if admit_args:
            admit_to_corpus(*admit_args)
        if memory is not None and result.claims and result.result_kind == "answer" and result.evidence is None:
            memory.write_claims(result.claims, list(result.sources), question)
        if memory is not None and discovery:
            memory.write_discovery_note(list(result.candidates), question)
    except Exception as exc:  # noqa: BLE001
        logger.warning("post-response task failed (%s)", type(exc).__name__)


def _public_evidence(result) -> dict:
    """D-055 response fields from the EvidenceBundle (allowlisted; span text never leaves except capped quotes)."""
    if result.evidence is None:
        return {}
    from vera.fulldoc_answer import public_evidence

    return public_evidence(result.evidence, result.citations if result.result_kind == "answer" else ())


def _fulldoc_sources(result) -> list[dict]:
    """Numbered admitted passages as AskSource rows: provenance only (no passage text); ids are canonical strings, so
    the integer DB id fields stay absent."""
    decided = {d.get("document_id"): d.get("licence_decision") for d in result.evidence.examined_full_documents}
    out = []
    for s in result.sources:
        out.append({"n": s["n"], "title": s.get("title") or "", "url": s.get("url") or "", "published": s.get("published"),
                    "source_type": s.get("source_type"), "provider": "full_document",
                    "identifier": s.get("canonical_source_id"), "retrieved_at": None,
                    "licence_decision": decided.get(s.get("document_id")), "span": s.get("span"), "span_id": s.get("span_id"),
                    "admission_passed": True, "acquisition": answer_contract.ACQ_ACQUIRED,
                    "disposition": "admitted", "licence": s.get("licence"),
                    **{k: s[k] for k in _FRESH_SOURCE_KEYS if s.get(k)}})
    return out


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key), Depends(enforce_caps)])  # W2 caps (public mode)
def ask(req: AskRequest, request: Request, background_tasks: BackgroundTasks):
    """PROG-1027 wrapper: today's handler is `_ask_impl`. Progress records are opt-in (flag + both headers); a
    request without them runs exactly as before. The single `done` event is emitted only here (INTERFACE-CONTRACT s2)."""
    rec = token = None
    try:
        rec = progress.open_record(request)  # inside the try; never raises
        if rec is not None:
            token = progress.SINK.set(rec)
            progress.emit(progress.Stage.RECEIVED, progress.State.DONE)
        resp = _ask_impl(req, request, background_tasks)
        if rec is not None:
            st = progress.outcome(resp)
            progress.emit(progress.Stage.DONE, st, **progress.terminal_counts(st, rec))
        return resp
    except BaseException:
        if rec is not None:
            progress.emit(progress.Stage.DONE, progress.State.ERROR)
        raise
    finally:
        if rec is not None:
            rec.close()
        if token is not None:
            progress.SINK.reset(token)


def _ask_impl(req: AskRequest, request: Request, background_tasks: BackgroundTasks):
    from vera.ask_service import REQUEST_BUDGET, request_budget
    budget = request_budget()  # CMF-1022: the request-level deadline starts ONCE, here, at handler entry
    if req.mode is DemoMode.force_bad:
        # Guardrail demo: inject one identified synthetic fault, rejected by real
        # schema validation. No OpenAI call is made — the demo is free and deterministic.
        fault = FORCE_BAD_FAULT
        try:
            AskResponse(**SYNTHETIC_FAULT_PAYLOADS[fault])
        except ValidationError as exc:
            logger.info(
                "force_bad invoked: synthetic fault '%s' rejected by schema validation: %s",
                fault.value, exc.errors(),
            )
            raise HTTPException(
                status_code=500,
                detail="Guardrail demo: synthetic malformed output was rejected by schema validation.",
            ) from exc

    key_status = classify_openai_api_key(os.getenv("OPENAI_API_KEY")) if OPENAI_FALLBACK else "ok"
    if key_status != "ok":
        logger.error("Rejected /ask: OPENAI_API_KEY is %s", key_status)
        raise HTTPException(status_code=500, detail=OPENAI_KEY_ERROR_DETAIL[key_status])

    if OPENAI_FALLBACK and CURRENT_PRICING is None:
        logger.error("Rejected /ask: no pricing record for model %s", MODEL)
        raise HTTPException(
            status_code=500,
            detail=f"Server misconfigured: no pricing record for model '{MODEL}' in config/model-pricing.json",
        )

    try:
        # scope router first (item #72 W1); free providers first, OpenAI last (item #77)
        grounding, cache = None, {}
        if GROUNDING is not None:

            def _restricted_key():  # RL-REL-1021-C: keyed by the active profile (AND vs OR never share); derived once
                import json as _json

                from vera.search_providers import OPENALEX_PROFILE

                prof = OPENALEX_PROFILE.get()
                return ("r", None if prof is None else _json.dumps(prof, sort_keys=True, default=str))

            def sources_fn(restricted=False):  # one search per request, reused if the provider chain falls through
                if restricted:  # FIX1-1021: off-topic reading list: scholarly providers only, no link following, no stored branch
                    key = _restricted_key()
                    if key not in cache:
                        from vera.scope_router import offtopic_providers

                        rcfg = {**GROUNDING, "providers": offtopic_providers(), "resolve": {"enabled": False}}
                        cache[key], cache[("u",) + key] = retrieve_all(req.question, rcfg)
                    return cache[key]
                if "s" not in cache:
                    from vera.dar_runtime import get_corpus  # lazy; flag-gated, returns None when DAR is off

                    dar = get_corpus()
                    cache["s"], cache["unused"] = retrieve_all(req.question, GROUNDING,
                                                               **({"dar_corpus": dar} if dar is not None else {}))
                return cache["s"]

            def _unused_fn(restricted=False):  # abstract-less originals of the same cached search (acquisition candidates only)
                if restricted:  # empty by design today: the off-topic search runs with resolve off
                    return cache.get(("u",) + _restricted_key()) or []
                return cache.get("unused") or []

            sources_fn.unused = _unused_fn
            grounding = (sources_fn, GROUNDING)
        _tok = REQUEST_BUDGET.set(budget)  # shared by every provider tried in the chain; never restarted
        try:
            result, answered_by = answer_via_chain(req.question, ASK_CHAIN, grounding)
        finally:
            REQUEST_BUDGET.reset(_tok)
    except RuntimeError as exc:
        # empty chain (no free provider key) or the default-client latch: provider-neutral, never a paid call
        if not (isinstance(exc, OpenAIFallbackDisabled) or str(exc).startswith("no provider available")):
            raise
        logger.error("/ask has no usable answering provider (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="No answering provider is configured.") from exc
    except AuthenticationError as exc:
        if not OPENAI_FALLBACK:
            logger.error("Answering provider rejected its key (%s)", type(exc).__name__)
            raise HTTPException(status_code=503, detail="The answering provider is misconfigured on the server.") from exc
        raise HTTPException(
            status_code=401,
            detail=(
                "OpenAI rejected the API key. Check that OPENAI_API_KEY in .env is current "
                "and belongs to an active account."
            ),
        ) from exc
    except RateLimitError as exc:
        if not OPENAI_FALLBACK:
            raise HTTPException(status_code=429, detail="The free answering provider is rate-limited; retry shortly.") from exc
        quota_markers = {"insufficient_quota", "credit_balance_exhausted"}
        if getattr(exc, "code", None) in quota_markers or getattr(exc, "type", None) in quota_markers:
            raise HTTPException(
                status_code=402,
                detail=(
                    "The OpenAI account for this API key has no remaining credit, so the "
                    "request was not answered. Add a payment method or credits at "
                    "https://platform.openai.com/settings/organization/billing/ and retry."
                ),
            ) from exc
        raise HTTPException(
            status_code=429,
            detail="OpenAI rate limit hit (too many requests). Wait a few seconds and retry.",
        ) from exc
    except OpenAIError as exc:
        logger.error("Answering provider request failed (%s, status %s, code %s)", type(exc).__name__,
                     getattr(exc, "status_code", None), getattr(exc, "code", None))  # never the raw provider text
        if not OPENAI_FALLBACK:
            raise HTTPException(status_code=502, detail="The answering provider failed; please retry.") from exc
        raise HTTPException(status_code=502, detail="OpenAI request failed. Please try again.") from exc

    record_request_cost(request, result.cost_usd)
    if getattr(result, "acquire_trigger", None):  # WED-ACQ2: once per request, only after the chain succeeded
        _acquire(req.question, result.acquire_trigger)
    if ((result.contract or {}).get("inline_acquisition") or {}).get("stored"):  # DAR-INLINE-1021: after the response
        from vera.inline_acquire import refresh_after_response
        background_tasks.add_task(refresh_after_response)  # stored documents join the served acquired corpus
    fulldoc = result.evidence is not None and result.result_kind == "answer"  # D-055: sources are admitted passages
    store = store_from_env(CORPUS) if result.sources and not fulldoc else None  # never re-admitted as web candidates
    memory = None
    recalled, memory_written = [], 0  # memory_written stays 0: writes are asynchronous (background, after admission)
    if result.accepted:  # item #84: recall and write only for in-scope questions; errors never fail /ask
        try:
            memory = memory_from_env(MEMORY_BLOCK)
            if memory is not None:
                recalled = memory.recall(req.question)  # recall is synchronous
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory failed (%s)", type(exc).__name__)
    memory_submitted = _memory_submitted(memory, result)
    is_answer = result.result_kind == "answer"
    discovery = _discovery_note_wanted(memory, result)
    recalled_discovery = []
    if result.accepted:  # computed after answering: a note is never an input to the answer (D8c)
        try:
            if "L2" in answer_contract.layers() and memory is not None:
                recalled_discovery = [n for n in memory.recall_discovery(req.question)
                                      if isinstance(n, dict) and isinstance(n.get("note"), str)]
        except Exception as exc:  # noqa: BLE001
            logger.warning("discovery recall failed (%s)", type(exc).__name__)
    # an empty chain cannot have answered (it 503s above); the guard stays only for a stubbed answer_via_chain
    admit_args = (req.question, list(result.sources), llm_call_for(ASK_CHAIN[0] if ASK_CHAIN else None), CORPUS, store) \
        if store is not None else None
    if admit_args or (memory is not None and ((result.claims and is_answer and not fulldoc) or discovery)):
        # item #79 phase C + item #84: after the response, never delaying it. Memory writes run AFTER corpus
        # admission so first-time sources are already in the corpus when their claims are saved.
        background_tasks.add_task(_after_response, admit_args, memory, result, req.question, discovery=discovery)
    evidence = _public_evidence(result)
    return AskResponse(
        **evidence,
        recalled=recalled,
        memory_written=memory_written,
        memory_submitted=memory_submitted,
        answer=result.answer,
        model=answered_by,
        tokens_used=result.tokens_used,
        cost_usd=result.cost_usd,
        # C-F1/C-F2: for every non-answer kind the public sources, traced sources and claim_check are empty;
        # result.sources stays internal (admission above).
        sources=(_fulldoc_sources(result) if fulldoc else public_sources(list(result.sources))) if is_answer else [],
        claim_check=result.claim_check if is_answer else None,
        traced_sources=[{"n": i, **{k: t.get(k) for k in ("title", "url", "published")}, "source_type": "traced"}
                        for i, t in enumerate(cache.get("unused") or [], 1)] if is_answer else [],
        result_kind=result.result_kind,
        outcome=answer_contract.outcome_for(result.result_kind, (result.contract or {}).get("fallback"),
                                     len(result.candidates or ())),
        candidates=[] if is_answer else list(result.candidates),
        contract=result.contract,
        recalled_discovery=recalled_discovery,
        discovery_note_submitted=1 if discovery else 0,
        **_research_fields(result),
    )


def _parse_after(raw) -> int | None:
    """None unless absent (-> 0) or a plain decimal integer >= 0."""
    if raw is None:
        return 0
    if isinstance(raw, str) and raw.isascii() and raw.isdecimal() and len(raw) <= 9:
        return int(raw)
    return None


@app.get("/ask/progress/{rid}", dependencies=[Depends(verify_api_key)])
async def ask_progress(request: Request, rid: str):
    """PROG-1027: events of one Ask for the browser session that sent it. No Path/Query/Header constraints (they would
    answer 422); every refusal is the identical 404 of an unmatched route."""
    after = _parse_after(request.query_params.get("after"))
    if not progress.enabled() or after is None:
        raise HTTPException(status_code=404)
    rec = progress.STORE.get_authorized(rid, request.headers.get("x-vera-session"))
    if rec is None:
        raise HTTPException(status_code=404)
    return {"events": rec.events_after(after, limit=progress.CFG.max_events_per_poll), "closed": rec.closed}
