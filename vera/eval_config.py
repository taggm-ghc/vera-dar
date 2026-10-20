"""One validated loader for the VERA-DAR eval configuration (goal #2, plan vera-plan-config-single-source).

Two files, never merged on disk:
  config/vera_eval_freeze.json   FROZEN and IMMUTABLE in this incarnation (R1, D1 option 2). Read-only here:
                                 question, requirements, objections and the seven success_criteria.
  config/vera_eval_run.json      VERSIONED companion: freeze pin, scoring parameter (fabrication rule),
                                 budget, generator model, judge pool/selection settings, model-to-family
                                 map, and (later) pipeline constants.

Rules enforced here (hardcoded-data standard, rules 4 and 6):
  * every missing / null / mistyped / out-of-range value is listed in ONE verbose EvalConfigError;
  * the freeze file must match the pin in the run config (sha256 of its newline-normalised bytes AND the
    16-hex requirements hash); there is no bypass flag;
  * a model whose family is unknown fails closed (family_for) unless an explicit, disclosed override is given.

PIN HASH CHOICE (plan 4.1a limit, checked 2026-10-02): the repo has no .gitattributes, global and local
core.autocrlf are unset, and the freeze file holds no CR bytes. A raw-bytes hash would still turn into a false
alarm on a machine that checks files out with CRLF, so the pin hash is computed over the file bytes with CRLF
normalised to LF (and a UTF-8 BOM removed). Today the normalised hash equals the raw hash. The requirements
hash16 is computed over PARSED content and is immune to line endings by construction.

Design: no module-level cached singleton (callers pass `config` or call load_eval_config()); this module must
never be imported by vera.ask_service (a bad eval file must not take down /ask; enforced by a test).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlsplit

_ROOT = Path(__file__).resolve().parent.parent
# Why: the two files named by R1 (D1 option 2); per-environment overrides are passed as arguments, not env.
FREEZE_PATH = _ROOT / "config" / "vera_eval_freeze.json"
RUN_PATH = _ROOT / "config" / "vera_eval_run.json"
MODEL_SELECTION_PATH = _ROOT / "config" / "model-selection.json"
# Why: the run config's own schema versions this loader understands (the freeze file has no schema_version).
SUPPORTED_RUN_SCHEMAS = frozenset({1})
# Why: R1 froze the file with this status (2026-10-01); anything else means it is not the frozen artefact.
FREEZE_STATUS = "FROZEN"
RUN_STATUS = "VERSIONED"
# Why: R1 decision A5/K: at least 1 requirement and K>=4 objections (previous load_freeze rule, unchanged).
MIN_REQUIREMENTS, MIN_OBJECTIONS = 1, 4

# Schema (types and ranges), NOT values: the values live only in the JSON files.
# key -> (kind, low, high)
_CRITERIA = {"relevance_threshold": ("float", 0, 1), "relevance_improvement_pp": ("float", 0, 100),
             "grounding_threshold": ("float", 0, 1), "reasoning_threshold": ("int", 1, 5),
             "reasoning_objections_improvement": ("int", 0, 100), "auditability_threshold": ("int", 1, 5),
             "cost_latency_threshold": ("int", 1, 5)}
_RUN_SCORING = {"grounding_max_fabricated": ("int", 0, 10_000)}
_RUN_KEYS = {"status", "schema_version", "run_config_version", "created_at", "created_by", "freeze_pin",
             "scoring", "budget", "generator", "judge", "model_families", "pipeline", "search", "notes"}
_JUDGE_KEYS = {"pool", "selection", "family_patterns", "forbidden_env_overrides"}
# Why (review F7): judge.pool.providers is validated at load against the providers vera.m6.judge has facts for
# (key env, base URL, prices). Kept here, not imported, because judge.py imports this module; a test pins this
# tuple to judge._PROVIDERS so the two cannot drift.
SUPPORTED_JUDGE_PROVIDERS = ("groq", "openai")
# Why (review F6): code-side MINIMUM safety set. The run config may add names/patterns but can never remove
# these: the four env names are the overrides that would change the judge model, price or family rule without
# being recorded in the selection; "gpt" and "openai" are the patterns that keep the generator's own family
# out of the judge pool. An empty or thinned list would silently disable the guard.
REQUIRED_FORBIDDEN_ENV = ("VERA_JUDGE_ALLOW_SAME_FAMILY", "VERA_JUDGE_GROQ_MODEL", "VERA_JUDGE_PRICE_IN",
                          "VERA_JUDGE_PRICE_OUT")
REQUIRED_REFUSE_OPENAI = ("gpt", "openai")


class EvalConfigError(RuntimeError):
    """Message is the full report of every problem found; nothing was scored."""


@dataclass(frozen=True)
class SuccessCriteria:
    relevance_min: float
    relevance_delta: float  # frozen file stores percentage points; stored here as a fraction (15 -> 0.15)
    grounding_min: float
    grounding_max_fabricated: int  # from the RUN config (scoring parameter), not the freeze file
    reasoning_min: int
    reasoning_delta: int
    audit_min: int
    cost_min: int


@dataclass(frozen=True)
class Budget:
    cost_usd: float
    latency_s: float


@dataclass(frozen=True)
class JudgeSettings:
    providers: tuple[str, ...]
    rule: str
    family_preference: tuple[str, ...]
    max_canary_attempts: int
    canary_max_tokens: int
    listing_timeout_s: float
    refuse_openai_patterns: tuple[str, ...]
    non_chat_patterns: tuple[str, ...]
    generic_patterns: tuple[tuple[str, str], ...]
    forbidden_env_overrides: tuple[str, ...]


@dataclass(frozen=True)
class FamilyResolution:
    model: str
    family: str
    source: str  # "map" or "override"


@dataclass(frozen=True)
class EvalConfig:
    freeze_path: str
    run_path: str
    schema_version: int
    run_config_version: str
    freeze_file_sha256: str
    run_file_sha256: str
    requirements_hash16: str
    question: str
    requirements: tuple
    objections: tuple
    success: SuccessCriteria
    budget: Budget
    generator_model: str
    judge: JudgeSettings
    model_families: Mapping[str, Mapping]
    pipeline: Mapping
    search: Mapping
    raw: Mapping  # the FREEZE file's parsed content only (read-only view); the run config is not merged in
    run_raw: Mapping  # the RUN config's parsed content (read-only view), for the fingerprint


# ----------------------------------------------------------------------------- hashing
def normalise_newlines(data: bytes) -> bytes:
    """CRLF -> LF and a leading UTF-8 BOM removed, so a checkout with other line endings hashes the same."""
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.replace(b"\r\n", b"\n")


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(normalise_newlines(data)).hexdigest()


def requirements_hash16(freeze: Mapping) -> str:
    """The existing freeze_hash() formula (vera.m6.eval_rubric.freeze_hash delegates here): sha256 over
    requirements + objections only, truncated to 16 hex characters. UNCHANGED; the wider run fingerprint is
    a separate field (vera.m6.fingerprint)."""
    payload = {"requirements": freeze.get("requirements", []), "objections": freeze.get("objections", [])}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- helpers
def _num(problems: list, label: str, v, kind: str, lo, hi) -> None:
    if v is None:
        problems.append(f"{label}: missing or null")
    elif isinstance(v, bool) or not isinstance(v, (int, float)):
        problems.append(f"{label}={v!r}: not a number")
    elif isinstance(v, float) and not math.isfinite(v):
        problems.append(f"{label}={v!r}: not finite")
    elif kind == "int" and v != int(v):
        problems.append(f"{label}={v!r}: must be a whole number")
    elif not lo <= v <= hi:
        problems.append(f"{label}={v!r}: outside [{lo}, {hi}]")


def _pos_num(problems: list, label: str, v) -> None:
    if v is None:
        problems.append(f"{label}: missing or null")
    elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        problems.append(f"{label}={v!r}: must be a finite number > 0")


def _str_list(problems: list, label: str, v, *, min_len: int = 1) -> list:
    if not isinstance(v, list) or len(v) < min_len or not all(isinstance(x, str) and x.strip() for x in v):
        problems.append(f"{label}={v!r}: must be a list of >= {min_len} non-empty string(s)")
        return []
    return v


def _read_json(path, role: str) -> tuple[bytes, dict]:
    try:
        raw = Path(path).read_bytes()
        d = json.loads(raw)
    except (OSError, ValueError) as e:
        raise EvalConfigError(f"{path}: cannot read/parse the {role} ({type(e).__name__}: {e}). "
                              "Owner: R1. Nothing was scored.") from None
    if not isinstance(d, dict):
        raise EvalConfigError(f"{path}: the {role} top level must be a JSON object, got {type(d).__name__}.")
    return raw, d


def _check_freeze(problems: list, d: dict, path) -> None:
    if d.get("status") != FREEZE_STATUS:
        problems.append(f"{path}: status={d.get('status')!r}, need {FREEZE_STATUS!r}")
    if not isinstance(d.get("question"), str) or not d["question"].strip():
        problems.append("question: missing, empty or not a string")
    for key, minimum in (("requirements", MIN_REQUIREMENTS), ("objections", MIN_OBJECTIONS)):
        items = d.get(key)
        if not isinstance(items, list) or len(items) < minimum:
            problems.append(f"{key}: need a list of >= {minimum} item(s), got "
                            f"{len(items) if isinstance(items, list) else repr(items)}")
            continue
        ids = []
        for i, it in enumerate(items):
            if not isinstance(it, dict) or not isinstance(it.get("id"), str) or not it["id"].strip():
                problems.append(f"{key}[{i}]: needs an object with a non-empty string id")
            else:
                ids.append(it["id"])
        if len(set(ids)) != len(ids):
            problems.append(f"{key}: duplicate ids {sorted({x for x in ids if ids.count(x) > 1})}")
    sc = d.get("success_criteria")
    if not isinstance(sc, dict):
        problems.append("success_criteria: missing or not an object")
        return
    for k in sorted(set(sc) - set(_CRITERIA)):
        problems.append(f"success_criteria.{k}: unknown key (typo?)")
    for k, (kind, lo, hi) in _CRITERIA.items():
        _num(problems, f"success_criteria.{k}", sc.get(k), kind, lo, hi)


def _check_run(problems: list, r: dict) -> dict:
    """Validate the run config; returns the parsed judge-settings kwargs (empty when invalid)."""
    for k in sorted(set(r) - _RUN_KEYS):
        problems.append(f"run config: unknown top-level key {k!r} (typo? success_criteria must stay in the "
                        "freeze file, never here)")
    if r.get("status") != RUN_STATUS:
        problems.append(f"run config status={r.get('status')!r}, need {RUN_STATUS!r}")
    if r.get("schema_version") not in SUPPORTED_RUN_SCHEMAS or isinstance(r.get("schema_version"), bool):
        problems.append(f"run config schema_version={r.get('schema_version')!r}, supported "
                        f"{sorted(SUPPORTED_RUN_SCHEMAS)}")
    if not isinstance(r.get("run_config_version"), str) or not r["run_config_version"].strip():
        problems.append("run config run_config_version: missing or empty (every change is a new version)")
    sc = r.get("scoring")
    if not isinstance(sc, dict):
        problems.append("scoring: missing or not an object")
    else:
        for k in sorted(set(sc) - set(_RUN_SCORING)):
            problems.append(f"scoring.{k}: unknown key (typo?)")
        for k, (kind, lo, hi) in _RUN_SCORING.items():
            _num(problems, f"scoring.{k}", sc.get(k), kind, lo, hi)
    b = r.get("budget")
    if not isinstance(b, dict):
        problems.append("budget: missing or not an object")
    else:
        _pos_num(problems, "budget.cost_usd", b.get("cost_usd"))
        _pos_num(problems, "budget.latency_s", b.get("latency_s"))
    g = r.get("generator")
    if not isinstance(g, dict) or not isinstance(g.get("model"), str) or not g["model"].strip():
        problems.append("generator.model: missing or not a non-empty string")
    fams = r.get("model_families")
    if not isinstance(fams, dict) or not fams:
        problems.append("model_families: missing, empty or not an object")
    else:
        for m, e in fams.items():
            if not isinstance(e, dict):
                problems.append(f"model_families[{m!r}]: not an object")
                continue
            for need in ("family", "source"):
                if not isinstance(e.get(need), str) or not e[need].strip():
                    problems.append(f"model_families[{m!r}].{need}: missing or empty (every entry needs a "
                                    "family and a source)")
        if isinstance(g, dict) and isinstance(g.get("model"), str) and g["model"] not in fams:
            problems.append(f"generator.model={g['model']!r} is not in model_families (its family is unknown, "
                            "so the self-grading guard could not be checked)")
    if not isinstance(r.get("pipeline"), dict):
        problems.append("pipeline: missing or not an object (use {} until pipeline constants move here)")
    _check_search(problems, r.get("search"))
    pin = r.get("freeze_pin")
    if not isinstance(pin, dict):
        problems.append("freeze_pin: missing or not an object")
    else:
        for need in ("file_sha256", "requirements_hash16"):
            if not isinstance(pin.get(need), str) or not pin[need].strip():
                problems.append(f"freeze_pin.{need}: missing or empty")
    return _check_judge(problems, r.get("judge"))


def _check_search(problems: list, s) -> None:
    """search block (M2 redo). Provider names are checked against vera.search_providers at chain build."""
    if not isinstance(s, dict):
        problems.append("search: missing or not an object")
        return
    if not isinstance(s.get("providers"), list) or not s["providers"] or not all(isinstance(x, str) for x in s["providers"]):
        problems.append("search.providers: need a non-empty list of provider names")
    _num(problems, "search.min_candidates", s.get("min_candidates"), "int", 1, 100)
    fb = s.get("fallback_seed")
    if not isinstance(fb, dict) or set(fb) != {"path", "sha256"}:
        problems.append("search.fallback_seed: need an object with exactly path and sha256 (both null = no fallback)")
    elif (fb["path"] is None) != (fb["sha256"] is None) or any(v is not None and not (isinstance(v, str) and v.strip()) for v in fb.values()):
        problems.append("search.fallback_seed: path and sha256 must be both null or both non-empty strings (frozen by hash)")
    if s.get("mode") not in ("fan_out", "first_sufficient"):
        problems.append("search.mode: fan_out or first_sufficient")
    _num(problems, "search.rrf_k", s.get("rrf_k"), "int", 1, 1000)
    _check_query_terms(problems, s.get("query_terms"))
    # per-provider validators (G8): arXiv stays required (the original route); the OpenAlex block is required
    # exactly when the provider is listed (or the block is present).
    listed = s["providers"] if isinstance(s.get("providers"), list) else []
    _check_arxiv(problems, s.get("arxiv"))
    if "openalex" in listed or "openalex" in s:
        _check_openalex(problems, s.get("openalex"))
    if "doaj" in listed or "doaj" in s:
        _check_doaj(problems, s.get("doaj"))
    if "duckduckgo_lite" in listed or "duckduckgo_lite" in s:
        _check_duckduckgo_lite(problems, s.get("duckduckgo_lite"))
    _check_max_share(problems, s.get("max_share"))


def _check_query_terms(problems: list, t) -> None:
    if not isinstance(t, dict):
        problems.append("search.query_terms: missing or not an object")
        return
    _num(problems, "search.query_terms.min_term_chars", t.get("min_term_chars"), "int", 1, 10)
    _num(problems, "search.query_terms.max_extra_terms", t.get("max_extra_terms"), "int", 1, 30)
    if "relax_on_zero" in t and not isinstance(t["relax_on_zero"], bool):
        problems.append("search.query_terms.relax_on_zero: need true or false (default true when absent)")
    for k in ("stopwords", "meta_words", "core_phrases"):
        v = t.get(k)
        if (not isinstance(v, list) or not v
                or not all(isinstance(x, str) and x.strip() and x == x.lower() and x == x.strip() for x in v)):
            problems.append(f"search.query_terms.{k}: need a non-empty list of lowercase, trimmed strings")


_SOURCE_CLASSES = {"news", "web_portal", "independent_research", "vendor_claim"}
_HOST_RE = re.compile(r"^[a-z0-9]([a-z0-9\-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9\-]*[a-z0-9])?)+$")


def _check_max_share(problems: list, m) -> None:
    if m is None:
        return  # optional: the code default caps duckduckgo_lite at 2
    if not isinstance(m, dict):
        problems.append("search.max_share: need an object {provider_name: max items in the final top-n}")
        return
    for k, v in m.items():
        if not isinstance(k, str) or not k.strip():
            problems.append(f"search.max_share: bad provider name {k!r}")
        _num(problems, f"search.max_share.{k}", v, "int", 0, 100)


def _check_duckduckgo_lite(problems: list, d) -> None:
    if not isinstance(d, dict):
        problems.append("search.duckduckgo_lite: missing or not an object")
        return
    if not isinstance(d.get("enabled"), bool):
        problems.append("search.duckduckgo_lite.enabled: must be true or false")
    if not isinstance(d.get("endpoint"), str) or not d["endpoint"].startswith("https://"):
        problems.append("search.duckduckgo_lite.endpoint: need an https URL")
    elif (urlsplit(d["endpoint"]).hostname or "") != "lite.duckduckgo.com":
        problems.append("search.duckduckgo_lite.endpoint: host must be lite.duckduckgo.com")
    if not isinstance(d.get("region"), str) or not re.fullmatch(r"[a-z]{2}-[a-z]{2}", d["region"]):
        problems.append("search.duckduckgo_lite.region: need a kl code such as us-en or wt-wt")
    ua = d.get("user_agent")
    if not isinstance(ua, str) or not ua.strip() or "discovery" not in ua.lower():
        problems.append("search.duckduckgo_lite.user_agent: need an identifying User-Agent that says it is discovery only")
    _num(problems, "search.duckduckgo_lite.min_interval_s", d.get("min_interval_s"), "float", 5.0, 3600)
    if d.get("max_concurrency") != 1 or isinstance(d.get("max_concurrency"), bool):
        problems.append("search.duckduckgo_lite.max_concurrency: must be 1 (single connection)")
    _num(problems, "search.duckduckgo_lite.daily_query_cap", d.get("daily_query_cap"), "int", 1, 500)
    _num(problems, "search.duckduckgo_lite.cache_ttl_hours", d.get("cache_ttl_hours"), "float", 0.01, 24 * 365)
    cd = d.get("cache_dir")
    if cd is not None and not (isinstance(cd, str) and cd.strip() and not cd.startswith(("/", "~")) and ".." not in cd.split("/")
                               and cd.startswith("tmp/")):
        problems.append("search.duckduckgo_lite.cache_dir: must be a repo-relative path under tmp/ (or null = no cache)")
    _num(problems, "search.duckduckgo_lite.max_response_bytes", d.get("max_response_bytes"), "int", 1024, 8 * 1024 * 1024)
    _num(problems, "search.duckduckgo_lite.max_query_chars", d.get("max_query_chars"), "int", 100, 2000)
    _num(problems, "search.duckduckgo_lite.max_queries_per_question", d.get("max_queries_per_question"), "int", 1, 4)
    g = d.get("site_groups")
    if not isinstance(g, dict) or not g or not set(g) <= {"news", "portals"}:
        problems.append("search.duckduckgo_lite.site_groups: need an object with keys from news, portals")
    else:
        for name, sites in g.items():
            if not isinstance(sites, list) or not all(isinstance(x, str) and _HOST_RE.match(x) for x in sites):
                problems.append(f"search.duckduckgo_lite.site_groups.{name}: need a list of lowercase host names")
        if not any(isinstance(v, list) and v for v in g.values()):
            problems.append("search.duckduckgo_lite.site_groups: at least one group needs a site")
    sc = d.get("source_class_by_domain")
    if sc is not None:
        if not isinstance(sc, dict) or not all(isinstance(k, str) and _HOST_RE.match(k) and isinstance(v, str)
                                               and v in _SOURCE_CLASSES for k, v in sc.items()):
            problems.append("search.duckduckgo_lite.source_class_by_domain: need {lowercase host: one of "
                            + ", ".join(sorted(_SOURCE_CLASSES)) + "}")
    if not isinstance(d.get("terms_source"), str) or not d["terms_source"].strip():
        problems.append("search.duckduckgo_lite.terms_source: missing (every provider fact needs its source)")


def _check_doaj(problems: list, d) -> None:
    if not isinstance(d, dict):
        problems.append("search.doaj: missing or not an object")
        return
    if not isinstance(d.get("enabled"), bool):
        problems.append("search.doaj.enabled: must be true or false")
    if not isinstance(d.get("endpoint"), str) or not d["endpoint"].startswith("https://"):
        problems.append("search.doaj.endpoint: need an https URL")
    elif (urlsplit(d["endpoint"]).hostname or "") != "doaj.org":
        problems.append("search.doaj.endpoint: host must be doaj.org")
    # DOAJ documents a limit of two requests per second: the gate must stay at or above 0.5 s
    _num(problems, "search.doaj.min_interval_s", d.get("min_interval_s"), "float", 0.5, 3600)
    _num(problems, "search.doaj.max_response_bytes", d.get("max_response_bytes"), "int", 1024, 8 * 1024 * 1024)
    _num(problems, "search.doaj.daily_search_budget", d.get("daily_search_budget"), "int", 1, 100000)
    _num(problems, "search.doaj.snippet_max_chars", d.get("snippet_max_chars"), "int", 50, 20000)
    _num(problems, "search.doaj.year_from", d.get("year_from"), "int", 1900, 2100)
    _num(problems, "search.doaj.year_to", d.get("year_to"), "int", 1900, 2100)
    if (isinstance(d.get("year_from"), int) and isinstance(d.get("year_to"), int)
            and not isinstance(d["year_from"], bool) and d["year_from"] > d["year_to"]):
        problems.append("search.doaj.year_from: must not exceed year_to")
    if not isinstance(d.get("terms_source"), str) or not d["terms_source"].strip():
        problems.append("search.doaj.terms_source: missing (every provider fact needs its source)")


def _check_openalex(problems: list, o) -> None:
    if not isinstance(o, dict):
        problems.append("search.openalex: missing or not an object")
        return
    if not isinstance(o.get("endpoint"), str) or not o["endpoint"].startswith("https://"):
        problems.append("search.openalex.endpoint: need an https URL")
    _pos_num(problems, "search.openalex.min_interval_s", o.get("min_interval_s"))
    if o.get("max_concurrency") != 1 or isinstance(o.get("max_concurrency"), bool):
        problems.append("search.openalex.max_concurrency: must be 1 (single connection)")
    _num(problems, "search.openalex.max_response_bytes", o.get("max_response_bytes"), "int", 1024, 8 * 1024 * 1024)
    _num(problems, "search.openalex.daily_search_budget", o.get("daily_search_budget"), "int", 1, 100000)
    _num(problems, "search.openalex.snippet_max_chars", o.get("snippet_max_chars"), "int", 50, 20000)
    for k in ("from_publication_date", "to_publication_date"):
        v = o.get(k)
        if not (isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v)):
            problems.append(f"search.openalex.{k}: YYYY-MM-DD")
    if not isinstance(o.get("terms_source"), str) or not o["terms_source"].strip():
        problems.append("search.openalex.terms_source: missing (every provider fact needs its source)")
    if o.get("mailto") is not None:
        problems.append("search.openalex.mailto: must be unset (no contact address is approved)")


def _check_arxiv(problems: list, a) -> None:
    if not isinstance(a, dict):
        problems.append("search.arxiv: missing or not an object")
        return
    if not isinstance(a.get("endpoint"), str) or not a["endpoint"].startswith("https://"):
        problems.append("search.arxiv.endpoint: need an https URL")
    _pos_num(problems, "search.arxiv.min_interval_s", a.get("min_interval_s"))
    if a.get("max_concurrency") != 1 or isinstance(a.get("max_concurrency"), bool):
        problems.append("search.arxiv.max_concurrency: must be 1 (single connection)")
    _num(problems, "search.arxiv.max_response_bytes", a.get("max_response_bytes"), "int", 1024, 8 * 1024 * 1024)
    if not isinstance(a.get("terms_source"), str) or not a["terms_source"].strip():
        problems.append("search.arxiv.terms_source: missing (every provider fact needs its source)")
    q = a.get("query")
    if not isinstance(q, dict):
        problems.append("search.arxiv.query: missing or not an object")
        return
    _num(problems, "search.arxiv.query.max_terms", q.get("max_terms"), "int", 1, 30)
    _num(problems, "search.arxiv.query.min_term_chars", q.get("min_term_chars"), "int", 1, 10)
    if q.get("operator") not in ("OR", "AND"):
        problems.append("search.arxiv.query.operator: OR or AND")
    for k in ("submitted_from", "submitted_to"):
        v = q.get(k)
        if v is not None and not (isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v)):
            problems.append(f"search.arxiv.query.{k}: YYYY-MM-DD or null")
    if (q.get("submitted_from") is None) != (q.get("submitted_to") is None):
        problems.append("search.arxiv.query: submitted_from and submitted_to must both be set or both null")


def _check_judge(problems: list, j) -> dict:
    if not isinstance(j, dict):
        problems.append("judge: missing or not an object")
        return {}
    for k in sorted(set(j) - _JUDGE_KEYS):
        problems.append(f"judge.{k}: unknown key (typo?)")
    pool, sel, pat = j.get("pool"), j.get("selection"), j.get("family_patterns")
    out: dict = {}
    if not isinstance(pool, dict):
        problems.append("judge.pool: missing or not an object")
    else:
        provs = _str_list(problems, "judge.pool.providers", pool.get("providers"))
        for pv in provs:
            if pv not in SUPPORTED_JUDGE_PROVIDERS:
                problems.append(f"judge.pool.providers: {pv!r} is not a supported judge provider "
                                f"(supported: {list(SUPPORTED_JUDGE_PROVIDERS)}; it has no key/URL/price facts "
                                "in vera/m6/judge.py)")
        out["providers"] = tuple(provs)
    if not isinstance(sel, dict):
        problems.append("judge.selection: missing or not an object")
    else:
        if not isinstance(sel.get("rule"), str) or not sel["rule"].strip():
            problems.append("judge.selection.rule: missing or empty")
        out["rule"] = sel.get("rule")
        out["family_preference"] = tuple(_str_list(problems, "judge.selection.family_preference",
                                                   sel.get("family_preference")))
        _num(problems, "judge.selection.max_canary_attempts", sel.get("max_canary_attempts"), "int", 1, 20)
        _num(problems, "judge.selection.canary_max_tokens", sel.get("canary_max_tokens"), "int", 16, 100_000)
        _pos_num(problems, "judge.selection.listing_timeout_s", sel.get("listing_timeout_s"))
        out["max_canary_attempts"] = sel.get("max_canary_attempts")
        out["canary_max_tokens"] = sel.get("canary_max_tokens")
        out["listing_timeout_s"] = sel.get("listing_timeout_s")
    if not isinstance(pat, dict):
        problems.append("judge.family_patterns: missing or not an object")
    else:
        refuse = _str_list(problems, "judge.family_patterns.refuse_openai", pat.get("refuse_openai"), min_len=0)
        lacking = [x for x in REQUIRED_REFUSE_OPENAI if x not in refuse]
        if isinstance(pat.get("refuse_openai"), list) and lacking:
            problems.append(f"judge.family_patterns.refuse_openai={refuse!r}: must always contain "
                            f"{list(REQUIRED_REFUSE_OPENAI)} (missing {lacking}); the generator family must "
                            "never be judge-eligible")
        out["refuse_openai_patterns"] = tuple(refuse)
        out["non_chat_patterns"] = tuple(_str_list(problems, "judge.family_patterns.non_chat", pat.get("non_chat")))
        gen = pat.get("generic")
        if (not isinstance(gen, list) or not gen or not all(
                isinstance(p, list) and len(p) == 2 and all(isinstance(x, str) and x.strip() for x in p)
                for p in gen)):
            problems.append(f"judge.family_patterns.generic={gen!r}: must be a non-empty list of "
                            "[pattern, family] string pairs")
            gen = []
        out["generic_patterns"] = tuple((p[0], p[1]) for p in gen)
    forb = _str_list(problems, "judge.forbidden_env_overrides", j.get("forbidden_env_overrides"),
                     min_len=0)
    lacking = [x for x in REQUIRED_FORBIDDEN_ENV if x not in forb]
    if isinstance(j.get("forbidden_env_overrides"), list) and lacking:
        problems.append(f"judge.forbidden_env_overrides={forb!r}: must always contain "
                        f"{list(REQUIRED_FORBIDDEN_ENV)} (missing {lacking}); these overrides would change the "
                        "judge without being recorded")
    out["forbidden_env_overrides"] = tuple(forb)
    return out


def load_model_selection_strict(path) -> str:
    try:
        d = json.loads(Path(path).read_text())
        sel = d["selected_model"]
        if not isinstance(sel, str) or not sel.strip():
            raise ValueError("selected_model is empty")
        return sel
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise EvalConfigError(f"{path}: cannot read selected_model ({type(e).__name__}: {e}). A missing "
                              "model-selection file is an error here (strict).") from None


# ----------------------------------------------------------------------------- loader
def load_eval_config(path=FREEZE_PATH, run_path=RUN_PATH, *,
                     model_selection_path=MODEL_SELECTION_PATH) -> EvalConfig:
    """Read, pin-check and validate both files. Raises EvalConfigError listing every problem."""
    f_bytes, d = _read_json(path, "freeze file")
    r_bytes, r = _read_json(run_path, "run config")
    f_sha, r_sha = file_sha256(f_bytes), file_sha256(r_bytes)
    pin = r.get("freeze_pin") if isinstance(r.get("freeze_pin"), dict) else {}
    # Pin check first (SD5): a changed freeze file is refused outright, with both values shown.
    if pin.get("file_sha256") != f_sha:
        raise EvalConfigError(
            f"{path}: sha256 (newline-normalised) {f_sha} != pinned {pin.get('file_sha256')!r} in {run_path}. "
            "The freeze file is immutable in this incarnation: restore it (compare bytes and line endings), "
            "do not edit the pin to match. A later incarnation uses a NEW freeze file with a new pin.")
    h16 = requirements_hash16(d)
    if pin.get("requirements_hash16") != h16:
        raise EvalConfigError(
            f"{path}: requirements hash {h16!r} != pinned {pin.get('requirements_hash16')!r} in {run_path}. "
            "The frozen requirements/objections changed: restore the file.")
    problems: list[str] = []
    _check_freeze(problems, d, path)
    judge_kw = _check_run(problems, r)
    selected = load_model_selection_strict(model_selection_path)
    gen_model = (r.get("generator") or {}).get("model") if isinstance(r.get("generator"), dict) else None
    if isinstance(gen_model, str) and gen_model != selected:
        problems.append(f"generator.model={gen_model!r} but model-selection.json selected_model={selected!r}; "
                        "the eval would score a different generator than the pipeline used")
    if problems:
        raise EvalConfigError(
            f"{path} + {run_path}: {len(problems)} problem(s), nothing was scored:\n  - " + "\n  - ".join(problems)
            + "\nOwner: R1. Fix the run config as a new run_config_version (the freeze file is never edited "
              "in this incarnation), never by editing code.")
    sc = d["success_criteria"]
    return EvalConfig(
        freeze_path=str(path), run_path=str(run_path), schema_version=int(r["schema_version"]),
        run_config_version=r["run_config_version"], freeze_file_sha256=f_sha, run_file_sha256=r_sha,
        requirements_hash16=h16, question=d["question"], requirements=tuple(deepcopy(d["requirements"])),
        objections=tuple(deepcopy(d["objections"])),
        success=SuccessCriteria(
            relevance_min=float(sc["relevance_threshold"]),
            relevance_delta=float(sc["relevance_improvement_pp"]) / 100,
            grounding_min=float(sc["grounding_threshold"]),
            grounding_max_fabricated=int(r["scoring"]["grounding_max_fabricated"]),
            reasoning_min=int(sc["reasoning_threshold"]), reasoning_delta=int(sc["reasoning_objections_improvement"]),
            audit_min=int(sc["auditability_threshold"]), cost_min=int(sc["cost_latency_threshold"])),
        budget=Budget(float(r["budget"]["cost_usd"]), float(r["budget"]["latency_s"])),
        generator_model=gen_model, judge=JudgeSettings(**judge_kw),
        model_families=MappingProxyType(deepcopy(r["model_families"])),
        pipeline=MappingProxyType(deepcopy(r["pipeline"])),
        search=MappingProxyType(deepcopy(r["search"])),
        raw=MappingProxyType(deepcopy(d)), run_raw=MappingProxyType(deepcopy(r)))


def family_for(model: str, cfg: EvalConfig, *, override: str | None = None) -> FamilyResolution:
    """Family of a model id from the run config's map. Unknown + no override fails closed."""
    entry = cfg.model_families.get(model)
    if entry:
        return FamilyResolution(model, entry["family"], "map")
    if override and override.strip():
        return FamilyResolution(model, override.strip().lower(), "override")
    raise EvalConfigError(
        f"Model {model!r} is not in model_families of {cfg.run_path}; refusing (fail closed) because its family "
        "is unknown and the self-grading guard cannot be checked. Add the model with a source URL (a new "
        "run_config_version, R1 approves) or set VERA_JUDGE_FAMILY_OVERRIDE=<family> (disclosed in the report).")
