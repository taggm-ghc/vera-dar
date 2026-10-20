"""Quasi-dynamic judge selection for the slim live MVP (plan S3, R1 decision D2, 2026-10-02).

No provider or model is pinned. At run time the judge is chosen from the models Groq lists as live,
keeping only non-OpenAI families (OpenAI is the generator's family, so it is refused), then a tiny
canary call proves the key, the model and JSON mode BEFORE any baseline spend. The selection
(provider, model, family, rule) is recorded and the judge is then fixed for the run.

Network seams (nothing here can call out unless the caller passes the real transport):
  * http_get(url, headers, timeout) -> parsed JSON dict   (model listing; None means "not provided",
    which fails verbosely instead of calling out; `default_http_get` is the real one, CLI only)
  * judge_transport(url, headers, body, timeout) -> parsed JSON dict of an OpenAI-style chat
    completion (canary and M6 judge calls); `default_judge_transport` is the real one.

GOAL #2 (this file is wired, not rebuilt): the pool providers, family patterns, family preference, canary
and listing bounds, rule id and forbidden env overrides now come from config/vera_eval_run.json via
vera.eval_config (single source). They are read lazily (module `__getattr__` keeps the old constant names
readable), so importing this module never reads a config file and a bad config fails where it is used,
verbosely; probing an unknown attribute raises AttributeError without reading config. Family of a model id
has two sources, strictest wins (see family_for): name patterns, plus the run config's model_families map. The family patterns are still UNVERIFIED against Groq's live list until Stage 0.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping

from vera.cost_ledger import CostLedger
from vera.eval_config import EvalConfig, load_eval_config
from vera.m6.judge import (JUDGE_MAX_TOKENS, _PROVIDERS, JudgeConfigError, JudgeLimits, JudgeUsage,
                           OpenAICompatJudge, generation_family)

# --- settings: single source is config/vera_eval_run.json (judge section) ------------------------
# Why: the canary prompt and the env var NAMES are code invariants, not eval parameters.
CANARY_PROMPT = 'Reply with exactly this JSON object and nothing else: {"ok": true}'
JUDGE_POOL_ENV = "VERA_JUDGE_POOL"
SAME_FAMILY_OVERRIDE_ENV = "VERA_JUDGE_ALLOW_SAME_FAMILY"  # existing override, see judge.py


def _cfg(config: EvalConfig | None = None) -> EvalConfig:
    return config or load_eval_config()


def _provider_facts(provider: str):
    try:
        return _PROVIDERS[provider]
    except KeyError:
        raise JudgeConfigError(f"Judge pool provider {provider!r} (run config judge.pool.providers) has no "
                               f"provider facts in vera/m6/judge.py (known: {sorted(_PROVIDERS)}).") from None


def _settings(config: EvalConfig | None = None):
    return _cfg(config).judge


_LAZY_NAMES = frozenset({
    "JUDGE_LISTING_TIMEOUT_S", "JUDGE_PROVIDER_ORDER", "KEY_ENV", "LISTING_URL", "CHAT_URL",
    "REFUSE_OPENAI_PATTERNS", "NON_CHAT_PATTERNS", "MODEL_FAMILY_PATTERNS", "JUDGE_FAMILY_PREFERENCE",
    "JUDGE_MAX_CANARY_ATTEMPTS", "CANARY_MAX_TOKENS", "RULE_ID", "FORBIDDEN_JUDGE_ENV"})


def __getattr__(name: str):
    """Back-compat constant names, resolved from the run config on access (never at import)."""
    if name not in _LAZY_NAMES:  # unknown probes (hasattr, getattr default, import *) never load config
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    j = _settings()
    table = {
        "JUDGE_LISTING_TIMEOUT_S": lambda: j.listing_timeout_s,
        "JUDGE_PROVIDER_ORDER": lambda: j.providers,
        "KEY_ENV": lambda: {p: _provider_facts(p).key_env for p in j.providers},
        "LISTING_URL": lambda: {p: _provider_facts(p).base_url + "/models" for p in j.providers},
        "CHAT_URL": lambda: {p: _provider_facts(p).base_url + "/chat/completions" for p in j.providers},
        "REFUSE_OPENAI_PATTERNS": lambda: j.refuse_openai_patterns,
        "NON_CHAT_PATTERNS": lambda: j.non_chat_patterns,
        "MODEL_FAMILY_PATTERNS": lambda: j.generic_patterns,
        "JUDGE_FAMILY_PREFERENCE": lambda: j.family_preference,
        "JUDGE_MAX_CANARY_ATTEMPTS": lambda: j.max_canary_attempts,
        "CANARY_MAX_TOKENS": lambda: j.canary_max_tokens,
        "RULE_ID": lambda: j.rule,
        "FORBIDDEN_JUDGE_ENV": lambda: j.forbidden_env_overrides,
    }
    return table[name]()


_KEYISH = re.compile(r"(gsk_[A-Za-z0-9]{6,}|sk-[A-Za-z0-9_\-]{8,}|tvly-[A-Za-z0-9_\-]{6,}|"
                     r"Bearer\s+[A-Za-z0-9._\-]{8,}|://[^/\s:@]+:[^/\s@]+@)")


# --- data ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    provider: str
    model: str
    family: str
    family_source: str


@dataclass(frozen=True)
class Refusal:
    provider: str
    model: str
    reason: str


@dataclass
class Selection:
    judge: object  # the selected judge (n=1) or the first of `judges`
    candidate: Candidate
    canary_failures: list
    refused: list
    rule: str
    listing: dict
    judges: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    pool_source: str = "default"
    canary_spend: dict = field(default_factory=dict)

    def as_record(self) -> dict:
        """Report-safe record: names, ids and short strings only."""
        c = self.candidate
        return {"provider": c.provider, "model": c.model, "family": c.family,
                "family_source": c.family_source, "rule": self.rule, "pool_source": self.pool_source,
                "candidates": [(x.provider, x.model, x.family) for x in self.candidates],
                "refusals": [(r.provider, r.model, r.reason) for r in self.refused],
                "canary_failures": list(self.canary_failures), "canary_spend": dict(self.canary_spend),
                "listing": {k: v for k, v in self.listing.items() if k != "listed_models"}
                | {"listed_count": len(self.listing.get("listed_models", []))},
                "panel": [(j.provider, j.model, j.family) for j in self.judges]}


def scrub(text: str, env: Mapping[str, str] | None = None) -> str:
    """Remove env secret values and key-shaped strings from text destined for reports/exceptions."""
    out = str(text)
    for k, v in (env or {}).items():
        if v and len(v) >= 4 and any(t in k.upper() for t in ("KEY", "URL", "PASSWORD", "TOKEN", "SECRET")):
            out = out.replace(v, "***")
    return _KEYISH.sub("***", out)


# --- family map ------------------------------------------------------------------------------
def family_for(model_id: str, settings=None, *, model_families: Mapping | None = None) -> tuple[str | None, str]:
    """(family, reason). family None = refused (reason says why); 'openai' = refused by the caller.
    Order (plan section 5, R11a): openai/gpt first; non-chat/distill/compound next; generic last.
    Two sources, strictest wins (goal #2 F5): (1) name patterns from the run config (judge.family_patterns);
    (2) the run config's explicit model_families map. The map can only ADD refusals, never remove one:
    a pattern refusal always stands; a map entry of 'openai' refuses; a map family that disagrees with the
    pattern family refuses. The map is read only when the caller supplies it (`model_families`) or when
    `settings` is omitted (then the whole run config is loaded); callers that pass only `settings` get the
    pattern path alone, which refuses at least everything the pre-goal-#2 code refused."""
    if settings is None:
        cfg = _cfg()
        settings = cfg.judge
        if model_families is None:
            model_families = cfg.model_families
    st = settings
    mid = (model_id or "").strip().lower()
    if not mid:
        return None, "empty model id"
    fam, why = _pattern_family(mid, st)
    entry = (model_families or {}).get((model_id or "").strip())
    mapped = entry.get("family") if isinstance(entry, Mapping) else None
    if mapped:
        mapped = str(mapped).strip().lower()
        if fam == "openai":
            return fam, why
        if mapped == "openai":
            return "openai", "model_families map: openai (openai family = generator family)"
        if fam is not None and mapped != fam:
            return None, (f"family conflict: model_families maps {mapped!r} but the id patterns say {fam!r}; "
                          "refused (stricter outcome)")
    return fam, why


def _pattern_family(mid: str, st) -> tuple[str | None, str]:
    for p in st.refuse_openai_patterns:
        if p in mid:
            return "openai", f"pattern:{p} (openai family = generator family)"
    for p in st.non_chat_patterns:
        if p in mid:
            return None, f"excluded: id contains {p!r} (non-chat, distilled or compound model)"
    hits = [(p, f) for p, f in st.generic_patterns if p in mid]
    fams = {f for _p, f in hits}
    if len(fams) > 1:
        return None, f"ambiguous family (matches {sorted(fams)}); refused"
    if not hits:
        return None, "unknown model family: no pattern matches this id; refused (add a verified pattern)"
    return hits[0][1], f"pattern:{hits[0][0]}"


def _family_rank(family: str, settings=None) -> int:
    pref = (settings or _settings()).family_preference
    try:
        return pref.index(family)
    except ValueError:
        return len(pref)


# --- real transports (only passed in by the CLI / runner; never default inside tests) -------
def default_http_get(url: str, headers: dict, timeout: float) -> dict:
    import requests
    r = requests.get(url, headers=headers, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


def default_judge_transport(url: str, headers: dict, body: dict, timeout: float) -> dict:
    import requests
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


# --- listing and candidates ------------------------------------------------------------------
def parse_pool(pool_env: str, config: EvalConfig | None = None) -> list[tuple[str, str]]:
    """'groq:model-a,groq:model-b' -> [(provider, model)]. Verbose on malformed/foreign entries."""
    providers = _settings(config).providers
    out = []
    for raw in (pool_env or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise JudgeConfigError(f"{JUDGE_POOL_ENV} entry {raw!r} is not 'provider:model'.")
        prov, model = raw.split(":", 1)
        prov, model = prov.strip().lower(), model.strip()
        if prov not in providers:
            raise JudgeConfigError(
                f"{JUDGE_POOL_ENV} entry {raw!r}: provider {prov!r} is not in the judge pool "
                f"{providers} (OpenAI is the generator family; Mistral is not in the pool).")
        if not model:
            raise JudgeConfigError(f"{JUDGE_POOL_ENV} entry {raw!r} has an empty model id.")
        out.append((prov, model))
    if not out:
        raise JudgeConfigError(f"{JUDGE_POOL_ENV} is set but has no entries.")
    return out


def list_models(provider: str, env: Mapping[str, str], http_get: Callable | None, *,
                config: EvalConfig | None = None) -> list[str]:
    """GET <base>/models ($0). Raises JudgeConfigError verbosely (key never printed)."""
    if http_get is None:
        raise JudgeConfigError("No http_get transport was provided, so the live model listing was not "
                               "attempted (nothing is called implicitly). Pass default_http_get from the CLI.")
    facts = _provider_facts(provider)
    key = env.get(facts.key_env, "")
    try:
        data = http_get(facts.base_url + "/models", {"Authorization": f"Bearer {key}"},
                        _settings(config).listing_timeout_s)
    except Exception as e:  # noqa: BLE001 - verbose re-raise, scrubbed
        raise JudgeConfigError(
            f"{provider} model listing failed: {scrub(repr(e), env)[:300]}. Check {facts.key_env} "
            "and connectivity; no judge can be selected and nothing was spent.") from None
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise JudgeConfigError(f"{provider} model listing has an unexpected shape (no 'data' list); "
                               "refusing to guess models.")
    ids = sorted({r["id"] for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)})
    if not ids:
        raise JudgeConfigError(f"{provider} model listing is empty: no live model to judge with.")
    return ids


def _collect(env: Mapping[str, str], *, http_get, pool_env: str | None, config: EvalConfig | None = None):
    """-> (candidates in preference order, refusals, listing record)."""
    cfg = _cfg(config)
    st = cfg.judge
    gen_family = generation_family(cfg)
    out: list[Candidate] = []
    refused: list[Refusal] = []
    listing: dict = {"status": "not_attempted", "provider": None, "model": None, "listed_models": [],
                     "checked_at": None}
    for provider in st.providers:
        key_env = _provider_facts(provider).key_env
        if not env.get(key_env):
            refused.append(Refusal(provider, "*", f"key absent ({key_env} not set)"))
            continue
        ids = list_models(provider, env, http_get, config=cfg)
        listing = {"status": "ok", "provider": provider, "model": None, "listed_models": ids,
                   "checked_at": datetime.now(timezone.utc).isoformat()}
        for mid in ids:
            fam, why = family_for(mid, st, model_families=cfg.model_families)
            if fam is None:
                refused.append(Refusal(provider, mid, why))
            elif fam == gen_family:
                refused.append(Refusal(provider, mid, why))
            else:
                out.append(Candidate(provider, mid, fam, why))
    if pool_env:
        pool = parse_pool(pool_env, cfg)
        listed = {(c.provider, c.model): c for c in out}
        restricted = []
        for prov, model in pool:
            c = listed.get((prov, model))
            if c is None:
                already = next((r for r in refused if (r.provider, r.model) == (prov, model)), None)
                reason = (f"pool entry refused earlier: {already.reason}" if already else
                          "pool entry not in the live listing (or provider key absent)")
                refused.append(Refusal(prov, model, f"{JUDGE_POOL_ENV}: {reason}"))
            elif c not in restricted:
                restricted.append(c)
        out = restricted  # pool order is the order (R1: volatile per-run input)
        pool_source = JUDGE_POOL_ENV
    else:
        out.sort(key=lambda c: (_family_rank(c.family, st), c.model))  # deterministic
        pool_source = "default"
    listing = dict(listing, pool_source=pool_source)
    return out, refused, listing


def candidates(env: Mapping[str, str], *, http_get, pool_env: str | None = None,
               config: EvalConfig | None = None) -> tuple[list[Candidate], list[Refusal]]:
    """Live, non-OpenAI, known-family candidates in preference order, plus refusals with reasons."""
    cands, refused, _listing = _collect(env, http_get=http_get, pool_env=pool_env, config=config)
    return cands, refused


def distinct_families(cands: list[Candidate]) -> list[Candidate]:
    seen, out = set(), []
    for c in cands:
        if c.family not in seen:
            seen.add(c.family)
            out.append(c)
    return out


# --- the judge -------------------------------------------------------------------------------
class SelectedJudge(OpenAICompatJudge):
    """OpenAICompatJudge with the selected model and the family decided by the pool patterns
    (family_source "pattern"). There is no code default model. Re-asserts family != generator family with NO override: the same-family env override,
    the model override and the price overrides are refused at construction (S5 F3), because the
    selection record would not disclose them."""

    def __init__(self, provider: str, model: str, family: str, *, limits: JudgeLimits | None = None,
                 ledger: CostLedger | None = None, api_key: str | None = None, transport=None,
                 env: Mapping[str, str] | None = None, config: EvalConfig | None = None):
        import os
        cfg = _cfg(config)
        st = cfg.judge
        if provider not in st.providers:
            raise JudgeConfigError(f"Provider {provider!r} is not in the judge pool {st.providers}.")
        # Checked BEFORE the base constructor, which would read (and act on) the same variables.
        stale = [n for n in st.forbidden_env_overrides if n in os.environ or (env is not None and n in env)]
        if stale:
            raise JudgeConfigError(
                f"Refusing to start the selected judge: env override(s) {stale} are set. They would change "
                "the judge model, price or family rule without being recorded in the selection. "
                "Unset them (their values are not printed).")
        if family == generation_family(cfg):
            raise JudgeConfigError(
                f"Selected judge family {family!r} (model {model!r}) equals the generation "
                "family; R1 requires a different family. Refused.")
        super().__init__(provider, limits, api_key=api_key, ledger=ledger, model=model, family=family,
                         family_source="pattern", config=cfg)
        self._transport = transport or default_judge_transport
        self._max_tokens = JUDGE_MAX_TOKENS
        self._canary_max_tokens = st.canary_max_tokens
        self.same_family_override = False  # never true here: the override env var is refused above

    def _call(self, prompt: str):
        body = {"model": self.model, "max_tokens": self._max_tokens, "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}]}
        headers = {"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        try:
            d = self._transport(self.url, headers, body, 120)
        except Exception as e:  # noqa: BLE001 - verbose, key scrubbed, no retry (bounded loops)
            raise RuntimeError(f"Judge call to {self.provider} model {self.model!r} failed: "
                               f"{scrub(repr(e), {_provider_facts(self.provider).key_env: self.api_key})[:300]}") from None
        u = d.get("usage", {}) or {}
        return (d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0),
                u.get("completion_tokens", 0))


def _canary(judge: SelectedJudge) -> dict:
    """One minimal judge_json call. Returns its spend; then resets judge.usage so M6's caps start at 0
    (judge_json counts the canary in judge.usage, and run_m6 caps on that same object)."""
    judge._max_tokens = judge._canary_max_tokens
    try:
        res = judge.judge_json("preflight_canary", CANARY_PROMPT)
        if not isinstance(res, dict) or "ok" not in res:
            raise RuntimeError(f"canary reply is not the expected JSON object: {str(res)[:80]!r}")
    finally:
        judge._max_tokens = JUDGE_MAX_TOKENS
        spend = judge.usage.as_dict()
        judge._last_canary_spend = spend  # also readable after a failed canary
        judge.usage = JudgeUsage()  # the ledger entry stays; M6's budget restarts at zero
    return spend


def _add(a: dict, b: dict) -> dict:
    return {k: round(a.get(k, 0) + b.get(k, 0), 6) for k in ("api_calls", "tokens", "cost_usd", "latency_s")}


def select_from(cands: list[Candidate], refused: list[Refusal], listing: dict, env: Mapping[str, str], *,
                limits: JudgeLimits | None, ledger: CostLedger | None, n: int = 1,
                judge_transport=None, config: EvalConfig | None = None) -> Selection:
    cfg = _cfg(config)
    st = cfg.judge
    if n < 1:
        raise JudgeConfigError(f"n must be >= 1, got {n}.")
    tried: list[str] = []
    ok: list[tuple[Candidate, SelectedJudge]] = []
    spend: dict = {"api_calls": 0, "tokens": 0, "cost_usd": 0.0, "latency_s": 0.0}
    attempts = 0
    for c in distinct_families(cands):
        if len(ok) >= n or attempts >= st.max_canary_attempts:
            break
        attempts += 1
        try:
            j = SelectedJudge(c.provider, c.model, c.family, limits=limits, ledger=ledger,
                              api_key=env.get(_provider_facts(c.provider).key_env), transport=judge_transport,
                              env=env, config=cfg)
            try:
                _canary(j)
            finally:
                spend = _add(spend, getattr(j, "_last_canary_spend", {}))
            ok.append((c, j))
        except Exception as e:  # noqa: BLE001 - next candidate; recorded, scrubbed
            tried.append(f"{c.provider}:{c.model}: {scrub(repr(e), env)[:200]}")
    if len(ok) < n:
        raise JudgeConfigError(
            f"No live non-OpenAI judge set of size {n}: {len(cands)} candidate(s), "
            f"{len(distinct_families(cands))} distinct family(ies), {attempts} canary attempt(s) "
            f"(limit {st.max_canary_attempts}), canary failures {tried}, refused {len(refused)} "
            "model(s) (see report). Nothing was spent beyond canaries; no run is created.")
    first_c, first_j = ok[0]
    return Selection(first_j, first_c, tried, refused, st.rule, listing, judges=[j for _c, j in ok],
                     candidates=cands, pool_source=listing.get("pool_source", "default"),
                     canary_spend=spend)


def select_judge(env: Mapping[str, str], *, http_get, limits: JudgeLimits | None = None,
                 ledger: CostLedger | None = None, n: int = 1, judge_transport=None,
                 config: EvalConfig | None = None) -> Selection:
    """List live models, filter, canary the best candidates, return the fixed judge (panel: n>1 takes
    distinct families). Verbose JudgeConfigError when nothing qualifies."""
    cfg = _cfg(config)
    cands, refused, listing = _collect(env, http_get=http_get, pool_env=env.get(JUDGE_POOL_ENV), config=cfg)
    if not cands:
        raise JudgeConfigError(
            f"No candidate judge after filtering: refused {len(refused)}: "
            + "; ".join(f"{r.provider}:{r.model} ({r.reason})" for r in refused[:10]))
    return select_from(cands, refused, listing, env, limits=limits, ledger=ledger, n=n,
                       judge_transport=judge_transport, config=cfg)
