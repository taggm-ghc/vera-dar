"""CHUNK-TAXO-JOB-1109: embedder slot for the chunk-taxonomy job (plan v3.8/v3.8a, OD-1109-DEMB-3).

Config `embedder` (config/chunk_taxo.json, or env CHUNK_TAXO_EMBEDDER):
  "none"        (default) the job stays inert with one log line; nothing is imported or sent.
  "cloudflare"  Cloudflare Workers AI REST (@cf/baai/bge-*), over HTTPS to api.cloudflare.com only. Credentials are the env vars
                CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN, read at call time; without them the adapter is not created and the
                job stays inert (one log line). AI Gateway is not used. No model is loaded in the API.
  any name registered with register() (tests use this for a deterministic fake).

The returned object satisfies vera.semantic_similarity.Embedder: embed(texts) -> list[list[float]]; it also has `name` (the model id
stored beside every vector) and may raise Deferred (quota or budget: resume after 00:00 UTC) or EmbedError (a code only).
Outbound control (OD-1109-DEMB-3): the only host is api.cloudflare.com, pinned in code (HOST) over https. No api_permissions.json
entry is added: server-side LLM calls such as api.groq.com are not governed by that gate (same precedent).
Secrets, the URL and response bodies are never logged or put in an exception message; callers log the exception class only.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Callable, Sequence

log = logging.getLogger("vera.chunk_taxo")

HOST = "api.cloudflare.com"
ENV_ACCOUNT, ENV_TOKEN = "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"
# Model facts from the Cloudflare docs read by the coordinator (v3.8a); every value is overridable in config `cloudflare.models`.
MODELS = {
    "bge-m3": {"max_tokens": 60000, "neurons_per_m": 1075, "dim": None},
    "bge-small-en-v1.5": {"max_tokens": 512, "neurons_per_m": 1841, "dim": 384},
    "bge-base-en-v1.5": {"max_tokens": 512, "neurons_per_m": 6058, "dim": 768},
}
DEFAULTS = {"model": "bge-m3", "send_allowed": True, "daily_neuron_budget": 8000, "chars_per_token": 3.0, "max_attempts": 4,
            "backoff_base_s": 1.0, "max_wait_s": 60.0, "timeout_s": 30,
            "max_response_bytes": 8_000_000,
            "reserve_neurons": 0}  # HYBRID-1109: the job stops at daily_neuron_budget - reserve_neurons (ask-time share)  # 8 MB: ~64 vectors x 1024 floats is ~1 MB of JSON; generous but bounded
_REGISTRY: dict[str, Callable[[dict], object]] = {}
_BUDGET = {"day": None, "neurons": 0.0, "requests": 0}  # per UTC day, process-wide (a new adapter instance keeps the day's count)
_BLOCK = threading.Lock()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every 3xx so the bearer token is never forwarded to another URL or host (the 3xx surfaces as an HTTPError)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _default_open(req, timeout):
    return urllib.request.build_opener(_NoRedirect).open(req, timeout=timeout)


class EmbedError(RuntimeError):
    """Failure with a fixed code as its only message."""


class Deferred(Exception):
    """Quota or budget reached: not a failure; resume after the next 00:00 UTC."""

    def __init__(self, code: str, retry_in_s: float):
        super().__init__(code)
        self.code, self.retry_in_s = code, float(retry_in_s)


def register(name: str, factory: Callable[[dict], object]) -> None:
    _REGISTRY[str(name)] = factory


def unregister(name: str) -> None:
    _REGISTRY.pop(str(name), None)


def reset_budget_for_tests() -> None:
    with _BLOCK:
        _BUDGET.update(day=None, neurons=0.0, requests=0)


def seconds_to_utc_midnight(now: float) -> float:
    d = datetime.fromtimestamp(now, tz=timezone.utc)
    nxt = (d + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1.0, (nxt - d).total_seconds() + 30.0)


def _l2(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def _split(text: str, max_chars: int) -> list[str]:
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)] or [""]


class CloudflareEmbedder:
    def __init__(self, ecfg: dict, env=None, opener=None, clock=time.time, sleep=time.sleep):
        c = dict(DEFAULTS)
        c.update({k: v for k, v in (ecfg or {}).items() if k in DEFAULTS and v is not None})
        models = {k: dict(v) for k, v in MODELS.items()}
        for k, v in ((ecfg or {}).get("models") or {}).items():
            models.setdefault(k, {}).update(v)
        self.model_short = str(c["model"]).split("/")[-1]
        if self.model_short not in models:
            raise EmbedError("model_unknown")
        self.mcfg = models[self.model_short]
        self.cfg, self.env, self.clock, self.sleep = c, (env if env is not None else os.environ), clock, sleep
        self.opener = opener or _default_open
        self.name = "@cf/baai/" + self.model_short
        self.dim = self.mcfg.get("dim")
        self.batch_limit = int((ecfg or {}).get("max_batch") or 64)
        self.stats_run = {"neurons": 0.0, "requests": 0, "deferred": 0, "attempts": 0, "last_http_status": 0}
        self._day_used = 0.0  # the UTC day's neurons as persisted in tax_job_runs (set by the job before each batch)

    def set_day_used(self, neurons: float) -> None:
        self._day_used = max(0.0, float(neurons))

    # -- config checks
    @staticmethod
    def usable(ecfg: dict, env) -> str:
        if (ecfg or {}).get("send_allowed", DEFAULTS["send_allowed"]) is not True:
            return "send_not_allowed"
        if not (env.get(ENV_ACCOUNT) or "").strip() or not (env.get(ENV_TOKEN) or "").strip():
            return "credentials_missing"
        return "ok"

    def _url(self) -> str:
        acct = (self.env.get(ENV_ACCOUNT) or "").strip()
        if not acct or not all(ch.isalnum() or ch in "-_" for ch in acct):
            raise EmbedError("credentials_missing")
        return f"https://{HOST}/client/v4/accounts/{acct}/ai/run/{self.name}"

    # -- budget
    def _day(self) -> str:
        return datetime.fromtimestamp(self.clock(), tz=timezone.utc).strftime("%Y-%m-%d")

    def _charge_check(self, est: float) -> None:
        with _BLOCK:
            if _BUDGET["day"] != self._day():
                _BUDGET.update(day=self._day(), neurons=0.0, requests=0)
            if max(_BUDGET["neurons"], self._day_used) + est > float(self.cfg["daily_neuron_budget"]) - float(self.cfg["reserve_neurons"]):
                self.stats_run["deferred"] += 1
                raise Deferred("budget_reached", seconds_to_utc_midnight(self.clock()))

    def _charge(self, neurons: float) -> None:
        with _BLOCK:
            _BUDGET["neurons"] += neurons
            _BUDGET["requests"] += 1
        self.stats_run["neurons"] += neurons
        self.stats_run["requests"] += 1

    def stats(self) -> dict:
        with _BLOCK:
            return {"neurons_today": round(_BUDGET["neurons"], 2), **{k: self.stats_run[k] for k in ("neurons", "requests", "deferred")}}

    # -- request
    def _post(self, texts: list[str]):
        self.stats_run["attempts"], self.stats_run["last_http_status"] = 0, 0  # TAXO-REASON-1111: ints only, per request
        timed_out = False
        if self.cfg["send_allowed"] is not True:
            raise EmbedError("send_not_allowed")
        url = self._url()
        token = (self.env.get(ENV_TOKEN) or "").strip()
        if not token:
            raise EmbedError("credentials_missing")
        body = json.dumps({"text": texts}).encode("utf-8")
        for attempt in range(int(self.cfg["max_attempts"])):
            req = urllib.request.Request(url, data=body, method="POST",
                                         headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
            retry_after = None
            self.stats_run["attempts"], self.stats_run["last_http_status"], timed_out = attempt + 1, 0, False
            try:
                with self.opener(req, timeout=float(self.cfg["timeout_s"])) as resp:
                    cap = int(self.cfg["max_response_bytes"])
                    raw = resp.read(cap + 1)
                    if len(raw) > cap:
                        raise EmbedError("response_too_large")
                    return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as exc:
                code = int(exc.code)
                self.stats_run["last_http_status"] = code
                try:
                    errs = json.loads(exc.read(65536).decode("utf-8")).get("errors") or []
                except Exception:  # noqa: BLE001
                    errs = []
                blob = json.dumps(errs).lower()
                if code in (429, 403) and any(w in blob for w in ("neuron", "daily", "allocation", "quota", "4006")):
                    self.stats_run["deferred"] += 1
                    raise Deferred("quota_reached", seconds_to_utc_midnight(self.clock())) from None
                if code == 429 or code >= 500:
                    ra = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
                    try:
                        retry_after = float(ra) if ra is not None else None
                    except ValueError:
                        retry_after = None
                else:
                    raise EmbedError(f"http_{code}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                # transient network failure: retried below; a timeout is told apart from other network errors (no text kept)
                timed_out = isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)
            except ValueError:
                raise EmbedError("bad_response") from None
            if attempt + 1 >= int(self.cfg["max_attempts"]):
                break
            wait = retry_after if retry_after is not None else float(self.cfg["backoff_base_s"]) * (2 ** attempt)
            self.sleep(min(max(0.0, wait), float(self.cfg["max_wait_s"])))
        raise EmbedError("timeout" if timed_out else "request_failed")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self.cfg["send_allowed"] is not True:
            raise EmbedError("send_not_allowed")
        texts = list(texts)
        max_chars = max(1, int(float(self.mcfg["max_tokens"]) * float(self.cfg["chars_per_token"])))
        parts, owner = [], []
        for i, t in enumerate(texts):
            for p in _split(str(t), max_chars):
                parts.append(p)
                owner.append(i)
        out_parts: list[list[float]] = []
        for lo in range(0, len(parts), self.batch_limit):
            chunk = parts[lo:lo + self.batch_limit]
            est_tokens = sum(len(p) for p in chunk) / float(self.cfg["chars_per_token"])
            est = est_tokens / 1e6 * float(self.mcfg["neurons_per_m"])
            self._charge_check(est)
            resp = self._post(chunk)
            res = resp.get("result") if isinstance(resp, dict) else None
            data = res.get("data") if isinstance(res, dict) else None
            if not isinstance(data, list) or len(data) != len(chunk) or resp.get("success") is False:
                raise EmbedError("bad_response")
            usage = res.get("usage") if isinstance(res, dict) else None
            tok = usage.get("total_tokens") or usage.get("prompt_tokens") if isinstance(usage, dict) else None
            neurons = (float(tok) / 1e6 * float(self.mcfg["neurons_per_m"])) if isinstance(tok, (int, float)) and tok >= 0 else est
            self._charge(neurons)
            for row in data:
                if not isinstance(row, list) or not row:
                    raise EmbedError("bad_response")
                v = [float(x) for x in row]
                if not all(math.isfinite(x) for x in v):
                    raise EmbedError("bad_response")
                if self.dim is None:
                    self.dim = len(v)  # detected from the first response (bge-m3's dim is not documented)
                elif len(v) != self.dim:
                    raise EmbedError("dim_mismatch")
                out_parts.append(v)
        out = []
        for i in range(len(texts)):
            vs = [out_parts[k] for k, o in enumerate(owner) if o == i]
            if len(vs) == 1:
                out.append(vs[0])
            else:  # split input: mean-pool the parts, then L2-normalise
                out.append(_l2([sum(c) / len(vs) for c in zip(*vs)]))
        return out

    def close(self) -> None:
        return None


def load_embedder(cfg: dict | None, env=None, opener=None):
    """Return (embedder, reason). embedder is None when the job must stay inert; reason is a short code. Never raises."""
    try:
        env = env if env is not None else os.environ
        name = str((cfg or {}).get("embedder") or "none").strip().lower()
        if name == "none":
            return None, "embedder_none"
        if name in _REGISTRY:
            return _REGISTRY[name](cfg or {}), "ok"
        if name == "cloudflare":
            ecfg = (cfg or {}).get("cloudflare") or {}
            why = CloudflareEmbedder.usable(ecfg, env)
            if why != "ok":
                return None, why
            return CloudflareEmbedder(ecfg, env=env, opener=opener), "ok"
        return None, "embedder_unknown"
    except Exception as exc:  # noqa: BLE001
        log.warning("chunk_taxo embedder load failed: CT_EMBED_ERROR (%s)", type(exc).__name__)
        return None, "embedder_error"
