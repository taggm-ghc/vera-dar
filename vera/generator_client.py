"""Shared generator client factory and fail-closed pricing (FREEGEN-1028).

The generator (the model that writes eval/baseline/pipeline answers) is configured in the `generator` block of
config/vera_eval_run.json:
    model        (required)  model id sent to the provider
    base_url     (optional)  OpenAI-compatible endpoint; absent = the OpenAI default (identical to before)
    api_key_env  (optional)  NAME of the env var holding the key; absent = OPENAI_API_KEY. Only the name lives in
                             config; the value is read from the environment at call time and never logged.
With neither optional key set, make_client() returns ask_service._get_client() unchanged, so behaviour (latch,
timeouts, caching, test monkeypatches) is exactly as before.

Cost: a model with no pricing record is an error everywhere (no silent $0). A free model carries an explicit
record with input = output = 0 in config/model-pricing.json.
"""
from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from vera.pricing.config import CONFIG_DIR, PricingRecord, latest_pricing_for, load_model_pricing

RUN_CONFIG_PATH = CONFIG_DIR / "vera_eval_run.json"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"


class GeneratorConfigError(RuntimeError):
    pass


class UnpricedModelError(RuntimeError):
    """The model has no record in config/model-pricing.json (fail closed; never report an unpriced cost as $0)."""


@dataclass(frozen=True)
class GeneratorSettings:
    model: str
    base_url: str | None = None
    api_key_env: str = DEFAULT_API_KEY_ENV

    @property
    def is_default_provider(self) -> bool:
        return self.base_url is None and self.api_key_env == DEFAULT_API_KEY_ENV


def generator_settings(run_path: Path | str | None = None) -> GeneratorSettings:
    path = Path(run_path) if run_path else RUN_CONFIG_PATH
    try:
        g = json.loads(path.read_text(encoding="utf-8")).get("generator")
    except (OSError, ValueError) as exc:
        raise GeneratorConfigError(f"cannot read the generator block from {path}: {exc!r}") from exc
    if not isinstance(g, dict) or not isinstance(g.get("model"), str) or not g["model"].strip():
        raise GeneratorConfigError(f"{path}: generator.model missing or not a non-empty string")
    base_url, key_env = g.get("base_url"), g.get("api_key_env", DEFAULT_API_KEY_ENV)
    if base_url is not None and (not isinstance(base_url, str) or not base_url.startswith("https://")):
        raise GeneratorConfigError(f"{path}: generator.base_url must be an https:// URL string when set")
    if not isinstance(key_env, str) or not key_env.strip():
        raise GeneratorConfigError(f"{path}: generator.api_key_env must be an env-var NAME (never a key value)")
    return GeneratorSettings(g["model"].strip(), base_url, key_env.strip())


def generator_model(run_path: Path | str | None = None) -> str:
    return generator_settings(run_path).model


@functools.lru_cache(maxsize=1)
def default_generator_model() -> str:
    """generator.model of config/vera_eval_run.json, read at FIRST USE (not at import) and then cached, so modules on
    the /ask import path start even when the eval run config is absent (FIXR-7DA-1028, VER-REL7DA-1-1028 #3). A read
    error is raised to the caller that needs the model and is not cached."""
    return generator_model()


def lazy_module_attr(module_globals: dict, name: str) -> str:
    """The model for a module's lazy constant `name`: an explicitly set module attribute wins (tests/overrides),
    else default_generator_model()."""
    return module_globals.get(name) or default_generator_model()


def make_client(settings: GeneratorSettings | None = None, *, env: Mapping[str, str] | None = None,
                client_cls: Callable | None = None, **client_kwargs):
    """Client for the configured generator. Default provider: the existing ask_service client. Otherwise an
    OpenAI-compatible client at base_url with the key read from the named env var (error names the var only)."""
    s = settings or generator_settings()
    if s.is_default_provider and client_cls is None and not client_kwargs:
        from vera.ask_service import _get_client
        return _get_client()
    if client_cls is None:
        from openai import OpenAI as client_cls
    if s.is_default_provider:  # OpenAI defaults: the SDK reads OPENAI_API_KEY itself, exactly as before
        return client_cls(**client_kwargs)
    key = (env if env is not None else os.environ).get(s.api_key_env)
    if not key:
        raise GeneratorConfigError(f"generator key env var {s.api_key_env!r} is not set")
    kw = dict(client_kwargs)
    kw.setdefault("max_retries", 0)
    if s.base_url:
        kw["base_url"] = s.base_url
    return client_cls(api_key=key, **kw)


def pricing_for(model: str) -> PricingRecord:
    rec = latest_pricing_for(model, load_model_pricing())
    if rec is None:
        raise UnpricedModelError(
            f"No pricing record for {model!r} in config/model-pricing.json; refusing to report an unpriced cost "
            "as $0. A free model needs an explicit record with input=0 and output=0.")
    return rec


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rec = pricing_for(model)
    return round(prompt_tokens / 1e6 * rec.input + completion_tokens / 1e6 * rec.output, 6)
