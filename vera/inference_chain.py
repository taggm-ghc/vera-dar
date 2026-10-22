"""Item #77: free-tier-first provider chain for /ask (scope router + answer) only.

Entries come from config/ask-provider-chain.json and are tried in order; an entry whose key env var is unset
is skipped. The selected OpenAI model is the last resort ONLY when `openai_fallback.enabled` is true in that file
(shipped false; missing, non-bool or unreadable means off). With it off the chain holds free entries only and
the API process never makes an OpenAI call. Any OpenAI-SDK error (rate limit, timeout, auth, 5xx, and 400s such as model_decommissioned) or an
empty answer moves the request to the next entry; the last entry's error propagates to main.py's existing
HTTP mapping. Logs name the provider, model and error class only, never message text or keys.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI, OpenAIError

from vera.ask_service import AskResult, answer_in_scope
from vera.pricing.config import PricingRecord

logger = logging.getLogger("vera")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "ask-provider-chain.json"


class EmptyAnswerError(OpenAIError):
    """A provider returned no answer text; treated like a provider failure so the chain moves on."""


@dataclass(frozen=True)
class ChainEntry:
    provider: str
    model: str
    pricing: PricingRecord
    base_url: str | None = None  # None: the default OpenAI client (last resort)
    api_key_env: str | None = None
    extra_body: dict = field(default_factory=dict)
    timeout_s: float = 20.0

    @property
    def label(self) -> str:
        return self.model if self.base_url is None else f"{self.provider}:{self.model}"

    def client(self) -> OpenAI | None:
        if self.base_url is None:
            return None
        from vera import rate_limit  # RATE-1022 A9: a skip surfaces as APIConnectionError -> next chain entry
        hc = rate_limit.openai_http_client(self.base_url, self.timeout_s)
        return OpenAI(base_url=self.base_url, api_key=os.environ[self.api_key_env], max_retries=0,
                      timeout=self.timeout_s, **({"http_client": hc} if hc is not None else {}))


def openai_fallback_enabled(path: Path = CONFIG_PATH) -> bool:
    """The one reader of the OpenAI-fallback switch (`openai_fallback.enabled`). Anything but a literal JSON true
    (missing block or key, non-bool, unreadable or malformed file) is OFF. No environment override."""
    try:
        raw = json.loads(Path(path).read_text())
        return raw["openai_fallback"]["enabled"] is True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def load_chain(openai_model: str, openai_pricing: PricingRecord | None, path: Path = CONFIG_PATH,
               env=os.environ, openai_enabled: bool | None = None) -> list[ChainEntry]:
    """Free entries that have a key, then OpenAI (only when the fallback is enabled and the model is priced).
    openai_enabled=None reads the switch from the `path` file being loaded (not CONFIG_PATH); pass True/False to
    override. An unreadable file gives an empty chain (no free entries, no OpenAI unless explicitly enabled)."""
    entries: list[ChainEntry] = []
    if openai_enabled is None:
        openai_enabled = openai_fallback_enabled(path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Provider chain config unreadable (%s); /ask has no free providers", type(exc).__name__)
        raw = {}
    timeout = float(raw.get("timeout_s", 20))
    checked = str(raw.get("checked_at", "unknown"))
    for p in raw.get("providers", []):
        if not (env.get(p["api_key_env"]) or "").strip():
            continue
        pricing = PricingRecord(provider=p["provider"], model=p["model"], input=p["input_per_1m"],
                                output=p["output_per_1m"], source_url=str(path.name),
                                retrieved_at=checked, effective_from=checked)
        entries.append(ChainEntry(p["provider"], p["model"], pricing, p["base_url"], p["api_key_env"],
                                  dict(p.get("extra_body") or {}), timeout))
    if openai_enabled and openai_pricing is not None:
        entries.append(ChainEntry("openai", openai_model, openai_pricing))
    return entries


def answer_via_chain(question: str, entries: list[ChainEntry], grounding=None) -> tuple[AskResult, str]:
    """Return (result, label of the entry that answered). Raises the last entry's error if all fail."""
    if not entries:
        raise RuntimeError("no provider available for /ask")
    for i, e in enumerate(entries):
        last = i == len(entries) - 1
        try:
            kw = {"grounding": grounding} if grounding is not None else {}
            result = answer_in_scope(question, e.model, e.pricing, client=e.client(), extra_body=e.extra_body or None,
                                     **kw)
            if not (result.answer or "").strip():
                raise EmptyAnswerError(f"empty answer from {e.label}")
            return result, e.label
        except OpenAIError as exc:
            if last:
                raise
            logger.warning("/ask provider %s failed (%s, status %s, code %s); trying next", e.label,
                           type(exc).__name__, getattr(exc, "status_code", "-"), getattr(exc, "code", "-"))
    raise AssertionError("unreachable")
