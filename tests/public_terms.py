"""Shared public-text term list (K3.I, plan K3 r2 R2-C).

Built at test time from config so a new provider or model is covered without editing a list:
`config/ask-provider-chain.json` providers, grounding providers, claim-check checkers (provider, model, family and every
"/"-split segment), the `openrouter_example` key, and `config/model-selection.json` `selected_model`; plus short
static extras (vendor, framework, environment and internal-stage terms). Import from tests of every public surface.

API: `public_terms() -> tuple[str, ...]` (lower-case, unique), `find_terms(text, terms=None) -> list[str]`
(whole-token, case-insensitive matches). A caller that has a documented exemption filters the tuple itself.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAIN = ROOT / "config" / "ask-provider-chain.json"
SELECTION = ROOT / "config" / "model-selection.json"

STATIC_TERMS = (
    "groq", "openai", "anthropic", "claude", "llama", "mistral", "gemini", "gpt-oss", "gpt", "nano", "qwen",
    "local-inference", "duckduckgo", "openalex", "doaj", "in the order tried", "provider chain", "model-selection",
    "snippet_max", "grounded_ask", "gate a", "gate b", "gate c", "gate_a", "gate_b", "gate_c", "write gate",
    "licence gate", "admitted by", "scorer", "rubric_version", "policy_version", "streamlit app", "vera_demo_source",
    "external_db_url", "vera_vjay", "fixture fallback", "onrender", "127.0.0.1", "localhost", "traceback",
)


def _read(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _add(out: list, value) -> None:
    if not isinstance(value, str):
        return
    v = value.strip().lower()
    if not v or v.startswith("<") or len(v) < 3:
        return
    out.append(v)
    out.extend(seg for seg in v.split("/") if len(seg) >= 3 and not seg.startswith("<"))


def config_terms(chain: Path = CHAIN, selection: Path = SELECTION) -> list[str]:
    cfg, out = _read(chain), []
    for p in cfg.get("providers", []):
        _add(out, p.get("provider")), _add(out, p.get("model"))
    ex = cfg.get("openrouter_example")
    if ex is not None:
        _add(out, "openrouter_example"), _add(out, ex.get("provider")), _add(out, ex.get("model"))
    g = cfg.get("grounding", {})
    for name in g.get("providers", []):
        _add(out, name)
    for c in (g.get("claim_check") or {}).get("checkers", []):
        for k in ("provider", "model", "family"):
            _add(out, c.get(k))
    _add(out, _read(selection).get("selected_model"))
    return out


def public_terms(chain: Path = CHAIN, selection: Path = SELECTION) -> tuple[str, ...]:
    return tuple(dict.fromkeys(config_terms(chain, selection) + list(STATIC_TERMS)))


def find_terms(text: str, terms=None) -> list[str]:
    """Terms found in `text` as whole tokens (a term is not matched inside a longer alphanumeric run)."""
    low = str(text).lower()
    hits = []
    for t in (public_terms() if terms is None else terms):
        if re.search(r"(?<![a-z0-9])" + re.escape(t.lower()) + r"(?![a-z0-9])", low):
            hits.append(t)
    return hits
