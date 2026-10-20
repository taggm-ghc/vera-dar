"""Licence gate (R1 #63, item 72 W4): decides, per candidate, whether its text may be stored.

R1 rule (2026-10-03): reject ONLY No-Derivatives (ND), no-educational-use and strictly-for-fee licences;
plain NonCommercial (NC) is ACCEPTED; no declared licence is deferred (held) for human review.
Term lists live in config/licence_rules.json. Detection is by the DECLARED licence id/url and the known
markers only; page text is never scanned. The GOVERNING licence depends on what fetch will store:
  * stored_text_kind == "abstract_metadata" (the arXiv abstract page): the METADATA licence governs (CC0);
  * anything else (full content): the CONTENT licence governs. A metadata licence (e.g. CC0 on an abstract)
    is NEVER permission to store content.
A rejecting marker on either record rejects. arXiv e-print URLs (pdf, html, e-print) are rejected outright.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlsplit

RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "licence_rules.json"

# Why: arXiv's terms forbid storing/serving e-prints without permission; only /abs/ pages may be stored.
ARXIV_HOSTS = ("arxiv.org", "www.arxiv.org", "export.arxiv.org")
ARXIV_ALLOWED_PREFIX = "/abs/"
METADATA_KIND = "abstract_metadata"


def _load_rules(path: Path = RULES_PATH) -> dict:
    try:
        r = json.loads(path.read_text(encoding="utf-8"))
        return {k: [str(x).lower() for x in r[k]] for k in
                ("nd_tokens", "no_educational_use_markers", "fee_only_markers")}
    except (OSError, ValueError, KeyError, TypeError) as e:  # fail verbosely: a gate without rules must not run
        raise RuntimeError(f"licence rules unreadable at {path}: {e!r}") from e


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _has_run(tokens: list[str], phrase: list[str]) -> bool:
    n = len(phrase)
    return n > 0 and any(tokens[i:i + n] == phrase for i in range(len(tokens) - n + 1))


def reject_reason(licence_id: str | None, licence_url: str | None = None) -> str | None:
    """Return 'ND', 'no educational use' or 'fee only' when the declared id/url carries that marker, else None.
    Plain NC is not a reason."""
    rules = _load_rules()
    toks = _tokens(f"{licence_id or ''} {licence_url or ''}")
    if set(rules["nd_tokens"]) & set(toks):
        return "ND (no derivatives)"
    if any(_has_run(toks, _tokens(m)) for m in rules["no_educational_use_markers"]):
        return "no educational use"
    if any(_has_run(toks, _tokens(m)) for m in rules["fee_only_markers"]):
        return "fee only"
    return None


def licence_decision(c: dict) -> tuple[str, str]:
    sp = urlsplit(c.get("url", ""))
    if sp.netloc.lower() in ARXIV_HOSTS and not sp.path.startswith(ARXIV_ALLOWED_PREFIX):
        return "reject", "arXiv e-print/PDF/HTML is never stored (arXiv terms); only the abstract page"
    lic = c.get("licence") or {}
    meta, content = lic.get("metadata") or {}, lic.get("content") or {}
    for rec in (meta, content):
        why = reject_reason(rec.get("id"), rec.get("url")) if (rec.get("id") or rec.get("url")) else None
        if why:
            return "reject", f"licence {rec.get('id') or rec.get('url')} is {why} (R1 rule)"
    governing = meta if c.get("stored_text_kind") == METADATA_KIND else content
    if not governing.get("id"):
        return "defer", "no declared licence for the stored text: held for review (#63)"
    return "allow", f"licence {governing['id']}"


def apply_licence_gate(scored: list[dict]) -> list[dict]:
    """Mutates and returns `scored`. A Gate A reject is kept; otherwise reject/defer override a fetch."""
    for c in scored:
        d, why = licence_decision(c)
        c["licence_decision"] = d
        if d != "allow" and c.get("gate_a_decision") != "reject":
            c["gate_a_decision"] = d
        c["gate_a_rationale"] = f"{c.get('gate_a_rationale', '')} Licence: {why}."
    return scored
