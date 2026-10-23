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
# Owner decision ARXIVHTML-1023 (ARX-SECREV-1023 R1, R3): the ONE exception to the rule above is
# https://arxiv.org/html/<id>vN carrying the arXiv route's version-bound licence evidence (licence.arxiv_evidence,
# set only by vera.arxiv_fulltext) for the SAME id and version, whose CC licence URL is in arxiv_html_cc_families.
_ARXIV_NEW_ID = re.compile(r"\d{4}\.\d{4,5}")
_ARXIV_VERSION = re.compile(r"v[1-9]\d*")
_CC_LICENCE_URL = re.compile(r"https://creativecommons\.org/licenses/([a-z\-]+)/\d\.\d/?")


def _load_rules(path: Path = RULES_PATH) -> dict:
    try:
        r = json.loads(path.read_text(encoding="utf-8"))
        return {k: [str(x).lower() for x in r[k]] for k in
                ("nd_tokens", "no_educational_use_markers", "fee_only_markers")}
    except (OSError, ValueError, KeyError, TypeError) as e:  # fail verbosely: a gate without rules must not run
        raise RuntimeError(f"licence rules unreadable at {path}: {e!r}") from e


# F-1 (ARX-SECREV-1023 R3, C1): Unicode full-stop variants that IDNA treats as label separators.
_UNICODE_DOTS = ("\u3002", "\uff0e", "\uff61")


def _arxiv_host(sp) -> str:
    """F-1 (C1): the host used to DETECT arXiv, never to admit: urlsplit hostname, lowercased, Unicode full-stop variants
    mapped to ".", trailing dots stripped, then IDNA-encoded (str.encode("idna"), encodings.idna: nameprep + punycode)
    and lowercased. "" when the URL has no host. Raises on any normalisation error; the caller defers (fail closed)."""
    h = (sp.hostname or "").lower()
    for d in _UNICODE_DOTS:
        h = h.replace(d, ".")
    h = h.rstrip(".")
    return h.encode("idna").decode("ascii").lower() if h else ""


def _arxiv_html_cc_ok(c: dict, sp) -> bool:
    """True only for https://arxiv.org/html/<id>vN(/) when c["licence"]["arxiv_evidence"] (built by the arXiv route from
    the abs page of that exact id and version) names the SAME id and version, its abs_url is that version's abs page,
    its licence_url equals the content licence URL, and that URL is a CC licence whose family is listed in config
    arxiv_html_cc_families (a JSON list). All regexes are full matches. Fails closed: anything absent, mismatched,
    malformed or unreadable, or any error, means False and the refusal stands."""
    try:
        # F-1 (C3): raw hostname AND raw netloc exactly "arxiv.org": no port, userinfo, case, trailing or Unicode dot.
        if sp.scheme != "https" or sp.hostname != "arxiv.org" or sp.netloc != "arxiv.org" or sp.query or sp.fragment:
            return False
        lic = c.get("licence")
        ev = lic.get("arxiv_evidence") if isinstance(lic, dict) else None
        content = lic.get("content") if isinstance(lic, dict) else None
        if not isinstance(ev, dict) or not isinstance(content, dict):
            return False
        aid, ver, curl = ev.get("id"), ev.get("version"), content.get("url")
        if not (isinstance(aid, str) and isinstance(ver, str) and isinstance(curl, str)):
            return False
        if not (_ARXIV_NEW_ID.fullmatch(aid) and _ARXIV_VERSION.fullmatch(ver)):
            return False
        if ev.get("abs_url") != f"https://arxiv.org/abs/{aid}{ver}" or ev.get("licence_url") != curl:
            return False
        if not re.fullmatch(rf"/html/{re.escape(aid)}{re.escape(ver)}/?", sp.path):
            return False
        fams = json.loads(RULES_PATH.read_text(encoding="utf-8"))["arxiv_html_cc_families"]
        if not isinstance(fams, list):
            return False
        m = _CC_LICENCE_URL.fullmatch(curl)
        return bool(m) and m.group(1) in {str(x).lower() for x in fams}
    except Exception:  # noqa: BLE001  fail closed: any error keeps the arXiv refusal
        return False


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
    try:
        sp = urlsplit(c.get("url", ""))
        host = _arxiv_host(sp)
    except Exception as e:  # noqa: BLE001  F-1 (C1): a host that cannot be normalised is held, never allowed
        return "defer", f"URL host could not be normalised ({type(e).__name__}): held for review (F-1, fail closed)"
    # F-1 (C2): any arXiv host whose RAW netloc is not exactly the normalised host (port incl. :443, userinfo, case,
    # trailing dot, Unicode dots) is refused on every path, /abs/ included.
    if host in ARXIV_HOSTS and (sp.netloc != host or (
            not sp.path.startswith(ARXIV_ALLOWED_PREFIX) and not _arxiv_html_cc_ok(c, sp))):
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
