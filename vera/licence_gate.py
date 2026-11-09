"""Licence gate (R1 #63, item 72 W4): decides, per candidate, whether its text may be stored.

R1 rule (2026-10-03): reject ONLY No-Derivatives (ND), no-educational-use and strictly-for-fee licences;
plain NonCommercial (NC) is ACCEPTED; no declared licence is deferred (held) for human review.
Term lists live in config/licence_rules.json. Detection is by the DECLARED licence id/url and the known
markers only; page text is never scanned. The GOVERNING licence depends on what fetch will store:
  * stored_text_kind == "abstract_metadata" (the arXiv abstract page): the METADATA licence governs (CC0);
  * anything else (full content): the CONTENT licence governs. A metadata licence (e.g. CC0 on an abstract)
    is NEVER permission to store content.
A rejecting marker on either record rejects. arXiv e-print URLs (pdf, html, e-print) are rejected outright.

DENY-ALL ALLOWLIST (LICENCE-ALLOWLIST-1104, owner 2026-11-04: "the human standard is Deny All; Approve specified
exceptions-from-Deny"; audit item E1). After the reject markers, the governing licence is STORABLE ("allow") only when
its normalised id (or, when the id is unrecognised, its normalised licence URL) matches an ACTIVE allowlist entry with
storable=true in config/licence_allowlist.json (the seed; a DB table is drafted, not applied). Recognised but
non-storable licences ("All rights reserved", publisher user/TDM licences, the arXiv distribution licence) and
unknown licences are link-only ("defer", never stored). Entries in any status other than "active" (e.g. "proposed"
rows written by scripts/licence_research.py) are never used. Plain NC stays accepted (R1); the three reject classes
stay rejected. Every decision carries a reason code (licence_reason_code) that the /ask path writes to
licence_observations, so unknown licence strings are counted there (reason_code 'licence_unknown').

FAIR-USE SHORT-QUOTE TIER (FAIRUSE-TIER-1108, owner decision OD-1108-FAIRUSE, option O2; vera/fair_use.py, config
fair_use.json, ships OFF): when enabled, a governing licence that matches an ACTIVE non-storable entry listed in
eligible_licence_ids (all rights reserved, the arXiv default licence) is "allow" with reason code
'licence_fair_use_quote', but only with the acquisition route's recorded access check (robots allowed, no TDM
reservation, no login). The reject markers run first; unknown, missing and conflicting licences are never covered.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlsplit

RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "licence_rules.json"
ALLOWLIST_PATH = Path(__file__).resolve().parent.parent / "config" / "licence_allowlist.json"
ACTIVE = "active"
# Reason codes (licence_observations.reason_code). 'licence_unknown' rows are the research queue.
RC_ALLOW, RC_REJECT, RC_ARXIV = "licence_allow", "licence_reject", "licence_reject_arxiv_url"
RC_MISSING, RC_UNKNOWN, RC_NOT_STORABLE = "licence_missing", "licence_unknown", "licence_not_storable"
RC_CONFLICT, RC_HOST = "licence_id_url_conflict", "licence_host_unnormalisable"
RC_ND_NEWS = "licence_nd_news_answer_only"  # H19 (OD-1106-NEWS): ND news with a satisfied access verdict; never stored
RC_US_FED_PD = "licence_us_federal_pd"  # GOV-PD-1108: no declared licence; US federal work (17 U.S.C. 105), mode on only
RC_FAIR_USE = "licence_fair_use_quote"  # FAIRUSE-TIER-1108 (OD-1108-FAIRUSE, O2): short-quote evidence tier, flag-gated

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
        if not m:  # FAIRUSE-TIER-1108: the arXiv default licence, only while the fair-use tier is enabled
            from vera import fair_use
            return fair_use.arxiv_default_url_ok(curl)
        return m.group(1) in {str(x).lower() for x in fams}
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


_CC_LICENCE_TEXT = re.compile(
    r"(?:cc|creative-commons)-(by|by-sa|by-nc|by-nc-sa|by-nd|by-nc-nd)(?:-(\d\.\d))?"
    r"(?:-(?:international|unported|generic))?(?:-(?:license|licence|public-license))?")
_CC_ZERO_TEXT = re.compile(r"(?:cc-?0|cc-zero|creative-commons-zero)(?:-(1\.0))?(?:-universal)?")


def norm_id(text: str | None) -> str:
    """Normalised licence id: lowercase, runs of anything but [a-z0-9.] become '-', '-' trimmed. A CC id written in any
    common spelling ("CC BY 4.0", "CC_BY_NC_4.0", "Creative Commons Attribution-NonCommercial 4.0 International" is NOT
    parsed: only the cc-<family>[-<version>] forms) becomes "cc-<family>[-<version>]" ; CC0 forms become "cc0-1.0"."""
    t = re.sub(r"(?<![0-9])\.|\.(?![0-9])", "-", str(text or "").lower())  # keep only version dots (4.0)
    t = re.sub(r"[^a-z0-9.]+", "-", t).strip("-.")
    m = _CC_LICENCE_TEXT.fullmatch(t)
    if m:
        return f"cc-{m.group(1)}" + (f"-{m.group(2)}" if m.group(2) else "")
    if _CC_ZERO_TEXT.fullmatch(t):
        return "cc0-1.0"
    return t


def norm_url(url: str | None) -> str:
    """Normalised licence URL: lowercase host+path without scheme, 'www.', query, fragment, trailing '/', and a final
    legalcode[.xx] / deed[.xx] segment. "" when it is not an http(s) URL."""
    s = str(url or "").strip().lower()
    if not re.match(r"https?://", s):
        return ""
    sp = urlsplit(s)
    host = (sp.hostname or "").removeprefix("www.")
    path = re.sub(r"/(?:legalcode|deed)(?:\.[a-z\-]+)?/?$", "", sp.path).rstrip("/")
    return f"{host}{path}" if host else ""


def _load_allowlist(path: Path | None = None) -> tuple[dict, dict]:
    """(by_alias, by_url) over ACTIVE entries only. Fails verbosely: a gate without its allowlist must not run."""
    p = path or ALLOWLIST_PATH
    try:
        entries = json.loads(p.read_text(encoding="utf-8"))["entries"]
        by_alias, by_url = {}, {}
        for e in entries:
            if e.get("status") != ACTIVE:
                continue  # deny-all: proposed / suspended / retired rows are never used
            for a in [e["id"], *e.get("aliases", [])]:
                by_alias.setdefault(norm_id(a), e)
            for u in e.get("urls", []):
                by_url.setdefault(norm_url(u), e)
        by_alias.pop("", None)
        by_url.pop("", None)
        return by_alias, by_url
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise RuntimeError(f"licence allowlist unreadable at {p}: {e!r}") from e


def _terms(e: dict) -> tuple:
    return tuple(e.get(k) for k in ("storable", "nc", "nd", "sa"))


def match_entry(licence_id: str | None, licence_url: str | None = None, path: Path | None = None):
    """(entry_or_None, how): how is 'id', 'url', 'conflict' or 'none'. The id governs; the URL is used only when the id
    is unrecognised. An id and URL that resolve to entries with different terms (storable, NC, ND, SA) are a conflict
    (no entry)."""
    by_alias, by_url = _load_allowlist(path)
    e_id = by_alias.get(norm_id(licence_id)) if licence_id else None
    e_url = by_url.get(norm_url(licence_url)) if licence_url else None
    if e_id and e_url and _terms(e_id) != _terms(e_url):  # e.g. "CC BY" + a .../by/4.0/ URL agree; BY-SA + BY do not
        return None, "conflict"
    if e_id:
        return e_id, "id"
    if e_url:
        return e_url, "url"
    return None, "none"


def nd_news_enabled(path: Path = RULES_PATH) -> bool:
    """H19: config/licence_rules.json nd_news.enabled is literally True. Missing, unreadable or anything else = False."""
    try:
        nd = json.loads(path.read_text(encoding="utf-8")).get("nd_news")
        return isinstance(nd, dict) and nd.get("enabled") is True
    except (OSError, ValueError, AttributeError):
        return False


def _other_reject(rec: dict) -> bool:
    """True when the record carries a no-educational-use or fee-only marker (those never get the ND news exception)."""
    rules = _load_rules()
    toks = _tokens(f"{rec.get('id') or ''} {rec.get('url') or ''}")
    return any(_has_run(toks, _tokens(m)) for m in rules["no_educational_use_markers"] + rules["fee_only_markers"])


def nd_news_permitted(c: dict) -> bool:
    """H19 (design section 2): True only when nd_news.enabled, the candidate has a CONTENT licence (the governing record
    is not the abstract-metadata one), that record normalises to a listed ND family, it carries no other reject marker,
    and the verdict looked up HERE from the URL host via news_access (A1 to A7, AD-6, NC-BASIS-1) is full_text with the
    same family. The metadata licence, c["source_class"], c["access"] and c["operations"] are never read (forgeable
    upstream). A licence version of 3.0 or unknown is the strictest reading and changes nothing here: the exception is
    answer-only either way, and operations stay governed by news_access.operation_verdict. Any error = False."""
    try:
        if not nd_news_enabled() or c.get("stored_text_kind") == METADATA_KIND:
            return False
        lic = c.get("licence")
        content = lic.get("content") if isinstance(lic, dict) else None
        url = c.get("url")
        if not isinstance(content, dict) or not isinstance(url, str) or not (content.get("id") or content.get("url")):
            return False
        if reject_reason(content.get("id"), content.get("url")) != "ND (no derivatives)" or _other_reject(content):
            return False
        from vera import news_access  # lazy: news_access imports this module
        fam = news_access._family(content.get("id"))
        rules = json.loads(RULES_PATH.read_text(encoding="utf-8")).get("nd_news") or {}
        if not content.get("id") or fam not in {news_access._family(f) for f in rules.get("families") or []}:
            return False
        cfg = news_access.reporting_lane.load_config()
        row = news_access.reporting_lane.outlet_row(url, cfg)
        acc_fam = news_access._family((((row or {}).get("access") or {}).get("licence") or {}).get("id"))
        if not acc_fam or acc_fam != fam:
            return False
        page = {"licence_href": content["url"]} if isinstance(content.get("url"), str) and content["url"] else None
        return news_access.mode(url, cfg, page) == news_access.FULL_TEXT
    except Exception:  # noqa: BLE001  fail closed
        return False


def _us_federal_pd(c: dict) -> dict | None:
    """GOV-PD-1108: the us-federal-public-domain decision when vera.gov_public_domain is in mode 'on' and allows this
    candidate's full content (host from data, recomputed from c["url"]; page test on the page text), else None.
    Any error = None (the missing-licence hold stands)."""
    try:
        from vera import gov_public_domain
        return gov_public_domain.gate_allow(c)
    except Exception:  # noqa: BLE001  fail closed
        return None


def _fair_use(entry: dict, c: dict) -> bool:
    """FAIRUSE-TIER-1108 (OD-1108-FAIRUSE, O2): vera.fair_use.admits. Any error = False (link-only stands)."""
    try:
        from vera import fair_use
        return fair_use.admits(entry, c)
    except Exception:  # noqa: BLE001  fail closed
        return False


def licence_status(c: dict) -> tuple[str, str, str]:
    """(decision, rationale, reason_code). decision is 'allow', 'defer' (link-only / held) or 'reject'."""
    try:
        sp = urlsplit(c.get("url", ""))
        host = _arxiv_host(sp)
    except Exception as e:  # noqa: BLE001  F-1 (C1): a host that cannot be normalised is held, never allowed
        return ("defer", f"URL host could not be normalised ({type(e).__name__}): held for review (F-1, fail closed)",
                RC_HOST)
    # F-1 (C2): any arXiv host whose RAW netloc is not exactly the normalised host (port incl. :443, userinfo, case,
    # trailing dot, Unicode dots) is refused on every path, /abs/ included.
    if host in ARXIV_HOSTS and (sp.netloc != host or (
            not sp.path.startswith(ARXIV_ALLOWED_PREFIX) and not _arxiv_html_cc_ok(c, sp))):
        return "reject", "arXiv e-print/PDF/HTML is never stored (arXiv terms); only the abstract page", RC_ARXIV
    lic = c.get("licence") or {}
    meta, content = lic.get("metadata") or {}, lic.get("content") or {}
    nd_news_ok = False
    for rec in (meta, content):
        why = reject_reason(rec.get("id"), rec.get("url")) if (rec.get("id") or rec.get("url")) else None
        if why and rec is content and why.startswith("ND") and nd_news_permitted(c):  # H19: flag-gated, off by default
            nd_news_ok = True
            continue
        if why:
            return "reject", f"licence {rec.get('id') or rec.get('url')} is {why} (R1 rule)", RC_REJECT
    if nd_news_ok:
        return ("defer", f"licence {content.get('id') or content.get('url')} is ND news with a satisfied access verdict: "
                         "answer-only, not stored (OD-1106-NEWS H19, R-037)", RC_ND_NEWS)
    governing = meta if c.get("stored_text_kind") == METADATA_KIND else content
    if not governing.get("id"):
        pd = _us_federal_pd(c) if governing is content and not content.get("url") else None
        if pd is not None:  # GOV-PD-1108: licence status only; needs the bound fetched full text (access gated on fetch)
            return ("allow", f"US federal government work, 17 U.S.C. 105 (rule {pd.get('rule')}, {pd.get('basis')}); "
                             f"no licence declared on the page", RC_US_FED_PD)
        return "defer", "no declared licence for the stored text: held for review (#63)", RC_MISSING
    entry, how = match_entry(governing.get("id"), governing.get("url"))
    if how == "conflict":
        return ("defer", f"licence id {governing['id']} and URL name different licences: link-only (deny-all)",
                RC_CONFLICT)
    if entry is None:
        return ("defer", f"licence {governing['id']} is not on the allowlist: link-only, queued for licence research "
                         "(deny-all)", RC_UNKNOWN)
    if entry.get("reject") is True:  # a recognised licence in an R1 reject class (e.g. a no-derivatives term)
        return "reject", f"licence {governing['id']} ({entry['id']}) is rejected: {entry.get('reason')}", RC_REJECT
    if entry.get("storable") is not True:
        if _fair_use(entry, c):  # FAIRUSE-TIER-1108: flag on, eligible licence, route-recorded access check
            from vera.fair_use import BASIS
            return ("allow", f"licence {governing['id']} ({entry['id']}): {BASIS}: short quotes with attribution only",
                    RC_FAIR_USE)
        return ("defer", f"licence {governing['id']} ({entry['id']}) is not storable: link-only (deny-all)",
                RC_NOT_STORABLE)
    return "allow", f"licence {governing['id']}", RC_ALLOW  # rationale text unchanged (downstream contract)


def licence_decision(c: dict) -> tuple[str, str]:
    d, why, _code = licence_status(c)
    return d, why


def _gov_pd_shadow(c: dict, code: str) -> None:
    """GOV-PD-1108 shadow mode: for a missing-licence candidate, record what the us-federal-public-domain rule would
    decide (reason code only) in c["gov_pd_shadow"]; the decision itself is unchanged. Off/on: nothing recorded."""
    try:
        from vera import gov_public_domain as g
        if code != RC_MISSING or g.mode() != "shadow":
            return
        d = g.decide(str(c.get("url") or ""), g.candidate_text(c), c.get("licence_href") or "")
        c["gov_pd_shadow"] = d.get("reason_code")
    except Exception:  # noqa: BLE001  shadow must never affect the gate
        pass


def apply_licence_gate(scored: list[dict]) -> list[dict]:
    """Mutates and returns `scored`. A Gate A reject is kept; otherwise reject/defer override a fetch."""
    for c in scored:
        d, why, code = licence_status(c)
        c["licence_decision"] = d
        c["licence_reason_code"] = code
        _gov_pd_shadow(c, code)
        if d != "allow" and c.get("gate_a_decision") != "reject":
            c["gate_a_decision"] = d
        c["gate_a_rationale"] = f"{c.get('gate_a_rationale', '')} Licence: {why}."
    return scored
