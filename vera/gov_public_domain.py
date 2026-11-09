"""GOV-PD-1108: licence rule "us-federal-public-domain" (17 U.S.C. 105). LICENCE STATUS ONLY, never access.

Owner 2026-11-08: "US .gov sites should always 'be fair game' for our purposes". Section 105 removes US copyright from
works of the United States Government (federal officers and employees, official duties). It does NOT cover state,
local or tribal government works, contractor or grantee works, copyrights transferred to the government, or
third-party material shown on a government page. So:

  * host test (data, never a hand list): the host is, or is a subdomain of, a domain in the vendored CISA dotgov-data
    snapshot (config/us_gov_domains_federal.json, refreshed by scripts/refresh_us_gov_domains.py) whose Domain type is
    federal, or of a federal zone outside .gov (.mil, fed.us; config/us_federal_pd.json with RFC evidence); a host under
    exclude_hosts (federal repositories of third-party works) never qualifies;
  * page test (narrowing only): a declared licence link, a copyright notice, a third-party text credit or a contractor
    or grantee statement in the page's visible text means the page falls back to the normal licence gate. An image
    credit is recorded but does not fall back: VERA-DAR uses page text only;
  * non-federal .gov (state, city, county, ...) and unknown hosts: no change at all.

Access is never decided here: robots.txt (read first), TDM reservations, Content-Signal/noai, terms, deny list, rate
limits and the API allowlist run before this module is consulted and are untouched by it. Pure: no I/O except reading
the two config files; never raises to a caller (any error = rule not applied, fail closed).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = _ROOT / "config" / "us_federal_pd.json"
MODES = ("off", "shadow", "on")
RULE_ID = "us-federal-public-domain"  # fallback only; config rule_id governs

# Reason codes (dev/trace fields; never page text, never a host).
RC_APPLIES = "gov_pd_applies"
RC_NOT_FEDERAL = "gov_pd_not_federal"          # any non-federal host, .gov state/local included
RC_GOV_NON_FEDERAL = "gov_pd_gov_non_federal"  # a .gov host in no federal row (state, city, county, unlisted ...)
RC_EXCLUDED = "gov_pd_excluded_host"
RC_DECLARED = "gov_pd_declared_licence"
RC_COPYRIGHT = "gov_pd_copyright_notice"
RC_THIRD_PARTY = "gov_pd_third_party_credit"
RC_CONTRACTOR = "gov_pd_contractor_statement"
RC_NO_TEXT = "gov_pd_page_unchecked"
RC_STALE = "gov_pd_dataset_stale"
RC_ERROR = "gov_pd_error"
RC_OFF = "gov_pd_off"

_UNICODE_DOTS = ("。", "．", "｡")
_cache: dict = {}


def reset_for_tests() -> None:
    _cache.clear()


def load_config(path: Path | None = None) -> dict:
    p = Path(path or CONFIG_PATH)
    key = ("cfg", str(p))
    if key not in _cache:
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
            _cache[key] = c if isinstance(c, dict) else {}
        except (OSError, ValueError):
            _cache[key] = {}
    return _cache[key]


def mode(cfg: dict | None = None) -> str:
    m = (cfg if cfg is not None else load_config()).get("mode")
    return m if m in MODES else "off"


def _dataset(cfg: dict) -> dict:
    ds = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    rel = str(ds.get("path") or "")
    p = (_ROOT / rel) if rel and not Path(rel).is_absolute() else Path(rel)
    key = ("ds", str(p), json.dumps(cfg.get("exclude_dataset_rows"), sort_keys=True, default=str),
           json.dumps(ds.get("federal_types"), default=str))
    if key not in _cache:
        try:
            snap = json.loads(p.read_text(encoding="utf-8"))
            types = {str(t) for t in ds.get("federal_types") or ()}
            rules = [r for r in cfg.get("exclude_dataset_rows") or () if isinstance(r, dict)]
            fed, dropped = {}, set()
            for row in snap.get("domains") or ():
                name = str(row.get("domain") or "").strip().lower().rstrip(".")
                org, sub = str(row.get("organization") or ""), str(row.get("suborganization") or "")
                if any(_row_excluded(r, org, sub) for r in rules):
                    dropped.add(name)  # narrowing only: e.g. contractor-operated DOE laboratories, USPS, Smithsonian
                    continue
                if name and str(row.get("type") or "") in types:
                    fed[name] = org
            _cache[key] = {"federal": fed, "dropped": dropped, "retrieved": str(snap.get("retrieved_at") or ""),
                           "source_url": str(snap.get("source_url") or ""), "sha256": str(snap.get("sha256") or "")}
        except (OSError, ValueError, TypeError, AttributeError):
            _cache[key] = {"federal": {}, "dropped": set(), "retrieved": "", "source_url": "", "sha256": ""}
    return _cache[key]


def _row_excluded(rule: dict, org: str, sub: str) -> bool:
    """One exclude_dataset_rows rule (all given conditions must hold; a rule with no organization condition never
    matches): organization (exact) or organization_regex (full match), then suborganization_regex (search: exclude on a
    hit) and/or suborganization_allow_regex (full match: exclude unless it matches). Case-insensitive. A malformed rule
    excludes the row (fail closed)."""
    try:
        if rule.get("organization") is not None:
            if org != str(rule["organization"]):
                return False
        elif rule.get("organization_regex"):
            if not re.fullmatch(str(rule["organization_regex"]), org, re.IGNORECASE):
                return False
        else:
            return False
        has_sub = False
        if rule.get("suborganization_regex"):
            has_sub = True
            if re.search(str(rule["suborganization_regex"]), sub, re.IGNORECASE):
                return True
        if rule.get("suborganization_allow_regex"):
            has_sub = True
            if not re.fullmatch(str(rule["suborganization_allow_regex"]), sub.strip(), re.IGNORECASE):
                return True
        return not has_sub  # an organisation-only rule excludes every row of that organisation
    except (re.error, TypeError, KeyError):
        return True


def norm_host(url_or_host: str) -> str:
    """Lowercased, Unicode full stops mapped, trailing dots stripped, IDNA-encoded. "" when unusable, and "" for a URL
    with userinfo, a backslash or a port other than 443 (parsers disagree on those; opus review finding 7)."""
    s = str(url_or_host or "").strip()
    try:
        if "\\" in s:
            return ""
        if "://" in s:
            sp = urlsplit(s)
            if sp.username is not None or sp.password is not None or "@" in sp.netloc or sp.port not in (None, 443):
                return ""
        h = urlsplit(s).hostname if "://" in s else s.split("/")[0].split(":")[0]
        h = (h or "").lower()
        for d in _UNICODE_DOTS:
            h = h.replace(d, ".")
        h = h.rstrip(".")
        return h.encode("idna").decode("ascii").lower() if h else ""
    except (ValueError, UnicodeError):
        return ""


def _under(host: str, suffix: str) -> bool:
    s = str(suffix or "").lower().strip(".")
    return bool(s) and (host == s or host.endswith("." + s))


def _stale(cfg: dict, ds: dict, today: _dt.date | None) -> bool:
    try:
        got = _dt.date.fromisoformat(ds["retrieved"][:10])
        limit = int((cfg.get("dataset") or {}).get("max_age_days"))
        return ((today or _dt.date.today()) - got).days > limit
    except (KeyError, ValueError, TypeError):
        return True  # no date or no bound: treat as stale (fail closed)


@dataclass(frozen=True)
class HostStatus:
    federal: bool
    code: str            # RC_* (no host inside)
    basis: str = ""      # "cisa_dotgov" | "zone:<suffix>" | ""
    organization: str = ""


def host_status(url: str, cfg: dict | None = None, today: _dt.date | None = None) -> HostStatus:
    """Federal or not, from data only. Never raises."""
    try:
        cfg = cfg if cfg is not None else load_config()
        host = norm_host(url)
        if not host:
            return HostStatus(False, RC_ERROR)
        if any(_under(host, (e or {}).get("suffix")) for e in cfg.get("exclude_hosts") or ()):
            return HostStatus(False, RC_EXCLUDED)
        for z in cfg.get("federal_zones_outside_gov") or ():
            if _under(host, (z or {}).get("suffix")):
                return HostStatus(True, RC_APPLIES, "zone:" + str(z.get("suffix")).lower())
        if not host.endswith(".gov"):
            return HostStatus(False, RC_NOT_FEDERAL)
        ds = _dataset(cfg)
        labels = host.split(".")
        for i in range(len(labels) - 1):
            name = ".".join(labels[i:])
            if name in ds["dropped"]:
                return HostStatus(False, RC_EXCLUDED)
            if name in ds["federal"]:
                if _stale(cfg, ds, today):
                    return HostStatus(False, RC_STALE)
                return HostStatus(True, RC_APPLIES, "cisa_dotgov", ds["federal"][name])
        return HostStatus(False, RC_GOV_NON_FEDERAL)
    except Exception:  # noqa: BLE001  fail closed
        return HostStatus(False, RC_ERROR)


def _compiled(cfg: dict, key: str) -> list:
    po = cfg.get("page_override") if isinstance(cfg.get("page_override"), dict) else {}
    ck = ("re", key, json.dumps(po.get(key) or [], sort_keys=True))
    if ck not in _cache:
        _cache[ck] = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in po.get(key) or ()]
    return _cache[ck]


def _image_credit_spans(text: str, cfg: dict) -> list[tuple[int, int]]:
    return [m.span() for r in _compiled(cfg, "image_credit_patterns") for m in r.finditer(text)]


def page_check(text: str | None, licence_href: str | None = "", cfg: dict | None = None) -> tuple[bool, str, bool]:
    """(clean, reason_code, image_credit). Narrowing only: any hit -> not clean. No text -> not clean (unchecked)."""
    cfg = cfg if cfg is not None else load_config()
    if licence_href:
        return False, RC_DECLARED, False
    t = str(text or "")
    if not t.strip():
        return False, RC_NO_TEXT, False
    if any(r.search(t) for r in _compiled(cfg, "declared_licence_patterns")):
        return False, RC_DECLARED, False  # a textual licence statement governs (e.g. an ND licence is never bypassed)
    if any(r.search(t) for r in _compiled(cfg, "copyright_notice_patterns")):
        return False, RC_COPYRIGHT, False
    if any(r.search(t) for r in _compiled(cfg, "contractor_patterns")):
        return False, RC_CONTRACTOR, False
    img = _image_credit_spans(t, cfg)
    for r in _compiled(cfg, "third_party_text_patterns"):
        for m in r.finditer(t):
            # only a hit INSIDE the image-credit phrase itself (the "courtesy of" of "Image courtesy of X") is skipped
            if any(s <= m.start() and m.end() <= e for s, e in img):
                continue
            return False, RC_THIRD_PARTY, bool(img)
    return True, RC_APPLIES, bool(img)


def decide(url: str, text: str | None = None, licence_href: str | None = "", cfg: dict | None = None,
           today: _dt.date | None = None) -> dict:
    """The rule's verdict for one page: {"rule", "mode", "applies", "reason_code", "basis", "organization",
    "image_credit", "dataset_retrieved", "licence"}. "licence" is {"id", "rule", "basis"} only when applies.
    Never raises; carries no page text and no host."""
    cfg = cfg if cfg is not None else load_config()
    out = {"rule": str(cfg.get("rule_id") or RULE_ID), "mode": mode(cfg), "applies": False, "reason_code": RC_OFF,
           "basis": "", "organization": "", "image_credit": False, "dataset_retrieved": "", "licence": None}
    try:
        hs = host_status(url, cfg, today)
        out.update(reason_code=hs.code, basis=hs.basis, organization=hs.organization[:120],
                   dataset_retrieved=_dataset(cfg)["retrieved"][:10] if hs.basis == "cisa_dotgov" else "")
        if not hs.federal:
            return out
        clean, code, img = page_check(text, licence_href, cfg)
        out.update(reason_code=code, image_credit=img)
        if clean:
            out["applies"] = True
            out["licence"] = {"id": str(cfg.get("licence_id") or "LicenseRef-us-government-work"),
                              "rule": out["rule"], "basis": hs.basis}
        return out
    except Exception:  # noqa: BLE001
        out.update(applies=False, reason_code=RC_ERROR, licence=None)
        return out


def candidate_text(c: dict) -> str | None:
    """The candidate's FETCHED full text, only when it is bound to what is stored: c["stored_text_kind"] ==
    "full_text", c["text"] non-empty and c["text_sha256"] equal to its sha256 (opus review finding 6: a snippet or
    summary in a text field must never stand in for the stored page). Else None."""
    if not isinstance(c, dict) or c.get("stored_text_kind") != "full_text":
        return None
    v = c.get("text")
    if not isinstance(v, str) or not v.strip():
        return None
    return v if c.get("text_sha256") == hashlib.sha256(v.encode("utf-8")).hexdigest() else None


def gate_allow(c: dict, cfg: dict | None = None) -> dict | None:
    """For vera.licence_gate: the decision when the rule ALLOWS this candidate's full content in mode 'on', else None.
    Everything is computed HERE: the host from c["url"] and the page test on the candidate's bound full text
    (candidate_text). No precomputed decision field is ever trusted (finding 5). No bound text = None (fail closed)."""
    try:
        cfg = cfg if cfg is not None else load_config()
        if mode(cfg) != "on" or not isinstance(c, dict):
            return None
        txt = candidate_text(c)
        if txt is None:
            return None
        d = decide(str(c.get("url") or ""), txt, str(c.get("licence_href") or ""), cfg)
        return d if d["applies"] else None
    except Exception:  # noqa: BLE001
        return None


def annotate_candidate(c: dict, text: str | None = None, licence_href: str | None = "",
                       cfg: dict | None = None) -> dict | None:
    """Hook for candidate sources (reference portal, discovered web links): in shadow or on mode, record
    c["gov_pd"] = decide(...) (codes only) for trace/shadow counts; off = no change. It GRANTS nothing: the licence gate
    recomputes everything from the bound full text (gate_allow). Returns the decision or None."""
    cfg = cfg if cfg is not None else load_config()
    if mode(cfg) == "off" or not isinstance(c, dict):
        return None
    url = str(c.get("url") or c.get("discovered_url") or "")
    d = decide(url, text if text is not None else candidate_text(c), licence_href, cfg)
    d["url"] = url
    c["gov_pd"] = d
    return d
