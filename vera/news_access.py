"""H19-T1 (OD-1106-NEWS): derived access verdict for no-derivatives (ND) news outlets. Pure, offline, no I/O beyond
the config load; nothing is stored. Design: work/H19H21-DESIGN-1106/design.md section 1 (checks A1 to A10).

mode(url, cfg, page) -> "full_text" | "link_only", with reasons (closed tokens). Fail closed: any missing field,
unknown status string or unreadable input gives link_only. A5 (robots) alone never yields full_text.

Config keys read from the reporting_lane config dict (all optional; shipped absent = fail closed):
  nc_basis: {"status": "SATISFIED"} - the owner-recorded noncommercial-eligibility statement NC-BASIS-1 (X-3).
            Anything but the literal "SATISFIED" is UNCONFIRMED and every ND row resolves to link_only with
            reason nc_basis_unconfirmed. The owner flips this ONE value once NC-BASIS-1 is on record.
  nd_news:  {"families": ["cc-by-nc-nd"]} - licence families this path accepts (default as shown).
  commercial_flags: {"fee"|"advertising"|"sale": bool} - any truthy flag makes every ND row link_only (acc_nc_ineligible).
AD-6: per-operation matrix OPERATIONS; a row's access.operations may only TIGHTEN it (forged looser values are ignored).
Gate order for evaluate(): robots -> licence -> processing -> output; each fail closed, a refusal stops later gates.
A licence version that is 3.0 or unknown is treated as 3.0 (strict): `strict_version` is reported; no row loosens.
Row `access` block shape: see design.md section 1. Page dict (optional, per fetch) keys: licence_href, text,
canonical_url, meta_robots, x_robots_tag, content_signal, tdm_reservation (signal strings from vera.public_fetch).
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from vera import licence_gate, reporting_lane

FULL_TEXT = "full_text"
LINK_ONLY = "link_only"
_DEFAULT_FAMILIES = ("cc-by-nc-nd",)
_RE_VER = r"-\d+(?:\.\d+)?$"


def _family(licence_id: Any) -> str:
    """Licence family: normalised id without the version suffix ('cc-by-nc-nd-4.0' -> 'cc-by-nc-nd')."""
    import re
    return re.sub(_RE_VER, "", licence_gate.norm_id(str(licence_id or "")))


def _s(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".").removeprefix("www.")
    except ValueError:
        return ""


def _families(cfg: dict) -> list[str]:
    nd = cfg.get("nd_news")
    fam = nd.get("families") if isinstance(nd, dict) else None
    out = [_family(f) for f in fam if isinstance(f, str)] if isinstance(fam, list) else []
    return out or list(_DEFAULT_FAMILIES)


def nc_basis_confirmed(cfg: dict | None = None) -> bool:
    """True only for nc_basis.status == "SATISFIED" (literal). Missing or anything else = UNCONFIRMED."""
    nb = (cfg if cfg is not None else reporting_lane.load_config()).get("nc_basis")
    return isinstance(nb, dict) and nb.get("status") == "SATISFIED"


G_ROBOTS, G_LICENCE, G_PROCESSING, G_OUTPUT = "robots", "licence", "processing", "output"
GATE_ORDER = (G_ROBOTS, G_LICENCE, G_PROCESSING, G_OUTPUT)

# AD-6 matrix. Rank: ALLOW < ALLOW_IF_NONCOMMERCIAL < CONDITIONAL < DEFER < NOT_PHASE_1 < BLOCK (stricter = higher).
_RANK = {"ALLOW": 0, "ALLOW_IF_NONCOMMERCIAL": 1, "ALLOW_WITH_OUTPUT_CHECKS": 1, "CONDITIONAL": 2, "DEFER": 3,
         "NOT_PHASE_1": 4, "BLOCK": 5}
OPERATIONS = {
    "metadata": "ALLOW",                      # discovery, provenance metadata, hash (text not retained)
    "rag_context": "ALLOW_IF_NONCOMMERCIAL",  # fetch and use as answer-time context
    "retain_original": "DEFER",               # phase 2; answer-only now
    "persist_embeddings": "NOT_PHASE_1",
    "fact_extraction": "ALLOW_WITH_OUTPUT_CHECKS",  # cross-source analysis; G1 to G5 on output
    "verbatim_quote": "CONDITIONAL",          # G6 budgets + attribution; AD-2 excludes third-party text
    "summary": "BLOCK", "close_paraphrase": "BLOCK", "translation": "BLOCK", "style_transform": "BLOCK",
    "training": "BLOCK", "commercial_use": "BLOCK",
}
_PROCESSING_OK = ("ALLOW", "ALLOW_IF_NONCOMMERCIAL", "ALLOW_WITH_OUTPUT_CHECKS", "CONDITIONAL")


def operation_verdict(row: dict | None, op: str) -> str:
    """Effective matrix verdict for op: the built-in AD-6 value, tightened (never loosened) by row access.operations.
    Unknown op = BLOCK."""
    base = OPERATIONS.get(op)
    if base is None:
        return "BLOCK"
    ov = ((row or {}).get("access") or {}).get("operations") if isinstance((row or {}).get("access"), dict) else None
    v = ov.get(op) if isinstance(ov, dict) else None
    if isinstance(v, str) and v in _RANK and _RANK[v] > _RANK[base]:
        return v
    return base


def licence_version(row: dict | None) -> str:
    """'3.0', '4.0' ... from the row licence id, or '' when absent. Informational."""
    import re
    lic = ((row or {}).get("access") or {}).get("licence") if isinstance(((row or {}).get("access")), dict) else None
    m = re.search(r"(\d+\.\d+)\s*$", licence_gate.norm_id(str((lic or {}).get("id") or ""))) if isinstance(lic, dict) else None
    return m.group(1) if m else ""


def strict_version(row: dict | None) -> bool:
    """3.0 or unknown version is read as 3.0 (strictest); only an explicit version >= 4.0 is not strict."""
    v = licence_version(row)
    try:
        return not (v and float(v) >= 4.0)
    except ValueError:
        return True


def _commercial(cfg: dict) -> bool:
    cf = cfg.get("commercial_flags")
    return isinstance(cf, dict) and any(bool(x) for x in cf.values())


def _g_robots(row: dict, acc: dict, page: dict | None) -> list[str]:
    r: list[str] = []
    rb = acc.get("robots") if isinstance(acc.get("robots"), dict) else {}
    if rb.get("status") != "ALLOWS" or rb.get("ai_agents_blocked") is True:
        r.append("acc_robots")
    if isinstance(page, dict):
        meta = _s(page.get("meta_robots")).lower()
        xrt = _s(page.get("x_robots_tag")).lower()
        if "noai" in meta or "noimageai" in meta:
            r.append("acc_optout_meta")
        if any(t in xrt for t in ("noai", "noimageai", "none", "noindex")):
            r.append("acc_optout_xrt")
        tdm = _s(page.get("tdm_reservation")).lower()
        if tdm == "unreadable":
            r.append("acc_tdm_unreadable")
        elif tdm in ("1", "true", "reserved"):
            r.append("acc_tdm_reserved")
    return r


def _g_licence(row: dict, acc: dict, cfg: dict, page: dict | None) -> list[str]:
    r: list[str] = []
    lic = acc.get("licence") if isinstance(acc.get("licence"), dict) else {}
    fam = _family(lic.get("id"))
    if not fam or not _s(lic.get("evidence_url")):
        r.append("acc_licence_unset")
    elif fam not in _families(cfg):
        r.append("acc_licence_family")
    ai = acc.get("terms_ai_clause") if isinstance(acc.get("terms_ai_clause"), dict) else {}
    if ai.get("status") == "RESTRICTED":
        r.append("acc_ai_restricted")
    elif ai.get("status") != "PERMITTED" or not _s(ai.get("evidence_url")):
        r.append("acc_ai_unresolved")
    if isinstance(page, dict) and fam:
        r += _page_licence_checks(row, fam, page)
    return r


def _page_licence_checks(row: dict, fam: str, page: dict) -> list[str]:
    r: list[str] = []
    href = _s(page.get("licence_href"))  # A7
    if href:
        tail = href.rstrip("/").split("/licenses/", 1)[-1].replace("/", "-") if "/licenses/" in href else href
        if _family("cc-" + licence_gate.norm_id(tail).removeprefix("cc-")) != fam:
            r.append("acc_page_licence_mismatch")
    text = page.get("text") if isinstance(page.get("text"), str) else ""
    low = text.lower()
    rh = _host(_s(row.get("host")))
    canon = _host(_s(page.get("canonical_url")))
    if (any(isinstance(m, str) and m and m.lower() in low for m in row.get("republished_markers") or [])  # A8
            or (canon and rh and canon != rh and not canon.endswith("." + rh))):
        r.append("acc_republished")
    if any(isinstance(m, str) and m and m.lower() in low for m in row.get("exclude_item_markers") or []):  # A10
        r.append("acc_item_excluded")
    return r


def _g_processing(row: dict, acc: dict, cfg: dict, op: str | None) -> list[str]:
    r: list[str] = []
    nc = acc.get("nc_eligible") if isinstance(acc.get("nc_eligible"), dict) else {}
    if nc.get("status") == "NOT_ELIGIBLE" or _commercial(cfg):
        r.append("acc_nc_ineligible")
    elif nc.get("status") != "SATISFIED" or not _s(nc.get("basis")):
        r.append("acc_nc_unresolved")
    if not nc_basis_confirmed(cfg):
        r.append("nc_basis_unconfirmed")
    cd = acc.get("crawl_delay_s")
    if cd is not None and (isinstance(cd, bool) or not isinstance(cd, (int, float)) or cd < 0):
        r.append("acc_delay_bad")
    if op is not None:
        ov = operation_verdict(row, op)
        if ov not in _PROCESSING_OK:
            r.append("acc_op_blocked" if ov == "BLOCK" else "acc_op_not_now")
    return r


def _prelude(url: str, cfg: dict) -> tuple[dict | None, list[str] | None]:
    row = reporting_lane.outlet_row(url, cfg) if isinstance(url, str) else None
    if not row or reporting_lane.outlet_verdict(url, cfg) != "allow":
        return None, ["acc_no_allow_row"]
    if not isinstance(row.get("access"), dict):
        return row, ["acc_no_access_block"]
    return row, None


def evaluate(url: str, op: str = "rag_context", cfg: dict | None = None, page: dict | None = None,
             output_refusal: list[str] | None = None) -> dict:
    """Gate-ordered decision for an operation: robots -> licence -> processing -> output. The first gate with a
    refusal stops evaluation: only its reasons are returned and later gates are not reached (gates_evaluated shows
    which ran). output_refusal = tokens from the caller's nd_guard run (gate 4); None = not run yet."""
    cfg = cfg if cfg is not None else reporting_lane.load_config()
    row, pre = _prelude(url, cfg)
    ev: list[str] = []
    if pre:
        return {"allowed": False, "gate": G_LICENCE, "reasons": pre, "gates_evaluated": [G_LICENCE],
                "has_access": False, "strict_version": True}
    acc, strict = row["access"], strict_version(row)
    checks = (lambda: _g_robots(row, acc, page), lambda: _g_licence(row, acc, cfg, page),
              lambda: _g_processing(row, acc, cfg, op), lambda: list(output_refusal or []))
    for g, fn in zip(GATE_ORDER, checks):
        ev.append(g)
        rs = fn()
        if rs:
            return {"allowed": False, "gate": g, "reasons": rs, "gates_evaluated": ev, "has_access": True,
                    "strict_version": strict}
    return {"allowed": True, "gate": None, "reasons": [], "gates_evaluated": ev, "has_access": True,
            "strict_version": strict}


def verdict(url: str, cfg: dict | None = None, page: dict | None = None) -> dict:
    """Record-level fetch-and-use verdict: {"mode", "reasons", "has_access"}. All gates 1 to 3 are collected
    (not short-circuited) so exceptions list every failing token. has_access False = row without an `access` block
    (today's non-ND behaviour; no exception is due). Empty reasons only when mode is full_text."""
    cfg = cfg if cfg is not None else reporting_lane.load_config()
    row, pre = _prelude(url, cfg)
    if pre:
        return {"mode": LINK_ONLY, "reasons": pre, "has_access": False}
    acc = row["access"]
    reasons = (_g_licence(row, acc, cfg, page) + _g_processing(row, acc, cfg, "rag_context")
               + _g_robots(row, acc, page))
    return {"mode": LINK_ONLY if reasons else FULL_TEXT, "reasons": reasons, "has_access": True}


def mode(url: str, cfg: dict | None = None, page: dict | None = None) -> str:
    return verdict(url, cfg, page)["mode"]


def reasons(url: str, cfg: dict | None = None, page: dict | None = None) -> list[str]:
    return list(verdict(url, cfg, page)["reasons"])


def full_text_ok(url: str, cfg: dict | None = None, page: dict | None = None) -> bool:
    """Design section 5: outlet_verdict == allow AND (row has no ND `access` block OR mode == full_text)."""
    cfg = cfg if cfg is not None else reporting_lane.load_config()
    if reporting_lane.outlet_verdict(url, cfg) != "allow":
        return False
    v = verdict(url, cfg, page)
    return (not v["has_access"]) or v["mode"] == FULL_TEXT
