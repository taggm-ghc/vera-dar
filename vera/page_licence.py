"""H19-FIX-1109 (OD-1106-NEWS design A7, A8, A10, AD-2/A11; OD-1109-T8C keeps ND news off): page-level licence checks
for every fetched page that can feed an answer (reporting lane pages and web-route pages, including the six-1109
original-host follows). Pure and offline: no I/O beyond the config load; nothing is stored, logged or traced here.

check(url, page) -> {"ok", "reasons", "text", "a11_dropped", "licence"}:
  A7  the page's own rel=license link: a no-derivatives, no-educational-use or fee-only licence -> page_licence_reject;
      for an outlet row with an ND `access` block, a link whose CC family differs from the row -> acc_page_licence_mismatch
      (vera.news_access._page_licence_checks, which also runs that row's A8/A10 markers).
  A8  republication or syndication: a configured republication cue in the text, the row's republished_markers, or a
      <link rel=canonical> on another site -> acc_republished.
  A10 item exclusion: the row's exclude_item_markers -> acc_item_excluded; an "All Rights Reserved" notice on a page
      whose open licence is claimed (row licence or the page's CC link) -> acc_item_excluded (the item's licence
      cannot be established). Without an open-licence claim the notice only sets the shown licence.
  A11 third-party material (AD-2), detected mechanically, never from topic lists: a page-level wire cue (wire dateline,
      wire byline, wire copyright line, "contributed to this report") -> acc_third_party_page; paragraphs that hold
      third-party markup (public_fetch `third_party_text`: blockquote, q, cite, figure, figcaption, media) are dropped,
      and a markup segment that cannot be located drops the page (acc_third_party_ambiguous, fail closed); sentences
      matching a wire-credit pattern are dropped. Nothing left -> acc_third_party_all.
Any reason means the page is a link only (fail closed). Patterns and markers: config/reporting_lane.json
`page_checks` (DEFAULTS below are the fallback when the block is missing or malformed).
`licence` is the licence to show with the page's citations: {"id", "label", "url"} or None.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

DEFAULTS = {
    "republished_markers": ["originally published by", "originally published in", "originally published on",
                            "republished from", "republished with permission", "this story was originally published",
                            "this article was originally published", "this article first appeared",
                            "this story first appeared", "reprinted with permission", "syndicated from"],
    "arr_markers": ["all rights reserved"],
    "page_wire_patterns": [r"^[^\n]{0,60}\((?:ap|reuters|afp|upi|dpa|efe|ansa)\)\s*[—–-]",
                           r"(?m)^\s*by\b[^\n]{0,80}\b(?:associated press|reuters|agence france[- ]presse|afp)\b",
                           r"\bcopyright\s+(?:©\s*)?\d{4}\s+(?:the\s+)?(?:associated press|reuters|"
                           r"agence france[- ]presse|afp)\b",
                           r"\bcontributed to this report\b"],
    "wire_credit_patterns": [r"\((?:ap|reuters|afp|upi|dpa|efe|ansa|xinhua)\)", r"\bassociated press\b",
                             r"\breuters\b", r"\bagence france[- ]presse\b", r"\bafp\b", r"\bunited press international\b",
                             r"\bap photo\b", r"\bgetty images\b", r"\bused with permission\b", r"\bcourtesy of\b"],
}
_SENT = re.compile(r"(?<=[.!?])\s+")
_CC_PATH = re.compile(r"/(licenses|publicdomain)/([a-z-]+)/(\d+(?:\.\d+)?)", re.I)


def _cfg(cfg: dict | None) -> dict:
    try:
        if cfg is None:
            from vera import reporting_lane
            cfg = reporting_lane.load_config()
        pc = cfg.get("page_checks") if isinstance(cfg, dict) else None
    except Exception:  # noqa: BLE001
        pc = None
    out = dict(DEFAULTS)
    if isinstance(pc, dict):
        for k in DEFAULTS:
            v = pc.get(k)
            if isinstance(v, list) and all(isinstance(x, str) and x for x in v) and v:
                out[k] = v
    return out


def _host(url: str) -> str:
    try:
        return (urlsplit(str(url or "")).hostname or "").lower().rstrip(".").removeprefix("www.")
    except ValueError:
        return ""


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _compile(pats) -> list:
    out = []
    for p in pats:
        try:
            out.append(re.compile(p, re.I | re.M))
        except re.error:
            out.append(None)  # a broken pattern is reported by the caller as a config fault (fail closed)
    return out


def licence_from_href(href: str) -> dict | None:
    """{"id", "label", "url"} for a creativecommons.org licence or public-domain link; a link elsewhere is shown as
    "Licence terms" with its URL. '' -> None."""
    href = str(href or "").strip()
    if not href:
        return None
    m = _CC_PATH.search(href) if "creativecommons.org" in _host(href) else None
    if m:
        kind, fam, ver = m.group(1).lower(), m.group(2).lower(), m.group(3)
        if kind == "publicdomain":
            lab = "CC0 " + ver if fam == "zero" else "Public Domain Mark " + ver
            return {"id": ("cc0-" if fam == "zero" else "pdm-") + ver, "label": lab, "url": href}
        return {"id": f"cc-{fam}-{ver}", "label": f"CC {fam.upper()} {ver}", "url": href}
    return {"id": None, "label": "Licence terms", "url": href}


def label_for_id(licence_id: str) -> str:
    """'CC-BY-NC-ND-4.0' -> 'CC BY-NC-ND 4.0'; anything else unchanged."""
    m = re.fullmatch(r"cc-([a-z-]+?)-(\d+(?:\.\d+)?)", str(licence_id or "").strip().lower())
    return f"CC {m.group(1).upper()} {m.group(2)}" if m else str(licence_id or "")


def _row_licence(row: dict | None) -> dict | None:
    acc = (row or {}).get("access")
    lic = acc.get("licence") if isinstance(acc, dict) else None
    if isinstance(lic, dict) and lic.get("id"):
        return {"id": str(lic["id"]), "label": label_for_id(lic["id"]), "url": lic.get("url") or None}
    return None


def filter_third_party(text: str, segments, pc: dict) -> tuple[str, int, list[str]]:
    """(filtered text, passages dropped, reasons). Paragraph (line) level for markup, sentence level for credits."""
    reasons: list[str] = []
    lines = [ln for ln in str(text or "").split("\n")]
    dropped = 0
    segs = [s for s in (segments or []) if isinstance(s, str)] if isinstance(segments, list) else []
    if segments is not None and not isinstance(segments, list):
        reasons.append("acc_third_party_ambiguous")
    drop_idx: set = set()
    for seg in segs:
        for frag in (" ".join(x.split()) for x in seg.split("\n")):
            if not frag:
                continue
            hit = [i for i, ln in enumerate(lines) if frag in " ".join(ln.split())]
            if not hit:
                reasons.append("acc_third_party_ambiguous")
                break
            drop_idx.update(hit)
    dropped += len(drop_idx)
    lines = [ln for i, ln in enumerate(lines) if i not in drop_idx]
    credit = _compile(pc["wire_credit_patterns"])
    if any(c is None for c in credit):
        reasons.append("acc_third_party_config")
        credit = [c for c in credit if c is not None]
    out = []
    for ln in lines:
        kept = []
        for s in _SENT.split(ln):
            if any(c.search(s) for c in credit):
                dropped += 1
            elif s.strip():
                kept.append(s)
        if kept:
            out.append(" ".join(kept))
    return "\n".join(out), dropped, sorted(set(reasons))


def check(url: str, page: dict, *, row: dict | None = None, cfg: dict | None = None) -> dict:
    """See the module docstring. Never raises: an internal error is a link only (reason page_check_error)."""
    try:
        from vera import licence_gate, news_access, reporting_lane
        rcfg = cfg if cfg is not None else reporting_lane.load_config()
        pc = _cfg(rcfg)
        if row is None:
            row = reporting_lane.outlet_row(url, rcfg)
        page = page if isinstance(page, dict) else {}
        text = str(page.get("text") or "")
        low = _norm(text)
        href = str(page.get("licence_href") or "").strip()
        reasons: list[str] = []
        acc = (row or {}).get("access")
        row_lic = _row_licence(row)
        page_lic = licence_from_href(href)
        # A7 (+ the access row's own A8/A10 markers)
        if isinstance(acc, dict):
            fam = news_access._family(((acc.get("licence") or {}) if isinstance(acc.get("licence"), dict) else {})
                                      .get("id"))
            if fam:
                reasons += news_access._page_licence_checks(row, fam, {"licence_href": href, "text": text,
                                                                        "canonical_url": page.get("canonical_url")})
            else:
                reasons.append("acc_licence_unset")
        elif href and licence_gate.reject_reason(None, href):
            reasons.append("page_licence_reject")
        # A8
        markers = list(pc["republished_markers"]) + [m for m in (row or {}).get("republished_markers") or []
                                                     if isinstance(m, str) and m]
        canon, here = _host(page.get("canonical_url") or ""), _host(page.get("url") or url)
        if (any(_norm(m) in low for m in markers)
                or (canon and here and canon != here and not canon.endswith("." + here)
                    and not here.endswith("." + canon))):
            reasons.append("acc_republished")
        # A10
        arr = any(_norm(m) in low for m in pc["arr_markers"])
        open_claim = bool(row_lic or (page_lic and page_lic.get("id")) or (row or {}).get("licence_note"))
        if any(isinstance(m, str) and m and _norm(m) in low for m in (row or {}).get("exclude_item_markers") or []) \
                or (arr and open_claim):
            reasons.append("acc_item_excluded")
        # A11
        wire = _compile(pc["page_wire_patterns"])
        if any(w is None for w in wire):
            reasons.append("acc_third_party_config")
        if any(w is not None and w.search(text) for w in wire):
            reasons.append("acc_third_party_page")
        ftext, n_drop, r11 = filter_third_party(text, page.get("third_party_text"), pc)
        reasons += r11
        if not ftext.strip():
            reasons.append("acc_third_party_all")
        lic = row_lic or page_lic
        if lic is None and isinstance((row or {}).get("licence_note"), str) and row["licence_note"].strip():
            lic = {"id": None, "label": row["licence_note"].strip(), "url": None}
        if lic is None and arr:
            lic = {"id": "all-rights-reserved", "label": "All rights reserved (short quotation)", "url": None}
        seen: list[str] = []
        for r in reasons:
            if r not in seen:
                seen.append(r)
        return {"ok": not seen, "reasons": seen, "text": ftext if not seen else "", "a11_dropped": n_drop,
                "licence": lic}
    except Exception:  # noqa: BLE001
        return {"ok": False, "reasons": ["page_check_error"], "text": "", "a11_dropped": 0, "licence": None}
