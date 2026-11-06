"""LANE-LINKS-1106 (WP5): "Current reporting (links only)" group from contract.inline_acquisition.lanes.

Main answer area only (the sidebar never lists individual sources). Need-to-know: title, outlet, date, link, a
read/link-only mark and the GDELT credit; never a portal name, registry state or internal cause. Every string is
escaped and passed through safe_markdown; no unsafe_allow_html.
"""
from __future__ import annotations

import re

from vera.ui_safety import safe_markdown

_MD = re.compile(r"([\\`*_{}\[\]()#+!|~<>$])")


def _t(v, n: int = 300) -> str:
    s = " ".join(str(v).split())[:n] if isinstance(v, (str, int, float)) else ""
    return _MD.sub(r"\\\1", s)


def _http(u) -> str:
    u = str(u or "").strip()
    return u if len(u) <= 2000 and u.lower().startswith(("http://", "https://")) and not re.search(r"[\s()<>\[\]]", u) else ""


def lane_links_of(data) -> tuple[list, list]:
    c = data.get("contract") if isinstance(data, dict) else None
    ia = c.get("inline_acquisition") if isinstance(c, dict) else None
    lanes = ia.get("lanes") if isinstance(ia, dict) else None
    if not isinstance(lanes, dict):
        return [], []
    links = [x for x in (lanes.get("links") or []) if isinstance(x, dict) and _http(x.get("url"))]
    credits = [x for x in (lanes.get("credits") or []) if isinstance(x, dict)]
    return links, credits


def render_lane_links(st, data) -> None:
    links, credits = lane_links_of(data)
    if not links:
        return
    st.markdown("#### Current reporting (links only)")
    for lk in links:
        meta = ", ".join(p for p in (_t(lk.get("outlet") or lk.get("entity")), _t(lk.get("published"), 40)) if p)
        mark = "read" if lk.get("read") is True else "link only"
        title = _t(lk.get("title")) or _t(_http(lk["url"]))
        st.markdown(safe_markdown(f"- [{title}]({_http(lk['url'])})" + (f" ({meta})" if meta else "") + f" - {mark}"))
    for cr in credits:
        name, url = _t(cr.get("name") or cr.get("text") or cr.get("title")), _http(cr.get("url"))
        if name and url:
            st.caption(safe_markdown(f"[{name}]({url})"))
        elif name:
            st.caption(name)
