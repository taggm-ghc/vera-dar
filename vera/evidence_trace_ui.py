"""NONASK-1103 E3: shared renderer for the evidence trace (Ask and Report). Main pane only, never st.sidebar.

render_trace(st, trace, public=None, redact=None) -> bool
  Returns False and renders nothing when the trace is empty, the label file/flag is off, or the trace was built for a
  different audience (a dev trace is never rendered in public mode). Every dynamic string goes through
  ui_safety.safe_markdown, after `redact` (applied to the raw string first, when given). No unsafe_allow_html.
"""
from __future__ import annotations

from vera import evidence_trace, ui_safety
from vera.sources_sidebar import _BACKEND_WORDS, plain, plain_url


def _make(redact):
    def s(x) -> str:
        raw = "" if x is None else str(x)
        if redact is not None:
            raw = str(redact(raw))  # raw string first: escaping would split hosts like a\\.b\\.com
        t = ui_safety.safe_markdown(raw)
        return plain_url(t) if t.strip().lower().startswith(("http://", "https://")) else plain(t)
    return s


def render_trace(st, trace, public=None, redact=None) -> bool:
    if not isinstance(trace, dict) or not trace or evidence_trace.load_labels() is None:
        return False
    if public is None:
        public = evidence_trace._is_public_mode()
    if public and trace.get("view") != "public":
        return False  # fail closed: the dev view never renders in public mode
    s = _make(redact)
    passages = [p for p in trace.get("passages") or () if isinstance(p, dict)]
    parts = [p for p in trace.get("parts") or () if isinstance(p, dict)]
    ad = trace.get("adequacy") if isinstance(trace.get("adequacy"), dict) else None
    shown = False
    if passages:
        st.markdown("**How VERA-DAR chose the evidence**")
        for p in passages:
            n = s(p.get("n"))
            if p.get("cited"):
                sup = [x for x in p.get("supports") or () if not _BACKEND_WORDS.search(str(x))]
                line = f"[{n}] " + (s(p.get("verdict")) if p.get("verdict") else "used")
                if sup:
                    line += ": " + " ".join(s(x) for x in sup)
                st.markdown(line)
                if p.get("excerpt"):
                    st.caption("Passage: " + s(p["excerpt"]))
            else:
                st.caption(f"[{n}] selected, not cited: " + s(p.get("reason") or ""))
        shown = True
    unc = [p for p in parts if not p.get("covered")]
    if unc or (ad and ad.get("verdict") in ("qualify", "abstain")):
        st.markdown("**Parts of your question the evidence did not cover**")
        for p in unc:
            st.markdown("- " + s(p.get("label") or "part"))
        shown = True
    claims = trace.get("claims") if isinstance(trace.get("claims"), dict) else None
    if claims and (claims.get("partial") or claims.get("unsupported")):
        st.markdown("**Citation check**")
        st.caption(f"supported {s(claims.get('supported', 0))}, partly supported {s(claims.get('partial', 0))}, "
                   f"removed {s(claims.get('unsupported', 0))}")
        shown = True
    nu = trace.get("not_used") if isinstance(trace.get("not_used"), dict) else None
    if nu:
        for text, k in nu.items():
            st.caption(f"{s(k)} not used: " + s(text))
        shown = True
    if not public:
        for c in trace.get("codes") or ():
            st.caption("code: " + s(c))
        shown = True
    return shown
