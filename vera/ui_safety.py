"""Output handling for model/web text shown in Streamlit (OWASP LLM05, item #72).

Model text is never rendered with unsafe_allow_html. Before it goes to st.markdown it is passed
through `safe_markdown`, which neutralises raw HTML and the markdown constructs that make the
browser fetch or navigate on its own (images, non-http(s) links). Plain text goes through st.text.
"""
from __future__ import annotations

import re

_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(\s*([^)\s]*)[^)]*\)")
_REF_DEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*\S+.*$", re.MULTILINE)


def safe_markdown(text) -> str:
    s = "" if text is None else str(text)
    s = _IMAGE.sub(lambda m: f"[image removed: {m.group(1)}]", s)
    s = _REF_DEF.sub("", s)

    def link(m):
        label, url = m.group(1), m.group(2)
        if url.lower().startswith(("http://", "https://")):
            return m.group(0)
        return label

    s = _LINK.sub(link, s)
    return s.replace("<", "&lt;").replace(">", "&gt;")
