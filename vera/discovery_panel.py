"""K2.I5: the Ask page's non-answer branch (reading list, decline, discovery notes). Fail-closed: anything that is not
an evidentiary answer lands here. This module never calls st.success, render_sources or render_memory_panel and never
reads sources, traced_sources, claim_check, recalled or memory_submitted (the answer branch owns those)."""
from vera import answer_contract
from vera.memory_panel import _as_count
from vera.sources_sidebar import LICENCE_SIDEBAR, _link, plain
from vera.ui_safety import safe_markdown

KIND_ANSWER, KIND_DECLINE, KIND_READING_LIST = "answer", "decline", "reading_list"
KIND_SELF_DESCRIPTION = "self_description"  # SELFREF-1028: fixed public self-description, shown as plain text


def _candidates(raw) -> list[dict]:
    """Public candidate fields only; non-lists and malformed items dropped; unknown acquisition -> not acquired."""
    if not isinstance(raw, list):
        return []
    out = []
    for c in raw:
        if not isinstance(c, dict) or not isinstance(c.get("title"), str) or not isinstance(c.get("url"), str):
            continue
        if not (c["title"].strip() or c["url"].strip()):
            continue
        state = c.get("acquisition")
        state = state if state in (answer_contract.ACQ_NOT_ACQUIRED, answer_contract.ACQ_ACQUIRED, answer_contract.ACQ_IN_CORPUS) \
            else answer_contract.ACQ_NOT_ACQUIRED
        out.append({"title": c["title"].strip(), "url": c["url"].strip(), "acquisition": state,
                    "published": c.get("published") if isinstance(c.get("published"), (str, int)) else None,
                    "identifier": c.get("identifier") if isinstance(c.get("identifier"), (str, int)) else None,
                    "has_licence": bool(c.get("licence_decision"))})
    return out


def _render_memory(st, data: dict) -> None:
    if _as_count(data.get("discovery_note_submitted")) > 0:
        st.info(answer_contract.text("discovery_note_submitted"))
    notes = data.get("recalled_discovery")
    if isinstance(notes, list) and notes:
        st.markdown("**Earlier reading-list notes**")
        st.caption(answer_contract.text("recalled_note_caption"))
        for item in notes:
            if not isinstance(item, dict) or not isinstance(item.get("note"), str):
                continue
            st.markdown(safe_markdown(plain(item["note"])))
            urls = [u for u in (item.get("urls") or []) if isinstance(u, str)] if isinstance(item.get("urls"), list) else []
            links = [_link(u, u) for u in urls]
            if links:
                st.markdown(" · ".join(links))


def render_discovery(st, data: dict) -> None:
    kind = data.get("result_kind")
    if kind in (KIND_DECLINE, KIND_SELF_DESCRIPTION):  # SELFREF-1028: text only, no reading-list heading or warning
        st.info(safe_markdown(data.get("answer") or ""))
        return
    cands = _candidates(data.get("candidates"))
    st.markdown("### Reading list")
    c_ = data.get("contract")
    if cands and isinstance(c_, dict) and c_.get("offtopic") is True and isinstance(data.get("answer"), str):
        st.info(safe_markdown(data["answer"]))  # FIX1-1021: outside-scope notice shown above the list
    if cands:
        st.warning(answer_contract.not_read_line([{"acquisition": c["acquisition"]} for c in cands]))
        for i, c in enumerate(cands, 1):
            line = f"{i}. {_link(c['title'] or c['url'], c['url'])}"
            if c["published"] not in (None, ""):
                line += f" ({plain(c['published'])})"
            st.markdown(line)
            parts = []
            if c["identifier"] not in (None, ""):
                parts.append(plain(c["identifier"]))
            parts.append(answer_contract.representation(c["acquisition"]))
            st.caption(" · ".join(parts))
        if any(c["has_licence"] for c in cands):
            st.caption(LICENCE_SIDEBAR)
    elif kind == KIND_READING_LIST:
        st.info(safe_markdown(data.get("answer") or ""))
    else:
        st.warning(answer_contract.text("not_read_line"))
    _render_memory(st, data)
