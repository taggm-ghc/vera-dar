"""Evidence findings recalled from memory (K1/K2). Discovery notes are rendered by vera/discovery_panel.py, never here."""


def _recalled_caption(confirm_required=None) -> str:
    """Visitor-facing caption: names no back-end component; claims owner confirmation only when it is required."""
    if confirm_required is None:
        from vera.disclosure import confirm_required as _flag
        confirm_required = _flag()
    base = ("Findings from earlier sessions that passed VERA-DAR's automated checks against the text they cite, which may be "
            "an abstract or index record rather than the full document")
    if confirm_required:
        return (base + ", and were confirmed by VERA-DAR's owner (confirmed is not proof of truth). "
                "Never your question or who asked.")
    return base + ". They were not individually confirmed by VERA-DAR's owner. Never your question or who asked."


def _submitted_line(n: int, confirm_required=None, noun: str = "finding") -> str:
    """Visitor-facing submission signal: states a submission only, never storage or recall."""
    if confirm_required is None:
        from vera.disclosure import confirm_required as _flag
        confirm_required = _flag()
    plural = noun + "s"
    outcome = "wait for owner review" if confirm_required else "may be recalled in later sessions"
    return f"{plural.capitalize()} are submitted, and those that pass {outcome}."  # n is deliberately not shown (owner ruling BARE)


def _as_count(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def render_memory_panel(st, data: dict) -> None:
    """Display recalled findings from memory. Handles missing keys gracefully.

    Args:
        st: Streamlit module
        data: Response dict containing optional 'recalled' and 'memory_submitted' keys.
              recalled: list of {"claim_text","source_title","source_url","saved_at"}
              memory_submitted: int (candidates handed to the writer; not stored rows)
              memory_written is a deprecated constant 0 and is ignored.
    """
    from vera.sources_sidebar import _link, plain
    from vera.ui_safety import safe_markdown

    recalled = data.get("recalled") or []
    submitted = _as_count(data.get("memory_submitted", 0))

    if not recalled and submitted <= 0:
        return

    if recalled:
        st.markdown("### 🧠 From VERA-DAR's memory")
        st.caption(_recalled_caption())
        for item in recalled:
            claim_text = item.get("claim_text", "")
            source_title = item.get("source_title", "")
            source_url = item.get("source_url", "")
            saved_at = item.get("saved_at", "")

            # Format the display
            if claim_text:
                st.markdown(safe_markdown(claim_text))
            if source_title or source_url:
                st.caption(_link(source_title or source_url, source_url))
            if saved_at:
                st.caption(f"From an earlier session ({plain(saved_at)})")

    if submitted > 0:
        st.info(_submitted_line(submitted))
