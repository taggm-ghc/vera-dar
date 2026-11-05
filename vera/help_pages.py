"""UI-ABOUT-1031 / OD-1029-UI: renderers for the About and How to use pages. Static prose is read from
config/ui_help_pages.json; disclosure wording comes from vera.disclosure (never copied into the JSON)."""
import json
import logging
from pathlib import Path

from vera import disclosure
from vera import sources_sidebar

logger = logging.getLogger(__name__)
HELP_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "ui_help_pages.json"
_UNSET = object()
CORPUS_HEADING = "What VERA-DAR draws on"
FALLBACK = "This page is not available right now. The notice at the top of the page still applies."


def load(path: Path = HELP_CONFIG_PATH) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.error("help page config unreadable: %s", Path(path).name)
        return None
    return data if isinstance(data, dict) else None


def wake_minutes() -> int | None:
    """Start-up allowance in whole minutes, read from config/ui_wake.json (max_total_s); None if unreadable."""
    try:
        from vera import ui_wake
        return max(1, round(ui_wake.load_config().max_total_s / 60))
    except Exception:  # noqa: BLE001 - help text must never break a page
        return None


def answer_sources() -> str:
    """Where answers come from, from the same two flags the disclosure uses: full-document answering and answering
    from documents found in the same request (disclosure._OFFTOPIC_ANSWERING). Never claims more than the flags enable."""
    if not disclosure.fulldoc_enabled():
        return "Written answers are currently switched off; VERA-DAR shows reading lists only."
    if disclosure.offtopic_sentence() == disclosure._OFFTOPIC_ANSWERING:
        return ("Those documents are either in VERA-DAR's collection or, when the collection cannot answer, open-access "
                "documents found in the same request; the About page explains both. Such answers may be wrong.")
    return "Those documents are in VERA-DAR's collection."


LIVE_STATS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "live_stats.json"
LIVE_STATS_LINE = "VERA-DAR keeps one anonymous record per answer, with the time rounded to the hour; no question text or identifiers."


def live_stats_enabled(path: Path = LIVE_STATS_CONFIG_PATH) -> bool:
    """config/live_stats.json key `enabled` is exactly true; absent, unreadable or anything else is False."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("enabled") is True
    except (OSError, ValueError, AttributeError):
        return False


def _fill(text: str, coverage: str | None = None) -> str:
    # WP6 (orchestrator decision): the third-party point appears once on non-About pages, in the short notice; the
    # Privacy section keeps the no-save point and points to About for the services that receive the question.
    # CORPUS-DIGEST-1104: {coverage} is the generated coverage description (already escaped); no fixed Scope text.
    privacy = disclosure._NO_SAVE.strip()
    return (text.replace("{privacy}", privacy)
            .replace("{answer_sources}", answer_sources())
            .replace("{wake_minutes}", str(wake_minutes()))
            .replace("{coverage}", coverage or ""))


def _section(st, sec: dict, coverage: str | None = None) -> None:
    st.subheader(str(sec.get("heading", "")))
    for p in sec.get("paragraphs") or []:
        if "{wake_minutes}" in str(p) and wake_minutes() is None:
            continue  # no hardcoded fallback figure: omit the sentence if the config cannot be read
        if "{coverage}" in str(p) and not coverage:
            continue  # owner 2026-11-04: no fixed Scope text exists; no generated coverage yet = omit the line
        st.markdown(_fill(str(p), coverage))


def render_about(st, cfg=_UNSET) -> None:
    cfg = load() if cfg is _UNSET else cfg
    page = (cfg or {}).get("about")
    st.title((page or {}).get("title", "About VERA-DAR"))
    if not isinstance(page, dict):
        st.info(FALLBACK)
        return
    st.markdown(str(page.get("intro", "")))
    for heading, text in disclosure.disclosure_sections():
        st.subheader(heading)
        st.markdown(text)
    _corpus = sources_sidebar._corpus(st)
    st.subheader(CORPUS_HEADING)  # HOTFIX-BLURB-1104: the corpus description is About-only (was the sidebar on every page)
    sources_sidebar.render_corpus(st, _corpus)
    sources_sidebar.render_corpus_notes(st, _corpus)  # CORPUS-DIGEST-1104: moved from the sidebar
    for sec in page.get("after_disclosure") or []:
        _section(st, sec)
    if live_stats_enabled():
        st.markdown(LIVE_STATS_LINE)


def render_how_to_use(st, cfg=_UNSET) -> None:
    cfg = load() if cfg is _UNSET else cfg
    page = (cfg or {}).get("how_to_use")
    st.title((page or {}).get("title", "How to use VERA-DAR"))
    if not isinstance(page, dict):
        st.info(FALLBACK)
        return
    st.markdown(str(page.get("intro", "")))
    for sec in page.get("sections") or []:  # HOTFIX-BLURB-1104: no coverage/retention text here (About-only)
        _section(st, sec)
