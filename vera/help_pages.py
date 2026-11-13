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
CORPUS_HEADING = "What VERA-DAR draws on"  # CORPUS-DESC-1107: no longer an About heading (kept so tests can assert its absence)
CORPUS_PAGE = "pages/corpus_description.py"
CORPUS_PAGE_TITLE = "Corpus Description"
CORPUS_LINK_DEFAULT = "Corpus Description: what VERA-DAR's collection covers"
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
    if sec.get("corpus_link"):  # CORPUS-DESC-1107: the JSON gives the label only; the page path lives here
        _page_link(st, CORPUS_PAGE, str(sec["corpus_link"]))


# PRIVACY-1104 (owner 2026-11-04: "toabnu content should be migrated away from About & How"). About skips the
# disclosure_sections() headings listed in config about.moved_to_terms; the Terms page appends them verbatim. In list mode
# (fulldoc off) the reliance, licence and no-profile sentences sit inside the single _BASE section and stay on About until
# vera/disclosure.py splits _BASE (follow-up in work/PRIVACY-1104/migration-map.md). Links live here, never in the JSON.
THIRD_PARTY_SERVICES = (("arXiv", "https://arxiv.org/"), ("OpenAlex", "https://openalex.org/"),
                        ("Europe PMC", "https://europepmc.org/"), ("Semantic Scholar", "https://www.semanticscholar.org/"),
                        ("Wikipedia", "https://en.wikipedia.org/"), ("DuckDuckGo", "https://duckduckgo.com/"))
# Owner 2026-11-05: no email contact; terms and privacy requests go through the public repository's issue tracker.
REPO_ISSUES_LINK = "the project's public GitHub repository ([taggm-ghc/vera-dar issues](https://github.com/taggm-ghc/vera-dar/issues))"
TERMS_PAGE = "pages/terms.py"
TERMS_LABEL = "Terms of Access, Browsing, and Use"
TERMS_PENDING = "This part of the terms is being finalised."
NEWS_HEADING = "News reports"


def render_about(st, cfg=_UNSET) -> None:
    cfg = load() if cfg is _UNSET else cfg
    page = (cfg or {}).get("about")
    st.title((page or {}).get("title", "About VERA-DAR"))
    if not isinstance(page, dict):
        st.info(FALLBACK)
        return
    if page.get("opening"):  # owner 2026-11-05: the About page opens with the VERA expansion (config about.opening)
        st.markdown(str(page["opening"]))
    for line in page.get("opening_expansions") or []:  # ABOUT-DARMAG-1106: -DAR and -MAG expansions after the VERA line
        st.markdown(str(line))
    st.markdown(str(page.get("intro", "")))
    moved = set(page.get("moved_to_terms") or [])  # PRIVACY-1104: terms-type disclosure sections render on Terms
    for heading, text in disclosure.disclosure_sections():
        if heading in moved:
            continue
        st.subheader(heading)
        st.markdown(text)
    # CORPUS-DESC-1107 (owner 2026-11-07: too granular for About): the corpus description is its own page; About keeps one
    # link line and the licence / how-it-answers / Trace Eval notes (no coverage line: that is on the Corpus Description page).
    _page_link(st, CORPUS_PAGE, str(page.get("corpus_link") or CORPUS_LINK_DEFAULT))
    sources_sidebar.render_corpus_notes(st, None)
    for sec in page.get("after_disclosure") or []:
        _section(st, sec)
        if sec.get("heading") == NEWS_HEADING and disclosure.reporting_lane_on():
            # NEWS-LANE-1104 / R-031 fix note: the GDELT credit stays on About (and next to GDELT items), even though
            # "Your question and what is stored" (which also carries it) moved to Terms.
            st.markdown(disclosure._LANE_GDELT_CREDIT.strip())


def _page_link(st, path: str, label: str) -> None:
    try:
        st.page_link(path, label=label)
    except Exception:  # noqa: BLE001  (page not found outside a multipage run)
        st.markdown(label + ".")


def render_corpus_page(st, cfg=_UNSET) -> None:
    """CORPUS-DESC-1107: the Corpus Description page. One count/newest line (render_corpus), the subject areas (domains,
    topics, subtopics; code-built in every state), the checked digest when one exists, and the current coverage."""
    cfg = load() if cfg is _UNSET else cfg
    page = (cfg or {}).get("corpus_page") or {}
    st.title(str(page.get("title") or CORPUS_PAGE_TITLE))
    for para in page.get("paragraphs") or []:
        st.markdown(str(para))
    data = sources_sidebar._corpus(st)
    sources_sidebar.render_corpus(st, data)
    line = sources_sidebar.unlabelled_line(data.get("unlabelled") if isinstance(data, dict) else None)  # CORPUS-UNCAT-1113
    if line:
        st.caption(line)
    sources_sidebar.render_stored_taxonomy(st, data)  # WP6: no output unless the API sent stored_taxonomy (flag off: never)
    cov = sources_sidebar.coverage_text(data)
    if cov:
        st.markdown(f"**Current coverage:** {cov}")


ASK_TERMS_DEFAULT = "By asking, you agree to the Terms"


def render_ask_terms_notice(st, cfg=_UNSET) -> None:
    """H14 (OD-1106-HITL): one short line under the Ask box, linked to the Terms page. Wording from config
    ui_help_pages.json `ask_terms_notice.label`. It is a notice of terms, not a record of consent."""
    cfg = load() if cfg is _UNSET else cfg
    label = str(((cfg or {}).get("ask_terms_notice") or {}).get("label") or ASK_TERMS_DEFAULT)
    try:
        st.page_link(TERMS_PAGE, label=label)
    except Exception:  # noqa: BLE001  (page not found outside a multipage run)
        st.caption(label + ".")


def _terms_para(st, text: str) -> None:
    # Placeholders ([[OWNER-n]], [[VERIFY-n]], [[GAP-..]]) never reach users: the paragraph is replaced, and a release
    # test must assert that the shipped config has none.
    st.markdown(TERMS_PENDING if "[[" in text else text.replace("{repo_issues}", REPO_ISSUES_LINK))


def render_terms(st, cfg=_UNSET) -> None:
    """PRIVACY-1104: Terms of Access, Browsing, and Use, including the privacy notice. Static prose from
    config/ui_help_pages.json `terms`; flag-driven disclosure sections are appended verbatim by heading."""
    cfg = load() if cfg is _UNSET else cfg
    page = (cfg or {}).get("terms")
    st.title((page or {}).get("title", TERMS_LABEL))
    if not isinstance(page, dict):
        st.info(FALLBACK)
        return
    eff = str(page.get("effective_date", ""))
    if eff and "[[" not in eff:
        st.caption(f"Effective date: {eff}")
    st.markdown(str(page.get("intro", "")))
    dsec = dict(disclosure.disclosure_sections())
    for sec in page.get("sections") or []:
        st.subheader(str(sec.get("heading", "")))
        for p in sec.get("paragraphs") or []:
            _terms_para(st, str(p))
        if sec.get("services"):
            st.markdown("\n".join(f"- [{name}]({url})" for name, url in THIRD_PARTY_SERVICES))
        for h in sec.get("disclosure") or []:
            if h in dsec:
                st.markdown(f"**{h}.** {dsec[h]}")
        if sec.get("gdelt_credit") and disclosure.reporting_lane_on():
            st.markdown(disclosure._LANE_GDELT_CREDIT.strip())
        if sec.get("live_stats") and live_stats_enabled():
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
