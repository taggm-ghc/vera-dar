"""Item #78: sidebar expander telling visitors what VERA-DAR's answers draw on.

Everything shown is read from files shipped with the app (scope router config, /ask provider chain config,
the Trace Eval results snapshot); no database, API, environment variable or host is read or shown.

Owner requirement: the sidebar NEVER discloses a specific individual source, document or quote.
Enforced structurally: snapshot_summary() returns aggregate fields only (SNAPSHOT_FIELDS), so no title, URL or id
can reach the sidebar renderer; render_sources_sidebar() takes no sampling/rng argument.
"""
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("vera")

ROOT = Path(__file__).resolve().parent.parent
SCOPE_PATH = ROOT / "config" / "scope_router.json"
CHAIN_PATH = ROOT / "config" / "ask-provider-chain.json"
SELECTION_PATH = ROOT / "config" / "model-selection.json"
SNAPSHOT_PATH = ROOT / "eval_results" / "trace_eval_v1.json"
SNAPSHOT_FIELDS = frozenset({"count", "years", "questions", "captured"})  # aggregate only; never per-source
_ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{2})(\d{2})\.\d{4,5}")
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~])")


def _read(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def plain(text: str) -> str:
    """Escape Markdown so a source title renders as literal text."""
    return _MD_SPECIAL.sub(r"\\\1", _CTRL.sub(" ", str(text)).strip())  # control chars (newlines) never survive (SEC U2)


_URLISH = re.compile(r"https?://\S+", re.I)  # use fullmatch: `$` would also accept a trailing newline (SEC U2)
_CTRL = re.compile(r"[\x00-\x1f\x7f]")
_CITE_N = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_numbers(answer):
    """Source numbers cited by the displayed answer text ([n] and [n, m] markers), not by citation rows (SEC U1).
    None (unknown: every passage counts as cited) when the answer is not text."""
    if not isinstance(answer, str):
        return None
    return {int(x) for m in _CITE_N.finditer(answer) for x in re.findall(r"\d+", m.group(1))}
_MD_URL_SPECIAL = re.compile(r"([\\`*\[\]<>])")


def plain_url(text: str) -> str:
    """Like plain(), but a URL-looking text keeps its dots, underscores and hyphens unescaped (escaped URLs showed
    backslashes and broke copy/paste); only characters that could open Markdown/HTML constructs are escaped."""
    t = str(text)
    return _MD_URL_SPECIAL.sub(r"\\\1", t) if _URLISH.fullmatch(t) and not _CTRL.search(t) else plain(t)


def scope_text(path: Path = SCOPE_PATH) -> str | None:
    cfg = _read(path)
    return str(cfg["scope"]) if cfg and cfg.get("scope") else None


# K3.I: UI labels for the page links (labels, not data). Order is the order shown.
NAV = (("streamlit_app.py", "Ask VERA-DAR"), ("pages/vera_demo.py", "Report"), ("pages/trace_eval.py", "Trace Eval"))

# K3.I: the only fields render_corpus reads from the API body (deny by default; `summary_model` is never read).
CORPUS_FIELDS = frozenset({"summary_source", "source_count", "corpus_updated_at", "summary", "topics", "summary_generated_at"})

LICENCE_SIDEBAR = ("For listed records, only the indexing services' metadata licence was checked; the articles' own "
                   "licences were not verified with the publisher.")
LICENCE_UNDER_ANSWER = LICENCE_SIDEBAR
MODEL_LINE = "A third-party hosted language model may be used to decide whether a question is in scope."
NO_DESCRIPTION = "description not available right now."
# Public text is constant or derived from allowlisted fields. Model-written text matching these is dropped (defence in depth).
_BACKEND_WORDS = re.compile(r"\b(?:groq|openai|anthropic|claude|llama|mistral|gemini|gpt|qwen|openrouter|gate [abc]|"
                            r"licence gate|write gate)\b", re.I)


def render_nav(st) -> None:
    """Page links with plain labels (the default list is hidden by .streamlit/config.toml). Called by the page shells,
    never from render_sources_sidebar: st.page_link raises StreamlitPageNotFoundError when the page is not found
    relative to the running main script (for example under AppTest of a pages/ file), which is caught and logged."""
    from streamlit.errors import StreamlitPageNotFoundError
    for path, label in NAV:
        try:
            st.sidebar.page_link(path, label=label)
        except StreamlitPageNotFoundError:
            logger.warning("navigation link %r unavailable in this run", label)


def grounding_cap(path: Path = CHAIN_PATH) -> int | None:
    """The per-source text cap the Ask page really uses (config), or None when unreadable (no fallback number)."""
    try:
        cap = int(((_read(path) or {}).get("grounding") or {})["snippet_max_chars"])
    except (KeyError, TypeError, ValueError):
        return None
    return cap if cap > 0 else None


def answer_providers(chain_path: Path = CHAIN_PATH, selection_path: Path = SELECTION_PATH) -> list[str]:
    """Provider:model names in the order /ask tries them (names only, no keys or endpoints)."""
    chain = _read(chain_path) or {}
    names = [f"{p['provider']} {p['model']}" for p in chain.get("providers", []) if p.get("provider") and p.get("model")]
    sel = _read(selection_path) or {}
    if sel.get("selected_model"):
        names.append(f"openai {sel['selected_model']}")
    return names


def _public_snapshot() -> dict | None:
    """P0-2: the aggregate file the Trace Eval page displays (published run, else the shipped historical file)."""
    for name in ("trace_eval_public.json", "trace_eval_historical_public.json"):
        d = _read(ROOT / "eval_results" / name)
        if d and d.get("schema") == "vera-trace-public-v1" and d.get("variants"):
            q = d.get("n_questions")
            if isinstance(q, int) and not isinstance(q, bool) and q > 0:
                g = str(d.get("generated_at") or "")[:10]
                return {"count": None, "years": None, "questions": q,
                        "captured": g if re.match(r"^\d{4}-\d\d-\d\d$", g) else None}
    return None


def snapshot_summary(path: Path | None = None) -> dict | None:
    """Aggregate facts about the Trace Eval snapshot: distinct-source count, arXiv id year span, question count, date.

    Deliberately returns no per-source field (title, URL, id); see SNAPSHOT_FIELDS."""
    if path is None:
        pub = _public_snapshot()
        if pub:
            return pub
        path = SNAPSHOT_PATH
    res = _read(path)
    if not res:
        return None
    sources = set()
    for t in res.get("traces", []):
        for s in t.get("sources") or []:
            if s.get("url"):
                sources.add(s["url"])
    years = sorted({2000 + int(m.group(1)) for u in sources if (m := _ARXIV_ID.search(u))})
    questions = len({t.get("id") for t in res.get("traces", []) if t.get("id")})
    return {"count": len(sources), "years": (years[0], years[-1]) if years else None,
            "questions": questions, "captured": str(res.get("generated_at") or "")[:10] or None}


CORPUS_REFRESH_S = 600  # per browser session: how often the sidebar re-asks the API (the API caches anyway)
CORPUS_RETRY_S = 20  # a FAILED fetch is remembered this long, so reruns do not re-block on a slow or waking API


def _corpus(st) -> dict | None:
    """Item #83: the API's corpus description, fetched server-side and kept for CORPUS_REFRESH_S per session."""
    import os
    import time

    from vera.public_mode import get_corpus_summary, upstream_base_url
    cached = st.session_state.get("_vera_corpus")
    if cached and time.monotonic() - cached[0] < (CORPUS_REFRESH_S if cached[1] is not None else CORPUS_RETRY_S):
        return cached[1]
    data = get_corpus_summary(upstream_base_url(), os.getenv("VERA_API_KEY", ""))
    st.session_state["_vera_corpus"] = (time.monotonic(), data)
    return data


def render_corpus(st, data: dict | None) -> None:
    if not data or data.get("summary_source") in (None, "disabled", "unavailable", "not_connected"):
        st.caption(f"**VERA-DAR's corpus:** {NO_DESCRIPTION}")
        return
    data = {k: v for k, v in data.items() if k in CORPUS_FIELDS}
    when = f", newest source {plain(data['corpus_updated_at'][:10])}" if data.get("corpus_updated_at") else ""
    st.caption(f"**VERA-DAR's corpus:** {int(data.get('source_count') or 0)} sources{when}. {LICENCE_SIDEBAR}")
    for line in (data.get("summary"), ("Topics: " + ", ".join(data["topics"])) if data.get("topics") else None):
        if line:
            st.caption(f"**VERA-DAR's corpus:** {NO_DESCRIPTION}" if _BACKEND_WORDS.search(str(line)) else plain(line))
    if data.get("summary_source") == "agentic_model":
        st.caption(f"Description written by a language model on {plain((data.get('summary_generated_at') or '')[:10])}; "
                   "refreshed when the corpus changes.")


_HOW_LIST = ("**How VERA-DAR answers:** for each in-scope question the Ask page searches published scholarly sources "
             "and the web live and lists the ones that may be relevant, with links. VERA-DAR has not read the full "
             "documents, so it does not use abstracts or snippets as evidence and does not write an answer from "
             "them. If no usable source is found it says so instead of answering from memory. Open the sources "
             "and read them yourself before relying on any claim.")
_HOW_FULLDOC = ("**How VERA-DAR answers:** for each in-scope question the Ask page first searches a small collection of "
                "full documents that the operator selected and licence-checked in advance. If passages in them address "
                "the question, it writes a short answer from up to six passages, cites them with brief quotes, and has "
                "an AI model check each cited sentence; unsupported sentences are removed. Otherwise it searches "
                "published scholarly sources and the web and lists documents it has not read. Open the sources and "
                "read them yourself before relying on any claim.")


def how_answers_text() -> str:
    """Chosen by the runtime fulldoc flag (S1), never static."""
    from vera.disclosure import fulldoc_enabled
    from vera.disclosure import offtopic_sentence
    return (_HOW_FULLDOC if fulldoc_enabled() else _HOW_LIST) + offtopic_sentence()


def render_sources_sidebar(st) -> None:
    with st.sidebar.expander("📚 What VERA-DAR draws on", expanded=False):
        render_corpus(st, _corpus(st))
        scope = scope_text()
        st.caption(f"**Scope:** {plain(scope)}" if scope else "**Scope:** not available")
        st.caption(how_answers_text() + " " + MODEL_LINE)
        snap = snapshot_summary()
        if snap and snap["count"] is None:  # aggregate-only run summary: no per-source count is published
            when = f", captured {snap['captured']}" if snap["captured"] else ""
            st.caption(f"**Trace Eval sources:** per-source counts are not published; the latest posted run covered {snap['questions']} test questions{when}.")
            return
        if not snap or not snap["count"]:
            st.caption("**Trace Eval sources:** not available")
            return
        span = f", first posted {snap['years'][0]} to {snap['years'][1]}" if snap["years"] else ""
        when = f", captured {snap['captured']} (UTC; replaced when the evaluation is re-run)" if snap["captured"] else ""
        st.caption(f"**Trace Eval sources:** {snap['count']} distinct sources "
                   f"retrieved for {snap['questions']} test questions{span}{when}.")


def _link(title: str, url: str) -> str:
    safe_url = str(url or "")
    if (not safe_url.lower().startswith(("https://", "http://")) or any(c in safe_url for c in " ()<>\"'")
            or _CTRL.search(safe_url)):
        t = str(title or "")
        if _URLISH.fullmatch(t) and not _CTRL.search(t) and not any(c in t for c in " ()<>\"'"):  # the title is itself a clean URL: make it a link
            return f"[{plain_url(t)}]({t})"
        return plain_url(title)
    return f"[{plain_url(title)}]({safe_url})"


def _row_n(s: dict, i: int):
    n = s.get("n")
    return n if isinstance(n, int) and not isinstance(n, bool) else i


def _doc_key(s: dict, i: int):
    for k in ("canonical_source_id", "document_id", "url"):
        if s.get(k):
            return (k, str(s[k]))
    return ("row", i)


def group_sources(sources: list[dict], cited=None) -> list[dict]:
    """One entry per document: first row supplies title/year/url; `cited` / `other` hold (n, span) of its passages.
    `cited` None = unknown, so every passage counts as cited. Documents with a cited passage come first."""
    groups: dict = {}
    for i, s in enumerate(sources, 1):
        g = groups.setdefault(_doc_key(s, i), {"row": s, "cited": [], "other": []})
        n = _row_n(s, i)
        g["cited" if cited is None or n in cited else "other"].append((n, s.get("span")))
    return sorted(groups.values(), key=lambda g: not g["cited"])  # stable: cited documents first


def _ranges(items) -> str:
    return ", ".join(plain(str(sp)) for _, sp in items if sp)


def render_sources(st, sources: list[dict], traced: list[dict], claim_check: dict | None = None,
                   fulldoc: bool | None = None, cited=None) -> None:
    """Sources under an Ask answer (main answer area, not the sidebar): numbered (used) and traced originals without a
    free abstract (not used). Abstract wording only for abstract-grounded rows; full-document rows say inspected passages."""
    from vera import ui_evidence
    sources = [s for s in (sources if isinstance(sources, list) else []) if isinstance(s, dict)]
    traced = [t for t in (traced if isinstance(traced, list) else []) if isinstance(t, dict)]
    if fulldoc is None:
        fulldoc = ui_evidence.is_fulldoc(None, sources)
    cap = None if fulldoc else grounding_cap()
    if sources:
        if fulldoc:
            st.markdown("**Sources** (documents whose inspected passages the answer was grounded in)")
        else:
            st.markdown("**Sources** (published records whose abstracts"
                        f"{f', up to {cap:,} characters each,' if cap else ''} the answer was grounded in)")
        announced = False
        for g in (group_sources(sources, cited) if fulldoc else [
                {"row": s, "cited": [(_row_n(s, i), s.get("span"))], "other": [], "single": True} for i, s in enumerate(sources, 1)]):
            s = g["row"]
            if fulldoc and not g["cited"] and not announced:
                announced = True
                st.markdown("**Also selected, not cited** (documents whose passages were selected but not used in a cited claim)")
            nums = ", ".join(str(n) for n, _ in (g["cited"] or g["other"]))
            date = f" ({plain(s['published'])})" if s.get("published") else ""
            st.markdown(f"[{nums}] {_link(s.get('title') or s.get('url') or 'source', s.get('url'))}{date}")
            # K3.I (RF2): provider, retrieved_at and licence_decision are not shown publicly (owner-gated view later).
            if g.get("single"):
                meta = " · ".join(plain_url(str(s[k])) for k in ("identifier", "span") if s.get(k))
            else:
                meta = " · ".join(x for x in (plain_url(str(s["identifier"])) if s.get("identifier") else "",
                                              f"passages used: {_ranges(g['cited'])}" if g["cited"] and _ranges(g["cited"]) else "",
                                              f"also selected, not cited: {_ranges(g['other'])}" if g["other"] and _ranges(g["other"]) else "") if x)
            if meta:
                st.caption(meta)
        if any(s.get("licence_decision") for s in sources):
            st.caption(LICENCE_UNDER_ANSWER)
    ui_evidence.render_claim_check(st, claim_check, cap, fulldoc)
    if traced:
        st.caption(("Also found as a published record, but its full text was not acquired for this answer, so not used: "
                    if fulldoc else
                    "Also found and matched to a published record, but no abstract was available for it, so not used: ")
                   + "; ".join(_link(t.get("title") or t.get("url") or "source", t.get("url")) for t in traced))
