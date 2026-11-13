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


# K3.I: UI labels for the page links (labels, not data). Order is the order shown.
NAV = (("streamlit_app.py", "Ask VERA-DAR"), ("pages/vera_demo.py", "Report"), ("pages/trace_eval.py", "Trace Eval"),
       ("pages/how_to_use.py", "How to use"), ("pages/about.py", "About"),
       ("pages/corpus_description.py", "Corpus Description"),  # CORPUS-DESC-1107 (owner direction): its own page
       ("pages/terms.py", "Terms of Access, Browsing, and Use"))  # PRIVACY-1104: a link on every page, not content

# K3.I: the only fields render_corpus reads from the API body (deny by default; `summary_model` is never read).
CORPUS_FIELDS = frozenset({"summary_source", "source_count", "corpus_updated_at", "summary", "topics", "summary_generated_at",
                           "coverage", "topic_shares", "subject_areas", "published_years", "frequent_subjects",
                           "stored_taxonomy", "unlabelled"})
# D7 (DIGEST-IMPROVE-1105): the digest sample includes the full documents stored as passages, first; the label says how
# many of the sampled sources they are. SAMPLE_EXCLUSION is kept only for digests built before that change (their
# topic_shares carry no full_documents_in_sample key, and their sample excluded those documents).
SAMPLE_EXCLUSION = "full documents stored as passages are not in the sample"


def _sample_basis(shares: dict, n: int) -> str:
    """What the share percentages count: the sampled sources, and how many of them are full documents."""
    if "full_documents_in_sample" not in shares:
        return f"share of the {n} sampled sources, each counted once under its main topic; {SAMPLE_EXCLUSION}"
    k = int(shares["full_documents_in_sample"])
    fd = f", including {k} full document{'s' if k != 1 else ''} stored as passages" if k > 0 else ""
    return f"share of the {n} sampled sources{fd}, each counted once under its main topic"

LICENCE_SIDEBAR = ("For listed records, only the indexing services' metadata licence was checked; the articles' own "
                   "licences were not verified with the publisher.")
LICENCE_UNDER_ANSWER = LICENCE_SIDEBAR
MODEL_LINE = "A third-party hosted language model may be used to check whether a request is safe to handle."
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


# Deny-All audit G1: model-written text (summary, topics, topic labels in the shares line, coverage) is shown ONLY when
# the digest carries this marker, which vera.corpus_description sets when every model-written field passed the
# claim/evidence check (cited cluster, checker approval with verified title ids, numbers only as approved facts or
# code-confirmed derivations). Anything else shows code-built lines only.
CHECKED = "agentic_checked"
CODE_ONLY_SOURCES = ("deterministic_fallback",)  # summary text built by code, not a model


def _model_text_ok(line) -> bool:
    return bool(line) and not _BACKEND_WORDS.search(str(line))


def coverage_text(data: dict | None) -> str | None:
    """CORPUS-DIGEST-1104: the generated description of current coverage, or None. Never the configured scope string:
    with no generated description the caller omits the line (owner: no fixed Scope text)."""
    if not data or data.get("summary_source") != CHECKED:
        return None
    cov = data.get("coverage")
    return plain(cov) if isinstance(cov, str) and cov.strip() and _model_text_ok(cov) else None


def _share_lines(shares) -> list[str]:
    """Main topics (smallest set reaching 80% of sampled documents, computed in code) and the remainder lines."""
    if not isinstance(shares, dict) or not isinstance(shares.get("main"), list) or not shares["main"]:
        return []
    try:
        n = int(shares["basis_documents"])
        main = ", ".join(f"{plain(t['label'])} {int(t['pct'])}%" for t in shares["main"] if _model_text_ok(t.get("label")))
        if not main or n <= 0:
            return []
        reach = f"{int(shares.get('main_pct') or 0)}%"
        lines = [f"**Main topics** ({reach} of documents; {_sample_basis(shares, n)}): {main}."]
        o = shares.get("other") or {}
        if int(o.get("topics") or 0) > 0:
            k = int(o["topics"])
            lines.append(f"Other topics: {int(o.get('pct') or 0)}% across {k} smaller topic{'s' if k != 1 else ''}.")
        u = shares.get("ungrouped") or {}
        if int(u.get("documents") or 0) > 0:
            lines.append(f"Not grouped under a topic: {int(u.get('pct') or 0)}%.")
        return lines
    except (KeyError, TypeError, ValueError):
        return []


def _pct(share) -> str:
    v = float(share)
    return "under 1%" if 0 < v < 0.005 else f"{int(round(v * 100))}%"


def _subject_lines(sa) -> list[str]:
    """CORPUS-DESC-1107: code-built domain > topic > subtopic lines (names from the DB topic registry, shares computed by
    code; no model text, no source title). [] when absent or malformed."""
    if not isinstance(sa, dict):
        return []
    try:
        n = int(sa["basis_documents"])
        if n <= 0:
            return []
        lines = [f"**Subject areas** (provisional; share of {n} documents; each document counts once, under the nearest subject "
                 "in VERA-DAR's subject list. The subject list is incomplete and still growing; new subject areas are added "
                 "after owner review):"]  # OD-1108-HITL item 8: never present the listed areas as the complete set
        for d in sa.get("domains") or []:
            label = plain(str(d["label"])[:1].upper() + str(d["label"])[1:])
            parts = []
            for t in d.get("topics") or []:
                subs = ", ".join(f"{plain(x['label'])} {_pct(x['share'])}" for x in t.get("subtopics") or [])
                parts.append(f"{plain(t['label'])} {_pct(t['share'])}" + (f" (including {subs})" if subs else ""))
            more = int(d.get("more_topics") or 0)
            if more:
                parts.append(f"{more} more topic{'s' if more != 1 else ''}")
            lines.append(f"{label}: {_pct(d['share'])}" + (f". Topics: {'; '.join(parts)}." if parts else "."))
        o = sa.get("other_domains") or {}
        if int(o.get("count") or 0) > 0:
            k = int(o["count"])
            lines.append(f"Other subject areas: {_pct(o.get('share') or 0)} across {k} more area{'s' if k != 1 else ''}.")
        if not sa.get("domains"):
            lines.append("No document matched a listed subject area yet.")
        else:  # TAXO-MATCH-FIX-1108: exact count, not only a rounded share
            u = sa.get("unmatched_count")
            u = int(u) if isinstance(u, int) and not isinstance(u, bool) else int(round(float(sa.get("unmatched_share") or 0) * n))
            if 0 < u <= n:
                lines.append(f"Not yet matched to a listed subject area: {u} of {n} documents ({_pct(u / n)}).")
        return lines if all(_model_text_ok(ln) for ln in lines) else []  # defence in depth: all or nothing
    except (KeyError, TypeError, ValueError):
        return []


def subjects_enabled(path: Path = CHAIN_PATH) -> bool:
    """CORPUS-INTERIM-1108: the switch corpus_description.subject_areas.enabled in config/ask-provider-chain.json (the API
    reads the same value). Only an explicit true shows per-subject counts, shares and the subject list; a missing, malformed
    or false value hides them (deny by default). Set true again only once a validated matcher ships."""
    sa = (((_read(path) or {}).get("corpus_description") or {}).get("subject_areas")) or {}
    return sa.get("enabled") is True


def _years_text(years) -> str:
    if isinstance(years, list) and len(years) == 2 and all(isinstance(y, str) and re.fullmatch(r"\d{4}", y) for y in years):
        return f"publication years {years[0]} to {years[1]}" if years[0] != years[1] else f"publication year {years[0]}"
    return ""


def _frequent_line(phrases) -> str | None:
    """CORPUS-INTERIM-1108: 'Frequent subjects include: a; b; c.' from code-built title phrases; no numbers, no sources."""
    if not isinstance(phrases, list):
        return None
    ok = [plain(p) for p in phrases if isinstance(p, str) and p.strip() and not re.search(r"\d", p) and _model_text_ok(p)]
    return f"Frequent subjects include: {'; '.join(ok)}." if ok else None


NO_SUBJECTS = "A description by subject area is not available right now."


def _interim_summary(text: str) -> str:
    """OD-1109-HD3: in interim mode the checked written description keeps no percentage, share, fraction or per-subject
    count (inferred subject distributions); see corpus_interim.strip_shares. Returns "" when nothing is left."""
    from vera import corpus_interim
    return corpus_interim.strip_shares(text)


def _n(v) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0


def taxonomy_lines(tx) -> list[str]:
    """WP6: caption blocks for the stored per-chunk taxonomy (vera.corpus_taxonomy payload); [] when absent or malformed.
    Names are taxonomy entities (domain, topic, subtopic), never a source. Both views of a document's subject areas are
    shown and labelled: 'main label' is the one stored document-level vote; 'contains passages on' is the union of the
    document's stored chunk labels (a document may appear under several subjects). Unlabelled counts are derived from
    missing rows (abstentions are not stored), so they are one combined figure."""
    try:
        if not isinstance(tx, dict) or not isinstance(tx.get("domains"), list):
            return []
        t = tx["totals"]
        docs, dch, lab_c = _n(t["documents"]), _n(t["documents_with_chunks"]), _n(t["chunks_labelled"])
        lines = [
            "**Stored subject labels** (automatic and approximate; made by a program that groups passages by meaning and names groups from frequent terms, not read or checked by a person. "
            "The subject list is incomplete, so some material is outside it). Two views of a document are shown. "
            "**Main label (vote)**: the one subject area stored for the whole document. **Contains passages on (union)**: "
            "every subject area at least one of the document's stored passages was labelled with, so a document can appear "
            "under several subjects and the figures do not add up to the document count."]
        for d in tx["domains"]:
            head = (f"**{plain(d['label'])}**: main label for {_n(d['vote_docs'])} documents; contains passages on it in "
                    f"{_n(d['union_docs'])} documents ({_n(d['chunks']):,} labelled passages)")
            rows = [head]
            for tp in d.get("topics") or []:
                rows.append(f"- {plain(tp['label'])}: main label for {_n(tp['vote_docs'])} documents; passages in "
                            f"{_n(tp['union_docs'])} documents ({_n(tp['chunks']):,} passages)")
                for sb_ in tp.get("subtopics") or []:
                    if _n(sb_["chunks"]):
                        rows.append(f"    - {plain(sb_['label'])}: passages in {_n(sb_['union_docs'])} documents ({_n(sb_['chunks']):,} passages)")
            lines.append("\n".join(rows))
        lines.append(
            f"**Not labelled or outside the subject list** (not told apart: unlabelled passages are not recorded by reason): "
            f"{_n(t['chunks_not_labelled']):,} of {_n(t['chunks']):,} stored passages have no subject label. "
            f"{_n(t['documents_vote_not_labelled'])} of {docs} documents have no main label; "
            f"{_n(t['documents_union_not_labelled'])} of the {dch} documents with stored passages have no labelled passage.")
        return lines if lab_c or any(d.get("vote_docs") for d in tx["domains"]) else []
    except (KeyError, TypeError, ValueError):
        return []


def render_stored_taxonomy(st, data: dict | None) -> None:
    """WP6: the stored per-chunk subject labels, appended by the Corpus page after render_corpus. Nothing is written unless the
    API sent a stored_taxonomy block (flag corpus_description.stored_taxonomy.enabled; default off sends none)."""
    for line in taxonomy_lines(data.get("stored_taxonomy") if isinstance(data, dict) else None):
        st.caption(line)


def _unl_pct(u: int, n: int) -> str:
    """CORPUS-UNCAT-1113: share u of n as text; never rounds a non-zero share to 0% or a partial share to 100%."""
    if u == 0:
        return "0%"
    if u == n:
        return "100%"
    p = (200 * u + n) // (2 * n)  # rounded half up, integer arithmetic
    return "under 1%" if p < 1 else "over 99%" if p >= 100 else f"{p}%"


def unlabelled_line(data) -> str | None:
    """CORPUS-UNCAT-1113: code-built caption stating how many documents and stored passages have no stored subject label, from the
    API's `unlabelled` block. No model text, no source names. None for a missing or malformed payload."""
    try:
        v = {k: data[k] for k in ("documents", "documents_unlabelled", "documents_without_passages", "passages", "passages_unlabelled")}
        if any(type(x) is not int or x < 0 for x in v.values()):
            return None
        n, u, x, nc, uc = v["documents"], v["documents_unlabelled"], v["documents_without_passages"], v["passages"], v["passages_unlabelled"]
        if n == 0 or u > n or x > n or uc > nc:
            return None
    except (KeyError, TypeError):
        return None
    xc = f", including {x:,} with no stored passages, which the labelling program cannot label" if x > 0 else ""
    pc = f"; {uc:,} of {nc:,} stored passages ({_unl_pct(uc, nc)}) have none" if nc > 0 else ""
    return (f"**Subject labels:** {u:,} of {n:,} documents ({_unl_pct(u, n)}) have no stored subject label "
            f"(no main label and no labelled passage){xc}{pc}. Documents not yet processed by the labelling program and those outside "
            "its subject list are not told apart. Labels are assigned automatically and are approximate; "
            "a document stored in several versions is counted once.")


def render_corpus(st, data: dict | None) -> None:
    """CORPUS-DIGEST-1104: count/newest line, the digest, subject topics with shares, and the model-written line.
    CORPUS-DESC-1107: plus the code-built subject areas (domains, topics, subtopics) in every state; rendered on the
    Corpus Description page. The licence note, how-it-answers and Trace Eval lines stay on About (render_corpus_notes)."""
    if not data or data.get("summary_source") in (None, "disabled", "unavailable", "not_connected"):
        st.caption(f"**VERA-DAR's corpus:** {NO_DESCRIPTION}")
        return
    data = {k: v for k, v in data.items() if k in CORPUS_FIELDS}
    interim = not subjects_enabled()  # CORPUS-INTERIM-1108: no per-subject numbers, assignments or subject list
    if interim:
        data = {**data, "subject_areas": None, "topic_shares": None, "topics": []}
    when = f", newest source {plain(data['corpus_updated_at'][:10])}" if data.get("corpus_updated_at") else ""
    yrs = f", {_years_text(data.get('published_years'))}" if interim and _years_text(data.get("published_years")) else ""
    st.caption(f"**VERA-DAR's corpus:** {int(data.get('source_count') or 0)} sources{when}{yrs}.")
    checked = data.get("summary_source") == CHECKED
    if data.get("summary") and data.get("summary_source") in CODE_ONLY_SOURCES and not interim:
        st.caption(plain(data["summary"]))  # code-built text
    subjects = _subject_lines(data.get("subject_areas"))
    if not checked:  # default deny: no model-written field without the checker-pass marker
        if interim:
            st.caption(_frequent_line(data.get("frequent_subjects")) or NO_SUBJECTS)
            return
        for line in subjects or [NO_SUBJECTS]:
            st.caption(line)
        return
    summary = _interim_summary(data["summary"]) if interim and data.get("summary") else data.get("summary")
    if interim and not summary:
        st.caption(_frequent_line(data.get("frequent_subjects")) or NO_SUBJECTS)
    if summary:
        st.caption(plain(summary) if _model_text_ok(summary) else f"**VERA-DAR's corpus:** {NO_DESCRIPTION}")
    for line in subjects:
        st.caption(line)
    shares = _share_lines(data.get("topic_shares"))
    if shares:
        for line in shares:
            st.caption(line)
    elif data.get("topics"):
        line = "Topics: " + ", ".join(data["topics"])
        st.caption(plain(line) if _model_text_ok(line) else f"**VERA-DAR's corpus:** {NO_DESCRIPTION}")
    if checked:
        st.caption(f"Description written by a language model on {plain((data.get('summary_generated_at') or '')[:10])}; "
                   "refreshed when the corpus changes.")


_HOW_LIST = ("**How VERA-DAR answers:** for each question the Ask page searches published scholarly sources "
             "and the web live and lists the ones that may be relevant, with links. VERA-DAR has not read the full "
             "documents, so it does not use abstracts or snippets as evidence and does not write an answer from "
             "them. If no usable source is found it says so instead of answering from memory. Open the sources "
             "and read them yourself before relying on any claim.")
_HOW_FULLDOC = ("**How VERA-DAR answers:** for each question the Ask page first searches a small collection of "
                "full documents that the operator selected and licence-checked in advance. If passages in them address "
                "the question, it writes a short answer from up to six passages, cites them with brief quotes, and has "
                "an AI model check each cited sentence; unsupported sentences are removed. Otherwise it searches "
                "published scholarly sources and the web and lists documents it has not read. Open the sources and "
                "read them yourself before relying on any claim.")


def how_answers_base() -> str:
    """The How-VERA-DAR-answers paragraph without the off-topic sentence: on the About page that sentence already has
    its own disclosure section, so it is not repeated (each disclosure text appears exactly once there)."""
    from vera.disclosure import fulldoc_enabled
    return _HOW_FULLDOC if fulldoc_enabled() else _HOW_LIST


def how_answers_text() -> str:
    """Chosen by the runtime fulldoc flag (S1), never static."""
    from vera.disclosure import fulldoc_enabled
    from vera.disclosure import offtopic_sentence
    return (_HOW_FULLDOC if fulldoc_enabled() else _HOW_LIST) + offtopic_sentence()


def trace_eval_line() -> str:
    snap = snapshot_summary()
    if snap and snap["count"] is None:  # aggregate-only run summary: no per-source count is published
        when = f", captured {snap['captured']}" if snap["captured"] else ""
        return f"**Trace Eval sources:** historical snapshot: per-source counts are not published; the latest posted run covered {snap['questions']} test questions{when}."
    if not snap or not snap["count"]:
        return "**Trace Eval sources:** not available"
    span = f", first posted {snap['years'][0]} to {snap['years'][1]}" if snap["years"] else ""
    when = f", captured {snap['captured']} (UTC; replaced when the evaluation is re-run)" if snap["captured"] else ""
    return (f"**Trace Eval sources:** historical snapshot: {snap['count']} distinct sources "
            f"retrieved for {snap['questions']} test questions{span}{when}.")


CORPUS_NOTES_HEADING = "VERA-DAR's corpus: licences and how it answers"  # CORPUS-DESC-1107: coverage is on Corpus Description


def corpus_notes_lines(data: dict | None) -> list[str]:
    """CORPUS-DIGEST-1104: the lines moved from the sidebar to the About page, wording kept. The fixed Scope line is
    replaced by the generated coverage description, and omitted when none exists (owner amendment)."""
    cov = coverage_text(data)
    return ([f"**Current coverage:** {cov}"] if cov else []) + [
        LICENCE_SIDEBAR, how_answers_base() + " " + MODEL_LINE, trace_eval_line()]


def render_corpus_notes(st, data: dict | None = None) -> None:
    st.subheader(CORPUS_NOTES_HEADING)
    for line in corpus_notes_lines(data):
        st.markdown(line)


def render_sources_sidebar(st) -> None:
    """Kept as a no-op for the page shells. Owner request: the sidebar shows one About entry only, the NAV link
    from render_nav ("About"); the second "How VERA-DAR answers and its limits → About" link was removed. The header
    links were removed from every page too (vera.disclosure.render_disclosure). The corpus description stays About-only
    (vera.help_pages.render_about, HOTFIX-BLURB-1104)."""
    return None


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
                   fulldoc: bool | None = None, cited=None, claim_check_layer: str | None = None) -> None:
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
            st.markdown("**Sources** (documents whose cited passages the answer was grounded in)"  # NOANSWER-RC-1107 r3
                        if ui_evidence.count_matching_on(ui_evidence._cfg())
                        else "**Sources** (documents whose inspected passages the answer was grounded in)")
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
    if claim_check_layer != "deferred":  # ANS-LAYERS-1101: layered UI renders the citation check in its own expander
        ui_evidence.render_claim_check(st, claim_check, cap, fulldoc)
    if traced:
        st.caption(("Also found as a published record, but its full text was not acquired for this answer, so not used: "
                    if fulldoc else
                    "Also found and matched to a published record, but no abstract was available for it, so not used: ")
                   + "; ".join(_link(t.get("title") or t.get("url") or "source", t.get("url")) for t in traced))
