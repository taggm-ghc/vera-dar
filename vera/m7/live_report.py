"""REPORT-1022: a functional VERA-DAR report from a real /ask run (the Example report page's live mode).

Flow: the page sends the visitor's question through the SAME client the Ask page uses (vera.public_mode.post_ask, the
single place the X-API-Key header is attached), times the round trip, projects the JSON body into an allowlisted
report dict (`build_live_report`) and draws it (`render_live_report`). The page is a pure function of that projection.

Exposure rules, enforced here and covered by tests/test_m7_live_report.py:
  * the API base URL and the API key never reach a widget, the projection or any rendered string: `resolve_api`
    returns them to the page only, and every rendered string also passes through the existing `judge_select.scrub`
    helper with the process env plus the resolved URL and its host, so an echoed URL, host or key-shaped string is
    replaced by *** (defence in depth);
  * deny by default: only the fields named below are read from the body; the model name, provider names, internal
    fallback tokens, gate or stage names, retrieved_at, text hashes and full passage text are never shown;
    search providers and reason codes appear only as aggregate counts under fixed human labels (unknown -> "other");
  * excerpts are trimmed to the existing quote cap (ui_evidence._quote_cap, config/fulldoc_answer.json quote_max_chars);
  * links are http(s) only (sources_sidebar._link); model and web text goes through ui_safety.safe_markdown or
    sources_sidebar.plain, never unsafe HTML;
  * error messages are fixed text (no exception text, no URL); the status code is shown in dev mode only.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

from vera import ui_evidence
from vera.sources_sidebar import _link, plain
from vera.ui_safety import safe_markdown

RECENT_KEY = "vera_report_recent"  # st.session_state key: this browser session's projected reports only
RECENT_MAX = 5  # named constant: how many recent reports a session keeps (UI convenience, not a decision threshold)
CANDIDATES_SHOWN_MAX = 10  # named constant: reading-list rows shown on a non-answer report
REF_MAX = 80  # named constant: max chars of a passage reference ("chars=<start>-<end>") shown

LIVE_BANNER = ("LIVE REPORT. Produced by a real VERA-DAR run for the question below. Automated checks, not human "
               "review, compared each claim with the passages shown, and can be wrong. Read the sources before "
               "relying on a claim.")
FIXTURE_FALLBACK_NOTE = ("OFFLINE FALLBACK. No live report is shown here. The sections below are an illustrative "
                         "example (or a stored evaluation, when one exists), not a run for your question.")

# Fixed human labels (deny by default). Keys are API values; labels never name a search service, model or stage.
SERVICE_KIND = {"arxiv": "scholarly index", "openalex": "scholarly index", "europepmc": "scholarly index",
                "doaj": "scholarly index", "crossref": "scholarly index", "semantic_scholar": "scholarly index",
                "ddg": "web search", "duckduckgo": "web search", "duckduckgo_lite": "web search", "web": "web search",
                "http": "linked page", "full_document": "VERA-DAR's document collection",
                "stored": "VERA-DAR's document collection", "db": "VERA-DAR's document collection"}
ACQ_LABEL = {"full_document_acquired": "full document acquired and verified",
             "not_acquired": "not acquired (link only)", "in_collection": "already in VERA-DAR's collection"}
LICENCE_LABEL = {"allow": "licence allows use", "accept": "licence allows use", "hold": "licence unclear (held)",
                 "reject": "licence does not allow use"}
VERDICT_LABEL = {"supported": "supported", "partial": "partly supported"}
INLINE_STATUS = {"ok": "A new search found documents and checked them for this question.",
                 "none": "A new search ran but found no usable full document.",
                 "deadline": "A new search ran but did not finish within its time limit.",
                 "search_failed": "A new search was attempted but the search services did not respond.",
                 "unanswered": "A new search ran, but the documents found did not answer the question.",
                 "off": "No new search was made for this question (automatic search is off).",
                 "busy": "No new search was made (the service was busy).",
                 "rate": "No new search was made (the search rate limit was reached).",
                 "daily_cap": "No new search was made (today's search limit was reached).",
                 "error": "No new search was made (an internal error occurred)."}
DOC_OUTCOME = {"persisted": "added to VERA-DAR's collection", "below_persist_min":
               "used for this answer only (too few qualifying passages to keep)",
               "in_corpus": "already in VERA-DAR's collection", "write_failed": "could not be stored"}
STAGE_LABEL = {"search": "search", "reuse": "reuse of stored documents", "evaluate": "document checks",
               "write": "storing documents", "corpus": "collection rebuild", "total": "acquisition total"}
OTHER = "other"
SCOPE_OVERRIDE_NOTE = ("Scope policy: this question is outside VERA-DAR's curated scope. An answer was allowed only from "
                       "documents VERA-DAR acquired after licence, content and relevance checks. A supported claim means "
                       "its source says so; it does not show the answer is complete or on point.")
SCOPE_DECLINED_NOTE = ("Scope policy: this question is outside VERA-DAR's curated scope and no answer was authorized. "
                       "Any search or listing that followed was a reading list only, not evidence for an answer.")


# --- server-side resolution (never rendered) -------------------------------------------------------------------

def resolve_api(public: bool, env=None, local_url_file=None, typed_key: str = "") -> tuple[str, str]:
    """(base_url, api_key) for the page's server-side call, mirroring streamlit_app.py: public mode uses only the
    server env (VERA_API_BASE_URL, VERA_API_KEY); dev mode uses the local default URL and a key the developer typed
    (the server key is never used in dev mode). Both values stay in the page; neither is ever displayed."""
    from vera.public_mode import upstream_base_url
    env = os.environ if env is None else env
    if public:
        return upstream_base_url(env, local_url_file=None), str(env.get("VERA_API_KEY", ""))
    return upstream_base_url(env, local_url_file=local_url_file), str(typed_key or "")


def redactor(base_url: str = "", env=None):
    """Return text -> text that removes env secret values, the resolved API URL and its host, and key-shaped strings.
    Reuses the existing helper (vera.judge_select.scrub); only the URL/host entries are added to its mapping."""
    from vera.judge_select import scrub
    m = dict(os.environ if env is None else env)
    if base_url:
        m["_REPORT_API_URL"] = base_url
        host = urlparse(base_url).hostname or ""
        if host:
            m["_REPORT_API_URL_HOST"] = host
    return lambda text: scrub("" if text is None else str(text), m)


# --- projection (pure; allowlist) ------------------------------------------------------------------------------

def _d(v) -> dict:
    return v if isinstance(v, dict) else {}


def _int(v):
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _num(v) -> float:
    try:
        return 0.0 if isinstance(v, bool) else max(float(v), 0.0)
    except (TypeError, ValueError):
        return 0.0


def _str(v, limit: int) -> str:
    return ui_evidence._clean(v, limit) if isinstance(v, str) else ""


def _source_key(s: dict, i: int):
    """B2-1022: one key per document, so several admitted passages of one publication group together."""
    for k in ("identifier", "url", "title"):
        v = s.get(k)
        if isinstance(v, str) and v.strip():
            return (k, v.strip().lower())
    return ("row", i)


def _scope_policy(contract: dict, is_answer: bool) -> str:
    """B2-1022: say what the scope policy decided and whether later work authorized an answer or was only listing."""
    if contract.get("offtopic") is not True:
        return ""
    if is_answer:
        return SCOPE_OVERRIDE_NOTE
    return SCOPE_DECLINED_NOTE


def _tally(values, labels: dict) -> dict:
    out: dict = {}
    for v in values:
        lab = labels.get(v, OTHER) if isinstance(v, str) else "not recorded"
        out[lab] = out.get(lab, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def docs_inline(inline: dict) -> list:
    return ui_evidence._list(inline.get("documents"))


def _pipeline(data: dict, cc: dict, contract: dict, is_answer: bool, n_cands: int, n_fetched: int) -> list[str]:
    """B2-1022: one line per stage, kept apart: discovered, fetched, inspected, admitted, claims, authorization, answer.
    A stage the response does not report is said to be not reported, never inferred."""
    adm = data.get("admitted_evidence")
    m = ui_evidence._count(adm)
    k = ui_evidence._distinct_docs(adm) if isinstance(adm, list) else 0
    n = ui_evidence._distinct_docs(data.get("examined_full_documents"))
    checked, supported = ui_evidence._count(cc.get("checked")), ui_evidence._count(cc.get("supported"))
    authorized = m > 0 and (contract.get("offtopic") is not True or is_answer)  # admitted, so the writer ran
    return [f"Listed as candidates (found, not read): {n_cands}.",
            f"Documents fetched for this question: {n_fetched}." if n_fetched else
            "Documents fetched for this question: not reported separately.",
            f"Documents inspected: {n}.",
            f"Passages admitted as evidence: {m}" + (f", from {k} document{'s' if k != 1 else ''}." if m and k else "."),
            f"Claims checked: {checked}; supported by their source passage: {supported}." if cc else
            "Claims checked: none.",
            "Evidence authorized for answer generation: " + ("yes." if authorized else "no."),
            "Answer generated: " + ("yes." if is_answer else "no.")]


def build_live_report(data, question: str, latency_s: float) -> dict:
    """Allowlisted, display-ready projection of one /ask body. Never raises on malformed input."""
    data = _d(data)
    cap = ui_evidence._quote_cap()
    sources = ui_evidence._list(data.get("sources"))
    cites = ui_evidence._list(data.get("citations"))
    cands = ui_evidence._list(data.get("candidates"))
    traced = ui_evidence._list(data.get("traced_sources"))
    examined = ui_evidence._list(data.get("examined_full_documents"))
    cc = _d(data.get("claim_check"))
    contract = _d(data.get("contract"))
    inline = _d(contract.get("inline_acquisition"))
    is_answer = data.get("result_kind") == "answer"  # fail-closed, as on the Ask page
    fulldoc = ui_evidence.is_fulldoc(data)

    claims = []
    for c in cites:
        claims.append({"claim": _int(c.get("claim_id")), "n": _int(c.get("n")),
                       "verdict": VERDICT_LABEL.get(c.get("verdict"), "checked"),
                       "passage": _str(c.get("quote_span") or c.get("span") or "", REF_MAX),
                       "quote": _str(c.get("quote"), cap)})

    groups, by_n, by_doc = [], {}, {}
    for i, s in enumerate(sources, 1):
        n = _int(s.get("n")) or i
        g = by_doc.get(_source_key(s, i))
        if g is None:
            g = {"n": n, "ns": [], "passages": [], "title": _str(s.get("title") or s.get("url") or "source", 300),
                 "url": _str(s.get("url"), 2000), "published": _str(s.get("published"), 40),
                 "identifier": _str(s.get("identifier"), 120), "licence": _str(s.get("licence"), 120),
                 "excerpts": [], "attribution": None}
            groups.append(g)
            by_doc[_source_key(s, i)] = g
        g["ns"].append(n)
        if _str(s.get("span"), REF_MAX):
            g["passages"].append(_str(s.get("span"), REF_MAX))
        by_n.setdefault(n, g)
    for c, row in zip(cites, claims):
        g = by_n.get(row["n"])
        at = _d(c.get("attribution"))
        if g is None:  # a citation whose source row is absent: group it under its attribution
            g = {"n": row["n"], "ns": [row["n"]], "passages": [], "title": _str(at.get("title") or "source", 300),
                 "url": _str(at.get("source"), 2000), "published": "", "identifier": "", "licence": "", "excerpts": [],
                 "attribution": None}
            groups.append(g)
            by_n[row["n"]] = g
        g["excerpts"].append({"claim": row["claim"], "verdict": row["verdict"], "quote": row["quote"],
                              "passage": row["passage"]})
        if at and g["attribution"] is None:
            g["attribution"] = {k: _str(at.get(k), 300 if k != "licence_url" else 2000)
                                for k in ("creator", "title", "source", "licence", "licence_url", "changes")}

    inspected = [{"title": _str(d.get("title") or "document", 300), "url": _str(d.get("url"), 2000),
                  "published": _str(d.get("published"), 40), "chunks_inspected": _int(d.get("chunks_inspected")),
                  "candidate_spans": _int(d.get("candidate_spans")), "admitted_spans": _int(d.get("admitted_spans"))}
                 for d in examined]

    uniq_sources = list({_source_key(x, i): x for i, x in reversed(list(enumerate(sources, 1)))}.values())[::-1]
    found = uniq_sources + cands + traced  # distinct documents, not one entry per admitted passage
    docs = ui_evidence._list(inline.get("documents"))
    ms = _d(inline.get("ms"))
    search = {
        "results": len(found),
        "by_service": _tally([r.get("provider") for r in found], SERVICE_KIND),
        "by_acquisition": _tally([r.get("acquisition") for r in found if r.get("acquisition") is not None], ACQ_LABEL),
        "by_licence": _tally([r.get("licence_decision") for r in found if r.get("licence_decision") is not None],
                             LICENCE_LABEL),
        "inline": None if not inline else {
            "status": INLINE_STATUS.get(inline.get("status"), "A new search was considered (outcome: other)."),
            "answer_eligible": ui_evidence._count(inline.get("n_answer_eligible")),
            "stored": ui_evidence._count(inline.get("n_persisted")),
            "reused": ui_evidence._count(inline.get("reused")),
            "documents": _tally(["persisted" if d.get("persisted") is True else d.get("reason") for d in docs],
                                DOC_OUTCOME)},
        "stages_ms": {STAGE_LABEL[k]: _int(ms.get(k)) for k in STAGE_LABEL if _int(ms.get(k)) is not None},
    }

    return {
        "kind": "live", "question": _str(question, 2000), "is_answer": is_answer,
        "partial": is_answer and cc.get("answer_completeness") == "partial",
        "answer": str(data.get("answer") or "") if is_answer else "",
        "no_answer_reason": "" if is_answer else ui_evidence.fallback_text(
            contract.get("fallback"), n_inspected=ui_evidence._count(data.get("examined_full_documents"))),
        "claim_check": ui_evidence.claim_check_text(data.get("claim_check"), None, fulldoc) or "",
        "claims": claims, "groups": groups, "inspected": inspected,
        "reading_list": [] if is_answer else [{"n": _int(c.get("n")), "title": _str(c.get("title") or "source", 300),
                                               "url": _str(c.get("url"), 2000),
                                               "published": _str(c.get("published"), 40)}
                                              for c in cands[:CANDIDATES_SHOWN_MAX]],
        "scope_policy": _scope_policy(contract, is_answer),
        "pipeline": _pipeline(data, cc, contract, is_answer, len(cands), len(docs_inline(inline))),
        "scope": ui_evidence.evidence_scope_line(data) or "",
        "scope_source": ui_evidence.evidence_source_line(data) or "",
        "search": search,
        "usage": {"tokens": ui_evidence._count(data.get("tokens_used")), "cost_usd": _num(data.get("cost_usd")),
                  "latency_s": round(_num(latency_s), 2)},
    }


# --- rendering -------------------------------------------------------------------------------------------------

def _md(r, text) -> str:
    return r(safe_markdown(text))


def _ln(r, title, url) -> str:
    return r(_link(r(title), r(url)))


def render_live_report(st, rep: dict, r=None) -> None:
    """Draw one projected live report. `r` is the redactor (applied to every rendered string)."""
    r = r or redactor()
    st.success(LIVE_BANNER)
    st.subheader("Question")
    st.info(_md(r, rep.get("question")))

    st.header("1. Answer")
    if rep.get("is_answer"):
        (st.warning if rep.get("partial") else st.success)(_md(r, rep.get("answer")))
        if rep.get("partial"):
            st.caption("Partial answer: some parts of the question were not answered.")
    else:
        st.info(r(rep.get("no_answer_reason")))

    if rep.get("scope_policy"):
        st.caption(r(rep["scope_policy"]))

    st.header("2. Claim check")
    if rep.get("claim_check"):
        st.caption(r(rep["claim_check"]))
    rows = [{"Claim": c["claim"], "Check result": c["verdict"], "Source": f"[{c['n']}]" if c["n"] is not None else "?",
             "Passage": r(c["passage"])} for c in rep.get("claims") or []]
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info("No cited claims in this report.")

    st.header("3. Sources and short excerpts, by document")
    for g in rep.get("groups") or []:
        date = f" ({plain(r(g['published']))})" if g.get("published") else ""
        nums = ", ".join(str(x) for x in (g.get("ns") or [g["n"]]))
        st.markdown(f"**[{nums}]** {_ln(r, g['title'], g['url'])}{date}")
        if len(g.get("ns") or []) > 1 and g.get("passages"):
            st.caption("Admitted passages: " + "; ".join(plain(r(p)) for p in g["passages"]))
        if g.get("identifier"):
            st.caption(plain(r(g["identifier"])))
        # INT-1022-B disposition: each source's own licence as reported in its record (never a fixed "CC BY"); the
        # cited attribution line below carries it too, so it is shown here only when that line has none.
        if g.get("licence") and not (g.get("attribution") or {}).get("licence"):
            st.caption(f"Licence: {plain(r(g['licence']))}")
        for e in g.get("excerpts") or []:
            where = f" (passage {plain(r(e['passage']))})" if e.get("passage") else ""
            body = f"“{plain(r(e['quote']))}”" if e.get("quote") else "excerpt not shown"
            st.markdown(f"- claim {e['claim'] if e['claim'] is not None else '?'}, {e['verdict']}: {body}{where}")
        at = g.get("attribution")
        if at:
            parts = [plain(r(at["creator"])) if at.get("creator") else None,
                     _ln(r, at.get("title") or "source", at.get("source")) if at.get("title") or at.get("source") else None,
                     _ln(r, at["licence"], at.get("licence_url")) if at.get("licence") else None]
            line = ". ".join(p for p in parts if p)
            if at.get("changes"):
                line += f". Changes: {plain(r(at['changes']))}"
            if line:
                st.caption(line)
    if not rep.get("groups"):
        st.info("No source was used for an answer.")
    if rep.get("inspected"):
        st.markdown("**Documents inspected** (passage counts; no document text is shown)")
        st.dataframe([{"Document": r(d["title"]), "Published": r(d["published"]),
                       "Passages inspected": d["chunks_inspected"], "Candidate passages": d["candidate_spans"],
                       "Admitted passages": d["admitted_spans"]} for d in rep["inspected"]],
                     hide_index=True, use_container_width=True)
    if rep.get("reading_list"):
        st.markdown("**Reading list** (found but not used for an answer)")
        for i, c in enumerate(rep["reading_list"], 1):
            date = f" ({plain(r(c['published']))})" if c.get("published") else ""
            st.markdown(f"[{c['n'] if c['n'] is not None else i}] {_ln(r, c['title'], c['url'])}{date}")

    st.header("4. Evidence scope")
    st.caption(r(rep.get("scope") or "This report carries no full-document evidence fields."))
    if rep.get("scope_source"):
        st.caption(r(rep["scope_source"]))
    for line in rep.get("pipeline") or []:
        st.caption(r(line))

    st.header("5. Search and acquisition")
    s = rep.get("search") or {}
    st.caption(f"Results found or used: {s.get('results', 0)}. Counts only; individual search services are not named.")
    for title, key in (("By kind of service", "by_service"), ("By acquisition", "by_acquisition"),
                       ("By licence check", "by_licence")):
        if s.get(key):
            st.markdown(f"**{title}:** " + "; ".join(f"{k}: {v}" for k, v in s[key].items()))
    inl = s.get("inline")
    if inl:
        st.markdown(f"**New search for this question:** {inl['status']}")
        st.caption(f"Answer-eligible documents: {inl['answer_eligible']}; stored: {inl['stored']}; "
                   f"reused from the collection: {inl['reused']}."
                   + ("" if not inl.get("documents") else " Per document: "
                      + "; ".join(f"{k}: {v}" for k, v in inl["documents"].items()) + "."))

    st.header("6. Cost and latency")
    u = rep.get("usage") or {}
    a, b, c = st.columns(3)
    a.metric("Tokens used", f"{int(u.get('tokens') or 0):,}")
    b.metric("Cost (USD)", f"${float(u.get('cost_usd') or 0):.6f}")
    c.metric("Latency (s)", f"{float(u.get('latency_s') or 0):.2f}")
    st.caption("Latency is the round trip measured by this page. Cost is as reported by the service; calls to the free-tier provider are recorded at $0.")
    if s.get("stages_ms"):
        st.dataframe([{"Stage": k, "Time (ms)": v} for k, v in s["stages_ms"].items()], hide_index=True,
                     use_container_width=True)


def remember(session: dict, rep: dict) -> list:
    """Keep the newest RECENT_MAX projected reports in this session (never the raw body)."""
    recent = [x for x in session.get(RECENT_KEY, []) if isinstance(x, dict) and x.get("question") != rep.get("question")]
    session[RECENT_KEY] = ([rep] + recent)[:RECENT_MAX]
    return session[RECENT_KEY]
