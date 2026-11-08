"""NOANSWER-PAGE-1108: the no-answer page says each sentence once, shows no empty section, and its counts agree.

Owner report on build eea3264 ("Where did vera come from", "what are quarks?"): the opening sentence repeated under an
empty "Reading list" heading (D1), the stop sentence twice (D2), "none of the documents fetched" with nothing fetched
(D3), an empty "Research and verification" section (D4), progress facts shown twice (D5), the background sentence twice
in two wordings (D6) and three reasons chained with "and ... and" (D7).

Real code paths, no hand-set display fields: candidate records -> research_outcome.annotate_diag ->
inline_acquire._supply_status / stop_clause -> ask_service._inline_notice (the answer notice) -> evidence_ledger.from_inline
-> ask_service._with_ledger (contract.inline_acquisition) -> main.py /ask (limitations) -> streamlit_app.py (AppTest).
Fake: only the answer chain (answer_via_chain returns the AskResult built from the above) and HTTP.
"""
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import tests.test_ui_render_extract as base
from vera import ask_service, evidence_ledger, inline_acquire, page_text, progress_ui, research_outcome, ui_evidence
from vera.ask_service import AskResult
from vera.fulldoc_evidence import EvidenceBundle
from vera.inline_acquire import Outcome

ROOT = Path(__file__).resolve().parent.parent
ACQ = json.loads((ROOT / "config" / "auto_acquire.json").read_text())
REASON_CODE = "insufficient_full_document_evidence"


@pytest.fixture(autouse=True)
def _legacy_page(monkeypatch):
    """The page is rendered through the legacy /ask call (the stream path ends in the same render_result)."""
    from vera import progress
    monkeypatch.setattr(progress, "stream_enabled", lambda: False)


def _rec(i, outcome, reason, **kw):
    r = {"portal": "openalex", "doi": f"10.9999/np.{i}", "url": f"https://example.org/np/{i}", "outcome": outcome,
         "reason": reason, **kw}
    return research_outcome.annotate_diag(r)


def _outcome(recs, stored=0, admitted=0, stop="no_progress", status="none"):
    o = Outcome(status=status)
    o.candidates = recs
    o.stored, o.admitted = stored, admitted
    codes: dict = {}
    for r in recs:
        if r["outcome"] not in ("admitted", "reused"):
            codes[r["reason"]] = codes.get(r["reason"], 0) + 1
    tgt = 6
    st = "met" if stored >= tgt else inline_acquire._supply_status(ACQ, codes, o)  # the real status rule
    o.acquisition = {"status": st, "stored": stored, "target": tgt, "rounds": 3, "stop": stop}
    return o


# The three representative attempts named in the contract.
def _found_none_read():  # the owner's page: 31 found, 27 judged not relevant before any fetch, 4 not retrieved
    return ([_rec(i, "dropped", "R-ACQ-IRRELEVANT") for i in range(27)]
            + [_rec(27 + i, "rejected", "R-ACQ-NODOI") for i in range(4)])


def _some_read_none_kept():  # 10 found: 5 not relevant before fetch, 3 fetched and rejected on passages, 2 no route
    return ([_rec(i, "dropped", "R-ACQ-IRRELEVANT") for i in range(5)]
            + [_rec(5 + i, "rejected", "R-ACQ-FEWSPANS") for i in range(3)]
            + [_rec(8 + i, "rejected", "R-ACQ-NODOI") for i in range(2)])


def _some_saved():  # 8 found: 2 read and saved, 4 not relevant before fetch, 2 no route
    return ([_rec(i, "admitted", "R-ACQ-OK", persisted=True) for i in range(2)]
            + [_rec(2 + i, "dropped", "R-ACQ-IRRELEVANT") for i in range(4)]
            + [_rec(6 + i, "rejected", "R-ACQ-NODOI") for i in range(2)])


CASES = {"found_none_read": (_found_none_read, 0), "some_read_none_kept": (_some_read_none_kept, 0),
         "some_saved": (_some_saved, 2)}


def _result(recs, stored, *, background=False, research=None):
    o = _outcome(recs, stored=stored, admitted=stored)
    if background:
        o.background = {"status": "ACQ-QUEUED", "queued": 2}
    if stored:  # the written-but-unanswered path (ask_service._inline lines ~649-652)
        notice = inline_acquire.notice(ACQ, "unanswered") + inline_acquire.stop_clause(o.acquisition)
    else:
        notice = ask_service._inline_notice(ACQ, o)
    bg = inline_acquire.background_note(ACQ, o)
    if bg:
        notice = ask_service._InlineNotice(str(notice))
        notice.bg_note = bg
    ledger = evidence_ledger.from_inline(o, acq=ACQ)
    ia = ask_service._with_ledger(o.public(), ledger, ACQ)
    head = ui_evidence.fallback_text(REASON_CODE, n_matched=0)  # the server's headline (ask_service._with_fetched: same text)
    answer, notes = ask_service._nonanswer_note(head, (), notice, strip=False)
    ev = EvidenceBundle(question="q", examined_full_documents=(
        {"document_id": "d1", "canonical_source_id": "c1", "title": "Stored", "url": "https://example.org/s",
         "chunks_inspected": 4, "candidate_spans": 2, "relevant_spans": 0, "admitted_spans": 0},),
        status="insufficient", corpus_source="db")
    contract = {"layers": ["L0", "L2"], "fallback": REASON_CODE, "inline_acquisition": ia}
    if research:
        contract["research"] = research
    return AskResult(answer=answer, tokens_used=1, cost_usd=0.0, sources=(), accepted=True, result_kind="reading_list",
                     candidates=(), evidence=ev, notes=notes, contract=contract)


def _body(monkeypatch, result):
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    import main
    from vera import public_mode as pm
    pm.reset_limiter(None)
    monkeypatch.setattr(main, "answer_via_chain", lambda q, e, g: (result, "groq:m"))
    monkeypatch.setattr(main, "store_from_env", lambda cfg: None)
    monkeypatch.setattr(main, "memory_from_env", lambda block: None)
    r = TestClient(main.app).post("/ask", json={"question": "Where did vera come from"},
                                  headers={"X-API-Key": "k" * 20})
    assert r.status_code == 200, r.text
    return r.json()


def _page(monkeypatch, body):
    at = base._run(monkeypatch, body)
    elems = [("markdown", e.value) for e in at.markdown] + [("info", e.value) for e in at.info] \
        + [("caption", e.value) for e in at.caption] + [("warning", e.value) for e in at.warning]
    return at, elems


_FIXED = ("💡 Ask a follow-up question", )


def _sentences(elems):
    out = []
    for _k, v in elems:
        if any(v.startswith(f) for f in _FIXED):
            continue
        out += [" ".join(s.replace("\\", "").replace("*", "").split()).casefold() for s in page_text.sentences(v)]
    return out


def _answer_area(elems):
    """Sentences of the result area only (from the why-no-answer line on; the page chrome above is not the result)."""
    head = ui_evidence.fallback_text(REASON_CODE, n_matched=0)
    i = next(i for i, (_k, v) in enumerate(elems) if head.split(",")[0] in v)
    return elems[i:]


@pytest.mark.parametrize("case", sorted(CASES))
def test_no_sentence_twice_and_no_empty_heading(monkeypatch, case):
    mk, stored = CASES[case]
    body = _body(monkeypatch, _result(mk(), stored))
    at, elems = _page(monkeypatch, body)
    # result area = everything rendered for the result (order within one element type kept by AppTest)
    res = [(k, v) for k, v in elems if not v.startswith("💡")]
    sents = [s for s in _sentences(res) if len(s) > 25]
    dup = {s for s in sents if sents.count(s) > 1}
    assert not dup, f"repeated on the page: {sorted(dup)}"
    heads = [v for k, v in elems if k == "markdown" and v.startswith("#")]
    assert "### Reading list" not in heads  # D1: no list, no heading
    for e in at.expander:  # D4: no empty section
        assert list(e.caption) or list(e.markdown) or list(e.table), f"empty section {e.label!r}"


def test_owner_page_d1_d2_d4(monkeypatch):
    body = _body(monkeypatch, _result(_found_none_read(), 0))
    at, elems = _page(monkeypatch, body)
    allv = " ".join(v for _k, v in elems)
    head = ui_evidence.fallback_text(REASON_CODE, n_matched=0)
    assert allv.count(head) == 1  # D1: the opening sentence once
    assert allv.count("stopped after two further searches found nothing new") == 1  # D2
    assert not any(e.label == "Research and verification" for e in at.expander)  # D4: nothing to put in it


def _nums(text, pat):
    m = re.search(pat, text)
    return int(m.group(1)) if m else None


@pytest.mark.parametrize("case", sorted(CASES))
def test_counts_agree_with_one_attempt_record(monkeypatch, case):
    """D3 + checklist item 7 (neighbour check): every count and process claim on the page equals the same attempt's
    records: found, judged not relevant, not retrieved, fetched (text examined), read (counted), saved."""
    mk, stored = CASES[case]
    recs = mk()
    body = _body(monkeypatch, _result(recs, stored))
    _at, elems = _page(monkeypatch, body)
    page = " ".join(v.replace("\\", "") for _k, v in elems)
    led = body["contract"]["inline_acquisition"]["ledger"]
    truth = {"found": len(recs),
             "irrelevant": sum(r["outcome"] == "dropped" and r["reason"] == "R-ACQ-IRRELEVANT" for r in recs),
             "fetched": sum(r["outcome"] == "rejected" and r["reason"] in ("R-ACQ-FEWSPANS", "R-ACQ-IRRELEVANT")
                            for r in recs),
             "read": sum(r["outcome"] == "admitted" for r in recs), "saved": stored}
    assert led["examined_unqualified"] == truth["fetched"] and led["counted"] == truth["read"]
    assert led["stored"] == truth["saved"]
    found = _nums(page, r"It found (\d+) documents")
    if found is not None:
        assert found == truth["found"]
        assert _nums(page, r"(\d+) judged not relevant") == truth["irrelevant"]
    assert _nums(page, r"Read (\d+) of the 6 sources targeted") == truth["read"]
    n_f = _nums(page, r"(?:none of the|the) (\d+) documents? fetched")
    if truth["fetched"]:
        assert n_f == truth["fetched"]
    else:  # D3: never a claim about fetched documents when none was fetched
        assert "documents fetched" not in page and "document fetched" not in page
    saved = _nums(page, r"(\d+) saved to VERA-DAR's collection")
    assert (saved or 0) == truth["saved"]


def test_background_sentence_once_and_reasons_listed(monkeypatch):
    """D6: the background hand-off is one sentence, said once; D7: the reasons are one plain list, not 'and ... and'."""
    att = [research_outcome.PortalAttempt("p1", "quota")]
    cand = [research_outcome.CandidateOutcome("c1", terminal_reason="queued_background")]
    ro = research_outcome.summarize_outcome(att, cand, budget_exhausted=True, older_evidence_available=False,
                                            new_evidence_admitted=False, extra_causes=("crawl_delay",))
    assert " and the search budget ran out and " not in ro.user_text
    assert ("because a search service's request limit was reached, the search budget ran out and a source asks "
            "crawlers to wait longer than this search could, so") in ro.user_text
    research = {"user_text": ro.user_text, "triggered": True}
    body = _body(monkeypatch, _result(_found_none_read(), 0, background=True, research=research))
    _at, elems = _page(monkeypatch, body)
    page = " ".join(v.replace("\\", "") for _k, v in elems)
    assert page.count("may be acquired in the background on a best-effort basis") == 1, page
    assert page.count("queued sources are not kept if the service restarts") == 1, page  # BGQUEUE-CHECK-1108
    assert "Research outcome:" in page  # the research outcome is still shown (its other sentence)


def _labels():
    return progress_ui.Labels.load()


def _ev(stage, state, **c):
    return {"type": "event", "stage": stage, "state": state, "counts": c}


def test_progress_area_states_each_fact_once():
    """D5: the owner's progress area: the status line states Checked 2 of 4 and 0 of 6 saved; the bars below do not
    repeat them as text. The ~10 s activity line stays."""
    lab = _labels()
    h = progress_ui.StreamHolder(t0=0.0, last_read=0.0)
    h.events = [_ev("search", "start", step=1), _ev("search", "done", found=31, shortlisted=4),
                _ev("acquire", "update", checked=2, shortlisted=4, usable=0, held=0, stored=0, target=6)]
    h.activity = {"act": 3, "k": 2, "elapsed": 12}

    class UI:
        def __init__(self): self.texts = []
        def status(self, label, **k): self.texts.append(label); return self
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def write(self, t): self.texts.append(t)
        def progress(self, frac, text=None): self.texts.append(text or "")
        def info(self, t): self.texts.append(t)
        def caption(self, t): self.texts.append(t)
    ui = UI()
    progress_ui.stream_tick(h, {}, ui, lambda: 13.0, lab)
    blob = " | ".join(ui.texts)
    assert blob.count("Checked 2 of 4") == 1, blob
    assert blob.count("0 of 6") == 1, blob
    assert "Still reading 2 source(s) at once" in blob  # the activity update is kept
    v = progress_ui.stream_view(h, lab, 13.0)
    assert len(v["bars"]) == 2  # the bars still show progress, without repeating the text


def test_page_text_unit():
    s = page_text.Seen()
    s.add("One here. Two there.")
    assert s.new("Two there. Three now. Three now.") == "Three now."


# ---- D8: a designed skip (variant rounds are scholarly-only) is never reported as "the search budget ran out" ----
def _portals(cap, scholarly_only):
    from vera import search_providers as sp
    c = {"search": {"ddg_first": {"enabled": True}, "europepmc": {"enabled": True}, "providers": ["openalex", "arxiv"]},
         "bounds": {"max_search_calls_per_job": cap}}
    fn = sp.acquisition_search(lambda q, n, t: [], c, ddg_search=lambda *a, **k: [], transport=lambda *a, **k: None,
                               get=lambda *a, **k: None, url_check=lambda *a, **k: True, wiki_get=lambda *a, **k: None)
    tok = sp.SCHOLARLY_ONLY.set(scholarly_only)
    try:
        with sp.portal_scope() as lst:
            try:
                fn("quarks", 5, 2.0)
            except Exception:  # noqa: BLE001  stub providers may fail; only the recorded portals matter
                pass
    finally:
        sp.SCHOLARLY_ONLY.reset(tok)
    return list(lst)


def _outcome_text(portals):
    from vera import research_record
    rec = research_record.summarize(triggers=["insufficient"], inline={"status": "none", "portals": portals,
                                                                        "candidates": []},
                                    used_new=False, older_available=False, fr_before=None)
    return rec.get("user_text") or rec.get("summary") or ""


def test_designed_scholarly_only_skip_is_not_a_budget_stop():
    round0 = _portals(7, False)
    variant = _portals(7, True)  # a later round: DDG and Wikipedia are skipped by design
    assert all(p["portal"] != "ddg" for p in variant), variant
    assert "budget ran out" not in _outcome_text(round0 + variant)
    # control: a portal really left without a search-call slot is still reported as the budget running out
    assert "budget ran out" in _outcome_text(_portals(1, False))


# ---- D9: the "not retrieved" reasons are the records' own reason classes, crawl delay included ----
def _with_crawl_delay():
    return ([_rec(i, "dropped", "R-ACQ-IRRELEVANT") for i in range(22)]
            + [_rec(22 + i, "rejected", "R-ACQ-CRAWL_DELAY") for i in range(5)]
            + [_rec(27 + i, "rejected", "R-ACQ-NODOI") for i in range(4)])


def test_not_retrieved_reasons_follow_the_reason_codes(monkeypatch):
    recs = _with_crawl_delay()
    body = _body(monkeypatch, _result(recs, 0))
    _at, elems = _page(monkeypatch, body)
    page = " ".join(v.replace("\\", "") for _k, v in elems)
    assert "9 not retrieved (" in page, page
    assert "5 because the source's required pause between requests did not fit in the time left" in page
    assert "4 with no identifier or retrieval route VERA-DAR can use" in page
    assert "already held, no identifier or no retrieval route" not in page  # the old fixed list


def test_restart_clause_only_when_something_was_queued(monkeypatch):
    """BGQUEUE-CHECK-1108: the not-kept-on-restart clause appears only with a real hand-off."""
    body = _body(monkeypatch, _result(_found_none_read(), 0))  # nothing queued
    _at, elems = _page(monkeypatch, body)
    assert "not kept if the service restarts" not in " ".join(v.replace("\\", "") for _k, v in elems)
