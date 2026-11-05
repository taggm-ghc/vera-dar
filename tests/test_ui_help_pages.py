"""UI-ABOUT-1031 / OD-1029-UI: the Ask-page required notice, the About page (full disclosure, split only) and the
How to use page. Offline; AppTest renders each page."""
import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from vera import disclosure, help_pages

ROOT = Path(__file__).resolve().parent.parent
@pytest.fixture(autouse=True)
def _web_read_off(monkeypatch):
    """Web route pinned off: these tests compare against the unwidened _BASE_FULLDOC; the web-on wording is tested in test_disclosure.py."""
    monkeypatch.setattr(disclosure, "web_read_on", lambda: False)


_NORM = lambda t: re.sub(r"\s+", " ", t).strip()  # noqa: E731


def _fd(tmp_path, on):
    p = tmp_path / "fd.json"
    p.write_text(json.dumps({"fulldoc": {"enabled": on}}))
    return p


@pytest.mark.parametrize("on", [True, False])
def test_sections_join_to_the_full_disclosure_verbatim(tmp_path, on):
    fp = _fd(tmp_path, on)
    secs = disclosure.disclosure_sections(fulldoc_path=fp)
    assert _NORM(" ".join(t for _, t in secs)) == _NORM(disclosure.disclosure_text(fulldoc_path=fp))
    assert all(h and t for h, t in secs)


def test_required_notice_keeps_the_seven_groups_verbatim_and_drops_the_movable_ones():
    req, full = disclosure.required_notice_text(), disclosure.disclosure_text()
    for must in ("VERA-DAR is an AI system.", "may be wrong. Check the sources and read them yourself before relying on any claim.",
                 "VERA-DAR keeps no profile of you and does not tailor answers to you.",
                 "are sent to third-party AI services.", disclosure._THIRD_PARTY.strip(), disclosure._NO_SAVE.strip(),
                 disclosure._AUTO_ACQUIRE_MAY_STORE.strip(), "Some requests are refused for safety reasons."):
        assert must in req, must
    for moved in ("first searches a small collection", "Licences for documents in its own collection",
                  "For listed records, only the indexing", "in the same request"):
        assert moved in full and moved not in req, moved
    assert len(req.split()) < int(len(full.split()) * 0.55)  # NEWS-LANE-1104 / retention sentence: both texts grew; still well under the full text


def test_required_notice_follows_flags(tmp_path):
    chain = tmp_path / "chain.json"
    chain.write_text(json.dumps({"corpus": {"enabled": False}}))
    t = disclosure.required_notice_text(chain_path=chain)
    assert disclosure._CORPUS_MAY_STORE.strip() not in t
    assert disclosure._THIRD_PARTY.strip() in t and disclosure._NO_SAVE.strip() in t
    assert disclosure.required_notice_text(fulldoc_path=_fd(tmp_path, False)).startswith(disclosure._BASE)


def test_json_holds_no_disclosure_constants_or_internal_terms():
    raw = (ROOT / "config" / "ui_help_pages.json").read_text()
    text = json.loads(raw)
    assert text["about"] and text["how_to_use"]["sections"]
    for c in (disclosure._THIRD_PARTY, disclosure._NO_SAVE, disclosure._AUTO_ACQUIRE_MAY_STORE, disclosure._BASE_FULLDOC):
        assert c.strip() not in raw
    from tests.test_item78_sources_sidebar import _no_internal_terms
    prose = " ".join(p for s in text["how_to_use"]["sections"] for p in s["paragraphs"])
    _no_internal_terms(prose)
    assert not re.search(r"https?://|www\.|onrender|localhost", raw, re.I)


def _run(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _text(at):
    return " ".join(e.value for k in (at.title, at.subheader, at.markdown, at.caption) for e in k)


def test_about_page_renders_full_disclosure_once_under_headings():
    at = _run("pages/about.py")
    t = _NORM(_text(at))
    assert not any(_NORM(disclosure.disclosure_text()) in _NORM(c.value) for c in at.caption)  # split under headings
    moved = set(json.loads((ROOT / "config" / "ui_help_pages.json").read_text())["about"]["moved_to_terms"])
    for h, body in disclosure.disclosure_sections():
        if h in moved:  # PRIVACY-1104: terms-type sections render on the Terms page only
            assert h not in [s.value for s in at.subheader]
        else:
            assert h in [s.value for s in at.subheader] and _NORM(body) in t
    assert [x.value for x in at.title] == ["About VERA-DAR"]
    assert disclosure._NO_SAVE.strip() not in t  # retention text moved to Terms


def test_how_to_use_page_renders_sections_and_uses_config_scope():
    at = _run("pages/how_to_use.py")
    heads = [s.value for s in at.subheader]
    for h in ("What to ask", "What you get back", "How to read the citations", "Limits", "Terms and privacy"):
        assert h in heads, h
    t = _text(at)
    assert "{scope}" not in t and "{privacy}" not in t and "{" not in t
    # HOTFIX-BLURB-1104: retention and third-party detail are About-only; How to use only points there
    assert disclosure._NO_SAVE.strip() not in _NORM(t) and disclosure._THIRD_PARTY.strip() not in _NORM(t)
    assert "The Terms of Access, Browsing, and Use page says what VERA-DAR keeps from your question" in _NORM(t)


def test_unreadable_config_shows_fallback(tmp_path):
    class St:
        def __init__(self): self.out = []
        def title(self, x): self.out.append(x)
        def info(self, x): self.out.append(x)
        def markdown(self, x): self.out.append(x)
    s = St()
    help_pages.render_about(s, cfg=help_pages.load(tmp_path / "missing.json"))
    assert help_pages.FALLBACK in s.out


def test_sidebar_nav_has_both_pages_and_files_exist():
    from vera.sources_sidebar import NAV
    assert NAV[-3:] == (("pages/how_to_use.py", "How to use"), ("pages/about.py", "About"),
                        ("pages/terms.py", "Terms of Access, Browsing, and Use"))
    assert all((ROOT / p).exists() for p, _ in NAV)
    assert {t for t, _ in disclosure.ABOUT_LINKS} == {"pages/about.py", "pages/how_to_use.py"}


# ---- UI-ABOUT-FIX-1031 -------------------------------------------------------------------------------------
def _howto_text():
    cfg = json.loads((ROOT / "config" / "ui_help_pages.json").read_text())
    return " ".join(p for s in cfg["how_to_use"]["sections"] for p in s["paragraphs"]), cfg


def test_json_has_no_internal_task_notes():
    raw = (ROOT / "config" / "ui_help_pages.json").read_text()
    cfg = json.loads(raw)
    assert "todo_owner" not in raw and "_note" not in raw and "todo_owner" not in cfg["how_to_use"]
    assert not re.search(r"OD-\d|UI-ABOUT|\bC5\b|main\.py", raw)


def test_answer_claim_matches_off_topic_answering():
    text, _ = _howto_text()
    assert "written only from passages in the collection" not in text
    assert "{answer_sources}" in text
    assert "in the same request" in disclosure._OFFTOPIC_ANSWERING  # wording derives from the approved sentence


def test_no_answer_wording_is_plain():
    text, _ = _howto_text()
    assert "No answer from VERA-DAR is a correct result" not in text
    assert "Declining to answer is a valid outcome, not an error" in text


def test_about_sidebar_text_matches_what_the_sidebar_shows():
    _, cfg = _howto_text()
    t = " ".join(p for s in cfg["about"]["after_disclosure"] for p in s["paragraphs"])
    assert "never names an individual document" not in t.lower() and "totals only" not in t
    for shown in ("number of sources", "newest source", "licence note", "language model", "current coverage", "topics",
                  "link to this page", "Trace Eval"):
        assert shown in t, shown
    assert "last updated" not in t and "scope" not in t.lower()  # CORPUS-DIGEST-1104: no fixed Scope text
    from vera import sources_sidebar as sb  # each claimed element exists in the renderer
    import inspect
    side = inspect.getsource(sb.render_corpus) + inspect.getsource(sb.render_sources_sidebar) + inspect.getsource(sb._about_link)
    for el in ("newest source", "topics", "CHECKED", "ABOUT_LINK_LABEL"):
        assert el in side, el
    for gone in ("LICENCE_SIDEBAR", "scope_text", "how_answers_text", "how_answers_base", "trace_eval_line"):
        assert gone not in side, gone  # moved off the sidebar
    notes = inspect.getsource(sb.corpus_notes_lines)  # ... to the About page
    for el in ("LICENCE_SIDEBAR", "how_answers_base", "trace_eval_line", "coverage_text"):
        assert el in notes, el
    assert "render_corpus_notes" in inspect.getsource(help_pages.render_about)


def test_wake_time_comes_from_config(monkeypatch):
    text, _ = _howto_text()
    assert "3 minutes" not in text and "{wake_minutes}" in text
    from vera import ui_wake
    assert help_pages.wake_minutes() == max(1, round(ui_wake.load_config().max_total_s / 60))
    monkeypatch.setattr(ui_wake, "load_config", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    assert help_pages.wake_minutes() is None  # no hardcoded fallback; the sentence is omitted
    out = []
    class St:
        def subheader(self, x): pass
        def markdown(self, x): out.append(x)
    help_pages._section(St(), {"heading": "h", "paragraphs": ["a {wake_minutes} b", "kept"]})
    assert out == ["kept"]


def test_report_paragraphs_describe_code_behaviour():
    text, cfg = _howto_text()
    assert "sends your question" not in text and "Guardrail" not in text and "rejected-answer" not in text
    assert "illustrative, not a real run" in text
    # labels live in the page: its OFFLINE label must contain the quoted text. Read the source rather than importing
    # the page: a bare-mode import executes its st.form outside any ScriptRunContext and leaves that form on the
    # shared delta-generator stack, so every later AppTest of a page with a form fails "Forms cannot be nested"
    # (NONASK-1103 WP6 root cause of the order-dependent test_ui_progress / test_ui_wake / test_ui_wake_page failures).
    src = (ROOT / "pages" / "vera_demo.py").read_text(encoding="utf-8")
    assert '"Offline example (illustrative, not a real run)"' in src and "mode == OFFLINE" in src


@pytest.mark.parametrize("fd,off,corp,auto,layers,recall", list(__import__("itertools").product(
    [True, False], ["none", "plain", "answering"], [True, False], [True, False], [("L0",), ("L0", "L2")], [True, False])))
def test_disclosure_state_equivalence_96(monkeypatch, fd, off, corp, auto, layers, recall):
    """Regression guard only, NOT a base-versus-change comparison: expected values are derived from the changed
    disclosure code. Sections join to the full text; the required notice equals it minus the mechanism (S2), licence (S4)
    and off-topic sentences, with the verbatim refusal sentence replacing the off-topic one. Base equivalence is
    evidenced separately by scratch/UI-ABOUT-REVIEW-1031/cmp.py (see README)."""
    from vera import answer_contract
    o = {"none": "", "plain": disclosure._OFFTOPIC, "answering": disclosure._OFFTOPIC_ANSWERING}[off]
    monkeypatch.setattr(disclosure, "fulldoc_enabled", lambda *a, **k: fd)
    monkeypatch.setattr(disclosure, "offtopic_sentence", lambda: o)
    monkeypatch.setattr(disclosure, "corpus_admission_enabled", lambda *a, **k: corp)
    monkeypatch.setattr(disclosure, "auto_acquire_enabled", lambda: auto)
    monkeypatch.setattr(answer_contract, "layers", lambda *a, **k: layers)
    monkeypatch.setattr(answer_contract, "evidence_recall_enabled", lambda *a, **k: recall)
    full, req = disclosure.disclosure_text(), disclosure.required_notice_text()
    secs = " ".join(t for _, t in disclosure.disclosure_sections())
    assert _NORM(secs) == _NORM(full)
    bf = disclosure.base_fulldoc()
    s2 = bf[bf.index("For each"):bf.index("Answers and lists")]
    s4 = bf[bf.index("Licences for documents"):bf.index("VERA-DAR keeps no")]
    expect = full
    if fd:
        expect = expect.replace(s2, "").replace(s4, "")
    if o:
        expect = expect.replace(o, " Some requests are refused for safety reasons.")
    assert _NORM(req) == _NORM(expect)
    assert req == req.strip() and "  " not in req
    assert disclosure._REFUSED in disclosure._OFFTOPIC and disclosure._REFUSED in disclosure._OFFTOPIC_ANSWERING


@pytest.mark.parametrize("fd,answering", [(True, True), (True, False), (False, True), (False, False)])
def test_answer_sources_follows_flags(monkeypatch, fd, answering):
    monkeypatch.setattr(disclosure, "fulldoc_enabled", lambda *a, **k: fd)
    monkeypatch.setattr(disclosure, "offtopic_sentence",
                        lambda: disclosure._OFFTOPIC_ANSWERING if answering else disclosure._OFFTOPIC)
    t = help_pages._fill("{answer_sources}")
    assert "{answer_sources}" not in t
    assert ("open-access documents found in the same request" in t) == (fd and answering)
    assert ("Those documents are in VERA-DAR's collection." == t) == (fd and not answering)
    assert ("switched off" in t) == (not fd)


def test_about_intro_and_no_subset_claim():
    cfg = json.loads((ROOT / "config" / "ui_help_pages.json").read_text())
    intro = cfg["about"]["intro"]
    assert "Other pages show a one-line notice that links here." in intro and "subset" not in intro


def test_how_to_use_text_matches_the_layered_ui():
    raw = (ROOT / "config" / "ui_help_pages.json").read_text()
    how = json.dumps(json.loads(raw)["how_to_use"])
    assert "Metrics" not in how and "tokens" not in how and "Under an answer" not in how
    assert "Research and verification" in how and "historical" in how.lower()


def test_live_stats_line_only_when_flag_on(tmp_path):
    f = tmp_path / "ls.json"
    assert help_pages.live_stats_enabled(tmp_path / "absent.json") is False
    f.write_text(json.dumps({"enabled": False}))
    assert help_pages.live_stats_enabled(f) is False
    f.write_text(json.dumps({"enabled": True}))
    assert help_pages.live_stats_enabled(f) is True

    class St:
        def __init__(self): self.md = []
        def title(self, x): pass
        def info(self, x): pass
        def subheader(self, x): pass
        def markdown(self, x): self.md.append(x)
        def caption(self, x): pass  # HOTFIX-BLURB-1104: About renders the corpus description
    from vera import sources_sidebar as sb_mod
    orig_corpus = sb_mod._corpus
    for on in (False, True):
        s = St()
        sb_mod.__dict__["_corpus"] = lambda st: None  # offline: no corpus description fetch
        orig = help_pages.live_stats_enabled
        help_pages.live_stats_enabled = lambda *a, **k: on
        try:
            help_pages.render_about(s)
            about_md = list(s.md)
            s.md.clear()
            help_pages.render_terms(s)  # PRIVACY-1104: the live-stats privacy line moved to the Terms page
        finally:
            help_pages.live_stats_enabled = orig
            sb_mod._corpus = orig_corpus
        assert help_pages.LIVE_STATS_LINE not in about_md
        assert s.md.count(help_pages.LIVE_STATS_LINE) == (1 if on else 0)


# ---- PRIVACY-1104: Terms of Access, Browsing, and Use ------------------------------------------------------
def test_terms_page_renders_moved_sections_and_hides_placeholders():
    at = _run("pages/terms.py")
    t = _NORM(_text(at))
    heads = [s.value for s in at.subheader]
    assert [x.value for x in at.title] == ["Terms of Access, Browsing, and Use"]
    for h in ("Acceptable use", "AI answers may be wrong; no warranty", "Third-party sources and licences", "Privacy",
              "Cookies", "Changes to these terms", "Governing law and contact"):
        assert h in heads, h
    moved = set(json.loads((ROOT / "config" / "ui_help_pages.json").read_text())["about"]["moved_to_terms"])
    for h, body in disclosure.disclosure_sections():
        if h in moved:
            assert _NORM(body) in t, h
    assert "[[" not in t and "{" not in t  # placeholders never render
    # owner decisions 2026-11-05: operator, governing law, no email (repository issues), effective date
    assert "Tagg Maiwald" in t and "State of Nevada, USA" in t and "github.com/taggm-ghc/vera-dar/issues" in t
    assert "@" not in t
    assert "Effective date: 2026-10-20" in [c.value for c in at.caption]
    assert "_streamlit_xsrf" in t
    assert disclosure.short_notice_text() in [c.value for c in at.caption]  # R-031 short notice on this page too


def test_terms_release_gate_no_placeholders_left():
    """Release gate: fails while any owner placeholder remains in the shipped config (expected to fail until filled)."""
    raw = (ROOT / "config" / "ui_help_pages.json").read_text()
    assert "[[" not in raw, sorted(set(re.findall(r"\[\[([A-Z]+-[A-Z0-9]+)", raw)))
