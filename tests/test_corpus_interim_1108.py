"""CORPUS-INTERIM-1108 (owner 2026-11-08): the Corpus Description page without per-subject numbers, assignments or the
subject list; digest + count + years, else deterministic frequent subject phrases; the switch restores the old view.
Real render paths (sources_sidebar.render_corpus, help_pages.render_corpus_page via AppTest); fake engine; no DB, no model."""
import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.test_corpus_desc_1107 import DIGEST, ENTRIES, FALLBACK, ROWS, _capture, _describer, _no_llm, _NORM, _offline  # noqa: F401
from vera import corpus_description as cd, corpus_interim as ci, sources_sidebar as sb

ROOT = Path(__file__).resolve().parent.parent
TITLES = ([f"Influenza forecasting with compartmental models, study {i}" for i in range(4)]
          + [f"Coral reef bleaching and ocean warming in region {i}" for i in range(3)]
          + ["A one-off paper on quantum chromodynamics lattices"])
PHR = ["compartmental models", "coral reef bleaching", "influenza forecasting"]
INTERIM = {"published_years": ["2005", "2026"], "frequent_subjects": ["influenza forecasting", "coral reef bleaching"]}
NUMBERS = re.compile(r"\d+\s*%|under 1%|not yet matched|subject areas|including|\bof \d+ documents", re.I)


@pytest.fixture
def off(monkeypatch):
    monkeypatch.setattr(sb, "subjects_enabled", lambda: False)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(sb, "subjects_enabled", lambda: True)


def test_shipped_switch_is_off_and_sidebar_reads_it():
    cfg = json.loads((ROOT / "config" / "ask-provider-chain.json").read_text())
    assert cfg["corpus_description"]["subject_areas"]["enabled"] is False and sb.subjects_enabled() is False
    assert not (ROOT / "config" / "corpus_disclosure.json").exists() and not hasattr(sb, "provisional_note")


def test_switch_reader_denies_by_default(tmp_path):
    p = tmp_path / "c.json"
    assert sb.subjects_enabled(p) is False  # missing
    p.write_text(json.dumps({"corpus_description": {"subject_areas": {}}}))
    assert sb.subjects_enabled(p) is False
    p.write_text(json.dumps({"corpus_description": {"subject_areas": {"enabled": True}}}))
    assert sb.subjects_enabled(p) is True


def test_phrases_are_frequent_multiword_without_numbers_or_titles():
    cfg = ci.load_cfg()
    out = ci.frequent_phrases(TITLES, cfg)
    assert out[:2] == ["influenza forecasting", "coral reef bleaching"] or set(PHR) & set(out)
    assert all(len(p.split()) >= 2 and not re.search(r"\d", p) for p in out)
    assert not any("quantum" in p for p in out)  # appears once: below min_docs
    assert not any(p in {t.casefold() for t in TITLES} for p in out)
    assert ci.frequent_phrases(["Two words"] * 5, cfg) == []  # a phrase equal to a whole title is never shown
    assert ci.frequent_phrases([], cfg) == []


def test_digest_shown_with_count_and_years_and_no_subject_numbers(off):
    st, cap = _capture()
    sb.render_corpus(st, {**DIGEST, **INTERIM, "topic_shares": {"basis_documents": 9, "main": [{"label": "x", "pct": 60}], "main_pct": 60}})
    text = _NORM(" ".join(cap))
    assert "314 sources, newest source 2026-11-07, publication years 2005 to 2026." in text
    assert "A collection on influenza and reefs." in text and "Description written by a language model" in text
    assert not NUMBERS.search(text.replace("Description written by a language model on 2026-11-07", "")), text
    assert "Frequent subjects" not in text and "Label flu" not in text


def test_fallback_shows_phrases_not_count_only_and_no_subject_list(off):
    st, cap = _capture()
    sb.render_corpus(st, {**FALLBACK, **INTERIM})
    text = _NORM(" ".join(cap))
    assert "Frequent subjects include: influenza forecasting; coral reef bleaching." in text
    assert "publication years 2005 to 2026" in text and text.count("Publication years") == 0  # no duplicate years line
    assert not NUMBERS.search(text) and "Epidemiology" not in text and "Label flu" not in text
    assert "not available" not in text


def test_no_individual_source_titles_on_page(off):
    st, cap = _capture()
    sb.render_corpus(st, {**FALLBACK, **INTERIM})
    text = _NORM(" ".join(cap)).casefold()
    assert not any(t.casefold() in text for t in TITLES)


def test_switch_on_restores_old_view(on):
    st, cap = _capture()
    sb.render_corpus(st, FALLBACK)
    text = _NORM(" ".join(cap))
    assert "Subject areas (provisional; share of 4 documents" in text and "Not yet matched to a listed subject area: 1 of 4" in text
    assert "provisional note" not in text.lower() and "Subject shares are provisional" not in text


def test_describer_off_has_no_subject_data_and_adds_phrases(monkeypatch):
    d = _describer(lambda: (True, ROWS))
    d.cfg = {**d.cfg, "subject_areas": {"enabled": False}}
    monkeypatch.setattr(ci, "read_interim", lambda eng, cfg: INTERIM)
    out = d.current(_no_llm)
    assert out["subject_areas"] is None and out["topic_shares"] is None
    assert out["frequent_subjects"] == INTERIM["frequent_subjects"] and out["published_years"] == ["2005", "2026"]
    from main import CorpusDescription
    assert CorpusDescription(**out).frequent_subjects == INTERIM["frequent_subjects"]


def test_describer_on_keeps_subject_areas(monkeypatch):
    monkeypatch.setitem(cd.SUBJECT_DEFAULTS, "enabled", True)
    out = _describer(lambda: (True, ROWS)).current(_no_llm)
    assert out["subject_areas"]["domains"] and "frequent_subjects" not in out


def test_page_via_apptest_shows_description_without_subject_numbers(monkeypatch, _offline):
    monkeypatch.setattr("vera.public_mode.get_corpus_summary", lambda *a, **k: {**FALLBACK, **INTERIM})
    monkeypatch.setattr("vera.public_mode.upstream_base_url", lambda: "http://x")
    monkeypatch.setenv("VERA_API_KEY", "k")
    at = AppTest.from_file(str(ROOT / "pages/corpus_description.py"), default_timeout=30).run()
    assert not at.exception
    text = _NORM(" ".join(e.value for kind in ("title", "markdown", "caption") for e in getattr(at.main, kind)))
    assert "Frequent subjects include:" in text and "publication years 2005 to 2026" in text
    assert not NUMBERS.search(text.replace("not yet show how many", "")), text
    assert "never names an individual source" in text


def test_api_default_is_off_when_key_missing():
    assert cd.subject_cfg({})["enabled"] is False and cd.subject_cfg({"subject_areas": {}})["enabled"] is False
    assert cd.subject_areas_for(None, {}, {}, registry_reader=lambda: (True, ROWS)) is None  # no engine needed: off
    assert cd.subject_cfg({"subject_areas": {"enabled": True}})["enabled"] is True


def test_interim_checked_summary_drops_shares_and_per_subject_counts(off):
    # OD-1109-HD3: the model-written description may state "label: N%" or "N sources" per subject; interim mode removes those sentences
    summ = ("A collection on influenza and reefs. Influenza: 60% of the sample. Coral reefs: 12 sources. "
            "About a third of papers concern ocean warming. It spans 2005 to 2026.")
    st, cap = _capture()
    sb.render_corpus(st, {**DIGEST, **INTERIM, "summary": summ})
    text = _NORM(" ".join(cap))
    assert "A collection on influenza and reefs." in text and "It spans 2005 to 2026." in text
    assert "60%" not in text and "12 sources" not in text and "of the sample" not in text
    assert "314 sources, newest source 2026-11-07, publication years 2005 to 2026." in text  # total count / years kept (interim block)


def test_interim_summary_only_shares_leaves_no_summary_text(off):
    assert sb._interim_summary("Flu: 60%. Reefs: 40 percent. 7 sources on coral.") == ""
    assert sb._interim_summary("Covers physics and biology.") == "Covers physics and biology."


def test_switch_on_keeps_summary_untouched(on):
    st, cap = _capture()
    sb.render_corpus(st, {**DIGEST, "summary": "A collection. Flu: 60%."})
    assert "Flu: 60%." in _NORM(" ".join(cap))


FORBIDDEN_LIVE = ("descending order of share", "Other / mixed", "Subject areas (", "automatic and approximate", "share of ",
                  "Main topics (", "topic without an approved label", "Not grouped under a topic", "Topics: ", "21%", "17%", "43%", "57%")


def test_live_shaped_data_shows_none_of_the_pct_b_strings(off):
    # OD-1109-HD3 acceptance: fixtures shaped like the live CORPUS-PCT-B page (366 sources, subject areas with topic shares, digest with shares)
    live = {**DIGEST, **INTERIM, "source_count": 366,
            "summary": ("A collection on AI coding assistants and epidemiology. AI coding assistants: 21%. "
                        "Main topics (57% of documents; share of the 150 sampled sources): topic without an approved label 17%. "
                        "Not grouped under a topic: 43%. It also covers reef science."),
            "coverage": "Not grouped under a topic: 43%.",
            "topic_shares": {"basis_documents": 150, "main": [{"label": "topic without an approved label", "pct": 17}], "main_pct": 57},
            "topics": ["topic without an approved label"],
            "subject_areas": {**cd_sa(), "basis_documents": 366}}
    st, cap = _capture()
    sb.render_corpus(st, live)
    text = _NORM(" ".join(cap))
    for bad in FORBIDDEN_LIVE:
        assert bad not in text, (bad, text)
    assert "366 sources" in text and "publication years 2005 to 2026" in text and "Description written by a language model on 2026-11-07" in text
    assert "A collection on AI coding assistants and epidemiology." in text and "It also covers reef science." in text
    help_text = " ".join(json.loads((ROOT / "config" / "ui_help_pages.json").read_text())["corpus_page"]["paragraphs"])
    for bad in ("descending order of share", "Other / mixed", "automatic and approximate", "share of the documents"):
        assert bad not in help_text, bad


def cd_sa():
    from tests.test_corpus_desc_1107 import SA
    return dict(SA)
