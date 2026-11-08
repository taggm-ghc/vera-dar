"""CORPUS-DISCLOSE-1108: interim, config-driven note that subject shares are provisional."""
import json
import re

from vera import sources_sidebar as ss

SA = {"basis_documents": 40, "domains": [{"label": "physics", "share": 0.5, "topics": []}], "unmatched_count": 20}
DATA = {"summary_source": "deterministic_fallback", "source_count": 40, "subject_areas": SA}


class _St:
    def __init__(self):
        self.out = []

    def caption(self, t):
        self.out.append(t)


def _cfg(tmp_path, monkeypatch, enabled=True, text="Subject shares are provisional and may change."):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"provisional_note": {"enabled": enabled, "text": text}}))
    monkeypatch.setattr(ss, "provisional_note", lambda path=p, _f=ss.provisional_note: _f(path))


def test_shipped_config_enabled_and_numberless():
    n = ss.provisional_note()
    assert n and "provisional" in n and not re.search(r"\d", n)


def test_renders_when_enabled(tmp_path, monkeypatch):
    _cfg(tmp_path, monkeypatch)
    st = _St()
    ss.render_corpus(st, DATA)
    assert sum("provisional and may change" in t for t in st.out) == 1


def test_absent_when_disabled(tmp_path, monkeypatch):
    _cfg(tmp_path, monkeypatch, enabled=False)
    st = _St()
    ss.render_corpus(st, DATA)
    assert not any("provisional and may change" in t for t in st.out)


def test_never_shows_a_number(tmp_path, monkeypatch):
    _cfg(tmp_path, monkeypatch, text="Shares from 40 documents are provisional.")
    assert ss.provisional_note() is None


def test_no_note_without_subject_figures(tmp_path, monkeypatch):
    _cfg(tmp_path, monkeypatch)
    st = _St()
    ss.render_corpus(st, {**DATA, "subject_areas": None})
    assert not any("provisional and may change" in t for t in st.out)


def test_checked_branch_with_shares_only(tmp_path, monkeypatch):
    _cfg(tmp_path, monkeypatch)
    shares = {"basis_documents": 10, "main_pct": 100, "main": [{"label": "Physics", "pct": 100}]}
    d = {"summary_source": ss.CHECKED, "source_count": 10, "topic_shares": shares, "summary": "x"}
    st = _St()
    ss.render_corpus(st, d)
    assert sum("provisional and may change" in t for t in st.out) <= 1
