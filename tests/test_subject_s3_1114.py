"""SUBJECT-BUILD-1114 WP-S3: per-passage association line in render_sources (T18-T19 of work/SUBJECT-1114/PLAN.md).
Fakes only: a recording stand-in for st, no database, no network, no model."""
import pytest

from vera import sources_sidebar as sb

CAVEAT = "These are assigned automatically by a program that has not been validated, and are approximate."


class _St:
    def __init__(self):
        self.cap = []
        self.md = []

    def markdown(self, text, *a, **k):
        self.md.append(text)

    def caption(self, text, *a, **k):
        self.cap.append(text)


def _render(sources, **kw):
    st = _St()
    sb.render_sources(st, sources, [], fulldoc=kw.pop("fulldoc", False), **kw)
    return st.cap


def _src(basis, description=None, **extra):
    d = {"n": 1, "title": "Doc", "url": "https://x.org/a", "assoc_basis": basis}
    if description is not None:
        d["assoc_description"] = description
    d.update(extra)
    return d


def _line(basis, description=None):
    lines = [c for c in _render([_src(basis, description)]) if CAVEAT in c]
    assert len(lines) == 1, lines
    return lines[0]


# ---- T18: wording per basis ----

def test_subtopic_names_each_level_and_omits_absent_ones():
    full = _line("subtopic", {"subtopic": "solar power (terms: solar, grid, panel)", "topic": "energy",
                              "domain": "science"})
    assert full.startswith("Sub-topic: " + sb.plain("solar power (terms: solar, grid, panel)") + " · topic: energy"
                           " · domain: science. " + CAVEAT)
    no_domain = _line("subtopic", {"subtopic": "wind", "topic": "energy"})
    assert "Sub-topic: wind · topic: energy. " in no_domain and "domain:" not in no_domain


def test_topic_basis_has_no_sub_topic_level():
    line = _line("topic", {"subtopic": "should not show", "topic": "energy", "domain": "science"})
    assert line.startswith("Topic: energy · domain: science. ")
    assert "Sub-topic" not in line and "should not show" not in line


def test_domain_basis_is_worded_as_domain_only():
    line = _line("domain", {"domain": "science", "topic": None, "subtopic": None})
    assert line.startswith("Domain: science (no topic or sub-topic record). " + CAVEAT)


def test_domain_only_record_never_appears_under_topic():
    for basis in ("domain", "topic", "subtopic"):
        line = _line(basis, {"domain": "science"})
        assert line.startswith("Domain: science (no topic or sub-topic record)."), (basis, line)
        assert "Topic" not in line and "topic: science" not in line


def test_document_vote_with_topic_is_worded_as_the_documents():
    line = _line("document_vote", {"topic": "energy", "domain": "science", "subtopic": None})
    assert line.startswith("Document's topic: energy · domain: science. " + CAVEAT)
    assert "Sub-topic" not in line


def test_document_vote_with_only_domain_is_worded_as_the_documents():
    line = _line("document_vote", {"domain": "science", "topic": None, "subtopic": None})
    assert line.startswith("Document's domain: science. " + CAVEAT)
    assert "Topic" not in line and "topic:" not in line


def test_none_basis_says_no_record_yet():
    line = _line("none", {"domain": None, "topic": None, "subtopic": None})
    assert line == "No topic or sub-topic record yet. " + CAVEAT


def test_caveat_present_on_every_basis_line():
    for basis, d in (("subtopic", {"subtopic": "a"}), ("topic", {"topic": "b"}), ("domain", {"domain": "c"}),
                     ("document_vote", {"topic": "d"}), ("none", None)):
        assert CAVEAT in _line(basis, d)


def test_markdown_and_html_in_labels_are_escaped():
    line = _line("subtopic", {"subtopic": "<b>bold</b> [link](javascript:x)", "topic": "*t*"})
    assert "<b>" not in line and "</b>" not in line and "](javascript" not in line.replace("\\]", "")
    assert "\\<b\\>bold\\</b\\>" in line
    assert "\\[link\\]" in line and "\\*t\\*" in line


@pytest.mark.parametrize("bad", ["", "   ", 7, ["x"]])
def test_non_string_or_blank_labels_are_dropped(bad):
    line = _line("topic", {"topic": bad, "domain": "science"})
    assert line.startswith("Domain: science (no topic or sub-topic record).")


def test_unknown_basis_gives_no_line():
    assert not [c for c in _render([_src("guessed", {"topic": "x"})]) if CAVEAT in c]


# ---- T19: per-passage lines, absence, existing behaviour ----

def test_two_passages_of_one_document_each_get_their_own_line():
    sources = [
        _src("subtopic", {"subtopic": "solar power", "topic": "energy"}, n=1, span="12:3",
             canonical_source_id="c1"),
        _src("subtopic", {"subtopic": "wind turbines", "topic": "energy"}, n=2, span="40:9",
             canonical_source_id="c1"),
    ]
    caps = _render(sources, fulldoc=True, cited=[1, 2])
    lines = [c for c in caps if CAVEAT in c]
    assert len(lines) == 2, caps
    assert "Passage 12:3: Sub-topic: solar power · topic: energy." in lines[0]
    assert "Passage 40:9: Sub-topic: wind turbines · topic: energy." in lines[1]


def test_no_line_when_assoc_basis_absent():
    caps = _render([{"n": 1, "title": "Doc", "url": "https://x.org/a", "span": "1:2"}])
    assert not [c for c in caps if CAVEAT in c or "record" in c]


def test_no_line_when_assoc_basis_none_value():
    caps = _render([_src(None, {"topic": "x"})])
    assert not [c for c in caps if CAVEAT in c]


def test_existing_sidebar_tests_module_still_imports():
    # The sidebar functions are untouched by WP-S3; their own tests run in tests/test_item78_sources_sidebar.py.
    assert callable(sb.render_sources) and callable(sb.render_sources_sidebar)
