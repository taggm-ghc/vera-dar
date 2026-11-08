"""CORPUS-PCT-B-1108 (owner 2026-11-08): the Corpus Description shows each subject area's share again (stored-data classifier),
in descending order, with the remainder as ONE last line "Other / mixed: N%". No "Not yet matched" headline or sentence."""
import re

import pytest

from tests.test_corpus_desc_1107 import DIGEST, ENTRIES, FALLBACK, ROWS, _capture, _describer, _no_llm, _NORM  # noqa: F401
from vera import corpus_description as cd, sources_sidebar as sb

ENT = [{"id": f"t{i}", "label": f"Topic {i}", "field": f"area {i}", "parent_topic_key": None} for i in range(6)]


def _sa(counts, unmatched, **scfg):
    primary, k = {}, 0
    for i, c in enumerate(counts):
        for _ in range(c):
            primary[k] = f"t{i}"; k += 1
    for _ in range(unmatched):
        primary[k] = None; k += 1
    return cd.subject_areas(ENT, primary, scfg)


def _lines(sa):
    return sb._subject_lines(sa)


def _pcts(lines):
    return [int(m.group(1)) for ln in lines[1:] for m in [re.match(r"[^:]+: (\d+)%", ln)] if m]


@pytest.mark.parametrize("counts,unmatched", [([5, 3, 1], 1), ([7, 7, 7], 0), ([1, 1, 1, 1, 1, 1], 4), ([9], 31), ([], 5)])
def test_shares_plus_other_mixed_sum_to_about_100(counts, unmatched):
    lines = _lines(_sa(counts, unmatched, display_domains=3))
    assert lines and lines[-1].startswith("Other / mixed: ") or sum(_pcts(lines)) == 100
    assert 98 <= sum(_pcts(lines)) <= 100 + len(counts)  # rounding of each share only


def test_areas_are_in_descending_order_even_if_the_input_is_not():
    sa = _sa([2, 6, 4], 0)
    sa["domains"] = list(reversed(sorted(sa["domains"], key=lambda d: -d["share"])))  # worst case: ascending
    pcts = _pcts(_lines(sa)[:-1] if _lines(sa)[-1].startswith("Other") else _lines(sa))
    assert pcts == sorted(pcts, reverse=True) and len(pcts) == 3


def test_other_mixed_is_the_last_line_and_the_only_remainder_line():
    lines = _lines(_sa([5, 3], 2))
    assert lines[-1] == "Other / mixed: 20%" and sum(ln.startswith("Other") for ln in lines) == 1
    assert _lines(_sa([], 5))[-1] == "Other / mixed: 100%"
    assert not any(ln.startswith("Other") for ln in _lines(_sa([4, 4], 0)))  # nothing left over: no line


def test_smaller_areas_beyond_the_display_limit_fold_into_other_mixed():
    lines = _lines(_sa([4, 3, 2, 1], 0, display_domains=2))
    assert lines[-1] == "Other / mixed: 30%" and sum(1 for ln in lines if ln.startswith("Area")) == 2


def test_header_says_automatic_and_approximate():
    assert "automatic and approximate" in _lines(_sa([3], 1))[0]


@pytest.mark.parametrize("data", [FALLBACK, DIGEST, {**FALLBACK, "subject_areas": _sa([], 5)}])
def test_no_not_yet_matched_text_anywhere_in_rendered_output(data):
    st, cap = _capture()
    sb.render_corpus(st, data)
    text = _NORM(" ".join(cap)).casefold()
    assert "not yet matched" not in text and "no document matched" not in text and "of 4 documents (" not in text
    assert "other / mixed" in text


def test_llm_prose_and_provenance_line_unchanged():
    st, cap = _capture()
    sb.render_corpus(st, DIGEST)
    text = _NORM(" ".join(cap))
    assert "A collection on influenza and reefs." in text and "Description written by a language model on 2026-11-07;" in text
    assert text.index("A collection on influenza") < text.index("Subject areas") < text.index("Description written by")


def test_a_failure_never_breaks_the_page(monkeypatch):
    d = _describer(lambda: (True, ROWS))

    def boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(cd, "subject_areas_for", boom)
    out = d.current(_no_llm)
    assert out.get("subject_areas") is None and out["source_count"]  # digest still served
    st, cap = _capture()
    sb.render_corpus(st, {**FALLBACK, "subject_areas": None})
    assert sb.NO_SUBJECTS in cap
    for bad in ("x", 7, {"basis_documents": "n"}, {"basis_documents": 4, "domains": [{"label": "a"}]}):
        assert sb._subject_lines(bad) == []  # malformed input: empty, never an exception
