"""R8-HEADLINE-1106: non-answer headline separates inspected documents from documents that supplied passages."""
from types import SimpleNamespace as N

from vera import ask_service as a, discovery_panel as dp, ui_evidence as u
from vera import fulldoc_corpus


def _ev(n, ids):
    return N(examined_full_documents=tuple(range(n)), admitted=[{"document_id": i} for i in ids], corpus_source="db")


def test_server_text():
    t = fulldoc_corpus.load_config()["answer"]["unsupported_text"]
    assert a._with_count(t, _ev(30, [1, 1, 2]), selected=True) == (
        "VERA-DAR checked 30 full documents; passages from 2 looked related, but they did not support an answer, so it has not answered.")


def test_ui_text_and_fetched():
    off = dict(fulldoc_corpus.load_config()["answer"], count_matching_documents=False)  # NOANSWER-RC-1107: legacy
    assert u.fallback_text("model_insufficient", cfg=off, n_inspected=30, n_selected=2).startswith(
        "VERA-DAR checked 30 full documents; passages from 2 looked related")
    t = u.fallback_text("model_insufficient", cfg=off, n_inspected=29, n_selected=2, n_fetched=1)
    assert "checked 29 full documents from VERA-DAR's collection and 1 fetched for this question;" in t
    ev = N(corpus_source="bundle", examined_full_documents=tuple(range(29)))
    ia = {"documents": [{"persisted": False, "reason": "below_persist_min"}]}
    s = a._with_fetched(a._with_count(fulldoc_corpus.load_config()["answer"]["unsupported_text"], ev, selected=True), ev, ia)
    assert "checked 29 full documents from VERA-DAR's collection and 1 fetched for this question;" in s


class _St:
    def __init__(self): self.out = []
    def markdown(self, t): self.out.append(t)
    caption = info = warning = markdown


def _data(admitted):
    cand = lambda u_: {"title": "T", "url": u_, "acquisition": "not_acquired", "identifier": None}
    return {"result_kind": "reading_list", "candidates": [cand("https://a.org/1"), cand("https://b.org/2")],
            "admitted_evidence": admitted,
            "contract": {"inline_acquisition": {"discovered": [{"url": "https://a.org/1", "outcome": "admitted"},
                                                               {"url": "https://b.org/2", "outcome": "admitted"}]}}}


def test_panel_considered_vs_unused():
    st = _St()
    dp.render_discovery(st, _data([{"url": "https://a.org/1/"}]))
    text = "\n".join(map(str, st.out))
    assert dp.OBTAINED_CONSIDERED_TEXT in text and text.count(dp.OBTAINED_TEXT) == 1


def test_panel_no_data_keeps_text():
    st = _St()
    dp.render_discovery(st, _data([]))
    text = "\n".join(map(str, st.out))
    assert dp.OBTAINED_CONSIDERED_TEXT not in text and text.count(dp.OBTAINED_TEXT) == 2
