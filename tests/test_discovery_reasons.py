"""DAR-PIPE-1022-UIFIX: each discovered candidate's outcome is shown in plain language in the MAIN panel; no internal
code reaches the page; the sidebar shows no candidate titles. Offline (mock st + AppTest, no server)."""
import json
import re
from types import SimpleNamespace
from unittest import mock

import pytest
from streamlit.testing.v1 import AppTest

from tests.test_discovery_panel import ROOT, _app, render, texts  # noqa: F401
from vera import discovery_panel as dp

CODES = json.loads((ROOT / "config" / "auto_acquire.json").read_text())["reason_codes"]
INTERNAL = re.compile(r"R-[A-Z]+-|ACQ-|R-LIC|gate_a|gate_b|link_only|not_evaluated|not_reached|already_stored|not_used", re.I)


def data_for(entries):
    cands = [{"title": f"Secret Title {i}", "url": f"https://x.org/{i}", "acquisition": "not_acquired"}
             for i in range(len(entries))]
    disc = [dict(e, url=f"https://x.org/{i}", service="s", record_id=str(i)) for i, e in enumerate(entries)]
    return {"result_kind": "reading_list", "answer": "x", "candidates": cands,
            "contract": {"inline_acquisition": {"status": "no_evidence", "discovered": disc}}}


# Job-level codes (never on a per-candidate entry); any other configured R-ACQ code must have its own text.
JOB_LEVEL = {"disabled", "queue_full", "rate", "daily_cap", "duplicate_question", "expired", "search_failed",
             "intent_denied"}


def test_every_configured_failure_code_has_plain_text():
    for name, code in CODES.items():
        if code.startswith("R-ACQ-") and name not in JOB_LEVEL:
            line = dp._reason_line({"outcome": "rejected", "reason": code})
            assert line and line != dp.REASON_GENERIC and not INTERNAL.search(line), (name, line)


LOADER = json.loads((ROOT / "config" / "fulltext_ingest.json").read_text())["reason_codes"]


def test_loader_level_refusal_codes_emitted_by_acquisition_have_plain_text():
    """Codes fulltext_loader / auto_acquire.licence_elements_decision raise on a fetched document (grep: licence_reject,
    licence_hold, licence_mismatch, abstract_only) must not fall back to the generic text."""
    for name in ("licence_reject", "licence_hold", "licence_mismatch", "abstract_only"):
        line = dp._reason_line({"outcome": "rejected", "reason": LOADER[name]})
        assert line and line != dp.REASON_GENERIC and not INTERNAL.search(line), (name, line)


@pytest.mark.parametrize("entry,want", [
    ({"outcome": "rejected", "reason": CODES["no_licence"]}, "Full text not openly licensed."),
    ({"outcome": "rejected", "reason": CODES["unreachable"]}, "Source unreachable."),
    ({"outcome": "rejected", "reason": CODES["host"]}, "Blocked by network safety check."),
    ({"outcome": "rejected", "reason": CODES["irrelevant"]}, "Not relevant enough to your question."),
    ({"outcome": "link_only", "reason": CODES["no_doi"]}, dp.REASON_TEXT[CODES["no_doi"]]),
    ({"outcome": "rejected", "reason": "R-ZZZ-NEW"}, dp.REASON_GENERIC),
    ({"outcome": "not_evaluated"}, "Not checked in time."),
    ({"outcome": "deferred", "reason": CODES["defer_cap"]}, dp.REASON_TEXT[CODES["defer_cap"]]),
])
def test_reason_renders_plain_and_hides_codes(entry, want):
    st = render(data_for([entry]))
    t = " | ".join(texts(st))
    assert want in t
    assert not INTERNAL.search(t)
    assert "R-ZZZ" not in t


@pytest.mark.parametrize("outcome", ["admitted", "already_stored", "not_used"])
def test_obtained_but_no_answer_is_truthful(outcome):
    st = render(data_for([{"outcome": outcome}]))
    t = " | ".join(texts(st))
    assert dp.OBTAINED_TEXT in t
    assert "not acquired" not in t.lower().replace("has not acquired", "") and "read for this answer" not in t
    assert "has not acquired or read the documents in this list" not in t  # M-U2: contradicted by the obtained row


def _disclosure(entries):
    st = render(data_for(entries))
    return [str(c.args[0]) for c in st.warning.call_args_list]


def test_page_disclosure_none_some_all_obtained():
    rej = {"outcome": "rejected", "reason": CODES["no_licence"]}
    got = {"outcome": "admitted"}
    none = _disclosure([rej, rej])
    assert none == [dp.answer_contract.text("not_read_line")]
    some = _disclosure([got, rej])
    assert some == [dp.NOT_READ_OTHERS] and "other documents" in some[0]
    all_ = _disclosure([got, got])
    assert all_ == [dp.OBTAINED_ALL] and "has not acquired or read" not in all_[0]
    # per-row text stays consistent with the banner
    t = " | ".join(texts(render(data_for([got, rej]))))
    assert t.count(dp.OBTAINED_TEXT) == 1 and "index record or abstract" in t


def test_lic_reject_text_is_accurate_for_nd_and_nonreuse_licences():
    line = dp._reason_line({"outcome": "rejected", "reason": "R-LIC-REJECT"})
    assert line == "The licence does not permit reuse of the full text." and "openly licensed" not in line


def test_each_candidate_gets_its_own_reason():
    st = render(data_for([{"outcome": "rejected", "reason": CODES["no_licence"]},
                          {"outcome": "rejected", "reason": CODES["unreachable"]}]))
    caps = [str(c.args[0]) for c in st.caption.call_args_list]
    assert any("not openly licensed" in c for c in caps) and any("unreachable" in c for c in caps)


def test_no_inline_block_changes_nothing():
    d = data_for([{"outcome": "rejected", "reason": CODES["no_licence"]}])
    d["contract"] = {}
    t = " | ".join(texts(render(d)))
    assert "not openly licensed" not in t


def test_app_main_has_reasons_sidebar_no_titles_no_codes(monkeypatch):
    at = _app(monkeypatch, data_for([{"outcome": "rejected", "reason": CODES["no_licence"]},
                                     {"outcome": "rejected", "reason": CODES["unreachable"]}]))
    assert not at.exception
    main = " | ".join([m.value for m in at.markdown] + [c.value for c in at.caption])
    assert "Full text not openly licensed." in main and "Source unreachable." in main
    assert not INTERNAL.search(main)
    side = " | ".join([m.value for m in at.sidebar.markdown] + [c.value for c in at.sidebar.caption]
                      + [m.value for m in at.sidebar.info] + [m.value for m in at.sidebar.warning])
    assert "Secret Title" not in side and "x.org" not in side


import itertools  # noqa: E402

_ROW = {"O": {"outcome": "admitted"}, "N": {"outcome": "rejected", "reason": "R-LIC-HOLD"},
        "A": {"outcome": "already_stored"}}  # admitted/already_stored in a reading list are rendered as "obtained"


@pytest.mark.parametrize("n_obt,n_stored,n_none", [c for c in itertools.product(range(3), repeat=3) if sum(c)])
def test_banner_is_truthful_for_every_row_combination(n_obt, n_stored, n_none):
    """Rows: obtained (admitted/already_stored; a reading list never uses a document), stored (in collection or
    full document acquired), not acquired. No banner statement may be false for any present row."""
    entries = [_ROW["O"]] * n_obt + [_ROW["N"]] * n_none
    d = data_for(entries + [{"outcome": "not_used"}] * 0)
    for i in range(n_stored):  # stored rows: acquisition state set on the candidate, no discovered entry
        d["candidates"].append({"title": f"Stored {i}", "url": f"https://s.org/{i}",
                                "acquisition": "in_collection" if i % 2 else "full_document_acquired"})
    st = render(d)
    banner = " ".join(str(c.args[0]) for c in st.warning.call_args_list)
    low = banner.lower()
    if n_none == 0:
        assert "has not acquired or read" not in low, banner      # nothing listed is unacquired
    else:
        assert "not acquired or read" in low, banner               # the unacquired rows are disclosed
    if n_obt or n_stored:
        assert "has not acquired or read the documents in this list." not in low, banner  # unqualified, false for held rows
        assert "the other documents" not in low or n_stored == 0, banner  # "other" would exclude stored rows wrongly
    if n_obt:
        assert "were not used in this answer" in low or "did not use" in low or "did not inspect" in low, banner
    if n_obt and n_stored:
        assert "obtained or stored" in low or "obtained the full text" in low, banner
    assert "evidence-grounded answer" in low
    assert not INTERNAL.search(banner)
