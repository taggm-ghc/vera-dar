"""GAPTESTS-1023 G6 + owner scenario (live observation on 7da574a): corpus miss, 9 stored documents inspected, 0 passages
admitted, 4 discovered candidates none acquired -> abstain; UI gives a per-candidate reason for each; no PMCID/abstract leak.
Real: main.py /ask, grounded_ask.retrieve_all, ask_service, inline_acquire, acq_discovered, discovery_panel (+ Streamlit AppTest).
Fake: scope classifier, Europe PMC/JATS, store, answer model, claim checker, provider search."""
import json
import re
from unittest import mock

import requests

from tests import acq_support as A
from tests import fulltext_synth
from tests._g23_support import SHIPPED_TEXT, http_ask, use_file
from tests._pipeline_support import ABSTRACT_SENTINEL, Q_EPI, disc_entries, discovered, h, reason_code, rig  # noqa: F401
from tests.test_acq_discovered import Router
from tests.test_discovery_panel import _app
from vera import discovery_panel as dp

DOIS = {"nd": "10.9999/own.nd", "nooa": "10.9999/own.nooa", "unr": "10.9999/own.unreach", "fail": "10.9999/own.irrel"}
PMC_ND, PMC_UNR, PMC_IR = "PMC7100001", "PMC7100003", "PMC7100004"
WANT_TEXT = {"nd": "The licence does not permit reuse of the full text.", "nooa": "Full text is not openly available.",
             "unr": "Source unreachable.", "fail": "Not relevant enough to your question."}   # literals, not read from the panel
TITLES = {"nd": "Owner Cand One ND", "nooa": "Owner Cand Two NoOA", "unr": "Owner Cand Three Unreachable",
          "fail": "Owner Cand Four Irrelevant"}
LISTED = ("nd", "nooa", "unr")  # DDFIX-1023/INTEG item 6: "fail" (R-ACQ-IRRELEVANT) is no longer listed
NINE = tuple((f"syn_c{i}", "clim" if i % 2 else "repro", f"10.9999/syn.c{i}") for i in range(9))


def _web():
    rec, jats = {}, {}
    rec[DOIS["nd"]] = A.epmc_record(DOIS["nd"], pmcid=PMC_ND, licence="cc by-nd")
    jats[PMC_ND] = fulltext_synth.jats("own_nd", "epi", DOIS["nd"])
    rec[DOIS["unr"]] = A.epmc_record(DOIS["unr"], pmcid=PMC_UNR)
    jats[PMC_UNR] = fulltext_synth.jats("own_unr", "epi", DOIS["unr"])
    rec[DOIS["fail"]] = A.epmc_record(DOIS["fail"], pmcid=PMC_IR)          # open, licensed, but off-topic body
    jats[PMC_IR] = fulltext_synth.jats("own_ir", "clim", DOIS["fail"])
    return Router(rec, jats, over={f"/{PMC_UNR}/fullTextXML": requests.ConnectionError("fake: unreachable")})


PMCS = {"nd": PMC_ND, "nooa": "PMC7100002", "unr": PMC_UNR, "fail": PMC_IR}


def _srcs():
    """Provider records carry an abstract (sentinel) and a PMCID, as the live OpenAlex records do; neither may reach the client."""
    return [discovered(doi=DOIS[k], wid=f"https://openalex.org/W9{i}", title=TITLES[k], pmcid=PMCS[k])
            for i, k in enumerate(("nd", "nooa", "unr", "fail"))]


def _body(rig, monkeypatch, tmp_path):
    use_file(rig, monkeypatch, tmp_path, text=SHIPPED_TEXT)     # real shipped config file
    rig.h.bundle(docs=NINE)                                     # the base corpus now holds 9 stored full documents, none on epi
    rig.setup(web=_web(), cands=[])
    resp, _s = http_ask(rig, monkeypatch, _srcs(), offtopic=False)   # in-scope corpus miss
    assert resp.status_code == 200, resp.text
    return resp.json()


def _rows(st):
    rows = []
    for name, args, _kw in st.method_calls:
        if name == "markdown" and re.match(r"^\d+\. ", str(args[0])):
            rows.append((str(args[0]), []))
        elif name == "caption" and rows:
            rows[-1][1].append(str(args[0]))
    return rows


def test_owner_scenario_abstains_with_nine_inspected_zero_admitted_four_not_acquired(rig, monkeypatch, tmp_path):
    body = _body(rig, monkeypatch, tmp_path)
    assert body["result_kind"] != "answer" and not body.get("admitted_evidence")
    assert not body.get("citations") and not (body.get("claim_check") or {}).get("claims")
    exam = body["examined_full_documents"]
    assert len(exam) == 9 and sum(d["admitted_spans"] for d in exam) == 0           # 9 stored documents inspected, 0 passages admitted
    assert rig.store.plans == [] and not rig.h.answer_calls                      # nothing stored, the writer never ran
    # DDFIX-1023/INTEG item 6 (owner 2026-10-23): the R-ACQ-IRRELEVANT record is not listed while relevant ones remain
    assert len(body["candidates"]) == 3 and all(c["acquisition"] == "not_acquired" for c in body["candidates"])
    assert TITLES["fail"] not in json.dumps(body["candidates"])
    disc = disc_entries(mock.Mock(contract=body["contract"]))
    assert len(disc) == 4 and all(e["outcome"] in ("rejected", "deferred") and e.get("reason") for e in disc), disc


def test_owner_scenario_pmcid_public_and_no_abstract_leak_in_the_response(rig, monkeypatch, tmp_path):
    """PMCID-PUBLIC-1023 (owner 2026-10-23, inverts the former G23 rule): a known PMCID is a public identifier and is
    shown with the record's other ids; abstract text is still never in the response."""
    body = _body(rig, monkeypatch, tmp_path)
    dump = json.dumps(body)
    assert ABSTRACT_SENTINEL not in dump
    known = [c for c in (body.get("research_outcome") or {}).get("candidates") or [] if (c.get("ids") or {}).get("pmcid")]
    assert {c["ids"]["pmcid"] for c in known} <= set(PMCS.values())
    for pmc in re.findall(r"PMC\d{5,}", dump):  # any PMC id shown is a real, known id of a listed record
        assert pmc in PMCS.values(), pmc


def test_owner_scenario_ui_each_of_the_four_has_its_own_reason_not_banner_only(rig, monkeypatch, tmp_path):
    body = _body(rig, monkeypatch, tmp_path)
    by_url = {e["url"]: e for e in body["contract"]["inline_acquisition"]["discovered"]}
    st = mock.MagicMock()
    dp.render_discovery(st, body)
    rows = _rows(st)
    assert len(rows) == 3  # DDFIX-1023/INTEG item 6: the irrelevant record is not listed
    assert not [r for r in rows if TITLES["fail"] in r[0]]
    reasons = {}
    for k in LISTED:
        (row,) = [r for r in rows if TITLES[k] in r[0]]
        want = WANT_TEXT[k]
        assert dp._reason_line(by_url[f"https://doi.org/{DOIS[k]}"]) == want      # panel maps this entry to the literal text
        assert want != dp.REASON_GENERIC
        assert any(want in c for c in row[1]), (k, want, row)           # in THIS candidate's own caption
        reasons[k] = want
    assert len(set(reasons.values())) == 3                                   # distinct causes are distinguishable
    text = json.dumps(st.method_calls)
    assert not re.search(r"R-ACQ-|R-LIC-|PMC\d{5,}|%s" % ABSTRACT_SENTINEL, text)


def test_owner_scenario_ui_apptest_renders_reasons_and_leaks_nothing(rig, monkeypatch, tmp_path):
    body = _body(rig, monkeypatch, tmp_path)
    at = _app(monkeypatch, body)
    assert not at.exception
    main_t = " | ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info] + [w.value for w in at.warning])
    for k in LISTED:
        assert TITLES[k] in main_t
        assert WANT_TEXT[k] in main_t, k
    assert TITLES["fail"] not in main_t  # DDFIX-1023/INTEG item 6
    assert not re.search(r"R-ACQ-|R-LIC-|PMC\d{5,}|%s" % ABSTRACT_SENTINEL, main_t)
    assert main_t.count(dp.REASON_GENERIC) == 0
    side = " | ".join([m.value for m in at.sidebar.markdown] + [c.value for c in at.sidebar.caption])
    assert "Owner Cand" not in side

