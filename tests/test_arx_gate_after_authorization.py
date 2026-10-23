"""ARX-GATE-V2 after-authorization tests (copy to work/arx-a2/tree/tests/test_arxiv_gate_v2.py ONLY after the owner has
applied, or approved at the permission prompt, licence_gate_arxiv_html.v2.diff). They exercise the REAL licence gate:
no monkeypatch of licence_gate.licence_decision / fulltext_loader.licence_decision / arxiv_fulltext._arxiv_url_rule.
OFFLINE: synthetic fixtures (tests/arxiv_synth.py), fake transports, stubbed answer model; no network, DB or LLM.
Run: work/venvs/vera-dar/bin/python -I scripts/vera_test_runner.py -- tests/test_arxiv_gate_v2.py tests/test_arxiv_evidence.py
Before authorization these tests are EXPECTED to fail (the gate still refuses all arXiv HTML)."""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from tests import accept_predicate as ap
from tests import acq_support as A
from tests import arxiv_synth as X
from tests.test_acq_discovered import Router, _direct, cfg_on, disc, run  # noqa: F401  (_direct: autouse)
from tests.test_acq_serve import _decide, h  # noqa: F401  (h is a fixture)
from tests.test_dar_inline_acquire import OFF_Q, rig  # noqa: F401  (rig is a fixture)
from vera import arxiv_fulltext, auto_acquire, fulldoc_db, licence_gate

HTML = "https://arxiv.org/html/2605.23989v1"
CC = "https://creativecommons.org/licenses/{}/4.0/"


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch):
    monkeypatch.setattr(arxiv_fulltext, "_sleep", lambda s: None)


def _ev(aid="2605.23989", ver="v1", lic_url=CC.format("by"), **over):
    e = arxiv_fulltext.evidence_block(aid, ver, lic_url)
    e.update(over)
    return e


def _cand(url=HTML, lic_id="cc by", lic_url=CC.format("by"), ev="default"):
    lic = {"content": {"id": lic_id, "url": lic_url, "source": "manifest declared"}}
    if ev == "default":
        ev = _ev(lic_url=lic_url)
    if ev is not None:
        lic["arxiv_evidence"] = ev
    return {"url": url, "licence": lic, "stored_text_kind": "full_text"}


def _d(c):
    return licence_gate.licence_decision(c)[0]


# ------------------------------------------------------------------------------------------------ gate level (real)
def test_cc_by_with_correct_evidence_is_admitted():
    assert _d(_cand()) == "allow"
    assert _d(_cand(url=HTML + "/")) == "allow"  # R3: optional trailing slash


@pytest.mark.parametrize("fam,lic_id", [("by-sa", "cc by-sa"), ("by-nc", "cc by-nc"), ("by-nc-sa", "cc by-nc-sa")])
def test_other_permitted_families_are_admitted(fam, lic_id):
    assert _d(_cand(lic_id=lic_id, lic_url=CC.format(fam))) == "allow"


def test_manifest_item_without_evidence_is_rejected():  # review (a)
    assert _d(_cand(ev=None)) == "reject"


def test_evidence_for_v1_with_html_v2_is_rejected():  # review (b)
    assert _d(_cand(url="https://arxiv.org/html/2605.23989v2")) == "reject"
    assert _d(_cand(url=HTML, ev=_ev(ver="v2"))) == "reject"


def test_evidence_id_mismatch_is_rejected():  # review (c)
    assert _d(_cand(ev=_ev(aid="2605.00001"))) == "reject"


@pytest.mark.parametrize("ev", [
    _ev(abs_url="https://arxiv.org/abs/2605.23989v2"),
    _ev(abs_url="https://arxiv.org/abs/2605.23989"),
    _ev(licence_url=CC.format("by-sa")),          # evidence licence differs from the content licence
    _ev(ver="1"), _ev(ver="v0"), _ev(aid="hep-th/9901001"), _ev(aid="2605.23989v1", ver=""),
    {"id": "2605.23989"}, "not a dict", [],
])
def test_malformed_or_mismatched_evidence_is_rejected(ev):
    assert _d(_cand(ev=ev)) == "reject"


@pytest.mark.parametrize("lic_id,lic_url,want", [
    ("cc by-nd", CC.format("by-nd"), "reject"),
    ("cc by-nc-nd", CC.format("by-nc-nd"), "reject"),
    ("arxiv nonexclusive", "http://arxiv.org/licenses/nonexclusive-distrib/1.0/", "reject"),
    ("cc0", "https://creativecommons.org/publicdomain/zero/1.0/", "reject"),
    (None, None, "reject"),
    ("cc by", "http://creativecommons.org/licenses/by/4.0/", "reject"),     # not https
    ("cc by", "https://creativecommons.org/licenses/by/4.0/legalcode", "reject"),
    ("cc by", "https://creativecommons.org/licenses/by/3.0/us/", "reject"),
])
def test_nd_default_cc0_missing_and_odd_licences_stay_refused(lic_id, lic_url, want):
    assert _d(_cand(lic_id=lic_id, lic_url=lic_url)) == want


@pytest.mark.parametrize("url", [
    "http://arxiv.org/html/2605.23989v1", "https://www.arxiv.org/html/2605.23989v1",
    "https://export.arxiv.org/html/2605.23989v1", "https://arxiv.org/html/2605.23989v1?x=1",
    "https://arxiv.org/html/2605.23989v1#s1", "https://arxiv.org/html/2605.23989",
    "https://arxiv.org/html/2605.23989v1/x", "https://arxiv.org/html/2605.23989v10",
    "https://arxiv.org/pdf/2605.23989v1", "https://arxiv.org/e-print/2605.23989v1",
])
def test_other_arxiv_urls_stay_refused_even_with_evidence(url):
    assert _d(_cand(url=url)) == "reject"


# F-1 (ARX-REAL-VERIFY): the pre-existing gate host check uses netloc, so these spellings skip the arXiv refusal.
# Fixed by the owner-applied F-1 v4 gate (work/arx-gate-proposal/v4); the former strict xfail marks are removed.


@pytest.mark.parametrize("url", [
    pytest.param("https://arxiv.org:443/html/2605.23989v1", id="port-443"),
    pytest.param("https://u@arxiv.org/html/2605.23989v1", id="userinfo"),
    pytest.param("https://u:p@arxiv.org/html/2605.23989v1", id="userinfo-password"),
    pytest.param("https://arxiv.org:8080/html/2605.23989v1", id="port-8080"),
    pytest.param("https://ArXiv.org/html/2605.23989v1", id="mixed-case-html"),
])
def test_f1_port_and_userinfo_html_refused_even_with_evidence(url):
    assert _d(_cand(url=url)) == "reject"


@pytest.mark.parametrize("url", [
    pytest.param("https://arxiv.org:443/pdf/2605.23989v1", id="port-pdf"),
    pytest.param("https://u@arxiv.org/e-print/2605.23989v1", id="userinfo-eprint"),
    pytest.param("https://arxiv.org./pdf/2605.23989v1", id="trailing-dot-pdf"),
    pytest.param("https://arxiv.org./html/2605.23989v1", id="trailing-dot-html"),
])
def test_f1_other_spellings_refused(url):
    assert _d(_cand(url=url)) == "reject"


@pytest.mark.parametrize("url", ["https://ARXIV.org/pdf/2605.23989v1", "https://ArXiv.Org/e-print/2605.23989v1"])
def test_mixed_case_host_pdf_and_eprint_refused(url):  # already true at the v2 gate; regression guard for v3 (mixed-case /html/ is an F-1 case above)
    assert _d(_cand(url=url)) == "reject"


def test_plain_https_arxiv_html_with_evidence_is_still_admitted():
    assert _d(_cand(url=HTML)) == "allow"


@pytest.mark.parametrize("content", ['{"arxiv_html_cc_families": "by"}', '{"nd_tokens": []}', "not json"])
def test_gate_fails_closed_on_malformed_family_config(monkeypatch, tmp_path, content):
    f = tmp_path / "rules.json"
    real = json.loads(licence_gate.RULES_PATH.read_text(encoding="utf-8"))
    try:
        data = json.loads(content)
        merged = {**{k: real[k] for k in ("nd_tokens", "no_educational_use_markers", "fee_only_markers")}, **data}
        f.write_text(json.dumps(merged), encoding="utf-8")
    except ValueError:
        f.write_text(content, encoding="utf-8")
    monkeypatch.setattr(licence_gate, "RULES_PATH", f)
    assert licence_gate._arxiv_html_cc_ok(_cand(), __import__("urllib.parse").parse.urlsplit(HTML)) is False


def test_shipped_family_lists_agree():  # R4: the gate list binds; the route list may not be wider
    gate = set(json.loads(licence_gate.RULES_PATH.read_text(encoding="utf-8"))["arxiv_html_cc_families"])
    assert gate == {"by", "by-sa", "by-nc", "by-nc-sa"}
    assert arxiv_fulltext.allowed_families(cfg_on()) == gate


# ------------------------------------------------------------------------------------------------ route + /ask (real)
def _go(web):
    return run(cfg_on(), [X.discovery_src()], web)


def test_2605_23989v1_admitted_through_the_real_gate():
    o, store = _go(Router({}, {}, X.routes()))
    assert o.status == "ok" and disc(o)["outcome"] == "admitted", o.public()
    prov = store.plans[0]["source"]["provenance"]
    assert prov["licence"]["arxiv_evidence"] == _ev(lic_url=CC.format("by"))
    assert prov["licence_decision"] == "allow" and prov["fulltext_url"] == HTML
    srcs, spans = A.rows_from_plans(store.plans)
    c = fulldoc_db.load_acquired_rows([{**s, "provenance": json.loads(s["provenance"])} for s in srcs], spans,
                                      auto_acquire._fd_cfg(), cfg_on())
    assert c.status == "ok" and c.acquired
    bad = copy.deepcopy(store.plans[0])
    bad["source"]["provenance"]["licence"]["arxiv_evidence"]["version"] = "v2"
    srcs, spans = A.rows_from_plans([bad])
    c = fulldoc_db.load_acquired_rows([{**s, "provenance": json.loads(s["provenance"])} for s in srcs], spans,
                                      auto_acquire._fd_cfg(), cfg_on())
    assert not c.acquired


@pytest.mark.parametrize("lic,reason", [(X.CC_BY_ND, "R-LIC-REJECT"), (X.ARXIV_DEFAULT, "R-ACQ-ARXIV_LICENCE_HOLD"),
                                        (None, "R-ACQ-LICENCE_ABSENT")])
def test_route_refusals_unchanged_through_the_real_gate(lic, reason):
    o, store = _go(Router({}, {}, X.routes(licence=lic)))
    assert disc(o)["reason"] == reason and store.plans == []


def test_2605_23989v1_answers_the_same_post_ask_request(rig, monkeypatch):
    import main
    web = Router({}, {}, X.routes())
    rig.setup(web=web, cands=[])
    rig.h.answer_text = "The transmission of the pathogen was compared with incidence during the outbreak [1]."
    main_mod = rig.h.install()
    monkeypatch.setattr(main_mod, "retrieve_all", lambda q, cfg, **kw: ([X.discovery_src()], []))
    monkeypatch.setattr(main_mod, "_acquire", lambda q, t: None)
    from vera import inline_acquire
    monkeypatch.setattr(inline_acquire, "refresh_after_response", lambda: None)
    _decide(monkeypatch, False, "layer2_llm", "llm_out:x", "off_topic")
    r = TestClient(main.app).post("/ask", json={"question": OFF_Q}, headers={"X-API-Key": "k" * 20})
    body = r.json()
    assert r.status_code == 200 and body["result_kind"] == "answer", body.get("contract")
    assert [e["outcome"] for e in body["contract"]["inline_acquisition"]["discovered"]] == ["admitted"]
    assert X.HTML_URL in [u for u, _h, _k in web.calls]
    assert body["admitted_evidence"] and all(a["document_id"].startswith("acq-") for a in body["admitted_evidence"])
    f = ap.grounded_failures(r.status_code, body)
    assert not [x for x in f if "build identity" not in x], f
    assert len(rig.h.answer_calls) == 1 and len(rig.h.check_calls) == 1
