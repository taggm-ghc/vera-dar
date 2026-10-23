"""ARX-GATE-V2 route side (ARX-SECREV-1023 R1 route part, R2, R4). OFFLINE: synthetic fixtures, fake transports.

R2: the abs page's FINAL URL must be exactly /abs/<id><ver> as requested, else R-ACQ-IDENTITY_MISMATCH with a precise
attempt reason and no content fetch. R1 (route part): the version-bound licence evidence block is built only by
vera.arxiv_fulltext and handed to the loader as an argument; item / conversion record / manifest / discovery data never
produce it, and the read path re-derives it only for rows the arXiv route wrote. R4: once the gate's family list exists
the route admits only families in both lists. The licence gate itself is NOT changed here (it still refuses arXiv HTML)."""
import copy
import json

import pytest

from tests import acq_support as A
from tests import arxiv_synth as X
from tests.test_acq_discovered import Router, _direct, cfg_on, disc, redirect, run  # noqa: F401  (_direct: autouse)
from vera import arxiv_fulltext, auto_acquire, fulldoc_db, fulltext_loader, licence_gate

ROBOTS = "https://arxiv.org/robots.txt"
CC_BY_HTTPS = "https://creativecommons.org/licenses/by/4.0/"
_REAL_DECISION = licence_gate.licence_decision


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch):
    monkeypatch.setattr(arxiv_fulltext, "_sleep", lambda s: None)


def _proposed_rule(c):
    """TEST-ONLY stand-in (same as tests/test_arxiv_fulltext.py) used ONLY to obtain stored rows for the read-path
    re-derivation checks below; it says nothing about the real gate."""
    u = str(c.get("url") or "")
    lic = ((c.get("licence") or {}).get("content") or {})
    if u.startswith("https://arxiv.org/html/") and "creativecommons.org" in str(lic.get("url") or ""):
        return _REAL_DECISION({**c, "url": u.replace("/html/", "/abs/", 1)})
    return _REAL_DECISION(c)


@pytest.fixture
def allow_arxiv_html(monkeypatch):
    """ARX-REAL-VERIFY: stand-in DISABLED; the real licence gate decides."""
    return None


@pytest.fixture
def spy_kwargs(monkeypatch):
    calls = []
    real = fulltext_loader.prepare_from_parts
    monkeypatch.setattr(fulltext_loader, "prepare_from_parts",
                        lambda *a, **k: calls.append(copy.deepcopy(k)) or real(*a, **k))
    return calls


def _go(web, cfg=None, src=None):
    return run(cfg or cfg_on(), [src or X.discovery_src()], web)


def _attempts():
    return arxiv_fulltext.last_attempts(X.AID + X.VER)


def _expected_evidence():
    return {"id": X.AID, "version": X.VER, "abs_url": X.ABS_URL, "licence_url": CC_BY_HTTPS, "source": "arxiv abs page"}


# ------------------------------------------------------------------------------------------------ R2
@pytest.mark.parametrize("final", [
    "https://arxiv.org/abs/2605.23989v2",          # another version
    "https://arxiv.org/abs/2605.23989",            # unversioned (= latest) when v1 was requested
    "https://arxiv.org/abs/2605.00001v1",          # another paper
    "https://arxiv.org/html/2605.23989v1",         # not an abs page
    "https://arxiv.org/abs/2605.23989v1?x=1",      # query
])
def test_abs_final_url_must_be_the_requested_version(final):
    web = Router({}, {}, {**X.routes(), X.ABS_URL: redirect(final), final: X.html_resp(X.abs_page())})
    o, store = _go(web)
    assert disc(o)["reason"] == "R-ACQ-IDENTITY_MISMATCH" and store.plans == []
    assert [u for u, _h, _k in web.calls] == [ROBOTS, X.ABS_URL, final]  # only the abs redirect chain; no content fetch
    last = _attempts()[-1]
    assert last["representation"] == "abs" and last["status"] == "refused"
    assert last["reason"] == "abs final URL is not the requested id/version abstract page" and last["final_url"] == final


def test_abs_final_url_trailing_slash_is_accepted():
    final = X.ABS_URL + "/"
    web = Router({}, {}, {**X.routes(), X.ABS_URL: redirect(final), final: X.html_resp(X.abs_page())})
    o, _store = _go(web)
    assert disc(o)["outcome"] == "admitted"  # passes R2; admitted at the v2 gate


def test_unversioned_record_requires_unversioned_abs_final():
    src = {**X.discovery_src(url="https://arxiv.org/abs/2605.23989"), "identifier": "arXiv:2605.23989"}
    unv = "https://arxiv.org/abs/2605.23989"
    web = Router({}, {}, {**X.routes(), unv: redirect(X.ABS_URL)})  # unversioned request lands on v1: refused
    o, _store = _go(web, src=src)
    assert disc(o)["reason"] == "R-ACQ-IDENTITY_MISMATCH"
    assert arxiv_fulltext.last_attempts("2605.23989")[-1]["reason"].startswith("abs final URL")


# ------------------------------------------------------------------------------------------------ R1 route side
def test_route_hands_the_version_bound_evidence_to_the_loader(spy_kwargs):
    o, _store = _go(Router({}, {}, X.routes()))
    assert disc(o)["outcome"] == "admitted"  # the v2 gate accepts the version-bound block
    assert spy_kwargs and spy_kwargs[-1].get("arxiv_evidence") == _expected_evidence()


def test_evidence_block_is_built_only_from_route_arguments():
    ev = arxiv_fulltext.evidence_block(X.AID, "v3", CC_BY_HTTPS)
    assert ev == {"id": X.AID, "version": "v3", "abs_url": "https://arxiv.org/abs/2605.23989v3",
                  "licence_url": CC_BY_HTTPS, "source": "arxiv abs page"}


def _item(**extra):
    return {"document_id": "d", "title": "t", "doi": "10.1/x", "landing_url": X.ABS_URL, "fulltext_url": X.HTML_URL,
            "fulltext_format": "html", "licence": "cc by", "licence_url": CC_BY_HTTPS, **extra}


def test_manifest_item_or_conversion_record_never_produce_evidence():
    cfg = fulltext_loader.load_config()
    forged = _expected_evidence()
    item = _item(arxiv_evidence=forged, licence_evidence=forged)
    meta = {"arxiv_evidence": forged, "meta": {"arxiv_evidence": forged}}
    with pytest.raises(fulltext_loader.Refused):  # the unchanged gate refuses arXiv HTML
        fulltext_loader.check_licence(item, meta, cfg)
    seen = []
    real = licence_gate.apply_licence_gate

    def spy(cands):
        seen.extend(copy.deepcopy(cands))
        return real(cands)
    import vera.fulltext_loader as fl_mod
    orig = fl_mod.apply_licence_gate
    fl_mod.apply_licence_gate = spy
    try:
        with pytest.raises(fulltext_loader.Refused):
            fulltext_loader.check_licence(item, meta, cfg)
    finally:
        fl_mod.apply_licence_gate = orig
    assert seen and all("arxiv_evidence" not in (c.get("licence") or {}) and "arxiv_evidence" not in c for c in seen)


def test_discovery_candidate_carries_no_evidence():
    c = arxiv_fulltext.discovered_candidate(X.discovery_src(), None, "preprint", {"epistemic": "x"}, cfg_on())
    assert c is not None and "arxiv_evidence" not in json.dumps(c)


def _rows(plans):
    srcs, spans = A.rows_from_plans(plans)
    return [{**s, "provenance": json.loads(s["provenance"])} for s in srcs], spans


def test_stored_route_rows_carry_and_rederive_the_evidence(allow_arxiv_html):
    o, store = _go(Router({}, {}, X.routes()))
    assert disc(o)["outcome"] == "admitted"
    prov = store.plans[0]["source"]["provenance"]
    assert prov["licence"]["arxiv_evidence"] == _expected_evidence()
    assert arxiv_fulltext.stored_evidence(prov) == _expected_evidence()
    srcs, spans = _rows(store.plans)
    c = fulldoc_db.load_acquired_rows(srcs, spans, auto_acquire._fd_cfg(), cfg_on())
    assert c.status == "ok" and c.acquired


@pytest.mark.parametrize("mutate", [
    lambda p: p["licence"]["arxiv_evidence"].update(version="v2"),
    lambda p: p["licence"]["arxiv_evidence"].update(id="2605.00001"),
    lambda p: p["licence"]["arxiv_evidence"].update(licence_url="https://creativecommons.org/licenses/by-sa/4.0/"),
    lambda p: p["acquisition"].update(resolver="europepmc"),
    lambda p: p["acquisition"]["arxiv"].update(version="v2"),
])
def test_read_path_refuses_rows_whose_evidence_does_not_rederive(allow_arxiv_html, mutate):
    o, store = _go(Router({}, {}, X.routes()))
    bad = copy.deepcopy(store.plans[0])
    mutate(bad["source"]["provenance"])
    assert arxiv_fulltext.stored_evidence(bad["source"]["provenance"]) is None
    srcs, spans = _rows([bad])
    c = fulldoc_db.load_acquired_rows(srcs, spans, auto_acquire._fd_cfg(), cfg_on())
    assert not c.acquired


def test_non_route_rows_never_get_evidence_even_if_stored():
    prov = {"acquisition": {"resolver": "europepmc", "arxiv": {"id": X.AID, "version": X.VER, "abs_url": X.ABS_URL,
                                                               "licence_url": CC_BY_HTTPS}},
            "licence": {"content": {}, "arxiv_evidence": _expected_evidence()}}
    assert arxiv_fulltext.stored_evidence(prov) is None
    assert arxiv_fulltext.stored_evidence({"licence": {"arxiv_evidence": _expected_evidence()}}) is None


# ------------------------------------------------------------------------------------------------ R4
def test_gate_family_list_narrows_the_route_list(monkeypatch):
    monkeypatch.setattr(arxiv_fulltext, "_gate_families", lambda: ("by",))
    o, store = _go(Router({}, {}, X.routes(licence="http://creativecommons.org/licenses/by-sa/4.0/")))
    assert disc(o)["reason"] == "R-ACQ-ARXIV_LICENCE_HOLD" and store.plans == []


def test_gate_family_list_cannot_widen_the_route_list(monkeypatch):
    monkeypatch.setattr(arxiv_fulltext, "_gate_families", lambda: ("by", "by-nd", "zero"))
    assert arxiv_fulltext.allowed_families(cfg_on()) == {"by"}
    monkeypatch.setattr(arxiv_fulltext, "_gate_families", lambda: ())  # malformed gate list: fail closed
    assert arxiv_fulltext.allowed_families(cfg_on()) == set()
    monkeypatch.setattr(arxiv_fulltext, "_gate_families", lambda: None)  # key not present yet: route list
    assert arxiv_fulltext.allowed_families(cfg_on()) == {"by", "by-sa", "by-nc", "by-nc-sa"}


def test_gate_family_reader_fails_closed_on_malformed_values(monkeypatch, tmp_path):
    for content, want in (('{"arxiv_html_cc_families": "by"}', ()), ('{"x": 1}', None), ("not json", ()),
                          ('{"arxiv_html_cc_families": ["BY", "by-sa"]}', ("by", "by-sa"))):
        f = tmp_path / "rules.json"
        f.write_text(content, encoding="utf-8")
        monkeypatch.setattr(licence_gate, "RULES_PATH", f)
        assert arxiv_fulltext._gate_families() == want, content
