"""FAIRUSE-TIER-1108 (owner decision OD-1108-FAIRUSE, option O2): the fair-use short-quote licence tier, its output
caps, attribution, the takedown path and the published statement. OFFLINE: synthetic fixtures, fake transports and
fake DB cursors; no network, DB or LLM. The flag is switched by monkeypatching vera.fair_use.load (no file writes)."""
import copy
import json

import pytest

from tests import acq_support as A
from tests import arxiv_synth as X
from tests.test_acq_discovered import Router, _direct, cfg_on, disc, run  # noqa: F401  (_direct: autouse)
from vera import arxiv_fulltext, auto_acquire, fair_use, fulldoc_db, fulltext_loader, licence_gate


SHIPPED = fair_use.load()
ON = {**SHIPPED, "enabled": True}
ARR, ARX = "LicenseRef-all-rights-reserved", "LicenseRef-arxiv-nonexclusive-distrib-1.0"
ARX_URL = "https://arxiv.org/licenses/nonexclusive-distrib/1.0/"
HTML = "https://arxiv.org/html/2605.23989v1"
ACCESS = fair_use.access_record("test route", "2026-11-08T00:00:00+00:00")


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(fair_use, "load", lambda path=None: copy.deepcopy(ON))


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch):
    monkeypatch.setattr(arxiv_fulltext, "_sleep", lambda s: None)


@pytest.fixture(autouse=True)
def _clean_withdrawals():
    fair_use.reset_for_tests()
    yield
    fair_use.reset_for_tests()


def _c(lic_id, lic_url=None, url="https://example.org/paper", access=ACCESS, **kw):
    c = {"url": url, "licence": {"content": {"id": lic_id, "url": lic_url}}, "stored_text_kind": "full_text", **kw}
    if access is not None:
        c["access_check"] = copy.deepcopy(access)
    return c


# ------------------------------------------------------------------------------------------------ shipped config
def test_shipped_config_is_off_with_calibrated_caps():
    assert SHIPPED["enabled"] is False  # ships OFF until the opus reviewer and the sonnet verifier both pass
    assert set(SHIPPED["eligible_licence_ids"]) == {ARR, ARX, "LicenseRef-arxiv-assumed-1991-2003"}
    assert fair_use.caps() == {"quote_max_chars": 200, "max_quotes_per_source": 2, "max_quote_chars_per_source": 400,
                               "max_share_of_document": 0.05}
    note = SHIPPED["_calibration_note"]
    assert "calibration-work.json" in note and "9 quotes" in note and "max 199" in note  # CLAUDE.md: no cap without a note


def test_every_eligible_id_is_an_active_non_storable_allowlist_entry():
    by_alias, _ = licence_gate._load_allowlist()
    for i in SHIPPED["eligible_licence_ids"]:
        e = by_alias[licence_gate.norm_id(i)]
        assert e["id"] == i and e["status"] == "active" and e["storable"] is False and e.get("reject") is not True


def test_caps_fall_back_to_strict_defaults_on_bad_values():
    assert fair_use.caps({"caps": {"quote_max_chars": "x", "max_quotes_per_source": -1, "max_share_of_document": 7}}) \
        == fair_use._DEFAULT_CAPS
    assert fair_use.caps({}) == fair_use._DEFAULT_CAPS


# ------------------------------------------------------------------------------------------------ licence gate
@pytest.mark.parametrize("lic_id,lic_url", [(ARR, None), ("All rights reserved", None), (ARX, ARX_URL),
                                            ("arxiv nonexclusive distrib 1.0", None)])
def test_flag_off_keeps_link_only(lic_id, lic_url):
    d, _why, code = licence_gate.licence_status(_c(lic_id, lic_url))
    assert (d, code) == ("defer", licence_gate.RC_NOT_STORABLE)


@pytest.mark.parametrize("lic_id,lic_url", [(ARR, None), ("All rights reserved", None), ("(c) all rights reserved", None),
                                            (ARX, ARX_URL), ("arxiv nonexclusive distrib 1.0", None)])
def test_flag_on_admits_eligible_licence_with_access_check(on, lic_id, lic_url):
    d, why, code = licence_gate.licence_status(_c(lic_id, lic_url))
    assert (d, code) == ("allow", licence_gate.RC_FAIR_USE) and "fair-use-quote (O2, OD-1108-FAIRUSE)" in why


@pytest.mark.parametrize("access", [None, {}, {**ACCESS, "robots": "unknown"}, {**ACCESS, "tdm_reservation": "reserved"},
                                    {**ACCESS, "login_required": True}, {**ACCESS, "login_required": 0},
                                    {**ACCESS, "robots": True}, "allowed"])
def test_flag_on_without_a_valid_access_check_stays_link_only(on, access):
    c = _c(ARR, access=None)
    if access is not None:
        c["access_check"] = access
    assert licence_gate.licence_status(c)[0] == "defer"


@pytest.mark.parametrize("lic_id,lic_url,want,code", [
    ("Some Publisher Licence v3", None, "defer", licence_gate.RC_UNKNOWN),       # unknown: not covered, recorded
    (None, None, "defer", licence_gate.RC_MISSING),                               # missing: not covered
    ("cc by-nd 4.0", None, "reject", licence_gate.RC_REJECT),                    # reject list unchanged
    ("all rights reserved no educational use", None, "reject", licence_gate.RC_REJECT),
    ("all rights reserved", None, "allow", licence_gate.RC_FAIR_USE),
    ("cc by 4.0", None, "allow", licence_gate.RC_ALLOW),                         # CC unchanged
    ("Elsevier user license", None, "reject", licence_gate.RC_REJECT),           # recognised reject entry
    ("open access", None, "defer", licence_gate.RC_NOT_STORABLE),                # a bare OA label is not eligible
    ("author manuscript", None, "defer", licence_gate.RC_NOT_STORABLE),
    (ARR, "https://creativecommons.org/licenses/by/4.0/", "defer", licence_gate.RC_CONFLICT),  # id/url disagree
])
def test_flag_on_scope(on, lic_id, lic_url, want, code):
    d, _why, got = licence_gate.licence_status(_c(lic_id, lic_url))
    assert (d, got) == (want, code)


def test_abstract_metadata_kind_never_uses_the_tier(on):
    c = _c(ARR, stored_text_kind="abstract_metadata")
    c["licence"] = {"metadata": {"id": ARR}}
    assert licence_gate.licence_status(c)[0] == "defer"


def test_arxiv_html_with_default_licence(monkeypatch):
    lic = {"content": {"id": ARX, "url": ARX_URL}, "arxiv_evidence": arxiv_fulltext.evidence_block("2605.23989", "v1", ARX_URL)}
    c = {"url": HTML, "licence": lic, "stored_text_kind": "full_text", "access_check": dict(ACCESS)}
    assert licence_gate.licence_status(c)[0] == "reject"  # flag off: arXiv HTML policy unchanged
    monkeypatch.setattr(fair_use, "load", lambda path=None: copy.deepcopy(ON))
    assert licence_gate.licence_status(c)[:3:2] == ("allow", licence_gate.RC_FAIR_USE)
    for bad in ("http://arxiv.org/licenses/nonexclusive-distrib/1.0/", "https://arxiv.org/licenses/assumed-1991-2003/"):
        b = copy.deepcopy(c)
        b["licence"]["content"]["url"] = bad
        b["licence"]["arxiv_evidence"]["licence_url"] = bad
        assert licence_gate.licence_status(b)[0] == "reject"
    for u in ("https://arxiv.org/pdf/2605.23989v1", "https://arxiv.org/e-print/2605.23989v1"):
        assert licence_gate.licence_status({**c, "url": u})[0] == "reject"  # e-print/PDF never


def _item(**kw):
    return {"fulltext_url": "https://example.org/p", "landing_url": "https://example.org/p", "licence": ARR,
            "licence_url": None, "access_check": dict(ACCESS), **kw}


_LCFG = {"stored_text_kind": "full_text", "reason_codes": {"licence_hold": "R-HOLD", "licence_reject": "R-REJ"}}


def test_loader_ignores_an_access_check_without_route_evidence(on):
    """Review F2: a manifest or other item cannot assert an access check; only the arXiv route's evidence carries it."""
    with pytest.raises(fulltext_loader.Refused):
        fulltext_loader.check_licence(_item(), {}, _LCFG)


def test_loader_records_basis_and_access_check_on_provenance(on):
    ev = arxiv_fulltext.evidence_block("2605.23989", "v1", ARX_URL)
    item = _item(fulltext_url=HTML, landing_url="https://arxiv.org/abs/2605.23989v1", licence=ARX, licence_url=ARX_URL)
    rec = fulltext_loader.check_licence(item, {}, _LCFG, arxiv_evidence=ev)
    assert rec["licence_decision"] == "allow"
    fu = rec["licence"]["fair_use"]
    assert fu["basis"] == "fair-use-quote (O2, OD-1108-FAIRUSE)"
    assert "quoted, cited and listed" in fu["use"] and "not republished or redistributed" in fu["use"]
    assert fu["access_check"] == {k: ACCESS[k] for k in ("robots", "tdm_reservation", "login_required", "checked_by",
                                                          "checked_at")}


def test_loader_without_access_check_holds(on):
    item = {"fulltext_url": "https://example.org/p", "landing_url": "https://example.org/p", "licence": ARR,
            "licence_url": None}
    cfg = {"stored_text_kind": "full_text", "reason_codes": {"licence_hold": "R-HOLD", "licence_reject": "R-REJ"}}
    with pytest.raises(fulltext_loader.Refused):
        fulltext_loader.check_licence(item, {}, cfg)


def test_family_helper_reads_the_arxiv_route_id():
    assert auto_acquire._family(ARX) == "nonexclusive-distrib"
    assert auto_acquire._family(ARX_URL) == "nonexclusive-distrib"
    assert auto_acquire._family("cc by 4.0") == "by"  # unchanged


# ------------------------------------------------------------------------------------------------ arXiv route e2e
def _go(lic):
    return run(cfg_on(), [X.discovery_src()], Router({}, {}, X.routes(licence=lic)))


@pytest.mark.usefixtures("legacy_arxiv_route")  # ARXIV-PORTAL-1110: pins the dormant arXiv evidence route
def test_arxiv_default_held_while_flag_off():
    o, store = _go(X.ARXIV_DEFAULT)
    assert disc(o)["reason"] == "R-ACQ-ARXIV_LICENCE_HOLD" and store.plans == []


@pytest.mark.usefixtures("legacy_arxiv_route")  # ARXIV-PORTAL-1110: pins the dormant arXiv evidence route
def test_arxiv_default_admitted_with_basis_when_on_and_kill_switch_at_read(monkeypatch):
    monkeypatch.setattr(fair_use, "load", lambda path=None: copy.deepcopy(ON))
    o, store = _go(X.ARXIV_DEFAULT)
    assert disc(o)["outcome"] == "admitted", disc(o)
    prov = store.plans[0]["source"]["provenance"]
    assert prov["licence_decision"] == "allow"
    assert prov["licence"]["content"]["id"] == ARX and prov["licence"]["content"]["url"] == ARX_URL
    assert prov["licence"]["fair_use"]["basis"] == fair_use.BASIS
    assert prov["licence"]["fair_use"]["access_check"]["robots"] == "allowed"
    att = prov["attribution"]
    assert att["source"].startswith("https://") and att["title"] and att["creator"] and att["licence_url"] == ARX_URL
    srcs, spans = A.rows_from_plans(store.plans)
    rows = [{**s, "provenance": json.loads(s["provenance"])} for s in srcs]
    c = fulldoc_db.load_acquired_rows(copy.deepcopy(rows), spans, auto_acquire._fd_cfg(), cfg_on())
    assert c.status == "ok" and len(c.acquired) == 1  # read path re-derives the same record
    monkeypatch.setattr(fair_use, "load", lambda path=None: copy.deepcopy(SHIPPED))
    c = fulldoc_db.load_acquired_rows(copy.deepcopy(rows), spans, auto_acquire._fd_cfg(), cfg_on())
    assert not c.acquired  # flag off: stored fair-use rows are unusable (kill switch)


@pytest.mark.usefixtures("legacy_arxiv_route")  # ARXIV-PORTAL-1110: pins the dormant arXiv evidence route
def test_arxiv_default_with_other_url_spelling_is_still_held(on):
    o, store = _go("http://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html")
    assert disc(o)["reason"] == "R-ACQ-ARXIV_LICENCE_HOLD" and store.plans == []


@pytest.mark.usefixtures("legacy_arxiv_route")  # ARXIV-PORTAL-1110: pins the dormant arXiv evidence route
@pytest.mark.parametrize("lic", [X.CC_BY_ND, "https://example.org/some-licence", None])
def test_arxiv_other_licences_unchanged_when_on(on, lic):
    o, store = _go(lic)
    assert disc(o).get("reason") in ("R-LIC-REJECT", "R-ACQ-ARXIV_LICENCE_HOLD", "R-ACQ-LICENCE_ABSENT")
    assert store.plans == []


@pytest.mark.usefixtures("legacy_arxiv_route")  # ARXIV-PORTAL-1110: pins the dormant arXiv evidence route
def test_robots_disallow_still_refuses_when_on(on):
    web = Router({}, {}, {**X.routes(licence=X.ARXIV_DEFAULT),
                          "https://arxiv.org/robots.txt": A.Resp(body=b"User-agent: *\nDisallow: /html/\n",
                                                                 ctype="text/plain")})
    o, store = run(cfg_on(), [X.discovery_src()], web)
    assert disc(o)["reason"] == "R-ACQ-ROBOTS" and store.plans == []


# ------------------------------------------------------------------------------------------------ output caps
def _att(**kw):
    return {"creator": "A. Author", "title": "T", "source": "https://arxiv.org/abs/2605.23989v1", "licence": ARX,
            "licence_url": ARX_URL, "changes": "Short excerpt", **kw}


def _src(n, did="d1", lic=ARX, url=ARX_URL):
    return {"n": n, "document_id": did, "licence": lic, "licence_url": url}


def _cit(n, q, **kw):
    return {"n": n, "quote": q, "quote_span": "chars=0-1", "attribution": _att(), **kw}


def test_caps_per_source_count_and_chars():
    srcs = [_src(1), _src(2), _src(3), _src(4, did="d2")]
    cits = [_cit(1, "a" * 150), _cit(2, "b" * 150), _cit(3, "c" * 50), _cit(4, "d" * 120)]
    assert fair_use.limit_citations(cits, srcs) == 1
    assert [bool(c["quote"]) for c in cits] == [True, True, False, True]  # third quote of d1 dropped
    assert cits[2]["quote_span"] is None


def test_caps_chars_per_source_and_quote_length():
    srcs = [_src(1), _src(2), _src(3, did="d3")]
    cits = [_cit(1, "a" * 200), _cit(2, "b" * 201), _cit(3, "c" * 200)]
    fair_use.limit_citations(cits, srcs)
    assert [bool(c["quote"]) for c in cits] == [True, False, True]
    cits = [_cit(1, "a" * 200), _cit(2, "b" * 200 + "")]
    srcs = [_src(1), _src(2)]
    fair_use.limit_citations(cits, srcs)
    assert all(c["quote"] for c in cits)  # 400 chars allowed exactly


def test_share_cap_for_short_documents():
    srcs = [_src(1), _src(2)]
    cits = [_cit(1, "a" * 100), _cit(2, "b" * 100)]
    fair_use.limit_citations(cits, srcs, {"d1": 3000})  # 5% of 3000 = 150 chars
    assert [bool(c["quote"]) for c in cits] == [True, False]


def test_quote_without_full_attribution_link_is_dropped():
    cits = [_cit(1, "a" * 50, attribution=_att(source="")), _cit(1, "b" * 50, attribution=_att(source="arxiv 1"))]
    fair_use.limit_citations(cits, [_src(1)])
    assert [c["quote"] for c in cits] == [None, None]


def test_cc_sources_are_untouched():
    cc = "https://creativecommons.org/licenses/by/4.0/"
    srcs = [_src(i, lic="cc by 4.0", url=cc) for i in (1, 2, 3)]
    cits = [_cit(i, "x" * 200) for i in (1, 2, 3)]
    assert fair_use.limit_citations(cits, srcs) == 0 and all(c["quote"] for c in cits)


def test_caps_apply_whatever_the_flag():
    srcs = [_src(i) for i in (1, 2, 3)]
    cits = [_cit(i, "x" * 10) for i in (1, 2, 3)]
    assert SHIPPED["enabled"] is False and fair_use.limit_citations(cits, srcs) == 1


def test_trace_excerpt_is_the_capped_quote_or_nothing():
    srcs = [_src(1), _src(2), _src(3, lic="cc by 4.0", url="https://creativecommons.org/licenses/by/4.0/")]
    cits = [_cit(1, "quoted text"), _cit(2, None)]
    tr = {"passages": [{"n": 1, "excerpt": "LONG " * 40}, {"n": 2, "excerpt": "other"}, {"n": 3, "excerpt": "cc text"}]}
    fair_use.limit_trace(tr, cits, srcs)
    assert [r["excerpt"] for r in tr["passages"]] == ["quoted text", None, "cc text"]


def test_answer_path_applies_caps(monkeypatch):
    """The two build_citations call sites in vera/fulldoc_answer.py are followed by the fair-use cap."""
    import inspect

    from vera import fulldoc_answer
    src = inspect.getsource(fulldoc_answer)
    assert src.count("_nd_quotes(citations, sources, cfg)  # H19 G6\n    _fair_use_quotes(citations, sources, ev)") == 2
    assert "fair_use.limit_trace(tr, citations or (), sources or ())" in src


# ------------------------------------------------------------------------------------------------ takedown
def test_withdrawn_matching(monkeypatch):
    monkeypatch.setattr(fair_use, "_config_entries", lambda path=None: [
        {"doi": "https://doi.org/10.1/ABC"}, {"url": "http://www.example.org/x/"}, {"arxiv_id": "2605.23989"},
        {"document_id": "acq-1"}])
    ks = fair_use.withdrawn_keys()
    assert fair_use.is_withdrawn({"doi": "10.1/abc"}, ks)
    assert fair_use.is_withdrawn({"url": "https://example.org/x"}, ks)
    assert fair_use.is_withdrawn({"url": "https://arxiv.org/abs/2605.23989v2"}, ks)
    assert fair_use.is_withdrawn({"document_id": "acq-1"}, ks)
    assert fair_use.is_withdrawn({"source_url": "https://example.org/x?utm=1"}, ks)
    assert fair_use.is_withdrawn({"source_url": "https://doi.org/10.1/abc"}, ks)  # memory recall carries only a URL
    assert not fair_use.is_withdrawn({"url": "https://example.org/y", "doi": "10.1/abd"}, ks)


def test_shipped_withdrawals_file_is_empty_and_valid():
    d = json.loads(fair_use.WITHDRAWALS_PATH.read_text(encoding="utf-8"))
    assert d["entries"] == [] and fair_use.withdrawn_keys() == set()


def test_admit_excludes_withdrawn_documents(monkeypatch):
    from vera import fulldoc_admission

    class Corpus:
        status, source_kind, unusable = "ok", "db", []
        acquired = [{"document_id": "acq-1", "url": "https://example.org/x"}]

        def retrieve(self, *a, **k):
            raise AssertionError("retrieval must not run without documents")

    monkeypatch.setattr(fair_use, "_config_entries", lambda path=None: [{"document_id": "acq-1"}])
    ev = fulldoc_admission.admit("what is x?", Corpus(), {})
    assert ev.status == "corpus_unusable" and not ev.admitted


def test_no_table_is_read_until_migration_023_ships():
    """AGENTS.md rule 7: no batch may need an unapplied migration; the DB reader ships with 023 (work/FAIRUSE-TIER-1108)."""
    src = fair_use.Path(fair_use.__file__).read_text(encoding="utf-8")
    assert "source_withdrawals FROM" not in src and "SELECT" not in src and not hasattr(fair_use, "refresh_from_db")


# ------------------------------------------------------------------------------------------------ published statement
def test_terms_statement_uses_the_existing_public_contact_route():
    d = json.loads((fair_use.CONFIG_PATH.parent / "ui_help_pages.json").read_text(encoding="utf-8"))
    sec = next(s for s in d["terms"]["sections"] if s["heading"] == "Third-party sources and licences")
    p = [x for x in sec["paragraphs"] if "Removal request" in x]
    assert len(p) == 1
    assert "quotes, cites and lists source references in discussion" in p[0]
    assert "does not republish or redistribute" in p[0] and "{repo_issues}" in p[0]
    assert "@" not in p[0] and "mailto" not in p[0]
    assert d["terms"]["effective_date"] == "2026-11-08"
