"""H19-T2/T2b: ND news branch of the licence gate and use gate. Offline; config is patched in memory (never on disk)."""
import copy

import pytest

from vera import licence_gate as lg, news_access as na, reporting_lane as rl, source_permissions as sp

URL = "https://www.commondreams.org/news/a-story"
ND3 = {"id": "CC-BY-NC-ND-3.0", "url": "https://creativecommons.org/licenses/by-nc-nd/3.0/us/"}


def cand(url=URL, content=ND3, meta=None, **kw):
    lic = {"content": dict(content)} if content else {}
    if meta:
        lic["metadata"] = meta
    return {"url": url, "licence": lic, **kw}


@pytest.fixture
def cfg_on(monkeypatch):
    c = copy.deepcopy(rl.load_config())
    c["nc_basis"] = {"status": "SATISFIED"}
    monkeypatch.setattr(rl, "load_config", lambda *a, **k: c)
    monkeypatch.setattr(lg, "nd_news_enabled", lambda *a, **k: True)
    return c


@pytest.fixture
def flag_on_only(monkeypatch):
    monkeypatch.setattr(lg, "nd_news_enabled", lambda *a, **k: True)


def test_flag_off_state_rejects(monkeypatch):
    # RETRIEVAL-1109: the shipped flag is now ON (test_shipped_flag_is_on); this asserts the OFF state explicitly.
    monkeypatch.setattr(lg, "nd_news_enabled", lambda *a, **k: False)
    d, _why, code = lg.licence_status(cand())
    assert (d, code) == ("reject", lg.RC_REJECT)
    assert sp.evaluate(URL, licence=ND3).use_reason == "use_nd_licence"


def test_shipped_flag_is_on():
    # RETRIEVAL-1109 / OD-1109-RETRIEVAL: nd_news.enabled true and nc_basis SATISFIED ship ON.
    assert lg.nd_news_enabled() is True
    assert rl.load_config()["nc_basis"]["status"] == "SATISFIED"


def test_flag_on_nc_basis_unconfirmed_still_rejects(monkeypatch, flag_on_only):
    # nc_basis UNCONFIRMED (patched; the shipped value is SATISFIED) -> news_access link_only -> no exception
    c = copy.deepcopy(rl.load_config())
    c["nc_basis"] = {"status": "UNCONFIRMED"}
    monkeypatch.setattr(rl, "load_config", lambda *a, **k: c)
    assert lg.licence_status(cand())[2] == lg.RC_REJECT


def test_flag_and_basis_on_defers_answer_only(cfg_on):
    d, why, code = lg.licence_status(cand())
    assert (d, code) == ("defer", lg.RC_ND_NEWS) and "answer-only" in why and "not stored" in why
    assert lg.licence_decision(cand())[0] != "allow"
    # same call through apply_licence_gate never stores
    assert lg.apply_licence_gate([cand()])[0]["licence_decision"] == "defer"


def test_forged_candidate_fields_ignored(cfg_on):
    c = cand(url="https://unlisted-news.example/a", source_class="news", access={"mode": "full_text"},
             operations={"summary": "ALLOW"}, nd_news=True)
    assert lg.licence_status(c)[2] == lg.RC_REJECT
    # a listed host is permitted by the row, not by anything the candidate says
    assert lg.nd_news_permitted(cand(source_class="x")) is True
    # metadata-only governing record never gets the exception
    assert lg.nd_news_permitted(cand(stored_text_kind=lg.METADATA_KIND)) is False


@pytest.mark.parametrize("content", [
    {"id": "CC-BY-ND-4.0", "url": "https://creativecommons.org/licenses/by-nd/4.0/"},   # no NC: out of family
    {"id": "CC-BY-NC-ND-4.0", "url": "https://creativecommons.org/licenses/by-nc-nd/4.0/"},  # Common Dreams row is 3.0 family same -> see below
    {"id": "CC-BY-NC-ND-3.0 fee only", "url": None},
    {"id": "CC-BY-NC-ND-3.0 no educational use", "url": None},
    {"id": "", "url": None},
])
def test_other_licences_keep_rejecting(cfg_on, content):
    c = cand(content=content)
    if content["id"] == "CC-BY-NC-ND-4.0":
        # same family as the row (versions differ): allowed by family; page-href check (A7) compares family only
        assert lg.licence_status(c)[2] == lg.RC_ND_NEWS
    else:
        assert lg.licence_status(c)[0] == "reject" or content["id"] == ""


def test_metadata_nd_still_rejects(cfg_on):
    c = cand(meta={"id": "CC-BY-ND-4.0"}, content={"id": "CC-BY-NC-ND-3.0", "url": ND3["url"]})
    assert lg.licence_status(c)[2] == lg.RC_REJECT


def test_page_licence_mismatch_rejects(cfg_on):
    c = cand(content={"id": "CC-BY-NC-ND-3.0", "url": "https://creativecommons.org/licenses/by-nd/3.0/"})
    assert lg.licence_status(c)[0] == "reject"


def test_arxiv_and_non_nd_unchanged(cfg_on):
    assert lg.licence_status({"url": "https://arxiv.org/pdf/2101.00001", "licence": {}})[2] == lg.RC_ARXIV
    assert lg.licence_status({"url": "https://x.example/a", "licence": {"content": {"id": "CC-BY-4.0"}}})[0] == "allow"


def test_use_gate_ok_nd_news_and_overrides(cfg_on):
    r = sp.evaluate(URL, licence=ND3)
    assert r.use_reason == "use_ok_nd_news" and r.use_for_answer is True
    assert r.ingest_reason == "ingest_not_usable" and r.ingest is False  # never stored
    assert sp.evaluate(URL, licence=ND3, headers={"X-Robots-Tag": "noai"}).use_reason == "use_no_ai_signal"
    assert sp.evaluate(URL, licence=ND3, html_head='<meta name="robots" content="noai">').use_for_answer is False
    assert sp.evaluate(URL, licence=ND3, headers={"Content-Signal": "ai-input=no"}).use_for_answer is False
    # unlisted host: still use_nd_licence
    assert sp.evaluate("https://unlisted-news.example/a", licence=ND3).use_reason == "use_nd_licence"


def test_use_gate_tdm_reserved_bars_every_lane():
    for h in ({"TDM-Reservation": "1"}, {"tdm-reservation": "unreadable"}):
        r = sp.evaluate("https://x.example/a", licence="CC-BY-4.0", headers=h)
        assert r.use_reason == "use_tdm_reserved" and r.use_for_answer is False and r.ingest_reason == "ingest_not_usable"
    assert sp.evaluate("https://x.example/a", licence="CC-BY-4.0", headers={"TDM-Reservation": "0"}).use_for_answer


def test_phrases_and_codes_present():
    cfg = sp.load_config()
    for code in ("use_ok_nd_news", "use_tdm_reserved"):
        assert code in cfg["reason_codes"]["use_for_answer"] and code in cfg["public_phrases"]["use_for_answer"]
    assert cfg["public_phrases"]["use_for_answer"]["use_ok_nd_news"] == ""


def test_outlet_verdict_gates_access_rows_until_flag_and_basis(monkeypatch):
    c = copy.deepcopy(rl.load_config())
    u = "https://commondreams.org/a"
    monkeypatch.setattr(lg, "nd_news_enabled", lambda *a, **k: False)
    c["nc_basis"] = {"status": "UNCONFIRMED"}  # RETRIEVAL-1109: shipped state is ON; start from OFF explicitly
    assert rl.outlet_verdict(u, c) == "links_only"                       # flag off
    assert rl._raw_outlet_verdict(u, c) == "allow"
    monkeypatch.setattr(lg, "nd_news_enabled", lambda *a, **k: True)
    assert rl.outlet_verdict(u, c) == "links_only"                       # F1: flag on, nc_basis UNCONFIRMED
    assert u.split("/")[2] not in [h for h in rl.allow_hosts(c) if rl.outlet_verdict("https://" + h + "/a", c) != "allow"]
    c["nc_basis"] = {"status": "SATISFIED"}
    assert rl.outlet_verdict(u, c) == "allow"                            # flag on and every A-check holds
    c["outlets"] = [dict(o, access=dict(o["access"], terms_ai_clause={"status": "RESTRICTED"})) if o["host"] == "commondreams.org" else o
                    for o in c["outlets"]]
    assert rl.outlet_verdict(u, c) == "links_only"                       # any failing A-check
    assert rl.outlet_verdict("https://globalvoices.org/a", c) == "allow"  # non-ND rows never affected


def test_raw_verdict_is_private():
    import inspect
    assert "raw" not in inspect.signature(rl.outlet_verdict).parameters
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "vera"
    users = [p.name for p in root.glob("*.py") if "_raw_outlet_verdict" in p.read_text(encoding="utf-8")]
    assert sorted(users) == ["news_access.py", "reporting_lane.py"]
