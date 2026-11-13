"""WPF-FOLLOW-1113 (P-1111-11 follow-ups from WPF-VERIFY-1112). Offline: the real selective_fetch.pinned_get runs with
requests.Session.get patched (tests.test_denial_guard_1111 helpers); requests per (host, path) are counted.
- A1/A2: D7 wiki_ref_portal.get_article and D8 reference_portal.fetch_page report a 401/403 (or a remembered denial)
  as `access_denied`, distinct from `unreachable` (brief rule 9); D8 stops the alternatives after it.
- A5: a 403 on a Europe PMC supplementaryFiles path under purpose metadata is remembered (download endpoint).
- A6: the Wikipedia search-API call maps a 401/403 to `route_denied`."""
import json
from pathlib import Path

import pytest

from tests.test_denial_guard_1111 import Resp, _clean, net  # noqa: F401  (_clean: autouse memory/DNS reset)
from vera import acq_discovered as ad
from vera import selective_fetch as sf

ROOT = Path(__file__).resolve().parent.parent
WIKI_HOST = "en.wikipedia.org"
D7_CFG = {"article_host": WIKI_HOST, "allowed_hosts": [WIKI_HOST], "timeout_s": 5, "max_response_bytes": 1000,
          "user_agent": "UA"}


def _d7(url):
    from vera import wiki_ref_portal as w
    return w.get_article(url, D7_CFG, http_get=sf.pinned_get, url_check=lambda u: True, clock=lambda: 0.0,
                         deadline=None)


# ---------------------------------------------------------------------------------------------- A1: D7 get_article
@pytest.mark.parametrize("status", [401, 403])
def test_a1_d7_denial_is_access_denied_and_never_resent(monkeypatch, status):
    from vera import wiki_ref_portal as w
    n = net(monkeypatch, {(WIKI_HOST, "/wiki/X"): [Resp(status)]})
    for _ in range(2):
        with pytest.raises(w.PortalError) as ei:
            _d7("https://en.wikipedia.org/wiki/X")
        assert str(ei.value) == "access_denied"
    assert n.count(WIKI_HOST, "/wiki/X") == 1


@pytest.mark.parametrize("resp,code", [(Resp(404), "unreachable"), (Resp(500), "unreachable"),
                                       (Resp(301, {"Location": "https://en.wikipedia.org/wiki/Y"}), "unreachable"),
                                       (Resp(429), "rate_limited")])
def test_a1_d7_other_statuses_keep_their_codes(monkeypatch, resp, code):
    from vera import wiki_ref_portal as w
    n = net(monkeypatch, {(WIKI_HOST, "/wiki/X"): [resp]})
    with pytest.raises(w.PortalError) as ei:
        _d7("https://en.wikipedia.org/wiki/X")
    assert str(ei.value) == code
    assert not ad.is_denied(urls=("https://en.wikipedia.org/wiki/X",)) and n.count(WIKI_HOST, "/wiki/X") == 1


# ---------------------------------------------------------------------------------------------- A2: D8 fetch_page
def _rp_cfg():
    return json.loads((ROOT / "config" / "reference_portals.json").read_text(encoding="utf-8"))


def _rp_deps(rp):
    return rp.Deps(http_get=sf.pinned_get, url_check=lambda u: (True, ""), robots_ok=lambda u, ua: True,
                   no_ai_input=lambda u, ua: False, tdmrep_ok=lambda u, p=None: True, crawl_ok=lambda u, ua: True,
                   limited=lambda portal, fn: fn)


@pytest.mark.parametrize("status", [401, 403])
def test_a2_d8_denial_is_access_denied_and_never_resent(monkeypatch, status):
    from vera import reference_portal as rp
    url = "https://en.wikipedia.org/wiki/Denied_Page"
    n = net(monkeypatch, {(WIKI_HOST, "/wiki/Denied_Page"): [Resp(status)]})
    for _ in range(2):
        with pytest.raises(rp.Skip) as ei:
            rp.fetch_page(url, {"rate_portal": "wikipedia"}, _rp_cfg(), _rp_deps(rp), lambda: 0.0, None)
        assert ei.value.code == "access_denied"
    assert n.count(WIKI_HOST, "/wiki/Denied_Page") == 1


@pytest.mark.parametrize("resp,code", [(Resp(404), "unreachable"), (Resp(500), "unreachable"),
                                       (Resp(302, {"Location": "https://en.wikipedia.org/wiki/Y"}), "unreachable")])
def test_a2_d8_other_statuses_keep_their_codes(monkeypatch, resp, code):
    from vera import reference_portal as rp
    url = "https://en.wikipedia.org/wiki/Other_Page"
    net(monkeypatch, {(WIKI_HOST, "/wiki/Other_Page"): [resp]})
    with pytest.raises(rp.Skip) as ei:
        rp.fetch_page(url, {"rate_portal": "wikipedia"}, _rp_cfg(), _rp_deps(rp), lambda: 0.0, None)
    assert ei.value.code == code


def test_a2_d8_access_denied_stops_the_alternatives(monkeypatch):
    """A 403 on the plural title stops the group: the singular alternative on the refusing host is not requested
    (a 404 still falls through to it, tests/test_reference_portal_1108.py)."""
    from tests import test_reference_portal_1108 as T
    from vera import reference_portal as rp
    rp.reset_for_tests()
    try:
        net_ = T.Net({T.WIKI: T.Resp(403, b""), T.WIKI_SING: T.Resp(200, T.PAGE)})
        cands, counts, net_ = T.run(net=net_)
        assert net_.calls == [T.WIKI] and counts.get("access_denied") == 1 and cands == []
    finally:
        rp.reset_for_tests()


# ---------------------------------------------------------------------------------------------- A5: Europe PMC
SUPP = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9/supplementaryFiles"
SUPP_KEY = ("www.ebi.ac.uk", "/europepmc/webservices/rest/PMC9/supplementaryFiles")


@pytest.mark.parametrize("scoped", [False, True])
def test_a5_supplementary_files_403_under_metadata_is_remembered(monkeypatch, scoped):
    from vera import api_permissions
    n = net(monkeypatch, {SUPP_KEY: [Resp(403)]})
    with api_permissions.holding("europepmc"):
        if scoped:
            with api_permissions.purpose_scope("metadata", SUPP):
                assert sf.pinned_get(SUPP).status_code == 403
                with pytest.raises(sf.DocumentDenied):
                    sf.pinned_get(SUPP)
        else:
            assert sf.pinned_get(SUPP, api_purpose="metadata").status_code == 403
            with pytest.raises(sf.DocumentDenied):
                sf.pinned_get(SUPP, api_purpose="metadata")
    assert n.count(*SUPP_KEY) == 1 and ad.is_denied(urls=(SUPP,))


def test_a5_escaped_marker_is_still_a_download_endpoint():
    from vera import api_permissions
    for path in ("/europepmc/webservices/rest/PMC9/%73upplementaryFiles",
                 "/europepmc/webservices/rest/PMC9/%66ullTextXML"):
        url = "https://www.ebi.ac.uk" + path
        with api_permissions.holding("europepmc"):
            dec = sf.api_gate(url, "metadata")
        assert dec is not None and sf._is_document_request(url, "metadata", dec) is True, path


def test_a5_metadata_search_stays_exempt_even_with_the_marker_in_the_query(monkeypatch):
    from vera import api_permissions
    s = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=supplementaryFiles"
    key = ("www.ebi.ac.uk", "/europepmc/webservices/rest/search")
    n = net(monkeypatch, {key: [Resp(403)]})
    with api_permissions.holding("europepmc"):
        sf.pinned_get(s, api_purpose="metadata")
        sf.pinned_get(s, api_purpose="metadata")
    assert n.count(*key) == 2 and not ad.is_denied(urls=(s,))


# ---------------------------------------------------------------------------------------------- A6: search API
class _R:
    def __init__(self, status):
        self.status_code, self.content, self.headers = status, b"", {}


def _search_cfg():
    from vera import wiki_ref_portal as w
    c = {**w.load_config(), "enabled": True}
    return {**c, "search_api": {**c["search_api"], "endpoint": "https://en.wikipedia.org/w/api.php"}}


@pytest.mark.parametrize("status,code", [(401, "route_denied"), (403, "route_denied"), (429, "rate_limited"),
                                         (500, "unreachable")])
def test_a6_search_api_refusal_is_route_denied(status, code):
    from vera import wiki_ref_portal as w
    with pytest.raises(w.PortalError) as ei:
        w.api_search("photosynthesis", WIKI_HOST, 5, _search_cfg(), get=lambda *a: _R(status))
    assert str(ei.value) == code
