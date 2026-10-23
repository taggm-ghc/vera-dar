"""ARX-GATE F-1 v4 tests (ARX-SECREV-1023 Round 3, C1-C4). Copy to tests/test_arx_gate_f1_v4.py ONLY after the owner has
applied work/arx-gate-proposal/v4/licence_gate_f1_host.v4.diff; at the same time remove the _F1 strict-xfail marks in
tests/test_arx_gate_after_authorization.py (they would XPASS-fail). Real gate, no monkeypatch; offline; no network/DB/LLM.
Before the diff is applied, the F-1, Unicode-dot, /abs/-spelling and normalisation-error cases fail."""
import pytest

from vera import arxiv_fulltext, licence_gate

AID, VER = "2605.23989", "v1"
CC_BY = "https://creativecommons.org/licenses/by/4.0/"
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"


def _html(url):  # full-text candidate with valid route evidence + CC BY content licence
    return {"url": url, "stored_text_kind": "full_text",
            "licence": {"content": {"id": "cc by", "url": CC_BY, "source": "manifest declared"},
                        "arxiv_evidence": arxiv_fulltext.evidence_block(AID, VER, CC_BY)}}


def _abs(url):  # abstract-page candidate: CC0 metadata governs
    return {"url": url, "stored_text_kind": "abstract_metadata",
            "licence": {"metadata": {"id": "cc0", "url": CC0}, "content": {"id": "cc by", "url": CC_BY}}}


def _d(c):
    return licence_gate.licence_decision(c)[0]


# C4 (a): the 9 F-1 cases (were strict xfail in test_arx_gate_after_authorization.py)
@pytest.mark.parametrize("url", [
    "https://arxiv.org:443/html/2605.23989v1", "https://u@arxiv.org/html/2605.23989v1",
    "https://u:p@arxiv.org/html/2605.23989v1", "https://arxiv.org:8080/html/2605.23989v1",
    "https://ArXiv.org/html/2605.23989v1", "https://arxiv.org:443/pdf/2605.23989v1",
    "https://u@arxiv.org/e-print/2605.23989v1", "https://arxiv.org./pdf/2605.23989v1",
    "https://arxiv.org./html/2605.23989v1",
])
def test_f1_spellings_refused_even_with_evidence(url):
    assert _d(_html(url)) == "reject"


# C4 (b): Unicode full-stop spellings (U+3002, U+FF0E, U+FF61) and other IDNA-equivalent spellings: never allowed
@pytest.mark.parametrize("host", ["arxiv。org", "arxiv．org", "arxiv｡org", "www。arxiv。org",
                                  "export．arxiv．org", "arxiv。org。", "ａrxiv.org",
                                  "arx­iv.org", "ARXIV．ORG"])
@pytest.mark.parametrize("path", ["/html/2605.23989v1", "/pdf/2605.23989v1", "/e-print/2605.23989v1",
                                  "/abs/2605.23989v1"])
def test_unicode_and_idna_spellings_never_allowed(host, path):
    url = f"https://{host}{path}"
    for c in (_html(url), _abs(url)):
        d = _d(c)
        assert d in ("reject", "defer") and d != "allow", (url, d)


# C4 (c): non-canonical /abs/ spellings are refused (C2: every path)
@pytest.mark.parametrize("url", [
    "https://arxiv.org./abs/2605.23989v1", "https://ArXiv.org/abs/2605.23989v1", "https://ARXIV.ORG/abs/2605.23989v1",
    "https://arxiv.org:443/abs/2605.23989v1", "https://u@arxiv.org/abs/2605.23989v1",
    "https://www.arxiv.org./abs/2605.23989v1", "https://Export.arxiv.org/abs/2605.23989v1",
])
def test_non_canonical_abs_spellings_refused(url):
    assert _d(_abs(url)) == "reject"


# C4 (d): unchanged behaviour on canonical spellings
@pytest.mark.parametrize("url", ["https://arxiv.org/abs/2605.23989v1", "https://www.arxiv.org/abs/2605.23989v1",
                                 "https://export.arxiv.org/abs/2605.23989v1", "http://arxiv.org/abs/2605.23989v1"])
def test_canonical_abs_unchanged(url):
    assert licence_gate.licence_decision(_abs(url)) == ("allow", "licence cc0")


@pytest.mark.parametrize("url", ["https://arxiv.org/html/2605.23989v1", "https://arxiv.org/html/2605.23989v1/"])
def test_plain_html_with_valid_evidence_and_cc_by_allowed(url):
    assert licence_gate.licence_decision(_html(url)) == ("allow", "licence cc by")


@pytest.mark.parametrize("url", ["https://www.arxiv.org/html/2605.23989v1", "https://export.arxiv.org/html/2605.23989v1",
                                 "http://arxiv.org/html/2605.23989v1"])
def test_other_canonical_hosts_and_http_html_still_refused(url):
    assert _d(_html(url)) == "reject"


# C1: any normalisation error -> defer, never allow
@pytest.mark.parametrize("url", ["https://arxiv..org/html/2605.23989v1", "https://" + "a" * 64 + ".org/x",
                                 "https://[::1/html/2605.23989v1", "https://arxiv。。org/pdf/2605.23989v1"])
def test_unnormalisable_host_is_deferred(url):
    d, why = licence_gate.licence_decision(_html(url))
    assert d == "defer" and "F-1" in why


# generic (non-arXiv) path unchanged, including internationalised hosts
@pytest.mark.parametrize("url", ["https://example.org/paper", "https://bücher.example/p", "https://u@example.org:8443/p",
                                 ""])
def test_generic_path_unchanged(url):
    c = {"url": url, "stored_text_kind": "full_text", "licence": {"content": {"id": "cc by", "url": CC_BY}}}
    assert licence_gate.licence_decision(c) == ("allow", "licence cc by")
