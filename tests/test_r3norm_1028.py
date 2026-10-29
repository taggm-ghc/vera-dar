"""R3NORM-1028: single canonical-source key normaliser, no-false-merge cases, legacy-key read compatibility. Offline."""
import pytest

from vera import url_key as uk
from vera.corpus_admission import canonical_url

SAME = [
    ("HTTPS://Example.COM/a/b", "https://example.com/a/b"),
    ("https://example.com:443/a", "https://example.com/a"),
    ("http://example.com:80/a", "http://example.com/a"),
    ("https://example.com/a///", "https://example.com/a"),
    ("https://example.com/a#frag", "https://example.com/a"),
    ("https://example.com/a?utm_source=x&b=2&a=1&fbclid=z", "https://example.com/a?a=1&b=2"),
    ("https://example.com/a?b=2&a=1", "https://example.com/a?a=1&b=2"),
    ("https://example.com/a%2fb", "https://example.com/a%2Fb"),
    ("http://dx.doi.org/10.1/ABC", "https://doi.org/10.1/abc"),
    ("https://doi.org/10.1/abc/", "https://doi.org/10.1/abc"),
    ("http://arxiv.org/pdf/2101.00001.pdf", "https://arxiv.org/abs/2101.00001"),
    ("https://arxiv.org/pdf/2101.00001v2", "https://arxiv.org/abs/2101.00001v2"),
    ("https://arxiv.org/abs/2101.00001/", "https://arxiv.org/abs/2101.00001"),
]


@pytest.mark.parametrize("raw,want", SAME)
def test_normalises(raw, want):
    assert uk.normalize_url(raw) == want
    assert uk.normalize_url(want) == want  # idempotent


def test_doi_forms_one_key():
    keys = {canonical_url({"doi": d, "url": "https://x.org/p"}) for d in
            ("10.1/X", "doi:10.1/x", " https://doi.org/10.1/X ", "http://dx.doi.org/10.1/x")}
    assert keys == {"https://doi.org/10.1/x"}


@pytest.mark.parametrize("a,b", [
    ("http://example.com/a", "https://example.com/a"),           # scheme not merged for unknown hosts
    ("https://example.com/A", "https://example.com/a"),           # path case is significant
    ("https://www.example.com/a", "https://example.com/a"),       # www. kept
    ("https://example.com/a?id=1", "https://example.com/a?id=2"),
    ("https://example.com/a?id=1", "https://example.com/a"),
    ("https://example.com/a", "https://example.com:8443/a"),
    ("https://arxiv.org/abs/2101.00001v1", "https://arxiv.org/abs/2101.00001v2"),  # versions differ
    ("https://example.com/a?ref=1", "https://example.com/a"),     # 'ref' is not a tracking parameter
    ("https://example.com/pdf/1.pdf", "https://example.com/pdf/1"),
])
def test_no_false_merge(a, b):
    assert uk.normalize_url(a) != uk.normalize_url(b)


def test_non_http_and_garbage_never_raise():
    assert uk.normalize_url("urn:isbn:123/") == "urn:isbn:123"
    assert uk.normalize_url("") == ""
    assert uk.normalize_url("http://[bad") == "http://[bad"


def test_legacy_keys_cover_old_spellings():
    src = {"url": "HTTPS://Example.com/a/"}
    assert uk.canonical_key(src) == "https://example.com/a"
    assert "HTTPS://Example.com/a" in uk.legacy_keys(src)        # old corpus_admission spelling
    assert "https://example.com/a" not in uk.legacy_keys(src)    # current key is not repeated
    assert uk.lookup_keys(src)[0] == uk.canonical_key(src)
    root = uk.legacy_keys({"url": "https://example.com/"})       # old m2 key kept '/' for the root
    assert "https://example.com/" in root


def test_old_equal_keys_unchanged():
    """A key already in the canonical lower-case spelling is identical under old and new code (existing rows)."""
    for u in ("https://arxiv.org/abs/1", "https://doi.org/10.1/x", "https://example.com/a?b=1"):
        assert uk.canonical_key({"url": u}) == u


class _Lookup:
    def __init__(self, stored):
        self.stored, self.asked = stored, []

    def rows(self, sql, params):
        self.asked.append(params["u"])
        return [(self.stored[params["u"]],)] if params["u"] in self.stored else []


def _loader(stored):
    from vera.fulltext_loader import RoLookup
    lk = RoLookup.__new__(RoLookup)
    fake = _Lookup(stored)
    lk.rows = fake.rows
    return lk, fake


def test_fulltext_lookup_finds_legacy_key_no_duplicate():
    lk, fake = _loader({"HTTPS://Example.com/a": 7})
    assert lk.canonical("https://example.com/a", {"url": "HTTPS://Example.com/a/"}) == 7
    assert fake.asked[0] == "https://example.com/a"  # new key first, then legacy


def test_fulltext_lookup_new_key_wins_and_miss_is_none():
    lk, _ = _loader({"https://example.com/a": 3, "HTTPS://Example.com/a": 7})
    assert lk.canonical("https://example.com/a", {"url": "HTTPS://Example.com/a"}) == 3
    lk2, _ = _loader({})
    assert lk2.canonical("https://example.com/a", {"url": "https://example.com/a"}) is None
