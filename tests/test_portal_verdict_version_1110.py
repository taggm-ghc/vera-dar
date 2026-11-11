"""PVFIX-A-1110: portal_verdict passes the item's own version/type to the version-gated repository rows (offline)."""
import pytest

from tests import host_role_support as HR
from vera import fulltext_loader as FL
from vera import host_role

ZEN = "https://zenodo.org/records/1234567"
ZENF = "https://zenodo.org/records/1234567/files/paper.pdf"


@pytest.fixture(autouse=True)
def _roles():
    HR.install()
    yield
    HR.reset()


def zen(**kw):
    return dict({"url": ZEN, "fulltext_url": ZENF, "doi": "10.5281/zenodo.1234567"}, **kw)


@pytest.mark.parametrize("item", [
    zen(version="publishedVersion"),
    zen(version="vor"),
    zen(version="info:eu-repo/semantics/publishedVersion"),
    {"url": "https://hal.science/hal-01234567", "version": "published"},
    {"url": "https://dspace.mit.edu/handle/1721.1/12345", "version": "publishedVersion"},
    {"url": "https://eprints.soton.ac.uk/12345/", "version": "publishedVersion"},
    zen(version="", type="dataset"),
    zen(version="", type="report"),
])
def test_published_or_nonarticle_serves(item):
    assert FL.portal_verdict(item) is None


def test_openreview_published_is_original_and_serves():
    item = {"url": "https://openreview.net/forum?id=abc", "version": "published"}
    assert FL.portal_verdict(item) is None
    assert host_role.role_for_url(item["url"], {"version": "published"}).role == "ORIGINAL"
    assert FL.portal_verdict({"url": "https://openreview.net/forum?id=abc"}).startswith("version_unqualified:")


@pytest.mark.parametrize("ver", ["acceptedVersion", "accepted", "camera-ready", "cameraready",
                                 "info:eu-repo/semantics/acceptedVersion", "submittedVersion", "info:eu-repo/semantics/submittedVersion",
                                 "preprint", "draft", "unknown", "none", "", "v1.0", "publishedversio", "published version"])
@pytest.mark.parametrize("base", [ZEN, "https://dspace.mit.edu/handle/1721.1/12345", "https://hal.science/hal-01234567"])
def test_other_versions_are_quarantined(base, ver):
    out = FL.portal_verdict({"url": base, "version": ver, "type": "article"})
    assert out and out.startswith("version_unqualified:")


def test_missing_version_article_type_and_no_meta_quarantined():
    assert FL.portal_verdict(zen()).startswith("version_unqualified:")
    assert FL.portal_verdict(zen(type="journal-article")).startswith("version_unqualified:")


@pytest.mark.parametrize("typ", ["preprint", "posted-content", "Preprint"])
def test_preprint_type_wins_over_published_version(typ):
    assert FL.portal_verdict(zen(version="publishedVersion", type=typ))  # PORTAL by the record-type or version rule


def test_version_in_unrelated_field_does_not_count():
    assert FL.portal_verdict({"url": "https://example.org/x", "title": "publishedVersion"}, canonical_url=ZEN).startswith("version_unqualified:")
    assert FL.portal_verdict({"url": "https://example.org/x"}, {"notes": {"version": "publishedVersion"}}, canonical_url=ZEN).startswith("version_unqualified:")
    # but a stored row's provenance version is read
    assert FL.portal_verdict({"url": "https://example.org/x"}, {"version": "publishedVersion"}, canonical_url=ZEN) is None


def test_gated_host_via_any_url_field_with_unqualified_version():
    assert FL.portal_verdict({"url": "https://example.org/x", "fulltext_url": ZENF, "version": "submittedVersion"}).startswith("version_unqualified:")


@pytest.mark.parametrize("item,rule_prefix", [
    ({"url": "https://arxiv.org/abs/2101.00001"}, None),
    ({"url": "https://www.biorxiv.org/content/10.1101/2020.01.01.123456v1"}, None),
    ({"url": "https://www.ncbi.nlm.nih.gov/pubmed/1"}, None),
    ({"url": "https://example.org/x", "doi": "10.48550/arxiv.2101.00001"}, None),
    ({"url": "https://example.org/x", "source": "PPR"}, None),
])
def test_non_gated_portals_unchanged_by_version(item, rule_prefix):
    before = FL.portal_verdict(item)
    after = FL.portal_verdict(dict(item, version="publishedVersion"))
    assert before and before == after


def test_shadow_library_with_version_still_excluded():
    out = FL.portal_verdict({"url": "https://sci-hub.se/10.1000/x", "version": "publishedVersion"})
    assert out
    assert host_role.role_for_url("https://sci-hub.se/10.1000/x", {"version": "publishedVersion"}).role == "EXCLUDED"


def test_accepted_versions_config_is_published_only():
    reg = host_role.get_registry()
    assert reg.rv["accepted_versions"] == {"published", "publishedversion", "vor"}


@pytest.mark.parametrize("ver", [1, 0, True, False, ["publishedVersion"], {"v": "published"}, "info:eu-repo/semantics/", "info:eu-repo/semantics/  "])
@pytest.mark.parametrize("where", ["item", "prov"])
def test_s1_undecidable_version_is_not_empty_even_for_dataset(ver, where):
    item = zen(type="dataset")
    prov = {}
    if where == "item":
        item["version"] = ver
    else:
        prov["version"] = ver
    out = FL.portal_verdict(item, prov)
    assert out and out.startswith("version_unqualified:")


def test_trailing_space_published_is_trimmed_and_serves():
    assert FL.portal_verdict(zen(version="PublishedVersion ")) is None
