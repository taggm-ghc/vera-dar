"""NONASK-1103 COMP contract for stream C: an APA reference ends with the licence after the identifier where known."""
from vera import apa7
from vera.auto_acquire import tasl_attribution
from vera.references_ui import csl_from_source

LIC_URL = "https://creativecommons.org/licenses/by/4.0/"


def _src(**kw):
    s = {"title": "Batching study", "url": "https://example.org/a", "authors": ["Ann Lee"], "published": "2024",
         "doi": "10.1234/abc", "source_type": "journal-article", "container_title": "J. Test"}
    s.update(kw)
    return s


def test_licence_after_identifier_from_attribution():
    s = _src(attribution=tasl_attribution("Batching study", "Ann Lee", "https://doi.org/10.1234/abc", "CC BY 4.0", LIC_URL))
    ref = apa7.reference(csl_from_source(s))
    assert ref.endswith(f"Licence: [{apa7._p('CC BY 4.0')}]({LIC_URL}).")
    assert ref.index("doi.org/10.1234/abc") < ref.index("Licence:")
    plain = apa7.reference(csl_from_source(s), links=False)
    assert plain.endswith(f"Licence: {apa7._p('CC BY 4.0')} ({LIC_URL}).")


def test_no_licence_adds_nothing():
    assert "Licence" not in apa7.reference(csl_from_source(_src()))
    assert "Licence" not in apa7.reference(csl_from_source(_src(licence="cc-by")))  # no URL, no attribution: fail soft


def test_unsafe_licence_url_dropped():
    s = _src(attribution={"licence": "CC BY 4.0", "licence_url": "javascript:alert(1)"})
    ref = apa7.reference(csl_from_source(s))
    assert ref.endswith(f"Licence: {apa7._p('CC BY 4.0')}.") and "javascript" not in ref
