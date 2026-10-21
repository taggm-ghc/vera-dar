"""DARINT G5: vera/jats_text.py on SYNTHETIC JATS/HTML fixtures (tests/fulltext_synth.py). No DB, no network."""
import json
from pathlib import Path

import pytest

from tests import fulltext_synth as fs
from vera import jats_text as jt

CFG = json.loads((Path(__file__).resolve().parent.parent / "config" / "fulltext_ingest.json").read_text())["converter"]


def conv(raw, fmt="jats-xml"):
    return jt.convert(raw, fmt, CFG)


def test_sections_citations_and_references():
    r = conv(fs.jats("d1", "epi", "10.9999/SYN.1"))
    t = r["text"]
    kinds = [s["kind"] for s in r["sections"]]
    assert kinds[0] == "title" and "abstract" in kinds and "body" in kinds and "back" in kinds
    for s in r["sections"]:
        assert 0 <= s["start"] < s["end"] <= len(t)
        if s["title"]:
            assert t[s["start"]:].startswith(s["title"])
    assert any(s["title"] == "1. Introduction" and s["level"] == 1 for s in r["sections"])
    assert any(s["title"] == "Background" and s["level"] == 2 for s in r["sections"])
    assert r["citations"] and all(t[c["start"]:c["end"]] == c["label"] for c in r["citations"])
    assert {c["rids"][0] for c in r["citations"]} <= {"r1", "r2", "r3"}
    assert [x["doi"] for x in r["references"]] == ["10.9999/ref.1", "10.9999/ref.2", "10.9999/ref.3"]
    assert "A synthetic reference" not in t  # reference list never in the text
    assert CFG["formula_placeholder"] in t and "R_0" not in t
    assert "  " not in t and "\n\n\n" not in t
    m = r["meta"]
    assert m["doi"] == "10.9999/syn.1" and m["year"] == 2021 and m["authors"] == ["A. Synth"]
    assert m["licence"]["url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert r["has_body"] and r["body_chars"] > r["abstract_chars"] > 0


def test_deterministic():
    raw = fs.jats("d1", "clim", "10.9999/x")
    assert conv(raw) == conv(raw)


def test_abstract_only_has_no_body():
    r = conv(fs.jats("d2", "epi", "10.9999/x", with_body=False))
    assert r["has_body"] is False and r["body_chars"] == 0 and r["abstract_chars"] > 0


def test_external_doctype_without_subset_is_accepted():
    raw = fs.jats("d3", "epi", "10.9999/x",
                  extra_prolog='<!DOCTYPE article PUBLIC "-//NLM//DTD JATS (Z39.96) v1.1" "JATS-archive.dtd">')
    assert conv(raw)["has_body"]


@pytest.mark.parametrize("prolog,needle", [
    ('<!DOCTYPE article [<!ENTITY a "aaaa">]>', "entity"),
    ('<!DOCTYPE article SYSTEM "x.dtd" [ ]>', "internal subset"),
    ('<!DOCTYPE article [<!ENTITY ext SYSTEM "file:///etc/passwd">]>', "entity"),
])
def test_entity_and_subset_refused(prolog, needle):
    with pytest.raises(jt.ConversionError, match=needle):
        conv(fs.jats("d4", "epi", "10.9999/x", extra_prolog=prolog))


def test_utf16_entity_trick_refused():
    raw = fs.jats("d5", "epi", "10.9999/x", extra_prolog='<!DOCTYPE article [<!ENTITY a "b">]>')
    raw = raw.decode().replace('encoding="UTF-8"', 'encoding="UTF-16"').encode("utf-16")
    with pytest.raises(jt.ConversionError, match="UTF-16"):
        conv(raw)


def test_declared_non_utf8_refused():
    raw = fs.jats("d6", "epi", "10.9999/x").replace(b'encoding="UTF-8"', b'encoding="ISO-8859-1"')
    with pytest.raises(jt.ConversionError, match="not UTF-8"):
        conv(raw)


def test_undefined_entity_refused():
    raw = fs.jats("d7", "epi", "10.9999/x").replace(b"SYNTHETIC thanks.", b"thanks&nbsp;all")
    with pytest.raises(jt.ConversionError, match="parse error"):
        conv(raw)


def test_oversize_refused():
    small = dict(CFG, max_raw_bytes=100)
    with pytest.raises(jt.ConversionError, match="max_raw_bytes"):
        jt.convert(fs.jats("d8", "epi", "10.9999/x"), "jats-xml", small)


def test_old_expat_refused():
    with pytest.raises(jt.ConversionError, match="expat"):
        jt.convert(fs.jats("d9", "epi", "10.9999/x"), "jats-xml", dict(CFG, min_expat_version=[99, 0, 0]))


def test_non_article_root_and_unknown_format_refused():
    with pytest.raises(jt.ConversionError, match="article"):
        conv(b"<?xml version='1.0'?><html/>")
    with pytest.raises(jt.ConversionError, match="unsupported"):
        conv(b"x", "pdf")


def test_html_conversion():
    r = conv(fs.html("h1", "repro", "10.9999/h"), "html")
    t = r["text"]
    assert "var x" not in t and "menu" not in t and "foot" not in t
    assert "Synthetic ref." not in t and r["references"][0]["doi"] == "10.9999/ref.1"
    assert r["citations"] and all(t[c["start"]:c["end"]] == "[1]" for c in r["citations"])
    assert [s["kind"] for s in r["sections"]][:3] == ["title", "abstract", "body"]
    assert r["meta"]["doi"] == "10.9999/h" and r["meta"]["licence"]["url"].endswith("/by/4.0/")
    assert r["has_body"]


def test_deep_nesting_refused_not_recursion_error():
    deep = b"<?xml version='1.0'?><article><body>" + b"<sec>" * 5000 + b"x" + b"</sec>" * 5000 + b"</body></article>"
    with pytest.raises(jt.ConversionError, match="max_element_depth"):
        conv(deep)
    html = b"<html><body>" + b"<div>" * 5000 + b"x" + b"</div>" * 5000 + b"</body></html>"
    with pytest.raises(jt.ConversionError, match="max_element_depth"):
        conv(html, "html")


def test_html_implicitly_closed_tags_do_not_count_as_depth():
    paras = "".join(f"<p>SYNTHETIC para {i} transmission incidence" for i in range(1000))
    items = "".join(f"<li>item {i}" for i in range(500))
    raw = f"<html><body><article><h1>T</h1><h2>Introduction</h2>{paras}<ul>{items}</ul></article></body></html>"
    r = conv(raw.encode(), "html")
    assert "SYNTHETIC para 999" in r["text"] and "item 499" in r["text"] and r["has_body"]


def test_html_genuinely_deep_is_refused():
    raw = b"<html><body><h2>Intro</h2>" + b"<div><span>" * 300 + b"x" + b"</span></div>" * 300 + b"</body></html>"
    with pytest.raises(jt.ConversionError, match="max_element_depth"):
        conv(raw, "html")


def test_jats_drops_embedded_xhtml_script_and_style():
    raw = fs.jats("d10", "epi", "10.9999/x").replace(
        b"<p>SYNTHETIC thanks.</p>", b"<p>SYNTHETIC thanks.<script>alert(1)</script><style>b{}</style></p>")
    t = conv(raw)["text"]
    assert "alert" not in t and "b{}" not in t and "SYNTHETIC thanks." in t


def _copernicus(refs_xml: str, body: str) -> bytes:
    return (f'<?xml version="1.0" encoding="UTF-8"?><article><front><article-meta><title-group><article-title>SYNTHETIC '
            f'Copernicus-style</article-title></title-group></article-meta></front><body><sec><title>Intro</title>'
            f'<p>{body}</p></sec></body><back><ref-list>{refs_xml}</ref-list></back></article>').encode()


def test_author_year_citations_when_no_xref():
    refs = ('<ref id="bib1.bib1"><label>1</label><mixed-citation>Hawkins, E. and Sutton, R.: SYNTHETIC title, '
            'J. Synth., 2009.</mixed-citation></ref><ref id="bib1.bib2"><label>2</label><mixed-citation>Meehl, G. A., '
            'Boer, G. J.: SYNTHETIC, 1997.</mixed-citation></ref><ref id="bib1.bib3"><label>3</label><mixed-citation>'
            'Lee, A.: one, 2010.</mixed-citation></ref><ref id="bib1.bib4"><label>4</label><mixed-citation>Lee, B.: two, '
            '2010.</mixed-citation></ref>')
    body = ("SYNTHETIC text (e.g., Hawkins and Sutton, 2009) and later Meehl et al., 1997, and ambiguous Lee, 2010, "
            "and unknown Nobody, 2001.")
    r = conv(_copernicus(refs, body))
    t = r["text"]
    got = {(c["label"], c["rids"][0], c["method"]) for c in r["citations"]}
    assert got == {("Hawkins and Sutton, 2009", "bib1.bib1", "author-year"), ("Meehl et al., 1997", "bib1.bib2",
                                                                               "author-year")}
    assert all(t[c["start"]:c["end"]] == c["label"] for c in r["citations"])
    assert r["references"][0]["first_author"] == "Hawkins" and r["references"][0]["year"] == "2009"


def test_xref_citations_take_precedence_and_are_marked():
    r = conv(fs.jats("d11", "epi", "10.9999/x"))
    assert r["citations"] and {c["method"] for c in r["citations"]} == {"xref"}


def test_pmc_efetch_wrapper_unwrapped_and_multi_refused():
    one = fs.jats("d12", "epi", "10.9999/x").decode().split("?>", 1)[1]
    raw = f'<?xml version="1.0" encoding="UTF-8"?><pmc-articleset>{one}</pmc-articleset>'.encode()
    assert conv(raw)["has_body"]
    two = f'<?xml version="1.0" encoding="UTF-8"?><pmc-articleset>{one}{one}</pmc-articleset>'.encode()
    with pytest.raises(jt.ConversionError, match="exactly one"):
        conv(two)


def test_ref_html_duplicates_not_counted_and_empty_app_group_suppressed():
    refs = ('<ref id="bib1.bib1"><label>1</label><mixed-citation>Hawkins, E.: SYNTHETIC, 2009.</mixed-citation></ref>'
            '<ref-html id="bib1.bib1"><label>1</label>Hawkins, E.: SYNTHETIC, 2009.</ref-html>')
    raw = _copernicus(refs, "SYNTHETIC (Hawkins, 2009).").replace(b"<back>", b"<back><app-group></app-group>")
    r = conv(raw)
    assert len(r["references"]) == 1 and len(r["citations"]) == 1
    assert "Appendix" not in r["text"] and not any(s["kind"] == "back" for s in r["sections"])


def test_supplementary_caption_kept_media_dropped():
    raw = fs.jats("d13", "epi", "10.9999/x").replace(
        b"<ack>", b'<sec><title>Supporting information</title><supplementary-material id="s1"><label>S1 Fig</label>'
                  b'<caption><p>SYNTHETIC supplementary caption.</p></caption><media xlink:href="f.tif"/>'
                  b'</supplementary-material></sec><ack>')
    t = conv(raw)["text"]
    assert "S1 Fig" in t and "SYNTHETIC supplementary caption." in t and "f.tif" not in t


# H1-C (FDC-1020): truncated and mis-declared JATS must be refused, never half-converted.
def test_jats_truncated_at_half_length_refused():
    raw = fs.jats("d14", "epi", "10.9999/x")
    with pytest.raises(jt.ConversionError, match="XML parse error"):
        conv(raw[: len(raw) // 2])


def test_jats_truncated_inside_tag_refused():
    raw = fs.jats("d15", "epi", "10.9999/x")
    cut = raw.index(b"<article-title") + len(b"<article-ti")  # stop mid-tag name
    with pytest.raises(jt.ConversionError, match="XML parse error"):
        conv(raw[:cut])


def test_jats_bytes_declared_html_refused():
    raw = fs.jats("d16", "epi", "10.9999/x")
    with pytest.raises(jt.ConversionError, match="no headed sections"):
        conv(raw, fmt="html")
