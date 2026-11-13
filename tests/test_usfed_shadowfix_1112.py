"""USFED-SHADOWFIX-1112 (plan work/USFED-ROUTE-1111 §5.6, passes 18-21): the ORIGINAL_HTML route's THIN floor is
measured on the stored converted text only and no route refusal carries R-ABSTRACT (SF-A); 6b prunes caption, credit and
CMS banner-colour blocks without touching a title wrapper on another host (SF-B). Offline: fake HTTP, no DB. Page fixtures
reconstruct the STRUCTURE (element nesting and class names) of the harvested DOE Explains pages read in this task; the
saved real pages are used only in the optional tests at the end (skipped when the scratch copies are absent)."""
from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

from tests import acq_support as A
from tests.test_original_html_wpg_1111 import (CC, DOE, DOE_HOST, PARA, SHIPPED, X, Web, code, lines, run, set_mode)
from vera import acq_config, auto_acquire, fulltext_loader, html_main, jats_text, original_html

FLOOR = int(SHIPPED["limits"]["min_text_chars"])
NEW_TOKENS = ("field_banner_color", "caption", "attribution")
BANNER_FIELD = ('<div class="block block-layout-builder block-field-blocknodepagefield-banner-color">'
                '<div class="field field--list_default field--field_banner_color field field--name-field-banner-color">'
                '<div class="title">Banner Color</div><div class="field__item">Light Text on a Dark Overlay (Default)'
                "</div></div></div>")
CAPTION = ('<div class="block block-layout-builder block-inline-blockmedia layout-align layout">'
           '<div class="grid-container caption"><div class="grid-row"><div class="grid-col-11">'
           '<div class="caption-text">An image of particle tracks left over after a collision of two nuclei.</div>'
           '<div class="caption-attribution">Image courtesy of Example National Laboratory</div></div></div></div></div>')


def doe_page(lead, sections=(), *, before_lead="", title="DOE Explains...Quarks and Gluons"):
    """DOE Explains shape: usa-banner, header/nav, <main> with the h1, the CMS banner-colour field, a media caption block,
    then the body field: lead paragraphs BEFORE the first h2 (the definition), then h2 sections, then a footer."""
    lead_h = "".join(f"<p>{p}</p>" for p in lead)
    secs = "".join(f"<h2>{h}</h2><p>{p}</p>" for h, p in sections)
    return (f"<!doctype html><html><head><title>{title} | Department of Energy</title></head><body>"
            '<div class="usa-banner">An official website of the United States government</div>'
            '<header><nav><a href="/">Home</a><a href="/science">Office of Science</a></nav></header>'
            f'<main><article><h1>{title}</h1>{before_lead}'
            '<div class="block block-layout-builder block-inline-blockbasic">'
            f'<div class="field field--text_default field--body">{lead_h}{secs}</div></div></article></main>'
            '<footer><p>Office of Science</p><a href="/privacy">Privacy</a></footer></body></html>').encode()


def convert_shipped(raw):
    ex = html_main.extract_main(raw, X, DOE_HOST, set())
    return jats_text.html_to_text(ex["html"], CC), ex


LEAD_HEAVY = doe_page([PARA * 5, PARA * 4], [("DOE Office of Science: Contributions to Quark Research", PARA * 2)],
                      before_lead=BANNER_FIELD + CAPTION)


# ================================================================================ SF-A: one floor, one measure, one code
def test_lead_heavy_page_body_chars_below_floor_but_text_above_is_not_refused(monkeypatch):
    conv, _ = convert_shipped(LEAD_HEAVY)
    assert conv["body_chars"] < FLOOR <= len(conv["text"])  # the real DOE shape: the definition sits before the first h2
    set_mode(monkeypatch, "on")
    plan, _isbn = run(Web({DOE: LEAD_HEAVY}), min_spans=1)  # red before SF-A: fulltext_loader.Refused R-ABSTRACT
    assert plan["admitted_spans"] >= 1


def test_text_below_floor_is_thin_never_abstract(monkeypatch):
    set_mode(monkeypatch, "on")
    thin = doe_page(["Quarks are particles inside protons."], [("Facts", "Gluons bind quarks.")])
    assert code(Web({DOE: thin})) == acq_config.reason(A.cfg(), "original_thin")


def test_h1_only_page_is_no_body_not_abstract(monkeypatch):
    raw = doe_page([PARA * 10])  # the title is the only heading: converts without error, has_body false
    conv, _ = convert_shipped(raw)
    assert conv["has_body"] is False and len(conv["text"]) >= FLOOR
    set_mode(monkeypatch, "on")
    assert code(Web({DOE: raw})) == acq_config.reason(A.cfg(), "original_no_body")  # red before: R-ABSTRACT


def test_summary_heading_is_not_an_article_abstract_for_this_route(monkeypatch):
    raw = doe_page([PARA], [("Summary", PARA * 8), ("How quarks combine", PARA)])
    conv, _ = convert_shipped(raw)
    assert conv["abstract_chars"] > 0 and conv["body_chars"] < 2 * conv["abstract_chars"] and len(conv["text"]) >= FLOOR
    set_mode(monkeypatch, "on")
    # red before: the body/abstract ratio refused it as R-ABSTRACT inside the loader. After SF-A the page passes the loader
    # and reaches the relevance gate (spans in an abstract-kind section are not admitted there: a later, separate stage).
    after_loader = {acq_config.reason(A.cfg(), "irrelevant"), acq_config.reason(A.cfg(), "few_spans")}
    try:
        plan, _isbn = run(Web({DOE: raw}), min_spans=1)
        assert plan["admitted_spans"] >= 1
    except auto_acquire._Stop as e:
        assert e.code in after_loader, e.code


def test_no_route_refusal_carries_r_abstract(monkeypatch):
    set_mode(monkeypatch, "on")
    abstract = acq_config.reason(A.cfg(), "original_no_body")
    for raw in (doe_page([PARA * 10]), doe_page(["Short."], [("Facts", "Tiny.")]), LEAD_HEAVY):
        try:
            run(Web({DOE: raw}), min_spans=1)
        except auto_acquire._Stop as e:
            assert e.code != "R-ABSTRACT" and e.code in (abstract, acq_config.reason(A.cfg(), "original_thin"))
        except fulltext_loader.Refused as e:  # noqa: PERF203
            pytest.fail(f"loader refused a page the route passed: {e.code}")


def test_relaxed_cfg_parity_and_copy():
    base = fulltext_loader.load_config()
    before = copy.deepcopy(base)
    r = original_html.relaxed_ingest_cfg(base, SHIPPED)
    assert r["min_fulltext_chars"] == FLOOR  # parity: the loader's floor IS the route's floor, same measure (len(text))
    assert r["min_body_chars"] == 1 and r["min_body_to_abstract_ratio"] == 0
    assert base == before and base["min_body_chars"] == 4000 and base["min_fulltext_chars"] == 8000  # input untouched
    meta = {"has_body": True, "body_chars": 300, "abstract_chars": 0}
    with pytest.raises(fulltext_loader.Refused):
        fulltext_loader.check_content("x" * 2000, meta, base)  # every other route keeps the article floors
    fulltext_loader.check_content("x" * FLOOR, meta, r)
    with pytest.raises(fulltext_loader.Refused):
        fulltext_loader.check_content("x" * (FLOOR - 1), meta, r)  # unreachable from the route (THIN first), still safe
    with pytest.raises(fulltext_loader.Refused):
        fulltext_loader.check_content("x" * 5000, {"has_body": False, "body_chars": 0, "abstract_chars": 0}, r)


def test_manifest_item_cannot_claim_the_route_floor():
    icfg = fulltext_loader.load_config()
    meta = {"has_body": True, "body_chars": 300, "abstract_chars": 0, "converter": jats_text.CONVERTER_ID}
    with pytest.raises(fulltext_loader.Refused):
        fulltext_loader.check_content("x" * 2500, meta, icfg)  # no route= -> article floors


# ================================================================================ SF-B: 6b tokens
def test_caption_credit_and_banner_field_pruned_credit_kept_in_provenance():
    conv, ex = convert_shipped(LEAD_HEAVY)
    ls = lines(conv["text"])
    for gone in ("banner color", "light text on a dark overlay (default)", "image courtesy of example national laboratory",
                 "an image of particle tracks left over after a collision of two nuclei."):
        assert gone not in ls, gone
    assert any("quarks are elementary particles" in ln for ln in ls)  # lead kept
    assert ex["stats"].get("token", 0) >= 2
    prov = html_main.capture_provenance(html_main.parse(LEAD_HEAVY), SHIPPED["provenance"])
    assert "Image courtesy of Example National Laboratory" in prov["statements"]  # 6e reads the full page first


def test_shipped_token_regex_carries_the_three_tokens_and_not_bare_banner():
    rx = re.compile(X["drop_token_re"], re.I)
    for t in ("field--field_banner_color", "caption", "caption-attribution", "caption-text", "hds-caption"):
        assert rx.search(t), t
    for t in ("recaptioned", "captions", "hds-page-intro-banner", "hero-banner", "banner", "attributions",
              "block-field-blocknodepagefield-banner-color-x"):
        assert not rx.search(t), t


@pytest.mark.parametrize("cls", ["hds-page-intro hds-page-intro-banner", "hero-banner", "page-banner"])
def test_title_wrapper_named_banner_keeps_h1_and_lead(cls):
    lead = "Dark matter is the invisible material that makes up most of the matter in the universe. " * 3
    raw = (f"<html><head><title>Dark Matter</title></head><body><main><div class=\"{cls}\"><h1>Dark Matter</h1>"
           f"<p>{lead}</p></div><h2>What we know</h2><p>{PARA * 3}</p></main></body></html>").encode()
    conv = jats_text.html_to_text(html_main.extract_main(raw, X, "science.nasa.gov", set())["html"], CC)
    assert "Dark Matter" in conv["text"] and "invisible material" in conv["text"]


def test_body_paragraph_next_to_a_caption_is_kept():
    raw = doe_page([PARA * 3], [("Facts", PARA * 2)], before_lead=CAPTION)
    conv, _ = convert_shipped(raw)
    assert conv["text"].count("Quarks are elementary particles") >= 5 and "particle tracks" not in conv["text"]


# ================================================================================ real saved pages (optional)
SCRATCH = Path(__file__).resolve().parents[3]  # <root>/scratch from the worktree
C0 = SCRATCH / "USFED-C0-1111" / "pages"
C0_OK = C0.is_dir() and any(C0.glob("*.html"))


def _without_new_tokens():
    x = copy.deepcopy(X)
    x["drop_token_re"] = re.sub(r"\|(?:" + "|".join(NEW_TOKENS) + r")(?=[|)])", "", x["drop_token_re"])
    assert not any(t in x["drop_token_re"] for t in NEW_TOKENS)
    return x


@pytest.mark.skipif(not C0_OK, reason="saved USFED-C0-1111 pages not present (never copied into the repo)")
def test_real_c0_pages_new_tokens_drop_no_heading_and_no_nasa_text():
    x0 = _without_new_tokens()
    for f in sorted(C0.glob("*.html")):
        raw = f.read_bytes()
        a = jats_text.html_to_text(html_main.extract_main(raw, x0, None, set())["html"], CC)
        b = jats_text.html_to_text(html_main.extract_main(raw, X, None, set())["html"], CC)
        ha = [s.get("title") for s in a["sections"]]
        assert [s.get("title") for s in b["sections"]] == ha, f.name  # no heading newly pruned
        if "nasa" in f.name:
            assert b["text"] == a["text"], f.name  # no NASA line lost
        lost = set(lines(a["text"])) - set(lines(b["text"]))
        assert all(len(ln) < 400 for ln in lost), (f.name, [ln[:40] for ln in lost])  # captions/credits, not prose blocks


@pytest.mark.skipif(not (C0 / "www_energy_gov_science_doe_explainsquarks_and_gluons.html").is_file(),
                    reason="saved USFED-C0-1111 quarks page not present")
def test_real_quarks_page_is_not_refused_r_abstract(monkeypatch):
    raw = (C0 / "www_energy_gov_science_doe_explainsquarks_and_gluons.html").read_bytes()
    conv, _ = convert_shipped(raw)
    assert conv["body_chars"] < FLOOR <= len(conv["text"])  # the readout's failing shape
    set_mode(monkeypatch, "on")
    try:
        run(Web({DOE: raw}))
    except auto_acquire._Stop as e:
        assert e.code != "R-ABSTRACT", e.code
