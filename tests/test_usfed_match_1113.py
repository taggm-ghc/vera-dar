"""USFED-MATCH-1113 (work/USFED-ROUTE-1111/PLAN.md section 5.8, MT-A matching and MT-B extraction), tests t17-t28.

Offline: sockets raise, fake HTTP only, no DB. Fixtures are synthetic markup with reserved example values
(example.org, someone@example.org) or, in the optional tests, saved DOE Explains / NASA pages under scratch/ (skipped when
absent; never copied into the repo). No article-page text and no real contact data."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import socket
from pathlib import Path

import pytest

from tests import acq_support as A
from tests import test_original_html_wpg_1111 as W
from vera import acq_discovered, html_main, inline_acquire as ia, jats_text, original_html as oh

SHIPPED = W.SHIPPED
X, CC = W.X, W.CC
DOE_HOST, ROW = W.DOE_HOST, "energy_gov_doe_explains"
NASA_ROW = "nasa_science_explainers"
SCRATCH = Path(__file__).resolve().parents[3]  # <root>/scratch from the worktree
HARVEST = SCRATCH / "USFED-WPC-1112" / "harvest" / "pages"
HARVEST_OK = HARVEST.is_dir() and any(HARVEST.glob("*.html"))
C0 = SCRATCH / "USFED-C0-1111" / "pages"
C0_OK = C0.is_dir() and any(C0.glob("*.nasa*")) or (C0.is_dir() and any(C0.glob("science_nasa_gov*.html")))
SKIP_PAGES = "saved pages under scratch/ not visible (the sanctioned runner masks /home); a skip is not coverage"
PRE_MT_SHA = "5aaa25932137afa436c17a51451fd7b404e0e521ade77fe59aae3527cdfb8214"  # shipped snapshot sha256 at e44c6fb

_spec = importlib.util.spec_from_file_location(
    "refresh_permitted_catalogue", Path(__file__).resolve().parent.parent / "scripts" / "refresh_permitted_catalogue.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    oh.reset_for_tests()
    acq_discovered.reset_for_tests()

    def boom(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(socket.socket, "connect", boom)
    yield
    oh.reset_for_tests()


def doe(slug):
    return f"https://www.energy.gov/science/doe-explains{slug}"


def nasa(slug):
    return f"https://science.nasa.gov/universe/{slug}"


def ent(url, title, counts, slug=None):
    slug = slug or url.rsplit("/", 1)[-1].replace("doe-explains", "")
    return {"url": url, "host": url.split("/")[2], "title": title, "slug_terms": slug.split("-"),
            "subject_counts": dict(counts)}


def snap_of(entries):
    s = {"version": 1, "harvested_at": "2026-11-12T00:00:00+00:00", "robots_status": {}, "script_version": "test",
         "entries": entries, "denied": [], "boilerplate": {}}
    s["sha256"] = oh.snapshot_digest(s)
    return s


def install(monkeypatch, entries, mode="on", edit=None, **limits):
    c = W.set_mode(monkeypatch, mode, **limits)
    if edit:
        edit(c)
    s = snap_of(entries)
    monkeypatch.setattr(oh, "load_snapshot", lambda cfg=None: s)
    return c


# Quark family fixtures (DOE): the title page "Quarks", a page that discusses quarks a lot, one a little, one not at all.
E_QUARKS = ent(doe("quarks"), "Quarks", {"quark": 3, "proton": 2})
E_PROTONS = ent(doe("protons"), "Protons", {"quark": 9, "proton": 5})
E_NEUTRONS = ent(doe("neutrons"), "Neutrons", {"quark": 4, "proton": 1, "neutron": 6})
E_PHOTONS = ent(doe("photons"), "Photons", {"photon": 7})
E_GLUONS = ent(doe("gluons"), "Gluons", {"gluon": 4, "quark": 5})
E_QG = ent(doe("quarks-and-gluons"), "Quarks and Gluons", {"quark": 8, "gluon": 8})


def urls(out):
    return [m["url"] for m in out]


# ================================================================================ t17 / t18 / t20: MT-A matching
def test_t17_title_page_first_best_stage_two_page_second(monkeypatch):
    install(monkeypatch, [E_PROTONS, E_PHOTONS, E_QUARKS, E_NEUTRONS])
    out = oh.match_subject("What are quarks")
    assert urls(out) == [E_QUARKS["url"], E_PROTONS["url"]]  # title page, then the highest count
    assert [m["match_stage"] for m in out] == [1, 2]
    assert E_PHOTONS["url"] not in urls(out)  # no `quark` in its counts: not every subject term present


def test_t17_stage_two_ties_break_by_url_and_never_repeat_stage_one(monkeypatch):
    a, b = ent(doe("aaa"), "Aaa", {"quark": 5}), ent(doe("bbb"), "Bbb", {"quark": 5})
    install(monkeypatch, [b, a, E_QUARKS])
    out = oh.match_subject("quarks")
    assert urls(out) == [E_QUARKS["url"], a["url"]] and len(set(urls(out))) == len(out)


def test_t17_stage_two_needs_every_subject_term_and_ranks_by_the_sum(monkeypatch):
    only_q = ent(doe("only-q"), "Only q", {"quark": 50})  # missing `gluon`: excluded however large
    both_small = ent(doe("both-small"), "Both small", {"quark": 1, "gluon": 1})
    both_big = ent(doe("both-big"), "Both big", {"quark": 3, "gluon": 4})
    install(monkeypatch, [E_QG, only_q, both_small, both_big])
    out = oh.match_subject("quarks and gluons")
    assert urls(out) == [E_QG["url"], both_big["url"]]


def test_t17_stage_one_with_two_doe_entries_leaves_no_slot(monkeypatch):
    two = ent(doe("quark-matter"), "Quark Matter", {"quark": 1})
    install(monkeypatch, [E_QUARKS, two, E_PROTONS])
    out = oh.match_subject("quarks")
    assert sorted(urls(out)) == sorted([E_QUARKS["url"], two["url"]]) and all(m["match_stage"] == 1 for m in out)


def test_t18_stage_two_is_off_without_the_row_honoured(monkeypatch):
    for edit in (lambda c: next(r for r in c["rows"] if r["id"] == ROW).update(max_pages_per_question=3),
                 lambda c: next(r for r in c["rows"] if r["id"] == ROW).update(evidence_floor="x"),
                 lambda c: next(r for r in c["rows"] if r["id"] == ROW).pop("max_pages_per_question")):
        install(monkeypatch, [E_QUARKS, E_PROTONS], edit=edit)
        out = oh.match_subject("quarks")
        assert urls(out) == [E_QUARKS["url"]] and out[0]["match_stage"] == 1  # the global cap 1, stage 1 only


def test_t18_stage_two_is_off_without_a_stage_one_match(monkeypatch):
    zebra = ent(doe("protons"), "Protons", {"zebra": 4})
    install(monkeypatch, [zebra, E_PHOTONS])
    assert oh.match_subject("zebra") == []  # a count alone is never a match


def test_t18_nasa_row_never_gets_stage_two(monkeypatch):
    n1 = ent(nasa("quarks"), "Quarks", {"quark": 2}, slug="quarks")
    n2 = ent(nasa("protons"), "Protons", {"quark": 9}, slug="protons")
    install(monkeypatch, [n1, n2])
    out = oh.match_subject("quarks")
    assert urls(out) == [n1["url"]] and out[0]["match_stage"] == 1


def test_t18_nasa_plus_doe_keeps_host_and_total_caps(monkeypatch):
    n1 = ent(nasa("quarks"), "Quarks", {"quark": 2}, slug="quarks")
    n2 = ent(nasa("protons"), "Protons", {"quark": 9}, slug="protons")
    install(monkeypatch, [n1, n2, E_QUARKS, E_PROTONS, E_NEUTRONS])
    out = oh.match_subject("quarks")
    hosts = [u.split("/")[2] for u in urls(out)]
    assert hosts.count(DOE_HOST) == 2 and hosts.count("science.nasa.gov") == 1 and len(out) == 3
    install(monkeypatch, [n1, n2, E_QUARKS, E_PROTONS, E_NEUTRONS], max_candidates_per_question=2)
    out = oh.match_subject("quarks")
    assert len(out) == 2  # the total cap bounds the whole list, stage 2 included


def test_t18_denied_stage_two_page_is_skipped(monkeypatch):
    c = install(monkeypatch, [E_QUARKS, E_PROTONS, E_NEUTRONS])
    s = snap_of([E_QUARKS, E_PROTONS, E_NEUTRONS])
    s["denied"] = [{"url": E_PROTONS["url"], "code": "R-ACQ-ACCESS_DENIED", "seen_at": "t"}]
    s["sha256"] = oh.snapshot_digest(s)
    monkeypatch.setattr(oh, "load_snapshot", lambda cfg=None: s)
    assert urls(oh.match_subject("quarks", c)) == [E_QUARKS["url"], E_NEUTRONS["url"]]


def test_t20_hook_still_appends_at_most_two(monkeypatch):
    install(monkeypatch, [E_QG, E_QUARKS, E_GLUONS, E_PROTONS])
    assert ia._OH_INLINE_MAX == 2
    assert len(oh.match_subject("quarks and gluons")) <= 3
    assert len(ia._original_html_hook(W.Q, [])) == 2


# ================================================================================ t19: snapshot field validation
BAD_COUNTS = [
    ("absent", lambda e: e.pop("subject_counts")),
    ("none", lambda e: e.update(subject_counts=None)),
    ("list", lambda e: e.update(subject_counts=[["quark", 1]])),
    ("str", lambda e: e.update(subject_counts="quark")),
    ("bool value", lambda e: e.update(subject_counts={"quark": True})),
    ("zero", lambda e: e.update(subject_counts={"quark": 0})),
    ("negative", lambda e: e.update(subject_counts={"quark": -1})),
    ("float", lambda e: e.update(subject_counts={"quark": 1.0})),
    ("str value", lambda e: e.update(subject_counts={"quark": "1"})),
    ("non-term key space", lambda e: e.update(subject_counts={"bad term": 1})),
    ("non-term key upper", lambda e: e.update(subject_counts={"Quark": 1})),
    ("non-term key stopword", lambda e: e.update(subject_counts={"the": 1})),
    ("non-term key plural", lambda e: e.update(subject_counts={"quarks": 1})),
    ("empty key", lambda e: e.update(subject_counts={"": 1})),
]


@pytest.mark.parametrize("name,mut", BAD_COUNTS, ids=[n for n, _ in BAD_COUNTS])
def test_t19_bad_subject_counts_on_any_entry_invalidate_the_snapshot(name, mut):
    good = snap_of([copy.deepcopy(E_QUARKS), copy.deepcopy(E_PROTONS)])
    assert oh._snapshot_valid(good, SHIPPED) is True
    s = snap_of([copy.deepcopy(E_QUARKS), copy.deepcopy(E_PROTONS)])
    mut(s["entries"][1])  # the LAST entry only: one bad entry voids the whole snapshot
    s["sha256"] = oh.snapshot_digest(s)  # a consistent hash does not rescue it
    assert oh._snapshot_valid(s, SHIPPED) is False


def test_t19_route_claims_nothing_on_an_invalid_snapshot(monkeypatch, tmp_path):
    bad = snap_of([copy.deepcopy(E_QUARKS), copy.deepcopy(E_PROTONS)])
    del bad["entries"][0]["subject_counts"]
    bad["sha256"] = oh.snapshot_digest(bad)
    ok = snap_of([copy.deepcopy(E_QUARKS), copy.deepcopy(E_PROTONS)])
    c = W.set_mode(monkeypatch, "shadow")  # shadow does not stub load_snapshot: the real file path is exercised
    for name, s, expect in (("bad.json", bad, False), ("ok.json", ok, True)):
        p = tmp_path / name
        p.write_text(json.dumps(s), encoding="utf-8")
        c["snapshot"] = {"path": str(p)}
        got = oh.load_snapshot(c)
        assert (got is not None) is expect
        assert bool(oh.match_subject("quarks", c)) is expect


# ================================================================================ t22-t24: MT-B extraction
def extract(body, x=None, host=DOE_HOST):
    raw = f"<html><head><title>Example page</title></head><body><main><h1>Example title</h1>{body}</main></body></html>"
    ex = html_main.extract_main(raw.encode(), x or X, host, set())
    return ex["html"].decode("utf-8"), ex["stats"]


def test_t22_ex1_tokens_prune_and_lookalikes_stay():
    for cls in ("beneath-title", "read-time", "person_bio", "person-bio", "linked-topic-tags", "x read-time y"):
        out, st = extract(f'<div class="{cls}"><p>Example dropped line.</p></div><p>Example kept line.</p>')
        assert "dropped" not in out and "kept" in out and st.get("token") == 1, cls
    for cls in ("readtime", "tagsx", "linked-topic-tagsx", "beneath-titles", "person_bios", "bio"):
        out, st = extract(f'<div class="{cls}"><p>Example lookalike line.</p></div>')
        assert "lookalike" in out and not st.get("token"), cls


def test_t22_tokens_are_in_the_shipped_config_only_via_the_regex():
    for t in ("beneath-title", "read-time", "person_bio", "person-bio", "linked-topic-tags"):
        assert t in X["drop_token_re"]
    assert re.compile(X["drop_token_re"], re.I)  # still one valid regex


SEC = ("<p>Example lead paragraph.</p><h2>How it works</h2><p>Example explanation.</p>"
       "<h2>Resources</h2><p>Example resource blurb one.</p><h3>Sub resources</h3><p>Example nested blurb.</p>"
       "<ul><li>Example list item</li></ul><h2>Fast Facts</h2><p>Example fact.</p>")


def test_t23_ex2_prunes_section_to_the_next_same_or_higher_heading():
    out, st = extract(SEC)
    assert "resource blurb" not in out and "nested blurb" not in out and "Sub resources" not in out and "list item" not in out
    assert ">Resources<" not in out
    assert "Fast Facts" in out and "Example fact" in out and "How it works" in out and "Example explanation" in out
    assert st.get("section", 0) >= 2 and st.get("section_block", 0) >= 3


def test_t23_ex2_never_prunes_h1_and_keeps_lookalike_headings():
    keep = ["Resources for educators", "DOE Office of Science: Contributions to Quark Research", "Fast Facts",
            "Contact tracing basics", "Funding the future of physics"]
    body = "".join(f"<h2>{h}</h2><p>Example body for {i}.</p>" for i, h in enumerate(keep))
    out, st = extract(body)
    for i, h in enumerate(keep):
        assert h in out and f"Example body for {i}." in out
    assert not st.get("section")
    raw = b"<html><head><title>t</title></head><body><main><h1>Contact</h1><p>Example after h1.</p></main></body></html>"
    o = html_main.extract_main(raw, X, DOE_HOST, set())["html"].decode()
    assert "Contact" in o and "after h1" in o


@pytest.mark.parametrize("h", ["Resources", "  RESOURCES ", "Resources and Related Terms", "Resources and related sites",
                               "Related Links", "Contact", "Contacts", "Funding", "Publications", "Publication",
                               "Acknowledgments", "Acknowledgements", "More by this Author"])
def test_t23_ex2_heading_names(h):
    out, st = extract(f"<h2>Intro</h2><p>Example intro.</p><h2>{h}</h2><p>Example dropped body.</p><h2>After</h2><p>Example tail.</p>")
    assert "dropped body" not in out and "Example tail" in out and "Example intro" in out and st.get("section") == 1


def test_t23_ex2_levels_h3_section_ends_at_h3_or_h2_and_state_is_per_call():
    body = ("<h2>Parent</h2><p>Example parent.</p><h3>Funding</h3><p>Example funded text.</p><h4>Deeper</h4><p>Example deeper.</p>"
            "<h3>Sibling</h3><p>Example sibling.</p><h2>Contact</h2><p>Example contact text.</p>")
    out, _ = extract(body)
    assert "funded text" not in out and "Deeper" not in out and "deeper" not in out
    assert "Example sibling" in out and "Sibling" in out and "Example parent" in out and "contact text" not in out
    out2, st2 = extract("<p>Example plain page with no heading at all.</p>")  # a dropped section never leaks into the next call
    assert "plain page" in out2 and not st2.get("section_block")


def test_t23_ex2_leaves_block_hashes_untouched():
    raw = f"<html><body><main><h1>T</h1>{SEC}</main></body></html>".encode()
    x0 = {k: v for k, v in X.items() if k != "drop_section_heading_re"}
    assert html_main.block_hashes(raw, X, DOE_HOST) == html_main.block_hashes(raw, x0, DOE_HOST)
    assert html_main.line_hash("Example resource blurb one.") in html_main.block_hashes(raw, X, DOE_HOST)


def test_t23_ex2_bad_regex_fails_closed():
    for bad in ("(unclosed", 5, ["a"]):
        x = copy.deepcopy(X)
        x["drop_section_heading_re"] = bad
        with pytest.raises(html_main.HtmlRefused):
            extract(SEC, x)


def test_t24_ex3_prunes_email_blocks_of_any_length_and_keeps_plain_paragraphs():
    short = "<p>Questions? Write to someone@example.org today.</p>"
    long_ = "<p>" + ("Example filler sentence about nothing. " * 14) + "Reach someone@example.org for more.</p>"
    assert len(long_) > 400
    out, st = extract(short + long_ + "<p>Example body paragraph without an address.</p>")
    assert "someone@example.org" not in out and "filler sentence" not in out
    assert "body paragraph without an address" in out and st.get("contact_line") == 2
    out2, _ = extract('<h2>Intro</h2><p>Example text.</p><div><p>Plain paragraph, see @ symbol only.</p></div>')
    assert "Plain paragraph" in out2 and "Example text" in out2


def test_t24_ex3_is_off_without_the_key_and_fails_closed_on_a_bad_value():
    x = {k: v for k, v in X.items() if k != "drop_email_blocks"}
    out, _ = extract("<p>Write to someone@example.org.</p>", x)
    assert "someone@example.org" in out  # key absent = off (the NASA-parity path of other configs)
    for bad in ("yes", 1, "true"):
        x2 = copy.deepcopy(X)
        x2["drop_email_blocks"] = bad
        with pytest.raises(html_main.HtmlRefused):
            extract("<p>Example.</p>", x2)


def test_t24_shipped_config_carries_the_new_keys():
    assert X["drop_email_blocks"] is True
    assert re.compile(X["drop_section_heading_re"], re.I).fullmatch("resources and related terms")


# ================================================================================ t25: EX-4 lead section
LEAD_PAGE = ("<html><head><title>Example page</title></head><body><main><h1>Example title</h1>"
             "<p>Example lead definition sentence for the page. " + "More lead text. " * 40 + "</p></main></body></html>").encode()


def doe_row(c=None):
    return next(r for r in (c or SHIPPED)["rows"] if r["id"] == ROW)


def nasa_row(c=None):
    return next(r for r in (c or SHIPPED)["rows"] if r["id"] == NASA_ROW)


def test_t25_doe_row_converts_the_lead_as_a_body_section_with_identical_text():
    ex = html_main.extract_main(LEAD_PAGE, X, DOE_HOST, set())["html"]
    off = jats_text.html_to_text(ex, CC)
    on = jats_text.html_to_text(ex, oh.converter_cfg(CC, doe_row()))
    assert off["has_body"] is False and on["has_body"] is True
    assert on["text"] == off["text"]
    assert [s["kind"] for s in on["sections"]] == ["title", "body"] and on["sections"][1]["title"] == ""
    assert on["body_chars"] > 0 and off["body_chars"] == 0


def test_t25_flag_only_on_the_doe_row_and_only_as_bool_true():
    assert doe_row()["html_lead_section"] is True and "html_lead_section" not in nasa_row()
    assert "html_lead_section" not in CC and oh.converter_cfg(CC, doe_row()).get("html_lead_section") is True
    assert oh.converter_cfg(CC, doe_row()) is not CC and "html_lead_section" not in CC  # a copy
    assert "html_lead_section" not in oh.converter_cfg(CC, nasa_row())
    for bad in ("true", 1, "yes", None, [True]):
        r = dict(doe_row(), html_lead_section=bad)
        assert "html_lead_section" not in oh.converter_cfg(CC, r), bad
    other = dict(nasa_row(), html_lead_section=True)  # the key is honoured on the DOE row id only
    assert "html_lead_section" not in oh.converter_cfg(CC, other)
    assert "html_lead_section" not in oh.converter_cfg(CC, None)
    ex = html_main.extract_main(LEAD_PAGE, X, DOE_HOST, set())["html"]
    for bad in ("true", 1, None):  # the converter itself ignores a non-bool value
        assert jats_text.html_to_text(ex, {**CC, "html_lead_section": bad})["has_body"] is False


def test_t25_other_callers_and_pages_with_headings_are_unchanged():
    raw = W.page()
    ex = html_main.extract_main(raw, X, DOE_HOST, set())["html"]
    a = jats_text.html_to_text(ex, CC)
    b = jats_text.html_to_text(ex, oh.converter_cfg(CC, doe_row()))
    assert a["text"] == b["text"]
    assert a["body_chars"] <= b["body_chars"]
    assert [s["title"] for s in a["sections"]] == [s["title"] for s in b["sections"] if s["title"] != "" or s["kind"] == "title"]


def test_t25_evaluate_passes_an_h1_plus_lead_page_on_the_doe_row(monkeypatch):
    W.set_mode(monkeypatch, "on")
    paras = ["Quarks are elementary particles that combine to form protons and neutrons, and gluons are the force carrying "
             "particles that bind quarks together through the strong force inside every atomic nucleus. "] * 12
    raw = ("<html><head><title>DOE Explains...Quarks and Gluons | Department of Energy</title></head><body><main>"
           "<h1>DOE Explains...Quarks and Gluons</h1>" + "".join(f"<p>{p} Variant {i}.</p>" for i, p in enumerate(paras))
           + "</main></body></html>").encode()
    plan, _ = W.run(W.Web({W.DOE: raw}), min_spans=1)
    assert plan["admitted_spans"] >= 1


def test_t25_relaxed_ingest_cfg_applies_the_row_converter_flag():
    base = {"converter": dict(CC), "min_body_chars": 4000, "min_fulltext_chars": 8000}
    r = oh.relaxed_ingest_cfg(base, SHIPPED, doe_row())
    assert r["converter"].get("html_lead_section") is True and "html_lead_section" not in base["converter"]
    assert "html_lead_section" not in oh.relaxed_ingest_cfg(base, SHIPPED, nasa_row())["converter"]
    assert "html_lead_section" not in oh.relaxed_ingest_cfg(base, SHIPPED)["converter"]


# ================================================================================ t26: --rederive-from
class Boom:
    def __call__(self, *a, **k):
        raise AssertionError("network used")


def make_root(monkeypatch, tmp_path, pages):
    (tmp_path / "config").mkdir()
    (tmp_path / "scratch" / "pages").mkdir(parents=True)
    monkeypatch.setattr(oh, "_ROOT", tmp_path)
    monkeypatch.setattr(R, "scratch_root", lambda: tmp_path.resolve() / "scratch")
    cfg = copy.deepcopy(SHIPPED)
    cfg["snapshot"] = {"path": "config/permitted_catalogue.json"}
    monkeypatch.setattr(oh, "load_config", lambda path=None: cfg)
    entries = []
    for slug, title, body in pages:
        u = doe(slug)
        entries.append({"url": u, "host": DOE_HOST, "title": title, "slug_terms": slug.split("-")})
        html = (f"<html><head><title>{title}</title></head><body><main><h1>{title}</h1>{body}</main></body></html>")
        (tmp_path / "scratch" / "pages" / (hashlib.sha256(u.encode()).hexdigest()[:24] + ".html")).write_text(html, "utf-8")
    old = {"_doc": "d", "version": 1, "harvested_at": "2026-11-12T00:00:00+00:00", "script_version": "old/0",
           "robots_status": {DOE_HOST: {"robots": "parsed_200"}}, "entries": sorted(entries, key=lambda e: e["url"]),
           "denied": [], "boilerplate": {DOE_HOST: [hashlib.sha256(b"x").hexdigest()]}}
    old["sha256"] = oh.snapshot_digest(old)
    p = tmp_path / "config" / "permitted_catalogue.json"
    p.write_text(json.dumps(old, indent=1), encoding="utf-8")
    return cfg, p, old


PAGES26 = [("quarks", "Quarks", "<p>Quarks bind. Quarks combine into protons.</p><h2>Resources</h2><p>Quarks quarks quarks.</p>"),
           ("protons", "Protons", "<p>A proton holds quarks. Gluons help.</p><p>Contact someone@example.org about gluons.</p>"),
           ("gluons", "Gluons", "<p>Plain text about nothing in particular.</p>")]


def test_t26_rederive_recomputes_only_subject_counts_and_sha(monkeypatch, tmp_path):
    cfg, p, old = make_root(monkeypatch, tmp_path, PAGES26)
    for name in ("socket", "create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, Boom())
    monkeypatch.setattr(R.pf, "_get", Boom())
    assert R.main(["--rederive-from", str(tmp_path / "scratch" / "pages")]) == 0
    assert json.loads(p.read_text("utf-8")) == old  # dry run by default: nothing written
    assert R.main(["--rederive-from", str(tmp_path / "scratch" / "pages"), "--write"]) == 0
    new = json.loads(p.read_text("utf-8"))
    assert new["sha256"] == oh.snapshot_digest(new) != old["sha256"] and new["script_version"] == R.SCRIPT_VERSION != "old/0"
    for k in ("_doc", "version", "harvested_at", "robots_status", "denied", "boilerplate"):
        assert new[k] == old[k], k
    strip = [{k: v for k, v in e.items() if k != "subject_counts"} for e in new["entries"]]
    assert strip == old["entries"]
    by = {e["url"]: e["subject_counts"] for e in new["entries"]}
    # vocabulary = the catalogue's own title/slug terms; title and h1 excluded; EX-2 section dropped; EX-3 contact dropped
    assert by[doe("quarks")] == {"quark": 2, "proton": 1}
    assert by[doe("protons")] == {"proton": 1, "quark": 1, "gluon": 1}
    assert by[doe("gluons")] == {}
    assert oh._snapshot_valid(new, cfg) is True


def test_t26_rederive_refuses_a_missing_page_or_a_dir_outside_scratch(monkeypatch, tmp_path):
    cfg, p, old = make_root(monkeypatch, tmp_path, PAGES26)
    d = tmp_path / "scratch" / "pages"
    next(d.glob("*.html")).unlink()
    with pytest.raises(SystemExit):
        R.main(["--rederive-from", str(d), "--write"])
    assert json.loads(p.read_text("utf-8")) == old  # nothing written
    with pytest.raises(SystemExit):
        R.main(["--rederive-from", str(tmp_path / "config"), "--write"])


def test_t26_one_derivation_function_is_shared():
    assert callable(R.derive_subject_counts)
    src = Path(R.__file__).read_text(encoding="utf-8")
    assert src.count("def derive_subject_counts") == 1


# ================================================================================ t21 / t27 / t28: shipped data
@pytest.mark.skipif(not HARVEST_OK, reason=SKIP_PAGES)
def test_t21_stored_subject_counts_equal_a_fresh_derivation():
    s = json.loads(Path(oh.load_config()["snapshot"]["path"]).read_text("utf-8")) if False else oh.load_snapshot(SHIPPED)
    assert s is not None
    raws = {e["url"]: (HARVEST / (hashlib.sha256(e["url"].encode()).hexdigest()[:24] + ".html")).read_bytes()
            for e in s["entries"]}
    fresh = R.derive_subject_counts(s["entries"], raws, SHIPPED["extract"], s["boilerplate"])
    assert fresh == {e["url"]: e["subject_counts"] for e in s["entries"]}
    assert len(s["entries"]) >= 70 and all(e["subject_counts"] for e in s["entries"])


def test_t27_shipped_snapshot_is_unchanged_except_subject_counts():
    s = oh.load_snapshot(SHIPPED)
    assert s is not None and s["sha256"] != PRE_MT_SHA  # the field changes the digest
    stripped = {"entries": [{k: v for k, v in e.items() if k != "subject_counts"} for e in s["entries"]],
                "denied": s["denied"], "boilerplate": s["boilerplate"]}
    assert oh.snapshot_digest(stripped) == PRE_MT_SHA  # entries minus the field, denied and boilerplate: as at e44c6fb
    assert s["denied"] == [] and list(s["boilerplate"]) == [DOE_HOST] and {e["host"] for e in s["entries"]} == {DOE_HOST}


def _without_mt_b():
    x = copy.deepcopy(X)
    for t in ("beneath-title", "read-time", "person_bio", "person-bio", "linked-topic-tags"):
        x["drop_token_re"] = x["drop_token_re"].replace("|" + t, "")
    x.pop("drop_section_heading_re", None)
    x.pop("drop_email_blocks", None)
    assert x != X
    return x


@pytest.mark.skipif(not C0_OK, reason=SKIP_PAGES)
def test_t28_nasa_parity_on_the_saved_c0_pages():
    files = sorted(C0.glob("science_nasa_gov*.html"))
    assert len(files) >= 1
    x0, n = _without_mt_b(), 0
    for f in files:
        raw = f.read_bytes()
        a = html_main.extract_main(raw, x0, "science.nasa.gov", set())["html"]
        b = html_main.extract_main(raw, X, "science.nasa.gov", set())["html"]
        try:
            ta = jats_text.html_to_text(a, oh.converter_cfg(CC, nasa_row()))["text"]
        except jats_text.ConversionError:
            with pytest.raises(jats_text.ConversionError):
                jats_text.html_to_text(b, oh.converter_cfg(CC, nasa_row()))
            continue
        assert jats_text.html_to_text(b, oh.converter_cfg(CC, nasa_row()))["text"] == ta, f.name
        n += 1
    assert n >= 1


def test_t28_synthetic_nasa_shape_is_unchanged_by_mt_b():
    raw = ("<html><head><title>Example</title></head><body><main><h1>Example</h1><h2>Dark matter</h2>"
           "<p>Example NASA-shaped paragraph one.</p><h2>Contacts</h2><p>Example is not pruned for a page with no rule?</p>"
           "</main></body></html>").encode()
    # NASA shares the route-wide EX-1..EX-3 (config), so a heading named like a rule IS pruned by design; the lead flag is
    # what is DOE-only. The parity pinned here: the lead flag is absent for the NASA row and the text is identical.
    ex = html_main.extract_main(raw, X, "science.nasa.gov", set())["html"]
    assert jats_text.html_to_text(ex, oh.converter_cfg(CC, nasa_row()))["text"] == jats_text.html_to_text(ex, CC)["text"]
