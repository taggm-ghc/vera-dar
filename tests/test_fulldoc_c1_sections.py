"""PLAN-FDA C1: excluded section headings + paragraph labels leave evidence_ranges (synthetic bundles only).
Real-bundle checks are done offline by HARN; nothing here reads a real bundle."""
import copy
import hashlib

from tests.fulldoc_bundle_synth import MANIFEST_NAME, SUMS_NAME, sec, sha_file, synth_cfg, write_bundle
from vera import fulldoc_corpus as fc

HEADINGS = ["How to cite", "Code availability", "Supporting information"]  # the approved three (D2/F6)
LABELS = ["How to cite this article:", "How to cite:"]
P = "the transmission of outbreak infection was compared with incidence under surveillance conditions."


def cfg_c1(**over):
    c = synth_cfg()
    c.setdefault("admission", {})
    c["admission"]["excluded_section_headings"] = HEADINGS
    c["admission"]["excluded_paragraph_labels"] = LABELS
    c["admission"].update(over)
    return c


def _load(tmp_path, secs, cfg=None, doc="syn_epi"):
    b = write_bundle(tmp_path, body_sections={doc: secs})
    c = fc.load_bundle(b, cfg or cfg_c1())
    assert c.status == "ok" and not c.unusable
    return c, next(a for a in c.acquired if a["document_id"] == doc)


def _admissible(a, text, needle):
    i = text.index(needle)
    return any(r[0] <= i and i + len(needle) <= r[1] for r in a["evidence_ranges"])


def test_heading_excluded_and_other_sections_stay(tmp_path):
    secs = sec("Results", [P + " RESULTSMARK"]) + sec("Code availability", ["CODEAVAILMARK see the repository."]) \
        + sec("Discussion", ["DISCMARK " + P])
    c, a = _load(tmp_path, secs)
    t = c.text("syn_epi")
    assert not _admissible(a, t, "CODEAVAILMARK") and not _admissible(a, t, "Code availability")
    assert _admissible(a, t, "RESULTSMARK") and _admissible(a, t, "DISCMARK")
    assert any(x["rule"] == "heading" and x["reason"] == "excluded_section" for x in a["excluded_ranges"])


def test_numbered_heading_variant(tmp_path):
    secs = sec("Code availability", ["NUMMARK code."], label="7.") + sec("Results", ["KEEPMARK " + P])
    c, a = _load(tmp_path, secs)
    assert "7. Code availability" in c.text("syn_epi")
    assert not _admissible(a, c.text("syn_epi"), "NUMMARK") and _admissible(a, c.text("syn_epi"), "KEEPMARK")


def test_heading_normalisation_case_space_punct_roman(tmp_path):
    secs = sec("SUPPORTING   Information:", ["SIMARK x"]) + sec("IV. how to cite.", ["HTCMARK y"]) \
        + sec("Results", ["KEEPMARK " + P])
    c, a = _load(tmp_path, secs)
    t = c.text("syn_epi")
    assert not _admissible(a, t, "SIMARK") and not _admissible(a, t, "HTCMARK") and _admissible(a, t, "KEEPMARK")


def test_heading_exact_match_only_no_variants(tmp_path):  # D10 pending: variants are NOT excluded
    secs = sec("Code and data availability", ["VARMARK1 x"]) + sec("Availability of code", ["VARMARK2 x"]) \
        + sec("Supporting Information files", ["VARMARK3 x"])
    c, a = _load(tmp_path, secs)
    t = c.text("syn_epi")
    assert all(_admissible(a, t, m) for m in ("VARMARK1", "VARMARK2", "VARMARK3"))
    assert a["excluded_ranges"] == []


def test_nested_subsections_leave_with_parent_and_sibling_stays(tmp_path):
    kid = sec("Repository details", ["KIDMARK x"], children=sec("Deep", ["DEEPMARK x"]))
    secs = sec("Supporting information", ["PARENTMARK x"], children=kid) + sec("Results", ["AFTERMARK " + P])
    c, a = _load(tmp_path, secs)
    t = c.text("syn_epi")
    for m in ("PARENTMARK", "KIDMARK", "DEEPMARK"):
        assert not _admissible(a, t, m), m
    assert _admissible(a, t, "AFTERMARK")


def test_nested_subsection_of_other_section_with_excluded_name_only_that_part(tmp_path):
    secs = sec("Results", ["RESMARK " + P], children=sec("Code availability", ["SUBMARK x"])) \
        + sec("Discussion", ["DISCMARK " + P])
    c, a = _load(tmp_path, secs)
    t = c.text("syn_epi")
    assert _admissible(a, t, "RESMARK") and not _admissible(a, t, "SUBMARK") and _admissible(a, t, "DISCMARK")


def test_label_removes_to_end_of_enclosing_section_only(tmp_path):
    secs = sec("Results", ["BEFOREMARK " + P, "How to cite this article: LABELMARK Doe J. Title. J. 2021.", "AFTERLABELMARK x"]) \
        + sec("Discussion", ["NEXTSECMARK " + P])
    c, a = _load(tmp_path, secs)
    t = c.text("syn_epi")
    assert _admissible(a, t, "BEFOREMARK") and _admissible(a, t, "NEXTSECMARK")
    for m in ("How to cite this article", "LABELMARK", "AFTERLABELMARK"):
        assert not _admissible(a, t, m), m


def test_label_blank_line_then_citation_separate_paragraphs(tmp_path):
    secs = sec("Results", ["BEFOREMARK " + P, "How to cite this article:", "CITEMARK Doe J. Title. J. 2021;1:1."])
    c, a = _load(tmp_path, secs)
    t = c.text("syn_epi")
    assert _admissible(a, t, "BEFOREMARK") and not _admissible(a, t, "CITEMARK")


def test_label_inside_nested_subsection_ends_at_that_subsection(tmp_path):
    secs = sec("Results", ["OUTERMARK " + P], children=sec("Notes", ["How to cite: INNERMARK x"])) \
        + sec("Discussion", ["DISCMARK " + P])
    c, a = _load(tmp_path, secs)
    t = c.text("syn_epi")
    assert _admissible(a, t, "OUTERMARK") and not _admissible(a, t, "INNERMARK") and _admissible(a, t, "DISCMARK")


def test_guard_in_body_prose_how_to_cite_stays_admissible(tmp_path):
    secs = sec("Methods", ["GUARD1MARK Readers ask how to cite this article: the answer is in the journal footer. " + P])
    c, a = _load(tmp_path, secs)
    assert _admissible(a, c.text("syn_epi"), "GUARD1MARK") and a["excluded_ranges"] == []


def test_guard_please_cite_this_work_as_and_supplementary_materials(tmp_path):
    secs = sec("Methods", ["Please cite this work as: GUARD2MARK Doe J. A CITATION file example."]) \
        + sec("Supplementary materials", ["GUARD3MARK " + P])
    c, a = _load(tmp_path, secs)
    t = c.text("syn_epi")
    assert _admissible(a, t, "GUARD2MARK") and _admissible(a, t, "GUARD3MARK") and a["excluded_ranges"] == []


def test_empty_or_absent_keys_change_nothing(tmp_path):
    secs = sec("Code availability", ["CODEAVAILMARK x"]) + sec("Results", ["How to cite: CITEMARK y"])
    b = write_bundle(tmp_path, body_sections={"syn_epi": secs})
    base = synth_cfg()
    for key in ("excluded_section_headings", "excluded_paragraph_labels"):
        base.get("admission", {}).pop(key, None)
    c = fc.load_bundle(b, base)
    a = next(x for x in c.acquired if x["document_id"] == "syn_epi")
    assert _admissible(a, c.text("syn_epi"), "CODEAVAILMARK") and _admissible(a, c.text("syn_epi"), "CITEMARK")
    assert a["excluded_ranges"] == []


def test_text_and_pins_unchanged_by_exclusions(tmp_path):
    secs = sec("Code availability", ["CODEAVAILMARK x"]) + sec("Results", ["How to cite: CITEMARK y"])
    b = write_bundle(tmp_path, body_sections={"syn_epi": secs})
    pins = {"manifest_sha256": sha_file(b / MANIFEST_NAME), "sums_sha256": sha_file(b / SUMS_NAME)}
    off = synth_cfg(pins)
    on = cfg_c1()
    on["pins"] = dict(pins)
    c_off, c_on = fc.load_bundle(b, off), fc.load_bundle(b, on)
    assert c_on.status == "ok"
    for x, y in zip(c_off.acquired, c_on.acquired):
        assert x["text_sha256"] == y["text_sha256"] and x["chars"] == y["chars"]
        assert x["text_sha256"] == hashlib.sha256(c_on.text(x["document_id"]).encode()).hexdigest()
    assert [u for u in c_on.unusable] == []


def test_excluded_ranges_recorded_with_reason_and_bounds(tmp_path):
    secs = sec("Supporting information", ["SIMARK x"]) + sec("Results", ["How to cite: CITEMARK y"])
    c, a = _load(tmp_path, secs)
    t = c.text("syn_epi")
    assert {x["rule"] for x in a["excluded_ranges"]} == {"heading", "label"}
    for x in a["excluded_ranges"]:
        assert x["reason"] == "excluded_section" and 0 <= x["start"] < x["end"] <= len(t)
    assert a["excluded_ranges"][0]["match"]


def test_retrieve_evidence_only_filters_before_topk(tmp_path):
    secs = sec("Code availability", ["transmission incidence outbreak infection CODEMARK " * 6])
    b = write_bundle(tmp_path, body_sections={"syn_epi": secs})
    c = fc.load_bundle(b, cfg_c1())
    q = "transmission incidence outbreak infection CODEMARK"
    plain = c.retrieve(q, 3)
    assert any("CODEMARK" in s["text"] for s in plain)  # without the filter the excluded chunk takes a slot
    ev = c.retrieve(q, 3, evidence_only=True)
    # filtered BEFORE top-k: the per-document slots the excluded chunks held are refilled with evidence chunks
    assert len(ev) > len([x for x in plain if "CODEMARK" not in x["text"] and x["document_id"] == "syn_epi"])
    assert all(s["in_evidence"] for s in ev) and not any("CODEMARK" in s["text"] for s in ev)
    assert all(s["text"] == c.text(s["document_id"])[s["start"]:s["end"]] for s in ev)
    assert c.retrieve(q, 3) == plain  # default unchanged and independent of the evidence index


def test_retrieve_evidence_only_drops_abstract_and_title_chunks(tmp_path):
    c = fc.load_bundle(write_bundle(tmp_path), cfg_c1())
    for s in c.retrieve("SYNTHETIC study sentence transmission", 12, evidence_only=True):
        a = next(x for x in c.acquired if x["document_id"] == s["document_id"])
        assert any(r[0] <= s["start"] and s["end"] <= r[1] for r in a["evidence_ranges"])


def test_get_corpus_none_when_absent_fail_closed(tmp_path):
    fc.reset_for_tests()
    b = write_bundle(tmp_path)
    cfg = synth_cfg()
    env = {cfg["bundle_dir_env"]: str(b)}
    cfg.pop("fulldoc", None)
    assert fc.get_corpus(cfg, env=env) is None  # key absent => disabled (fail closed, D-ON-4 / SEC5)
    cfg2 = copy.deepcopy(cfg)
    cfg2["fulldoc"] = {"enabled": True}
    assert fc.get_corpus(cfg2, env=env) is not None
    cfg3 = copy.deepcopy(cfg)
    cfg3["fulldoc"] = {"enabled": False}
    assert fc.get_corpus(cfg3, env=env) is None
    fc.reset_for_tests()


def test_guard_data_availability_sections_stay_admissible(tmp_path):  # D10 = B: no variants, "data"/"availability" is not a signal
    secs = sec("Data availability", ["DATAMARK " + P]) + sec("Code and data availability", ["CDMARK " + P])
    c, a = _load(tmp_path, secs)
    t = c.text("syn_epi")
    assert _admissible(a, t, "DATAMARK") and _admissible(a, t, "CDMARK") and a["excluded_ranges"] == []


def test_guard_availability_of_code_stays_admissible(tmp_path):  # D10 = B: variant, not excluded
    c, a = _load(tmp_path, sec("Availability of code", ["AOCMARK " + P]))
    assert _admissible(a, c.text("syn_epi"), "AOCMARK") and a["excluded_ranges"] == []


def test_guard_supporting_information_files_stays_admissible(tmp_path):  # D10 = B: only exactly "Supporting information"
    c, a = _load(tmp_path, sec("Supporting Information files", ["SIFMARK " + P]))
    assert _admissible(a, c.text("syn_epi"), "SIFMARK") and a["excluded_ranges"] == []
