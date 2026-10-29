"""IMPL-1028-E1FA7: E1 term-concept groups, WP-FA7 per-part allocation, topical_fit order mode, generic_terms_v2.
SYNTHETIC passages only (boxing/MMA cases mimic HYP-1028 H3-H5 and REPLAY-1028 b09); no network, DB or env."""
import copy
import json
from pathlib import Path

from vera import fulldoc_admission as fa
from vera import search_query as sq
from vera import term_expansion as te

ROOT = Path(__file__).resolve().parent.parent
FD = json.loads((ROOT / "config" / "fulldoc_answer.json").read_text())
GROUPS = te.load(str(ROOT / "config" / "term_expansions.json"))
Q = "What are key differences and commonalities between boxing and mixed martial arts?"
QA = "What are key differences and commonalities between boxing and MMA?"
QM = "What do raised MMA levels mean in vitamin B12 deficiency?"
ALIAS = "Boxing and MMA fighters suffer head impacts in training and competition."
SPELLED = "Boxing and mixed martial arts fighters suffer head impacts in training and competition."
MED = ["Methylmalonic acid (MMA) levels rise in vitamin B12 deficiency and are measured in serum.",
       "Raised MMA with low B12 indicates deficiency; MMA testing is common in clinics."]
B09 = ("Officers trained in mixed martial arts techniques reported fewer use-of-force injuries during arrests than "
       "officers trained in baton tactics.")


def doc(i, title):
    return {"document_id": i, "canonical_source_id": f"https://x/{i}", "title": title, "url": f"https://x/{i}",
            "doi": None, "published": "2020", "authors": ["A"], "licence": "CC-BY",
            "licence_url": "https://creativecommons.org/licenses/by/4.0/", "licence_decision": "allow",
            "attribution": "a", "text_sha256": "h", "chars": 100, "evidence_ranges": [[0, 10 ** 9]]}


class Corpus:
    status = "ok"

    def __init__(self, docs):
        self.acquired = [doc(i, t) for i, (t, _) in docs.items()]
        self.full, self.cands = {}, []
        for i, (_t, spans) in docs.items():
            pos, txt = 0, ""
            for j, s in enumerate(spans):
                self.cands.append({"span_id": f"{i}:{j}", "document_id": i, "start": pos, "end": pos + len(s),
                                   "text": s, "score": 1.0 - 0.01 * j})
                txt += s + " "
                pos += len(s) + 1
            self.full[i] = txt

    def text(self, i):
        return self.full[i]

    def retrieve(self, q, k):
        return list(self.cands)

    def inspection(self):
        return {}


def cfg(**on):
    c = copy.deepcopy(FD)
    a = c["admission"]
    for k, v in on.items():
        a[k] = dict(a.get(k) or {}, **v)
    return c


def run(q, docs, c):
    return fa.admit(q, Corpus(docs), c)


def view(b):
    return ([(a["span_id"], round(a["relevance"], 6)) for a in b.admitted], b.trace, b.rejected, b.parts_unsupported)


SCEN = {
    "alias": (Q, {"d1": ("Combat sports", [ALIAS, ALIAS + " More."])}),
    "aliasq": (QA, {"d1": ("Combat sports", [ALIAS, SPELLED])}),
    "spelled": (Q, {"d1": ("Combat sports", [SPELLED, SPELLED + " More."])}),
    "med": (QM, {"m1": ("B12", MED), "m2": ("Fight", [ALIAS])}),
}


# ---- defaults reproduce current output (golden: switches absent == explicitly off) ----
def test_shipped_switches_are_off():
    a = FD["admission"]
    assert a["term_groups"]["enabled"] is False and a["part_allocation"]["enabled"] is False
    assert a["topical_fit"]["order"] is False and a["generic_terms_v2"]["enabled"] is False
    assert a["topical_fit"]["fit_lo"] is None and a["topical_fit"]["fit_hi"] is None
    assert a["max_spans_total"] == 6  # unchanged by this work package
    for k in ("term_groups", "part_allocation", "topical_fit", "generic_terms_v2"):
        assert "UNCALIBRATED" in a[k]["_calibration_note"]


def test_defaults_identical_to_absent_keys():
    absent = copy.deepcopy(FD)
    for k in ("term_groups", "part_allocation", "topical_fit", "generic_terms_v2"):
        absent["admission"].pop(k)
    for signal in ("legacy", "idf_weighted"):
        for q, docs in SCEN.values():
            c1, c2 = copy.deepcopy(FD), copy.deepcopy(absent)
            c1["admission"]["signal"] = c2["admission"]["signal"] = signal
            b1, b2 = run(q, docs, c1), run(q, docs, c2)
            assert repr(b1) == repr(b2) and b1.parts_unsupported == ()


def test_key_concepts_default_unchanged():
    for q in (Q, QA, QM):
        assert sq.key_concepts(q) == sq.key_concepts(q, groups=())
        assert sq.query_variants(q) == []


# ---- E1 term groups ----
def test_seed_groups_and_ambiguous_flag():
    mma = next(g for g in GROUPS if g.id == "mixedmartialarts")
    assert ("mma", True) in mma.phrases and mma.spelled == "mixed martial arts" and mma.token == "xgmixedmartialarts"
    assert te.parse({"groups": [{"id": "a1", "phrases": [{"text": "x y"}, {"text": "xy"}]}]}) == ()  # bad id
    dup = {"groups": [{"id": "aaa", "phrases": [{"text": "foo"}, {"text": "bar"}]},
                      {"id": "bbb", "phrases": [{"text": "foo"}, {"text": "baz"}]}]}
    assert te.parse(dup) == ()  # a phrase in two groups fails closed


def test_weighted_group_token_preserves_spelled_out_arithmetic():
    on = cfg(term_groups={"enabled": True})
    icfg, n = fa._routing(on)["identification"], 5
    tg = fa._term_ctx(Q, on, icfg, n)
    assert tg is not None and tg.w == {"xgmixedmartialarts": 3}
    # today: spelled 4/5; alias 1/5 (H5). Grouped + weighted: both 4/5, denominator still 5.
    assert fa._relevance(Q, SPELLED, icfg, n) == fa._relevance(Q, SPELLED, icfg, n, tg) == 0.8
    assert fa._relevance(Q, ALIAS, icfg, n) == 0.2 and fa._relevance(Q, ALIAS, icfg, n, tg) == 0.8
    # F2: the floor counts a group as the spelled-out stems really present (alias alone = 1 hit), never its weight
    assert fa._matched(Q, ALIAS, icfg, n, tg) == (2, 5) and fa._matched(Q, SPELLED, icfg, n, tg) == fa._matched(Q, SPELLED, icfg, n)


def test_alias_span_admitted_only_when_on():
    q, docs = SCEN["alias"]
    assert not run(q, docs, FD).admitted
    b = run(q, docs, cfg(term_groups={"enabled": True}))
    assert [a["span_id"] for a in b.admitted] == ["d1:0", "d1:1"] and b.trace["term_groups"] == 1
    q, docs = SCEN["spelled"]  # spelled-out form: identical admission with the switch on
    assert view(run(q, docs, FD))[0] == view(run(q, docs, cfg(term_groups={"enabled": True})))[0]


def test_alias_question_matches_spelled_span():
    q, docs = SCEN["aliasq"]
    b = run(q, docs, cfg(term_groups={"enabled": True}))
    assert {a["span_id"] for a in b.admitted} == {"d1:0", "d1:1"}


def test_ambiguous_alias_guard_medical_mma():
    on = cfg(term_groups={"enabled": True})
    assert te.triggered(QM, GROUPS) == ()
    assert fa._term_ctx(QM, on, fa._routing(on)["identification"], 5) is None
    q, docs = SCEN["med"]
    assert repr(run(q, docs, on)) == repr(run(q, docs, FD))  # no expansion: byte-identical output
    assert sq.query_variants(QM, GROUPS) == [] and sq.key_concepts(QM, groups=GROUPS) == sq.key_concepts(QM, groups=())
    # a cue (boxing, UFC, fighters ...) or the spelled-out form makes the alias count
    assert te.triggered("Is MMA more dangerous than boxing?", GROUPS)
    assert te.triggered("mixed martial arts (MMA) injuries", GROUPS)
    assert not te.triggered("MMA polymer bone cement fatigue", GROUPS)


def test_key_concepts_and_variants_with_groups():
    kc = sq.key_concepts(QA, groups=GROUPS)
    assert "mixed martial arts|mma" in kc[0] and "mma" not in kc[0] and "boxing" in kc[0]
    assert "mixed martial arts|mma" in sq.matched_concepts(kc, "Injuries among MMA fighters")
    assert "mixed martial arts|mma" in sq.matched_concepts(kc, "Mixed martial arts rules")
    assert sq.query_variants(QA, GROUPS) == ["mixed martial arts"]
    assert sq.query_variants(Q, GROUPS) == ["mma"]


# ---- WP-FA7 per-part allocation ----
A_SPANS = [f"Boxing differs from mixed martial arts in rule {i}, unlike the other sport." for i in range(10)]
B_SPANS = ["Both boxing and martial arts share rounds and weight classes.",
           "Boxing and martial arts gyms share conditioning drills."]


def test_part_allocation_gives_starved_part_a_slot():
    docs = {"A": ("Differences", A_SPANS), "B": ("Commonalities", B_SPANS)}
    off = run(Q, docs, FD)
    assert {a["document_id"] for a in off.admitted} == {"A"} and len(off.admitted) == 6  # H3: part starved
    b = run(Q, docs, cfg(part_allocation={"enabled": True}))
    by = [a["document_id"] for a in b.admitted]
    assert len(by) == 6 and by.count("B") >= 1 and b.parts_unsupported == ()
    assert b.trace["parts"]["n"] == 4 and b.trace["parts"]["quota"] == 1  # boxing, mixed martial arts, contrast, common
    rel = [a["relevance"] for a in b.admitted]
    assert rel == sorted(rel, reverse=True)  # admitted stays in rank order
    assert all(a["relevance"] >= FD["admission"]["min_relevance"] for a in b.admitted)


def test_part_with_no_passage_is_reported():
    b = run(Q, {"A": ("Differences", A_SPANS)}, cfg(part_allocation={"enabled": True}))
    assert b.parts_unsupported == ("common",) and b.trace["parts"]["unsupported"] == 1
    assert len(b.admitted) == len(run(Q, {"A": ("Differences", A_SPANS)}, FD).admitted)


def test_weak_cue_offtopic_passage_not_promoted():  # REPLAY-1028 b09 ("than")
    on = cfg(part_allocation={"enabled": True})
    parts = fa._alloc_parts(Q, on["admission"]["part_allocation"], fa._routing(on)["identification"], None)
    assert [p.label for p in parts if p.kind == "facet"] == ["contrast", "common"]
    fits = dict(zip([p.label for p in parts], fa._fits(B09, parts)))
    assert fits["contrast"] is False and fits["common"] is False
    docs = {"A": ("Differences", A_SPANS[:6]), "C": ("Policing", [B09])}
    assert [a["span_id"] for a in run(Q, docs, on).admitted] == [a["span_id"] for a in run(Q, docs, FD).admitted]


def test_single_facet_question_not_split_and_floor_never_bypassed():
    q = "What are the differences between boxing and kickboxing?"
    on = cfg(part_allocation={"enabled": True, "concepts": False})
    assert fa._alloc_parts(q, on["admission"]["part_allocation"], fa._routing(on)["identification"], None) == []
    docs = {"A": ("x", A_SPANS), "L": ("low", ["Unlike rowing, both share nothing about weather."])}
    b = run(Q, docs, cfg(part_allocation={"enabled": True}))
    assert all(a["document_id"] != "L" for a in b.admitted)  # a cue never admits a span below the relevance floor


# ---- topical_fit order mode ----
H_SPANS = ["Herpes infection outbreak in boxing gyms: herpes lesions, herpes transmission and infection control.",
           "Commonalities and differences: boxing and mixed martial arts gyms both had herpes infection outbreaks."]
G_SPANS = ["Boxing and mixed martial arts differ in rules; boxing and mixed martial arts fighters train striking."]


def test_topical_fit_unset_is_noop_and_set_reorders():
    docs = {"H": ("Herpes in gyms", H_SPANS), "G": ("Boxing vs MMA", G_SPANS)}
    off = run(Q, docs, FD)
    unset = run(Q, docs, cfg(topical_fit={"order": True}))
    assert view(off)[0] == view(unset)[0] and unset.trace["topical_fit"] == "unset"
    assert off.admitted[0]["document_id"] == "H"
    b = run(Q, docs, cfg(topical_fit={"order": True, "fit_lo": 0.2, "fit_hi": 0.5}))
    assert b.admitted[0]["document_id"] == "G" and {a["span_id"] for a in b.admitted} == {a["span_id"] for a in off.admitted}


# ---- generic_terms_v2 (DIAG-ANS3 fix 6; "share" deliberately excluded) ----
def test_generic_terms_v2():
    a = FD["admission"]["generic_terms_v2"]
    assert a["add"] == ["commonalities", "commonality", "similarity"] and "share" not in a["add"]
    p1 = "Head impacts are common in boxing."
    assert run(Q, {"d1": ("x", [p1])}, FD).admitted  # H4 baseline: admitted on the "commo" stem
    assert not run(Q, {"d1": ("x", [p1])}, cfg(generic_terms_v2={"enabled": True})).admitted


# ---- revision 2 (REVIEW-1028-E1FA7 F1-F3) ----
def test_f1_generic_cues_do_not_defeat_the_guard():
    for q in ("MMA cage pressure in PMMA cages", "Are MMA levels raised in combat veterans?", "What sport raises MMA?",
              "MMA", "What is MMA?", "Do MMAs matter?", "dilemma and comma"):
        assert te.triggered(q, GROUPS) == () or all(g.id != "mixedmartialarts" for g in te.triggered(q, GROUPS)), q
        assert te.query_variants(q, GROUPS) == [], q
    for q in ("MMA vs boxing", "UFC and MMA rule changes", "MMA grappling drills", "mixed-martial-arts injuries"):
        assert any(g.id == "mixedmartialarts" for g in te.triggered(q, GROUPS)), q


def test_f1_collision_counts_helper_on_scope_sets():
    qs = [it["text"] for f in ("scope_set_v1.json", "scope_set_v2.json")
          for it in json.loads((ROOT / "config" / f).read_text())["items"]]
    c = te.collision_counts(qs + ["Raised MMA in B12 deficiency", "MMA vs boxing"], GROUPS)["mixedmartialarts:mma"]
    assert c["alias"] == c["alias_triggered"] + c["alias_untriggered"] and c["alias_untriggered"] >= 1
    assert c["alias_triggered"] >= 1


def test_f2_lone_alias_never_clears_a_floor_the_base_would_not():
    span = "UFC rules changed in 2001 for safety."
    for signal in ("legacy", "idf_weighted"):
        base = cfg()
        base["admission"]["signal"] = signal
        on = cfg(term_groups={"enabled": True})
        on["admission"]["signal"] = signal
        docs = {"u": ("UFC history", [span, span + " Again."])}
        assert len(run("UFC", docs, on).admitted) <= len(run("UFC", docs, base).admitted)
    on = cfg(term_groups={"enabled": True})
    icfg = fa._routing(on)["identification"]
    tg = fa._term_ctx("UFC", on, icfg, 5)
    assert tg is not None and fa._matched("UFC", span, icfg, 5, tg) == (1, 1)  # weight 3, but ONE floor hit


KNEE = ["Knee injuries in marathon runners are caused by overuse.", "Runners with knee pain often increase mileage."]


def test_f3_no_allocation_on_single_part_question():
    q = "What causes knee injuries in marathon runners?"
    on = cfg(part_allocation={"enabled": True})
    assert fa._alloc_parts(q, on["admission"]["part_allocation"], fa._routing(on)["identification"], None) == []
    b = run(q, {"k": ("Knees", KNEE)}, on)
    assert b.parts_unsupported == () and "parts" not in b.trace
    assert view(b)[0] == view(run(q, {"k": ("Knees", KNEE)}, FD))[0]
    # "between X and Y" without a facet cue is not a comparison
    q2 = "What is the relationship between diet and cancer?"
    assert fa._alloc_parts(q2, on["admission"]["part_allocation"], fa._routing(on)["identification"], None) == []


def test_f3_concept_parts_are_key_concepts_or_groups():
    on = cfg(part_allocation={"enabled": True}, term_groups={"enabled": True})
    icfg = fa._routing(on)["identification"]
    tg = fa._term_ctx(QA, on, icfg, 5)
    parts = fa._alloc_parts(QA, on["admission"]["part_allocation"], icfg, tg)
    assert [(p.kind, p.cues) for p in parts if p.kind == "concept"] == [
        ("concept", ("boxing",)), ("concept", ("mixed martial arts", "mma"))]
    q = "What are the differences between boxing and kickboxing?"  # one facet + explicit comparison
    parts = fa._alloc_parts(q, on["admission"]["part_allocation"], icfg, None)
    assert [p.cues for p in parts] == [("boxing",), ("kickboxing",)]


def test_literal_group_token_in_text_is_not_a_match():
    on = cfg(term_groups={"enabled": True})
    docs = {"i": ("Inj", ["xgmixedmartialarts and boxing differ greatly.", "Boxing and mixed martial arts differ greatly."])}
    b = run(Q, docs, on)
    assert [a["span_id"] for a in b.admitted] == ["i:1"]


def test_group_token_survives_without_generic_terms():
    on = cfg(term_groups={"enabled": True})
    on["admission"]["relevance"].pop("question_generic_terms")
    icfg = fa._routing(on)["identification"]
    tg = fa._term_ctx(QA, on, icfg, 5)
    assert "xgmixedmartialarts" in fa._q_stems(QA, icfg, 5, tg)
