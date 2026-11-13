"""USFED-RESUME-1113 RS-A (work/USFED-ROUTE-1111/PLAN.md section 5.7; owner ruling OD-1113-USFED-TWO-PAGES): two
distinct DOE pages per question, DOE row only. t8-t16 of section 5.7.

- match_subject: the DOE row's own cap (row key max_pages_per_question, honoured only as int 2 on row id
  energy_gov_doe_explains, no other row on its host); every other case keeps limits.max_candidates_per_host.
- Persistent floor for the DOE row (evidence_floor "distinct_pages"): counted per question over distinct pages, at most
  one distinct passage (sha256 of its normalised text) per page, as an order-independent maximum matching
  (original_html.distinct_page_marks); every other route keeps auto_acquire.persist_eligible.
- Background and list jobs of the DOE row are refused R-ACQ-FEWSPANS.

Offline: sockets raise, fake HTTP, stub store, no DB. Page fixtures are the synthetic DOE Explains structure of
tests/test_original_html_wpg_1111.py (no article-page content); plans come from the real route's evaluate."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import socket
from types import SimpleNamespace

import pytest

from tests import acq_support as A
from tests import test_acq6_rounds_1103 as R
from tests import test_original_html_wpg_1111 as W
from vera import acq_config, acq_discovered, inline_acquire as ia, original_html
from vera import auto_acquire as aa

DOE, NASA = W.DOE, W.NASA
DOE2 = "https://www.energy.gov/science/doe-explainsgluons"
DOE3 = "https://www.energy.gov/science/doe-explainsquark-matter"
NASA2 = "https://science.nasa.gov/universe/quarks"
NASA3 = "https://science.nasa.gov/universe/gluons"
ROW = "energy_gov_doe_explains"
FEW = "R-ACQ-FEWSPANS"
PARA_B = ("Gluons carry the strong interaction between quarks, and because gluons themselves carry colour charge they "
          "also attract one another, which confines quarks inside hadrons such as protons and neutrons. ")


def sha(s: str) -> str:
    return hashlib.sha256(" ".join(s.split()).casefold().encode("utf-8")).hexdigest()


P, Q2, S3 = sha("passage p"), sha("passage q"), sha("passage s")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    original_html.reset_for_tests()
    acq_discovered.reset_for_tests()

    def boom(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(socket.socket, "connect", boom)
    yield
    original_html.reset_for_tests()


def doe_row(c):
    return next(r for r in c["rows"] if r["id"] == ROW)


def nasa_row(c):
    return next(r for r in c["rows"] if r["id"] == "nasa_science_explainers")


def on(monkeypatch, urls=(DOE, DOE2, NASA), titled=True, edit=None):
    """Mode on over the SHIPPED config (deep copy), an in-index snapshot of `urls`; `edit(c)` changes the copy."""
    c = W.set_mode(monkeypatch, "shadow")
    c["mode"] = "on"
    if edit:
        edit(c)
    snap = W.index_snapshot(list(urls))
    if titled:
        for e in snap["entries"]:
            e["title"], e["slug_terms"] = "Quarks and Gluons", ["quarks", "gluons"]
        snap["sha256"] = original_html.snapshot_digest(snap)
    monkeypatch.setattr(original_html, "load_snapshot", lambda cfg=None: snap)
    return c


def hosts(out):
    return [m["url"].split("/")[2] for m in out]


# ================================================================================ shipped config (RS-A keys)
def test_shipped_row_carries_the_keys_and_nothing_else_changed():
    r = doe_row(W.SHIPPED)
    assert r["max_pages_per_question"] == 2 and type(r["max_pages_per_question"]) is int
    assert r["evidence_floor"] == "distinct_pages"
    assert "OD-1113-USFED-TWO-PAGES" in r["_calibration_note"]
    assert W.SHIPPED["mode"] == "off"
    assert W.SHIPPED["limits"]["max_candidates_per_host"] == 1 and W.SHIPPED["limits"]["max_candidates_per_question"] == 3
    n = nasa_row(W.SHIPPED)
    assert "max_pages_per_question" not in n and "evidence_floor" not in n


def test_row_cap_never_exceeds_the_inline_cap():
    assert doe_row(W.SHIPPED)["max_pages_per_question"] <= ia._OH_INLINE_MAX


# ================================================================================ t8 / t9 / t13: match_subject
def test_t8_doe_subject_with_two_matches_returns_two_doe_pages(monkeypatch):
    on(monkeypatch)
    out = original_html.match_subject("quarks and gluons")
    assert hosts(out).count("www.energy.gov") == 2 and len(out) == 3
    assert {m["url"] for m in out if "energy.gov" in m["url"]} == {DOE, DOE2}
    assert len(ia._original_html_hook(W.Q, [])) == ia._OH_INLINE_MAX == 2


def test_t8_three_doe_matches_still_cap_at_two(monkeypatch):
    on(monkeypatch, urls=(DOE, DOE2, DOE3))
    assert hosts(original_html.match_subject("quarks and gluons")) == ["www.energy.gov"] * 2


def test_t9_nasa_row_keeps_the_global_cap(monkeypatch):
    on(monkeypatch, urls=(NASA2, NASA3))
    assert len(original_html.match_subject("quarks and gluons")) == 1


def test_t9_doe_row_without_the_keys_keeps_the_global_cap(monkeypatch):
    def drop(c):
        doe_row(c).pop("max_pages_per_question", None)
        doe_row(c).pop("evidence_floor", None)
    on(monkeypatch, edit=drop)
    assert hosts(original_html.match_subject("quarks and gluons")).count("www.energy.gov") == 1


def test_t9_second_row_on_the_doe_host_fails_closed(monkeypatch):
    def second(c):
        r = copy.deepcopy(doe_row(c))
        r.update({"id": "energy_gov_other", "path_stems": ["/science/other-explains"]})
        r.pop("max_pages_per_question", None)
        r.pop("evidence_floor", None)
        c["rows"].append(r)
    on(monkeypatch, edit=second)
    assert hosts(original_html.match_subject("quarks and gluons")).count("www.energy.gov") == 1


@pytest.mark.parametrize("key,val", [("max_pages_per_question", True), ("max_pages_per_question", 3),
                                     ("max_pages_per_question", "2"), ("max_pages_per_question", 2.0),
                                     ("max_pages_per_question", None), ("evidence_floor", "per_page"),
                                     ("evidence_floor", None), ("evidence_floor", "DISTINCT_PAGES")])
def test_t13_bad_values_fail_closed(monkeypatch, key, val):
    on(monkeypatch, edit=lambda c: doe_row(c).__setitem__(key, val))
    assert hosts(original_html.match_subject("quarks and gluons")).count("www.energy.gov") == 1
    a, b = mk(DOE, [P]), mk(DOE2, [Q2])
    assert original_html.distinct_page_marks([a, b], A.cfg()) == {}  # per-document floor applies (not eligible at 1)
    assert not aa.persist_eligible(a, A.cfg())


def test_t13_keys_on_the_nasa_row_are_ignored(monkeypatch):
    def move(c):
        for k in ("max_pages_per_question", "evidence_floor"):
            nasa_row(c)[k] = doe_row(c).pop(k, None) or (2 if k.startswith("max") else "distinct_pages")
    on(monkeypatch, urls=(NASA2, NASA3, DOE, DOE2), edit=move)
    h = hosts(original_html.match_subject("quarks and gluons"))
    assert h.count("science.nasa.gov") == 1 and h.count("www.energy.gov") == 1
    assert original_html.distinct_page_marks([mk(NASA2, [P], row="nasa_science_explainers"),
                                              mk(NASA3, [Q2], row="nasa_science_explainers")], A.cfg()) == {}


def test_t13_keys_on_the_nasa_row_do_not_disable_the_doe_row(monkeypatch):
    def add(c):
        nasa_row(c).update({"max_pages_per_question": 2, "evidence_floor": "distinct_pages"})
    on(monkeypatch, urls=(NASA2, NASA3, DOE, DOE2), edit=add)
    h = hosts(original_html.match_subject("quarks and gluons"))
    assert h.count("science.nasa.gov") == 1 and h.count("www.energy.gov") == 2


# ================================================================================ real plans from the route
_BASE: dict = {}


def real_plan(monkeypatch, url=DOE, body=None):
    """A real plan dict from the shipped route's evaluate (mode on, inline min_spans)."""
    on(monkeypatch, urls=(DOE, DOE2, DOE3, NASA))
    paras = body if body is not None else [W.PARA * 2 for _ in range(4)]
    plan, _ = W.run(W.Web({url: W.page(paras)}), c=W.cand(url), min_spans=acq_config.answer_min_spans(A.cfg()))
    return plan


@pytest.fixture
def base(monkeypatch):
    if "p" not in _BASE:
        _BASE["p"] = real_plan(monkeypatch)
    on(monkeypatch, urls=(DOE, DOE2, DOE3, NASA))
    return _BASE["p"]


def mk(url, shas, row=ROW, base_plan=None, **top):
    """A plan of the real shape (from `base`) for `url` whose admitted passages have digests `shas`."""
    p = copy.deepcopy(base_plan if base_plan is not None else _BASE.get("p") or _SHAPE)
    p["canonical_url"] = url
    p["admitted_spans"] = len(shas)
    p["original_html"] = {"row": row, "canonical_url": url, "admitted_sha": sorted(shas)}
    p.update(top)
    return p


_SHAPE = {"canonical_url": DOE, "admitted_spans": 1, "spans": [], "source": {"content": "", "content_hash": "x",
                                                                              "provenance": {"acquisition": {}}}}


def test_evaluate_records_route_local_admitted_digests(base):
    oh = base["original_html"]
    assert oh["row"] == ROW and oh["canonical_url"] == DOE
    d = oh["admitted_sha"]
    assert d and d == sorted(d) and all(re.fullmatch(r"[0-9a-f]{64}", x) for x in d)
    assert len(d) <= base["admitted_spans"]
    assert "admitted_sha" not in json.dumps(base["source"])  # route-local: never in the stored provenance


def test_admitted_digests_come_from_the_same_admission_call(monkeypatch):
    calls = []
    real = aa.relevance_gate

    def gate(*a, **k):
        calls.append(sorted(k))
        return real(*a, **k)
    monkeypatch.setattr(aa, "relevance_gate", gate)
    p = real_plan(monkeypatch)
    assert len(calls) == 1 and p["original_html"]["admitted_sha"]


# ================================================================================ t10 / t11 / t11b / t12: tally
def marks(plans, **k):
    return original_html.distinct_page_marks(plans, A.cfg(), **k)


def elig(plans, **k):
    m = marks(plans, **k)
    return [m.get(id(p)) for p in plans]


def test_t10_two_doe_pages_one_distinct_passage_each_are_both_eligible(base):
    assert elig([mk(DOE, [P]), mk(DOE2, [Q2])]) == [True, True]


def test_t11_one_page_with_two_passages_alone_is_not(base):
    assert elig([mk(DOE, [P, Q2])]) == [False]


@pytest.mark.parametrize("twin", [DOE, DOE + "/", "https://WWW.ENERGY.GOV/science/doe-explainsquarks-and-gluons",
                                  "https://www.energy.gov/science/doe-explainsQuarks-and-gluons"])
def test_t11_the_same_page_twice_is_not(base, twin):
    assert True not in elig([mk(DOE, [P]), mk(twin, [Q2])])


def test_t11_reused_paragraph_on_two_pages_credits_one(base):
    assert True not in elig([mk(DOE, [P]), mk(DOE2, [P])])
    assert True not in elig([mk(DOE2, [P]), mk(DOE, [P])])


def test_t11_maximum_matching_in_both_orders(base):
    assert elig([mk(DOE, [P, Q2]), mk(DOE2, [P])]) == [True, True]
    assert elig([mk(DOE2, [P]), mk(DOE, [P, Q2])]) == [True, True]


def test_t11_real_reused_paragraph_pair_strong_force_class(monkeypatch):
    """Section 5.6 class 5: two distinct real-route pages whose only admitted passage is the same reused paragraph.
    The real admission runs; the wrapper keeps only its first admitted passage (count and text from the same call), so
    each page admits exactly one passage, as in the readout."""
    real = aa.relevance_gate

    def first_only(*a, admitted_out=None, **k):
        got: list = []
        real(*a, admitted_out=got, **k)
        if admitted_out is not None:
            admitted_out.extend(got[:1])
        return min(len(got), 1)
    monkeypatch.setattr(aa, "relevance_gate", first_only)
    a = real_plan(monkeypatch, DOE)
    b = real_plan(monkeypatch, DOE2)
    assert a["original_html"]["admitted_sha"] == b["original_html"]["admitted_sha"]
    assert len(a["original_html"]["admitted_sha"]) == 1  # not vacuous: one distinct passage on each page
    assert elig([a, b]) == [False, False]
    c = real_plan(monkeypatch, DOE2, body=[PARA_B * 2 for _ in range(4)])
    assert set(c["original_html"]["admitted_sha"]) - set(a["original_html"]["admitted_sha"])
    assert elig([a, c]) == [True, True]


@pytest.mark.parametrize("flag", ["answer_only", "lane", "source_lane"])
def test_t11b_answer_only_or_lane_partner_does_not_credit(base, flag):
    a, b = mk(DOE, [P]), mk(DOE2, [Q2], **{flag: True})
    assert elig([a, b]) == [False, False]


def test_t11b_variant_only_partner_does_not_credit(base):
    a, b = mk(DOE, [P]), mk(DOE2, [Q2])
    assert elig([a, b], exclude=(id(b),)) == [False, False]


def test_t11b_no_cross_row_or_cross_route_sum(base):
    a = mk(DOE, [P])
    n = mk(NASA, [Q2], row="nasa_science_explainers")
    e = copy.deepcopy(_BASE["p"])
    e.pop("original_html")
    e.update({"canonical_url": "https://doi.org/10.5555/example.1", "admitted_spans": 1})
    m = marks([a, n, e])
    assert m.get(id(a)) is False and id(n) not in m and id(e) not in m
    assert not aa.persist_eligible(n, A.cfg()) and not aa.persist_eligible(e, A.cfg())


def test_t11b_a_plan_without_digests_or_with_a_forged_row_does_not_credit(base):
    a, b = mk(DOE, [P]), mk(DOE2, [])
    assert elig([a, b]) == [False, False]
    f = mk(NASA, [Q2])  # claims the DOE row on a NASA URL
    assert elig([a, f]) == [False, False]
    g = mk(DOE2, ["not-a-digest"])
    assert elig([a, g]) == [False, False]


def test_t12_one_doe_page_with_one_passage_is_not(base):
    assert elig([mk(DOE, [P])]) == [False]


def test_t12_non_doe_plans_keep_the_per_document_floor(base):
    e2 = mk("https://doi.org/10.5555/example.2", [P, Q2])
    e2.pop("original_html")
    e1 = mk("https://doi.org/10.5555/example.1", [P])
    e1.pop("original_html")
    assert marks([e2, e1]) == {}
    assert aa.persist_eligible(e2, A.cfg()) and not aa.persist_eligible(e1, A.cfg())


def test_tally_is_pure(base):
    a, b = mk(DOE, [P]), mk(DOE2, [Q2])
    before = json.dumps([a, b], sort_keys=True)
    marks([a, b])
    assert json.dumps([a, b], sort_keys=True) == before


# ================================================================================ t14: background and list jobs
def test_t14_background_job_of_the_doe_row_is_fewspans(monkeypatch):
    on(monkeypatch, urls=(DOE, NASA))
    web = W.Web({DOE: W.page()})
    with pytest.raises(aa._Stop) as e:
        W.run(web)  # min_spans None: background job; the same page passes inline (real_plan)
    assert e.value.code == FEW


def test_t14_list_job_of_the_doe_row_is_fewspans(monkeypatch):
    on(monkeypatch, urls=(DOE, NASA))
    web = W.Web({DOE: W.page()})
    with pytest.raises(aa._Stop) as e:
        W.run(web, job=A.job(question=W.Q, trigger=aa.LIST_TRIGGER))
    assert e.value.code == FEW


def test_t14_without_the_keys_background_is_unchanged(monkeypatch):
    def drop(c):
        doe_row(c).pop("max_pages_per_question", None)
        doe_row(c).pop("evidence_floor", None)
    on(monkeypatch, urls=(DOE, NASA), edit=drop)
    plan, _ = W.run(W.Web({DOE: W.page()}))
    assert plan["canonical_url"] == DOE


# ================================================================================ inline persist loop (t10-t12, t15, t16)
class H2(R.H):
    """The A6 rounds harness: evaluate returns the plan carried by the candidate (a real-shape plan)."""

    def evaluate(self, job, score, doi, c, isbn, acq, deps, icfg, fdcfg, clock, deadline, d, min_spans=1, **k):
        self.evaluated.append(doi)
        return copy.deepcopy(c["_plan"]), None


def cand(tag, plan, **kw):
    return {"doi": f"10.5555/usfed.{tag}", "title": f"Synthetic {tag}", "_usable": True, "_plan": plan, **kw}


def written(h):
    return sorted(p["canonical_url"] for p in h.writes)


def acq_cfg():
    return R.cfg(acq6_rounds_enabled=False)


def test_inline_two_doe_pages_are_both_written(base, monkeypatch):
    h = H2(monkeypatch, [[cand("a", mk(DOE, [P])), cand("b", mk(DOE2, [Q2]))]])
    h.run(acq_cfg())
    assert written(h) == sorted([DOE, DOE2])


def test_inline_reused_paragraph_pair_writes_nothing(base, monkeypatch):
    h = H2(monkeypatch, [[cand("a", mk(DOE, [P])), cand("b", mk(DOE2, [P]))]])
    o = h.run(acq_cfg())
    assert written(h) == [] and {d.get("reason") for d in o.documents} == {"below_persist_min"}


def test_inline_answer_only_partner_does_not_credit(base, monkeypatch):
    h = H2(monkeypatch, [[cand("a", mk(DOE, [P])), cand("b", mk(DOE2, [Q2], answer_only=True), answer_only=True)]])
    h.run(acq_cfg())
    assert written(h) == []


def test_inline_variant_only_partner_does_not_credit(base, monkeypatch):
    h = H2(monkeypatch, [[cand("a", mk(DOE, [P]))], [cand("b", mk(DOE2, [Q2]))]])
    h.run(R.cfg())  # rounds on: b is found only by a variant query (answer-only)
    assert len(h.searches) >= 2 and "10.5555/usfed.b" in h.evaluated
    assert written(h) == []


def test_inline_reused_stored_row_does_not_credit(base, monkeypatch):
    stored = mk(DOE2, [Q2])
    h = H2(monkeypatch, [[cand("a", mk(DOE, [P]))]])
    monkeypatch.setattr(ia, "_stored_for", lambda ranked, *a, **k: (A.rows_from_plans([stored]), list(ranked)))
    h.run(acq_cfg())
    assert written(h) == []


def test_inline_doe_plus_epmc_one_each_writes_nothing_and_non_doe_two_still_written(base, monkeypatch):
    e1 = mk("https://doi.org/10.5555/example.1", [P])
    e1.pop("original_html")
    e2 = mk("https://doi.org/10.5555/example.2", [P, Q2])
    e2.pop("original_html")
    h = H2(monkeypatch, [[cand("a", mk(DOE, [P])), cand("e1", e1), cand("e2", e2)]])
    h.run(acq_cfg())
    assert written(h) == ["https://doi.org/10.5555/example.2"]


def test_t15_partner_write_failure_counts_pair_partial_and_no_further_write(base, monkeypatch, caplog):
    h = H2(monkeypatch, [[cand("a", mk(DOE, [P])), cand("b", mk(DOE2, [Q2]))]])
    tried = []

    def write(p):
        tried.append(p["canonical_url"])
        if p["canonical_url"] == DOE2:
            raise RuntimeError("write failed")
        h.writes.append(p)
        return {"status": "stored"}
    h.deps.store = SimpleNamespace(write=write)
    with caplog.at_level(logging.INFO):
        o = h.run(acq_cfg())
    assert sorted(tried) == sorted([DOE, DOE2]) and written(h) == [DOE] and o.stored == 1
    ev = [r.getMessage() for r in caplog.records if "original_html_pair_partial" in r.getMessage()]
    assert len(ev) == 1
    for r in caplog.records:
        m = r.getMessage()
        assert "energy.gov" not in m and "https://" not in m


def test_t15_both_writes_ok_is_not_partial(base, monkeypatch, caplog):
    h = H2(monkeypatch, [[cand("a", mk(DOE, [P])), cand("b", mk(DOE2, [Q2]))]])
    with caplog.at_level(logging.INFO):
        h.run(acq_cfg())
    assert not [r for r in caplog.records if "original_html_pair_partial" in r.getMessage()]


def _progress(monkeypatch, plans):
    ev = []
    monkeypatch.setattr(ia, "_emit", lambda stage, state, **c: ev.append((stage, state, c)))
    h = H2(monkeypatch, [[cand(f"x{i}", p) for i, p in enumerate(plans)]])
    h.run(acq_cfg(), stream=True)
    ups = [c for s, st, c in ev if s == "acquire" and st == "update" and "usable" in c]
    return ups[-1] if ups else None


def test_t16_progress_count_uses_the_tally(base, monkeypatch):
    last = _progress(monkeypatch, [mk(DOE, [P]), mk(DOE2, [Q2])])
    assert last is not None and last["usable"] == 2 and last["stored"] == 2


def test_t16_progress_count_reused_paragraph_is_zero(base, monkeypatch):
    last = _progress(monkeypatch, [mk(DOE, [P]), mk(DOE2, [P])])
    assert last is not None and last["usable"] == 2 and last["stored"] == 0
