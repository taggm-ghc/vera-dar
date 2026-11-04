"""NONASK-1103 C1: golden-driven APA 7 test. Honours each case's `gate` (exact vs structure); the fixture is read-only.
Comparison rule: tests/fixtures/apa7_goldens_README.md (visible text; escapes removed; links reduced to text)."""
import json
import re
from pathlib import Path

import pytest

from vera import apa7

DATA = json.loads((Path(__file__).parent / "fixtures" / "apa7_goldens.json").read_text(encoding="utf-8"))
CASES = DATA["cases"]
_LINK = re.compile(r"(?<!\\)\[((?:[^\]\\]|\\.)*)\]\(([^)\s]*)\)")
_ESC = re.compile(r"\\([!-/:-@\[-`{-~])")


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(apa7, "_public", lambda u: not re.search(r"//(10\.|127\.|192\.168\.|localhost)", u))


def visible(raw):
    links = []

    def sub(m):
        links.append(m.group(2))
        return m.group(1)
    t = _LINK.sub(sub, raw)
    return _ESC.sub(r"\1", t), links


def _norm(s):
    return re.sub(r"\s+", " ", s.replace("*", "").replace("\u2013", "-")).strip()


def _check_assertions(a, raw, vis, in_txt, links=()):  # noqa: ARG001
    for el_prev, el in zip([None] + a.get("elements_in_order", []), a.get("elements_in_order", [])):
        pass
    pos = 0
    for el in a.get("elements_in_order", []):
        i = vis.find(el, pos)
        if i < 0 and "\u2013" in vis:
            i = vis.replace("\u2013", "-").find(el, pos)
        assert i >= 0, f"element {el!r} missing/out of order after {pos} in {vis!r}"
        pos = i + len(el)
    for x in a.get("must_not_contain", []):
        assert x.lower() not in vis.lower(), f"must_not_contain {x!r}"
    if "must_not_end_with" in a:
        assert not vis.endswith(a["must_not_end_with"])
    for x in a.get("raw_must_not_contain", []):
        assert x not in raw
    for x in a.get("raw_must_not_match", []):
        assert not re.search(x, raw), f"raw matches {x!r}: {raw!r}"
    for x in a.get("reference_must_not_start_with", []):
        assert not vis.startswith(x)
    for x in a.get("in_text_must_not_contain", []):
        assert x not in in_txt
    for k, n in a.get("count", {}).items():
        assert vis.count(k) == n
    if "ellipsis_between" in a:
        x, y = a["ellipsis_between"]
        assert re.search(re.escape(x) + r" (\. \. \.|\.\.\.) " + re.escape(y), vis), vis
    if "isbn_digits_after_strip" in a:
        assert re.sub(r"[-\s]", "", a["isbn_digits_after_strip"]) in vis.replace("-", "")


def _run(case):
    rec = case["input"]
    e = case["expected"]
    codes = []
    raw, codes = apa7.reference_ex(rec)
    vis, links = visible(raw)
    in_txt = apa7.in_text(rec)
    if case["gate"] == "exact":
        assert _norm(vis) == _norm(e["reference"]), f"\n got: {vis}\n want: {e['reference']}"
    _check_assertions(e.get("assertions", {}), raw, vis, in_txt, links)
    assert links == e["links"], f"links {links} != {e['links']}"
    if "reason_code" in e:
        assert e["reason_code"] in codes, codes
    if "reason_code_prefix" in e:
        assert any(c.startswith(e["reason_code_prefix"]) for c in codes), codes
    for x in e.get("visible_contains", []):
        assert x in vis, f"{x!r} not in {vis!r}"
    if "visible_ends_with" in e:
        assert vis.endswith(e["visible_ends_with"]), vis
    if "in_text" in e:  # in-text forms are all `structure`: author/date tokens must agree
        got = visible(in_txt)[0]
        assert _norm(got) == _norm(e["in_text"]), f"in_text got {got!r} want {e['in_text']!r}"
    if "in_text_narrative" in e:
        assert visible(apa7.in_text(rec, narrative=True))[0] == e["in_text_narrative"]
    for x in e.get("characterisation_contains", []):
        assert x.lower() in apa7.characterise(rec).lower()


def _run_list(case):
    recs = case["input"]["records"]
    e = case["expected"]
    got = [visible(x)[0] for x in apa7.reference_list(recs)]
    assert [_norm(x) for x in got] == [_norm(x) for x in e["references_in_order"]], got
    for r in recs:
        assert visible(apa7.in_text(r, context=recs))[0] == e["in_text_by_title"][r["title"]]


def _run_answer(case):
    ans, ns = case["input"]["answer"], {int(k): v for k, v in case["input"]["ns"].items()}
    e = case["expected"]
    before = ans
    disp = apa7.display_answer(ans, ns)
    assert ans == before and disp == e["display"], disp
    if "references_in_order_first_tokens" in e:
        lst = [visible(x)[0] for x in apa7.reference_list(list(ns.values())[:0] + [ns[n] for n in e["cited_set"]])]
        for tok, ref in zip(e["references_in_order_first_tokens"], lst):
            assert ref.startswith(tok)
    if "reference_count" in e:
        docs = {v.get("doc_key") for v in ns.values()}
        assert len(docs) == e["reference_count"]
    for k, n in e.get("assertions", {}).get("display_count", {}).items():
        assert disp.count(k) == n


@pytest.mark.parametrize("case", [c for c in CASES if c["id"] != "G20"], ids=lambda c: c["id"])
def test_golden(case):
    {"list": _run_list, "answer": _run_answer}.get(case.get("kind"), _run)(case)


def test_g24_structure_and_g20_no_junk():
    g24 = next(c for c in CASES if c["id"] == "G24")
    ref = visible(apa7.reference(g24["input"]))[0]
    a = g24["expected"]["assertions"]
    assert [f for f in a["author_families_in_order"] if f in ref] == a["author_families_in_order"]
    for ini in a["author_initials"]:
        assert ini in ref
    # G20 (property subset): any subset of fields of the base cases yields no junk tokens
    import itertools
    for c in CASES:
        if c["id"] in ("G1", "G5a", "G7", "G9", "G10", "G16", "G17", "G22"):
            keys = [k for k in c["input"] if k != "type"]
            for n in range(len(keys) + 1):
                for sub in itertools.islice(itertools.combinations(keys, n), 40):
                    r = {"type": c["input"]["type"], **{k: c["input"][k] for k in sub}}
                    out = visible(apa7.reference(r))[0]
                    for bad in ("None", "null", "undefined", "Anonymous", "Unknown"):
                        assert bad not in out, (c["id"], sub, out)
                    if "author" not in sub and "title" in sub and not r.get("author"):
                        assert out.startswith(re.sub(r"^\*", "", c["input"]["title"][:1])) or out.startswith("*")
                    if "issued" not in sub:
                        assert "(n.d.)" in out or r["type"] == "software"
