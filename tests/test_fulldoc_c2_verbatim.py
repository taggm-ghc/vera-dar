"""PLAN-FDA C2 tests: verbatim accounting against the full bundle text. Synthetic text only, no network, no DB."""
import json
import random
import re

import pytest

from vera import fulldoc_answer as fa
from vera import fulldoc_verbatim as fv
from vera.ask_service import AskResult
from vera.fulldoc_evidence import EvidenceBundle

WORDS = ("quartz lantern meadow orbit velvet harbor cinder prairie tundra copper willow ember glacier thistle marble "
         "falcon sorrel basalt nimbus cobalt juniper saffron drift timber ripple anchor beacon canyon dusk estuary "
         "fjord gossamer hollow indigo jasper kelp lagoon mosaic nectar opal pebble quill raven summit tangle umber "
         "valley wicker yarrow zenith").split()


def prose(seed, n):
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        out.append(" ".join(rnd.choice(WORDS) for _ in range(rnd.randint(9, 14))).capitalize() + ".")
    return " ".join(out)


DOC_A = prose(1, 60)
DOC_B = prose(2, 60)  # a bundle document that is never admitted
CHK = fv.VerbatimChecker([DOC_A, DOC_B], [], {})


def sent(i, doc=DOC_A):
    return [s for s in doc.split(". ")][i].rstrip(".") + "."


def counted(text, chk=CHK):
    return chk.profile(text).counted


# ---- normalisation -------------------------------------------------------------------------------------------

def test_normalisation_rules():
    assert fv.normalize("Café  “quoted” — well-\nknown co­op[19,20] (S1 Text) x²") == \
        "café \"quoted\" - wellknown coop x"
    assert fv.normalize("ﬁnal ＡBC") == "final abc"  # NFKC expansion / fullwidth


@pytest.mark.parametrize("t", ["Plain text.", "A [1] b­ c  d", "ﬁsh ß ’q’", "é x-\ny", "  lead and trail  "])
def test_mapped_equals_fast(t):
    assert fv.normalize_mapped(t)[0] == fv.normalize(t)


def test_lcs_len_matches_dp():
    rnd = random.Random(5)
    for _ in range(40):
        a = "".join(rnd.choice("abc") for _ in range(rnd.randint(0, 25)))
        b = "".join(rnd.choice("abc") for _ in range(rnd.randint(0, 25)))
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a)):
            for j in range(len(b)):
                dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
        assert fv.lcs_len(a, b) == dp[-1][-1]


# ---- detection -----------------------------------------------------------------------------------------------

def test_verbatim_vs_paraphrase():
    s = sent(3)
    assert counted(s) >= len(s) - 2
    para = "Meanwhile, " + " ".join(reversed(s.rstrip(".").split())) + " is what we note."
    assert counted(para) == 0


def test_short_run_not_counted_long_run_counted():
    toks = sent(4).rstrip(".").split()
    assert counted(" ".join(toks[:7]) + " zzzz unrelated words follow here.") == 0
    assert counted("Intro words that differ. " + " ".join(toks[:9]) + " and then a closing remark that differs.") > 0


def test_source_markers_do_not_hide_verbatim():
    toks = sent(5).rstrip(".").split()
    marked = " ".join(w + (" [1]" if i % 3 == 0 else "") + ("²" if i % 4 == 0 else "") for i, w in enumerate(toks)) + "."
    assert counted(marked) >= len(" ".join(toks)) - 2


def test_soft_hyphen_and_linebreak_hyphenation():
    s = sent(6)
    toks = s.rstrip(".").split()
    toks[2] = toks[2][:2] + "­" + toks[2][2:]
    toks[4] = toks[4][:3] + "-\n" + toks[4][3:]
    assert counted(" ".join(toks) + ".") > len(s) // 2


def test_nfkc_expanding_chars_use_larger_length():
    ref = "the ofﬁce staff reviewed ﬁnal ﬁgures for eight separate quarterly meetings held yearly"
    chk = fv.VerbatimChecker([ref], [], {})
    p = chk.profile(ref)
    assert p.counted >= len(ref)  # original ligature = 1 char, normalised = 2: the larger is counted
    assert p.counted >= len(fv.normalize(ref))


def test_every_fifth_word_edit_caught_by_lcs():
    s = sent(7)
    toks = s.rstrip(".").split()
    assert len(toks) >= 9
    edited = [("zork" if i % 5 == 4 else w) for i, w in enumerate(toks)]
    text = " ".join(edited) + "."
    nogap = {"gap_tolerance_tokens": 0}
    assert not fv.VerbatimChecker([DOC_A], [], {**nogap, "sentence_lcs_ratio": 1.01}).profile(text).counted  # runs alone miss it
    p = fv.VerbatimChecker([DOC_A], [], nogap).profile(text)  # LCS rule alone catches it: whole sentence counted
    assert p.counted >= len(text) - 1
    assert CHK.profile(text).counted >= len(text) - 6  # default config (gap tolerance + LCS) also catches it


def test_non_admitted_bundle_document_is_counted():
    s = sent(9, DOC_B)
    only_a = fv.VerbatimChecker([DOC_A], [], {})
    assert counted(s, only_a) == 0 and counted(s) >= len(s) - 2


def test_metadata_subtracts_only_matching_characters():
    toks = sent(8).rstrip(".").split()
    title = " ".join(toks[:3])
    ref = " ".join(toks) + "."
    chk = fv.VerbatimChecker([ref], [title], {})
    full = fv.VerbatimChecker([ref], [], {}).profile(ref).counted
    got = chk.profile(ref).counted
    assert 0 < got < full and full - got >= len(title) - 1
    # title, DOI and an author on their own are not counted at all
    meta = ["A Long Reproducibility Study of Quartz Lanterns Across Many Meadows", "10.1234/abcd.5678", "Marlowe Quillfeather"]
    chk2 = fv.VerbatimChecker([" ".join(meta)], meta, {})
    assert chk2.profile(" ".join(meta)).counted == 0


def test_numbers_and_technical_phrase_not_counted():
    ref = "values 12.5 mg 100 ml 14 days 2020 18 mg 40 percent recorded"
    chk = fv.VerbatimChecker([ref], [], {"min_verbatim_tokens": 5})
    assert chk.profile("12.5 mg 100 ml 14 days 2020 18 mg 40 percent").counted == 0
    assert fv.VerbatimChecker([ref], [], {"min_verbatim_tokens": 5, "exclude_numbers": False}).profile(
        "12.5 mg 100 ml 14 days 2020 18 mg 40 percent").counted > 0
    phrase = "findable accessible interoperable reusable data principles applied to every local archive"
    assert fv.VerbatimChecker([phrase], [], {}).profile(phrase).counted > 0
    assert fv.VerbatimChecker([phrase], [], {"technical_phrases": [phrase]}).profile(phrase).counted == 0


def test_combined_total_counts_inline_quote_once():
    s = sent(3)
    p = CHK.profile(s)
    q = s[: len(s) // 2]
    total, gen, quoted = fv.combined_total([p], [q])
    assert total == gen and quoted == len(q)
    total2, _, _ = fv.combined_total([p], ["entirely different quote text"])
    assert total2 == gen + len("entirely different quote text")


# ---- enforcement through answer() ----------------------------------------------------------------------------

def span(i, text, start=10):
    attr = {"creator": f"Author {i}", "title": f"T{i}", "source": f"https://x.org/{i}", "licence": "CC-BY-4.0",
            "licence_url": "https://creativecommons.org/licenses/by/4.0/", "changes": "Short excerpt."}
    return {"span_id": f"s{i}", "document_id": f"d{i}", "canonical_source_id": f"https://x.org/{i}", "title": f"T{i}",
            "url": f"https://x.org/{i}", "published": "2024", "authors": [], "licence": "CC-BY-4.0", "start": start,
            "end": start + len(text), "text": text, "text_sha256": "ab" * 32, "relevance": 0.9, "attribution": attr}


def bundle(admitted, title="T1", authors=()):
    docs = ({"document_id": "d1", "canonical_source_id": "https://x.org/1", "title": title, "authors": list(authors), "url": "https://x.org/1", "doi": None,
             "published": "2024", "licence": "CC-BY-4.0", "licence_decision": "allow", "text_sha256": "cd" * 32, "chars": 9,
             "chunks_inspected": 1, "candidate_spans": 1, "admitted_spans": 1},)
    return EvidenceBundle(question="q", examined_full_documents=docs, admitted=tuple(admitted), status="admitted")


GCFG = {"removed_citation_note": "Note: bad citations removed.",
        "claim_check": {"enabled": True, "timeout_s": 5, "max_claims": 40,
                        "checkers": [{"provider": "t", "model": "m", "api_key_env": "FD_TEST_KEY", "base_url": None}]}}


def cfg(**ans):
    return {"answer": {"system_prompt": "P\n{sources}", "quote_max_chars": 200, "partial_marker": " (PARTIAL)",
                       "removed_note": "Note: removed.", "unchecked_note": "Note: unchecked.",
                       "max_total_quote_chars": 700, **ans},
            "fallback_tokens": {"insufficient": "ins", "model_insufficient": "model_ins", "corpus_unusable": "cu",
                                "no_valid_citations": "nvc", "checker_unavailable": "cku",
                                "claim_check_insufficient": "cci", "claim_check_disabled": "ccd"}}


def run(draft, verdicts, ev, config=None, refs=None):
    def check(chk, msgs):
        n = len(json.loads(msgs[1]["content"])["claims"])
        return json.dumps({"claims": [{"id": i, "verdict": verdicts[i % len(verdicts)], "reason": "r"} for i in range(n)]})
    return fa.answer("q", ev, "m", None, chained={}, grounding_cfg=GCFG, cfg=config or cfg(),
                     complete_fn=lambda *a, **k: AskResult(draft, 1, 0.0), check_call=check, reference_texts=refs)


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("FD_TEST_KEY", "x")


def test_run_over_200_chars_is_dropped_but_short_sentence_kept():
    long_passage = " ".join(DOC_B.split(". ")[:6]) + "."
    assert len(long_passage) > 200
    ev = bundle([span(1, DOC_A[:900])])
    out = run(f"{long_passage} [1] Nothing here matches the corpus at all, a fresh remark. [1]", ("supported",), ev,
              refs=[DOC_A, DOC_B])
    assert out.fallback is None
    assert long_passage[:60] not in out.result.answer
    assert out.result.claim_check["verbatim_dropped"] >= 1
    assert "fresh remark" in out.result.answer


def test_only_overlong_verbatim_gives_fallback():
    long_passage = " ".join(DOC_B.split(". ")[:6]) + "."
    out = run(f"{long_passage} [1]", ("supported",), bundle([span(1, DOC_A[:900])]), refs=[DOC_A, DOC_B])
    assert out.result is None and out.fallback == "cci"


def test_verbatim_budget_in_summary_and_quote_boundaries():
    ev = bundle([span(1, DOC_A[:1500])])
    out = run("A fresh original remark about nothing in the corpus [1].", ("supported",), ev, refs=[DOC_A, DOC_B])
    b = out.result.claim_check["verbatim_budget"]
    assert set(b) == {"quoted", "generated_verbatim", "total", "cap"} and b["cap"] == 700
    assert b["total"] < 700 and all(len(c["quote"] or "") <= 200 for c in out.result.citations)


def _claims_with_gen(n_chars_target):
    """Answer sentences that are each exactly one source sentence (generated verbatim, < 200 chars)."""
    sents = [s + "." for s in DOC_A.split(". ")[:40]]
    return sents


def test_total_boundary_699_700():
    # one verbatim sentence; its counted length L: total < cap required, so cap == L must drop, cap == L + 1 must keep
    s = [x for x in DOC_A.split(". ") if 60 < len(x) < 180][0] + "."
    s = s[:-1] + " [1]."
    L = CHK.profile(s).counted
    ev = bundle([span(1, DOC_A[:1500])])
    keep = run(s, ("supported",), ev, config=cfg(max_total_quote_chars=L + 1), refs=[DOC_A, DOC_B])
    drop = run(s, ("supported",), ev, config=cfg(max_total_quote_chars=L), refs=[DOC_A, DOC_B])
    assert keep.fallback is None and keep.result.claim_check["verbatim_budget"]["total"] == L
    assert all(c["quote"] is None for c in keep.result.citations)  # no room left: quotes nulled, refs kept
    assert keep.result.citations and all(c["span_id"] for c in keep.result.citations)
    assert drop.result is None and drop.fallback == "cci"


def test_quote_200_201_boundary():
    span_text = "Alpha " + " ".join(["wordy"] * 80) + " omega."
    from vera.fulldoc_answer import pick_quote
    for cap in (200, 201):
        assert len(pick_quote(span_text, "wordy", cap)) <= cap
    out = run("An original remark [1].", ("supported",), bundle([span(1, span_text)]), refs=[span_text])
    assert all(len(c["quote"]) <= 200 for c in out.result.citations if c["quote"])


def test_step3_drops_lowest_ranked_verbatim_heavy_sentence():
    sents = [x + "." for x in DOC_A.split(". ") if 70 < len(x) < 130][:6]
    assert len(sents) >= 5
    draft = " ".join(s[:-1] + " [1]." for s in sents)
    ev = bundle([span(1, DOC_A[:2500])])
    out = run(draft, ("supported", "supported", "supported", "partial", "supported", "supported"), ev,
              config=cfg(max_total_quote_chars=450), refs=[DOC_A, DOC_B])
    b = out.result.claim_check["verbatim_budget"]
    assert b["total"] < 450 and out.result.claim_check["verbatim_dropped"] >= 1
    assert "(PARTIAL)" not in out.result.answer  # the partial (lowest rank) went first


def test_model_insufficient_token_distinct_and_default_back_compat():
    ev = bundle([span(1, DOC_A[:900])])
    c = cfg(insufficient_marker="INSUFFICIENT_EVIDENCE")
    o = fa.answer("q", ev, "m", None, chained={}, grounding_cfg=GCFG, cfg=c, complete_fn=lambda *a, **k: AskResult("INSUFFICIENT_EVIDENCE", 1, 0.0))
    assert o.fallback == "model_ins"
    del c["fallback_tokens"]["model_insufficient"]
    o = fa.answer("q", ev, "m", None, chained={}, grounding_cfg=GCFG, cfg=c, complete_fn=lambda *a, **k: AskResult("INSUFFICIENT_EVIDENCE", 1, 0.0))
    assert o.fallback == "ins"


def test_pol_partial_only_falls_back_and_switch():
    ev = bundle([span(1, DOC_A[:900])])
    d = "An original remark about nothing [1]."
    assert run(d, ("partial",), ev).fallback == "cci"
    assert run(d, ("partial",), ev, config=cfg(require_supported_claim=False)).result is not None
    assert run(d, ("supported",), ev).result is not None


def long_run(doc=DOC_A, start=0, min_chars=270):
    """Consecutive source sentences joined into ONE sentence (no internal full stops): a single >200-char verbatim run."""
    out, k = [], start
    while len(" ".join(out)) < min_chars:
        out.append(sent(k, doc).rstrip("."))
        k += 1
    return " ".join(out)


def test_b1_number_inside_long_verbatim_run_does_not_split_it():
    lr = long_run()
    toks = lr.split()
    withnum = " ".join(toks[:20] + ["42"] + toks[20:])
    assert len(lr) > 200
    p = CHK.profile(withnum)
    assert p.max_run > 200, p.max_run
    out = run(withnum[:-1] + " [1].", ("supported",), bundle([span(1, DOC_A[:1500])]), refs=[DOC_A, DOC_B])
    assert out.result is None and out.fallback == "cci"


def test_b1_author_word_substring_does_not_split_long_run():
    ref = " ".join(["the standard approach to handling standard errors in a standard way requires care and patience"] * 4)
    ref = " ".join(f"{w}{i}x" if i % 17 == 0 else w for i, w in enumerate(ref.split()))
    chk = fv.VerbatimChecker([ref], ["and", "and Kay"], {})
    assert len(ref) > 250 and chk.profile(ref).max_run > 200
    # whole-token matching only: "and" cuts the word "and" but never "standard"
    cuts = chk._exclusions(fv.normalize("standard and standard"))
    assert cuts == [(9, 12)]


def test_b1_author_strings_contribute_names_not_common_words():
    got = fa._metadata_strings([{"authors": ["Marlowe Quillfeather and Kay Thornbury"]}])
    assert "Marlowe Quillfeather and Kay Thornbury" in got and "Quillfeather" in got and "Thornbury" in got
    assert not {"and", "Kay"} & set(got)


def test_b1_excluded_title_inserted_mid_run_is_still_caught():
    title = "Zebra Study of Opal Tundra Winds"
    lr = long_run()
    toks = lr.split()
    mixed = " ".join(toks[:20] + [title] + toks[20:])
    chk = fv.VerbatimChecker([DOC_A, DOC_B], [title], {})
    assert chk.profile(mixed).max_run > 200
    ev = bundle([span(1, DOC_A[:1500])], title=title)
    out = run(mixed + " [1].", ("supported",), ev, refs=[DOC_A, DOC_B])
    assert out.result is None and out.fallback == "cci"


def test_b1_excluded_doi_fragment_and_number_together_in_long_run():
    lr = long_run(start=10)
    toks = lr.split()
    mixed = " ".join(toks[:15] + ["10.1234/abcd.5678"] + toks[15:30] + ["3.5", "mg"] + toks[30:])
    chk = fv.VerbatimChecker([DOC_A, DOC_B], ["10.1234/abcd.5678"], {})
    assert chk.profile(mixed).max_run > 200


def _simple_counter(text, refs, quotes=(), n=8):
    """Independent naive counter: exact word 8-grams of any reference document, plain regex tokens, no normalisation
    helpers or checker code. Returns (total chars of the union of marked word spans + quotes, longest marked run)."""
    grams = set()
    for r in refs:
        w = re.findall(r"[a-z0-9]+", r.lower())
        grams |= {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
    words = [(m.group().lower(), m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9]+", text)]
    marked = [False] * len(words)
    for i in range(len(words) - n + 1):
        if tuple(w[0] for w in words[i:i + n]) in grams:
            for j in range(i, i + n):
                marked[j] = True
    chars, longest, run_start, run_end = 0, 0, None, None
    for (w, a, b), m in zip(words, marked):
        if m:
            if run_start is None:
                run_start = a
            run_end = b
            chars += b - a
        else:
            if run_start is not None:
                longest = max(longest, run_end - run_start)
            run_start = None
    if run_start is not None:
        longest = max(longest, run_end - run_start)
    return chars + sum(len(q) for q in quotes if q and q not in text), longest


def test_property_every_response_within_caps():
    rnd = random.Random(11)
    authors = ["Marlowe Quillfeather and Kay Thornbury"]
    ev = bundle([span(1, DOC_A[:2500]), span(2, DOC_B[:2500])], authors=authors)
    pool = [x + "." for x in DOC_A.split(". ")[:25]] + [x + "." for x in DOC_B.split(". ")[:25]]
    answered = trials = 0
    for trial in range(60):
        parts = []
        for _ in range(rnd.randint(1, 7)):
            kind = rnd.choice(["verbatim", "edited", "fresh", "long", "number", "author"])
            base = rnd.choice(pool)
            if kind == "edited":
                t = base.rstrip(".").split()
                base = " ".join("zork" if i % 5 == 4 else w for i, w in enumerate(t)) + "."
            elif kind == "fresh":
                base = prose(rnd.randint(100, 999), 1)
            elif kind == "long":
                base = long_run(rnd.choice([DOC_A, DOC_B]), rnd.randint(0, 20), rnd.choice([220, 300, 420])) + "."
            elif kind in ("number", "author"):
                t = base.rstrip(".").split()
                ins = "17.5 mg" if kind == "number" else "Quillfeather"
                base = " ".join(t[:4] + [ins] + t[4:]) + "."
            parts.append(f"{base[:-1]} [{rnd.choice([1, 2])}].")
        trials += 1
        out = run(" ".join(parts), ("supported", "partial"), ev, refs=[DOC_A, DOC_B])
        if out.result is None:
            continue
        answered += 1
        quotes = [c["quote"] for c in out.result.citations if c["quote"]]
        b = out.result.claim_check["verbatim_budget"]
        assert all(len(q) <= 200 for q in quotes)
        assert b["total"] < 700, (trial, b)
        # independent recount (naive exact-8-gram counter, not the checker) over the PUBLISHED text + quotes. No
        # slack: the legacy answer text holds no notes/markers beyond the verified sentences themselves.
        total, _ = _simple_counter(out.result.answer, [DOC_A, DOC_B], quotes)
        assert total < 700, (trial, total)
        for sentence in re.split(r"(?<=[.!?])\s+", out.result.answer):  # the >200 rule is per sentence
            assert _simple_counter(sentence, [DOC_A, DOC_B])[1] <= 200, (trial, sentence)
    assert answered >= 0.3 * trials, (answered, trials)  # the property is exercised, not vacuous


@pytest.mark.parametrize("inserts", [["Zebra Study of Opal Tundra Winds", "Quartz Atlas of Cobalt Drift"],
                                     ["Zebra Study of Opal Tundra Winds", "12 kg", "Quartz Atlas of Cobalt Drift"]])
def test_b1_chained_excluded_strings_mid_run_are_still_caught(inserts):
    titles = [i for i in inserts if i[0].isalpha()]
    toks = long_run().split()
    mixed = " ".join(toks[:20] + inserts + toks[20:])
    chk = fv.VerbatimChecker([DOC_A, DOC_B], titles, {})
    assert chk.profile(mixed).max_run > 200
    ev = bundle([span(1, DOC_A[:1500])], title=titles[0], authors=[titles[1]])
    out = run(mixed + " [1].", ("supported",), ev, refs=[DOC_A, DOC_B])
    assert out.result is None and out.fallback == "cci"


@pytest.mark.parametrize("order", ["attr_title", "title_attr"])
def test_b1_inserted_text_ending_with_sentence_break_still_bridges(order):
    attr = "Short excerpt of Quartz Atlas, converted to plain text."
    title = "Zebra Study of Opal Tundra Winds"
    ins = [attr, title] if order == "attr_title" else [title, attr]
    s1, s2 = sent(0) + " " + sent(1), sent(2) + " " + sent(3)
    assert len(s1) > 100 and len(s2) > 100
    mixed = " ".join([s1] + ins + [s2])
    chk = fv.VerbatimChecker([DOC_A, DOC_B], [attr, title], {})
    assert chk.profile(mixed).max_run > 200
    # (answer level: claim_check splits such text into separate sentences, each judged on its own; the profile-level
    # bridge matters for any single text profiled as a unit, e.g. one claim sentence containing an abbreviation stop.)
