"""H19-T5: vera.nd_guard G1 to G6 and G9 on fixture text written for this test (no publisher text). Offline, pure."""
import copy
import json
from pathlib import Path

from vera import fulldoc_answer as fa, nd_guard as g

ART = ("The county board voted on Tuesday to close two rural clinics by June. "
       "Officials said the closures would save about four million dollars each year. "
       "Residents packed the meeting room and many spoke against the plan. "
       "A nurse described driving ninety minutes to the nearest hospital. "
       "The board chair promised a review of transport options within sixty days. "
       "Opponents are collecting signatures for a ballot measure in the autumn. "
       "State health regulators have not yet approved the closures. "
       "The clinics together served roughly nine thousand patients last year.")
OTHER = "Rural hospital closures have risen across several states in the past decade, according to a policy survey."
ND = {"n": 1, "snippet": ART, "canonical_text": ART, "title": "County board closes two clinics", "licence_class": "nd",
      "url": "https://www.example-news.org/2026/clinics", "relevance": 0.9}
OS = {"n": 2, "snippet": OTHER, "title": "Survey", "url": "https://other.example/s", "relevance": 0.8}


def rec(i, sentence, cites, verdict="supported"):
    return {"id": i, "sentence": sentence, "citations": cites, "verdict": verdict}


def test_no_nd_span_is_identity():
    rs = [rec(1, "Anything at all goes here today.", [2])]
    out, rep = g.apply(rs, [OS], {})
    assert out is rs and rep == {}
    assert g.limit_quotes([{"n": 2, "quote": "x" * 500}], [OS]) == 0


def test_g1_verbatim_run_dropped_paraphrase_kept():
    copy_ = rec(1, "Officials said the closures would save about four million dollars each year [1].", [1])
    own = rec(2, "Two rural clinics face closure after a county vote [1].", [1])
    mix = rec(3, "Rural closures are rising nationally [2].", [2])
    out, rep = g.apply([copy_, own, mix, rec(4, "Clinic access is a state issue [2].", [2]),
                        rec(5, "Policy surveys track hospital closures [2].", [2])], [ND, OS], {})
    assert copy_ not in out and own in out and rep["g1"] == 1


def test_g2_g3_share_and_count_caps():
    cites = [rec(i, f"Fact number {i} about the county plan differs entirely [1].", [1]) for i in range(1, 7)]
    others = [rec(10 + i, f"Independent finding {i} on unrelated closures nationwide [2].", [2]) for i in range(3)]
    out, rep = g.apply(cites + others, [ND, OS], {"max_coverage": 1.0, "max_sequence": 99})
    n_nd = sum(1 for r in out if 1 in r["citations"])
    assert n_nd <= 4 and n_nd / len(out) <= 0.4 and rep["g2g3"] >= 1
    # a compliant 2-fact answer (2 ND of 5 sentences = 0.4) survives
    ok = [rec(1, "Two clinics are slated to close [1].", [1]), rec(2, "Regulators have yet to sign off [1].", [1])] + [
        rec(10 + i, f"Independent finding {i} on unrelated closures nationwide [2].", [2]) for i in range(3)]
    kept, rep = g.apply(ok, [ND, OS], {})
    assert len(kept) == 5 and not any(rep.values())


def test_g2_single_source_answer_is_emptied_g8():
    only = [rec(1, "The board decided to shut two clinics [1].", [1])]
    out, _ = g.apply(only, [ND, OS], {})
    assert out == []  # G8: the caller's require_supported_claim then takes the honest-notice fallback


def test_g4_coverage_drops_structural_paraphrase():
    sents = ["The county board voted Tuesday to close two rural clinics by June.",
             "Officials said the closures would save four million dollars each year.",
             "Residents packed the meeting room, many speaking against the plan.",
             "A nurse described driving ninety minutes to the nearest hospital."]
    rs = [rec(i + 1, t + " [1]", [1]) for i, t in enumerate(sents)] + [rec(9, "National closures are rising [2].", [2]) for _ in range(1)]
    out, rep = g.apply(rs, [ND, OS], {"max_share": 1.0, "max_sentences": 99, "max_sequence": 99,
                                      "min_verbatim_tokens": 99})
    assert rep["g4"] >= 1 and len(out) < len(rs)
    assert rs[4] in out  # the unrelated sentence survives


def test_g5_sequence_run_tail_dropped():
    sents = ["The board voted Tuesday to close two rural clinics by June.",
             "Residents packed the meeting room and many spoke against the plan.",
             "A nurse described driving ninety minutes to the nearest hospital.",
             "Opponents are collecting signatures for a ballot measure in the autumn.",
             "State health regulators have not yet approved the closures."]
    rs = [rec(i + 1, t + " [1]", [1]) for i, t in enumerate(sents)]
    off = {"max_share": 1.0, "max_sentences": 99, "max_coverage": 1.0, "min_verbatim_tokens": 99, "near_copy": 2.0,
           "max_cover_per_sentence": 99, "max_words_per_source": 999, "union_containment": 2.0}
    out, rep = g.apply(rs, [ND], off)
    assert rep["g5"] == 2 and out == rs[:3]
    # the same facts in reversed order are not an in-order retelling
    out2, rep2 = g.apply(list(reversed(rs)), [ND], off)
    assert rep2["g5"] == 0 and len(out2) == 5


def test_g6_quote_budgets():
    cs = [{"n": 1, "quote": "a" * 150, "quote_span": "x"}, {"n": 1, "quote": "b" * 150, "quote_span": "y"},
          {"n": 1, "quote": "c" * 150, "quote_span": "z"}, {"n": 1, "quote": "d" * 250, "quote_span": "w"},
          {"n": 2, "quote": "e" * 300, "quote_span": "v"}]
    assert g.limit_quotes(cs, [ND, OS], {}) == 2
    assert [bool(c["quote"]) for c in cs] == [True, True, False, False, True]  # 300 for source 1 (<=400); source 2 not ND
    assert cs[2]["quote_span"] is None


def test_g9_single_article_request_forces_quote_and_link():
    q = "Please summarise https://www.example-news.org/2026/clinics for me"
    assert g.g9_sources(q, [ND, OS]) == [1]
    assert g.g9_sources("Summarize 'County board closes two clinics'", [ND, OS]) == [1]
    assert g.g9_sources("Translate this article into French", [ND, OS]) == [1]  # ambiguous: fail closed
    assert g.g9_sources("rewrite this link in a pirate style", [ND]) == [1]
    assert g.g9_sources("What is happening with rural clinics?", [ND, OS]) == []
    assert g.g9_sources("Summarise the survey", [OS]) == []  # no ND source at all
    rs = [rec(1, "The board will close two clinics by June [1].", [1]), rec(2, "National closures are rising [2].", [2])]
    out, rep = g.apply(rs, [ND, OS], {"max_coverage": 1.0, "max_sequence": 99, "max_share": 1.0}, q)
    assert out == [rs[1]] and rep["g9"] == 1 and rep["g9_forced"] == [1]
    ql = g.g9_quote_and_link(q, [ND, OS])
    it = ql["items"][0]
    assert ql["forced"] == [1] and it["url"] == ND["url"] and it["quote"] in ART and len(it["quote"]) <= 200
    # no sentence of the quote-and-link form is a paraphrase: the quote is verbatim
    assert g.apply(rs, [ND, OS], {}, "What is new?")[1].get("g9") == 0


def test_hook_prompt_payload_and_config():
    cfg = json.loads((Path(__file__).resolve().parent.parent / "config" / "fulldoc_answer.json").read_text(encoding="utf-8"))
    c = cfg["answer"]["nd_guard"]
    assert "_calibration_note" in c and "UNCALIBRATED" in c["_calibration_note"]
    srcs = [dict(ND, span_id="s", snippet=ART)]
    msgs = fa.messages_for("q", srcs, cfg)
    assert json.loads(msgs[1]["content"])["passages"][0]["licence"] == "ND"
    assert "no-derivatives" in msgs[0]["content"]
    plain = fa.messages_for("q", [OS | {"span_id": "s"}], cfg)
    assert "no-derivatives" not in plain[0]["content"] and "licence" not in json.loads(plain[1]["content"])["passages"][0]
    kept, n = fa._nd_guard([rec(1, "Officials said the closures would save about four million dollars each year [1].", [1])],
                           [ND], cfg, "")
    assert kept == [] and n == 1
    same = [rec(1, "x y z [2].", [2])]
    assert fa._nd_guard(same, [OS], cfg, "") == (same, 0)


CONDENSED = [  # F2 reviewer fixture shape: 3 condensed paraphrases that together cover all 8 article sentences
    "County board voted Tuesday closing two rural clinics by June, saving officials about four million dollars yearly, "
    "though residents packed the meeting room speaking against the plan [1].",
    "A nurse said she drove ninety minutes to reach a hospital, and the board chair promised reviewing transport "
    "options within sixty days while opponents collect ballot measure signatures [1].",
    "State health regulators have not approved closures, after serving roughly nine thousand patients in 2025 [1].",
]


def test_f2_condensed_paraphrase_of_one_article_is_dropped():
    rs = [rec(i + 1, t, [1]) for i, t in enumerate(CONDENSED)] + [
        rec(10 + i, f"Independent finding {i} on unrelated closures nationwide [2].", [2]) for i in range(5)]
    out, rep = g.apply(rs, [ND, OS], {}, "What is happening with the clinics?")
    assert rep["g1"] == 0  # not caught by the verbatim check: the containment checks must catch it
    assert not [r for r in out if 1 in r["citations"]], rep
    assert len([r for r in out if 2 in r["citations"]]) == 5
    # per-source word budget alone also limits paraphrased text
    longs = [rec(i, f"Distinct clinic detail number {i} about transport funding staffing patients budget " + "x" * i + " [1].", [1])
             for i in range(1, 4)]
    assert g.apply(longs, [ND, OS], {"max_words_per_source": 12, "max_share": 1.0, "max_coverage": 1.0})[1]["g10"] >= 1


def test_f5_missing_canonical_text_and_disabled_fail_closed():
    nocanon = dict(ND); nocanon.pop("canonical_text")
    rs = [rec(1, "A short fact [1].", [1]), rec(2, "Other fact [2].", [2])]
    out, rep = g.apply(rs, [nocanon, OS], {})
    assert out == [rs[1]] and rep["closed"] == 1
    out, rep = g.apply(rs, [ND, OS], {"enabled": False})
    assert out == [rs[1]] and rep["closed"] == 1
    # no ND source at all: untouched even when disabled
    assert g.apply(rs, [OS], {"enabled": False})[0] is rs


def test_f3_widened_g9_cues_from_config():
    for q in ("Summarizing this article please", "paraphrasing the story", "what are the key points of the article",
              "main takeaways?", "explain this article to me", "TL;DR", "what does this article say", "give me the gist"):
        assert g.request_cue(q), q
    assert g.request_cue("overview of clinics", {"g9_cues": ["nothingmatches"]}) is False
    cfg = json.loads((Path(__file__).resolve().parent.parent / "config" / "fulldoc_answer.json").read_text(encoding="utf-8"))
    assert isinstance(cfg["answer"]["nd_guard"]["g9_cues"], list) and g.request_cue("key points", cfg)
    assert g.g9_sources("what does this article say", [ND, OS]) == [1]
