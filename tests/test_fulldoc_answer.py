"""D-055 Option C step 3 tests: stubbed model and checker, no network."""
import json

import pytest

from vera import fulldoc_answer as fa
from vera.ask_service import AskResult
from vera.fulldoc_evidence import EvidenceBundle

LONG = ("Alpha systems reduce latency by forty percent in field trials. " + "Filler sentence about unrelated matters. " * 12
        + "Beta caching doubles throughput under heavy load when the cache is warm and the workload is read heavy. "
        + "Closing remarks follow here with more words. " * 10)
CFG = {
    "answer": {"system_prompt": "Use passages.\n{sources}", "quote_max_chars": 60, "partial_marker": " (PARTIAL)",
               "removed_note": "Note: removed.", "unchecked_note": "Note: unchecked parts left out."},
    "fallback_tokens": {"insufficient": "insufficient_full_document_evidence", "corpus_unusable": "full_document_corpus_unusable",
                        "no_valid_citations": "no_valid_citations", "checker_unavailable": "checker_unavailable",
                        "claim_check_insufficient": "claim_check_insufficient", "claim_check_disabled": "claim_check_disabled",
                        "model_insufficient": "model_insufficient"},
}
GCFG = {"removed_citation_note": "Note: bad citations removed.",
        "claim_check": {"enabled": True, "timeout_s": 5, "max_claims": 10,
                        "checkers": [{"provider": "t", "model": "m", "api_key_env": "FD_TEST_KEY", "base_url": None}]}}


def span(i, text, start=10):
    return {"span_id": f"s{i}", "document_id": f"d{i}", "canonical_source_id": f"https://x.org/{i}", "title": f"T{i}",
            "url": f"https://x.org/{i}", "published": "2024", "authors": [], "licence": "CC-BY-4.0", "start": start,
            "end": start + len(text), "text": text, "text_sha256": "ab" * 32, "relevance": 0.9,
            "attribution": ATTR(i)}


def ATTR(i):
    return {"creator": f"Author {i}", "title": f"T{i}", "source": f"https://x.org/{i}", "licence": "CC-BY-4.0",
            "licence_url": "https://creativecommons.org/licenses/by/4.0/", "changes": "Short excerpt."}


def bundle(status="admitted", admitted=None):
    adm = admitted if admitted is not None else (span(1, LONG), span(2, "Gamma results are short."))
    docs = ({"document_id": "d1", "canonical_source_id": "https://x.org/1", "title": "T1", "url": "https://x.org/1", "doi": None,
             "published": "2024", "licence": "CC-BY-4.0", "licence_decision": "allow", "text_sha256": "cd" * 32, "chars": 999,
             "chunks_inspected": 5, "candidate_spans": 3, "admitted_spans": 2},)
    return EvidenceBundle(question="q", examined_full_documents=docs, admitted=tuple(adm), status=status)


class Stub:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def __call__(self, messages, model, pricing, **kw):
        self.calls += 1
        self.messages = messages
        return AskResult(answer=self.text, tokens_used=100, cost_usd=0.01)


def checker(verdicts):
    def call(chk, msgs):
        n = len(json.loads(msgs[1]["content"])["claims"])
        return json.dumps({"claims": [{"id": i, "verdict": verdicts[i], "reason": "r"} for i in range(n)]})
    return call


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("FD_TEST_KEY", "x")


def run(text, verdicts=("supported",), ev=None, gcfg=GCFG, call="default"):
    stub = Stub(text)
    out = fa.answer("q", ev or bundle(), "m", None, chained={}, grounding_cfg=gcfg, cfg=CFG, base_tokens=5, base_cost=0.5,
                    complete_fn=stub, check_call=checker(verdicts) if call == "default" else call)
    return out, stub


def test_success_path_and_citation_binding():
    out, stub = run("Beta caching doubles throughput under heavy load [1]. Gamma results are short [2].",
                    ("supported", "supported"))
    r = out.result
    assert out.fallback is None and r.accepted and r.result_kind == "answer" and r.contract is None
    assert r.tokens_used == 105 and r.cost_usd == pytest.approx(0.51) and out.tokens_used == 105
    assert len(r.sources) == 2 and r.evidence.status == "admitted" and r.claim_check["outcome"] == "checked"
    assert "Beta caching" in stub.messages[0]["content"] and "[2] T2 (published 2024)" in stub.messages[0]["content"]
    c = r.citations[0]
    assert set(c) == {"claim_id", "n", "span_id", "document_id", "canonical_source_id", "licence", "span", "verdict", "quote",
                      "quote_span", "attribution"}
    assert c["attribution"] == ATTR(1)
    a, b = (int(x) for x in c["quote_span"][6:].split("-"))
    assert b - a == len(c["quote"])
    assert (c["n"], c["span_id"], c["document_id"], c["licence"], c["span"], c["verdict"]) == (
        1, "s1", "d1", "CC-BY-4.0", f"chars=10-{10 + len(LONG)}", "supported")
    assert len(r.citations) == 2 and r.citations[1]["span_id"] == "s2"


def test_unsupported_removed_and_not_cited():
    out, _ = run("Beta caching doubles throughput [1]. Gamma is huge [2].", ("supported", "unsupported"))
    r = out.result
    assert "Gamma is huge" not in r.answer and "Note: removed." in r.answer
    assert [c["n"] for c in r.citations] == [1]


def test_partial_marked():
    out, _ = run("Beta caching doubles throughput [1]. Gamma is short [2].", ("supported", "partial"))
    assert "Gamma is short [2]. (PARTIAL)" in out.result.answer or "(PARTIAL)" in out.result.answer
    assert [c["verdict"] for c in out.result.citations] == ["supported", "partial"]


def test_zero_valid_citations_fallback():
    out, _ = run("No citations here. Or bad one [9].")
    assert out.result is None and out.fallback == "no_valid_citations" and out.tokens_used == 105


def test_out_of_range_removed():
    out, _ = run("Beta caching doubles throughput [1, 7].")
    assert out.fallback is None and "7" not in out.result.answer.split("Note")[0]
    assert all(c["n"] == 1 for c in out.result.citations)


def test_checker_unavailable():
    out, _ = run("Beta caching doubles throughput [1].", call=lambda chk, msgs: "garbage")
    assert out.result is None and out.fallback == "checker_unavailable"


def test_claim_check_insufficient():
    out, _ = run("Beta caching doubles throughput [1].", ("unsupported",))
    assert out.result is None and out.fallback == "claim_check_insufficient"


@pytest.mark.parametrize("g", [None, {"claim_check": {"enabled": False}}, {}])
def test_claim_check_disabled_no_model_call(g):
    out, stub = run("x [1].", gcfg=g)
    assert stub.calls == 0 and out.fallback == "claim_check_disabled" and out.result is None and out.tokens_used == 5


def test_not_grounded_no_model_call():
    out, stub = run("x [1].", ev=bundle(status="insufficient", admitted=()))
    assert stub.calls == 0 and out.fallback == "insufficient_full_document_evidence"
    out, stub = run("x [1].", ev=bundle(status="corpus_unusable", admitted=()))
    assert stub.calls == 0 and out.fallback == "full_document_corpus_unusable"


def test_quote_cap_and_substring():
    for cap in (20, 60, 200):
        q = fa.pick_quote(LONG, "Beta caching doubles throughput under heavy load", cap)
        assert q in LONG and 0 < len(q) <= cap and q != LONG
    q = fa.pick_quote(LONG, "Beta caching doubles throughput", 200)
    assert q.startswith("Beta caching")
    assert fa.pick_quote("Short span.", "anything", 200) == "Short span."
    out, _ = run("Beta caching doubles throughput [1].")
    assert all(len(c["quote"]) <= 60 and c["quote"] in LONG for c in out.result.citations)


def test_public_evidence_no_span_text():
    out, _ = run("Beta caching doubles throughput [1]. Gamma results are short [2].", ("supported", "supported"))
    ev = out.result.evidence
    pub = fa.public_evidence(ev, out.result.citations)
    blob = json.dumps(pub)
    assert pub["evidence_status"] == "admitted" and len(pub["admitted_evidence"]) == 2
    assert all("text" not in a for a in pub["admitted_evidence"]) and pub["examined_full_documents"][0]["chars"] == 999
    for a in ev.admitted:
        t = a["text"]
        for i in range(0, max(1, len(t) - 299)):
            if len(t) >= 300:
                assert t[i:i + 300] not in blob
    assert "Filler sentence" not in blob.replace(out.result.citations[0]["quote"], "")


def test_uncited_sentences_never_published():  # lead D-055: claim_check judges cited sentences only
    out, _ = run("Beta caching doubles throughput [1]. Uncited claim that the moon is cheese.\nAnother uncited line.")
    a = out.result.answer
    assert "moon" not in a and "Another uncited" not in a and "Beta caching doubles throughput [1]." in a
    assert "Note: unchecked parts left out." in a and out.result.claim_check["uncited_dropped"] == 2


def test_only_cited_and_verified_no_unchecked_note():
    out, _ = run("Beta caching doubles throughput [1].")
    assert "unchecked" not in out.result.answer and out.result.claim_check["uncited_dropped"] == 0


def test_citation_after_full_stop_is_attached_to_its_sentence():
    assert fa.attach_trailing_citations("Beta caching doubles throughput. [1] Next.") == \
        "Beta caching doubles throughput [1]. Next."
    assert fa.attach_trailing_citations("A holds. [1][2]\nB.") == "A holds [1][2].\nB."
    out, _ = run("Beta caching doubles throughput. [1]")
    assert "Beta caching doubles throughput [1]." in out.result.answer


def test_model_insufficient_marker_means_no_answer_and_no_check():
    cfg = {**CFG, "answer": {**CFG["answer"], "insufficient_marker": "INSUFFICIENT_EVIDENCE",
                              "system_prompt": "Say {insufficient_marker} if off-question.\n{sources}"}}
    calls = []
    out = fa.answer("q", bundle(), "m", None, chained={}, grounding_cfg=GCFG, cfg=cfg,
                    complete_fn=lambda *a, **k: AskResult("INSUFFICIENT_EVIDENCE", 3, 0.0),
                    check_call=lambda *a: calls.append(a))
    assert out.result is None and out.fallback == "model_insufficient" and calls == []  # C2/F14: its own token
    assert CFG["fallback_tokens"]["model_insufficient"] != CFG["fallback_tokens"]["insufficient"]
    old = {**cfg, "fallback_tokens": {k: v for k, v in cfg["fallback_tokens"].items() if k != "model_insufficient"}}
    out = fa.answer("q", bundle(), "m", None, chained={}, grounding_cfg=GCFG, cfg=old,
                    complete_fn=lambda *a, **k: AskResult("INSUFFICIENT_EVIDENCE", 3, 0.0), check_call=lambda *a: None)
    assert out.fallback == "insufficient_full_document_evidence"  # key absent: whole-draft behaviour unchanged


def test_partial_only_answer_falls_back_d7():
    out, _ = run("Beta caching doubles throughput [1].", ("partial",))
    assert out.result is None and out.fallback == "claim_check_insufficient"


def test_verbatim_budget_present_in_summary():
    out, _ = run("Beta caching doubles throughput [1].")
    b = out.result.claim_check["verbatim_budget"]
    assert set(b) == {"quoted", "generated_verbatim", "total", "cap"} and b["total"] < b["cap"]


def test_long_verbatim_copy_of_a_passage_is_never_published():
    long_span = bundle().admitted[0]["text"]
    if len(long_span) <= CFG["answer"]["quote_max_chars"]:
        pytest.skip("fixture span shorter than the quote cap")
    out, _ = run(long_span.strip() + " [1]. Beta caching doubles throughput [1].", ("supported", "supported"))
    assert out.result is None or long_span.strip()[:CFG["answer"]["quote_max_chars"] + 1] not in out.result.answer


def test_quote_budget_drops_lowest_ranked_quotes_keeps_span_refs():
    recs = [{"id": k, "sentence": f"Beta caching doubles throughput {k} [1]. ", "citations": [1],
             "verdict": "partial" if k == 0 else "supported"} for k in range(6)]
    srcs = fa.sources_from(bundle())
    cites = fa.build_citations(recs, srcs, 60, total_cap=130)
    quoted = [c for c in cites if c["quote"]]
    assert sum(len(c["quote"]) for c in quoted) <= 130 and len(quoted) < len(cites)
    assert all(c["span_id"] and c["span"] and c["attribution"] for c in cites)
    assert next(c for c in cites if c["claim_id"] == 0)["quote"] is None  # the partial claim is ranked lowest
    assert all(c["quote_span"] is None for c in cites if c["quote"] is None)


def test_passage_without_attribution_is_never_cited():
    srcs = fa.sources_from(bundle(admitted=(dict(span(1, LONG), attribution=None),)))
    recs = [{"id": 0, "sentence": "Beta [1].", "citations": [1], "verdict": "supported"}]
    assert fa.build_citations(recs, srcs, 60, 700) == []
