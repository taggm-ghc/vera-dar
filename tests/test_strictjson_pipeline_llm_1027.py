"""STRICTJSON-1027 SJ3: pipeline_llm.complete_json strict replies (C3) and M5 _llm_check row hardening (C6).
T8-T12, T10b, T10c. Fake provider client only; no network."""
from types import SimpleNamespace

import pytest

from tests.m45_support import FakeLLM, corpus  # noqa: F401
from vera import pipeline_llm as pl
from vera.m4.evidence_relations import build_relations
from vera.m5.claim_verification import verify_claims
from vera.pipeline_llm import LLMError

CLAIM = "Developers completed the task 55.8% faster in a controlled experiment"


class FakeClient:
    """chat.completions.create returns the queued replies in order; records the number of calls."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        content = self.replies.pop(0)
        return SimpleNamespace(usage=SimpleNamespace(total_tokens=7, prompt_tokens=4, completion_tokens=3),
                               choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def fake(monkeypatch):
    def install(*replies):
        client = FakeClient(replies)
        monkeypatch.setattr(pl, "_client", lambda: client)
        monkeypatch.setattr(pl, "load_model_pricing", lambda: None)
        monkeypatch.setattr(pl, "default_model", lambda: "fake-model")
        return client
    return install


DUP_VERDICTS = '{"verdicts":[{"index":0,"verdict":"unsupported","issue":"x"}],"verdicts":[{"index":0,"verdict":"supported","issue":"x"}]}'
DUP_IN_ROW = '{"verdicts":[{"index":0,"verdict":"unsupported","verdict":"supported","issue":"x"}]}'
CLEAN_SUPPORTED = '{"verdicts":[{"index":0,"verdict":"supported","issue":"x"}]}'


def resp(*claims):
    return {"response_text": "Answer text.", "claims": [{"claim_text": t, "evidence_span_ids": ids} for t, ids in claims]}


def test_t8_duplicate_twice_raises_after_two_attempts(fake):
    client = fake(DUP_VERDICTS, DUP_IN_ROW)
    with pytest.raises(LLMError) as e:
        pl.complete_json("sys", "usr")
    assert client.calls == 2 == pl.MAX_JSON_RETRIES + 1
    assert "duplicate_key" in str(e.value)


def test_t8_tokens_summed_on_retry_success(fake):
    fake(DUP_VERDICTS, CLEAN_SUPPORTED)
    r = pl.complete_json("sys", "usr")
    assert r.tokens == 14


@pytest.mark.parametrize("bad", ['{"score":NaN}', '{"score":1e999}', '{"vеrdicts":[]}'])
def test_t8b_other_strict_rejections_spend_the_retry(fake, bad):
    client = fake(bad, bad)
    with pytest.raises(LLMError):
        pl.complete_json("sys", "usr")
    assert client.calls == 2


def test_t9_duplicate_then_clean_succeeds(fake):
    client = fake(DUP_IN_ROW, CLEAN_SUPPORTED)
    r = pl.complete_json("sys", "usr")
    assert r.data == {"verdicts": [{"index": 0, "verdict": "supported", "issue": "x"}]} and client.calls == 2


def test_t10_verify_claims_through_complete_json_duplicate_never_supported(fake, corpus):  # noqa: F811
    fake(DUP_VERDICTS, DUP_IN_ROW)
    r = verify_claims(resp((CLAIM, ["s1"])), corpus, llm=None)
    c = r["verified_response"]["claims"][0]
    assert c["verification_status"] == "unverified" and c["issues"] == ["llm_verification_unavailable"]
    assert r["verification_status"] != "verified"


@pytest.mark.parametrize("rows", [
    [{"index": 0, "verdict": "unsupported", "issue": "x"}, {"index": 0, "verdict": "supported", "issue": "x"}],
    [{"index": 0, "verdict": "supported", "issue": "x"}, {"index": 0, "verdict": "unsupported", "issue": "x"}],
    [{"index": 0}, {"index": 0, "verdict": "supported", "issue": "x"}],  # verdict-less row still counts (E4)
])
def test_t10b_repeated_index_rejects_reply(corpus, rows):  # noqa: F811
    r = verify_claims(resp((CLAIM, ["s1"])), corpus, llm=FakeLLM(strict={"verdicts": rows}))
    c = r["verified_response"]["claims"][0]
    assert c["verification_status"] != "supported" and c["issues"] == ["llm_verification_unavailable"]


@pytest.mark.parametrize("bad_index", [True, "0", 0.0, None, [0]])
def test_t10c_non_int_index_skipped(corpus, bad_index):  # noqa: F811
    second = "Field data from three companies showed 26% more completed tasks per week"
    rows = [{"index": bad_index, "verdict": "supported", "issue": "x"}, {"index": 1, "verdict": "supported", "issue": "x"}]
    r = verify_claims(resp((CLAIM, ["s1"]), (second, ["s3"])), corpus, llm=FakeLLM(strict={"verdicts": rows}))
    a, b = r["verified_response"]["claims"]
    assert a["verification_status"] == "unverified" and a["issues"] == ["no_llm_verdict_for_claim"]
    assert b["verification_status"] == "supported"


@pytest.mark.parametrize("verdict", ["refuted", "contradicted", "unsupported ", " supported", "yes", "", None, 1])
def test_r2_sj_1_unknown_verdict_never_supported(corpus, verdict):  # noqa: F811
    rows = [{"index": 0, "verdict": verdict, "issue": "x"}]
    r = verify_claims(resp((CLAIM, ["s1"])), corpus, llm=FakeLLM(strict={"verdicts": rows}))
    c = r["verified_response"]["claims"][0]
    assert c["verification_status"] == "unverified" and c["issues"] == ["no_llm_verdict_for_claim"]


def test_r2_sj_1_case_variants_still_accepted(corpus):  # noqa: F811
    rows = [{"index": 0, "verdict": "UNSUPPORTED", "issue": "x"}]
    r = verify_claims(resp((CLAIM, ["s1"])), corpus, llm=FakeLLM(strict={"verdicts": rows}))
    assert r["verified_response"]["claims"][0]["verification_status"] == "unsupported"


def test_t11_build_relations_duplicate_relations_raises(fake, corpus):  # noqa: F811
    dup = ('{"relations":[{"from":"E1","to":"E2","type":"refutes","confidence":0.9,"rationale":"r"}],'
           '"relations":[]}')
    fake(dup, dup)
    with pytest.raises(LLMError):
        build_relations(corpus, llm=None)


def test_t12_clean_replies_unchanged(fake, corpus):  # noqa: F811
    fake(CLEAN_SUPPORTED)
    r = verify_claims(resp((CLAIM, ["s1"])), corpus, llm=None)
    assert r["verified_response"]["claims"][0]["verification_status"] == "supported" and r["verification_status"] == "verified"
    rows = [{"index": 0, "verdict": "supported", "issue": "x"}, {"index": 1, "verdict": "partial", "issue": "y"}]
    second = "Field data from three companies showed 26% more completed tasks per week"
    r2 = verify_claims(resp((CLAIM, ["s1"]), (second, ["s3"])), corpus, llm=FakeLLM(strict={"verdicts": rows}))
    assert [c["verification_status"] for c in r2["verified_response"]["claims"]] == ["supported", "overreach"]
