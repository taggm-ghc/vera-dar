"""STRICT-C-1028 (A1c): untrusted parse sites use vera.strict_json. Per module: a duplicate-key (or NaN) input behaves
exactly like malformed input; a normal input is unchanged. Fakes only: no network, no DB, no env values."""
import json
from types import SimpleNamespace

import pytest

from tests import fulltext_synth as fs
from tests.test_acq_search_profile import FIX, _prov
from tests.test_item71_doaj import CFG as _DCFG, DOAJ
from scripts import ingest_fulltext as cli
from vera import corpus_description as cd
from vera import fulldoc_corpus
from vera import fulltext_loader as fl
from vera import search_providers as sp
from vera.gate_a import score_candidates
from vera.m3.llm import LLMError, OpenAIJSONClient
from vera.m6.judge import parse_json_loose

DUP = '{"a": 1, "a": 2}'
BAD = "not json"


def _resp(text):
    return SimpleNamespace(text=text, status_code=200)


# ---- J1: providers
def test_openalex_dup_key_same_as_malformed():
    p = _prov([])
    msgs = []
    for body in (BAD, '{"results": [], "results": [{}]}', '{"results": [], "x": NaN}'):
        with pytest.raises(sp.ProviderResponseError) as e:
            p.parse(_resp(body), 5)
        msgs.append(type(e.value))
    assert len(set(msgs)) == 1
    assert len(p.parse(_resp(FIX), 5)) == len(p.parse(_resp(json.dumps(json.loads(FIX))), 5)) > 0


def test_doaj_dup_key_same_as_malformed():
    p = sp.DOAJProvider(DOAJ, transport=None, query_terms=_DCFG.get("query_terms") if isinstance(_DCFG, dict) else None,
                        sleep=lambda s: None)
    for body in (BAD, '{"results": [], "results": [{}]}'):
        with pytest.raises(sp.ProviderResponseError):
            p.parse(_resp(body), 5)
    assert p.parse(_resp(json.dumps({"total": 0, "results": []})), 5) == []
    assert p._zero_hits(_resp('{"results": []}')) is True
    assert p._zero_hits(_resp('{"results": [], "results": [1]}')) is False  # == malformed


def test_europepmc_dup_key_same_as_malformed():
    ecfg = {"search_url": "stub://x"}
    ok = json.dumps({"resultList": {"result": [{"doi": "10.1/X", "pmcid": "PMC1", "isOpenAccess": "Y"}]}})
    dup = ok[:-1] + ", \"resultList\": {}}"
    for fn in (lambda tx: sp.europepmc_search("q", 5, 1, ecfg, transport=tx),
               lambda tx: sp.europepmc_ids_lookup([("pmcid", "PMC1")], 1, ecfg, transport=tx)):
        errs = []
        for body in (BAD, dup):
            with pytest.raises(ValueError) as e:
                fn(lambda *a: _resp(body))
            errs.append(type(e.value))
        assert errs[0] == errs[1]
    assert sp.europepmc_ids_lookup([("pmcid", "PMC1")], 1, ecfg, transport=lambda *a: _resp(ok))["PMC1"]["doi"] == "10.1/x"


# ---- J2: manifest and conversion record
def test_manifest_items_dup_key_is_none_like_malformed():
    assert fulldoc_corpus._manifest_items(BAD.encode()) is None
    assert fulldoc_corpus._manifest_items(b'{"sources": [{"a": 1}], "sources": []}') is None
    assert fulldoc_corpus._manifest_items(b'{"sources": [{"a": 1}, 2]}') == [{"a": 1}]


def test_conversion_record_dup_key_refused_like_malformed(tmp_path):
    m = fs.write_corpus(tmp_path)
    assert cli.main(["convert", str(m), "--raw-dir", str(tmp_path / "raw")]) in (0, 2)
    item = json.loads(m.read_text())["sources"][0]
    cfg = fl.load_config()
    text, meta = fl.load_text(item, m.parent, cfg)  # normal: unchanged
    mp = (m.parent / item["text_file"]).with_suffix(".meta.json")
    good = mp.read_text()
    codes = []
    for bad in ("{ not json", good.rstrip().rstrip("}") + ', "text_sha256": "0"}'):
        mp.write_text(bad)
        with pytest.raises(fl.Refused) as e:
            fl.load_text(item, m.parent, cfg)
        codes.append(e.value.code)
    assert codes[0] == codes[1]


# ---- J3: LLM replies
def _cands():
    return [{"url": "https://s0.com", "title": "T0", "snippet": "x"}]


def test_gate_a_dup_key_defers_like_invalid_json():
    row = {"id": 0, "query_fit": 0.9, "evidentiary_value": 0.9, "cost_risk": 0.1, "uncertainty": "low",
           "rationale": {"query_fit": "f", "evidentiary_value": "e", "cost_risk": "c"}}
    ok = json.dumps({"scores": [row]})
    dup = ok[:-1] + ', "scores": []}'
    base = score_candidates("q", _cands(), lambda m: (BAD, 1, 1))[0]["gate_a_decision"]
    assert score_candidates("q", _cands(), lambda m: (dup, 1, 1))[0]["gate_a_decision"] == base == "defer"
    assert "gate_a_decision" in score_candidates("q", _cands(), lambda m: (ok, 1, 1))[0]


def _llm_client(content):
    c = OpenAIJSONClient.__new__(OpenAIJSONClient)
    # FREEGEN-IMPL-1028 fail-closed cost: an unpriced model raises, so the fixture carries an explicit 0 record
    from vera.pricing.config import PricingRecord
    c.model, c.timeout_s = "m", 1.0
    c._pricing = PricingRecord(provider="fake", model="m", input=0.0, output=0.0, source_url="u",
                               retrieved_at="2026-10-28T00:00:00", effective_from="2026-10-28")
    from vera.m3.llm import CallLog
    c.log = CallLog()
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]))))
    return c


def test_m3_llm_dup_key_is_llmerror_like_non_json():
    for bad in (BAD, DUP):
        with pytest.raises(LLMError, match="non-JSON"):
            _llm_client(bad).complete_json("s", "u", "p")
    assert _llm_client('{"a": 1}').complete_json("s", "u", "p") == {"a": 1}


def test_judge_parse_dup_key_raises_like_malformed():
    for bad in (BAD, DUP, '{"a": NaN}'):
        with pytest.raises(ValueError):
            parse_json_loose(bad)
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose('here: {"a": 1} done') == {"a": 1}


def test_corpus_description_chain_dup_key_fails_like_malformed(monkeypatch):
    import vera.ask_service as asv

    def entry(content):
        cl = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]))))
        return SimpleNamespace(client=lambda: cl, model="m", extra_body=None, label="L")
    monkeypatch.setattr(asv, "_get_client", lambda: None, raising=False)
    for bad in (BAD, '{"summary": "a", "summary": "b"}'):
        with pytest.raises(RuntimeError, match="no provider produced"):
            cd.generate_with_chain([entry(bad)])([])
    assert cd.generate_with_chain([entry('{"summary": "S", "topics": ["t"]}')])([]) == ("S", ["t"], "L")
