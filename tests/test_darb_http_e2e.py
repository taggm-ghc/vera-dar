"""DAR-B-1023-C3: full POST /ask end-to-end (TestClient, in process), discovery -> acquisition -> admission -> answer.
Fakes only: scope classifier, provider search (seam under REAL retrieve_all), Europe PMC/JATS HTTP, answer LLM, claim checker,
in-memory store. Enablement is the SHIPPED config/auto_acquire.json text read through the REAL acq_config.load_config(path).
Negative controls: kill switch env, and a private config copy with use_discovered=false. Fail-loud DB-connect / socket guards."""
import json
import re
import socket

import pytest

from tests._gap_support import (SHIPPED_TEXT, REAL_LOAD_CONFIG, disabled_config_text, http_ask, stored_text, use_config_file)
from tests._pipeline_support import (ABSTRACT_SENTINEL, DOI, PMC, epmc_web, h, passages_sent_to_model, rig)  # noqa: F401
from tests.test_dar_pipeline_audit_gaps import EPMC_FULLTEXT, EPMC_SEARCH, _fetched, _raw_provider_record
from vera import acq_config, ask_service, fulldoc_corpus

INTERNAL = re.compile(r"R-ACQ-|R-LIC-|R-ABSTRACT|gate_a|link_only|openalex\.org|" + re.escape(PMC))


@pytest.fixture
def no_db_no_socket(monkeypatch):
    """Fail-loud: any engine creation / DB driver connect / outbound socket connect during the request is recorded and raises."""
    hits = []

    def boom(kind):
        def _f(*a, **k):
            hits.append(kind)
            raise AssertionError(f"forbidden {kind} attempted")
        return _f
    import sqlalchemy
    monkeypatch.setattr(sqlalchemy, "create_engine", boom("sqlalchemy.create_engine"))
    for mod in ("psycopg2", "psycopg"):
        try:
            m = __import__(mod)
            monkeypatch.setattr(m, "connect", boom(f"{mod}.connect"))
        except ImportError:
            pass
    monkeypatch.setattr(socket.socket, "connect", boom("socket.connect"))
    monkeypatch.setattr(socket, "create_connection", boom("socket.create_connection"))
    return hits


def _insufficient_base(monkeypatch):
    real = ask_service._fulldoc
    tok = fulldoc_corpus.load_config()["fallback_tokens"]["model_insufficient"]
    monkeypatch.setattr(ask_service, "_fulldoc", lambda *a, **k: (None, tok, 0, 0.0, "x")
                        if (k.get("ephemeral") is None and not k.get("acquired_only")) else real(*a, **k))


def _shipped_file(rig, monkeypatch, tmp_path, text):
    """Route every load_config() through the REAL parser on a file holding `text` (the shipped text, byte for byte, unless a
    negative control edits it); the rig's private cfg is the parsed result, so nothing is set in memory."""
    p = tmp_path / "auto_acquire.json"
    p.write_text(text, encoding="utf-8")
    loaded = REAL_LOAD_CONFIG(p)
    assert loaded is not None
    monkeypatch.setattr(acq_config, "load_config", lambda *a, **k: REAL_LOAD_CONFIG(p))
    rig.cfg = loaded
    return p


def _scenario(name, rig, monkeypatch, tmp_path, text):
    """(a) in-scope miss | (b) off-topic decline. Returns (resp, searches)."""
    _shipped_file(rig, monkeypatch, tmp_path, text)
    web, _ = epmc_web()
    rig.setup(web=web, cands=[])
    if name == "in_scope":
        _insufficient_base(monkeypatch)
        return http_ask(rig, monkeypatch, [_raw_provider_record()], offtopic=False)
    return http_ask(rig, monkeypatch, [_raw_provider_record()], offtopic=True)


def _public_text(body):
    """Everything a visitor can see: the answer and the candidates list (NOT the contract trace block)."""
    return json.dumps({"answer": body.get("answer"), "candidates": body.get("candidates"),
                       "sources": body.get("sources")})


@pytest.mark.parametrize("scenario", ["in_scope", "off_topic"])
def test_shipped_config_enables_full_http_discovery_to_answer(scenario, rig, monkeypatch, tmp_path, no_db_no_socket):
    # the shipped file itself (read via the real loader) is what turns the feature on; nothing is flipped in memory
    assert json.loads(SHIPPED_TEXT)["inline"]["use_discovered"] is True
    shipped = REAL_LOAD_CONFIG(acq_config.CONFIG_PATH)
    assert shipped["enabled"] is True and shipped["inline"]["use_discovered"] is True
    resp, searches = _scenario(scenario, rig, monkeypatch, tmp_path, SHIPPED_TEXT)
    assert (tmp_path / "auto_acquire.json").read_text() == SHIPPED_TEXT
    assert rig.cfg["inline"]["use_discovered"] is True and rig.cfg == shipped

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["result_kind"] == "answer", body.get("contract")
    assert not body["contract"].get("offtopic")  # DDFIX-1023/SCOPE: an off_topic verdict takes the research path

    # discovery ran once, through the real retrieve_all, with the provider seam
    assert len(searches) == 1 and searches[0]["providers"]
    # the record was resolved and fetched from the stubbed Europe PMC endpoints, once each
    assert len(_fetched(rig, EPMC_SEARCH)) == 1 and len(_fetched(rig, EPMC_FULLTEXT)) == 1
    assert all(u.startswith("https://www.ebi.ac.uk/europepmc/") for u in rig.web.urls)

    # admission recorded, exactly one discovered candidate, outcome admitted
    ia = body["contract"]["inline_acquisition"]
    assert ia["status"] == "ok" and ia["admitted"] == 1 and ia["persisted"] is True
    disc = ia["discovered"]
    assert [e["outcome"] for e in disc] == ["admitted"] and disc[0]["url"] == f"https://doi.org/{DOI}"
    assert len(rig.outcomes) == 1 and len(rig.store.plans) == 1
    assert len(rig.h.answer_calls) == 1 and len(rig.h.check_calls) == 1

    # the new document: one stored plan; its identity and its spans
    plan = rig.store.plans[0]
    assert plan["source"]["provenance"], "no provenance stored with the new document"
    adm = body["admitted_evidence"]
    assert adm and {a["document_id"] for a in adm} and all(a["document_id"].startswith("acq-") for a in adm)
    doc_ids = {a["document_id"] for a in adm}
    assert len(doc_ids) == 1
    span_ranges = [(s["start_index"], s["end_index"]) for s in plan["spans"]]

    # numbered sources all belong to the newly admitted document and to its stored spans; no abstract-grounded source
    srcs = body["sources"]
    assert srcs and [s["n"] for s in srcs] == list(range(1, len(srcs) + 1))
    for s in srcs:
        assert s["provider"] == "full_document" and s["source_type"] == "full_text_passage"
        assert s["acquisition"] == "full_document_acquired" and s["admission_passed"] is True
        assert s["licence_decision"] == "allow" and s["identifier"] == f"https://doi.org/{DOI}"
        doc, _, rng = s["span_id"].partition(":")
        a, b = (int(x) for x in rng.split("-"))
        assert doc in doc_ids and any(lo <= a < b <= hi for lo, hi in span_ranges), (s["span_id"], span_ranges)  # inside a stored span
        assert s["span"] == "chars=" + rng
    # provenance present on each public source (licence, identifier, retrieval origin) and in the stored row
    assert all(s["licence"] and s["url"] and s["published"] for s in srcs)

    # citations: every [n] marker in the answer is a valid numbered source of the new document
    markers = {int(m) for grp in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", body["answer"]) for m in re.split(r"\s*,\s*", grp)}
    assert markers and markers <= {s["n"] for s in srcs}, (markers, body["answer"])
    cited = [s for s in srcs if s["n"] in markers]
    assert cited and all(s["span_id"].split(":")[0] in doc_ids for s in cited)
    # passages the writer saw were verbatim stored text of the new document, nothing from the abstract
    stored = stored_text(rig)
    sent = passages_sent_to_model(rig)
    assert sent and all(t in stored for t in sent)
    assert ABSTRACT_SENTINEL not in json.dumps(body) and ABSTRACT_SENTINEL not in stored
    assert body["claim_check"]["outcome"] == "checked" and body["claim_check"]["supported"] >= 1

    # public candidates carry no PMCID / abstract / internal codes (and the answer text neither)
    assert body["candidates"] == [] or not INTERNAL.search(json.dumps(body["candidates"]))
    assert ABSTRACT_SENTINEL not in json.dumps(body["candidates"])
    assert not re.search(r"R-ACQ-|R-LIC-|gate_a|link_only", body["answer"])
    assert PMC not in body["answer"] and PMC not in json.dumps(body["candidates"])

    # offline: no DB engine/driver connect and no socket connect was attempted
    assert no_db_no_socket == []


def _assert_not_acquired(rig, resp, searches):
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["result_kind"] != "answer", body.get("answer")
    assert _fetched(rig, EPMC_FULLTEXT) == [] and _fetched(rig, EPMC_SEARCH) == [] and rig.web.urls == []
    assert rig.store.plans == [] and rig.h.answer_calls == []
    assert not body.get("admitted_evidence")
    assert "discovered" not in ((body.get("contract") or {}).get("inline_acquisition") or {})
    assert not any(s.get("acquisition") == "full_document_acquired" for s in body.get("sources") or [])
    assert ABSTRACT_SENTINEL not in json.dumps(body) and PMC not in json.dumps(body)
    assert not INTERNAL.search(json.dumps(body.get("candidates") or []))


@pytest.mark.parametrize("scenario", ["in_scope", "off_topic"])
def test_kill_switch_env_blocks_acquisition_over_http(scenario, rig, monkeypatch, tmp_path, no_db_no_socket):
    killname = REAL_LOAD_CONFIG(acq_config.CONFIG_PATH)["kill_switch_env"]
    assert killname == "VERA_AUTOACQ_DISABLED"
    monkeypatch.setenv(killname, "1")
    resp, searches = _scenario(scenario, rig, monkeypatch, tmp_path, SHIPPED_TEXT)
    _assert_not_acquired(rig, resp, searches)
    assert no_db_no_socket == []


@pytest.mark.parametrize("scenario", ["in_scope", "off_topic"])
def test_use_discovered_false_private_config_copy_blocks_acquisition_over_http(scenario, rig, monkeypatch, tmp_path, no_db_no_socket):
    text = disabled_config_text()
    assert json.loads(text)["inline"]["use_discovered"] is False and json.loads(SHIPPED_TEXT)["inline"]["use_discovered"] is True
    resp, searches = _scenario(scenario, rig, monkeypatch, tmp_path, text)
    assert rig.cfg["inline"]["use_discovered"] is False
    _assert_not_acquired(rig, resp, searches)
    assert no_db_no_socket == []


def test_control_same_request_without_switch_does_answer(rig, monkeypatch, tmp_path, no_db_no_socket):
    """Only the kill-switch env differs from test_kill_switch_*: absent -> answer (guards the controls against vacuity)."""
    monkeypatch.delenv("VERA_AUTOACQ_DISABLED", raising=False)
    resp, _s = _scenario("off_topic", rig, monkeypatch, tmp_path, SHIPPED_TEXT)
    assert resp.json()["result_kind"] == "answer" and len(rig.store.plans) == 1
