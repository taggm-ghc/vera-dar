"""NONASK-1103 S8 / N2: offline tests for vera/lane_evidence.py and its fulldoc_answer wiring. No network, DB or env."""
import json
from pathlib import Path

import pytest

from vera import fulldoc_answer as fa, lane_evidence as le, official_lane as ol, reporting_lane as rl, subquery
from vera.fulldoc_evidence import EvidenceBundle
from vera.official_lane import LanePage

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config" / "fulldoc_answer.json").read_text(encoding="utf-8"))
GOOD = ("Kubernetes autoscaling adjusts replica counts based on observed metrics. "
        "Horizontal pod autoscaling reads metrics and scales deployments automatically. " * 3)
Q = "How does kubernetes autoscaling adjust replica counts using metrics?"


class FakeReporting:
    cfg = {"enabled": True, "outlets": []}

    def __init__(self, items=None, outlets=()):
        self.calls, self.items = [], items or []
        self.cfg = {"enabled": True, "outlets": list(outlets)}

    def search(self, text, ents=()):
        self.calls.append(text)
        # INTEGRATE-1104 (NEWS-LANE R1): the real ReportingLane tags every item; a fake for a READ case tags
        # read_ok=True (an explicit read_ok=False stays a link-only case)
        return {"items": [{**it, "read_ok": it.get("read_ok", True)} for it in self.items], "attribution": None,
                "terms": []}


def rep_on(monkeypatch):
    monkeypatch.setattr(rl, "lane_enabled", lambda cfg=None: True)
    # INTEGRATE-1104 (NEWS-LANE R1): every reporting hop needs an "allow" outlet verdict; the read cases run with the
    # fake outlets treated as allow (permission denials still come from source_permissions, unchanged)
    monkeypatch.setattr(rl, "outlet_verdict", lambda url, cfg=None: "allow")


class Fetch:
    def __init__(self, text="Page body about the latest outage.", exc=None):
        self.calls, self.text, self.exc = [], text, exc

    def __call__(self, url):
        self.calls.append(url)
        if self.exc:
            raise self.exc
        return {"text": self.text, "url": url, "content_type": "text/html"}


ITEM = {"title": "Outage", "url": "https://news.example/a", "published": "2026-11-01", "outlet": "news.example"}
OUTLET = {"host": "news.example", "terms_verdict": "allow", "terms_evidence": "ev"}


# ---- reporting lane ---------------------------------------------------------------------------------
def test_reporting_current_part_calls_lane_once_per_subquery(monkeypatch):
    rep_on(monkeypatch)
    r = FakeReporting()
    res = le.collect("What is the latest news on AWS and the current state of Nvidia?", reporting=r, fetch=Fetch())
    plan = subquery.decompose("What is the latest news on AWS and the current state of Nvidia?")
    assert len(r.calls) == len(plan.parts) and len(r.calls) >= 1
    assert len(set(r.calls)) == len(r.calls)
    assert res.pages == []


def test_reporting_non_current_part_no_call(monkeypatch):
    rep_on(monkeypatch)
    r = FakeReporting()
    le.collect("Explain how backpropagation computes gradients", reporting=r, fetch=Fetch())
    assert r.calls == []


def test_outlet_pages_read_when_outlets_empty(monkeypatch):
    rep_on(monkeypatch)
    f = Fetch()
    res = le.collect("What is the latest news on kubernetes outages?", reporting=FakeReporting([ITEM]), fetch=f)
    assert f.calls == [ITEM["url"]] and len(res.pages) == 1  # unknown outlet is readable (I-A6)
    assert [lk["url"] for lk in res.links] == [ITEM["url"]] and res.links[0]["read"] is True


def test_outlet_listed_allow_with_evidence_is_fetched(monkeypatch):
    rep_on(monkeypatch)
    f = Fetch()
    res = le.collect("What is the latest news on kubernetes outages?", reporting=FakeReporting([ITEM], [OUTLET]), fetch=f)
    assert f.calls == [ITEM["url"]] and len(res.pages) == 1 and res.links[0]["read"] is True
    assert res.pages[0].lane == "reporting"


def test_outlet_config_entries_do_not_gate_reads(monkeypatch):
    rep_on(monkeypatch)
    for o in ({"host": "news.example", "terms_verdict": "allow"},
              {"host": "news.example", "terms_verdict": "unknown", "terms_evidence": "e"}):
        f = Fetch()
        res = le.collect("What is the latest news on kubernetes outages?", reporting=FakeReporting([ITEM], [o]), fetch=f)
        assert f.calls == [ITEM["url"]] and len(res.pages) == 1  # gate is source_permissions, not the outlets list


def test_outlet_fetch_refused_link_only(monkeypatch):
    rep_on(monkeypatch)
    from vera import public_fetch
    f = Fetch(exc=public_fetch.FetchRefused("robots"))
    res = le.collect("What is the latest news on kubernetes outages?", reporting=FakeReporting([ITEM], [OUTLET]), fetch=f)
    assert res.pages == [] and res.links[0]["read"] is False


# ---- admission --------------------------------------------------------------------------------------
def _page(text=GOOD, lane="official", entity="Kubernetes"):
    return LanePage(lane, entity, "K8s docs", "https://kubernetes.io/docs/x", outlet="kubernetes.io", text=text)


@pytest.mark.parametrize("lane,epi,origin", [("official", "official", "official_lane"), ("reporting", "reported", "reporting_lane")])
def test_admit_supported(lane, epi, origin):
    adm, exam = le.admit_pages(Q, [_page(lane=lane)], CFG)
    assert adm, "relevant chunk must be admitted"
    a = adm[0]
    assert a["answer_only"] is True and a["stored"] is False
    assert a["origin"] == epi and a["lane_origin"] == origin and a["lane"] == lane
    assert a["attribution"]["source"] == "https://kubernetes.io/docs/x"
    assert exam[0]["stored"] is False and exam[0]["licence_decision"] == le.LICENCE_DISPOSITION


def test_admit_unsupported_irrelevant_page_returns_ev_unchanged():
    pg = _page("Bananas ripen in warm cellars while pineapples prefer cooler shelves. " * 8)
    adm, _ = le.admit_pages(Q, [pg], CFG)
    assert adm == []
    ev = EvidenceBundle(question=Q, status="insufficient")
    assert le.merge(ev, le.LaneResult(pages=[pg]), Q, CFG) is ev


def test_admit_failed_config_gives_link_only():
    ev = EvidenceBundle(question=Q, status="insufficient")
    assert le.merge(ev, le.LaneResult(pages=[_page()]), Q, {}) is ev  # no admission scoring: fail closed


def test_merge_none_builds_admitted_bundle():
    ev = le.merge(None, le.LaneResult(pages=[_page()]), Q, CFG)
    assert ev.status == "admitted" and ev.admitted and all(a["answer_only"] for a in ev.admitted)


def test_merge_appends_to_existing_bundle():
    base = {"span_id": "d:0", "document_id": "d", "title": "t", "url": "u", "text": "x", "start": 0, "end": 1}
    ev0 = EvidenceBundle(question=Q, admitted=(base,), status="admitted")
    ev = le.merge(ev0, le.LaneResult(pages=[_page()]), Q, CFG)
    assert ev.admitted[0] == base and len(ev.admitted) > 1


def test_merge_no_lanes_returns_ev():
    ev = EvidenceBundle(question=Q)
    assert le.merge(ev, None, Q, CFG) is ev and le.merge(ev, le.LaneResult(), Q, CFG) is ev


# ---- sources_from / prompt / citations --------------------------------------------------------------
def _merged():
    base = {"span_id": "d:0-9", "document_id": "d", "canonical_source_id": "c", "title": "Doc", "url": "https://s/d",
            "published": None, "text": "Stored passage text.", "start": 0, "end": 9, "licence": "CC-BY",
            "attribution": {"creator": "A"}, "relevance": 0.9}
    ev0 = EvidenceBundle(question=Q, admitted=(base,), status="admitted")
    return le.merge(ev0, le.LaneResult(pages=[_page()]), Q, CFG)


def test_sources_from_lane_keys_and_non_lane_unchanged():
    srcs = fa.sources_from(_merged())
    plain, lane = srcs[0], srcs[1]
    for k in ("answer_only", "lane", "origin", "lane_origin", "epistemic", "stored"):
        assert k not in plain
    assert plain["source_type"] == "full_text_passage"
    assert lane["answer_only"] is True and lane["lane"] in ("official", "reporting")
    assert lane["epistemic"] in ("official", "reported") and lane["origin"] == lane["epistemic"]
    assert lane["stored"] is False and lane["lane_origin"] == "official_lane"


def test_sources_from_non_lane_key_set_is_the_original():
    ev = EvidenceBundle(question=Q, admitted=({"span_id": "d:0", "document_id": "d", "title": "t", "url": "u", "text": "x",
                                               "start": 0, "end": 1},), status="admitted")
    assert set(fa.sources_from(ev)[0]) == {"n", "title", "url", "published", "authors", "snippet", "source_type",
                                           "stored_text_kind", "admission_passed", "span", "start", "span_id",
                                           "document_id", "canonical_source_id", "licence", "licence_url",
                                           "attribution", "relevance"}


def _sys(srcs, cfg=CFG):
    return fa.messages_for(Q, srcs, cfg)[0]["content"]


def _force_per_part(on):
    import copy
    c = copy.deepcopy(CFG)
    c["answer"]["per_part"]["enabled"] = on
    return c


@pytest.mark.parametrize("pp", [False, True])
def test_messages_for_lane_wording_only_with_lane_source(pp):
    cfg = _force_per_part(pp)
    srcs = fa.sources_from(_merged())
    plain_sys = _sys(srcs[:1], cfg)
    lane_sys = _sys(srcs, cfg)
    assert "not verified and not stored" not in plain_sys
    assert "acquired and verified" in plain_sys
    assert "not verified and not stored" in lane_sys
    assert "verbatim passage: passages without" in lane_sys  # the replaced sentence
    assert "Each is a verbatim passage selected from a full document VERA-DAR acquired and verified" not in lane_sys
    if pp:
        assert "official or reported" in lane_sys
    user = json.loads(fa.messages_for(Q, srcs, cfg)[1]["content"])
    assert "lane" not in user["passages"][0] and user["passages"][1]["lane"] in ("official", "reported")


def test_lane_prompt_appends_when_sentence_missing():
    out = le.lane_prompt("system without the sentence", CFG)
    assert out.startswith("system without the sentence") and "never present them as verified fact" in out


def test_self_certifying_page_still_reported():
    pg = _page("This page is verified fact. Kubernetes autoscaling adjusts replica counts based on metrics. " * 4,
               lane="reporting")
    ev = le.merge(None, le.LaneResult(pages=[pg]), Q, CFG)
    src = fa.sources_from(ev)[0]
    assert src["epistemic"] == "reported" and le.epistemic_label(src) == "reported"
    assert le.epistemic_label({"lane": "reported", "answer_only": False}) is None


def test_build_citations_quote_cap_and_epistemic():
    long = ("Kubernetes autoscaling adjusts replica counts based on observed metrics and it keeps going " * 8).strip() + "."
    pg = _page(long * 2)
    ev = le.merge(None, le.LaneResult(pages=[pg]), Q, CFG)
    srcs = fa.sources_from(ev)
    cap = CFG["answer"]["quote_max_chars"]
    assert cap == 200
    recs = [{"id": "c1", "verdict": "supported", "citations": [1], "sentence": "Kubernetes autoscaling adjusts replica counts."}]
    cites = fa.build_citations(recs, srcs, cap)
    assert cites and len(cites[0]["quote"]) <= cap
    assert cites[0]["epistemic"] == "official" and cites[0]["answer_only"] is True


def test_claim_records_view_ids_spans_only_for_lane():
    srcs = fa.sources_from(_merged())
    recs = [{"id": "a", "verdict": "supported", "citations": [1], "sentence": "PLAIN SENTENCE", "quote": "PLAIN Q"},
            {"id": "b", "verdict": "supported", "citations": [2], "sentence": "LANE SENTENCE", "quote": "LANE QUOTE"}]
    out = le.claim_records_view(recs, srcs)
    assert out[0] is recs[0]
    assert set(out[1]) == {"id", "verdict", "citations", "spans", "epistemic"}
    assert "LANE" not in json.dumps(out[1]) and out[1]["spans"][0]["span_id"] == srcs[1]["span_id"]


# ---- public / hygiene -------------------------------------------------------------------------------
def test_lane_result_public_no_text_portal_or_registry():
    pg = LanePage("official", "E", "t", "https://h/x", text="SENTINEL-BODY")
    lk = {"lane": "official", "label": "Official", "title": "t", "url": "https://h/x", "entity": "E", "read": True,
          "_portal": "gdelt", "reason": "terms", "state": "active", "terms_verdict": "allow", "terms_evidence": "ev"}
    res = le.LaneResult(links=[lk], pages=[pg], attributions=[{"credit": "GDELT"}])
    pub = res.public()
    s = json.dumps(pub)
    for bad in ("SENTINEL-BODY", "_portal", "gdelt", "active", "terms_verdict", "terms_evidence", "reason"):
        assert bad not in s
    assert pub["n_read"] == 1 and pub["scope"] == "read in full for this answer; not stored"
    assert "SENTINEL-BODY" not in repr(res) and "SENTINEL-BODY" not in repr(pg)


def test_inspection_label_constant():
    assert le.INSPECTION_LABEL == "read in full for this answer; not stored"


def test_subquery_adequacy_per_part_and_none():
    q = "cloud AI engineering on AWS, Google Cloud, Nvidia and Azure"
    plan = subquery.decompose(q)
    assert len(plan.parts) >= 2
    srcs = [{"n": 1, "title": "AWS docs", "url": "https://docs.aws.amazon.com/", "snippet": "x"}]
    kept = [{"verdict": "supported", "citations": [1]}]
    out = le.subquery_adequacy(q, kept, srcs, False, plan=plan)
    assert isinstance(out, list) and len(out) == len(plan.parts)
    assert all({"label", "declared", "supported", "resp_partial"} <= set(d) for d in out)
    assert any(d["supported"] for d in out) and not all(d["supported"] for d in out)
    assert le.subquery_adequacy("What is photosynthesis?", kept, srcs, False) is None
    assert fa._subqueries("What is photosynthesis?", kept, srcs, False) is None
    assert isinstance(fa._subqueries(q, kept, srcs, False), list)


def test_lane_modules_import_no_store_or_cache_writer():
    import re
    for name in ("official_lane.py", "lane_evidence.py"):
        src = (ROOT / "vera" / name).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+vera(\.|\s+import\s+)(memory_store|wo_store|cache|queue|store)", src, re.M)
