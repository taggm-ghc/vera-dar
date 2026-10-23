"""P0DTA-1023 end-to-end regression of the live discovery-to-acquisition failure (build 261d039).

Live: "Which are the prevailing proven-useful archetypes for agentic AI orchestrated agents?" -> 9 corpus documents,
0 passages admitted, both triggers fired, a portal failed, 2 candidates not acquired (one arXiv:2605.23989v1).
OFFLINE: the unchanged test rig (tests/_pipeline_support.py: fake deps, fake answer writer and claim checker, no DB,
no socket); every portal and HTTP response is faked; a fake clock advances only through the arXiv crawl-delay waits.
Synthetic text only (tests/fulltext_synth.py words; tests/arxiv_synth.py page shapes); only the record identity is real.
The REAL licence gate, arXiv route, admission and claim-check plumbing decide. No config file and no quota is changed.
"""
import copy

import pytest

from tests import acq_support as A
from tests import arxiv_synth as X
from tests import fulltext_synth as fs
from tests._pipeline_support import GROUND, PRICING, DiscSearch, Router, _decide, h, rig  # noqa: F401  (h, rig: fixtures)
from vera import acq_discovered, arxiv_fulltext, ask_service, inline_acquire, search_providers
from vera import research_outcome as ro

Q_LIVE = "Which are the prevailing proven-useful archetypes for agentic AI orchestrated agents?"
ROBOTS = "https://arxiv.org/robots.txt"
OTHER_DOI = "10.9999/p0dta.agentic.closed"  # the second (OpenAlex) candidate: no open-access route
AGENTIC = ["agentic", "orchestrated", "agents", "archetypes", "prevailing", "proven", "useful", "orchestration",
           "planner", "supervisor", "worker", "pipeline", "evaluation", "deployments", "patterns"]
NEAR = ["agents", "temperature", "ocean", "glacier", "aerosol", "carbon", "anomaly", "atmosphere", "forcing",
        "ensemble", "radiative", "circulation", "projection", "warming", "precipitation"]  # one shared word only


class Clock:
    def __init__(self):
        self.t = 5000.0

    def __call__(self):
        return self.t


@pytest.fixture
def live(rig, monkeypatch):  # noqa: F811
    monkeypatch.setitem(fs.TOPICS, "agentic", AGENTIC)
    monkeypatch.setitem(fs.TOPICS, "near", NEAR)
    acq_discovered.reset_for_tests()
    acq_discovered._ROBOTS.clear()  # cold robots cache
    getattr(arxiv_fulltext, "_HOST_LAST", {}).clear()  # absent before P0DTA-ARX
    clk = Clock()
    waits = []
    monkeypatch.setattr(arxiv_fulltext, "_sleep", lambda s: (waits.append(float(s)), setattr(clk, "t", clk.t + float(s))))
    real = inline_acquire.acquire
    monkeypatch.setattr(inline_acquire, "acquire", lambda q, a, **k: real(q, a, **{"clock": clk, **k}))
    from vera import research_record as rr
    seen, real_assess = [], rr.assess
    monkeypatch.setattr(rr, "assess", lambda rows, *a, **k: seen.append(list(rows or ())) or real_assess(rows, *a, **k))
    rig.clk, rig.waits, rig.assessed = clk, waits, seen
    yield rig
    getattr(arxiv_fulltext, "_HOST_LAST", {}).clear()  # absent before P0DTA-ARX
    acq_discovered._ROBOTS.clear()


def _ask(rig):
    # 9 corpus documents, each sharing one word with the question: candidate passages, none admitted
    rig.h.bundle(docs=tuple((f"syn_near{i}", "near", f"10.9999/p0dta.near{i}") for i in range(9)))
    rig.h.install()
    rig.cfg["inline"]["use_discovered"] = True
    html = X.html_resp(X.latexml_html(topic="agentic"))
    web = Router({}, {}, {**X.routes(html=html),
                          ROBOTS: A.Resp(body=b"User-agent: *\nCrawl-delay: 15\n", ctype="text/plain")})

    def search(q, n, t):  # acquisition search: one portal times out, the others return nothing new
        search_providers.record_portal("europepmc", "timeout", 0)
        search_providers.record_portal("openalex", "completed", 0)
        return []
    rig.setup(web=web, cands=[], search=search)
    rig.h.answer_text = "Supervisor and worker agents are the prevailing orchestrated archetypes [1]."
    ddg = {"provider": "duckduckgo_lite", "url": X.HTML_URL, "doi": None, "title": X.TITLE,
           "snippet": "agentic orchestrated agents archetypes"}
    oa = {"provider": "openalex", "doi": OTHER_DOI, "url": f"https://doi.org/{OTHER_DOI}", "openalex_id": "https://openalex.org/W9",
          "title": "SYNTHETIC agentic orchestration patterns", "snippet": "agentic orchestrated agents archetypes",
          "stored_text_kind": "abstract_metadata"}
    ds = DiscSearch([ddg, oa])
    before = copy.deepcopy(rig.cfg)
    _decide(rig.mp, True, "layer1", "ok", "")
    r = ask_service.answer_in_scope(Q_LIVE, "m", PRICING, grounding=(ds, GROUND[1]))
    return r, web, before


def test_live_question_acquires_versioned_arxiv_html_and_answers_with_that_citation(live):
    r, web, before = _ask(live)
    ia = (r.contract or {}).get("inline_acquisition") or {}
    research = (r.contract or {}).get("research") or {}
    # triggers: insufficiency AND freshness both fired on the 9-document, 0-admitted base evidence
    trig = research.get("triggers") or []
    assert "insufficient" in trig and any(t in trig for t in ("stale", "freshness_unknown")), research
    base_rows = live.assessed[0]  # the base evidence the triggers were assessed on: 9 documents, none admitted
    assert len({str(d.get("document_id")) for d in base_rows}) == 9
    assert all(str(d.get("document_id")).startswith("syn_near") and not d.get("admitted") for d in base_rows)
    # discovery -> selection: the DDG arXiv html URL is a routed candidate with the versioned id
    cands = ia.get("candidates") or []
    arx = [c for c in cands if c.get("arxiv_id") == X.AID + X.VER]
    assert len(arx) == 1, ia
    # fetch of the versioned HTML route under a 15 s crawl-delay on a cold robots cache (full delay honoured)
    urls = [c[0] for c in web.calls]
    assert ROBOTS in urls and X.ABS_URL in urls and X.HTML_URL in urls, urls
    assert [round(w) for w in live.waits] == [15, 15]
    # licence gate (CC BY on the abs page) -> admission -> stored provenance pins the versioned html route
    assert arx[0]["outcome"] == "admitted" and arx[0]["status"] == "admitted", arx[0]
    (plan,) = live.store.plans
    assert plan["source"]["provenance"]["acquisition"]["arxiv"]["html_url"] == X.HTML_URL
    # a cited answer whose citation is that paper
    assert r.result_kind == "answer", (r.result_kind, r.answer)
    cited = {str(s.get("document_id")) for s in r.sources or ()}
    adm = {str(a.get("document_id")) for a in r.evidence.admitted}
    assert cited and cited <= adm and all(d.startswith("acq-") for d in cited)
    assert plan["source"]["provenance"]["document_id"] in cited
    assert {(s.get("url"), s.get("title")) for s in r.sources} == {(X.ABS_URL, X.TITLE)}
    # every candidate: exactly one terminal status and a reason; the portal failure is reported
    for c in cands:
        assert c.get("status") in ro.DIAG_STATUSES and c.get("status_reason"), c
    assert sum(ia.get("diag_status_counts", {}).values()) == len(cands)
    assert any(p.get("status") == "timeout" for p in ia.get("portals") or []), ia.get("portals")
    # no quota value changed
    assert live.cfg == before
