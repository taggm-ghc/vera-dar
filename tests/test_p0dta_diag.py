"""P0DTA-DIAG: per-candidate acquisition diagnostics. Status recording only: one terminal status + reason + stage."""
import pytest

from tests import acq_support as A
from tests import fulltext_synth as fs
from tests import test_acq_discovered as D
from vera import acq_config, auto_acquire, inline_acquire
from vera import research_outcome as ro
from vera import research_record as rr

R = {"deadline": "R-ACQ-DEADLINE", "cap": "R-ACQ-DEFER_CAP", "known": "R-ACQ-KNOWN", "nodoi": "R-ACQ-NODOI",
     "notoa": "R-ACQ-NOTOA", "lic": "R-ACQ-LICENCE_MISMATCH", "inj": "R-ACQ-INJECT", "unreach": "R-ACQ-UNREACHABLE",
     "irrel": "R-ACQ-IRRELEVANT", "fmt": "R-ACQ-FORMAT_UNSUPPORTED", "rate": "R-ACQ-RATELIMITED",
     "ft": "R-ACQ-FETCH_TIMEOUT", "docb": "R-ACQ-DOC_BUDGET", "daily": "R-ACQ-DAILY", "cd": "R-ACQ-CRAWL_DELAY",
     "dup": "R-ACQ-DUPLICATE"}


@pytest.mark.parametrize("outcome,reason,want", [
    ("dropped", R["cap"], ("discovered_not_selected", "ranked_below_cap", "resolve")),
    ("dropped", R["nodoi"], ("discovered_not_selected", "ineligible_no_identifier", "resolve")),
    ("dropped", R["notoa"], ("discovered_not_selected", "ineligible_not_open_access", "resolve")),
    ("dropped", R["known"], ("discovered_not_selected", "duplicate_or_in_corpus", "resolve")),
    ("dropped", R["dup"], ("discovered_not_selected", "duplicate", "resolve")),
    ("link_only", R["nodoi"], ("discovered_not_selected", "ineligible_no_identifier", "resolve")),
    ("rejected", R["fmt"], ("selected_not_fetched", "no_route", "resolve")),
    ("rejected", R["known"], ("selected_not_fetched", "already_in_corpus", "resolve")),
    ("rejected", R["cd"], ("selected_not_fetched", "crawl_delay", "fetch")),
    ("rejected", R["lic"], ("fetched_rejected", "licence_rejected", "licence")),
    ("rejected", R["inj"], ("fetched_rejected", "injection_rejected", "admission")),
    ("rejected", R["unreach"], ("fetched_rejected", "fetch_failed", "fetch")),
    ("rejected", R["ft"], ("fetched_rejected", "fetch_timeout", "fetch")),
    ("rejected", R["irrel"], ("acquired_not_admitted", "relevance_rejected", "relevance")),
    ("not_used", None, ("acquired_not_admitted", "over_document_budget", "admission")),
    ("reuse_rejected", R["irrel"], ("acquired_not_admitted", "relevance_rejected", "relevance")),
    ("not_evaluated", R["deadline"], ("skipped", "deadline", "fetch")),
    ("rejected", R["deadline"], ("skipped", "deadline", "fetch")),
    ("not_evaluated", R["cap"], ("skipped", "candidate_cap", "fetch")),
    ("not_evaluated", R["docb"], ("skipped", "doc_budget", "fetch")),
    ("not_evaluated", R["daily"], ("skipped", "quota", "resolve")),
    ("rejected", R["rate"], ("skipped", "rate_limited", "fetch")),  # source limiter skip; terminal stays fetch_failed
    ("admitted", None, ("admitted", "admitted", "admission")),
    ("reused", None, ("admitted", "reused_stored", "admission")),
])
def test_status_mapping(outcome, reason, want):
    assert ro.diag_status(outcome, reason) == want


def test_every_status_in_vocabulary_is_reachable_and_nothing_else():
    got = {ro.diag_status(o, r)[0] for o, r in [("dropped", R["cap"]), ("rejected", R["fmt"]), ("rejected", R["lic"]),
                                                ("not_used", None), ("not_evaluated", R["deadline"]), ("admitted", None)]}
    assert got == set(ro.DIAG_STATUSES)


def test_unknown_reason_is_never_admitted():
    assert ro.diag_status("rejected", "R-WHATEVER")[0] == "fetched_rejected"
    assert ro.diag_status("weird", None)[0] != "admitted"


def test_public_id_order_and_annotate_keeps_existing_keys():
    assert ro.public_id({"doi": "10.1/x", "pmcid": "PMC1", "url": "https://a.org"}) == "10.1/x"
    assert ro.public_id({"arxiv_id": "2401.00001", "pmcid": "PMC1"}) == "2401.00001"
    assert ro.public_id({"pmcid": "PMC1", "url": "https://a.org"}) == "PMC1"
    assert ro.public_id({"url": "https://a.org"}) == "https://a.org"
    assert ro.public_id({}) is None
    rec = {"doi": "10.1/x", "outcome": "rejected", "reason": R["lic"]}
    ro.annotate_diag(rec)
    assert rec["outcome"] == "rejected" and rec["reason"] == R["lic"]
    assert (rec["public_id"], rec["status"], rec["status_reason"], rec["stage"]) == \
        ("10.1/x", "fetched_rejected", "licence_rejected", "licence")


def test_counts_cover_all_statuses():
    c = ro.diag_counts([{"status": "admitted"}, {"status": "skipped"}, {"status": "skipped"}, {"status": "bogus"}, {}])
    assert c["admitted"] == 1 and c["skipped"] == 2 and set(c) == set(ro.DIAG_STATUSES) and sum(c.values()) == 3


# ---------------------------------------------------------------------------------------- terminal reasons + sentences
def _terminals():
    return {
        "deadline_reached": {"outcome": "not_evaluated", "reason": R["deadline"]},
        "candidate_cap": {"outcome": "not_evaluated", "reason": R["cap"]},
        "doc_budget_reached": {"outcome": "not_evaluated", "reason": R["docb"]},
        "quota": {"outcome": "not_evaluated", "reason": R["daily"]},
        "fetch_timeout": {"outcome": "rejected", "reason": R["ft"]},
    }


def test_terminal_reasons_are_distinct_not_budget_exhausted():
    t = _terminals()
    assert rr._terminal(t["deadline_reached"]) == "deadline_reached"
    assert rr._terminal(t["candidate_cap"]) == "candidate_cap"
    assert rr._terminal(t["doc_budget_reached"]) == "doc_budget_reached"
    assert rr._terminal({"outcome": "not_used", "reason": None}) == "doc_budget_reached"
    assert rr._terminal(t["quota"]) == "quota"
    assert rr._terminal(t["fetch_timeout"]) == "fetch_timeout"
    assert rr._terminal({"outcome": "rejected", "reason": R["cd"]}) == "crawl_delay"
    assert set(("deadline_reached", "candidate_cap", "doc_budget_reached", "quota", "fetch_timeout", "unreachable",
                "crawl_delay")) <= set(ro.TERMINAL_REASONS)


@pytest.mark.parametrize("cand,phrase", [
    ({"outcome": "not_evaluated", "reason": R["deadline"]}, "time limit for this search was reached"),
    ({"outcome": "not_evaluated", "reason": R["cap"]}, "more candidates were found than one request can check"),
    ({"outcome": "not_evaluated", "reason": R["docb"]}, "limit on documents read per request"),
    ({"outcome": "not_evaluated", "reason": R["daily"]}, "limit on new searches was reached"),
    ({"outcome": "queued_background", "reason": R["ft"], "unfinished": "rejected"}, "took too long to respond"),
    ({"outcome": "queued_background", "reason": R["cd"], "unfinished": "rejected"}, "wait longer than this search could"),
])
def test_each_stop_has_its_own_sentence_and_never_says_budget(cand, phrase):
    r = rr.summarize(triggers=["insufficient"], inline={"status": "none", "portals": [
        {"portal": "openalex", "status": "completed", "results": 1}], "candidates": [cand]},
        used_new=False, older_available=False, fr_before=None)
    assert phrase in r["user_text"] and "budget ran out" not in r["user_text"], r["user_text"]


def test_job_quota_refusal_is_an_explicit_quota_outcome():
    for st in ("rate", "daily_cap"):
        r = rr.summarize(triggers=["insufficient"], inline={"status": st}, used_new=False, older_available=False,
                         fr_before=None)
        assert "limit on new searches was reached" in r["user_text"] and "budget ran out" not in r["user_text"]
        assert "job_quota" in r["report_detail"]["incomplete_causes"]


def test_record_and_api_fields_carry_counts_and_ui_labels_have_no_gate_names():
    cands = [{"doi": "10.1/a", "outcome": "admitted", "reason": None}, {"doi": "10.1/b", "outcome": "rejected", "reason": R["lic"]}]
    for c in cands:
        ro.annotate_diag(c)
    rec = rr.summarize(triggers=["insufficient"], inline={"status": "ok", "candidates": cands,
                                                         "portals": [{"portal": "p", "status": "completed", "results": 2}]},
                       used_new=True, older_available=False, fr_before=None)
    assert rec["diag_status_counts"]["admitted"] == 1 and rec["diag_status_counts"]["fetched_rejected"] == 1
    pub = rr.public_fields(rec)["research_outcome"]
    assert pub["diag_status_counts"] == rec["diag_status_counts"] and pub["candidates"][0]["status"] == "admitted"
    assert set(ro.DIAG_LABELS) == set(ro.DIAG_STATUSES)
    assert not any("R-" in v or "gate" in v.lower() for v in ro.DIAG_LABELS.values())


# ---------------------------------------------------------------------------------------- end to end (offline, fake clock)
class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(acq_config, "db_configured", lambda cfg, env: True)
    monkeypatch.setattr(auto_acquire, "_worker_loop", lambda *a, **k: None)
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    acq = A.cfg()
    acq["inline"]["enabled"] = True
    yield acq
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()


def test_document_budget_stop_is_recorded_per_candidate_as_doc_budget(env):
    env["bounds"]["max_docs_per_job"] = 1
    env["inline"]["fetch_workers"] = 1
    dois = [f"10.9999/dg.ok{i}" for i in range(4)]
    web = A.Web({x: A.epmc_record(x, pmcid=f"PMC73000{i}") for i, x in enumerate(dois)},
                {f"PMC73000{i}": fs.jats(f"dg{i}", "epi", x) for i, x in enumerate(dois)})
    d = A.deps([A.cand(x) for x in dois], web)
    out = inline_acquire.acquire(A.Q_EPI, env, env=A.ENV_ON, clock=Clock(), deps_factory=lambda a, e: d)
    cs = out.public()["candidates"]
    left = [c for c in cs if c["status"] == "skipped"]
    assert out.admitted == 1 and left
    assert all(c["status_reason"] == "doc_budget" and c["reason"] == R["docb"] for c in left)
    assert not any(c["reason"] == R["cap"] for c in left)
    assert sum(out.public()["diag_status_counts"].values()) == len(cs)


def test_deadline_stop_is_recorded_per_candidate_as_deadline(env):
    clk = Clock()

    class SlowWeb(A.Web):
        def __call__(self, url, **k):
            clk.t += 50.0  # past inline.deadline_s 45 on the first read
            return super().__call__(url, **k)
    dois = [f"10.9999/dl.ok{i}" for i in range(3)]
    web = SlowWeb({x: A.epmc_record(x, pmcid=f"PMC74000{i}") for i, x in enumerate(dois)},
                  {f"PMC74000{i}": fs.jats(f"dl{i}", "epi", x) for i, x in enumerate(dois)})
    env["inline"]["fetch_workers"] = 1
    d = A.deps([A.cand(x) for x in dois], web)
    out = inline_acquire.acquire(A.Q_EPI, env, env=A.ENV_ON, clock=clk, deps_factory=lambda a, e: d)
    cs = out.public()["candidates"]
    assert len(cs) == 3 and all(c["status"] in ro.DIAG_STATUSES for c in cs)
    assert any(c["status"] == "skipped" and c["status_reason"] == "deadline" for c in cs)


def test_slow_fetch_is_fetch_timeout_not_deadline(env):
    doi, pmc = "10.9999/ft.slow", "PMC7500001"
    clk = Clock()

    class SlowWeb(A.Web):
        def __call__(self, url, **k):
            r = super().__call__(url, **k)
            if "fullTextXML" in url:
                clk.t += 21.0  # > fetch_timeout_s 20, far inside inline deadline_s 45
            return r
    web = SlowWeb({doi: A.epmc_record(doi, pmcid=pmc)}, {pmc: fs.jats("ft", "epi", doi)})
    d = A.deps([A.cand(doi)], web)
    out = inline_acquire.acquire(A.Q_EPI, env, env=A.ENV_ON, clock=clk, deps_factory=lambda a, e: d)
    (c,) = out.public()["candidates"]
    assert c["reason"] == R["ft"] and c["status"] == "fetched_rejected" and c["status_reason"] == "fetch_timeout"
    assert c["stage"] == "fetch"
    rec = rr.summarize(triggers=["insufficient"], inline=out.public(), used_new=False, older_available=False,
                       fr_before=None)
    assert "budget ran out" not in rec["user_text"]


def test_in_request_duplicate_is_r_acq_duplicate_and_stored_match_stays_known(env):
    doi = "10.9999/dup.one"
    web = A.Web({doi: A.epmc_record(doi, pmcid="PMC7600001")}, {"PMC7600001": fs.jats("dup", "epi", doi)})
    d = A.deps([A.cand(doi), A.cand(doi)], web)
    out = inline_acquire.acquire(A.Q_EPI, env, env=A.ENV_ON, clock=Clock(), deps_factory=lambda a, e: d)
    second = out.public()["candidates"][1]
    assert second["outcome"] == "dropped" and second["reason"] == R["dup"] and second["status_reason"] == "duplicate"


def test_held_discovered_records_appear_in_candidates():
    cfg = D.cfg_on()
    held = {"provider": "openalex", "url": "https://example.org/no-identifier", "title": "No identifier here",
            "snippet": "transmission incidence outbreak", "licence": D.CC0, "stored_text_kind": "abstract_metadata"}
    web, _raw = D.epmc_web()
    o, _ = D.run(cfg, [D.oa_src(), held], web)
    pub = o.public()
    hs = [c for c in pub["candidates"] if c.get("held")]
    assert hs and all(c["status"] == "discovered_not_selected" and c["status_reason"] == "ineligible_no_identifier"
                      and c["public_id"] == "https://example.org/no-identifier" for c in hs)
    assert sum(pub["diag_status_counts"].values()) == len(pub["candidates"])


# ---------------------------------------------------------------------------------------- verifier fixes (rev 2)
def _b_get_stop(clock_step, deadline):
    from vera import acq_discovered
    clk = Clock()
    host = A.cfg()["fetch"]["allowed_hosts"][0]

    def slow_get(url, **k):
        clk.t += clock_step
        return A.Resp(ctype="text/html", body=b"x")
    d = A.deps([], slow_get)
    d.url_check = lambda u: (True, "")
    with pytest.raises(auto_acquire._Stop) as e:
        acq_discovered._b_get(f"https://{host}/p", {host}, {"text/html"}, 10**6, 3, d, A.cfg(), clk, deadline, [], robots=False, timeout_s=5.0)
    return e.value.code


def test_landing_fetch_wall_clock_is_fetch_timeout_but_request_deadline_stays_deadline():
    assert _b_get_stop(6.0, None) == R["ft"]            # per-fetch timeout_s 5 exceeded, no request deadline
    assert _b_get_stop(6.0, 1000.0 + 600.0) == R["ft"]  # request deadline far away
    assert _b_get_stop(6.0, 1000.0 + 3.0) == R["deadline"]  # the request deadline passed first


def test_duplicate_has_its_own_terminal_reason_not_provenance():
    assert ro.reason_from_code(R["dup"]) == "duplicate" and "duplicate" in ro.TERMINAL_REASONS
    assert rr._terminal({"outcome": "dropped", "reason": R["dup"]}) == "duplicate"


@pytest.mark.parametrize("st,phrase", [("busy", "service was busy"), ("budget_unknown", "could not be determined")])
def test_busy_and_budget_unknown_have_truthful_sentences(st, phrase):
    r = rr.summarize(triggers=["insufficient"], inline={"status": st}, used_new=False, older_available=False, fr_before=None)
    assert phrase in r["user_text"] and "budget ran out" not in r["user_text"], r["user_text"]


@pytest.mark.parametrize("cand", [{"outcome": "rejected", "reason": R["ft"]},
                                  {"outcome": "queued_background", "reason": R["ft"], "unfinished": "rejected"}])
def test_none_after_fetch_timeout_never_implies_nothing_was_found(cand):
    r = rr.summarize(triggers=["insufficient"], inline={"status": "none", "portals": [
        {"portal": "openalex", "status": "completed", "results": 1}], "candidates": [cand]},
        used_new=False, older_available=False, fr_before=None)
    assert r["kind"] == "incomplete_search" and r["complete"] is False
    assert "took too long to respond" in r["user_text"]
    assert "No sufficiently valuable source" not in r["user_text"] and "did not meet" not in r["user_text"]
