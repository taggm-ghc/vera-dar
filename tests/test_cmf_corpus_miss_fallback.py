"""CMF-1022-T1 RED tests (v3): corpus-miss research fallback, aligned to the real seams in d2-codepath.md / s1-security.md.
Target tree: work/demo-1022/handoff-rel45/clone-20261022-170816 (HEAD 0094955); the file must sit in that tree's tests/
(NOT copied there; the offline runner refuses other paths). Offline: reuses tests/acq_support.py (Web/Resp/Store/Lookup/
cand/deps/epi_setup/cfg), tests/test_dar_inline_acquire.py::Rig, tests/test_acq_serve helpers. No new module is assumed.
I1 (CMF-1022): the defects fixed by vet.md s8 changes 1-6 are now normal asserts. Still xfail (deferred, out of scope):
E (per-candidate reasons, R2-cand) and the _guarded_get 2a/3a LATENT test. The release reports E as NOT MET.

Seams: ask_service.answer_in_scope (:115) + retry_tokens (:192); acq_config.trigger_for_decline/layer2_category_ok
(:111-123, :221); inline_acquire._run_body / Outcome.public() (candidates list, key name SEAM-TBD(D2));
auto_acquire.licence_elements_decision (:601); ask_service._inline (:272); auto_acquire._guarded_get (:365-451).
Ruling: the P0 directive removes subject-scope gating only; scope_router.REFUSE_CATEGORIES are safety refusals and never
start discovery or acquisition.
"""
import pytest

from tests import acq_support as A
from tests import test_dar_inline_acquire as TI
from tests import test_fulldoc_e2e as e2e
from tests.test_acq_serve import BASE_NO_EPI, _decide, h  # noqa: F401  (h is a fixture)
from vera import acq_config, ask_service, auto_acquire, fulldoc_corpus, inline_acquire, scope_router

# ------------------------------------------------------------------------------------------------ A: token gate
TRIGGER_TODAY = ["insufficient", "corpus_unusable", "nonresponsive"]  # already start _inline (guard)
# OWNER DIRECTION: answer-failure gate = no_valid_citations, claim_check_insufficient, model_insufficient AND
# checker_unavailable must start discovery; premise must NOT; claim_check_disabled is OPEN-OWNER (xfail, not strict).
TRIGGER_NEW = ["no_valid_citations", "claim_check_insufficient", "model_insufficient"]  # checker_unavailable removed (OD-CMF-1022g)
# premise and claim_check_disabled stay OUT (HITL-CMF-6); checker_unavailable removed from the gate (OD-CMF-1022g).
# The premise entry is the REAL token (fulldoc_answer.PREMISE_FALLBACK), so adding it to retry_tokens would fail this test;
# _wire resolves config keys through fallback_tokens.get(token, token) and passes any other string through unchanged.
from vera.fulldoc_answer import PREMISE_FALLBACK  # noqa: E402
NO_TRIGGER = [PREMISE_FALLBACK, "claim_check_disabled", "checker_unavailable"]
Q_OFF = "What is the best recipe for sourdough bread with rye flour and a long overnight proof?"


def _pr():
    return e2e.PRICING


def _wire(mp, *, token=None, decision=None, clean=True, override=False):
    """Stub the router, the base-corpus result and _inline; returns the list of _inline calls."""
    calls = []
    cfg = A.cfg()
    cfg["scope_expansion"]["override_layer2_declines"] = override
    mp.setattr(ask_service, "_acq_cfg_if_on", lambda: cfg)
    mp.setattr(ask_service, "_layer1_clean", lambda q: clean)
    dec = decision or scope_router.ScopeDecision(True, "layer1", "ok")
    mp.setattr(scope_router, "route", lambda *a, **k: dec)
    if token is not None:
        ft = fulldoc_corpus.load_config()["fallback_tokens"]
        tok = ft.get(token, token)  # "PREMISE" is not a config key: a literal non-retry token
        mp.setattr(ask_service, "_fulldoc", lambda *a, **k: (None, tok, 0, 0.0, "x"))
    mp.setattr(ask_service, "_inline", lambda *a, **k: (calls.append(k.get("in_scope")), None)[1])
    return calls


@pytest.mark.parametrize("token", TRIGGER_TODAY)
# level: L1
def test_A_existing_retry_tokens_still_start_inline(monkeypatch, token):
    calls = _wire(monkeypatch, token=token)
    ask_service.answer_in_scope(Q_OFF, "m", _pr())
    assert len(calls) == 1  # exactly one _inline per request


# level: L1 (discovery triggered)
@pytest.mark.parametrize("token", TRIGGER_NEW)
def test_A_new_miss_tokens_start_inline(monkeypatch, token):
    calls = _wire(monkeypatch, token=token)
    ask_service.answer_in_scope(Q_OFF, "m", _pr())
    assert len(calls) == 1


@pytest.mark.parametrize("token", NO_TRIGGER)
def test_A_premise_token_never_starts_inline(monkeypatch, token):
    calls = _wire(monkeypatch, token=token)
    ask_service.answer_in_scope(Q_OFF, "m", _pr())
    assert not calls


# non-refusal scope declines MUST start discovery; REFUSE_CATEGORIES MUST NOT (strict guards)
L2_REFUSE = sorted(scope_router.REFUSE_CATEGORIES)


# level: L1. off_topic already starts discovery today (positive regression, no xfail).
def test_A_off_topic_decline_starts_discovery(monkeypatch):
    dec = scope_router.ScopeDecision(False, "layer2_llm", "llm_out:not about research", category="off_topic")
    # OD-1028(6) Path A (DDFIX-1023/SCOPE precedent): off_topic now takes the normal research path; a corpus miss there starts the in-scope inline
    calls = _wire(monkeypatch, token="insufficient", decision=dec, override=True)
    ask_service.answer_in_scope(Q_OFF, "m", _pr())
    assert calls == [True], "layer-2 off_topic verdict must start research (inline, research path)"


# level: L1. in_scope (the only practical widening) starts discovery; "" (unknown/invalid category) must NOT (V1 dissent;
# matches tests/test_acq_serve.py:123).
def test_A_in_scope_layer2_decline_starts_discovery(monkeypatch):
    dec = scope_router.ScopeDecision(False, "layer2_llm", "llm_out:not about research", category="in_scope")
    # OD-1028(6) Path A CT-4: the label has no routing effect, so in_scope takes the research path like off_topic;
    # a corpus miss there starts the in-scope inline (same adaptation as test_A_off_topic_decline_starts_discovery)
    calls = _wire(monkeypatch, token="insufficient", decision=dec, override=True)
    ask_service.answer_in_scope(Q_OFF, "m", _pr())
    assert calls == [True], "layer-2 decline category 'in_scope' must start research (inline, research path)"


def test_A_empty_category_layer2_decline_does_not_start_discovery(monkeypatch):
    dec = scope_router.ScopeDecision(False, "layer2_llm", "llm_out:not about research", category="")
    calls = _wire(monkeypatch, decision=dec, override=True)
    res = ask_service.answer_in_scope(Q_OFF, "m", _pr())
    assert not calls and res.acquire_trigger is None


# level: L1 guard
@pytest.mark.parametrize("cat", L2_REFUSE)
def test_A_refuse_categories_never_acquire(monkeypatch, cat):
    dec = scope_router.ScopeDecision(False, "layer2_llm", "llm_out:refused", category=cat)
    calls = _wire(monkeypatch, decision=dec, override=True)
    res = ask_service.answer_in_scope(Q_OFF, "m", _pr())
    assert not calls and res.acquire_trigger is None


# level: L1. Code guard: refusal categories (and "") stay closed even when the config lists them.
@pytest.mark.parametrize("cat", L2_REFUSE + [""])
def test_A_refuse_categories_closed_even_when_listed_in_config(monkeypatch, cat):
    cfg = A.cfg()
    cfg["scope_expansion"]["override_layer2_declines"] = True
    cfg["scope_expansion"]["layer2_categories_allowed"] = sorted(scope_router.REFUSE_CATEGORIES) + ["", "off_topic", "in_scope"]
    dec = scope_router.ScopeDecision(False, "layer2_llm", "llm_out:x", category=cat)
    assert acq_config.layer2_category_ok(cfg, dec) is False
    calls = []
    monkeypatch.setattr(ask_service, "_acq_cfg_if_on", lambda: cfg)
    monkeypatch.setattr(ask_service, "_layer1_clean", lambda q: True)
    monkeypatch.setattr(scope_router, "route", lambda *a, **k: dec)
    monkeypatch.setattr(ask_service, "_inline", lambda *a, **k: (calls.append(1), None)[1])
    res = ask_service.answer_in_scope(Q_OFF, "m", _pr())
    assert not calls and res.acquire_trigger is None


def test_A_shipped_config_allows_exactly_off_topic_and_in_scope():
    assert acq_config.load_config()["scope_expansion"]["layer2_categories_allowed"] == ["off_topic", "in_scope"]


# level: L1 guard
# never-floor (acq_config.py:111-123): deny / empty / classifier_malformed OFF regardless of category or config
@pytest.mark.parametrize("layer,why", [("layer1", "deny:injection"), ("layer1", "deny:unsafe"), ("layer2_llm", "empty"),
                                       ("layer2_llm", "classifier_malformed:json")])
def test_A_never_floor_deny_empty_malformed(monkeypatch, layer, why):
    cfg = A.cfg()
    assert acq_config.trigger_for_decline(cfg, layer, why) is False
    cfg["triggers"]["never_on_reasons_prefix"] = []  # config cannot re-enable the floor
    cfg["triggers"]["on_decline_reasons_prefix"] = [""]
    assert acq_config.trigger_for_decline(cfg, layer, why) is False
    dec = scope_router.ScopeDecision(False, layer, why, category="off_topic")
    calls = _wire(monkeypatch, decision=dec, override=True)
    res = ask_service.answer_in_scope(Q_OFF, "m", _pr())
    assert not calls and res.acquire_trigger is None


# ------------------------------------------------------------------------------------------------ B: eligible JATS acquired
@pytest.fixture
def inline_env(monkeypatch):
    monkeypatch.setattr(acq_config, "db_configured", lambda cfg, env: True)
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    acq = A.cfg()
    acq["inline"]["enabled"] = True
    return acq


def _acquire(acq, cands, web, q=A.Q_EPI):
    d = A.deps(cands, web)
    return inline_acquire.acquire(q, acq, env=A.ENV_ON, deps_factory=lambda a, e: d), d


# level: L2
def test_B_eligible_doi_jats_candidate_is_resolved_and_acquired(inline_env):
    web, _raw = A.epi_setup(doi="10.9999/cmf.b1", pmcid="PMC7000001")
    out, d = _acquire(inline_env, [A.cand("10.9999/cmf.b1")], web)
    assert out.admitted >= 1
    assert any("PMC7000001/fullTextXML" in u for u in web.urls)  # resolved DOI -> PMC id -> JATS fetched


# ------------------------------------------------------------------------------------------------ E: per-candidate status
from vera import research_outcome as _ro  # noqa: E402  (P0DTA-DIAG: the status vocabulary is research_outcome.DIAG_STATUSES)
STATUSES = set(_ro.DIAG_STATUSES)


def _ident(c):
    return c.get("doi") or c.get("pmcid") or c.get("record_id") or c.get("url")


# level: L2
def test_E_every_candidate_gets_exactly_one_terminal_status_with_reason(inline_env):
    from tests import fulltext_synth as fs
    dois = {n: f"10.9999/cmf.e{n}" for n in range(1, 8)}
    pm = {n: f"PMC71000{n}" for n in range(1, 8)}
    recs = {dois[1]: A.epmc_record(dois[1], pmcid=pm[1]),                       # 1 admitted
            dois[2]: A.epmc_record(dois[2], pmcid=pm[2], licence="cc by-nd"),   # 2 unlicensed
            dois[3]: A.epmc_record(dois[3], pmcid=pm[3]),                       # 3 inaccessible (no JATS -> 404)
            dois[4]: A.epmc_record(dois[4], pmcid=pm[4]),                       # 4 irrelevant (other topic)
            dois[5]: A.epmc_record(dois[5], pmcid=pm[5])}                       # 5 unsupported_format (PDF served)
    jats = {pm[1]: fs.jats("e1", "epi", dois[1]), pm[2]: fs.jats("e2", "epi", dois[2]),
            pm[4]: fs.jats("e4", "clim", dois[4])}
    web = A.Web(recs, jats, overrides={f"{pm[5]}/fullTextXML": A.Resp(ctype="application/pdf", body=b"%PDF-1.4")})
    cands = [A.cand(dois[n]) for n in range(1, 6)] + [A.cand(dois[1]),          # 6 duplicate of 1 (dedup skip)
                                                      {"title": "no identifier", "url": "https://x.org/a", "snippet": "s"}]  # 7 prefilter skip
    out, _ = _acquire(inline_env, cands, web)
    pub = out.public()
    listed = pub.get("candidates")
    assert isinstance(listed, list) and len(listed) == len(cands), "one entry per candidate, skips included"
    for e in listed:
        assert e.get("status") in STATUSES and e.get("status_reason") and e.get("public_id"), e
    assert sum(pub["diag_status_counts"].values()) == len(cands)
    # real vocabulary: the pipeline's outcome + R-code, with the diagnostic status / stage derived from them
    row = [(e["outcome"], e["reason"], e["status"], e["stage"]) for e in listed]
    assert row[0] == ("admitted", None, "admitted", "admission")
    assert row[1][:1] == ("rejected",) and row[1][1].startswith("R-LIC") and row[1][2:] == ("fetched_rejected", "licence")
    # rows 2, 4, 6 are deterministic here (fakes, no clock): a 404 is unreachable and queued for the background worker
    assert listed[2]["status_reason"] == "fetch_failed"
    assert row[2] == ("queued_background", "R-ACQ-UNREACHABLE", "fetched_rejected", "fetch")                # no JATS (404)
    assert row[3] == ("rejected", "R-ACQ-IRRELEVANT", "acquired_not_admitted", "relevance")
    assert row[4] == ("rejected", "R-ACQ-CONVERT", "fetched_rejected", "fetch")                              # PDF served
    assert listed[4]["status_reason"] == "extraction_failed"
    assert row[5] == ("dropped", "R-ACQ-DUPLICATE", "discovered_not_selected", "resolve")             # in-request duplicate
    assert row[6] == ("queued_background", "R-ACQ-NODOI", "discovered_not_selected", "resolve")           # URL-only: follow_url queue


# ------------------------------------------------------------------------------------------------ R3: no licence element = HOLD
# level: L1 (licence gate on acquisition)
def test_R3_jats_without_licence_element_is_held_not_granted_by_metadata():
    from vera import fulltext_loader as fl
    icfg = acq_config.ingest_cfg(fl.load_config(), A.cfg())
    with pytest.raises(fl.Refused) as ei:
        auto_acquire.licence_elements_decision([], "cc by", icfg)
    assert ei.value.code == icfg["reason_codes"]["licence_hold"]


# level: L1 read-time stored-row path (fulldoc_db.load_acquired_rows re-decides every stored acquired row):
# a stored row whose recorded jats_licences is [] becomes per-document unusable (R-LIC-HOLD).
def test_L1_read_time_stored_row_with_empty_licence_elements_is_refused():
    import json
    from vera import fulldoc_db
    auto_acquire.reset_for_tests()
    store = A.Store()
    out = auto_acquire.run_job(A.job(), A.cfg(), A.deps([A.cand("10.9999/acq.epi4")], A.epi_setup()[0], store=store),
                               clock=A.clock_at(), env=A.ENV_ON)
    auto_acquire.reset_for_tests()
    assert out["stored"] == 1  # a document with a recognised licence element is still stored
    src, spans = A.rows_from_plans(store.plans)
    ok = fulldoc_db.load_acquired_rows(src, spans, fulldoc_corpus.load_config(), A.cfg())
    assert ok.acquired and not ok.unusable
    p = json.loads(src[0]["provenance"])
    p["meta"]["jats_licences"] = []
    c = fulldoc_db.load_acquired_rows([{**src[0], "provenance": json.dumps(p)}], spans, fulldoc_corpus.load_config(), A.cfg())
    assert not c.acquired and any(u["code"] == "R-LIC-HOLD" for u in c.unusable), c.unusable


# ------------------------------------------------------------------------------------------------ 3b: malformed ports (unit)
@pytest.mark.parametrize("url", ["https://host.example.org:abc/x", "https://host.example.org:99999/x",
                                 "https://host.example.org:-1/x", "https://host.example.org:0/x",
                                 "http://host.example.org:65536/x"])
def test_3b_malformed_or_zero_port_refused_without_exception(monkeypatch, url):
    import socket
    from vera import selective_fetch
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    ok, why = selective_fetch.is_public_url(url)  # must not raise ValueError
    assert ok is False and why


# ------------------------------------------------------------------------------------------------ D: same-request retry
# level: L3 (newly acquired evidence admitted and used in a supported answer, same request, stubs end to end)
def test_D_new_token_retry_answers_in_same_request_from_acquired_doc(h, monkeypatch):
    rig = TI.Rig(h, monkeypatch)
    real = ask_service._fulldoc
    tok = fulldoc_corpus.load_config()["fallback_tokens"]["model_insufficient"]

    def wrap(*a, **k):  # base-corpus attempt -> model_insufficient; the inline (acquired_only/ephemeral) call is real
        if k.get("ephemeral") is None and not k.get("acquired_only"):
            return (None, tok, 0, 0.0, "x")
        return real(*a, **k)
    monkeypatch.setattr(ask_service, "_fulldoc", wrap)
    r = rig.run(q=TI.OFF_Q, layer="layer1", reason="ok", category="", accept=True)
    assert r.result_kind == "answer"
    assert rig.outcomes and rig.outcomes[0].admitted >= 1 and len(rig.outcomes) == 1  # single-shot
    assert r.evidence is not None and r.evidence.admitted  # the answer used the newly acquired passages
    assert h.answer_calls  # answer writer ran in this request


# ------------------------------------------------------------------------------------------------ G: _guarded_get bounds
ALLOW = A.cfg()["fetch"]["allowed_hosts"][0]


# level: L2 (bounded, SSRF-checked traversal during acquisition)
def _with_check(get, check=lambda u: (True, "")):
    d = A.deps([], get)
    d.url_check = check
    return d


def _stop_code(url, deps, cfg=None, **kw):
    with pytest.raises(auto_acquire._Stop) as ei:
        auto_acquire._guarded_get(url, cfg or A.cfg(), deps, 1_000_000, 5.0, False, **kw)
    return ei.value.code


def test_G_non_allowlisted_or_non_https_host_never_requested():
    calls = []
    d = _with_check(lambda *a, **k: calls.append(a) or A.Resp(), lambda u: (True, ""))
    cfg = A.cfg()
    assert _stop_code("https://evil.example/x", d) == acq_config.reason(cfg, "host")
    assert _stop_code(f"http://{ALLOW}/x", d) == acq_config.reason(cfg, "host")
    assert not calls


def test_G_ssrf_check_failure_stops_before_any_request():
    calls = []
    d = _with_check(lambda *a, **k: calls.append(a) or A.Resp(), lambda u: (False, "private"))
    assert _stop_code(f"https://{ALLOW}/x", d) == acq_config.reason(A.cfg(), "host")
    assert not calls


def test_G_redirect_to_internal_host_is_blocked_and_never_fetched():
    seen = []

    def get(url, **k):
        seen.append(url)
        return A.Resp(status=302, headers={"Location": "https://internal.invalid/admin"})
    d = _with_check(get, lambda u: (False, "private") if "internal" in u else (True, ""))
    assert _stop_code(f"https://{ALLOW}/x", d) == acq_config.reason(A.cfg(), "host")
    assert not any("internal" in u for u in seen)


def test_G_redirect_loop_is_bounded():
    seen = []

    def get(url, **k):
        seen.append(url)
        return A.Resp(status=302, headers={"Location": f"https://{ALLOW}/again"})
    d = _with_check(get)
    assert _stop_code(f"https://{ALLOW}/x", d) == acq_config.reason(A.cfg(), "unreachable")
    assert len(seen) <= 4  # selective_fetch.MAX_REDIRECTS (3) + the first hop


def test_G_oversize_body_is_capped():
    d = _with_check(lambda *a, **k: A.Resp(ctype="application/xml", body=b"x" * 3_000_000))
    with pytest.raises(auto_acquire._Stop) as ei:
        auto_acquire._guarded_get(f"https://{ALLOW}/x", A.cfg(), d, 1000, 5.0, False)
    assert ei.value.code == acq_config.reason(A.cfg(), "too_large")


@pytest.mark.xfail(reason="RED (I2/E1): the chain/cycle, userinfo and port-443 checks apply only when chain is set; "
                          "the fallback path should apply them unconditionally", strict=False)
def test_G_userinfo_and_non443_port_rejected_without_chain():
    calls = []
    d = _with_check(lambda *a, **k: calls.append(a) or A.Resp(ctype="application/xml", body=b"<a/>"))
    assert _stop_code(f"https://u:p@{ALLOW}/x", d) == acq_config.reason(A.cfg(), "host")
    assert _stop_code(f"https://{ALLOW}:8443/x", d) == acq_config.reason(A.cfg(), "host")
    assert not calls


# ------------------------------------------------------------------------------------------------ D2/D3 and latency bound
@pytest.fixture
def wired(monkeypatch):
    from types import SimpleNamespace
    calls = {"fulldoc": [], "acquire": 0}
    dec = SimpleNamespace(accepted=True, layer="layer2_llm", reason="llm_in:x", tokens_used=3, cost_usd=0.0)
    monkeypatch.setattr(scope_router, "route", lambda *a, **k: dec)
    monkeypatch.setattr(ask_service, "_acq_cfg_if_on", lambda: {"inline": {"in_scope": True}})
    monkeypatch.setattr(ask_service, "_layer1_clean", lambda q: True)
    monkeypatch.setattr(ask_service, "_acq_on_insufficient", lambda acq: False)
    monkeypatch.setattr(inline_acquire, "icfg_of", lambda acq: {"in_scope": True})
    monkeypatch.setattr(fulldoc_corpus, "flag_enabled", lambda cfg: True)
    monkeypatch.setattr(inline_acquire, "notice", lambda acq, status: f"NOTICE:{status}")

    def acquire(question, acq, served=None):
        calls["acquire"] += 1
        return calls["outcome"]
    calls["outcome"] = SimpleNamespace(corpus=object(), status="ok")
    monkeypatch.setattr(inline_acquire, "acquire", acquire)
    monkeypatch.setattr(ask_service, "_reading_list", lambda q, m, p, d, c, g, contract, tokens=None, cost=None:
                        ask_service.AskResult(answer="RL", tokens_used=tokens or 0, cost_usd=cost or 0.0,
                                              result_kind="reading_list", contract={"layers": ["L0"]}))
    return calls


def _fd(calls, base_token, inline_result):
    def fulldoc(question, model, pricing, decision, chained, grounding, acquired_only=False, ephemeral=None, used=None):
        calls["fulldoc"].append("inline" if ephemeral is not None else "base")
        if ephemeral is not None:
            return inline_result
        return (None, base_token, 1, 0.0, "TEXT")
    return fulldoc


# level: L1 (stubs). D2: base passages admitted but the writer failed -> the acquired-only retry answers.
def test_D2_base_writer_failed_then_acquired_only_retry_answers(wired, monkeypatch):
    good = ask_service.AskResult(answer="ANSWER", tokens_used=1, cost_usd=0.0, result_kind="answer", contract={})
    monkeypatch.setattr(inline_acquire, "answer_note", lambda acq, o: "")
    wired_o = wired["outcome"]
    wired_o.public = lambda: {"status": "ok"}
    tok = fulldoc_corpus.load_config()["fallback_tokens"]["no_valid_citations"]
    monkeypatch.setattr(ask_service, "_fulldoc", _fd(wired, tok, good))
    r = ask_service.answer_in_scope("What is X?", "m", _pr())
    assert wired["acquire"] == 1 and wired["fulldoc"] == ["base", "inline"] and r.result_kind == "answer"


# level: L1 (stubs). D3: the retry also fails -> both fallback tokens are kept in the contract.
@pytest.mark.parametrize("base,inl", [("no_valid_citations", "claim_check_insufficient"),
                                      ("model_insufficient", "checker_unavailable")])
def test_D3_retry_failure_keeps_fallback_and_inline_fallback(wired, monkeypatch, base, inl):
    ft = fulldoc_corpus.load_config()["fallback_tokens"]
    monkeypatch.setattr(ask_service, "_fulldoc", _fd(wired, ft[base], (None, ft[inl], 1, 0.0, "T")))
    r = ask_service.answer_in_scope("What is X?", "m", _pr())
    assert wired["acquire"] == 1 and r.result_kind == "reading_list"  # exactly one retry, no loop
    assert r.contract["fallback"] == ft[base] and r.contract.get("inline_fallback") == ft[inl], r.contract


def _clock(monkeypatch, elapsed):
    seq = iter([0.0])
    monkeypatch.setattr(ask_service.time, "monotonic", lambda: next(seq, float(elapsed)))


# level: L1 (stubs). Latency bound from existing values only: need = inline deadline_s (45) + 3 x chain timeout_s (20)
# + 2 x grounding deadline_s (10, when grounded) -> 105 s ungrounded here; skipped retry reports {"status": "deadline"}.
def test_latency_inline_retry_skipped_when_remaining_time_is_short(wired, monkeypatch):
    tok = fulldoc_corpus.load_config()["fallback_tokens"]["no_valid_citations"]
    monkeypatch.setattr(ask_service, "_fulldoc", _fd(wired, tok, None))
    _clock(monkeypatch, 80.0)  # 100 s left < 105 s
    r = ask_service.answer_in_scope("What is X?", "m", _pr())
    assert wired["acquire"] == 0 and wired["fulldoc"] == ["base"] and r.result_kind == "reading_list"
    assert r.contract["inline_acquisition"] == {"status": "deadline"} and r.contract["fallback"] == tok


def test_latency_inline_retry_runs_when_enough_time_remains(wired, monkeypatch):
    tok = fulldoc_corpus.load_config()["fallback_tokens"]["no_valid_citations"]
    monkeypatch.setattr(ask_service, "_fulldoc", _fd(wired, tok, None))
    _clock(monkeypatch, 70.0)  # 110 s left >= 105 s
    ask_service.answer_in_scope("What is X?", "m", _pr())
    assert wired["acquire"] == 1


@pytest.mark.parametrize("elapsed,runs", [(74.9, True), (75.0, True), (75.1, False)])  # need 105: remaining need+0.1/need/need-0.1
def test_latency_boundary_need_plus_minus_point_one(wired, monkeypatch, elapsed, runs):
    tok = fulldoc_corpus.load_config()["fallback_tokens"]["no_valid_citations"]
    monkeypatch.setattr(ask_service, "_fulldoc", _fd(wired, tok, None))
    _clock(monkeypatch, elapsed)
    ask_service.answer_in_scope("What is X?", "m", _pr())
    assert wired["acquire"] == (1 if runs else 0)


def test_latency_need_is_125_with_shipped_grounding_and_matches_calibration_path():
    from vera.grounded_ask import load_grounding
    need = ask_service.inline_need_s({"inline": {"deadline_s": 45}}, (None, load_grounding()))
    assert need == 125.0  # 45 + 3*20 + 2*10: the calibration note's worst inline path (125 s)


def test_latency_unreadable_term_skips_with_budget_unknown(wired, monkeypatch):
    tok = fulldoc_corpus.load_config()["fallback_tokens"]["no_valid_citations"]
    monkeypatch.setattr(ask_service, "_fulldoc", _fd(wired, tok, None))
    monkeypatch.setattr(ask_service, "_chain_timeout_s", lambda: None)
    r = ask_service.answer_in_scope("What is X?", "m", _pr())
    assert wired["acquire"] == 0 and r.contract["inline_acquisition"] == {"status": "budget_unknown"}


def test_one_inline_per_request_and_one_deadline_across_a_two_provider_chain(wired, monkeypatch):
    """The same RequestBudget (deadline + inline flag) is shared by every provider attempt in the chain: a failover does
    not restart the clock and cannot run a second _inline."""
    ft = fulldoc_corpus.load_config()["fallback_tokens"]
    monkeypatch.setattr(ask_service, "_fulldoc", _fd(wired, ft["no_valid_citations"], (None, ft["claim_check_insufficient"], 1, 0.0, "T")))
    b = ask_service.request_budget()
    first = ask_service.answer_in_scope("What is X?", "m", _pr(), budget=b)
    second = ask_service.answer_in_scope("What is X?", "m", _pr(), budget=b)  # provider 2 of the chain
    assert wired["acquire"] == 1 and b.inline_used is True  # max one _inline per request
    assert first.contract.get("inline_fallback") and "inline_acquisition" not in second.contract
    # context carriage used by main.py: the second attempt picks up the same budget without an argument
    wired["acquire"] = 0
    b2 = ask_service.request_budget()
    tok = ask_service.REQUEST_BUDGET.set(b2)
    try:
        ask_service.answer_in_scope("What is X?", "m", _pr())
        ask_service.answer_in_scope("What is X?", "m", _pr())
    finally:
        ask_service.REQUEST_BUDGET.reset(tok)
    assert wired["acquire"] == 1 and b2.inline_used is True


def test_deadline_elapsed_carries_over_between_providers(wired, monkeypatch):
    """Elapsed time of a failed first provider counts against the second: the budget's deadline is fixed at creation."""
    tok = fulldoc_corpus.load_config()["fallback_tokens"]["no_valid_citations"]
    monkeypatch.setattr(ask_service, "_fulldoc", _fd(wired, tok, None))
    b = ask_service.RequestBudget(deadline=1000.0)
    monkeypatch.setattr(ask_service.time, "monotonic", lambda: 1000.0 - 104.0)  # 104 s left < 105 s
    r = ask_service.answer_in_scope("What is X?", "m", _pr(), budget=b)
    assert wired["acquire"] == 0 and r.contract["inline_acquisition"] == {"status": "deadline"}
