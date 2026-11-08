"""PURPOSE-GRANTS-1108: purpose-grant vocabulary in shadow mode. Offline: no network, no DB.

Pins: (1) matching (grant, carve-out, negation, bare mention, conditions); (2) what a grant may and may never lift;
(3) site-owner channels only; (4) shadow mode changes no access or use decision and records codes in the trace and
verbatim evidence in the owner-only ring; (5) the config carries no per-site list.
"""
import json
import re
from urllib import robotparser

import pytest

from vera import public_fetch, purpose_grants as pg, request_trace as rt, robots_ai, source_permissions as sp

CFG = json.loads(pg.CONFIG_PATH.read_text(encoding="utf-8"))
RT_BASE = json.loads(rt.CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _reset():
    pg.reset_for_tests()
    c = json.loads(json.dumps(RT_BASE))
    c["db_sink"]["enabled"] = False
    rt.configure(c)
    yield
    pg.reset_for_tests()
    rt.end_context()


def cl(text, channel="robots_comment"):
    return pg.classify_line(text, channel, CFG)


# ------------------------------------------------------------------------------------------------ matching
@pytest.mark.parametrize("text,kind,purposes", [
    ("Crawling for research purposes is permitted", "grant", ("research",)),
    ("Research crawlers are welcome", "grant", ("research",)),
    ("Allowed for non-commercial research", "grant", ("non_commercial", "research")),
    ("Educational use permitted with attribution", "grant", ("educational",)),
    ("AI crawlers are blocked, except for research", "carveout", ("research",)),
    ("All AI bots are disallowed unless for academic research", "carveout", ("academic", "research")),
    ("AI crawlers blocked; research use permitted", "grant", ("research",)),
    ("Not for research", "negated", ("research",)),
    ("Research use is not permitted", "negated", ("research",)),
    ("No research use", "negated", ("research",)),
    ("Research bots are blocked", "negated", ("research",)),
    ("Research use is blocked, AI is allowed", "negated", ("research",)),
    ("Teaching use may not be granted", "negated", ("teaching",)),
])
def test_classify(text, kind, purposes):
    g = cl(text)
    assert g is not None and g.kind == kind and g.purposes == purposes, g


@pytest.mark.parametrize("text", [
    "Our research department maintains this file",  # bare mention: no permit, no negation
    "See the research section of our site",
    "This site is run by a not for profit",  # 'not' inside a purpose phrase is no negation
    "User-agent rules below",
    "",
])
def test_bare_mentions_are_no_grant(text):
    assert cl(text) is None


def test_condition_phrases_are_not_negations_and_are_recorded():
    g = cl("Research use permitted, no redistribution, please cite us")
    assert g.kind == "grant" and set(g.conditions) == {"no_redistribution", "attribution"}
    assert "non_commercial" in cl("Non-commercial use is welcome").conditions
    assert "contact_first" in cl("Research use allowed; contact us first").conditions


def test_scope_flags_and_verbatim():
    g = cl("AI crawlers for research are welcome")
    assert g.ai_scoped and not g.tdm_scoped and g.verbatim == "AI crawlers for research are welcome"
    assert cl("Text and data mining for research is permitted").tdm_scoped
    assert re.fullmatch(r"[0-9a-f]{16}", g.evidence_id)


def test_scan_robots_comments_and_structured_fields():
    txt = ("﻿# AI crawlers are blocked, except for research\nUser-agent: *\nDisallow: /private # research welcome\n"
           "Content-Signal: research=yes, ai-input=no\n# our research team\n")
    gs = pg.scan_robots(txt, CFG)
    assert [(g.channel, g.kind) for g in gs] == [("robots_comment", "carveout"), ("robots_comment", "grant"),
                                                 ("robots_field", "structured")]


def test_scan_headers_and_tdmrep():
    hs = pg.scan_headers({"Content-Usage": "ai-use=n, education=y", "TDM-Policy": "https://a.example/p.json"}, CFG)
    assert [(g.channel, g.kind) for g in hs] == [("http_header", "structured"), ("http_header", "policy_unread")]
    rules = [{"location": "/", "tdm-reservation": 1, "tdm-policy": "https://a.example/policy.json"},
             {"location": "/open/", "tdm-reservation": 0}]
    t = pg.scan_tdmrep(rules, "/x", CFG)
    assert len(t) == 1 and t[0].kind == "policy_unread" and "policy.json" in t[0].verbatim
    assert pg.scan_tdmrep([{"location": "/", "tdm-reservation": 1}], "/x", CFG) == ()
    assert pg.scan_tdmrep("garbage", "/x", CFG) == ()


def test_directive_paths_and_agent_names_are_never_grants():
    """Abuse: 'research' in a path, an agent name or a sitemap URL is not a statement of permission."""
    txt = ("User-agent: ResearchBot\nAllow: /research/\nAllow: /education-allowed/\n"
           "Sitemap: https://a.example/research-permitted.xml\n")
    assert pg.scan_robots(txt, CFG) == ()


def test_line_and_grant_caps():
    long = "# research use permitted " + "x " * 2000
    g = pg.scan_robots(long, CFG)[0]
    assert len(g.verbatim) <= CFG["limits"]["verbatim_chars"]
    many = "\n".join("# research use permitted %d" % i for i in range(100))
    assert len(pg.scan_robots(many, CFG)) == CFG["limits"]["max_grants"]


# ------------------------------------------------------------------------------------------------ assessment
def A(restriction, *texts, channel="robots_comment"):
    return pg.assess(restriction, [cl(t, channel) for t in texts], CFG)


def test_carveout_would_lift_statement_and_no_ai():
    assert A("robots_ai_statement", "AI crawlers are blocked, except for research")["decision"] == "would_admit"
    assert A("use_no_ai_signal", "AI crawlers for research are welcome")["decision"] == "would_admit"


def test_unlinked_grant_goes_to_review_not_admit():
    a = A("robots_ai_statement", "Educational use permitted")
    assert a["decision"] == "review" and a["why"] == "unlinked"
    assert A("use_tdm_reserved", "AI crawlers for research are welcome")["decision"] == "review"  # not TDM-scoped
    assert A("use_tdm_reserved", "Text and data mining for research is permitted")["decision"] == "would_admit"


@pytest.mark.parametrize("restriction", ["robots_ai_agent_block", "read_robots_disallow", "read_deny_list",
                                         "read_registry_forbid", "read_login_paywall", "read_no_crawl",
                                         "use_nd_licence", "use_registry_forbid", "licence_reject", "crawl_delay",
                                         "rate_limit", "some_unknown_code"])
def test_never_overridable(restriction):
    a = A(restriction, "AI crawlers are blocked, except for research")
    assert a["decision"] == "not_overridable"


def test_negation_conflict_contact_first_and_policy():
    assert A("robots_ai_statement", "Not for research")["decision"] == "negated"
    a = A("robots_ai_statement", "AI crawlers are blocked, except for research", "No research use")
    assert a["decision"] == "review" and a["why"] == "conflict"
    a = A("robots_ai_statement", "AI crawlers for research are welcome; contact us first")
    assert a["decision"] == "review" and a["why"] == "contact_first"
    pol = pg.scan_tdmrep([{"location": "/", "tdm-reservation": 1, "tdm-policy": "https://a.example/p"}], "/", CFG)
    assert pg.assess("use_tdm_reserved", pol, CFG)["decision"] == "review"
    s = pg.scan_robots("Content-Signal: research=yes", CFG)
    assert pg.assess("use_no_ai_signal", s, CFG)["why"] == "structured"


def test_untrusted_channels_never_count():
    """Abuse: a crafted page or a third-party comment area claiming a grant is not a site-owner channel."""
    for ch in ("page_meta", "page_body", "third_party", "html_comment"):
        g = cl("AI crawlers are blocked, except for research", ch)
        assert pg.assess("robots_ai_statement", [g], CFG) is None
    # an http_header grant cannot lift a robots.txt statement (the statement's own channel only)
    h = cl("AI crawlers for research are welcome", "http_header")
    assert pg.assess("robots_ai_statement", [h], CFG)["decision"] == "not_overridable"


def test_several_restrictions_all_must_lift():
    g = [cl("AI crawlers for research are welcome")]
    a = pg.shadow(["use_tdm_reserved", "use_no_ai_signal"], g, "https://a.example/x")
    assert a["decision"] == "review" and a["restriction"] == "use_tdm_reserved"
    a = pg.shadow(["use_no_ai_signal", "use_nd_licence"], g, "https://a.example/y")
    assert a["decision"] == "not_overridable" and a["restriction"] == "use_nd_licence"


# ------------------------------------------------------------------------------------------------ shadow wiring
ROBOTS = "# AI crawlers are blocked, except for non-commercial research\nUser-agent: *\nAllow: /\n"


def _rp(text):
    rp = robotparser.RobotFileParser()
    rp.parse(text.splitlines())
    return robots_ai.annotate(rp, text)


def test_robots_block_unchanged_and_evidence_recorded():
    rp = _rp(ROBOTS)
    url = "https://a.example/page"
    assert robots_ai.block_reason(rp, url) == "robots_ai_statement"  # the block stands
    assert robots_ai.allowed(rp, "VERA-research-fetcher/0.1", url) is True
    ev = pg.shadow_evidence()
    assert len(ev) == 1 and ev[0]["decision"] == "would_admit" and ev[0]["host"] == "a.example"
    assert ev[0]["verbatim"] == "AI crawlers are blocked, except for non-commercial research"
    assert ev[0]["conditions"] == ("non_commercial",)
    robots_ai.block_reason(rp, url)
    assert len(pg.shadow_evidence()) == 1  # deduplicated


def test_hyperscaler_group_never_blocks_so_no_shadow_record():
    # ROBOTS-POLICY-1108 (H13, OD-1108-DISPOSITIONS-2): a hyperscaler-only group is advisory and never blocks, so
    # there is no restriction for a purpose grant to shadow; it is recorded by robots_ai.advisory() instead.
    txt = "# research use welcome\nUser-agent: GPTBot\nDisallow: /\nUser-agent: *\nAllow: /\n"
    rp = _rp(txt)
    assert robots_ai.block_reason(rp, "https://a.example/x") is None
    assert robots_ai.advisory(rp, "https://a.example/x") == ("GPTBot",)
    assert pg.shadow_evidence() == []


def test_shadow_writes_codes_only_trace_event():
    rt.start("q", {"endpoint": "/ask"})
    robots_ai.block_reason(_rp(ROBOTS), "https://a.example/page")
    rt.finish("answer", "complete")
    v = rt.recent(rt.CURRENT.get().trace_id)
    ev = [e for e in v["events"] if e["stage"] == "purpose_grant"]
    assert len(ev) == 1 and ev[0]["code"] == "would_admit"
    d = ev[0]["detail"]
    assert d["mode"] == "shadow" and d["restriction"] == "robots_ai_statement" and d["kind"] == "carveout"
    assert d["purposes"] == "non_commercial:research" and d["conditions"] == "non_commercial"
    assert re.fullmatch(r"[0-9a-f]{16}", d["evidence"][0]) and "dropped_keys" not in d
    blob = json.dumps(v["events"])
    assert "a.example" not in blob and "blocked, except" not in blob


def test_evaluate_record_unchanged_with_grant():
    url = "https://a.example/p"
    kw = dict(headers={"Content-Signal": "ai-input=no"}, registry=[], deny_list=[], config={})
    base = sp.evaluate(url, **kw)
    pg.reset_for_tests()
    pg.load_config()
    r = sp.evaluate(url, robots_txt_entries="# AI crawlers for research are welcome\n", **kw)
    assert (r.use_for_answer, r.use_reason, r.ingest_reason) == (base.use_for_answer, base.use_reason,
                                                                 base.ingest_reason)
    assert r.use_reason == "use_no_ai_signal"
    assert pg.shadow_evidence()[0]["decision"] == "would_admit"


def test_tdmrep_reservation_unchanged_and_recorded():
    rules = [{"location": "/", "tdm-reservation": 1, "tdm-policy": "https://a.example/policy.json"}]
    assert public_fetch.tdmrep_reserved(rules, "/x") is True
    public_fetch._tdm_purpose_shadow(rules, "https://a.example/x")
    ev = pg.shadow_evidence()
    assert ev[0]["channel"] == "tdmrep" and ev[0]["decision"] == "review"


def test_mode_off_records_nothing(monkeypatch):
    off = dict(CFG, mode="off")
    monkeypatch.setattr(pg, "_CFG", off)
    assert pg.mode() == "off"
    assert robots_ai.block_reason(_rp(ROBOTS), "https://a.example/page") == "robots_ai_statement"
    assert pg.shadow_evidence() == []


@pytest.mark.parametrize("m", ["on", "ON", "admit", "", None, 1])
def test_no_admit_mode_exists(m):
    assert pg.mode(dict(CFG, mode=m)) == "off"


def test_malformed_config_fails_to_off_and_scans_nothing(monkeypatch):
    with pytest.raises(ValueError):
        _validate({})
    bad = dict(CFG, never_overridable=list(CFG["never_overridable"]) + ["robots_ai_statement"])
    with pytest.raises(ValueError):
        _validate(bad)

    def boom(path=None):
        raise ValueError("purpose_grants config malformed")
    monkeypatch.setattr(pg, "load_config", boom)
    assert pg.mode() == "off" and pg.scan_robots(ROBOTS) == () and pg.shadow("robots_ai_statement", [1]) is None
    rp = _rp(ROBOTS)  # robots evaluation itself is untouched
    assert robots_ai.block_reason(rp, "https://a.example/") == "robots_ai_statement"
    assert pg.shadow_evidence() == []


def _validate(c):
    """load_config's validation on an in-memory dict (no file written; no temp directory used)."""
    import unittest.mock as um
    with um.patch.object(pg.Path, "read_text", return_value=json.dumps(c)):
        return pg.load_config(pg.CONFIG_PATH.with_name("purpose_grants_test_only.json"))


# ------------------------------------------------------------------------------------------------ config hygiene
def test_config_has_no_per_site_list_and_h13_stays_binding():
    host = re.compile(r"\b[a-z0-9-]+\.(?:com|org|net|edu|gov|io|uk|de|fr|ai|info)\b", re.I)
    vals = json.dumps({k: v for k, v in CFG.items() if not k.startswith("_")})
    assert not host.search(vals) and "://" not in vals
    assert not (set(CFG["overridable"]) & set(CFG["never_overridable"]))
    assert "robots_ai_agent_block" in CFG["never_overridable"]
    assert CFG["mode"] in pg.MODES


# ------------------------------------------------------------------------------------------------ PURPOSE-GRANTS-FIX-1108
# The verifier's 30 probe strings (kind, decision against robots_ai_statement); F1/F2/F3 outcomes as fixed.
PROBES = [
    ("AI crawlers are blocked, except for research", "carveout", "would_admit"),
    ("AI crawlers are blocked except for research purposes", "carveout", "would_admit"),
    ("Not for research use", "negated", "negated"),
    ("Research use is not permitted", "negated", "negated"),
    ("Research purposes only", None, None),
    ("Research crawlers are welcome", "grant", "would_admit"),
    ("Our research department is located in Building 4", None, None),
    ("Educational use permitted", "grant", "review"),
    ("Educational use is not allowed", "negated", "negated"),
    ("Commercial use prohibited; research is allowed", "grant", "review"),
    ("All bots are prohibited. Research is not an exception", "negated", "negated"),
    ("No AI training. Exception: none, not even for research", "negated", "negated"),
    ("AI scraping is banned, but researchers may crawl", "carveout", "would_admit"),
    ("Researchers may not crawl", "negated", "negated"),
    ("do not crawl for research", "negated", "negated"),
    ("We block AI bots unless you are doing academic research", "carveout", "would_admit"),
    ("AI bots are welcome, no research allowed", "negated", "negated"),
    ("Allowed: non-commercial research. Contact us first", "grant", "review"),
    ("AI training is not permitted, even for educational purposes", "negated", "negated"),
    ("AI crawlers blocked. Research and teaching are exempt", "grant", "would_admit"),
    ("for research purposes only, no AI", "negated", "negated"),
    ("Disallow research bots but allow everyone else", "negated", "negated"),  # F1
    ("Nothing is allowed for research or education", "negated", "negated"),  # F2
    ("Permission is not granted for research", "negated", "negated"),
    ("Research use is allowed only with prior permission from the owner", "grant", "review"),
    ("AI crawlers are not blocked for research", "negated", "negated"),  # no separator: stays safe
    ("Do not use this for AI except where research is concerned", "carveout", "would_admit"),
    ("ai bots blocked - research ok", "carveout", "would_admit"),  # F3: dash-separated later clause
    ("may not be used for research", "negated", "negated"),
    ("Sponsored by the Institute for Research Allowed Inc", "grant", "review"),
]


@pytest.mark.parametrize("text,kind,decision", PROBES)
def test_verifier_probe_strings(text, kind, decision):
    g = cl(text)
    assert (g.kind if g else None) == kind, g
    a = pg.assess("robots_ai_statement", [g], CFG) if g else None
    assert (a["decision"] if a else None) == decision, a


@pytest.mark.parametrize("text", [
    "Disallow research bots but allow everyone else",
    "Block research crawlers, allow all others",
    "Nobody is permitted to use this for research",
    "None allowed for research",
    "ai bots blocked - research bots disallowed",  # the later clause is itself a restriction
    "ai bots blocked - research not ok",
    "Disallow research bots - allow everyone else",  # a dash does not turn a restriction object into a grant
])
def test_restriction_objects_and_new_negations_never_admit(text):
    g = cl(text)
    a = pg.assess("robots_ai_statement", [g], CFG) if g else None
    assert a is None or a["decision"] != "would_admit", (g, a)


@pytest.mark.parametrize("text,secret", [
    ("Research ok, mail jane.doe@example.org", "jane.doe"),
    ("Research ok, mail jane dot doe at example dot org: jane (at) example.org", "example.org"),
    ("Research allowed; call +1 (555) 123-4567", "555"),
    ("Research allowed, ask https://jane@example.org/contact", "jane"),
    ("Research allowed, tel:+441234567890 or @jane_doe", "jane_doe"),
])
def test_contact_details_redacted_before_storage(text, secret):
    g = cl(text)
    assert g is not None and secret not in g.verbatim and secret not in repr(pg.redact(text)), g.verbatim
    pg.shadow("robots_ai_statement", [g], "https://a.example/p")
    for e in pg.shadow_evidence():
        assert secret not in json.dumps(e, default=str)


def test_restriction_list_is_built_inside_the_hook_try(monkeypatch):
    def boom():
        raise RuntimeError("restriction list failed")
    sp._purpose_shadow("https://a.example/p", {"content-signal": "research=yes"}, None, boom)  # must not raise
    assert sp._purpose_shadow("https://a.example/p", {}, None, lambda: ["use_no_ai_signal"]) is None
