"""H19-T1: offline tests for vera/news_access.py. No network, no DB."""
import copy

from vera import news_access as na

GOOD = {"host": "example-news.org", "terms_verdict": "allow", "terms_evidence": "e",
        "access": {"licence": {"id": "CC-BY-NC-ND-4.0", "evidence_url": "https://x/l"},
                   "nc_eligible": {"status": "SATISFIED", "basis": "b"},
                   "terms_ai_clause": {"status": "PERMITTED", "evidence_url": "https://x/t"},
                   "robots": {"status": "ALLOWS", "ai_agents_blocked": False},
                   "crawl_delay_s": 10}}
URL = "https://www.example-news.org/a/1"


def cfg(row=None, nc=True, **kw):
    c = {"outlets": [copy.deepcopy(row if row is not None else GOOD)], "unlisted_verdict": "links_only"}
    if nc:
        c["nc_basis"] = {"status": "SATISFIED"}
    c.update(kw)
    return c


def mut(path, val):
    r = copy.deepcopy(GOOD)
    d = r
    for p in path[:-1]:
        d = d[p]
    if val is None:
        d.pop(path[-1], None)
    else:
        d[path[-1]] = val
    return r


def test_all_satisfied_full_text():
    v = na.verdict(URL, cfg())
    assert v == {"mode": "full_text", "reasons": [], "has_access": True}
    assert na.full_text_ok(URL, cfg())


def test_nc_basis_unconfirmed_default_and_flip():
    for c in (cfg(nc=False), cfg(nc_basis={"status": "UNCONFIRMED"}), cfg(nc_basis={"status": "satisfied"}),
              cfg(nc_basis=True)):
        v = na.verdict(URL, c)
        assert v["mode"] == "link_only" and v["reasons"] == ["nc_basis_unconfirmed"]
    assert na.mode(URL, cfg()) == "full_text"  # the one config value flips it


def test_a1_rows():
    assert na.reasons(URL, {"outlets": []}) == ["acc_no_allow_row"]
    r = mut(["terms_verdict"], "links_only")
    assert na.reasons(URL, cfg(r)) == ["acc_no_allow_row"]
    r = mut(["terms_evidence"], "")
    assert na.mode(URL, cfg(r)) == "link_only"
    r = copy.deepcopy(GOOD); r.pop("access")
    v = na.verdict(URL, cfg(r))
    assert v["mode"] == "link_only" and v["has_access"] is False
    assert na.full_text_ok(URL, cfg(r))  # no access block: non-ND behaviour unchanged


def test_a2_licence():
    assert "acc_licence_unset" in na.reasons(URL, cfg(mut(["access", "licence", "id"], None)))
    assert "acc_licence_unset" in na.reasons(URL, cfg(mut(["access", "licence", "evidence_url"], "")))
    assert "acc_licence_family" in na.reasons(URL, cfg(mut(["access", "licence", "id"], "CC-BY-ND-4.0")))
    assert "acc_licence_family" in na.reasons(URL, cfg(mut(["access", "licence", "id"], "CC-BY-4.0")))
    assert na.mode(URL, cfg(mut(["access", "licence", "id"], "CC BY-NC-ND 3.0"))) == "full_text"


def test_a3_nc():
    assert "acc_nc_unresolved" in na.reasons(URL, cfg(mut(["access", "nc_eligible", "status"], "UNRESOLVED")))
    assert "acc_nc_ineligible" in na.reasons(URL, cfg(mut(["access", "nc_eligible", "status"], "NOT_ELIGIBLE")))
    assert "acc_nc_unresolved" in na.reasons(URL, cfg(mut(["access", "nc_eligible", "basis"], "")))
    assert "acc_nc_unresolved" in na.reasons(URL, cfg(mut(["access", "nc_eligible", "status"], "bogus")))


def test_a4_ai():
    assert "acc_ai_restricted" in na.reasons(URL, cfg(mut(["access", "terms_ai_clause", "status"], "RESTRICTED")))
    assert "acc_ai_unresolved" in na.reasons(URL, cfg(mut(["access", "terms_ai_clause", "status"], "UNRESOLVED")))
    assert "acc_ai_unresolved" in na.reasons(URL, cfg(mut(["access", "terms_ai_clause", "evidence_url"], "")))


def test_a5_robots_and_alone_not_enough():
    assert "acc_robots" in na.reasons(URL, cfg(mut(["access", "robots", "status"], "DISALLOWS")))
    assert "acc_robots" in na.reasons(URL, cfg(mut(["access", "robots", "ai_agents_blocked"], True)))
    # A5 alone: robots ALLOWS, everything else unset -> never full_text
    r = {"host": "example-news.org", "terms_verdict": "allow", "terms_evidence": "e",
         "access": {"robots": {"status": "ALLOWS", "ai_agents_blocked": False}}}
    v = na.verdict(URL, cfg(r))
    assert v["mode"] == "link_only" and "acc_robots" not in v["reasons"]
    assert {"acc_licence_unset", "acc_nc_unresolved", "acc_ai_unresolved"} <= set(v["reasons"])


def test_a6_delay():
    assert na.mode(URL, cfg(mut(["access", "crawl_delay_s"], None))) == "full_text"
    for bad in (-1, "10", True):
        assert "acc_delay_bad" in na.reasons(URL, cfg(mut(["access", "crawl_delay_s"], bad)))


def test_a7_page_licence():
    ok = {"licence_href": "https://creativecommons.org/licenses/by-nc-nd/4.0/"}
    assert na.mode(URL, cfg(), ok) == "full_text"
    for href in ("https://creativecommons.org/licenses/by/4.0/", "https://creativecommons.org/licenses/by-nd/4.0/"):
        assert "acc_page_licence_mismatch" in na.reasons(URL, cfg(), {"licence_href": href})


def test_a8_a9_a10_page():
    r = mut(["republished_markers"], ["originally published by the Other"])
    r["exclude_item_markers"] = ["All Rights Reserved"]
    assert "acc_republished" in na.reasons(URL, cfg(r), {"text": "This was Originally Published by the Other."})
    assert "acc_item_excluded" in na.reasons(URL, cfg(r), {"text": "All Rights Reserved"})
    assert na.mode(URL, cfg(r), {"text": "plain"}) == "full_text"
    assert "acc_optout_meta" in na.reasons(URL, cfg(), {"meta_robots": "noai"})
    assert "acc_optout_xrt" in na.reasons(URL, cfg(), {"x_robots_tag": "noindex"})
    assert "acc_tdm_reserved" in na.reasons(URL, cfg(), {"tdm_reservation": "1"})
    assert "acc_tdm_unreadable" in na.reasons(URL, cfg(), {"tdm_reservation": "unreadable"})


def test_fail_closed_garbage():
    for u in ("", None, "not a url", "https://other.org/x"):
        assert na.mode(u, cfg()) == "link_only"
    assert na.mode(URL, {}) == "link_only"
    r = copy.deepcopy(GOOD); r["access"] = "yes"
    assert na.mode(URL, cfg(r)) == "link_only"
    assert not na.full_text_ok(URL, {"outlets": [dict(GOOD, terms_verdict="exclude")]})


# ---- AD-6: operations matrix, gate order, commercial flag, version strictness ----
def test_matrix_defaults_and_unknown_op():
    row = GOOD
    assert na.operation_verdict(row, "metadata") == "ALLOW"
    assert na.operation_verdict(row, "rag_context") == "ALLOW_IF_NONCOMMERCIAL"
    assert na.operation_verdict(row, "retain_original") == "DEFER"
    assert na.operation_verdict(row, "persist_embeddings") == "NOT_PHASE_1"
    for op in ("summary", "close_paraphrase", "translation", "style_transform", "training", "commercial_use", "zzz"):
        assert na.operation_verdict(row, op) == "BLOCK"


def test_forged_operations_cannot_loosen_but_can_tighten():
    r = copy.deepcopy(GOOD)
    r["access"]["operations"] = {"summary": "ALLOW", "training": "ALLOW", "verbatim_quote": "BLOCK", "metadata": "weird"}
    assert na.operation_verdict(r, "summary") == "BLOCK" and na.operation_verdict(r, "training") == "BLOCK"
    assert na.operation_verdict(r, "verbatim_quote") == "BLOCK"
    assert na.operation_verdict(r, "metadata") == "ALLOW"
    for op in ("summary", "training", "translation"):
        e = na.evaluate(URL, op, cfg(r))
        assert not e["allowed"] and e["gate"] == "processing" and e["reasons"] == ["acc_op_blocked"]


def test_evaluate_ops_allowed_and_deferred():
    for op in ("metadata", "rag_context", "fact_extraction", "verbatim_quote"):
        assert na.evaluate(URL, op, cfg())["allowed"], op
    for op in ("retain_original", "persist_embeddings"):
        e = na.evaluate(URL, op, cfg())
        assert not e["allowed"] and e["reasons"] == ["acc_op_not_now"]


def test_gate_order_stops_at_first_refusal():
    c = cfg(mut(["access", "robots", "status"], "DISALLOWS"), nc=False)  # also bad licence below
    c["outlets"][0]["access"]["licence"]["id"] = "CC-BY-ND-4.0"
    e = na.evaluate(URL, "summary", c)
    assert e["gate"] == "robots" and e["reasons"] == ["acc_robots"] and e["gates_evaluated"] == ["robots"]
    c = cfg(mut(["access", "licence", "id"], "CC-BY-ND-4.0"), nc=False)
    e = na.evaluate(URL, "summary", c)
    assert e["gate"] == "licence" and e["gates_evaluated"] == ["robots", "licence"]
    e = na.evaluate(URL, "summary", cfg(nc=False))
    assert e["gate"] == "processing" and e["reasons"] == ["nc_basis_unconfirmed", "acc_op_blocked"]
    assert e["gates_evaluated"] == ["robots", "licence", "processing"]
    e = na.evaluate(URL, "fact_extraction", cfg(), output_refusal=["g1_verbatim"])
    assert e["gate"] == "output" and e["reasons"] == ["g1_verbatim"] and e["gates_evaluated"][-1] == "output"
    e = na.evaluate(URL, "fact_extraction", cfg(), output_refusal=[])
    assert e["allowed"] and e["gates_evaluated"] == ["robots", "licence", "processing", "output"]
    # page-level opt-out is a robots-gate refusal even when everything else is fine
    e = na.evaluate(URL, "rag_context", cfg(), {"meta_robots": "noai", "licence_href": "https://creativecommons.org/licenses/by/4.0/"})
    assert e["gate"] == "robots" and e["gates_evaluated"] == ["robots"]


def test_commercial_flag_makes_all_rows_nc_ineligible():
    for k in ("fee", "advertising", "sale"):
        c = cfg(commercial_flags={k: True})
        v = na.verdict(URL, c)
        assert v["mode"] == "link_only" and "acc_nc_ineligible" in v["reasons"]
    assert na.mode(URL, cfg(commercial_flags={"fee": False, "advertising": False, "sale": False})) == "full_text"


def test_strict_version():
    assert na.strict_version(mut(["access", "licence", "id"], "CC-BY-NC-ND-3.0")) is True
    assert na.strict_version(mut(["access", "licence", "id"], "CC-BY-NC-ND")) is True
    assert na.strict_version(GOOD) is False
    assert na.evaluate(URL, "rag_context", cfg(mut(["access", "licence", "id"], "CC-BY-NC-ND-3.0")))["strict_version"]


def test_no_access_block_and_no_row_evaluate():
    r = copy.deepcopy(GOOD); r.pop("access")
    e = na.evaluate(URL, "rag_context", cfg(r))
    assert not e["allowed"] and e["reasons"] == ["acc_no_access_block"] and e["has_access"] is False
    assert na.evaluate("https://other.org/", "rag_context", cfg())["reasons"] == ["acc_no_allow_row"]
