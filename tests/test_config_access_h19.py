"""H19-T1c: the shipped config/reporting_lane.json access blocks match work/H19H21-DESIGN-1106/outlet-verdicts.md and
everything stays link-only while NC-BASIS-1 is UNCONFIRMED. Offline."""
import copy
import json
from pathlib import Path

from vera import news_access as na, reporting_lane as rl

CFG = json.loads((Path(__file__).resolve().parent.parent / "config" / "reporting_lane.json").read_text(encoding="utf-8"))
ROWS = {o["host"]: o for o in CFG["outlets"]}
PERMITTED = ["commondreams.org", "democracynow.org", "votebeat.org", "chalkbeat.org", "kffhealthnews.org"]


def url(h):
    return f"https://www.{h}/a/1"


def test_nc_basis_ships_unconfirmed_and_every_nd_row_is_link_only():
    assert CFG["nc_basis"]["status"] == "UNCONFIRMED" and not na.nc_basis_confirmed(CFG)
    for h in PERMITTED + ["truthout.org", "texastribune.org", "propublica.org"]:
        v = na.verdict(url(h), CFG)
        assert v["mode"] == "link_only" and "nc_basis_unconfirmed" in v["reasons"] and v["has_access"], h


def test_permitted_rows_resolve_full_text_once_basis_is_flipped():
    c = copy.deepcopy(CFG)
    c["nc_basis"] = {"status": "SATISFIED"}
    for h in PERMITTED:
        assert na.verdict(url(h), c) == {"mode": "full_text", "reasons": [], "has_access": True}, h


def test_permitted_row_values():
    for h in PERMITTED:
        a = ROWS[h]["access"]
        assert ROWS[h]["terms_verdict"] == "allow" and ROWS[h]["terms_evidence"]
        assert a["terms_ai_clause"]["status"] == "PERMITTED" and a["robots"]["status"] == "SUMMARY_UNTESTED_UA"
        assert a["licence"]["id"].startswith("CC-BY-NC-ND-") and a["licence"]["evidence_url"] and a["evidence_level"] == "summary"
        assert a["feed_url"].startswith("https://")
    assert ROWS["democracynow.org"]["access"]["crawl_delay_s"] == 10
    assert ROWS["commondreams.org"]["access"]["licence"]["id"] == "CC-BY-NC-ND-3.0"
    assert ROWS["propublica.org"]["access"]["licence"]["id"] == "CC-BY-NC-ND-3.0"
    for h in ("votebeat.org", "chalkbeat.org", "kffhealthnews.org"):
        assert ROWS[h]["access"]["licence"]["id"] == "CC-BY-NC-ND-4.0"
    assert "/republish/" in ROWS["votebeat.org"]["deny_path_prefixes"]
    assert any("texas tribune" in m for m in ROWS["chalkbeat.org"]["republished_markers"])
    assert "all rights reserved" in ROWS["kffhealthnews.org"]["exclude_item_markers"]


def test_exceptions_link_only_with_their_tokens_even_when_basis_flipped():
    c = copy.deepcopy(CFG)
    c["nc_basis"] = {"status": "SATISFIED"}
    assert "acc_ai_restricted" in na.verdict(url("truthout.org"), c)["reasons"]
    tt = na.verdict(url("texastribune.org"), c)["reasons"]
    assert "acc_licence_unset" in tt and "acc_ai_restricted" in tt
    assert na.mode(url("truthout.org"), c) == na.mode(url("texastribune.org"), c) == "link_only"
    page = {"text": "This story was originally published by The Texas Tribune."}
    assert "acc_republished" in na.verdict(url("chalkbeat.org"), c, page)["reasons"]
    assert "acc_item_excluded" in na.verdict(url("kffhealthnews.org"), c, {"text": "All Rights Reserved"})["reasons"]


def test_propublica_reconciliation_stays_exclude():
    assert ROWS["propublica.org"]["terms_verdict"] == "exclude" and rl.outlet_verdict(url("propublica.org"), CFG) == "exclude"
    assert "reconciliation" in ROWS["propublica.org"]["access"]
    assert ROWS["theconversation.com"]["terms_verdict"] == "exclude" and "access" not in ROWS["theconversation.com"]


def test_no_lane_sees_allow_for_access_rows_while_flag_off():
    for h in PERMITTED:
        assert rl.outlet_verdict(url(h), CFG) == "links_only", h
    assert rl.outlet_verdict(url("globalvoices.org"), CFG) == "allow"
