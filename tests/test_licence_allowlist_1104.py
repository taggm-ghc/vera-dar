"""LICENCE-ALLOWLIST-1104: deny-all licence allowlist (owner 2026-11-04, option 1) and the licence research flow."""
import json

import pytest

from vera import corpus_admission
from vera.licence_gate import (ALLOWLIST_PATH, RC_ALLOW, RC_CONFLICT, RC_MISSING, RC_NOT_STORABLE, RC_UNKNOWN,
                               _load_allowlist, apply_licence_gate, licence_decision, licence_status, match_entry,
                               norm_id, norm_url)
from scripts import licence_research as lr


def c(content=None, url=None, meta=None, kind="full_content", page="https://x.org/a"):
    return {"url": page, "stored_text_kind": kind, "gate_a_decision": "fetch", "gate_a_rationale": "r",
            "licence": {"metadata": {"id": meta} if meta else {},
                        "content": {"id": content, "url": url} if (content or url) else {}}}


@pytest.mark.parametrize("lid", ["CC-BY-4.0", "CC BY 4.0", "cc by", "CC-BY", "cc-by-sa", "CC BY-SA 4.0", "CC0",
                                 "cc0 1.0 universal", "CC-BY-3.0", "Public Domain", "OGL-UK-3.0", "MIT", "Apache-2.0",
                                 "US Government Work", "Public Domain Mark",
                                 "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International"])
def test_allowlisted_licences_pass(lid):
    d, why, code = licence_status(c(content=lid))
    assert (d, code) == ("allow", RC_ALLOW), why


@pytest.mark.parametrize("lid", ["CC-BY-NC-4.0", "CC BY-NC 4.0", "cc-by-nc-sa-4.0", "CC-BY-NC-SA", "by-nc", "cc by-nc"])
def test_plain_nc_accepted(lid):
    assert licence_decision(c(content=lid))[0] == "allow"


@pytest.mark.parametrize("lid", ["All rights reserved", "Copyright. All Rights Reserved.", "proprietary",
                                 "Elsevier TDM", "arXiv nonexclusive distrib 1.0", "open-access"])
def test_recognised_non_storable_is_link_only(lid):
    d, _why, code = licence_status(c(content=lid))
    assert (d, code) == ("defer", RC_NOT_STORABLE)


@pytest.mark.parametrize("lid", ["some publisher licence", "proprietary-eula", "CC BY 2.0", "Custom Terms v7",
                                 "cc-by-4.0-de", "Creative Commons Attribution"])
def test_unknown_is_link_only_and_queued(lid):
    d, why, code = licence_status(c(content=lid))
    assert (d, code) == ("defer", RC_UNKNOWN) and "link-only" in why


@pytest.mark.parametrize("lid", ["CC-BY-ND-4.0", "CC BY-NC-ND 4.0", "cc by-nd", "Strictly for fee", "Fee Only",
                                 "No Educational Use", "subscription required"])
def test_nd_fee_and_no_education_rejected(lid):
    assert licence_decision(c(content=lid))[0] == "reject"


@pytest.mark.parametrize("lid,url", [("Elsevier user license", None), ("Custom", "https://www.elsevier.com/open-access/userlicense/1.0/")])
def test_elsevier_user_licence_rejected_nd_class(lid, url):
    d, why, code = licence_status(c(content=lid, url=url))
    assert (d, code) == ("reject", "licence_reject") and "ND" in why


def test_missing_and_metadata_rules_unchanged():
    assert licence_status(c())[2] == RC_MISSING
    assert licence_decision(c(url="https://creativecommons.org/licenses/by/4.0/"))[0] == "defer"  # URL-only = undeclared
    assert licence_decision(c(meta="CC0-1.0"))[0] == "defer"
    assert licence_decision(c(meta="CC0-1.0", kind="abstract_metadata"))[0] == "allow"
    assert licence_decision(c(meta="All rights reserved", kind="abstract_metadata"))[0] == "defer"


def test_url_used_only_when_id_unrecognised_and_conflict_holds():
    by4 = "http://www.creativecommons.org/licenses/by/4.0/legalcode.en"
    assert licence_decision(c(content="CC BY (version not stated)", url=by4))[0] == "allow"
    assert licence_decision(c(content="Custom", url="https://example.org/terms"))[0] == "defer"
    assert licence_status(c(content="CC-BY-SA-4.0", url=by4))[2] == RC_CONFLICT
    assert licence_decision(c(content="All rights reserved", url=by4))[0] == "defer"  # recognised id governs


def test_normalisers():
    assert norm_id("CC_BY_NC_SA_4.0") == norm_id("cc by-nc-sa 4.0") == "cc-by-nc-sa-4.0"
    assert norm_id("CC Zero") == norm_id("CC0-1.0") == "cc0-1.0"
    assert norm_url("HTTPS://www.CreativeCommons.org/licenses/by/4.0/deed.en") == "creativecommons.org/licenses/by/4.0"
    assert norm_url("not a url") == ""


def test_seed_shape_and_evidence():
    entries = json.loads(ALLOWLIST_PATH.read_text())["entries"]
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids))
    for e in entries:
        assert {"id", "status", "storable", "attribution_required", "nc", "reason", "evidence_url"} <= set(e), e["id"]
        assert e["status"] in ("active", "proposed")
        if e["status"] == "active":
            assert e["approved_by"] and str(e["evidence_url"] or "https://").startswith("http")
        if e["storable"] and e["status"] == "active":
            assert e["nd"] is False and e["evidence_url"].startswith("https://")


def test_proposed_rows_are_never_used(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"entries": [
        {"id": "LicenseRef-x", "status": "proposed", "storable": True, "aliases": ["x licence"], "urls": []}]}))
    assert match_entry("x licence", path=p) == (None, "none")
    assert _load_allowlist(p) == ({}, {})


def test_missing_allowlist_fails_verbosely(tmp_path):
    with pytest.raises(RuntimeError, match="licence allowlist unreadable"):
        _load_allowlist(tmp_path / "nope.json")


def test_unknown_recorded_in_licence_observations_rows():
    cs = apply_licence_gate([c(content="Mystery Licence 2"), c(content="CC-BY-4.0"), c(content="All rights reserved")])
    codes = [r["reason_code"] for x in cs for r in corpus_admission.licence_observation_rows(x, 1, now="2026-11-04")]
    assert codes == [RC_UNKNOWN] * 2 + [RC_ALLOW] * 2 + [RC_NOT_STORABLE] * 2
    rows = corpus_admission.licence_observation_rows(cs[0], 1, now="2026-11-04")
    gov = [r for r in rows if r["governing"]][0]
    assert gov["spdx"] == "Mystery Licence 2" and gov["decision"] == "hold" and gov["url"] is None
    assert "reason_code = 'licence_unknown'" in lr.QUEUE_SQL and "licence_observations" in lr.QUEUE_SQL


def test_queue_is_bounded_ranked_and_skips_known():
    obs = [{"licence": "Mystery A", "count": 3, "first_seen": "2026-11-01", "last_seen": "2026-11-02"},
           {"licence": "mystery-a", "count": 2, "first_seen": "2026-10-30", "last_seen": "2026-11-03"},
           {"licence": "CC-BY-4.0", "count": 99}, {"licence": "CC BY 2.0", "count": 50},  # active / proposed: known
           {"licence": "Mystery B", "count": 1}]
    q = lr.build_queue(obs, max_items=5)
    assert [m["key"] for m in q] == ["mystery-a", "mystery-b"]
    assert q[0]["count"] == 5 and q[0]["first_seen"] == "2026-10-30" and q[0]["last_seen"] == "2026-11-03"
    assert len(lr.build_queue(obs, max_items=1)) == 1
    with pytest.raises(lr.ResearchError):
        lr.build_queue(obs, max_items=0)


@pytest.mark.parametrize("finding,outcome", [
    ({"grants_store_and_redistribute": True, "nd": False, "nc": True}, "storable"),
    ({"grants_store_and_redistribute": True, "nd": True}, "reject"),
    ({"grants_store_and_redistribute": True, "nd": False, "fee_only": True}, "reject"),
    ({"grants_store_and_redistribute": True, "nd": False, "no_educational_use": True}, "reject"),
    ({"grants_store_and_redistribute": None, "nd": False}, "link-only"),
    ({"grants_store_and_redistribute": True, "nd": None}, "link-only")])
def test_proposal_classification(finding, outcome):
    base = {"licence": "Mystery", "evidence_url": "https://lic.example/terms", "evidence_read_state": "full text"}
    assert lr.proposal({**base, **finding})["proposed_outcome"] == outcome


def test_proposal_flow_never_activates(tmp_path, monkeypatch):
    f = tmp_path / "findings.jsonl"
    f.write_text(json.dumps({"licence": "Mystery Open", "evidence_url": "https://lic.example/t",
                             "evidence_read_state": "full text", "grants_store_and_redistribute": True,
                             "nd": False, "quote": "you may copy and redistribute"}) + "\n")
    before = ALLOWLIST_PATH.read_bytes()
    out = tmp_path / "proposals.jsonl"
    assert lr.main(["propose", "--findings", str(f), "--out", str(out)]) == 0
    row = json.loads(out.read_text())
    assert row["status"] == "proposed" and row["approved_by"] is None and row["storable"] is True
    assert ALLOWLIST_PATH.read_bytes() == before
    assert licence_decision(c(content="Mystery Open"))[0] == "defer"
    assert lr.main(["propose", "--findings", str(f), "--out", str(ALLOWLIST_PATH)]) == 2  # refused
    assert ALLOWLIST_PATH.read_bytes() == before


def test_proposal_requires_evidence():
    with pytest.raises(lr.ResearchError):
        lr.proposal({"licence": "X", "evidence_read_state": "snippet"})
    with pytest.raises(lr.ResearchError):
        lr.proposal({"licence": "X", "evidence_url": "https://e", "evidence_read_state": "s", "quote": "w " * 30})
