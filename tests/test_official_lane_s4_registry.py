"""OFFICIAL-LANE-S4A step 1 (OD-OL-1, owner 2026-11-04): the verified registry verdicts in config/entity_registry.json.
Offline: reads the config file only. Works from repo/tests or from work/OFFICIAL-LANE-S4A/staged/tests (the sibling
vera/ and config/ directories are used in both layouts)."""
import importlib.util
import json
import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent
REG_PATH = _BASE / "config" / "entity_registry.json"
QUAL = ("Treat the access verdicts as operational policy decisions, not definitive legal determinations of permitted "
        "AI use.")
EV = "work/OFFICIAL-REGISTRY-VERIFY-1104/registry-verification.md"
READ = {"Google Cloud", "Kafka", "Pulsar", "Slack"}
LINK = {"AWS", "Nvidia", "Azure", "Snowflake", "Teams", "Zoom"}
H5 = {"FEC", "Census Bureau", "EAC"}  # OD-1106-HITL H5 (a); tests/test_h5_enable_1106.py covers them


def _load(name):
    spec = importlib.util.spec_from_file_location(f"s4a_{name}", _BASE / "vera" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ol = _load("official_lane")


def _doc():
    return json.loads(REG_PATH.read_text(encoding="utf-8"))


def _active():
    return {r["entity"]: r for r in _doc()["entities"] if r.get("state") == "active" and r["entity"] not in H5}


def test_active_set_is_the_signed_off_ten():
    assert set(_active()) == READ | LINK


def test_verdicts_match_the_owner_table():
    rows = _active()
    for name in READ:
        assert ol.verdict(rows[name]) == "allow", name
        assert rows[name]["access_verdict"] == "allow" and rows[name]["terms_verdict"] == "allow"
    for name in LINK:
        assert ol.verdict(rows[name]) == "links_only", name
        assert rows[name]["access_verdict"] == "forbid" and rows[name]["terms_verdict"] == "links_only"


def test_each_row_verified_with_evidence_date_and_qualification():
    for name, r in _active().items():
        v = r["verification"]
        assert v["status"] == "verified" and v["date"] == "2026-11-04", name
        assert v["evidence"].startswith(EV) and "OD-OL-1" in v["decision"]
        assert v["verdict"] == ("read-permitted" if name in READ else "link-only")
        assert v["qualification"] == QUAL
        assert EV in r["access_evidence"] and EV in r["terms_evidence"]
    assert QUAL in _doc()["_verification_note"]


def test_paths_to_avoid_recorded_for_every_row():
    rows = _active()
    for name, r in rows.items():
        assert isinstance(r["avoid_paths"], list) and r["avoid_paths_note"], name
        assert all(isinstance(p, str) and p for p in r["avoid_paths"])
        if name in LINK:
            assert "*" in r["avoid_paths"]
    assert rows["Kafka"]["avoid_paths"] == []
    assert "/walkthroughs/*" in rows["Google Cloud"]["avoid_paths"]
    assert "/oauth*" in rows["Slack"]["avoid_paths"]


def test_domain_changes_slack_and_zoom():
    rows = _active()
    assert "docs.slack.dev" in rows["Slack"]["domains"]
    assert rows["Zoom"]["domains"] == ["support.zoom.com"]
    assert not any("support.zoom.us" in d for r in _doc()["entities"] for d in r.get("domains") or [])


def test_hosts_ascii_lowercase_exact_and_entry_urls_on_row():
    for name, r in _active().items():
        for h in r["domains"]:
            assert h.isascii() and h == h.lower() and "*" not in h and "/" not in h, (name, h)
            assert h.encode("idna").decode("ascii") == h
        for u in r.get("docs_entry_urls") or []:
            assert u.startswith("https://") and ol.host_ok(u, r), (name, u)


def test_read_rows_entry_urls_are_allowed_and_allowlists_well_formed():
    rows = _active()
    for name, r in rows.items():
        ap = r["allowed_paths"]
        assert isinstance(ap, dict) and r["allowed_paths_note"] and r["path_case_insensitive"] is False, name
        for host, spec in ap.items():
            assert host in r["domains"], (name, host)
            assert all(e.startswith("/") for e in spec.get("exact", []))
            assert all(x.startswith("/") and x.endswith("/") for x in spec.get("prefix", []))
        if name in LINK:
            assert ap == {}, name
        assert r["avoid_paths_note"].startswith("DOCUMENTATION ONLY")
    for name in READ:
        assert rows[name]["docs_entry_urls"], name
        for u in rows[name]["docs_entry_urls"]:
            assert ol.page_ok(u, rows[name]), (name, u)
    assert "Deny All" in _doc()["_verification_note"]


def test_stale_r042_wording_marked_superseded():
    assert _doc()["_terms_note"].startswith("SUPERSEDED")
    assert "UNVERIFIED against live sites at this date" not in _doc()["_calibration_note"]


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"{n} passed")
