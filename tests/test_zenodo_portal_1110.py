"""PHYS-SUPPLY-1110 (3): zenodo.org rate portal. Terms and robots read 2026-11-09 (summarizing fetch): Crawl-delay 10,
guest 60 requests/min and 2000/h, licence per deposited record. The portal paces requests only; it opens no acquisition
route (the repository-PDF route stays pending and off). Config reads and a fake transport only: no socket, no DB."""
from __future__ import annotations

import json
import pathlib

import pytest

from vera import rate_limit

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _cfg(name):
    return json.loads((ROOT / "config" / name).read_text(encoding="utf-8"))


def test_zenodo_portal_is_at_or_below_the_robots_crawl_delay_and_guest_limits():
    p = _cfg("rate_limits.json")["portals"]["zenodo"]
    assert p["hosts"] == ["zenodo.org"] and p["max_concurrency"] == 1 and p["burst"] == 1
    assert p["rate_per_s"] <= 1 / 10  # robots Crawl-delay 10 (stricter than 60 per minute)
    assert p["daily_cap"] is not None and p["daily_cap"] <= 2000  # under one hour of the documented guest allowance
    assert 429 in p["throttle_statuses"] and all(s["read"] == "2026-11-09" for s in p["sources"])
    assert rate_limit.get_registry().portal_for_url("https://zenodo.org/records/1234567") is not None


def test_portal_opens_no_acquisition_route_and_repository_pdf_row_stays_pending_and_off():
    from vera import acq_config
    acq = acq_config.load_config()
    assert "zenodo.org" not in {h.lower() for h in (acq.get("fetch") or {}).get("allowed_hosts") or ()}
    assert "zenodo.org" not in {h.lower() for h in (acq.get("arxiv_route") or {}).get("hosts") or ()}
    assert acq["repository_pdf_route"]["enabled"] is False
    row = next(h for h in _cfg("repository_pdf.json")["hosts"] if h["host"] == "zenodo.org")
    assert row["status"] == "pending" and row["rate_portal"] == "zenodo" and row["crawl_delay_s"] >= 10
    assert row["allowed_path_pattern"] == r"^/records/[0-9]+/files/[^/]+\.pdf$"


def test_acquisition_fetch_of_a_zenodo_record_is_still_refused_before_any_request():
    from tests import acq_support as A
    from vera import acq_config, auto_acquire
    sent = []
    deps = A.deps([], lambda *a, **k: sent.append(a) or pytest.fail("sent"))
    cfg = A.cfg()
    with pytest.raises(auto_acquire._Stop) as e:
        auto_acquire._guarded_get("https://zenodo.org/records/1234567/files/x.pdf", cfg, deps, 1000, 5.0, False)
    assert e.value.code == acq_config.reason(cfg, "host") and sent == []
