"""STREAM-1031 T10g: labels carry counts and stages only; no portal names from config/rate_limits.json."""
import json
import re
from pathlib import Path

CFG = Path(__file__).resolve().parent.parent / "config"
LABELS = json.loads((CFG / "progress-stages.json").read_text(encoding="utf-8"))
PORTALS = json.loads((CFG / "rate_limits.json").read_text(encoding="utf-8"))["portals"]


def _strings():
    def walk(n):
        if isinstance(n, dict):
            for k, v in n.items():
                if not k.startswith("_"):
                    yield from walk(v)
        elif isinstance(n, str):
            yield n
    yield from walk(LABELS)


def _names():
    out = set()
    for key, p in PORTALS.items():
        out.add(key.lower().replace("_", " "))
        out.add(key.lower())
        for h in p.get("hosts", []):
            out.add(h.lower())
    return out


def test_t10g_no_portal_names_hosts_urls():
    names = _names()
    assert "openalex" in names and len(names) > 10
    for t in _strings():
        low = t.lower()
        assert "http" not in low and "www." not in low, t
        for n in names:
            assert not re.search(r"(?<![a-z])" + re.escape(n) + r"(?![a-z])", low), (n, t)


def test_stream_labels_present_and_placeholders_known():
    s = LABELS["stages"]
    assert {"limited", "busy"} <= set(s["acquire"])
    assert s["search"]["done"] == "{found} possible sources found; {shortlisted} to check"
    assert set(LABELS["why"]) == {"1", "2", "3", "4", "5"}
    for k in ("unavailable", "connection_lost", "never_connected", "stalled", "busy"):
        assert k in LABELS["terminals"]
    known = {"rid", "inspected", "admitted", "found", "checked", "usable", "budget", "stored", "target", "attempt", "minutes",
             "shortlisted", "held", "step", "ceiling", "why_text",
             "wait_s", "k", "n", "elapsed"}  # NOISY-UI-1031 activity counts
    for t in _strings():
        assert set(re.findall(r"\{(\w+)\}", t)) <= known, t
