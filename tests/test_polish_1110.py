"""POLISH-1110: retrieve_all carries landing_url through to the listed candidate record (P-1110-11 wording is in test_arch_fix_1110_bd)."""
from vera import grounded_ask as ga

CFG = {"enabled": True, "providers": ["p"], "num_results": 5, "deadline_s": 5, "snippet_max_chars": 300,
       "resolve": {"enabled": False}}


def test_landing_url_survives_retrieve_all():
    rows = [{"title": "Alpha study", "url": "https://example.org/a.pdf", "snippet": "alpha study text",
             "landing_url": "https://example.org/a"},
            {"title": "Alpha beta", "url": "https://example.org/b", "snippet": "alpha beta text"}]
    out, _ = ga.retrieve_all("alpha study", CFG, search=lambda _q: rows)
    by = {s["url"]: s for s in out}
    assert by["https://example.org/a.pdf"]["landing_url"] == "https://example.org/a"
    assert "landing_url" not in by["https://example.org/b"]
