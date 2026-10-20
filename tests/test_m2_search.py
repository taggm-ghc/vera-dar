"""Search tests: offline only. Fixtures are HAND-MADE (written from the documented Atom shape), NOT recorded
arXiv responses. Any real transport call fails the test."""
import ast
import hashlib
import json
from pathlib import Path

import pytest
import requests

import vera.search_providers as sp
from vera.cost_ledger import CostLedger
from vera.licence_gate import apply_licence_gate, licence_decision
from vera.search_and_fetch import (SearchConfigError, SearchError, SearchResults, normalize_url,
                                   search_question)
from vera.search_providers import ArxivProvider, RateGate, SeedFileProvider, licence_record

ROOT = Path(__file__).resolve().parent.parent
ARXIV_SETTINGS = {"endpoint": "https://export.arxiv.org/api/query", "min_interval_s": 3.0, "max_concurrency": 1,
                  "max_response_bytes": 100000, "terms_source": "x",
                  "query": {"max_terms": 8, "min_term_chars": 2, "operator": "OR",
                            "submitted_from": "2023-01-01", "submitted_to": "2026-12-31"}}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real network transport called in a test")
    monkeypatch.setattr(sp, "arxiv_default_transport", boom)
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)


class Resp:
    def __init__(self, status=200, text="", headers=None, payload=None):
        self.status_code, self.text, self.headers, self.payload = status, text, headers or {}, payload


def entry(i, url=None, doi=None):
    url = url or f"http://arxiv.org/abs/2401.0000{i}v1"
    d = f"<arxiv:doi>{doi}</arxiv:doi>" if doi else ""
    return (f"<entry><id>{url}</id><title> T{i}\n x</title><summary> abstract {i} </summary>"
            f"<published>2024-01-0{i}T00:00:00Z</published>{d}"
            f'<link title="pdf" href="http://arxiv.org/pdf/2401.0000{i}v1"/></entry>')


def feed(*entries):
    return ('<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">'
            + "".join(entries) + "</feed>")


class FakeProvider:
    required_env = ()
    endpoint = "fake"

    def __init__(self, name, seq, interval=0.0, basis="no_published_price_keyless"):
        self.name, self.seq, self.gate, self.basis, self.calls = name, list(seq), RateGate(interval), basis, []

    def call(self, q, n, t):
        self.calls.append(q)
        x = self.seq.pop(0)
        if isinstance(x, Exception):
            raise x
        return x

    def parse(self, resp, n):
        return [{"url": u, "title": "t", "snippet": "s"} for u in resp.payload][:n]

    def cost_for(self, resp):
        return 0.0, True, self.basis

    def failure_basis(self):
        return self.basis


def ok(*urls):
    return Resp(payload=list(urls))


def run(providers, q="q", **kw):
    return search_question(q, kw.pop("n", 5), providers=providers, sleep=lambda s: None, **kw)


def test_dedupe_rank_filter_and_contract():
    out = run([FakeProvider("f", [ok("https://a.com/x/", "https://A.com/x#frag", "ftp://bad/y", "https://b.com/z")])])
    assert [c["url"] for c in out] == ["https://a.com/x/", "https://b.com/z"]
    assert [c["rank"] for c in out] == [1, 2] and all(c["provider"] == "f" for c in out)
    assert isinstance(out, SearchResults)


def test_known_zero_cost_with_basis_on_every_record():
    led = CostLedger()
    run([FakeProvider("f", [Resp(429, headers={"Retry-After": "0"}), requests.ConnectionError("x"), ok("https://a.com")])],
        ledger=led)
    assert led.unknown_cost_calls() == 0 and led.total_usd() == 0.0
    assert [e.ok for e in led.entries] == [False, False, True]
    assert {e.cost_basis for e in led.entries} == {"no_published_price_keyless"}


def test_failure_raises_searcherror_never_empty():
    with pytest.raises(SearchError) as e:
        run([FakeProvider("f", [Resp(500), Resp(500), Resp(500)])])
    assert e.value.args[0].count("attempt") == 3
    with pytest.raises(SearchError, match="401"):
        run([FakeProvider("f", [Resp(401, text="no")])])
    assert run([FakeProvider("f", [ok()])]) == []  # a genuine empty result is the only [] 


def test_empty_chain_and_bad_args():
    with pytest.raises(SearchConfigError):
        run([])
    with pytest.raises(ValueError):
        search_question(" ", providers=[])
    with pytest.raises(ValueError):
        search_question("q", 0, providers=[])


def test_rate_limit_fake_clock_no_real_sleep():
    t = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        t[0] += s
    p = FakeProvider("f", [ok("https://a.com"), ok("https://b.com")], interval=3.0)
    for _ in range(2):
        search_question("q", 5, providers=[p], sleep=sleep, clock=lambda: t[0])
    assert slept == [3.0] and len(p.calls) == 2  # second call waited the 3 s interval


def test_interval_beyond_deadline_stops():
    p = FakeProvider("f", [ok("https://a.com")], interval=100.0)
    p.gate.stamp(0.0)
    with pytest.raises(SearchError, match="deadline"):
        search_question("q", providers=[p], sleep=lambda s: None, clock=lambda: 0.0, deadline_s=10)


def test_fewer_than_min_advances_then_flags_below_minimum():
    a = FakeProvider("a", [ok("https://a.com/1")])
    b = FakeProvider("b", [ok(*[f"https://b.com/{i}" for i in range(5)])])
    out = run([a, b], min_candidates=5)
    assert out.meta["route_used"] == "b" and not out.meta["below_minimum"] and len(out) == 5
    c = FakeProvider("c", [ok("https://c.com/1", "https://c.com/2")])
    out = run([c], min_candidates=5)
    assert out.meta["below_minimum"] is True and len(out) == 2  # disclosed, not silent
    a2 = FakeProvider("a", [ok(*[f"https://a.com/{i}" for i in range(5)])])
    b2 = FakeProvider("b", [])
    run([a2, b2], min_candidates=5)
    assert b2.calls == []  # enough from the first route: no advance


def test_failed_first_route_advances_and_all_failed_lists_each():
    a = FakeProvider("a", [Resp(500)] * 3)
    b = FakeProvider("b", [ok("https://b.com")])
    assert run([a, b])[0]["provider"] == "b"
    with pytest.raises(SearchError) as e:
        run([FakeProvider("a", [Resp(500)] * 3), FakeProvider("b", [Resp(503)] * 3)])
    assert "a:" in e.value.args[0] and "b:" in e.value.args[0]


# ---- arXiv parsing (hand-made fixtures) and the abstract-page-only rule
def test_arxiv_parse_abstract_pages_only_with_metadata_licence():
    body = feed(entry(1, doi="10.1/X"), entry(2, url="http://arxiv.org/pdf/2401.00002v1"),
                entry(3, url="http://arxiv.org/html/2401.00003v1"), entry(4))
    items = ArxivProvider(ARXIV_SETTINGS).parse(Resp(text=body), 10)
    assert [i["url"] for i in items] == ["https://arxiv.org/abs/2401.00001v1", "https://arxiv.org/abs/2401.00004v1"]
    assert items[0]["title"] == "T1 x" and items[0]["doi"] == "10.1/X"
    assert items[0]["licence"]["metadata"]["id"] == "CC0-1.0" and items[0]["licence"]["content"]["id"] is None


@pytest.mark.parametrize("body", [
    "<notxml", "<rss/>", '<!DOCTYPE x [<!ENTITY a "b">]><feed xmlns="http://www.w3.org/2005/Atom"/>',
    feed("<entry><id>http://arxiv.org/api/errors#x</id><title>Error</title><summary>bad</summary></entry>"),
    "x" * 200000])
def test_arxiv_bad_body_is_failure_not_empty(body):
    with pytest.raises(sp.ProviderResponseError):
        ArxivProvider(ARXIV_SETTINGS).parse(Resp(text=body), 5)
    with pytest.raises(SearchError):
        run([ArxivProvider(ARXIV_SETTINGS, transport=lambda *a: Resp(text=body))], q="ai coding productivity")


def test_arxiv_end_to_end_with_injected_transport_and_no_email():
    seen = {}

    def tx(url, params, headers, timeout):
        seen.update(url=url, params=params, headers=headers)
        return Resp(text=feed(entry(1)))
    led = CostLedger()
    out = run([ArxivProvider(ARXIV_SETTINGS, transport=tx)], q="ai coding productivity", ledger=led)
    assert out[0]["provider"] == "arxiv" and seen["params"]["max_results"] == 5
    assert "all:coding" in seen["params"]["search_query"] and "submittedDate:[202301010000" in seen["params"]["search_query"]
    assert "@" not in json.dumps(seen["headers"])  # no contact e-mail is sent
    assert led.entries[0].cost_basis == "no_published_price_keyless" and led.entries[0].cost_known


def test_default_transport_is_the_only_network_path():
    p = ArxivProvider(ARXIV_SETTINGS)  # no transport injected: the (patched) default must be reached
    with pytest.raises(AssertionError, match="real network"):
        p.call("coding assistants productivity", 3, 5)


# ---- licence gate
def cand(meta=None, content=None, kind="full_content", url="https://x.org/a", gate="fetch"):
    return {"url": url, "licence": licence_record({"id": meta, "url": None, "source": "t"} if meta else None,
                                                  {"id": content, "url": None, "source": "t"} if content else None),
            "stored_text_kind": kind, "gate_a_decision": gate, "gate_a_rationale": "r"}


def test_licence_gate_rules():
    assert licence_decision(cand(content="CC-BY-4.0"))[0] == "allow"
    assert licence_decision(cand())[0] == "defer"                                   # none declared
    assert licence_decision(cand(meta="CC0-1.0"))[0] == "defer"                     # CC0 metadata != content licence
    assert licence_decision(cand(meta="CC0-1.0", kind="abstract_metadata"))[0] == "allow"
    for lic in ("CC-BY-ND-4.0", "CC BY-NC-ND 4.0"):     # R1 rule: ND rejects; plain NC is accepted (item 72 W4)
        assert licence_decision(cand(content=lic))[0] == "reject", lic
    for lic in ("CC-BY-NC-4.0", "by-nc-sa"):
        assert licence_decision(cand(content=lic))[0] == "allow", lic
    assert licence_decision(cand(meta="CC-BY-ND-4.0", content="CC-BY-4.0"))[0] == "reject"
    assert licence_decision(cand(content="CC-BY-4.0", url="https://arxiv.org/pdf/2401.1"))[0] == "reject"
    assert licence_decision(cand(content="CC-BY-4.0", url="https://arxiv.org/abs/2401.1"))[0] == "allow"


def test_apply_gate_overrides_fetch_but_keeps_gate_a_reject():
    cs = apply_licence_gate([cand(), cand(content="MIT"), cand(content="CC-BY-NC-4.0", gate="reject"), cand()])
    assert [c["gate_a_decision"] for c in cs] == ["defer", "fetch", "reject", "defer"]
    assert "Licence:" in cs[0]["gate_a_rationale"]


# ---- seed fallback frozen by hash and disclosed
def seed_doc(query="q"):
    return {"schema": "vera-seed-v1",
            "search_strategy": {"queries": [query], "tools": ["web"], "found_at": "2026-10-02",
                                "inclusion_rule": "r", "agent": "a", "model": "m"},
            "items": [{"url": f"https://s.org/{i}", "title": "t", "snippet": "s", "found_at": "2026-10-02",
                       "found_by": {"agent": "a", "model": "m"}, "reliability_tier": "A"} for i in range(3)]}


def write_seed(tmp_path, doc):
    f = tmp_path / "seed.json"
    f.write_text(json.dumps(doc))
    return f, hashlib.sha256(f.read_bytes()).hexdigest()


def test_seed_fallback_used_only_when_needed_and_disclosed(tmp_path):
    f, h = write_seed(tmp_path, seed_doc())
    a = FakeProvider("a", [ok("https://a.com/1")])
    out = run([a, SeedFileProvider(f, h)], min_candidates=2)
    assert out.meta["fallback_used"] and out.meta["seed_sha256"] == h and out[0]["provider"] == f"seed:{h[:8]}"
    assert out[0]["licence"]["content"]["id"] is None  # undeclared: the gate will defer it
    led = CostLedger()
    run([FakeProvider("a", [Resp(500)] * 3), SeedFileProvider(f, h)], ledger=led)
    assert led.entries[-1].cost_basis == "out_of_band_seed" and led.entries[-1].cost_known


def test_seed_changed_file_or_wrong_query_or_bad_schema_is_config_error(tmp_path):
    f, h = write_seed(tmp_path, seed_doc())
    f.write_text(json.dumps(seed_doc("other")))  # changed after freezing
    with pytest.raises(SearchConfigError, match="sha256"):
        run([SeedFileProvider(f, h)])
    f2, h2 = write_seed(tmp_path, seed_doc("different query"))
    with pytest.raises(SearchConfigError, match="query"):
        run([SeedFileProvider(f2, h2)])
    bad = seed_doc(); del bad["items"][0]["reliability_tier"]
    f3, h3 = write_seed(tmp_path, bad)
    with pytest.raises(SearchConfigError, match="reliability_tier"):
        run([SeedFileProvider(f3, h3)])
    with pytest.raises(SearchConfigError, match="unreadable"):
        run([SeedFileProvider(tmp_path / "missing.json", "0" * 64)])


def test_normalize_url():
    assert normalize_url("HTTPS://Ex.com/a/#x") == "https://ex.com/a"


# ---- no Tavily left in vera/
def test_no_tavily_reference_in_vera():
    word = "tav" + "ily"
    for py in (ROOT / "vera").rglob("*.py"):
        src = py.read_text()
        assert word not in src.lower(), py
        ast.parse(src)
    for rel in ("README.md", ".env.example", "config/vera_eval_run.json"):
        assert word not in (ROOT / rel).read_text().lower(), rel
