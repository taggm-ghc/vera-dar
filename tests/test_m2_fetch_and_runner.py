import hashlib

import pytest

from vera.cost_ledger import CostLedger
from vera.gate_a import GateAConfig
from vera.m2_runner import run_m2
from vera.selective_fetch import FetchResult, extract_text, fetch_candidate


class R:
    def __init__(self, status=200, body=b"", ctype="text/html; charset=utf-8", headers=None):
        self.status_code, self._b = status, body
        self.headers = {"Content-Type": ctype, **(headers or {})}
        self.encoding = "utf-8"

    def iter_content(self, n):
        for i in range(0, len(self._b), n):
            yield self._b[i:i + n]


OK = lambda u: (True, "")


def test_fetch_hash_and_text_and_provenance():
    body = b"<html><script>evil()</script><body><p>Hello</p><div>World</div></body></html>"
    fr = fetch_candidate("https://a.com", http_get=lambda *a, **k: R(body=body), url_check=OK)
    assert fr.ok and fr.content_text == "Hello\nWorld"
    assert fr.content_hash == hashlib.sha256(fr.content_text.encode()).hexdigest()
    assert fr.provenance["raw_sha256"] == hashlib.sha256(body).hexdigest()
    assert fr.provenance["final_url"] == "https://a.com" and fr.provenance["bytes"] == len(body)


def test_fetch_failures_are_results_not_exceptions():
    cases = [(R(404), "http 404"), (R(body=b"%PDF", ctype="application/pdf"), "unsupported"),
             (R(body=b"<html><script>x</script></html>"), "no extractable"),
             (R(body=b"a" * 2_000_001, ctype="text/plain"), "exceeds")]
    for resp, msg in cases:
        fr = fetch_candidate("https://a.com", http_get=lambda *a, **k: resp, url_check=OK)
        assert not fr.ok and msg in fr.error


def test_ssrf_block_and_redirect_recheck():
    blocked = lambda u: (False, "private") if "internal" in u else (True, "")
    assert "blocked" in fetch_candidate("http://internal/x", http_get=None, url_check=blocked).error
    redirect = R(302, headers={"Location": "http://internal/admin"})
    fr = fetch_candidate("https://a.com", http_get=lambda *a, **k: redirect, url_check=blocked)
    assert not fr.ok and "blocked" in fr.error


def test_redirect_loop_bounded():
    loop = R(302, headers={"Location": "https://a.com/again"})
    fr = fetch_candidate("https://a.com", http_get=lambda *a, **k: loop, url_check=OK)
    assert "too many redirects" in fr.error


def test_extract_text_plain():
    assert extract_text("a  b\n\n  c ", "text/plain") == "a b\nc"


class MemStore:
    """Mirrors PostgresStore semantics: canonical URL + content hash -> reuse, else version+1."""

    def __init__(self):
        self.cands, self.sources, self.iters = {}, {}, []

    def existing_urls(self, run_id):
        from vera.search_and_fetch import normalize_url
        return {normalize_url(c["url"]) for c in self.cands.values()}

    def start_iteration(self, *a):
        self.iters.append({"args": a})
        return len(self.iters)

    def insert_candidate(self, run_id, it, c):
        cid = len(self.cands) + 1
        self.cands[cid] = {**c, "fetch_status": "pending"}
        return cid

    def register_source(self, candidate_id, fr):
        key = (fr.url, fr.content_hash)
        if key in self.sources:
            sid, ver, _ = self.sources[key]
            return sid, ver, False
        ver = 1 + sum(1 for k in self.sources if k[0] == fr.url)
        self.sources[key] = (len(self.sources) + 1, ver, True)
        return self.sources[key]

    def update_candidate(self, cid, fetch_status, error=None):
        self.cands[cid].update(fetch_status=fetch_status, fetch_error=error)

    def finish_iteration(self, it, results_count):
        self.iters[it - 1]["results_count"] = results_count


def fake_search(q, n, ledger=None):
    ledger.record(kind="search", provider="arxiv", cost_usd=0.008)
    return [{"url": f"https://s{i}.com", "title": "t", "snippet": "s", "rank": i + 1, "provider": "arxiv",
             "licence": {"metadata": {"id": "CC0-1.0"}, "content": {"id": "CC-BY-4.0"}}} for i in range(4)]


def fake_score(question, cs, cfg=None, ledger=None):
    ledger.record(kind="llm_gate_a", provider="x", cost_usd=0.001)
    decs = ["fetch", "fetch", "defer", "reject"]
    return [{**c, "gate_a_score": 0.9 - i * .2, "gate_a_decision": decs[i], "gate_a_rationale": "r",
             "gate_a_factors": {}} for i, c in enumerate(cs)]


def fake_fetch(url, ledger=None):
    if url.endswith("s1.com"):
        return FetchResult(url=url, ok=False, error="http 500")
    return FetchResult(url=url, ok=True, content_text="body", content_hash="h-" + url, provenance={})


def test_runner_end_to_end_with_fakes():
    store, led = MemStore(), CostLedger()
    res = run_m2("q", 7, store, search_fn=fake_search, score_fn=fake_score, fetch_fn=fake_fetch, ledger=led)
    assert [c["url"] for c in res.admitted] == ["https://s0.com"]
    assert len(res.fetch_failed) == 1 and len(res.deferred) == 1 and len(res.rejected) == 1
    st = {c["url"]: (c["gate_a_decision"], c["fetch_status"]) for c in store.cands.values()}
    assert st == {"https://s0.com": ("fetch", "fetched"), "https://s1.com": ("fetch", "failed"),
                  "https://s2.com": ("defer", "pending"), "https://s3.com": ("reject", "pending")}
    assert store.iters[0]["results_count"] == 4 and res.cost_usd == pytest.approx(0.009) and len(res.cost_calls) == 2
    assert res.admitted[0]["source_version"] == 1


def test_research_skips_known_urls_and_versions_changes():
    store = MemStore()
    run_m2("q", 7, store, search_fn=fake_search, score_fn=fake_score, fetch_fn=fake_fetch)
    res2 = run_m2("q", 7, store, iteration=2, search_fn=fake_search, score_fn=fake_score, fetch_fn=fake_fetch)
    assert res2.admitted == [] and store.iters[1]["results_count"] == 4
    # same URL, new hash -> version 2
    a = FetchResult(url="u", ok=True, content_hash="h1"); b = FetchResult(url="u", ok=True, content_hash="h2")
    assert store.register_source(1, a)[1] == 1 and store.register_source(1, b)[1] == 2 and store.register_source(1, a)[1] == 1


def test_search_failure_closes_iteration_and_raises():
    store = MemStore()
    def bad(q, n, ledger=None):
        raise RuntimeError("search down")
    with pytest.raises(RuntimeError):
        run_m2("q", 7, store, search_fn=bad, score_fn=fake_score, fetch_fn=fake_fetch)
    assert "results_count" not in store.iters[0]  # NULL = failed, distinct from 0 = nothing found


def test_deadline_demotes_fetches_to_deferred():
    store = MemStore()
    t = iter([0, 999] + [999] * 50)
    res = run_m2("q", 7, store, search_fn=fake_search, score_fn=fake_score, fetch_fn=fake_fetch,
                 deadline_s=10, clock=lambda: next(t))
    assert res.admitted == [] and len(res.deferred) == 3
    assert "deadline" in res.deferred[0]["gate_a_rationale"]


def test_cost_log_written_when_env_set(tmp_path, monkeypatch):
    import json
    log = tmp_path / "cost.jsonl"
    monkeypatch.setenv("VERA_COST_LOG", str(log))
    run_m2("q", 7, MemStore(), search_fn=fake_search, score_fn=fake_score, fetch_fn=fake_fetch)
    line = json.loads(log.read_text())
    assert line["stage"] == "m2" and line["total_usd"] == pytest.approx(0.009) and len(line["calls"]) == 2


# ---- pacing of arXiv abstract-page fetches (R1 approved the 3 s limit; applies to page fetches too)
def test_host_pacer_spaces_same_host_fetches_and_ignores_other_hosts():
    from vera.selective_fetch import HostPacer
    t = {"now": 100.0}
    slept = []

    def sleep(s):
        slept.append(s)
        t["now"] += s

    p = HostPacer({"arxiv.org": 3.0}, clock=lambda: t["now"], sleep=sleep)
    p.wait("https://arxiv.org/abs/2401.00001")   # first call: no wait
    t["now"] += 1.0
    p.wait("https://arxiv.org/abs/2401.00002")   # 1 s later: must wait the remaining 2 s
    p.wait("https://example.org/page")           # unpaced host: no wait
    assert slept == [2.0]


def test_paced_host_interval_matches_run_config():
    import json
    from pathlib import Path
    from vera.selective_fetch import PACED_HOSTS
    cfg = json.loads((Path(__file__).resolve().parent.parent / "config" / "vera_eval_run.json").read_text())
    assert PACED_HOSTS["arxiv.org"] == cfg["search"]["arxiv"]["min_interval_s"]
