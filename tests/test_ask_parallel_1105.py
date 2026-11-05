"""ASK-PARALLEL-1105 (owner rule feedback_nonblocking_spawned_steps). OFFLINE: the /ask grounding fan-out calls its live
providers concurrently, and the DDG landing-page DOI lookups run concurrently, with results, ordering, ledger records
and portal records unchanged from the serial loop; one portal (or host) never sees overlapping calls."""
import threading
import time

import pytest
import requests

import vera.search_and_fetch as sf
import vera.search_providers as sp
from tests.test_m2_search import FakeProvider, ok
from vera.cost_ledger import CostLedger
from vera.search_providers import ProviderUnavailable


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real network transport called in a test")
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)


class _Overlap:
    def __init__(self):
        self.lock, self.now, self.peak, self.per = threading.Lock(), 0, 0, {}

    def enter(self, key):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
            k = self.per.setdefault(key, [0, 0])
            k[0] += 1
            k[1] = max(k[1], k[0])

    def leave(self, key):
        with self.lock:
            self.now -= 1
            self.per[key][0] -= 1


class SlowProvider(FakeProvider):
    def __init__(self, name, seq, delay, ov):
        super().__init__(name, seq)
        self.delay, self.ov = delay, ov

    def call(self, q, n, t):
        self.ov.enter(self.name)
        try:
            time.sleep(self.delay)
            return super().call(q, n, t)
        finally:
            self.ov.leave(self.name)


def _serial_reference(live, question, n, kw):
    """The pre-change serial loop, as a reference: (kind, value, ms) per provider in chain order."""
    out = []
    for p in live:
        t0 = time.monotonic()
        try:
            out.append(("ok", sf._attempts(p, question, n, **kw), 0))
        except sf._SCE as e:
            out.append(("sce", e, 0))
        except sf._ProviderFailed as f:
            out.append(("failed", f, 0))
        out[-1] = out[-1][:2] + ((time.monotonic() - t0) * 1000,)
    return out


def _chain(delays, ov):
    ddg, oa, doaj, ax = delays
    return [SlowProvider("duckduckgo_lite", [ok("https://doi.org/10.9/d1", "https://example.org/d2")], ddg, ov),
            SlowProvider("openalex", [ok("https://doi.org/10.9/a", "https://doi.org/10.9/d1")], oa, ov),
            SlowProvider("doaj", [ProviderUnavailable("doaj daily query cap (quota) spent")], doaj, ov),
            SlowProvider("arxiv", [ok(*[f"https://arxiv.org/abs/2401.0000{i}" for i in range(1, 4)])], ax, ov)]


def _run(delays, ov, ledger):
    return sf.search_question("q", 6, providers=_chain(delays, ov), sleep=lambda s: None, mode="fan_out",
                              max_share={}, ledger=ledger, max_retries=1)


def _items(res):
    return [{k: v for k, v in x.items() if k != "retrieved_at"} for x in res]  # wall-clock stamp only


def test_fan_out_providers_run_concurrently_and_results_match_serial(monkeypatch):
    delays = (0.30, 0.20, 0.10, 0.25)  # first in chain is slowest: workers finish in a different order than they start
    ov, led = _Overlap(), CostLedger()
    t0 = time.monotonic()
    res = _run(delays, ov, led)
    par_s = time.monotonic() - t0
    assert ov.peak >= 2, ov.peak  # providers overlapped
    assert par_s < 0.8 * sum(delays), par_s  # serial would be >= sum(delays) = 0.85 s
    assert all(v[1] == 1 for v in ov.per.values())  # each portal: one call in flight at a time

    monkeypatch.setattr(sf, "_call_live_concurrently", _serial_reference)
    ov2, led2 = _Overlap(), CostLedger()
    ref = _run(delays, ov2, led2)
    assert ov2.peak == 1
    assert _items(res) == _items(ref)  # same items, same order
    assert res.meta == ref.meta  # route_tried, portals (chain order), provider_counts, failures, notes
    strip = lambda L: [{k: v for k, v in e.items() if k != "at"} for e in L.as_dicts()]
    assert strip(led) == strip(led2)  # ledger records in chain order


def test_same_portal_providers_share_one_worker_in_chain_order():
    ov = _Overlap()
    a = SlowProvider("openalex", [ok("https://doi.org/10.9/a")], 0.15, ov)
    b = SlowProvider("openalex", [ok("https://doi.org/10.9/b")], 0.15, ov)
    order = []
    for p in (a, b):
        orig = p.call
        p.call = (lambda _o, _p: (lambda q, n, t: (order.append(id(_p)), _o(q, n, t))[1]))(orig, p)
    c = SlowProvider("arxiv", [ok("https://arxiv.org/abs/2401.00001")], 0.15, ov)
    sf.search_question("q", 5, providers=[a, c, b], sleep=lambda s: None, mode="fan_out", max_share={})
    assert ov.per["openalex"][1] == 1  # never two openalex calls in flight
    assert order == [id(a), id(b)]  # chain order within the portal


def test_fan_out_request_context_reaches_workers():
    import contextvars
    cv = contextvars.ContextVar("t1105", default=None)
    seen = []

    class P(FakeProvider):
        def call(self, q, n, t):
            seen.append(cv.get())
            return super().call(q, n, t)
    tok = cv.set("ctx")
    try:
        sf.search_question("q", 5, providers=[P("openalex", [ok("https://doi.org/10.9/a")]),
                                              P("arxiv", [ok("https://arxiv.org/abs/2401.00001")])],
                           sleep=lambda s: None, mode="fan_out", max_share={})
    finally:
        cv.reset(tok)
    assert seen == ["ctx", "ctx"]


def test_fan_out_unexpected_error_still_propagates():
    class Boom(FakeProvider):
        def call(self, q, n, t):
            raise KeyError("x")
    with pytest.raises(KeyError):
        sf.search_question("q", 5, providers=[Boom("openalex", []), FakeProvider("arxiv", [ok("https://a.b/c")])],
                           sleep=lambda s: None, mode="fan_out", max_share={})


# ----------------------------------------------------------------------------------------------- landing lookups
def _results(urls):
    return [{"url": u, "title": f"T{i}", "discovery_group": "portals"} for i, u in enumerate(urls)]


URLS = ["https://journals.example-a.org/article/1", "https://www.example-b.com/paper/2",
        "https://pubs.example-c.net/doi/3", "https://journals.example-a.org/article/4"]


def _fake_landing(ov, delay, calls):
    def fake(url, timeout_s, max_bytes, **kw):
        host = url.split("/")[2]
        ov.enter(host)
        try:
            calls.append(url)
            time.sleep(delay)
            return None if url.endswith("/2") else f"10.5555/{url[-1]}"
        finally:
            ov.leave(host)
    return fake


def _serial_landing(urls, lt, lb, **kw):
    return [sp.landing_doi(u, lt, lb, **kw) for u in urls], set()


def test_landing_lookups_run_concurrently_and_match_serial(monkeypatch):
    dcfg = {"landing_groups": ["portals"], "max_landing_fetches": 3, "landing_timeout_s": 5}
    ov, calls = _Overlap(), []
    monkeypatch.setattr(sp, "landing_doi", _fake_landing(ov, 0.25, calls))
    monkeypatch.setattr(sp, "_landing_group", lambda u: "host:" + u.split("/")[2])
    t0 = time.monotonic()
    out, counts = sp.ddg_to_candidates(_results(URLS), {}, dcfg, deadline=time.monotonic() + 60,
                                       clock=time.monotonic, calls_left=0)
    par_s = time.monotonic() - t0
    assert ov.peak >= 2 and par_s < 0.6, (ov.peak, par_s)  # serial: 3 x 0.25 s = 0.75 s
    assert sorted(calls) == sorted(URLS[:3])  # the same three landings (max_landing_fetches), the fourth not fetched

    ov2, calls2 = _Overlap(), []
    monkeypatch.setattr(sp, "landing_doi", _fake_landing(ov2, 0.0, calls2))
    monkeypatch.setattr(sp, "_landing_dois_concurrently", _serial_landing)
    ref, rcounts = sp.ddg_to_candidates(_results(URLS), {}, dcfg, deadline=time.monotonic() + 60,
                                        clock=time.monotonic, calls_left=0)
    assert out == ref and counts == rcounts  # candidates in result order, identical counts
    assert calls2 == URLS[:3]


def test_landing_same_host_never_overlaps(monkeypatch):
    dcfg = {"landing_groups": ["portals"], "max_landing_fetches": 4, "landing_timeout_s": 5}
    ov, calls = _Overlap(), []
    monkeypatch.setattr(sp, "landing_doi", _fake_landing(ov, 0.1, calls))
    monkeypatch.setattr(sp, "_landing_group", lambda u: "host:" + u.split("/")[2])
    sp.ddg_to_candidates(_results(URLS), {}, dcfg, deadline=time.monotonic() + 60, clock=time.monotonic, calls_left=0)
    assert ov.per["journals.example-a.org"][1] == 1  # the two example-a landings ran one after another
    assert [u for u in calls if "example-a" in u] == [URLS[0], URLS[3]]  # in result order


def test_landing_group_uses_rate_limit_portal():
    from vera import rate_limit
    a = rate_limit.get_registry().portal_for_url("https://api.openalex.org/works")
    assert a is not None and sp._landing_group("https://api.openalex.org/works") == "portal:" + a.name
    assert sp._landing_group("https://unmapped.example.invalid/p") == "host:unmapped.example.invalid"


def test_landing_same_group_rechecks_deadline_like_serial(monkeypatch):
    t = [0.0]

    def fake(url, timeout_s, max_bytes, **kw):
        t[0] += 6.0  # the first landing takes 6 s of the 10 s left; the second same-host one must not start (lt 5)
        return "10.5555/x"
    monkeypatch.setattr(sp, "landing_doi", fake)
    monkeypatch.setattr(sp, "_landing_group", lambda u: "host:same")
    urls = ["https://j.example.org/a/1", "https://j.example.org/a/2"]
    out, counts = sp.ddg_to_candidates(_results(urls), {}, {"landing_groups": ["portals"], "landing_timeout_s": 5},
                                       deadline=10.0, clock=lambda: t[0], calls_left=0)
    assert counts["ddg_landing_get"] == 1 and t[0] == 6.0
    assert [c.get("doi") for c in out] == ["10.5555/x", None]  # the skipped one stays a URL-only candidate


def test_fan_out_config_error_raises_and_ledger_records_every_call():
    class Bad(FakeProvider):
        def call(self, q, n, t):
            raise sf._SCE("bad config")
    led, good = CostLedger(), FakeProvider("arxiv", [ok("https://arxiv.org/abs/2401.00001")])
    with pytest.raises(sf.SearchConfigError):
        sf.search_question("q", 5, providers=[Bad("openalex", []), good], sleep=lambda s: None, mode="fan_out",
                           max_share={}, ledger=led)
    assert [e.provider for e in led.entries] == ["arxiv"]  # the call that was made is accounted for


def test_fan_out_rate_limit_deadline_reaches_workers():
    from vera import rate_limit
    seen = []

    class P(FakeProvider):
        def call(self, q, n, t):
            seen.append((rate_limit.DEADLINE.get(), rate_limit.QUEUE.get()))
            return super().call(q, n, t)
    with rate_limit.queue_scope(12345.0):
        sf.search_question("q", 5, providers=[P("openalex", [ok("https://doi.org/10.9/a")]),
                                              P("arxiv", [ok("https://arxiv.org/abs/2401.00001")])],
                           sleep=lambda s: None, mode="fan_out", max_share={})
    assert seen == [(12345.0, True)] * 2


# ------------------------------------------ ASK-PARALLEL-FIX-1105: per-host in-flight slot on every hop (S-K defect 1)
_PAGE = b'<html><head><meta name="citation_doi" content="10.1234/abc"></head></html>'


class _R:
    def __init__(self, status, headers=None, body=b""):
        self.status_code, self.headers, self._b = status, headers or {}, body

    def iter_content(self, n):
        yield self._b

    def close(self):
        pass


class _HostNet:
    """Fake transport counting in-flight requests per host (peak) and overall (peak); optional redirects."""
    def __init__(self, redirects=None, hold=0.2):
        self.redirects, self.hold = redirects or {}, hold
        self.lock, self.cur, self.peak, self.total, self.total_peak = threading.Lock(), {}, {}, 0, 0

    def __call__(self, url, headers=None, timeout=None, allow_redirects=False, stream=False, **kw):
        host = url.split("/")[2]
        with self.lock:
            self.cur[host] = self.cur.get(host, 0) + 1
            self.peak[host] = max(self.peak.get(host, 0), self.cur[host])
            self.total += 1
            self.total_peak = max(self.total_peak, self.total)
        try:
            time.sleep(self.hold)
            if url in self.redirects:
                return _R(302, {"Location": self.redirects[url]})
            return _R(200, {"Content-Type": "text/html"}, _PAGE)
        finally:
            with self.lock:
                self.cur[host] -= 1
                self.total -= 1


def _land(urls, net):
    return sp._landing_dois_concurrently(urls, 5.0, 100000, get=net, url_check=lambda u: True, deadline=None,
                                         clock=time.monotonic, acq={}, robots_ok=lambda u: True)


def test_fix_redirects_converging_on_one_unportaled_host_never_overlap():
    red = {"https://first-a.example/x": "https://publisher.example/p1",
           "https://first-b.example/y": "https://publisher.example/p2"}
    net = _HostNet(red)
    out, skipped = _land(list(red), net)
    assert net.peak["publisher.example"] == 1
    assert out == ["10.1234/abc", "10.1234/abc"] and not skipped  # both still resolved (serialised, not dropped)
    assert net.total_peak == 2  # the two different first hosts still ran at the same time


def test_fix_different_hosts_still_run_in_parallel():
    net = _HostNet(hold=0.3)
    t0 = time.monotonic()
    out, _ = _land([f"https://h{i}.example/p" for i in range(3)], net)
    assert out == ["10.1234/abc"] * 3 and net.total_peak == 3
    assert time.monotonic() - t0 < 0.8  # serial would take >= 0.9 s


def test_fix_host_slot_held_through_body_read_and_released():
    held = []

    class Slow(_R):
        def iter_content(self, n):
            with sp._host_slot("slot.example", 0.0) as free:  # another landing would find the host busy mid-read
                held.append(not free)
            yield self._b

    sp.landing_doi("https://slot.example/a", 5.0, 100000, get=lambda u, **k: Slow(200, {"Content-Type": "text/html"},
                                                                                  _PAGE),
                   url_check=lambda u: True, robots_ok=lambda u: True)
    assert held == [True]
    assert "slot.example" not in sp._HOST_SLOTS  # the refcounted entry is dropped once no landing uses it


def test_fix_busy_host_slot_skips_instead_of_overlapping():
    calls = []
    with sp._host_slot("busy.example", 1.0) as got:
        assert got
        t0 = time.monotonic()
        r = sp.landing_doi("https://busy.example/a", 0.6, 100000, get=lambda *a, **k: calls.append(1),
                           url_check=lambda u: True, robots_ok=lambda u: True)
    assert r is None and calls == [] and time.monotonic() - t0 < 1.0
