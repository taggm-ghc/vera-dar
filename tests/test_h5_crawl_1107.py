"""H5-CRAWL-1107 (owner OD-1107-HITL, 2026-11-07, H-5(a) approved): per-host Crawl-delay pacing on acquisition.

Owner, verbatim: "Crawl-delay: Approve H-5(a), with applicable source rate restrictions enforced. Permit compliant
inline acquisition and use background queues where pacing exceeds the request budget. Do not claim deferred
acquisition will resolve the current question without an implemented mechanism."

Covered: the flag ships true; concurrent workers serialize per host (slots booked under the lock); the spacing is the
stricter of robots Crawl-delay and the config/rate_limits.json rate; a slot past the budget is not booked and stops
with R-ACQ-CRAWL_DELAY (no request, not an attempt); inline acquisition fetches what fits and hands the rest to the
EXISTING best-effort background queue; the user-facing text claims background work only when it was really queued;
a server 429 is a rate limit, not an attempt. Fake clocks, fake sleeps and fake HTTP only: no network, no DB."""
from __future__ import annotations

import threading
from urllib import robotparser

import pytest

from tests import acq_support as A
from tests import fulltext_synth
from tests import test_fulldoc_e2e as e2e
from tests.test_acq_serve import h  # noqa: F401  (fixture)
from vera import acq_config, acq_discovered, auto_acquire, inline_acquire, rate_limit, research_record

EBI = "www.ebi.ac.uk"
JATS = f"https://{EBI}/europepmc/webservices/rest/PMC1/fullTextXML"
CD10 = "User-agent: *\nCrawl-delay: 10\n"
Q = e2e.Q_EPI


def seed_robots(host: str, text: str, now: float = 0.0) -> None:
    """Cache a PARSED robots.txt (with its Crawl-delay) and an empty tdmrep.json for host in the hop gate."""
    from vera import robots_ai
    p = robotparser.RobotFileParser()
    p.parse(text.splitlines())
    robots_ai.annotate(p, text)
    acq_discovered._ROBOTS[host] = (now, p)
    acq_discovered._TDMREP[host] = (now, [])


class Clock:
    """Thread-safe fake monotonic clock; sleep(s) moves it to max(now, caller's now + s) (parallel sleepers)."""

    def __init__(self, t: float = 100.0):
        self.t, self.lock = t, threading.Lock()

    def __call__(self):
        with self.lock:
            return self.t

    def sleep(self, s):
        with self.lock:
            self.t = max(self.t, self.t + max(0.0, s))


def _cfg(on=True):
    c = A.cfg()
    c["crawl_delay"] = {"enforce_on_acquisition": on}
    return c


@pytest.fixture
def reset(monkeypatch):
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    acq_discovered.reset_for_tests()
    monkeypatch.delenv("VERA_AUTOACQ_DISABLED", raising=False)
    monkeypatch.setattr(auto_acquire, "_worker_loop", lambda *a, **k: None)  # queued jobs are only inspected
    yield
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    acq_discovered.reset_for_tests()


# ------------------------------------------------------------------------------------------------ shipped config
def test_flag_ships_true():
    assert acq_config.load_config()["crawl_delay"]["enforce_on_acquisition"] is True


# ------------------------------------------------------------------------------------------------ serialization
def test_concurrent_workers_serialize_per_host_under_the_lock(reset, monkeypatch):
    """4 workers (inline.fetch_workers) hit the same host at the same instant: each books its own slot, 10 s apart;
    with the old read-then-sleep code all four computed the same wait and fired together."""
    seed_robots(EBI, CD10)
    clk = [100.0]
    waits, lk = [], threading.Lock()
    monkeypatch.setattr(acq_discovered, "_sleep", lambda s: (lk.acquire(), waits.append(s), lk.release()))
    acq_discovered._CRAWL_LAST[EBI] = 100.0  # a request was just sent
    go = threading.Barrier(4)

    def worker():
        go.wait()
        acq_discovered.crawl_wait(JATS, _cfg(), lambda: clk[0], None)

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(5)
    assert sorted(round(w, 6) for w in waits) == [10.0, 20.0, 30.0, 40.0]
    assert acq_discovered._CRAWL_LAST[EBI] == 140.0
    # another host is not delayed by this one
    seed_robots("arxiv.org", "User-agent: *\nCrawl-delay: 15\n")
    assert acq_discovered.crawl_wait("https://arxiv.org/abs/2101.00001", _cfg(), lambda: clk[0], None) == 0.0


class _GatedLast(dict):
    """_CRAWL_LAST stand-in: worker A's read of the host's last slot returns its value, then blocks until worker B
    has booked (or 1 s passes). With the lock held across read and write, B cannot book meanwhile, so A's wait times
    out and the two slots differ. Without the lock (the dropped-lock mutant), B books from the same stale value and
    both workers get the same slot. Deterministic: the interleaving is forced by events, not by timing luck."""

    def __init__(self, *a):
        super().__init__(*a)
        self.a_read, self.b_done, self.first = threading.Event(), threading.Event(), True

    def get(self, k, d=None):
        v = super().get(k, d)
        if self.first and threading.current_thread().name == "h5-A":
            self.first = False
            self.a_read.set()
            self.b_done.wait(1.0)
        return v


def test_read_and_book_are_one_locked_step_forced_interleaving(reset, monkeypatch):
    seed_robots(EBI, CD10)
    gated = _GatedLast({EBI: 100.0})
    monkeypatch.setattr(acq_discovered, "_CRAWL_LAST", gated)
    monkeypatch.setattr(acq_discovered, "_sleep", lambda s: None)
    got = {}

    def run(name):
        got[name] = acq_discovered.crawl_wait(JATS, _cfg(), lambda: 100.0, None)
        if name == "B":
            gated.b_done.set()

    a = threading.Thread(target=run, args=("A",), name="h5-A")
    a.start()
    assert gated.a_read.wait(5)
    b = threading.Thread(target=run, args=("B",), name="h5-B")
    b.start()
    a.join(5)
    b.join(5)
    assert sorted([got["A"], got["B"]]) == [10.0, 20.0]  # distinct slots; the mutant gives [10.0, 10.0]
    assert dict(gated)[EBI] == 120.0


def test_booked_slot_is_never_moved_earlier(reset, monkeypatch):
    monkeypatch.setattr(acq_discovered, "_sleep", lambda s: None)
    seed_robots(EBI, CD10)
    acq_discovered._CRAWL_LAST[EBI] = 150.0  # a future slot booked by another worker
    acq_discovered._ROBOTS.pop(EBI)  # robots not known: no spacing for this read, but the booking stays
    acq_discovered.crawl_wait(f"https://{EBI}/robots.txt", _cfg(), lambda: 100.0, None)
    assert acq_discovered._CRAWL_LAST[EBI] == 150.0


def test_flag_off_has_no_effect(reset, monkeypatch):
    waits = []
    monkeypatch.setattr(acq_discovered, "_sleep", waits.append)
    seed_robots(EBI, CD10)
    acq_discovered._CRAWL_LAST[EBI] = 100.0
    assert acq_discovered.crawl_wait(JATS, _cfg(False), lambda: 100.0, None) == 0.0 and waits == []


# ------------------------------------------------------------------------------------------------ stricter of two
def test_spacing_is_the_stricter_of_crawl_delay_and_the_configured_rate(reset):
    seed_robots(EBI, CD10)
    seed_robots("arxiv.org", "User-agent: *\nCrawl-delay: 1\n")
    rp = lambda host: acq_discovered._ROBOTS[host][1]  # noqa: E731
    assert acq_discovered.crawl_interval(JATS, rp(EBI)) == 10.0  # Crawl-delay 10 > 1 / 5 per s
    arx = acq_discovered.crawl_interval("https://arxiv.org/abs/x", rp("arxiv.org"))
    assert arx == pytest.approx(15.0, rel=1e-3)  # 1 / 0.066667 per s (rate_limits.json arxiv_web, ROUTES-1108) > Crawl-delay 1
    # from config, not hard-coded: a slower configured rate is honoured
    cfg = rate_limit.load_config()
    cfg["portals"]["europepmc"]["rate_per_s"] = 0.05
    rate_limit.set_registry_for_tests(rate_limit.Registry(cfg))
    assert acq_discovered.crawl_interval(JATS, rp(EBI)) == pytest.approx(20.0)
    # no Crawl-delay: 0 here (the portal's token bucket on deps.http_get paces it alone)
    assert acq_discovered.crawl_interval(JATS, True) == 0.0


# ------------------------------------------------------------------------------------------------ budget
def test_slot_past_the_budget_is_not_booked_and_is_not_an_attempt(reset, monkeypatch):
    monkeypatch.setattr(acq_discovered, "_sleep", lambda s: None)
    seed_robots(EBI, CD10)
    cfg, now = _cfg(), lambda: 100.0
    acq_discovered._CRAWL_LAST[EBI] = 100.0
    # deadline 125: slots 110 and 120 fit (min_call_s 1 left), 130 does not
    assert acq_discovered.crawl_wait(JATS, cfg, now, 125.0) == 10.0
    assert acq_discovered.crawl_wait(JATS, cfg, now, 125.0) == 20.0
    with pytest.raises(auto_acquire._Stop) as e:
        acq_discovered.crawl_wait(JATS, cfg, now, 125.0)
    assert e.value.code == "R-ACQ-CRAWL_DELAY" and acq_discovered._CRAWL_LAST[EBI] == 120.0  # nothing booked
    assert e.value.code not in research_record._FETCH_YES  # a wait is never an attempt
    assert inline_acquire._transient(cfg, e.value.code)  # background-eligible


def test_crawl_delay_stop_on_a_gate_read_is_not_cached_as_robots(reset, monkeypatch):
    monkeypatch.setattr(acq_discovered, "_sleep", lambda s: None)
    seed_robots(EBI, CD10, now=-10000.0)  # expired: the next request re-reads robots.txt, paced by the old delay
    acq_discovered._CRAWL_LAST[EBI] = 100.0
    web = A.Web({}, {"PMC1": b"<article/>"}, {"/robots.txt": A.Resp(body=CD10.encode(), ctype="text/plain")})
    with pytest.raises(auto_acquire._Stop) as e:
        auto_acquire._guarded_get(JATS, _cfg(), A.deps([], web), 100000, 5.0, False, clock=lambda: 100.0,
                                  deadline=105.0)
    assert e.value.code == "R-ACQ-CRAWL_DELAY" and web.urls == []
    assert acq_discovered._ROBOTS[EBI][1] not in (False,)  # never a fail-closed robots verdict


# ------------------------------------------------------------------------------------------------ 429
@pytest.mark.parametrize("fn", ["guarded", "b_get"])
def test_server_429_is_rate_limited_not_an_attempt_and_background_eligible(reset, fn):
    A.seed_access(EBI)
    web = A.Web({}, {}, {"fullTextXML": A.Resp(status=429)})
    cfg = _cfg()
    with pytest.raises(auto_acquire._Stop) as e:
        if fn == "guarded":
            auto_acquire._guarded_get(JATS, cfg, A.deps([], web), 100000, 5.0, False, clock=A.clock_at())
        else:
            acq_discovered._b_get(JATS, {EBI}, {"application/xml"}, 100000, 2, A.deps([], web), cfg, A.clock_at(),
                                  None, None)
    code = e.value.code
    assert code == "R-ACQ-RATELIMITED-THROTTLED" and acq_config.is_rate_limited(cfg, code)
    assert code not in research_record._FETCH_YES
    assert not research_record._fetch_made({"status": "fetched_rejected", "reason": code})
    assert inline_acquire.continue_eligible(cfg, {"outcome": "rejected", "reason": code})


# ------------------------------------------------------------------------------------------------ inline + queue
def _ebi_run(n, env, clk, monkeypatch):
    """n Europe PMC candidates, robots Crawl-delay 10 on www.ebi.ac.uk, 4 inline workers, fake clock and sleep."""
    monkeypatch.setattr(acq_discovered, "_sleep", clk.sleep)
    seed_robots(EBI, CD10, now=clk())
    dois = [f"10.9999/acq.h5c{i}" for i in range(n)]
    recs, jats = {}, {}
    for i, d in enumerate(dois):
        pmc = f"PMC{7100000 + i}"
        recs[d] = A.epmc_record(d, pmcid=pmc)
        jats[pmc] = fulltext_synth.jats(f"acq_{i}", "epi", d)
    web = A.Web(recs, jats)
    sent = []
    inner = web.__call__

    def get(url, **kw):
        sent.append(clk())
        return inner(url, **kw)
    cfg = A.cfg()
    cands = [A.cand(d) for d in dois]
    d = A.deps(cands, get)
    o = inline_acquire.acquire(Q, cfg, env=env, clock=clk, deps_factory=lambda a, e: d)
    return cfg, o, sent, dois


def test_inline_fetches_what_fits_and_queues_the_rest(reset, monkeypatch):
    clk = Clock(100.0)
    cfg, o, sent, dois = _ebi_run(8, dict(A.ENV_ON), clk, monkeypatch)
    ts = sorted(sent)
    assert ts and all(b - a >= 10.0 - 1e-9 for a, b in zip(ts, ts[1:]))  # never two requests inside the Crawl-delay
    deadline = 100.0 + float(cfg["inline"]["deadline_s"])
    assert all(t < deadline for t in ts)  # inline requests stay inside the request budget
    rec = {c["doi"]: c for c in o.candidates if c.get("doi") in dois}
    queued = [c for c in rec.values() if c["outcome"] == inline_acquire.QUEUED]
    done = [c for c in rec.values() if c["outcome"] != inline_acquire.QUEUED]
    assert queued and len(queued) + len(done) == len(dois)
    assert any(c["reason"] == "R-ACQ-CRAWL_DELAY" for c in queued)
    q = auto_acquire._S.queue
    in_queue = {auto_acquire._doi(c) for j in list(q.queue) for c in (j.cands or ())}
    assert {c["doi"] for c in queued} == in_queue  # handed to the EXISTING queue, nothing claimed beyond it
    note = inline_acquire.background_note(cfg, o)
    assert note and _honest(note)


def test_no_background_claim_when_nothing_was_queued(reset, monkeypatch):
    """Queue refuses (full): the candidates are not queued and no text claims a later acquisition."""
    monkeypatch.setattr(auto_acquire, "enqueue_candidates", lambda q, c, **k: {
        "status": "R-ACQ-QUEUE_FULL", "queued": [], "already": [], "refused": list(range(len(c)))})
    clk = Clock(100.0)
    cfg, o, _sent, dois = _ebi_run(8, dict(A.ENV_ON), clk, monkeypatch)
    recs = [c for c in (o.candidates or []) if c.get("doi") in dois]
    assert recs and any(c["reason"] == "R-ACQ-CRAWL_DELAY" for c in recs)  # pacing exceeded the budget
    assert not [c for c in recs if c["outcome"] == inline_acquire.QUEUED]
    assert o.queued == 0 and inline_acquire.background_note(cfg, o) == ""
    for k in ("inline_notice_none", "inline_notice_deadline", "inline_notice_unanswered", "inline_notice_search_failed"):
        assert _honest(cfg["scope_expansion"][k]) and "background" not in cfg["scope_expansion"][k].lower()


def _honest(t: str) -> bool:
    """No promise that deferred acquisition answers this question, and no count promise."""
    low = t.lower()
    banned = ("will answer", "will be answered", "will be added", "will update", "ask again in", "about ten minutes",
              "will find", "will collect", "asking again later", "ask again later")
    return not any(b in low for b in banned)


def test_background_texts_are_honest():
    cfg = acq_config.load_config()
    assert _honest(inline_acquire.BACKGROUND_NOTE) and "may" in inline_acquire.BACKGROUND_NOTE
    for k in ("inline_notice_background_persist", "inline_notice_background_persist_stopped"):
        t = cfg["scope_expansion"][k]
        assert _honest(t) and "this answer will not update" in t.lower()
        assert "not guaranteed" in t  # H5-CRAWL-1107 r3: honest, need-to-know wording
        assert not any(w in t.lower() for w in ("memory", "restart", "update,", "queue", "server"))
    assert _honest(inline_acquire.BACKGROUND_NOTE_PERSIST) and _honest(inline_acquire.BACKGROUND_NOTE_PERSIST_STOPPED)
    assert inline_acquire.BACKGROUND_NOTE_PERSIST == cfg["scope_expansion"]["inline_notice_background_persist"]
    assert (inline_acquire.BACKGROUND_NOTE_PERSIST_STOPPED
            == cfg["scope_expansion"]["inline_notice_background_persist_stopped"])
    se = cfg["scope_expansion"]
    assert "reading_list_notice_answering" not in se and "reading_list_notice_listing" not in se
    assert not hasattr(acq_config, "reading_list_notice")  # r2: dead text with no mechanism behind it, removed
