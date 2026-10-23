"""WAKE-1023: offline regression tests for the bounded UI API wake window (vera.ui_wake.ensure_awake) against the
SHIPPED config/ui_wake.json (health_timeout_s 10, max_total_s 160, backoff ending 11). Fake clock/sleep and fake HTTP
only: no sockets, no network, no files written. Success for API ready at 30/75/120 s; clean failure for 180/240 s and
never-ready within the ~3 minute promise; unreachable/incorrect URL; hard upper bound; question preserved."""
import itertools
import math
import random
from pathlib import Path

import pytest
import requests
from requests.adapters import HTTPAdapter
from requests.models import Response

from vera import public_mode, ui_wake

CFG = ui_wake.load_config(Path(__file__).resolve().parent.parent / "config" / "ui_wake.json")
MAXW = CFG.max_total_s
URL = "https://api.example.invalid"


def hard_bound(cfg) -> float:
    """Worst-case wall of one window for a client that honours its timeout: first probe, sleeps+probes until the
    deadline, plus one probe started just before it; also bounded by the finite schedule."""
    t = cfg.health_timeout_s
    return min(t + cfg.max_total_s + t, t * (len(cfg.backoff_s) + 1) + sum(cfg.backoff_s))


class FakeNet:
    """t0 = process start of the scenario. mode: 'hang' (unready probes burn the full timeout) or 'refuse' (fail fast)."""
    def __init__(self, clk, ready_at, slow_until=0.0, slow_latency=4.0, unready="hang"):
        self.clk, self.ready_at, self.slow_until, self.slow_latency, self.unready = clk, ready_at, slow_until, slow_latency, unready
        self.probes = []

    def get(self, url, timeout=None):
        t = self.clk.t - self.clk.t0
        if t < self.ready_at:
            if self.unready == "hang":
                self.clk.t += timeout
            self.probes.append((round(t, 1), "down"))
            raise requests.ConnectionError("down")
        lat = self.slow_latency if t < self.slow_until else 0.05
        if lat > timeout:
            self.clk.t += timeout
            self.probes.append((round(t, 1), "timeout"))
            raise requests.Timeout("slow")
        self.clk.t += lat
        self.probes.append((round(t, 1), "200"))
        return type("R", (), {"status_code": 200})()


class Clk:
    def __init__(self):
        self.t0 = self.t = 5000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Box:
    def update(self, **kw):
        self.kw = kw


class Status:
    def __init__(self, st): self.st = st
    def __enter__(self): self.st.statuses.append("open"); return Box()
    def __exit__(self, *a): return False


class ST:
    def __init__(self):
        self.session_state, self.statuses, self.errors, self.infos = {}, [], [], []

    def status(self, label, expanded=False): return Status(self)
    def error(self, m): self.errors.append(m)
    def info(self, m): self.infos.append(m)


@pytest.fixture
def env(monkeypatch):
    clk = Clk()
    monkeypatch.setattr(ui_wake, "clock", clk)
    monkeypatch.setattr(ui_wake, "sleep", clk.sleep)
    monkeypatch.setattr(ui_wake, "_in_script_run", lambda: True)
    ui_wake.reset_hook_cooldown()
    monkeypatch.delenv("VERA_API_RESTART_HOOK", raising=False)
    return clk


def run(clk, net, monkeypatch, reloads=1):
    """Each reload = one page_load_check on the same session (a failed window pops LAST_OK, so each reload re-probes).
    Returns (ok, wall_s_of_each_window, st)."""
    monkeypatch.setattr(ui_wake, "http_get", net.get)
    st, walls, ok = ST(), [], False
    for _ in range(reloads):
        s = clk.t
        ui_wake.page_load_check(st, "https://api.example.invalid", allow_restart_hook=False, cfg=CFG)
        walls.append(round(clk.t - s, 1))
        if ui_wake.LAST_OK_KEY in st.session_state:
            ok = True
            break
    return ok, walls, st


# (scenario, ready_at, unready mode, expected-success under current config)
@pytest.mark.parametrize("ready_at", [30, 75, 120])
@pytest.mark.parametrize("unready", ["hang", "refuse"])
def test_ready_at(env, monkeypatch, ready_at, unready):
    net = FakeNet(env, ready_at, unready=unready)
    ok, walls, st = run(env, net, monkeypatch)
    assert ok is True
    if not ok:
        assert st.errors == [CFG.failed_message]
    # window never exceeds config bound (+ one probe timeout of slack)
    assert walls[0] <= MAXW + CFG.health_timeout_s * 2 + 1


def test_h3_slow_health_first_40s_api_up(env, monkeypatch):
    """/health takes 4 s until t=40: the 10 s timeout sees the reply on the first probe."""
    net = FakeNet(env, ready_at=0, slow_until=40, slow_latency=4.0)
    ok, walls, st = run(env, net, monkeypatch)
    assert ok is True
    assert walls[0] == pytest.approx(4.0, abs=0.01) and len(net.probes) == 1


def test_h3_slow_health_until_70_succeeds(env, monkeypatch):
    """Slow (4 s) until 70 s: still within the 10 s per-attempt timeout, so the first probe succeeds."""
    net = FakeNet(env, ready_at=0, slow_until=70, slow_latency=4.0)
    ok, walls, st = run(env, net, monkeypatch)
    assert ok is True and len(net.probes) == 1


def test_never_ready_fails_cleanly(env, monkeypatch):
    for mode in ("hang", "refuse"):
        env.t0 = env.t
        net = FakeNet(env, ready_at=10**9, unready=mode)
        ok, walls, st = run(env, net, monkeypatch)
        assert ok is False
        assert st.errors == [CFG.failed_message]
        assert "kept above" in CFG.failed_message
        assert ui_wake.LAST_OK_KEY not in st.session_state
        assert st.infos == []  # no restart hook configured
        assert walls[0] <= MAXW + CFG.health_timeout_s * 2 + 1


def test_reload_restarts_window(env, monkeypatch):
    """API ready at 120 s: succeeds in the first window; a failed window would pop LAST_OK and reload would re-probe."""
    net = FakeNet(env, ready_at=120, unready="hang")
    ok, walls, st = run(env, net, monkeypatch, reloads=3)
    assert ok is True
    assert len(walls) == 1  # shipped window already covers 120 s


class TrackST:
    """Fake st that timestamps every st.* call: each one is a Streamlit yield point where a STOP/RERUN is honoured."""
    def __init__(self, clk):
        self.clk, self.session_state, self.calls, self.errors, self.infos, self.labels = clk, {}, [], [], [], []

    def _y(self, name):
        self.calls.append((name, round(self.clk.t, 3)))

    def status(self, label, expanded=False):
        st = self
        st._y("status"); st.labels.append(label)

        class _S:
            def __enter__(self_): return self_
            def __exit__(self_, *a): return False
            def update(self_, **kw): st._y("status.update"); st.labels.append(kw["label"])
        return _S()

    def error(self, m): self._y("error"); self.errors.append(m)
    def info(self, m): self._y("info"); self.infos.append(m)


@pytest.fixture
def clk(monkeypatch):
    c = Clk()
    monkeypatch.setattr(ui_wake, "clock", c)
    monkeypatch.setattr(ui_wake, "sleep", c.sleep)
    monkeypatch.setattr(ui_wake, "_in_script_run", lambda: True)
    ui_wake.reset_hook_cooldown()
    monkeypatch.delenv("VERA_API_RESTART_HOOK", raising=False)
    return c


def fake_get(clk, behaviour, probes, cost=0.0):
    """behaviour(url) -> int status or an exception instance. `cost` = seconds each attempt takes (capped by timeout)."""
    def get(url, timeout=None):
        clk.t += min(cost, timeout)
        r = behaviour(url)
        probes.append((round(clk.t - clk.t0, 1), url.split("/", 3)[-1] if "/" in url else url, getattr(r, "__name__", None)
                       or (type(r).__name__ if isinstance(r, Exception) else r)))
        if isinstance(r, Exception):
            raise r
        return type("R", (), {"status_code": r})()
    return get


def window(clk, st, base):
    s = clk.t
    ui_wake.page_load_check(st, base, allow_restart_hook=False, cfg=CFG)
    return round(clk.t - s, 1)


def expect_clean_fail(st, wall, probes):
    assert ui_wake.LAST_OK_KEY not in st.session_state
    assert st.errors == [CFG.failed_message]
    assert st.labels == [CFG.starting_message.format(max_total_s=CFG.max_total_s), CFG.failed_message]
    assert wall <= hard_bound(CFG) + 1e-6


# --- unreachable / incorrect URL cases (each fails after the FULL window, same message as a cold start) ----------
CASES = {
    # name: (base_url, behaviour, per-attempt cost seconds)
    "dns_failure_fast": (URL, lambda u: requests.ConnectionError("Name or service not known"), 0.0),
    "dns_failure_slow_resolver_5s": (URL, lambda u: requests.ConnectionError("Temporary failure in name resolution"), 5.0),
    "connection_refused": (URL, lambda u: requests.ConnectionError("Connection refused"), 0.0),
    "connect_timeout_blackhole": (URL, lambda u: requests.ConnectTimeout("timed out"), 1e9),
    "wrong_path_404": (URL, lambda u: 404, 0.05),
    "render_router_502": (URL, lambda u: 502, 0.05),
    "render_router_503": (URL, lambda u: 503, 0.05),
    "redirect_loop": (URL, lambda u: requests.TooManyRedirects("Exceeded 30 redirects"), 0.05),
}


@pytest.mark.parametrize("name", list(CASES))
def test_unreachable_fails_after_full_window(clk, monkeypatch, name):
    base, beh, cost = CASES[name]
    probes = []
    monkeypatch.setattr(ui_wake, "http_get", fake_get(clk, beh, probes, cost))
    st = TrackST(clk)
    wall = window(clk, st, base)
    expect_clean_fail(st, wall, probes)
    assert len(probes) == len(CFG.backoff_s) + 1 or wall >= CFG.max_total_s  # whole schedule used: no fast-fail path


def test_unset_base_url_probes_localhost_and_fails(clk, monkeypatch):
    """VERA_API_BASE_URL unset on the UI service -> DEFAULT_LOCAL_URL (127.0.0.1:8001); nothing listens on Render."""
    base = public_mode.upstream_base_url(env={}, local_url_file=None)
    assert base == public_mode.DEFAULT_LOCAL_URL == "http://127.0.0.1:8001"
    probes = []
    monkeypatch.setattr(ui_wake, "http_get", fake_get(clk, lambda u: requests.ConnectionError("Connection refused"), probes))
    st = TrackST(clk)
    wall = window(clk, st, base)
    expect_clean_fail(st, wall, probes)
    assert all(p[1] == "health" for p in probes)  # probes go to 127.0.0.1:8001/health; host never rendered
    assert not any("127.0.0.1" in x for x in st.labels + st.errors)


def test_no_scheme_url_never_probes_but_still_waits(clk, monkeypatch):
    """A base URL without http(s):// is rejected before any request, yet the full sleep schedule still runs."""
    probes = []
    monkeypatch.setattr(ui_wake, "http_get", fake_get(clk, lambda u: 200, probes))
    st = TrackST(clk)
    wall = window(clk, st, "api.example.invalid")
    expect_clean_fail(st, wall, probes)
    assert probes == []
    assert wall == pytest.approx(min(sum(CFG.backoff_s), CFG.max_total_s), abs=0.01)


# --- redirects through the real requests stack (HTTPAdapter.send replaced; no socket is opened) ----------------
def _resp(req, code, location=None):
    r = Response()
    r.status_code, r.url, r.request, r._content = code, req.url, req, b"{}"
    r.headers["Content-Type"] = "application/json"
    if location:
        r.headers["Location"] = location
    return r


@pytest.mark.parametrize("name,route,healthy", [
    ("redirect_301_to_health_200", {"/health": (301, "https://api.example.invalid/v2/health"), "/v2/health": (200, None)}, True),
    ("redirect_302_to_login_200", {"/health": (302, "https://login.example.invalid/signin"), "/signin": (200, None)}, True),
    ("redirect_to_404", {"/health": (308, "https://api.example.invalid/x"), "/x": (404, None)}, False),
])
def test_redirects_real_requests(clk, monkeypatch, name, route, healthy):
    seen = []

    def send(self, request, **kw):
        path = request.url.split(".invalid", 1)[1]
        seen.append(path)
        code, loc = route[path]
        return _resp(request, code, loc)

    monkeypatch.setattr(HTTPAdapter, "send", send)
    monkeypatch.setattr(ui_wake, "http_get", requests.get)  # the production seam value
    st = TrackST(clk)
    wall = window(clk, st, URL)
    assert (ui_wake.LAST_OK_KEY in st.session_state) is healthy
    if healthy:
        # FINDING: any redirect chain ending in 200 counts as healthy; the body is not checked (a login/parking page
        # on the wrong host would mask H4). Recorded as a residual risk, not changed (no code changes authorised).
        assert wall == 0 and st.calls == []
    else:
        expect_clean_fail(st, wall, seen)


# --- hard upper bound: no config or adversary yields an unbounded wait ------------------------------------------
def test_config_is_finite():
    assert math.isfinite(CFG.max_total_s) and 0 < CFG.max_total_s <= 300
    assert math.isfinite(CFG.health_timeout_s) and 0 < CFG.health_timeout_s <= 30
    assert 0 < len(CFG.backoff_s) <= 20 and all(math.isfinite(b) and b > 0 for b in CFG.backoff_s)
    # owner: no indefinite wait. Bound = first probe + deadline + one last probe; overshoot past max_total_s <= 2 x timeout
    assert hard_bound(CFG) <= CFG.max_total_s + 2 * CFG.health_timeout_s <= 240


def test_hard_bound_holds_against_random_adversary(clk, monkeypatch):
    rng = random.Random(1023)
    worst = 0.0
    for _ in range(400):
        clk.t0 = clk.t
        costs = itertools.cycle([rng.choice([0, 0.01, 1, 2.9, 3.1, 9.9, 10.1, 1e9]) for _ in range(5)])
        probes = []
        monkeypatch.setattr(ui_wake, "http_get", fake_get(clk, lambda u: requests.Timeout("t"), probes, next(costs)))

        def get(url, timeout=None, _c=costs):
            clk.t += min(next(_c), timeout)
            probes.append(1)
            raise requests.Timeout("t")
        monkeypatch.setattr(ui_wake, "http_get", get)
        st = TrackST(clk)
        wall = window(clk, st, URL)
        worst = max(worst, wall)
        assert wall <= hard_bound(CFG) + 1e-6


# --- cancellation: Streamlit honours STOP/RERUN only at st.* calls (script_runner._maybe_handle_execution_control_
# request). Inside the wake loop there is none between the opening status and the final update, so a click/rerun in
# the same session waits for the whole window, and a browser reload (new session) leaves the old window running. ---
def test_no_yield_point_inside_window(clk, monkeypatch):
    probes = []
    monkeypatch.setattr(ui_wake, "http_get", fake_get(clk, lambda u: requests.ConnectionError("down"), probes))
    st = TrackST(clk)
    start = clk.t
    wall = window(clk, st, URL)
    names = [n for n, _ in st.calls]
    assert names == ["status", "status.update", "error"]
    gap = st.calls[1][1] - st.calls[0][1]  # seconds during which a STOP/RERUN cannot take effect
    assert gap >= min(sum(CFG.backoff_s), CFG.max_total_s) - 1e-6


# --- R2 (owner 2026-10-23): startup sweep 30/75/120/180/240 s, slow /health > per-attempt timeout, permanent failure ---


@pytest.mark.parametrize("ready_at", [30, 75, 120, 180, 240, 10**9])
@pytest.mark.parametrize("unready", ["hang", "refuse"])
def test_startup_sweep(clk, monkeypatch, ready_at, unready):
    clk.t0 = clk.t
    net = FakeNet(clk, ready_at, unready=unready)
    monkeypatch.setattr(ui_wake, "http_get", net.get)
    st = TrackST(clk)
    wall = window(clk, st, URL)
    ok = ui_wake.LAST_OK_KEY in st.session_state
    assert wall <= hard_bound(CFG) + 1e-6
    if ready_at == 10**9:
        assert not ok
    else:
        assert ok is (ready_at <= 120)  # shipped p160strict: 30/75/120 s succeed, 180/240 s fail cleanly
    if not ok:
        expect_clean_fail(st, wall, net.probes)
        assert all(t < ready_at for t, r in net.probes)  # failure only because the window ended before readiness
    elif wall > 0:
        assert st.labels == [CFG.starting_message.format(max_total_s=CFG.max_total_s), CFG.ready_message]


@pytest.mark.parametrize("slow_until", [40, 100, 170])
def test_slow_health_exceeds_attempt_timeout(clk, monkeypatch, slow_until):
    """Process up from t=0, but /health replies take timeout+1 s until slow_until, then 0.05 s."""
    clk.t0 = clk.t
    net = FakeNet(clk, ready_at=0, slow_until=slow_until, slow_latency=CFG.health_timeout_s + 1)
    monkeypatch.setattr(ui_wake, "http_get", net.get)
    st = TrackST(clk)
    wall = window(clk, st, URL)
    ok = ui_wake.LAST_OK_KEY in st.session_state
    assert wall <= hard_bound(CFG) + 1e-6
    assert ok is any(r == "200" for _, r in net.probes)
    if not ok:
        expect_clean_fail(st, wall, net.probes)


def test_question_preserved_on_failed_window(clk, monkeypatch):
    """take_ready_question drops only the internal queue entry; the typed text lives in the text_input widget state,
    which ui_wake never touches, and the failure message says it is kept."""
    net = FakeNet(clk, ready_at=10**9, unready="refuse")
    monkeypatch.setattr(ui_wake, "http_get", net.get)
    st = TrackST(clk)
    st.session_state["$$WIDGET_question"] = "Why do many scientific studies fail to replicate?"
    ui_wake.queue_question(st.session_state, {"question": "Why do many scientific studies fail to replicate?"})
    assert ui_wake.take_ready_question(st, URL, allow_restart_hook=False, cfg=CFG) is None
    assert ui_wake.PENDING_KEY not in st.session_state
    assert st.session_state["$$WIDGET_question"] == "Why do many scientific studies fail to replicate?"
    assert "kept above" in CFG.failed_message and st.errors == [CFG.failed_message]
    # success path returns the queued payload once healthy
    clk.t0 = clk.t
    net2 = FakeNet(clk, ready_at=0)
    monkeypatch.setattr(ui_wake, "http_get", net2.get)
    ui_wake.queue_question(st.session_state, {"question": "q2"})
    assert ui_wake.take_ready_question(st, URL, allow_restart_hook=False, cfg=CFG) == {"question": "q2"}


def test_starting_message_tracks_max_total():
    msg = CFG.starting_message.format(max_total_s=CFG.max_total_s)
    assert "~3 minutes" in msg and hard_bound(CFG) <= 180  # literal promise covers the hard wall bound
