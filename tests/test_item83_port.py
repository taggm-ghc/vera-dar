"""Item #83: connection triage and corpus description ported from AI-Internship. Offline only."""
import requests
import pytest

from vera import api_triage as at, corpus_description as cd

HOST = "sentinel-host-do-not-show.example"
BASE = f"https://{HOST}"


class Resp:
    def __init__(self, code, body=None):
        self.status_code, self._body = code, body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def _raise(exc):
    def f(url):
        raise exc
    return f


def _routes(health, target=None):
    return lambda url: health if url.endswith("/health") else target


@pytest.mark.parametrize("base,get,earliest", [
    ("", None, "1. API address configured"),
    (BASE, _raise(requests.ConnectionError(f"connect failed to {BASE}")), "2. API reachable"),
    (BASE, _raise(requests.Timeout(f"timed out {BASE}")), "3. API awake"),
    (BASE, _routes(Resp(429)), "3. API awake"),
    (BASE, _routes(Resp(502)), "3. API awake"),
    (BASE, _routes(Resp(500, {"status": "bad"})), "4. API healthy"),
    (BASE, _routes(Resp(200, {"status": "ok"}), Resp(404, {})), "5. /ask reachable"),
])
def test_triage_earliest_failure_and_no_host(base, get, earliest):
    steps = at.triage_api(base, "/ask", http_get=get)
    assert at.earliest_failure(steps)["step"] == earliest
    assert HOST not in " ".join(s["detail"] for s in steps)


def test_triage_caps_429_is_not_a_wake_bounce():
    steps = at.triage_api(BASE, "/ask", http_get=_routes(Resp(200, {"status": "ok"}), Resp(405, {"detail": "x"})))
    assert at.earliest_failure(steps) is None  # GET on POST-only /ask answers 405: the endpoint exists


def test_ask_page_shows_triage_on_failures():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "streamlit_app.py").read_text()
    # PROG-E F2 / item #83: flag-on public path _render_terminal adds the third call site (was == 2).
    assert src.count("_show_triage(api_base_url)") == 3


CFG = {"enabled": True, "read_db_url_env": "VERA_DB_URL_RO", "prompt_version": "v1", "max_titles": 5, "retry_s": 300}


class Describer(cd.CorpusDescriber):
    """Stamp and profile injected (no DB)."""
    def __init__(self, overview, **kw):
        super().__init__(CFG, env={"VERA_DB_URL_RO": "x"}, engine_factory=lambda *a, **k: object(), **kw)
        self.overview = overview

    def current(self, generate):
        import vera.corpus_description as mod
        orig = mod.stamp_and_profile
        mod.stamp_and_profile = lambda e, n: (dict(self.overview), {"titles": ["T"], "published_years": ["2024", "2026"]})
        try:
            return super().current(generate)
        finally:
            mod.stamp_and_profile = orig


def test_regenerates_only_when_the_corpus_changes():
    calls = []
    gen = lambda m: calls.append(1) or (f"summary {len(calls)}", ["topic"], "groq:m")
    d = Describer({"source_count": 2, "corpus_updated_at": "A"})
    assert d.current(gen)["summary"] == "summary 1" and d.current(gen)["summary"] == "summary 1"
    d.overview["corpus_updated_at"] = "B"
    assert d.current(gen)["summary"] == "summary 2" and len(calls) == 2


def test_empty_corpus_needs_no_model_and_says_so():
    d = Describer({"source_count": 0, "corpus_updated_at": None})
    out = d.current(lambda m: (_ for _ in ()).throw(AssertionError("called")))
    assert out["summary_source"] == "deterministic_fallback" and "no stored sources yet" in out["summary"]


def test_model_failure_falls_back_and_cools_down():
    t = [0.0]
    d = Describer({"source_count": 3, "corpus_updated_at": "A"}, clock=lambda: t[0])
    calls = []

    def boom(m):
        calls.append(1)
        raise RuntimeError("down")
    assert d.current(boom)["summary_source"] == "deterministic_fallback"
    d.current(boom)
    assert len(calls) == 1  # cooldown
    t[0] = 301.0
    d.current(boom)
    assert len(calls) == 2


def test_not_connected_without_ro_url():
    out = cd.CorpusDescriber(CFG, env={}).current(lambda m: ("x", [], "m"))
    assert out["summary_source"] == "not_connected"


def test_sanitise_plain_text():
    assert cd.sanitise("a [b](https://x) ![i](https://y) <i>c</i> https://z d") == "a b c d"
