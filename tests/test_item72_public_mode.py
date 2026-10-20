"""Item #72 W2: public-mode access control, key/URL exposure sentinel tests, LLM05 output handling."""
import ast
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from vera import public_mode as pm
from vera.ui_safety import safe_markdown

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "streamlit_app.py")
SENTINEL = "SENTINEL-VERA-KEY-do-not-leak-0123456789"


class FakeResp:
    status_code = 200

    def json(self):
        return {"result_kind": "answer", "answer": "<script>alert(1)</script> ![x](http://evil/p.png) [a](javascript:x)",
                "tokens_used": 3, "cost_usd": 0.0001}


def _blob(at):
    parts = [repr(at.session_state.filtered_state)]
    for el in at.main:
        parts.append(repr(el.proto))
    for el in at.sidebar:
        parts.append(repr(el.proto))
    return "\n".join(parts)


@pytest.fixture
def sentinel_env(monkeypatch):
    monkeypatch.setenv("VERA_API_KEY", SENTINEL)
    monkeypatch.setenv("VERA_API_BASE_URL", "http://upstream.invalid:9")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)


@pytest.mark.parametrize("mode", [None, "1", "0"])
def test_sentinel_not_in_widgets_state_or_render(sentinel_env, monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    else:
        monkeypatch.setenv("VERA_PUBLIC_MODE", mode)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert SENTINEL not in _blob(at)
    if mode != "0":
        assert not at.sidebar.text_input  # no key widget, no URL widget in public mode
        assert "upstream.invalid" not in _blob(at)
    else:
        assert [w.value for w in at.sidebar.text_input][1] == ""  # key blank in dev mode


def test_request_helper_attaches_key_server_side_and_output_is_inert(sentinel_env, monkeypatch):
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, headers=headers)
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].set_value("hello")
    at.button[0].click().run()
    assert sent["headers"]["X-API-Key"] == SENTINEL
    assert sent["url"] == "http://upstream.invalid:9/ask"
    assert not at.sidebar.text_input
    shown = "\n".join(e.value for e in at.success)
    assert "<script>" not in shown and "&lt;script&gt;" in shown
    assert "![" not in shown and "javascript:" not in shown
    assert SENTINEL not in _blob(at).replace(repr(sent), "")


def test_dev_mode_never_sends_server_key(sentinel_env, monkeypatch):
    monkeypatch.setenv("VERA_PUBLIC_MODE", "0")
    sent = {}
    monkeypatch.setattr("requests.post", lambda url, json=None, headers=None, timeout=None:
                        sent.update(url=url, headers=headers) or FakeResp())
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.text_input[0].set_value("http://attacker.invalid")
    at.text_input[0].set_value("hello")
    at.button[0].click().run()
    assert not sent  # no key typed -> no request at all, server key not used
    at.sidebar.text_input[1].set_value("typed-key")
    at.button[0].click().run()
    assert sent["headers"]["X-API-Key"] == "typed-key" and sent["url"].startswith("http://attacker.invalid")


def test_public_mode_default_and_dev_values():
    assert pm.is_public_mode({}) and pm.is_public_mode({"VERA_PUBLIC_MODE": "1"})
    assert pm.is_public_mode({"VERA_PUBLIC_MODE": "garbage"})
    for v in ("0", "false", "off", "dev", "LOCAL"):
        assert not pm.is_public_mode({"VERA_PUBLIC_MODE": v})


def test_upstream_url_only_from_env_in_public():
    assert pm.upstream_base_url({"VERA_API_BASE_URL": "http://x:1/"}) == "http://x:1"
    assert pm.upstream_base_url({}) == pm.DEFAULT_LOCAL_URL


def test_no_unsafe_allow_html_on_model_text():
    allowed = {"vera/m7/demo_ui.py": 'html.escape(span_text)'}
    for f in [ROOT / "streamlit_app.py", *(ROOT / "pages").glob("*.py"), *(ROOT / "vera" / "m7").glob("*.py")]:
        for line in f.read_text().splitlines():
            if "unsafe_allow_html" in line and not line.lstrip().startswith("#"):
                rel = str(f.relative_to(ROOT))
                assert rel in allowed and allowed[rel] in line, f"{rel}: {line.strip()}"


def test_safe_markdown_neutralises_markup():
    out = safe_markdown("<img src=x onerror=1> ![a](http://e/p.png) [ok](https://a.b) [bad](javascript:x)")
    assert "<" not in out and "![" not in out and "javascript" not in out and "[ok](https://a.b)" in out


def test_demo_page_neutralises_markup_in_model_text():
    from vera.m7.demo_ui import render_demo
    from vera.m7.fixture import build_fixture_report
    rep = build_fixture_report()
    rep["engineered"]["response_text"] = "<script>alert(1)</script> ![x](http://evil/p.png)"

    def app(report):
        from vera.m7.demo_ui import render_demo as r
        r(report, "t")

    at = AppTest.from_function(app, kwargs={"report": rep}, default_timeout=30).run()
    assert not at.exception
    vals = "\n".join(m.value for m in at.markdown)
    assert "<script>" not in vals and "![x]" not in vals


# --- caps ------------------------------------------------------------------------------------

def _caps(**kw):
    return pm.Caps(**{**dict(req_per_min_per_client=2, req_per_day_per_client=3, req_per_day_global=100,
                             cost_usd_per_day_global=1.0, cost_usd_per_day_per_client=0.5), **kw})


def test_limiter_minute_day_cost_and_global():
    t = [1_000_000.0]
    lim = pm.CapLimiter(_caps(), clock=lambda: t[0])
    assert lim.allow("a") and lim.allow("a") and not lim.allow("a")
    assert lim.allow("b")
    t[0] += 61
    assert lim.allow("a") and not lim.allow("a")  # third of day used; day cap 3
    t[0] += 61
    assert not lim.allow("a")  # per-day cap
    lim.record_cost("b", 0.6)
    assert not lim.allow("b")  # per-client cost cap
    g = pm.CapLimiter(_caps(req_per_day_global=1), clock=lambda: t[0])
    assert g.allow("x") and not g.allow("y")
    t[0] += 86400
    assert lim.allow("a")  # new UTC day


def test_client_identity_proxy_handling():
    class R:
        client = type("C", (), {"host": "10.0.0.1"})()
        headers = {"x-forwarded-for": "6.6.6.6, 1.2.3.4"}
    assert pm.client_id(R, pm.Caps()) == "10.0.0.1"  # untrusted: header ignored
    assert pm.client_id(R, pm.Caps(trust_proxy_headers=True)) == "1.2.3.4"  # rightmost, not forgeable left


def test_ask_route_429_generic(monkeypatch):
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    import main
    pm.reset_limiter(pm.CapLimiter(_caps(req_per_min_per_client=1)))
    try:
        c = TestClient(main.app)
        body = {"question": "q", "mode": "force_bad"}
        h = {"X-API-Key": "k" * 20}
        assert c.post("/ask", json=body, headers=h).status_code == 500  # force_bad path, counted
        r = c.post("/ask", json=body, headers=h)
        assert r.status_code == 429
        assert r.json() == {"detail": pm.load_caps().rate_limit_detail}
        assert "openai" not in r.text.lower()
    finally:
        pm.reset_limiter(None)


def test_caps_config_loads():
    c = pm.load_caps()
    assert c.req_per_day_per_client > 0 and c.trust_proxy_headers is False


# ---- K5.2.I: no raw-JSON panel in public mode; dev mode keeps it -------------------------
MODEL_SENTINEL = "SENTINELMODEL7f3"
CHECKER_SENTINEL = "SENTINELCHECKER9c1"


class RichResp:
    status_code = 200

    def json(self):
        return {"result_kind": "answer", "answer": "Plain answer.", "model": MODEL_SENTINEL, "tokens_used": 12, "cost_usd": 0.0,
                "sources": [], "traced_sources": [], "recalled": [], "memory_written": 0,
                "claim_check": {"outcome": "checked", "checked": 1, "supported": 1, "partial": 0,
                                "removed": 0, "checker": CHECKER_SENTINEL}}


def _ask(monkeypatch, mode):
    monkeypatch.setenv("VERA_API_KEY", "k-test")
    monkeypatch.setenv("VERA_API_BASE_URL", "http://upstream.invalid:9")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr("requests.post", lambda *a, **k: RichResp())
    if mode is None:
        monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    else:
        monkeypatch.setenv("VERA_PUBLIC_MODE", mode)
    at = AppTest.from_file(APP, default_timeout=30).run()
    if mode == "0":
        at.sidebar.text_input[1].set_value("typed-key")
    at.text_input[0].set_value("hello")
    at.button[0].click().run()
    assert not at.exception
    return at


def test_ka1_public_run_has_no_json_panel_and_no_model_sentinel(monkeypatch):
    at = _ask(monkeypatch, None)
    assert at.success  # the answer rendered
    assert not at.json
    assert not any("Full Response" in e.label or "JSON" in e.label for e in at.expander)
    assert MODEL_SENTINEL not in _blob(at)


def test_ka1b_checker_label_not_public(monkeypatch):
    at = _ask(monkeypatch, None)
    assert CHECKER_SENTINEL not in _blob(at)


def test_ka2_dev_mode_still_shows_json_panel(monkeypatch):
    at = _ask(monkeypatch, "0")
    assert len(at.json) == 1
    assert MODEL_SENTINEL in at.json[0].value
    assert any("Full Response" in e.label for e in at.expander)


def _ancestors_ok(tree, call, pred):
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            ch._parent = node
    n = call
    while hasattr(n, "_parent"):
        n = n._parent
        if isinstance(n, ast.If) and pred(n.test) and any(call in ast.walk(b) for b in n.body):
            return True
    return False


def _is_not_public(test):
    return (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Name) and test.operand.id == "PUBLIC")


def test_ka3_every_st_json_call_is_inside_if_not_public():
    tree = ast.parse(Path(APP).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "json" and isinstance(n.func.value, ast.Name) and n.func.value.id == "st"]
    assert calls, "expected the dev-mode st.json call"
    for c in calls:
        assert _ancestors_ok(tree, c, _is_not_public), f"st.json at line {c.lineno} not under 'if not PUBLIC'"


def test_ka4_no_unread_full_response_stored_in_session_state():
    assert "last_ask_response" not in Path(APP).read_text()
