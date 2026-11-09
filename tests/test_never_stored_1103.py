"""NONASK-1103 I-SL-B: R-037 never-stored sentinel suite for the Official and Current-reporting lanes (plan 6.8, 6.8a).

Offline: stub fetch / http_get / model / checker; no network, no DB, no env files. Two sentinels are planted in a page
whose visible text is question-relevant: IN sits inside the admitted chunk (may reach ONLY the user answer quote and
citation quotes), OUT sits in chunks admission does not take (may reach NOWHERE). Sinks: DB/store writers, logs, trace,
contract, research record / public outcome, background queue, caches, memory, claim records, exceptions, reprs."""
import builtins
import functools
import gc
import json
import logging
import re
import socket
import sys
from types import SimpleNamespace

import pytest

from tests import leak_sentinel as ls
from vera import (ask_service, auto_acquire, evidence_trace, fulldoc_answer, fulldoc_corpus, inline_acquire,
                  lane_evidence, live_stats, memory_store, official_lane, public_fetch, reporting_lane)
from vera.official_lane import LanePage

Q = "What are the current Google Cloud Run limits and how does it compare with AWS Lambda?"
GCFG = {"removed_citation_note": "Note: bad citations removed.",
        "claim_check": {"enabled": True, "timeout_s": 5, "max_claims": 10,
                        "checkers": [{"provider": "t", "model": "m", "api_key_env": "FD_TEST_KEY", "base_url": None}]}}
OFFICIAL_URL = "https://docs.cloud.google.com/docs"
NEWS_URL = "https://news.example.org/story-1"
FILLER = ("Unrelated gardening notes about tomatoes and compost heaps, nothing about the question at all, just filler "
          "words repeated for length. ")


def page_text(inn, out, extra=""):
    relevant = (f"Google Cloud Run limits: the current request timeout limits for Google Cloud Run services are "
                f"documented here, and Cloud Run compares with AWS Lambda on concurrency limits ({inn}). {extra}")
    return relevant + " " + FILLER * 14 + f" Archive footer {out}. " + FILLER * 3


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("FD_TEST_KEY", "x")
    inn, out = ls.new("LANEIN"), ls.new("LANEOUT")
    return SimpleNamespace(inn=inn, out=out)


class Rec:
    """Records every call to a patched writer; nothing is ever written."""
    def __init__(self):
        self.calls = []

    def hook(self, label):
        def f(*a, **k):
            self.calls.append((label, a, k))
            return None
        return f


@pytest.fixture
def sinks(monkeypatch):
    rec = Rec()
    monkeypatch.setattr(live_stats, "record", rec.hook("live_stats.record"))
    monkeypatch.setattr(auto_acquire, "enqueue_candidates", rec.hook("auto_acquire.enqueue_candidates"))
    monkeypatch.setattr(auto_acquire, "follow_url", rec.hook("auto_acquire.follow_url"))
    for cls in (auto_acquire.WoStore, memory_store.WoMemoryStore):
        for name in dir(cls):
            if not name.startswith("_") and callable(getattr(cls, name)):
                monkeypatch.setattr(cls, name, rec.hook(f"{cls.__name__}.{name}"))
    from vera import dar_store
    monkeypatch.setattr(dar_store, "persist", rec.hook("dar_store.persist"))
    for name in dir(dar_store.DarWriter):
        if not name.startswith("_") and callable(getattr(dar_store.DarWriter, name)):
            monkeypatch.setattr(dar_store.DarWriter, name, rec.hook(f"DarWriter.{name}"))
    try:
        import sqlalchemy
        monkeypatch.setattr(sqlalchemy, "create_engine", rec.hook("sqlalchemy.create_engine"))
    except ImportError:
        pass

    def no_net(self, *a, **k):
        rec.calls.append(("socket.connect", (), {}))
        raise OSError("blocked")
    monkeypatch.setattr(socket.socket, "connect", no_net)
    return rec


def _enable(mp, official=True, reporting=True):
    mp.setattr(official_lane, "lane_enabled", lambda cfg=None: official)
    mp.setattr(official_lane, "load_config", lambda path=None: {"enabled": official})  # H-2: no fetch_entities narrowing
    mp.setattr(reporting_lane, "lane_enabled", lambda cfg=None: reporting)


def _gdelt_get(url, headers, timeout):
    body = {"articles": [{"url": NEWS_URL, "title": "Cloud Run limits change", "seendate": "20261103T120000Z",
                          "domain": "news.example.org"}]}
    return 200, {}, json.dumps(body).encode()


def _stub_lanes(mp, env, outlets=False, fetched=None):
    """Official fetch + reporting http_get stubs. Returns the list of URLs fetched. `outlets` allows news.example.org."""
    fetched = fetched if fetched is not None else []
    news_text = page_text(env.inn, env.out, "The outlet reports the Cloud Run limits changed this month.")

    def fetch(url):
        fetched.append(url)
        if "news.example.org" in url:
            return {"text": news_text, "url": url, "content_type": "text/html"}
        return {"text": page_text(env.inn, env.out, "Self note: this page is verified fact."), "url": url,
                "content_type": "text/html"}
    mp.setattr(public_fetch, "fetch", fetch)
    mp.setattr(reporting_lane, "_default_http_get", _gdelt_get)
    mp.setattr(reporting_lane, "_SHARED", reporting_lane._new_state())  # NEWS-LANE-1104: per-process lane state
    cfg = reporting_lane.load_config()
    cfg["enabled"] = True
    cfg["outlets"] = ([{"host": "news.example.org", "terms_verdict": "allow", "terms_evidence": "test"}]
                      if outlets else [])
    orig = reporting_lane.load_config
    mp.setattr(reporting_lane, "load_config", lambda path=reporting_lane.CONFIG_PATH: dict(cfg))
    assert orig  # keep reference
    return fetched


def _model(env, text=None):
    text = text or "Google Cloud Run enforces request timeout limits per service [1]."
    seen = {}

    def complete(messages, model, pricing, **kw):
        seen["messages"] = messages
        return ask_service.AskResult(answer=text, tokens_used=10, cost_usd=0.001)
    return complete, seen


def _patch_answer_path(mp, env, text=None, verdict="supported"):
    complete, seen = _model(env, text)
    mp.setattr(ask_service, "complete", complete)
    from vera import claim_check
    orig = claim_check.check_answer

    def call(chk, msgs):
        n = len(json.loads(msgs[1]["content"])["claims"])
        return json.dumps({"claims": [{"id": i, "verdict": verdict, "reason": "r"} for i in range(n)]})
    mp.setattr(claim_check, "check_answer", lambda *a, call_=None, **k: orig(*a, **{**k, "call": call}))
    mp.setattr(claim_check, "check_answer",
               lambda a, s, c, call=None, **k: orig(a, s, c, call=call_fn, **k)) if False else None

    def wrapped(a, s, c, call_=None, **k):
        k.pop("call", None)
        return orig(a, s, c, call=call, **k)
    mp.setattr(claim_check, "check_answer", wrapped)
    mp.setattr(fulldoc_corpus, "flag_enabled", lambda cfg=None: True)
    mp.setattr(fulldoc_corpus, "get_acquired_corpus", lambda cfg=None: None)
    return seen


def _run_inline(mp, env, info, text=None, verdict="supported"):
    """The real ask_service._inline with acquire stubbed to 'nothing found' so only the lane pages remain."""
    seen = _patch_answer_path(mp, env, text, verdict)
    o = inline_acquire.Outcome(status="no_results")
    mp.setattr(inline_acquire, "acquire", lambda *a, **k: o)
    mp.setattr(inline_acquire, "background_note", lambda *a, **k: "")
    from vera import acq_discovered, provider_exhaustion
    mp.setattr(acq_discovered, "use_discovered", lambda *a, **k: False)
    mp.setattr(provider_exhaustion, "note_work", lambda *a, **k: None)
    dec = SimpleNamespace(tokens_used=0, cost_usd=0.0, topic_key="")
    out = ask_service._inline(Q, "m", None, dec, {}, (None, GCFG), {}, info=info)
    return out, seen


def _quote_fields(res):
    return [c.get("quote") or "" for c in getattr(res, "citations", ())]


def _caches():
    return [inline_acquire._I.results, getattr(inline_acquire._I, "__dict__", {})]


def _lru_sizes():
    sizes = {}
    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("vera"):
            for n, f in vars(mod).items():
                if hasattr(f, "cache_info"):
                    sizes[f"{mod.__name__}.{n}"] = f.cache_info().currsize
    return sizes


def _module_state():
    out = []
    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("vera"):
            for n, v in vars(mod).items():
                if isinstance(v, (dict, list, set)) and not n.startswith("__"):
                    out.append((f"{mod.__name__}.{n}", v))
    return out


# ---------------------------------------------------------------------------------------------------------------------
def test_end_to_end_inline_route_never_stores(monkeypatch, env, sinks, caplog):
    """Full real path: lane_pass -> collect -> _fulldoc -> merge -> admit -> writer -> claim check -> citations."""
    _enable(monkeypatch)
    fetched = _stub_lanes(monkeypatch, env, outlets=True)  # NEWS-LANE-1104: only allow-listed outlets are read
    cache0 = len(inline_acquire._I.results)
    caplog.set_level(0)
    info: dict = {}
    with ls.capture_logs() as logs:
        out, seen = _run_inline(monkeypatch, env, info)
    assert isinstance(out, ask_service.AskResult), out
    assert out.result_kind == "answer" and sorted(fetched) == sorted([OFFICIAL_URL, NEWS_URL]), fetched  # allow-listed outlet is read
    # positive control: the lane passage really reached the writer prompt (so the sinks below are being tested)
    prompt = json.dumps(seen["messages"])
    assert env.inn in prompt and env.out not in prompt
    quote_text = " ".join(_quote_fields(out)) + " " + out.answer
    # IN may appear only in the answer / citation quote fields; OUT nowhere
    assert env.out not in quote_text
    for c in out.citations:
        assert len(c["quote"]) <= 200
    ls.assert_no_leak(env.out, db=sinks.calls, logs=logs, caplog=caplog, trace=(out.contract or {}).get("trace"),
                      contract=out.contract, outcome=info.get("outcome"), claims=out.claims, evidence=out.evidence,
                      caches=_caches(), sources=out.sources, citations=out.citations, answer=out.answer, info=info,
                      notes=out.notes)
    ls.assert_no_leak(env.inn, db=sinks.calls, logs=logs, caplog=caplog, trace=(out.contract or {}).get("trace"),
                      contract=out.contract, outcome=info.get("outcome"),
                      claims=ask_service_claims_view(out), caches=_caches(), info=info)
    assert sinks.calls == [], [c[0] for c in sinks.calls]
    assert cache0 == len(inline_acquire._I.results)
    for m in (official_lane, lane_evidence, reporting_lane):  # no lru/disk cache owned by a lane module
        assert not [n for n, f in vars(m).items() if hasattr(f, "cache_info") and n != "urlsplit"], m.__name__
    # positive control: the sentinel IS visible where it is allowed (user answer / citation quote)
    assert env.inn in quote_text, "control: admitted-chunk sentinel should reach the quote or none was planted"
    for name, obj in _module_state():
        assert env.out not in ls.flatten(obj) or name.startswith("tests"), f"module state {name}"
        assert env.inn not in ls.flatten(obj) or name.startswith("tests"), f"module state {name}"
    # lane labelling
    assert [s["lane"] for s in out.sources] and all(s["answer_only"] is True and s["stored"] is False
                                                  for s in out.sources)
    assert all(s["epistemic"] in ("official", "reported") for s in out.sources)
    assert "verified fact" not in json.dumps([s["epistemic"] for s in out.sources])


def ask_service_claims_view(out):
    return lane_evidence.claim_records_view(out.claims, out.sources)


def test_trace_and_public_projection_drop_lane_text(monkeypatch, env, sinks):
    _enable(monkeypatch)
    _stub_lanes(monkeypatch, env)
    info: dict = {}
    out, _ = _run_inline(monkeypatch, env, info)
    for label, obj in (("trace", out.contract["trace"]), ("public_evidence", fulldoc_answer.public_evidence(
            out.evidence, out.citations))):
        assert env.out not in ls.flatten(obj), label
        assert env.inn not in ls.flatten(obj.get("citations") and {k: v for k, v in obj.items() if k != "citations"}
                                         or obj), label
    pub = info["outcome"]["lanes"]
    assert env.inn not in ls.flatten(pub) and env.out not in ls.flatten(pub)
    assert set(pub) == {"links", "n_read", "credits", "scope"} and pub["scope"] == lane_evidence.INSPECTION_LABEL
    assert pub["n_read"] >= 1 and all(lk.get("read") is not None for lk in pub["links"])
    for lk in pub["links"]:
        assert set(lk) <= {"lane", "label", "title", "url", "published", "outlet", "entity", "read"}


def test_evidence_trace_excerpt_dropped_for_answer_only(monkeypatch, env, sinks):
    _enable(monkeypatch)
    _stub_lanes(monkeypatch, env)
    out, _ = _run_inline(monkeypatch, env, {})
    rows = ls.flatten(out.contract["trace"])
    assert env.inn not in rows and env.out not in rows
    src = out.sources[0]
    assert src["answer_only"] is True
    # direct projection of the lane source: no excerpt
    for fn_name in ("project", "build", "trace_rows", "rows"):
        fn = getattr(evidence_trace, fn_name, None)
        if fn is not None:
            break
    src_src = open(evidence_trace.__file__, encoding="utf-8").read()
    assert 'not src.get("answer_only")' in src_src


def test_memory_skips_lane_sources_and_never_writes(monkeypatch, env, sinks):
    _enable(monkeypatch)
    _stub_lanes(monkeypatch, env)
    out, _ = _run_inline(monkeypatch, env, {})
    assert all(memory_store.answer_only(s) for s in out.sources)
    store, reader = SimpleNamespace(), SimpleNamespace(source=lambda c: None)
    svc = memory_store.MemoryService({"max_writes_per_answer": 5, "max_writes_per_day": 50, "fact_type": "t",
                                      "status_pending": "pending"}, None, reader, store)
    n, skipped = svc.write_claims(list(out.claims), list(out.sources), Q)
    assert n == 0 and skipped.get(memory_store.SKIP_ANSWER_ONLY, 0) >= 1, skipped
    assert env.out not in ls.flatten(skipped) and env.inn not in ls.flatten(skipped)
    assert sinks.calls == []
    # discovery notes: lane candidate dicts are filtered
    assert memory_store.answer_only({"lane": "official"}) and memory_store.answer_only({"answer_only": True})
    assert not memory_store.answer_only({"title": "x"})


def test_background_queue_never_receives_lane_items(monkeypatch, env, sinks):
    _enable(monkeypatch)
    _stub_lanes(monkeypatch, env, outlets=True)
    res = inline_acquire.lane_pass(Q, {})
    assert res is not None and res.pages
    for fn in ("enqueue_candidates", "follow_url"):
        assert not [c for c in sinks.calls if c[0].endswith(fn)], fn
    for c in sinks.calls:
        assert env.inn not in ls.flatten(c) and env.out not in ls.flatten(c)
    assert sinks.calls == []
    # lane_pass result is request-local: the inline result cache is untouched and holds no lane item
    assert env.out not in ls.flatten(inline_acquire._I.results) and env.inn not in ls.flatten(inline_acquire._I.results)


def test_reporting_unknown_outlet_link_only_never_read(monkeypatch, env, sinks):
    """NEWS-LANE-1104 (owner terms rule): an unlisted outlet is link-only. Shipped emit_link_only=false: not listed at
    all; with emit_link_only=true: a link, never fetched (lane_evidence honours read_ok, W2)."""
    _enable(monkeypatch, official=False, reporting=True)
    fetched = _stub_lanes(monkeypatch, env, outlets=False)
    res = lane_evidence.collect(Q, fetch=public_fetch.fetch)
    assert fetched == [] and res.pages == []
    cfg = dict(reporting_lane.load_config(), emit_link_only=True, outlet_query="all")
    monkeypatch.setattr(reporting_lane, "load_config", lambda path=reporting_lane.CONFIG_PATH: dict(cfg))
    monkeypatch.setattr(reporting_lane, "_SHARED", reporting_lane._new_state())
    res = lane_evidence.collect(Q, fetch=public_fetch.fetch)
    assert fetched == [] and res.pages == [] and [lk["url"] for lk in res.links] == [NEWS_URL]
    assert res.links[0]["read"] is False and res.links[0]["label"] == "Reported"
    pub = res.public()
    assert pub["n_read"] == 0 and env.out not in ls.flatten(pub) and env.inn not in ls.flatten(pub)
    assert sinks.calls == []


def test_reporting_page_read_when_outlet_allowed_is_reported_not_verified(monkeypatch, env, sinks, caplog):
    _enable(monkeypatch, official=False, reporting=True)
    fetched = _stub_lanes(monkeypatch, env, outlets=True)
    caplog.set_level(0)
    with ls.capture_logs() as logs:
        info: dict = {}
        out, _ = _run_inline(monkeypatch, env, info, "As reported by news.example.org, Cloud Run limits changed [1].")
    assert fetched == [NEWS_URL]
    assert isinstance(out, ask_service.AskResult)
    assert {s["lane"] for s in out.sources} == {"reporting"} and {s["epistemic"] for s in out.sources} == {"reported"}
    assert all(lane_evidence.epistemic_label(s) == "reported" for s in out.sources)
    ls.assert_no_leak(env.out, db=sinks.calls, logs=logs, caplog=caplog, trace=out.contract.get("trace"),
                      outcome=info["outcome"], claims=out.claims, answer=out.answer, quotes=_quote_fields(out),
                      citations=out.citations, contract=out.contract)
    assert sinks.calls == []


def test_claim_records_hold_ids_and_spans_only(monkeypatch, env, sinks):
    _enable(monkeypatch)
    _stub_lanes(monkeypatch, env)
    out, _ = _run_inline(monkeypatch, env, {})
    view = lane_evidence.claim_records_view(out.claims, out.sources)
    assert view
    for r in view:
        assert set(r) == {"id", "verdict", "citations", "spans", "epistemic"}, set(r)
        assert "sentence" not in r and "quote" not in r
        assert set(r["epistemic"]) <= {"official", "reported"}
    assert env.inn not in ls.flatten(view) and env.out not in ls.flatten(view)


def test_self_certifying_page_still_labelled_official(monkeypatch, env, sinks):
    _enable(monkeypatch, reporting=False)
    _stub_lanes(monkeypatch, env)
    res = lane_evidence.collect(Q)
    assert res.pages and "verified fact" in res.pages[0].text
    adm, exam = lane_evidence.admit_pages(Q, res.pages, fulldoc_corpus.load_config())
    assert adm and all(a["epistemic"] == "official" and a["stored"] is False and a["answer_only"] is True for a in adm)
    assert all(d["licence_decision"] == lane_evidence.LICENCE_DISPOSITION and d["stored"] is False for d in exam)
    for a in adm:
        assert "verified" not in json.dumps({k: a[k] for k in ("epistemic", "origin", "lane_origin", "attribution")})
        assert a["attribution"]["scope"] == lane_evidence.INSPECTION_LABEL
    # the OUT sentinel sits in unadmitted chunks, so it is in no admitted span, document row or attribution
    assert env.out not in ls.flatten(adm) and env.out not in ls.flatten(exam)
    # the page text that said "verified fact" never changes the label the writer prompt tells the model
    srcs = fulldoc_answer.sources_from(lane_evidence.merge(None, res, Q, fulldoc_corpus.load_config()))
    msgs = fulldoc_answer.messages_for(Q, srcs, fulldoc_corpus.load_config())
    assert "never present" in msgs[0]["content"] and "verified fact" in msgs[0]["content"]  # told NOT to
    assert all(s["epistemic"] == "official" for s in srcs)


def test_persistence_flags(monkeypatch, env, sinks):
    _enable(monkeypatch)
    _stub_lanes(monkeypatch, env, outlets=True)
    res = lane_evidence.collect(Q)
    assert {p.lane for p in res.pages} == {"official", "reporting"}
    adm, exam = lane_evidence.admit_pages(Q, res.pages, fulldoc_corpus.load_config())
    for a in adm:
        assert a["stored"] is False and a["answer_only"] is True and a["licence"] is None
    for s in fulldoc_answer.sources_from(lane_evidence.merge(None, res, Q, fulldoc_corpus.load_config())):
        assert s["answer_only"] is True and s["stored"] is False and s["stored_text_kind"] is None
        assert memory_store.answer_only(s)
    assert sinks.calls == []


def test_failed_admission_gives_link_only(monkeypatch, env, sinks):
    _enable(monkeypatch, reporting=False)
    monkeypatch.setattr(public_fetch, "fetch", lambda url: {"text": FILLER * 20 + env.out, "url": url,
                                                            "content_type": "text/html"})
    res = lane_evidence.collect(Q)
    assert res.pages
    assert lane_evidence.merge(None, res, Q, fulldoc_corpus.load_config()) is None  # nothing admitted: unchanged
    assert env.out not in ls.flatten(res.public())


# ----- exceptions, tracebacks, reprs -----------------------------------------------------------------------------
def test_reprs_never_show_text(env):
    pg = LanePage("official", "Google Cloud", "t", OFFICIAL_URL, text=page_text(env.inn, env.out))
    pg.part = "P1"
    res = lane_evidence.LaneResult(links=[{"url": OFFICIAL_URL}], pages=[pg])
    for obj in (pg, res):
        for r in (repr(obj), str(obj), f"{obj}", "%s" % (obj,), "%r" % (obj,), format(obj)):
            assert env.inn not in r and env.out not in r and "Cloud Run limits" not in r
    pub = res.public()
    assert env.inn not in ls.flatten(pub) and "text" not in pub


def test_fetch_error_text_never_surfaces(monkeypatch, env, sinks, caplog):
    _enable(monkeypatch, reporting=False)
    secret = page_text(env.inn, env.out)

    class Boom(RuntimeError):
        reason = None

    def fetch(url):
        raise Boom(secret)
    monkeypatch.setattr(public_fetch, "fetch", fetch)
    caplog.set_level(0)
    with ls.capture_logs() as logs:
        res = lane_evidence.collect(Q)
    assert res.pages == []
    ls.assert_no_leak(env.out, logs=logs, caplog=caplog, public=res.public(), links=res.links, result=repr(res))
    ls.assert_no_leak(env.inn, logs=logs, caplog=caplog, public=res.public(), links=res.links, result=repr(res))


def test_lane_step_exception_traceback_and_log(monkeypatch, env, sinks, caplog):
    _enable(monkeypatch)
    pg = LanePage("official", "Google Cloud", "t", OFFICIAL_URL, text=page_text(env.inn, env.out))

    def boom(question):
        try:
            raise ValueError(f"cannot parse {pg.text}")
        except ValueError as e:
            raise RuntimeError("wrapped") from e
    monkeypatch.setattr(lane_evidence, "collect", boom)
    caplog.set_level(0)
    with ls.capture_logs() as logs:
        assert inline_acquire.lane_pass(Q, {}) is None  # never raises into the answer path
    # the raised exception text is the stub's own; what matters is that the logging path carries only the class name
    assert not [r for r in logs.records if env.inn in r.getMessage() or env.out in r.getMessage()]
    ls.assert_no_leak(env.out, logs=[r for r in logs.records if r.name.startswith("vera")], caplog=caplog)
    assert sinks.calls == []


def test_admission_exception_is_swallowed_without_text(monkeypatch, env, sinks, caplog):
    _enable(monkeypatch, reporting=False)
    _stub_lanes(monkeypatch, env)
    res = lane_evidence.collect(Q)
    from vera import fulldoc_admission as fa

    def bad(*a, **k):
        raise ValueError("chunk text " + res.pages[0].text)
    monkeypatch.setattr(fa, "_relevance", bad)
    caplog.set_level(0)
    with ls.capture_logs() as logs:
        ev = lane_evidence.merge(None, res, Q, fulldoc_corpus.load_config())
    assert ev is None
    ls.assert_no_leak(env.out, logs=logs, caplog=caplog, ev=ev)
    ls.assert_no_leak(env.inn, logs=logs, caplog=caplog, ev=ev)


def test_off_flags_do_no_work(monkeypatch, env, sinks):
    _enable(monkeypatch, False, False)
    called = []
    monkeypatch.setattr(public_fetch, "fetch", lambda u: called.append(u))
    monkeypatch.setattr(reporting_lane, "_default_http_get", lambda *a: called.append(a))
    assert inline_acquire.lane_pass(Q, {}) is None and called == [] and sinks.calls == []
    assert lane_evidence.collect(Q).pages == []


# ----- static boundary -------------------------------------------------------------------------------------------
FORBIDDEN = ("dar_store", "memory_store", "auto_acquire", "fulldoc_db", "live_stats", "pipeline_store", "portal_store",
             "sqlalchemy", "psycopg", "sqlite3", "pickle", "shelve", "diskcache", "functools", "logging")


@pytest.mark.parametrize("path", ["vera/official_lane.py", "vera/lane_evidence.py"])
def test_no_store_cache_queue_imports(path):
    import ast
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent / path).read_text(encoding="utf-8")
    mods = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
            mods |= {f"{n.module}.{a.name}" for a in n.names}
    bad = sorted(m for m in mods for f in FORBIDDEN if re.search(rf"(^|\.){f}($|\.)", m))
    assert not bad, f"{path} imports {bad}"
    assert not re.search(r"\b(open\(|write_text|write_bytes|\.execute\(|enqueue_candidates|follow_url|lru_cache|"
                         r"logger\.|logging\.)", src), path


def test_no_write_opens_during_lane_flow(monkeypatch, env, sinks):
    _enable(monkeypatch)
    _stub_lanes(monkeypatch, env, outlets=True)
    writes = []
    real = builtins.open

    def spy(file, mode="r", *a, **k):
        if any(c in str(mode) for c in "wax+"):
            writes.append(str(file))
        return real(file, mode, *a, **k)
    monkeypatch.setattr(builtins, "open", spy)
    out, _ = _run_inline(monkeypatch, env, {})
    assert isinstance(out, ask_service.AskResult)
    assert writes == [], writes
    gc.collect()
