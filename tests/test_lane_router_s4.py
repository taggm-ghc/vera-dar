"""OFFICIAL-LANE-S4A S4 tests for vera/lane_router.py (offline: stub temporality and news_intent, inline registry; no
network/DB). G11 tests use the real vera.temporality and vera.news_intent (config reads only). Works from repo/tests
or work/OFFICIAL-LANE-S4A/staged/tests (the sibling vera/ directory is loaded in both layouts)."""
import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

_BASE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("s4a_lane_router", _BASE / "vera" / "lane_router.py")
lr = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = lr
_spec.loader.exec_module(lr)

REG = [{"entity": "Google Cloud", "state": "active"}, {"entity": "AWS", "state": "active"},
       {"entity": "Kubernetes", "state": "draft"}, {"entity": "NoState"}]
ACQ_ON = {"lane_router": {"enabled": True, "run_on_sufficient": ["official", "reporting", "bogus"]},
          "inline": {"web_read_enabled": True}}


class Temp:
    """Time cue only on explicit words; records the volatile flag it was called with."""
    calls = []

    @staticmethod
    def detect(text, volatile_entity=False, cfg=None):
        Temp.calls.append(volatile_entity)
        return NS(current=volatile_entity or any(w in text.lower() for w in ("latest", "today", "this week")))


class News:
    @staticmethod
    def news_intent(text):
        return "make the news" in text.lower()


def _plan(parts, acq=ACQ_ON, news=News, question="q"):
    sp = NS(parts=tuple(NS(label=l, text=t, entities=e) for l, t, e in parts))
    return lr.plan(question, acq, subq_plan=sp, registry=REG, temporality=Temp, news_intent=news)


def test_off_returns_none():
    assert _plan([("P1", "anything", ())], acq={"lane_router": {"enabled": "true"}}) is None
    assert _plan([("P1", "anything", ())], acq={}) is None
    assert _plan([("P1", "anything", ())], acq=None) is None


def test_any_topic_gets_research_and_web_never_fewer():
    for q in ("sourdough hydration", "Roman law of contracts", "boson sampling", "who won the cup final", ""):
        p = _plan([("P1", q, ())])
        assert p.parts[0].lanes[:2] == ("research", "web")


def test_registry_entity_adds_official_only_for_active_rows():
    p = _plan([("P1", "Google Cloud TPU quotas", ("Google Cloud",)), ("P2", "Kubernetes pods", ("Kubernetes",)),
               ("P3", "x", ("NoState",))])
    assert "official" in p.parts[0].lanes and p.parts[0].entities == ("Google Cloud",)
    assert "official" not in p.parts[1].lanes  # draft row: no lane, no host guessed
    assert "official" not in p.parts[2].lanes  # a row without state is not active (fail closed)


def test_volatile_entity_alone_is_not_a_time_cue():
    Temp.calls.clear()
    p = _plan([("P1", "AWS Lambda limits", ("AWS",))])
    assert "reporting" not in p.parts[0].lanes
    assert Temp.calls and not any(Temp.calls)  # always called with volatile_entity=False (G11)


def test_time_or_news_cue_adds_reporting():
    p = _plan([("P1", "latest AWS outage", ("AWS",)), ("P2", "what make the news about the midterms", ())])
    assert "reporting" in p.parts[0].lanes and "volatile_with_time_cue" in p.parts[0].basis
    assert "reporting" in p.parts[1].lanes and "news_cue" in p.parts[1].basis


def test_failing_news_detector_adds_nothing_and_never_raises():
    class Bad:
        @staticmethod
        def news_intent(text):
            raise RuntimeError("x")
    p = _plan([("P1", "plain question", ())], news=Bad)
    assert p.parts[0].lanes == ("research", "web")


def test_trace_is_closed_tokens_only():
    p = _plan([("P1", "latest Google Cloud pricing secret-text", ("Google Cloud",))])
    tr = p.trace()
    assert set(tr["lanes"]) <= set(lr.LANES) and set(tr["basis"]) <= set(lr.BASIS)
    assert "secret-text" not in repr(tr)
    assert p.run_on_sufficient == ("official", "reporting")


def test_web_off_drops_only_web():
    acq = {"lane_router": {"enabled": True}, "inline": {"web_read_enabled": False}}
    p = _plan([("P1", "x", ())], acq=acq)
    assert p.parts[0].lanes == ("research",)


def test_plan_never_raises_on_bad_inputs():
    class Boom:
        @staticmethod
        def detect(*a, **k):
            raise RuntimeError("t")
    sp = NS(parts=(NS(label="P1", text="x", entities=()),))
    assert lr.plan("q", ACQ_ON, subq_plan=sp, registry=REG, temporality=Boom, news_intent=News) is None
    lr.plan("q", ACQ_ON, subq_plan=sp, registry="not-a-list", temporality=Temp)  # must not raise


def test_router_cannot_decline_or_choose_by_subject():
    """Nothing in routing can decline: no decline/accept/offtopic token, no subject lexicon, no lane removal."""
    src = (_BASE / "vera" / "lane_router.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    tree = ast.parse(code)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for bad in ("decline_template", "offtopic_warning", "accepted", "refuse", "lexicon", "source_type", "remove"):
        assert bad not in names, bad
    assert set(lr.PartRoute.__dataclass_fields__) == {"label", "lanes", "basis", "entities"}


def test_router_imports_no_store_cache_or_corpus_module():
    tree = ast.parse((_BASE / "vera" / "lane_router.py").read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mods |= {f"{n.module}.{a.name}" for a in n.names}
        elif isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
    assert mods <= {"__future__.annotations", "dataclasses.dataclass", "dataclasses.field", "vera.temporality",
                    "vera.news_intent", "vera.subquery"}, mods


# ---- G11 with the real temporality and news_intent ---------------------------------------------------------
def test_g11_reporting_cue_real_modules():
    from vera import temporality
    assert temporality.detect("AWS Lambda limits", volatile_entity=True).current is True  # the old side effect
    assert lr.reporting_cue("AWS Lambda limits") is False  # G11: a vendor name alone is no cue
    assert lr.reporting_cue("Google Cloud TPU quotas", "What are Google Cloud TPU quotas?") is False
    assert lr.reporting_cue("the latest AWS outage") is True


def test_g11_reporting_cue_news_on_question_and_never_raises():
    assert lr.reporting_cue("Slack", "did the Slack outage make the news", temporality=Temp, news_intent=News) is True

    class Boom:
        @staticmethod
        def detect(*a, **k):
            raise RuntimeError("t")
    assert lr.reporting_cue("latest", temporality=Boom, news_intent=News) is False


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"{n} passed")
