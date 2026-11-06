"""MIDTERMS-WP4-1106: GDELT query built from the event core; one broader retry after min_interval_s. Offline."""
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from vera import reporting_lane as rl

ROOT = Path(__file__).resolve().parent.parent
Q = ("What are implications of the US 2026 Midterm elections? Keep searching until you find an answer.")
ONE = json.dumps({"articles": [{"url": "https://x.org/a", "title": "T", "seendate": "20261104T000000Z"}]}).encode()
NONE = json.dumps({"articles": []}).encode()


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


class Fake:
    def __init__(self, clock, *responses):
        self.clock, self.r, self.calls = clock, list(responses), []

    def __call__(self, url, headers, timeout):
        self.calls.append((self.clock.t, parse_qs(urlsplit(url).query)["query"][0]))
        self.clock.t += 0.5  # the call itself takes time
        return (200, {}, self.r.pop(0))


def lane(clock, fake, **kw):
    c = rl.load_config(ROOT / "config" / "reporting_lane.json")
    c.update(enabled=True, fallback_order=["gdelt"], outlets=[], outlet_query="all", emit_link_only=True,
             min_interval_s=6)
    c.update(**kw)
    return rl.ReportingLane(c, http_get=fake, sleep=clock.sleep, clock=clock, rand=lambda: 0.5)


def test_query_is_event_core_without_filler():
    ck = Clock()
    f = Fake(ck, ONE)
    res = lane(ck, f).search(Q)
    assert len(f.calls) == 1
    q = f.calls[0][1]
    assert q == '"midterm elections" 2026'
    for w in ("searching", "until", "find", "keep", "answer", "implications"):
        assert w not in q.lower()
    d = res["diag"][0]
    assert (d["reason"], d["raw_items"], d["items"], d["attempts"]) == ("ok", 1, 1, 1)


def no_sleep(_s):
    raise AssertionError("the lane must never sleep in the request path")


def test_no_items_defers_broader_query_never_sleeps_then_next_call_uses_it():
    ck = Clock()
    ck.sleep = no_sleep
    f = Fake(ck, NONE, ONE)
    L = lane(ck, f)
    d1 = L.search(Q)["diag"][0]
    assert [c[1] for c in f.calls] == ['"midterm elections" 2026']  # spacing not elapsed: no second call
    assert (d1["reason"], d1["attempts"], d1["retry_deferred"], d1["raw_items"]) == ("no_items", 1, True, 0)
    assert L.search(Q)["diag"][0]["reason"] == "spacing"  # inside the interval: skipped as before
    ck.t += 6
    res = L.search(Q)
    assert [c[1] for c in f.calls][1] == "(midterm OR elections) 2026"  # the next call uses the broader query
    assert f.calls[1][0] - f.calls[0][0] >= 6
    assert res["diag"][0]["reason"] == "ok" and len(res["items"]) == 1
    assert L._next == {}


def test_broader_retry_runs_in_same_call_only_when_spacing_already_elapsed():
    ck = Clock()
    ck.sleep = no_sleep
    f = Fake(ck, NONE, ONE)
    d = lane(ck, f, min_interval_s=0).search(Q)["diag"][0]
    assert len(f.calls) == 2 and (d["attempts"], d["reason"]) == (2, "ok")


def test_empty_precise_and_broader_record_counts():
    ck = Clock()
    f = Fake(ck, NONE, NONE)
    L = lane(ck, f, min_interval_s=0)
    d = L.search(Q)["diag"][0]
    assert len(f.calls) == 2
    assert (d["reason"], d["raw_items"], d["items"], d["attempts"]) == ("no_items", 0, 0, 2)


def test_raw_vs_post_policy_counts():
    ck = Clock()
    body = json.dumps({"articles": [{"url": "https://bad.org/a"}, {"url": "https://ok.org/b"}]}).encode()
    f = Fake(ck, body)
    outlets = [{"host": "bad.org", "terms_verdict": "exclude"}]
    d = lane(ck, f, outlets=outlets).search(Q)["diag"][0]
    assert (d["raw_items"], d["items"], d["attempts"]) == (2, 1, 1)


def test_no_event_noun_keeps_term_query_and_no_retry():
    ck = Clock()
    f = Fake(ck, NONE, ONE)
    lane(ck, f).search("Kubernetes outage reports")
    assert len(f.calls) == 1 and "midterm" not in f.calls[0][1]


def test_synonyms_from_config_are_ored():
    ck = Clock()
    f = Fake(ck, ONE)
    c = rl.load_config(ROOT / "config" / "reporting_lane.json")
    c["news_intent"] = dict(c["news_intent"], event_synonyms={"midterm": ["midterm vote"]})
    lane(ck, f, news_intent=c["news_intent"]).search(Q)
    assert f.calls[0][1] == '("midterm elections" OR "midterm vote") 2026'


class Fake429(Fake):
    def __call__(self, url, headers, timeout):
        self.calls.append((self.clock.t, parse_qs(urlsplit(url).query)["query"][0]))
        return (429, {}, b"")


def test_gdelt_429_is_recorded_as_gdelt_response_and_backoff_is_honoured():
    ck = Clock()
    f = Fake429(ck)
    d = lane(ck, f).search(Q)["diag"][0]
    assert len(f.calls) == 1 and ck.slept == []  # the 60 s backoff window is not bypassed
    assert (d["reason"], d["throttle_source"]) == ("throttled", "gdelt_response")


def test_local_spacing_skip_is_recorded_not_slept():
    ck = Clock()
    f = Fake(ck, ONE)
    L = lane(ck, f)
    L._last_call["gdelt"] = ck.t - 2  # a call 2 s ago: the local limiter skips (existing behaviour kept)
    d = L.search(Q)["diag"][0]
    assert f.calls == [] and ck.slept == []
    assert (d["reason"], d["throttle_source"]) == ("spacing", "local_spacing")
