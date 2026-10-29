"""PROG-1027 W4b-3: poll_tick (the fragment body) called directly, plain pytest, no AppTest; config.toml (R4-PROG-S-1)."""
import tomllib
from pathlib import Path

import pytest

from vera import progress
from vera import progress_ui as pu
from vera.public_mode import GONE

ROOT = Path(__file__).resolve().parent.parent
LABELS = pu.Labels.load()
CFG = progress.validate_config({"enabled": True, "max_records": 8, "max_events_per_record": 8, "max_events_per_poll": 8,
                                "ttl_s": 600, "poll_interval_s": 1.0, "poll_margin_s": 5, "first_event_grace_s": 10,
                                "get_timeout_s": 2, "max_count": 10000})
RID = "ab12cd34" + "0" * 24
HOST = "https://sentinel-host.invalid"


class UI:
    def __init__(self):
        self.out = []

    def caption(self, t):
        self.out.append(t)


def inflight(done=False, **hk):
    h = pu.Holder()
    for k, v in hk.items():
        setattr(h, k, v)
    if done:
        h.done.set()
    return pu.Inflight(rid=RID, holder=h, t0=100.0)


def ev(seq, stage, state):
    return {"v": "1", "seq": seq, "ts": "t", "stage": stage, "state": state, "counts": {}}


def tick(inf, ses, ui, fetch, now=101.0):
    return pu.poll_tick(inf, ses, ui, fetch, lambda: now, CFG, LABELS)


def test_terminal_twice_gives_one_completion_and_one_durable_result():
    inf = inflight(done=True, status=200, result={"result_kind": "answer"})
    ses, ui = {"vera_inflight": inf}, UI()
    assert tick(inf, ses, ui, lambda a: []) is True
    assert "vera_inflight" not in ses and ses["vera_last_result"].terminal == "answered"
    first = ses["vera_last_result"]
    assert tick(inf, ses, ui, lambda a: []) is False  # fin guard
    assert ses["vera_last_result"] is first


def test_no_terminal_while_worker_running_renders_label_and_caption():
    inf = inflight()
    ses, ui = {"vera_inflight": inf}, UI()
    assert tick(inf, ses, ui, lambda a: [ev(1, "received", "done")]) is False
    assert ui.out[0].startswith("Question received") and ui.out[-1] == LABELS.reload_caption
    assert "vera_inflight" in ses


@pytest.mark.parametrize("fetch_result", [None, [{"v": "2"}]])
def test_unknown_or_missing_events_never_raise(fetch_result):
    inf = inflight()
    ses, ui = {"vera_inflight": inf}, UI()
    assert tick(inf, ses, ui, lambda a: fetch_result, now=200.0) is False  # past the grace: fixed text only
    assert HOST not in " ".join(ui.out)


def test_raising_fetch_shows_fixed_text_and_does_not_leak():
    def boom(after):
        raise RuntimeError("cannot reach " + HOST)
    inf = inflight()
    ses, ui = {"vera_inflight": inf}, UI()
    assert tick(inf, ses, ui, boom) is False
    assert ui.out == [LABELS.progress_unavailable]


def test_raising_fetch_with_done_holder_still_completes_once():
    def boom(after):
        raise RuntimeError(HOST)
    inf = inflight(done=True, status=200, result={"result_kind": "answer"})
    ses, ui = {"vera_inflight": inf}, UI()
    assert tick(inf, ses, ui, boom) is True
    assert ses["vera_last_result"].terminal == "answered" and ses["vera_last_result"].result == {"result_kind": "answer"}


def test_raising_label_lookup_is_guarded():
    class BadLabels(pu.Labels):
        def stage_label(self, *a, **k):
            raise KeyError("x")
    labels = BadLabels(LABELS._raw)
    inf = inflight()
    ses, ui = {"vera_inflight": inf}, UI()
    assert pu.poll_tick(inf, ses, ui, lambda a: [ev(1, "received", "done")], lambda: 101.0, CFG, labels) is False


def test_backstop_ends_the_run_with_poll_timeout():
    inf = inflight()
    ses, ui = {"vera_inflight": inf}, UI()
    assert tick(inf, ses, ui, lambda a: GONE, now=100.0 + 180 + 5) is True
    assert ses["vera_last_result"].terminal == "poll_timeout_unreceived"


def test_start_tracked_single_flight_and_sending_before_worker():
    order, ses = [], {}
    ui = UI()
    ui.caption = lambda t: order.append(("ui", t))
    spawned = []
    mk = iter(["r1" * 16, "r2" * 16])
    args = ("http://h", "q", "normal", "k")
    inf = pu.start_tracked(ses, lambda: next(mk), "T" * 43, lambda: 1.0, lambda *a: None, args, ui, LABELS,
                           spawn=lambda *a: (order.append(("spawn",)), spawned.append(a)))
    assert order[0][0] == "ui" and order[0][1].startswith("Sending your question") and order[1] == ("spawn",)
    assert spawned[0][3] == ("http://h", "q", "normal", "k", "r1" * 16, "T" * 43)
    again = pu.start_tracked(ses, lambda: next(mk), "T" * 43, lambda: 2.0, lambda *a: None, args, ui, LABELS,
                             spawn=lambda *a: spawned.append(a))
    assert again is inf and len(spawned) == 1


def test_streamlit_config_hides_error_details():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["client"]["showErrorDetails"] is False
