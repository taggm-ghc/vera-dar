"""PROGRESS-COUNT-1106: saved never exceeds its total, one position notion, collapsed log lines. Offline."""
from vera import inline_acquire as ia
from vera import progress
from vera.progress_ui import Labels, StreamHolder, stream_view

def _labels():
    return Labels.load()


def _ev(i, checked, held=0, stored=5, target=6):
    return {"stage": "acquire", "state": "update", "seq": i,
            "counts": {"checked": checked, "shortlisted": 31, "usable": 13, "held": held, "stored": stored, "target": target}}


def test_label_clamps_stored_to_target():
    t = _labels().stage_label("acquire", "update", _ev(1, 30, stored=7)["counts"], "")
    assert "6 of 6 saved" in t and "7 of 6" not in t


def test_log_collapses_count_updates():
    h = StreamHolder(t0=0.0, last_read=0.0)
    for i in range(12):
        h.events.append(_ev(i, 19 + i, stored=min(5 + i, 7)))
        h.event_at.append(float(i))  # 11 s span: one refresh line at most
    v = stream_view(h, _labels(), 12.0)
    assert len(v["log"]) <= 2 and v["log"][-1].startswith("Checked 30 of 31")


def test_log_new_line_when_held_changes_or_10s():
    h = StreamHolder(t0=0.0, last_read=0.0)
    for i, (held, at) in enumerate([(0, 0), (0, 1), (1, 2), (1, 3), (1, 20)]):
        h.events.append(_ev(i, 10 + i, held=held))
        h.event_at.append(float(at))
    assert len(stream_view(h, _labels(), 21.0)["log"]) == 3


def test_slow_read_reports_in_flight_count_not_rank_index(monkeypatch):
    got = []
    monkeypatch.setattr(progress, "set_activity", lambda code, **kw: got.append((code, kw)))
    monkeypatch.setattr(progress, "clear_activity", lambda: None)

    class F:  # distinct hashable stand-ins for futures
        pass
    a, b, c = F(), F(), F()
    started = {a: (7, 0.0), b: (9, 5.0), c: (11, 6.0)}
    ia._slow_read({a, b, c}, started, 31, {"mark": 0}, now=lambda: 25.0)
    assert got == [(progress.ACT_SLOW_READ, {"k": 3, "n": 31, "elapsed": 25})]
