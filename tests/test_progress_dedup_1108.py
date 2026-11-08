"""SELFQ-VARIANTS-1108: progress de-duplication and the rate-limit wait wording. Offline."""
import pytest

from vera import progress_ui


# ---- progress ----
class _H:
    def __init__(self, evs, activity=None):
        self.events, self.activity, self.t0, self.last_line_at = evs, activity, 0.0, 0.0


def _ev(stage, state, **c):
    return {"type": "event", "stage": stage, "state": state, "counts": c}


def _labels():
    return progress_ui.load_labels() if hasattr(progress_ui, "load_labels") else progress_ui.Labels.load()


def test_start_line_not_repeated_and_header_not_copied(monkeypatch):
    lab = _labels()
    st = next(s for s, d in lab._stages.items() if d.get("start", "").startswith("Searching open-access"))
    evs = [_ev(st, "start", step=1), _ev(st, "start", step=1)]
    v = progress_ui.stream_view(_H(evs), lab, 3.0)
    assert v["log"].count("Searching open-access sources (step 1)") == 1
    assert v["activity"].startswith("Working on your question")  # not a copy of the label


def test_stream_tick_header_line_not_written_twice():
    lab = _labels()
    st = next(s for s, d in lab._stages.items() if d.get("start", "").startswith("Searching open-access"))

    class UI:
        def __init__(self): self.w, self.label = [], None
        def status(self, label, **k):
            self.label = label
            return self
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def write(self, t): self.w.append(t)
        def progress(self, *a, **k): pass
        def info(self, t): self.w.append("INFO " + t)
        def caption(self, t): pass
    h = progress_ui.StreamHolder(t0=0.0, last_read=0.0)
    h.events = [_ev(st, "start", step=1)]
    ui = UI()
    progress_ui.stream_tick(h, {}, ui, lambda: 1.0, lab)
    assert ui.label == "Searching open-access sources (step 1)"
    assert "Searching open-access sources (step 1)" not in ui.w


@pytest.mark.parametrize("w,bad", [(0, "about 0"), (1, None)])
def test_rate_wait_wording(w, bad):
    lab = _labels()
    t = lab.activity_label({"act": 1, "wait_s": w})
    assert t and "about 0" not in t
    if w == 0:
        assert "briefly" in t
    assert "about 0 s" not in (lab.activity_label({"act": 4, "wait_s": 0}) or "")


def test_checked_update_lines_update_in_place():
    lab = _labels()
    st = next(s for s, d in lab._stages.items() if "Checked {checked}" in d.get("update", ""))
    evs = [_ev(st, "update", checked=c, shortlisted=18, usable=0, held=0, stored=0, target=6) for c in (15, 15, 15, 16)]
    v = progress_ui.stream_view(_H(evs), lab, 3.0)
    assert sum(1 for l in v["log"] if l.startswith("Checked ")) == 1
    assert len(set(v["log"])) == len(v["log"])
