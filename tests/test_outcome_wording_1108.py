"""WORDING-LABELS-1110 (a): the notice never claims a "sources targeted" ratio; it states the obtained count and the aim."""
import re

from vera import ui_evidence as _u

_OLD = re.compile(r"of the \d+ (new )?sources targeted|Read 0 of the 6")


def test_no_sources_targeted_wording_1110():
    for st in ("shortfall", "deadline", "relevance_rejected", "exhausted", None):
        for led in (None, {"counted": 0, "full_text": 0, "partial_text": 0, "examined_unqualified": 0, "stored": 0}):
            d = {"contract": {"inline_acquisition": {"acquisition": {"status": st, "target": 6, "stored": 0}}}}
            line = _u.acquisition_notice_line(d)
            assert line is None or not _OLD.search(line)
    d = {"contract": {"inline_acquisition": {"acquisition": {"status": "supply_limited", "target": 6, "stored": 0}}}}
    line = _u.acquisition_notice_line(d)
    assert line is None or "(aim: at least 6)" in line
