"""STRICTJSON-1027 SJ4: auto_acquire.resolve_oa strict Europe PMC JSON (C4), T13-T15. Fake http_get; no network."""
import json
import logging

import pytest

from tests import acq_support as A
from vera import acq_config, auto_acquire

DOI = "10.9999/strict.1027"
MARK = "SYNTHETICRECORDTEXT"


def _core(record_json: str) -> bytes:
    return ('{"hitCount":1,"resultList":{"result":[' + record_json + "]}}").encode()


def _rec(**over) -> dict:
    r = A.epmc_record(DOI, title=MARK)
    r.update(over)
    return r


def _body(rec: dict, extra: str) -> str:
    """Record JSON with `extra` key/value text appended inside the object (for duplicate keys json.dumps cannot make)."""
    return json.dumps(rec)[:-1] + ", " + extra + "}"


def _resolve(raw: bytes):
    web = A.Web({}, {}, overrides={"/rest/search": A.Resp(body=raw)})
    return auto_acquire.resolve_oa(DOI, A.cfg(), A.deps([], web))


def _stop_code(raw: bytes):
    with pytest.raises(auto_acquire._Stop) as e:
        _resolve(raw)
    return e.value.code


NOT_OA = acq_config.reason(A.cfg(), "not_oa")


def test_t13_duplicate_licence_nd_then_by_rejected():
    rec = _rec()
    del rec["license"]
    raw = _core(_body(rec, '"license": "cc by-nd", "license": "cc by"'))
    assert _stop_code(raw) == NOT_OA


@pytest.mark.parametrize("make,code", [
    (lambda: _body({k: v for k, v in _rec().items() if k != "isOpenAccess"}, '"isOpenAccess": "N", "isOpenAccess": "Y"'),
     "duplicate_key"),
    (lambda: _body({k: v for k, v in _rec().items() if k != "pubTypeList"},
                   '"pubTypeList": {"pubType": ["preprint"]}, "pubTypeList": {"pubType": ["research-article"]}'),
     "duplicate_key"),
    (lambda: _body({k: v for k, v in _rec().items() if k != "source"}, '"source": "PPR", "source": "MED"'),
     "duplicate_key"),
    (lambda: _body(_rec(), '"License": "cc by-nd"'), "duplicate_key"),
    (lambda: _body(_rec(), '"score": 1e999'), "non_finite"),
    (lambda: _body(_rec(), '"score": NaN'), "non_finite"),
    (lambda: _body(_rec(), '"licеnse": "cc by-nd"'), "non_ascii_key"),
])
def test_t14_strict_reject_is_not_oa_with_fixed_log(caplog, make, code):
    with caplog.at_level(logging.WARNING, logger="vera.autoacq"):
        assert _stop_code(_core(make())) == NOT_OA
    msgs = [r.getMessage() for r in caplog.records if "strict JSON" in r.getMessage()]
    assert msgs == [f"resolve_oa strict JSON reject ({code})"]
    assert not any(MARK in r.getMessage() or "cc by" in r.getMessage() for r in caplog.records)


def test_t15_clean_oa_record_unchanged():
    raw = _core(json.dumps(_rec()))
    out = _resolve(raw)
    assert out["pmcid"] == "PMC1234567" and out["licence"] == "cc by" and out["doi"] == DOI and out["title"] == MARK
    web = A.Web({DOI: _rec()}, {})  # the standard fake path gives the identical dict
    assert auto_acquire.resolve_oa(DOI, A.cfg(), A.deps([], web)) == out


def test_t15_genuine_non_oa_has_no_strict_log(caplog):
    with caplog.at_level(logging.WARNING, logger="vera.autoacq"):
        assert _stop_code(_core(json.dumps(_rec(isOpenAccess="N")))) == NOT_OA
    assert not any("strict JSON" in r.getMessage() for r in caplog.records)
