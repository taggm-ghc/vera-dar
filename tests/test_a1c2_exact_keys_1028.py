"""STRICT-C-1028 fix round 1 (A1c-2): exact-key mode for free-text-keyed objects; remaining untrusted sites."""
import json
from types import SimpleNamespace

import pytest

from tests.test_acq_search_profile import _prov
from vera import source_resolver as sr
from vera import strict_json as sj
from vera import search_providers as sp
from vera.m6 import judge as jg

AII = '{"The":[0],"the":[2],"café":[1],"α":[3]}'


def test_exact_mode_accepts_case_and_nonascii_words_default_rejects():
    assert sj.loads(AII, keys="exact") == {"The": [0], "the": [2], "café": [1], "α": [3]}
    with pytest.raises(sj.StrictJSONError):
        sj.loads(AII)  # default unchanged


def test_exact_mode_rejects_exact_duplicates_and_nan_nested():
    for bad in ('{"a":1,"a":2}', '{"x":{"the":[0],"the":[1]}}', '{"a": NaN}', '{"a": 1e999}'):
        with pytest.raises(sj.StrictJSONError):
            sj.loads(bad, keys="exact")
    assert sj.loads_or_none('{"a":1,"a":2}', keys="exact") is None
    with pytest.raises(sj.StrictJSONError):
        sj.loads("{}", keys="bogus")


def _oa(aii):
    return json.dumps({"results": [{"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/x", "title": "T title here",
                                    "publication_year": 2024, "publication_date": "2024-01-01", "type": "article",
                                    "primary_location": {"landing_page_url": "https://e.org/p", "is_oa": True},
                                    "abstract_inverted_index": aii, "authorships": []}]})


def test_openalex_parse_realistic_abstract_with_words_differing_by_case_and_nonascii():
    p = _prov([])
    aii = {"The": [0, 5], "the": [2], "café": [1], "α": [3], "study": [4]}
    out = p.parse(SimpleNamespace(text=_oa(aii), status_code=200), 5)
    assert len(out) == 1 and "café" in out[0]["snippet"]


def test_openalex_exact_duplicate_word_key_still_fails():
    body = _oa({"x": [0]}).replace('{"x": [0]}', '{"the": [0], "the": [1]}')
    assert body.count('"the"') == 2
    with pytest.raises(sp.ProviderResponseError):
        _prov([]).parse(SimpleNamespace(text=body, status_code=200), 5)


def _http(text):
    return lambda *a, **k: SimpleNamespace(status_code=200, text=text)


def test_source_resolver_sites_exact_for_openalex_default_for_s2():
    ok = json.dumps({"title": "A title of several words", "publication_date": "2024-01-01",
                     "abstract_inverted_index": {"The": [0], "the": [1], "café": [2]}})
    assert sr.lookup_doi("10.1/x", 1, http_get=_http(ok)) is not None
    assert sr._s2("10.1/x", 1, http_get=_http('{"abstract": "a"}')) == {"abstract": "a"}


def test_source_resolver_dup_and_nan_behave_like_malformed():
    for bad in ('{"abstract": "a", "abstract": "b"}', '{"abstract": NaN}', "not json"):
        assert sr._s2("10.1/x", 1, http_get=_http(bad)) == {}
    for bad in ('{"title": "t", "title": "u"}', '{"title": NaN}', "not json"):
        assert sr.lookup_doi("10.1/x", 1, http_get=_http(bad)) == sr.lookup_doi("10.1/x", 1, http_get=_http("not json"))


def test_judge_http_sites_reject_dup_and_nan():
    for bad in ('{"data": [], "data": [1]}', '{"data": NaN}'):
        with pytest.raises(ValueError):
            sj.loads(bad)
    j = jg.parse_json_loose  # judge._call / list_models use strict_json.loads (same loader, covered above)
    assert j('{"a": 1}') == {"a": 1}


def test_response_json_strict_and_double_fallback():
    r = lambda t: SimpleNamespace(text=t, json=lambda: {"fb": 1})
    assert sj.response_json(r('{"a": 1}')) == {"a": 1}
    assert sj.response_json(r("")) == {"fb": 1}
    for bad in ('{"data": [], "data": [1]}', '{"data": NaN}'):
        with pytest.raises(ValueError):
            sj.response_json(r(bad))
    assert sj.response_json(r(AII), keys="exact")["café"] == [1]
