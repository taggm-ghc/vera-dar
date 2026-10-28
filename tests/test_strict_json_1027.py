"""STRICTJSON-1027 SJ1: vera/strict_json.py (U1-U9). Offline, stdlib only."""
import json
import math
import re
import unicodedata

import pytest

from vera import strict_json as sj
from vera.strict_json import StrictJSONError

# Frozen reference copy of the hotfix normaliser (HOTFIX-DUPKEY-1027 patch
# work/hotfix-dupkey-1027/impl/hotfix-dupkey.patch, sha256 8e77701fff11a944d8d1a067c6757ec9dec3dd50f346cefb221f4b90bdd35244)
_REF_DROP = frozenset({"Cf", "Cc", "Mn"})


def _ref_base(s):
    return "".join(ch for ch in unicodedata.normalize("NFKC", s) if unicodedata.category(ch) not in _REF_DROP)


def _ref_norm(s):
    return re.sub(r"\s+", "", _ref_base(s)).casefold()


def _code(text, **kw):
    with pytest.raises(StrictJSONError) as e:
        sj.loads(text, **kw)
    return e.value.code


# U1: exact duplicates at every level
@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"x":{"a":1,"a":2}}', '[{"a":1,"a":2}]'])
def test_u1_exact_duplicate_rejected(text):
    assert _code(text) == "duplicate_key"


# U2: ASCII duplicates after normalising (case, whitespace)
@pytest.mark.parametrize("text", ['{"License":1,"license":2}', '{"lic ense":1,"license":2}',
                                  '{"verdict":"unsupported","VERDICT":"supported"}'])
def test_u2_normalised_duplicate_rejected(text):
    assert _code(text) == "duplicate_key"


# U3: non-ASCII keys (labelled non-ASCII tests; rejected whatever the normaliser does)
@pytest.mark.parametrize("key", ["licеnse", "ｌicense", "lic​ense", "ﬁle"])
def test_u3_non_ascii_key_rejected(key):
    assert _code(json.dumps({key: 1, "other": 2})) == "non_ascii_key"


def test_u3_non_ascii_key_equal_after_normalising_is_still_rejected():
    assert _code(json.dumps({"lic​ense": 1, "license": 2})) in ("duplicate_key", "non_ascii_key")


def test_non_ascii_value_is_allowed():
    assert sj.loads('{"title":"café а"}') == {"title": "café а"}


# U4: non-finite numbers
@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999", '{"a":[1e999]}',
                                  '{"score":NaN}'])
def test_u4_non_finite_rejected_and_allowed_with_constants(text):
    assert _code(text) == "non_finite"
    got, ref = sj.loads(text, allow_constants=True), json.loads(text)

    def flat(x):
        return flat(x["a"][0]) if isinstance(x, dict) and "a" in x else flat(x["score"]) if isinstance(x, dict) \
            else x
    a, b = flat(got), flat(ref)
    assert (math.isnan(a) and math.isnan(b)) or a == b


@pytest.mark.parametrize("text", ["1.5e3", "1e308", "-0.0", "0.1", "12345678901234567890"])
def test_u4_finite_numbers_parse_as_json(text):
    assert repr(sj.loads(text)) == repr(json.loads(text))


# U5: never RecursionError / TypeError
@pytest.mark.parametrize("text,code", [("[" * 100000, "too_deep"), (None, "not_str"), (b"{}", "not_str"),
                                       (1, "not_str"), ('{"a":', "bad_json"), ("", "bad_json"),
                                       ("1" * 5000, "bad_json")])
def test_u5_failures_are_strict_errors(text, code):
    assert _code(text) == code
    assert sj.loads_or_none(text) is None


def test_u5_deep_nesting_raises_too_deep_or_bad_json():
    with pytest.raises(StrictJSONError):
        sj.loads('{"a":' * 3000)


def test_strict_error_is_value_error():
    assert issubclass(StrictJSONError, ValueError)


# U6: equivalence with json.loads on valid real reply shapes
VALID = [
    '{"claims":[{"id":0,"verdict":"supported","reason":"r"},{"id":1,"verdict":"partial","reason":"r"}]}',
    '{"claims":[{"id":0,"verdict":"unsupported","reason":"r","responsive":"no"}]}',
    '{"claims":[]}',
    '{"verdicts":[{"index":0,"verdict":"supported","issue":"x"},{"index":1,"verdict":"partial","issue":""}]}',
    '{"verdicts":[]}',
    '{"relations":[{"from":"E1","to":"E2","type":"refutes","confidence":0.8,"rationale":"x"}]}',
    '{"relations":[]}',
    '{"resultList":{"result":[{"doi":"10.1/x","pmcid":"PMC1234567","isOpenAccess":"Y","license":"cc by",'
    '"source":"MED","pubTypeList":{"pubType":["research-article"]},"title":"T","authorString":"A B.",'
    '"pubYear":"2021"}]}}',
    '{"resultList":{"result":[]}}',
    '{"hitCount":1,"resultList":{"result":[{"bookOrReportDetails":{"isbn13":"9780000000000"}}]}}',
    '{"in_scope":true,"reason":"ok","category":"in_scope"}',
    '{"in_scope":false,"reason":"x","category":"off_topic"}',
    '{"a":null,"b":true,"c":false,"d":-1,"e":2.5,"f":"\\u00e9"}',
    '[1,2,3]', '"s"', "3", "null", "true",
    '{"nested":{"deep":{"deeper":[{"k":1},{"k":2}]}}}',
    '{"a":1,"b":{"a":2}}',
    '{"key with space":1}',
    '  {"padded": 1}  ',
]


@pytest.mark.parametrize("text", VALID)
def test_u6_valid_equivalent_to_json(text):
    assert sj.loads(text) == json.loads(text)
    assert sj.loads_or_none(text) == json.loads(text)


# U7: max_chars
def test_u7_max_chars():
    body = '{"a":"' + "x" * 10 + '"}'
    assert sj.loads(body, max_chars=len(body)) == {"a": "x" * 10}
    assert _code(body, max_chars=len(body) - 1) == "oversize"
    bad = "{" * 20
    assert _code(bad, max_chars=len(bad) - 1) == "oversize"  # checked before parsing
    assert sj.loads('"' + "y" * 200000 + '"', max_chars=None) == "y" * 200000


# U8: normaliser equal to the hotfix reference
NORM_INPUTS = ["harm\u0000ful", "harm͏ful", "harm​ful", "Harm ful", "ﬁle", "Ｈarm",
               "a\tb\nc", "ADVICE", "rule_change", "é", "é"]


@pytest.mark.parametrize("s", NORM_INPUTS)
def test_u8_normaliser_matches_hotfix(s):
    assert sj.norm_key(s) == _ref_norm(s) and sj.base_norm(s) == _ref_base(s)


@pytest.mark.parametrize("s", NORM_INPUTS[:4])
def test_u8_harmful_spellings(s):
    assert sj.norm_key(s) == "harmful"


# U9: fixed messages only
FAILS = ['{"secretkey":1,"secretkey":2}', '{"Secret":1,"secret":2}', '{"sеcret":1}', '{"secret":NaN}',
         '{"secret":1e999}', "[" * 100000, '{"secret":', None]


@pytest.mark.parametrize("text", FAILS)
def test_u9_message_is_fixed_code(text):
    with pytest.raises(StrictJSONError) as e:
        sj.loads(text, max_chars=None)
    assert str(e.value) == e.value.code and e.value.code in sj.CODES
    assert "secret" not in str(e.value).lower() and "secret" not in repr(e.value).lower()
    assert e.value.__cause__ is None
