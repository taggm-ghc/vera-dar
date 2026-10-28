"""STRICTJSON-1027 SJ5 / T16b: scope_router._strict_loads after the strict_json refactor behaves exactly like the
HOTFIX-DUPKEY-1027 version. The reference below is a frozen copy of the hotfix helpers, taken from the accepted patch
work/hotfix-dupkey-1027/impl/hotfix-dupkey.patch (sha256 8e77701fff11a944d8d1a067c6757ec9dec3dd50f346cefb221f4b90bdd35244).
Fixtures: every parametrized reply in tests/test_hotfix_dupkey_1027.py (read from its source, not imported) plus the
edge cases named in plan v2 T16b. Offline, stdlib only."""
import ast
import json
import math
import re
import unicodedata
from pathlib import Path

import pytest

from vera import scope_router as sr

# ---------------------------------------------------------------- frozen hotfix reference (verbatim from the patch)
MAX_REPLY_CHARS = 16384
_DROP_CATS = frozenset({"Cf", "Cc", "Mn"})


class _ReplyRejected(ValueError):
    """A classifier reply with a duplicate key (after normalising) or a non-ASCII key."""


def _base(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", s) if unicodedata.category(ch) not in _DROP_CATS)


def _norm(s: str) -> str:
    """Keys and equality checks: no whitespace at all, case-folded."""
    return re.sub(r"\s+", "", _base(s)).casefold()


def _norm_ws(s: str) -> str:
    """Text scans: whitespace runs kept as one space, so merged words such as "categorical advice" do not match."""
    return re.sub(r"\s+", " ", _base(s)).casefold().strip()


def _pairs_hook(pairs):  # json applies it at every nesting level
    keys = [_norm(k) for k, _ in pairs]
    if len(set(keys)) != len(keys) or any(not k.isascii() for k, _ in pairs):
        raise _ReplyRejected
    return dict(pairs)


def _strict_loads(content):
    if not isinstance(content, str) or len(content) > MAX_REPLY_CHARS:
        return None
    try:
        return json.loads(content, object_pairs_hook=_pairs_hook)
    except (TypeError, ValueError, RecursionError):  # JSONDecodeError and _ReplyRejected are ValueError
        return None


# ---------------------------------------------------------------- fixtures
def _off(**extra):
    return json.dumps({"in_scope": False, "category": "off_topic", **extra})


def _hotfix_fixtures() -> list:
    src = (Path(__file__).resolve().parent / "test_hotfix_dupkey_1027.py").read_text(encoding="utf-8")
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "parametrize" and len(node.args) >= 2 \
                and isinstance(node.args[1], (ast.List, ast.Tuple)):
            for el in node.args[1].elts:
                items = el.elts if isinstance(el, ast.Tuple) else [el]
                for it in items:
                    try:
                        v = eval(compile(ast.Expression(it), "<fixture>", "eval"), {"json": json, "_off": _off})  # noqa: S307
                    except Exception:  # noqa: BLE001  (not a self-contained expression)
                        continue
                    out.append(json.dumps(v) if isinstance(v, dict) else v)
    return out


def _edge_fixtures() -> list:
    head = '{"in_scope":false,"category":"off_topic","reason":"'
    cap1 = head + "a" * (MAX_REPLY_CHARS + 1 - len(head) - 2) + '"}'
    cap0 = head + "a" * (MAX_REPLY_CHARS - len(head) - 2) + '"}'
    return [cap1, cap0, "NaN", '{"x":NaN}', "Infinity", "-Infinity", "1e999", '{"x":-1e999}', None, b"{}", 1, 1.5,
            "[" * 100000, "[" * 10000, '{"a":' * 3000, '{"a":' * 100000,
            '{"cat\u0000egory":1,"category":2}', '{"cat͏egory":1,"category":2}',
            '{"cat​egory":1,"category":2}', '{"cat\\u200begory":1,"category":2}', '{"a":1,"\\u0061":2}',
            '﻿{"a":1}', '{"\\ud800":1}', "1" * 5000, '{"x":{"a":1,"a":2}}', '{"Category":1,"category":2}',
            '{"in_scope":true,"reason":"ok","category":"in_scope"}', "", "  ", "[]", "{}", "null", '"s"']


FIXTURES = _hotfix_fixtures() + _edge_fixtures()


def _same(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


def test_fixture_harvest_is_not_vacuous():
    assert len(_hotfix_fixtures()) >= 20 and len(FIXTURES) >= 50


@pytest.mark.parametrize("i", range(len(FIXTURES)))
def test_t16b_strict_loads_equivalent_to_hotfix(i):
    x = FIXTURES[i]
    assert _same(sr._strict_loads(x), _strict_loads(x))


@pytest.mark.parametrize("s", ["harm\u0000ful", "harm͏ful", "harm​ful", "Harm  ful", "ﬁle", "Ｈarm",
                               "categorical advice", " a\tb ", "é"])
def test_t16b_normalisers_equivalent(s):
    assert sr._norm(s) == _norm(s) and sr._base(s) == _base(s) and sr._norm_ws(s) == _norm_ws(s)


def test_t16_shared_module_used_and_no_local_copies():
    src = Path(sr.__file__).read_text(encoding="utf-8")
    assert "def _pairs_hook" not in src and "def _base" not in src and "def _norm(" not in src
    assert "_DROP_CATS" not in src
    loads = [ln.strip() for ln in src.splitlines() if "json.loads(" in ln]
    # only the owner-written config file loader (pre-existing) and refusal_token's lenient, duplicate-preserving embedded-JSON decoder (v2)
    assert loads == ['raw = json.loads(Path(path).read_text(encoding="utf-8"))',
                     "stack.append(json.loads(s, object_pairs_hook=lambda pairs: pairs))"]  # v2: embedded-JSON decode
    assert sr.MAX_REPLY_CHARS == MAX_REPLY_CHARS


def test_t16_refusal_token_strict_decode_fails_closed_on_duplicates():  # v2: top-level duplicates are malformed, so refusal_token fails closed (not a pair walk)
    pairs = json.loads('{"category":"harmful","category":"off_topic"}', object_pairs_hook=lambda p: p)
    assert pairs == [("category", "harmful"), ("category", "off_topic")]
    assert sr.refusal_token('{"category":"harmful","category":"off_topic"}') == "harmful"
    assert 'json.loads(s, object_pairs_hook=lambda pairs: pairs)' in Path(sr.__file__).read_text(encoding="utf-8")
