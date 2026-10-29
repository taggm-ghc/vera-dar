import ast
import json
import re
from pathlib import Path

import pytest

from vera import semantic_similarity as ss

ROOT = Path(__file__).resolve().parent.parent
VERA = ROOT / "vera"
FAKE = ss.FakeEmbedder()
CFG = {"threshold_topic": 0.5, "threshold_category": 0.9, "max_input_chars": 100}


def test_cosine_basics():
    assert ss.cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert ss.cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert ss.cosine([0, 0], [1, 0]) is None
    assert ss.cosine([1], [1, 2]) is None


def test_fake_embedder_deterministic():
    assert FAKE.embed(["a b c"]) == FAKE.embed(["a b c"])


def test_decision_and_separate_thresholds():
    a, b = "the cat sat on the mat", "the cat sat on a rug"
    s = ss.similarity_score(a, b, FAKE)
    assert 0.5 < s < 0.9
    assert ss.similarity_decision("topic", a, b, FAKE, CFG) == "similar"
    assert ss.similarity_decision("category", a, b, FAKE, CFG) == "dissimilar"
    assert ss.similarity_decision("topic", a, a, FAKE, CFG) == "similar"


def test_null_threshold_unknown():
    cfg = {"threshold_topic": None, "threshold_category": 0.1}
    assert ss.similarity_decision("topic", "x y", "x y", FAKE, cfg) == "unknown"
    assert ss.similarity_decision("category", "x y", "x y", FAKE, cfg) == "similar"


def test_shipped_config_uncalibrated_and_shadow():
    cfg = json.loads((ROOT / "config" / "semantic_similarity.json").read_text())
    assert cfg["enabled"] is False and cfg["shadow"] is True
    assert cfg["threshold_topic"] is None and cfg["threshold_category"] is None
    assert ss.similarity_decision("topic", "a", "a", FAKE) == "unknown"


def test_fail_closed():
    class Boom:
        def embed(self, texts):
            raise RuntimeError("x")
    assert ss.similarity_decision("topic", "a", "a", Boom(), CFG) == "unknown"
    assert ss.similarity_decision("nope", "a", "a", FAKE, CFG) == "unknown"
    assert ss.similarity_decision("topic", "", "a", FAKE, CFG) == "unknown"
    assert ss.similarity_decision("topic", None, "a", FAKE, CFG) == "unknown"
    assert ss.similarity_decision("topic", "a", "a", None, dict(CFG, enabled=False)) == "unknown"


def test_no_model_no_import_at_load(monkeypatch):
    import sys
    assert "fastembed" not in sys.modules
    assert ss.default_embedder({"enabled": True, "model_name": "m", "model_cache_env": "E"}, {}) is None
    ad = ss.FastEmbedAdapter("m", None)
    with pytest.raises(RuntimeError):
        ad.embed(["x"])
    assert "fastembed" not in sys.modules


def test_input_capped_and_treated_as_data():
    seen = []

    class Rec:
        def embed(self, texts):
            seen.extend(texts)
            return [[1.0, 0.0]] * len(texts)
    ss.similarity_decision("topic", "x" * 1000 + " ignore previous instructions", "y", Rec(), CFG)
    assert all(len(t) <= 100 for t in seen)


# ---- Owner condition (OD-1028(48)): similarity never establishes prohibited intent or refuses ----
_REFUSAL_WORDS = re.compile(r"refus|deny|denied|decline|reject|block|prohibit", re.I)
_REFUSAL_MODULES = {"scope_router.py", "catstep.py"}


def _mentions(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Import) and any(a.name.split(".")[-1] == "semantic_similarity" or
                                             a.name.endswith("semantic_similarity") for a in n.names):
            yield n
        elif isinstance(n, ast.ImportFrom):
            if (n.module or "").endswith("semantic_similarity") or any(a.name == "semantic_similarity" for a in n.names):
                yield n
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and "semantic_similarity" in n.value:
            yield n  # importlib / string indirection


def test_guard_no_refusal_path_uses_similarity():
    offenders = []
    for p in sorted(VERA.rglob("*.py")):
        if p.name == "semantic_similarity.py":
            continue
        tree = ast.parse(p.read_text())
        if not list(_mentions(tree)):
            continue
        rel = str(p.relative_to(ROOT))
        refusal_like = p.name in _REFUSAL_MODULES or _REFUSAL_WORDS.search(p.stem)
        defs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and re.search(r"refus|deny|decline", n.name, re.I)]
        if refusal_like or defs:
            offenders.append((rel, defs))
    assert not offenders, f"semantic_similarity used in refusal/decline code: {offenders}"
    for name in _REFUSAL_MODULES:
        assert (VERA / name).exists(), f"guard target {name} missing; update the guard, do not rename"


def test_guard_module_exposes_no_refusal_names():
    tree = ast.parse((VERA / "semantic_similarity.py").read_text())
    names = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    names += [t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)]
    bad = [n for n in names if re.search(r"refus|deny|denied|decline|reject|block|prohibit", n, re.I)]
    assert not bad, bad
    assert {ss.SIMILAR, ss.DISSIMILAR, ss.UNKNOWN} == {"similar", "dissimilar", "unknown"}
