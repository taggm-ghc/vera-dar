"""Checks that the rule-to-test map is accurate and matches AGENTS.md."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))
from rule_coverage_map import RULE_TESTS  # noqa: E402


def _entries():
    return [(r, t) for r, ts in RULE_TESTS.items() for t in ts]


def _claimed_rules():
    """Rule numbers named by 'Rules N to M' in the AGENTS.md release-gate sentence."""
    text = (ROOT / "AGENTS.md").read_text()
    m = re.search(r"Rules (\d+) to (\d+) are [^\n]*release gate", text)
    assert m, "AGENTS.md has no 'Rules N to M ... release gate' sentence"
    return {str(i) for i in range(int(m.group(1)), int(m.group(2)) + 1)}


@pytest.mark.parametrize("rule,node", _entries())
def test_named_test_exists(rule, node):
    path, _, name = node.partition("::")
    name = name.split("[", 1)[0]  # parametrized ids allowed
    f = ROOT / path
    assert f.is_file(), f"rule {rule}: {path} missing"
    assert re.search(rf"^def {re.escape(name)}\b", f.read_text(), re.M), (
        f"rule {rule}: {name} not defined in {path}"
    )


def test_map_covers_exactly_the_rules_agents_md_claims():
    assert set(RULE_TESTS) == _claimed_rules()


def test_every_rule_has_a_test():
    assert all(RULE_TESTS[r] for r in RULE_TESTS)
