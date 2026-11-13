"""TEST-HARDEN-1113: non-vacuity of the CPU growth-ratio helper (a quadratic function must be rejected)."""
from tests import growth_support as gs


def _linear(text):
    return sum(1 for ch in text if ch == "a")


def _quadratic(text):
    n = 0
    for i in range(len(text)):
        for j in range(i, len(text), 7):
            n += 1
    return n


def _make(k):
    return "ab" * k


def test_linear_function_stays_under_the_bound():
    assert gs.growth_ratio(_linear, _make, 4000) < gs.RATIO_BOUND


def test_quadratic_function_is_rejected():
    assert gs.growth_ratio(_quadratic, _make, 250) > gs.RATIO_BOUND


def test_bigger_input_costs_more_cpu_than_smaller():
    ratio, small, big = gs.growth_stats(_linear, _make, 4000)
    assert big > small and ratio > 1.5
