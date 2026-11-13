"""TEST-HARDEN-1113 (P-1110-05): load-independent complexity check for tests.

An absolute wall-clock bound fails when the host is busy. This helper times a callable on the same input shape at size
n and GROWTH*n in CPU time (time.process_time) and returns the ratio. A linear function grows about GROWTH-fold on any
host; a quadratic one about GROWTH**2-fold. Same method as tests/test_gov_pd_contractor_1111.py (RELFIX-1112): the
small run is repeated until it costs at least MIN_SMALL_CPU, and the best of three interleaved runs is compared.

Caveat: process_time counts CPU of every thread in the process, so a spinning leftover thread would inflate it; the
waiters other tests leave behind are blocked, not spinning. Keep the n and GROWTH*n sizes below any input cap of the
function under test (assert that in the test), or the ratio is flat and proves nothing.
"""
import time

GROWTH = 8
RATIO_BOUND = 24      # 3x headroom over linear (8); a quadratic function gives about 64
MIN_SMALL_CPU = 0.005
CPU_BACKSTOP = 20.0   # CPU seconds for one run at full original size; a hang guard, not the real check


def _cpu(fn, arg, reps):
    t0 = time.process_time()
    for _ in range(reps):
        fn(arg)
    return time.process_time() - t0


def growth_stats(fn, make, n, growth=GROWTH):
    """(ratio, small_cpu, big_cpu) of fn(make(growth*n)) over fn(make(n)); interleaved best of 3, same repeat count."""
    small, big = make(n), make(growth * n)
    reps = 1
    while _cpu(fn, small, reps) < MIN_SMALL_CPU and reps < 65536:
        reps *= 2
    ts, tb = [], []
    for _ in range(3):
        ts.append(_cpu(fn, small, reps))
        tb.append(_cpu(fn, big, reps))
    s, b = min(ts), min(tb)
    return b / max(s, 1e-9), s, b


def _ref_loop():
    s = 0
    for i in range(1_000_000):
        s += i * i
    return s


def normalized_cost(fn, arg, reps=2):
    """CPU of fn(arg) divided by the CPU of a fixed pure-Python reference loop, each best of `reps`, interleaved.
    Host speed cancels out (a loaded or slow host inflates both), so the number is a machine-independent cost."""
    cs, rs = [], []
    for _ in range(reps):
        t0 = time.process_time()
        fn(arg)
        cs.append(time.process_time() - t0)
        t0 = time.process_time()
        _ref_loop()
        rs.append(time.process_time() - t0)
    return min(cs) / max(min(rs), 1e-9)


def growth_ratio(fn, make, n, growth=GROWTH):
    return growth_stats(fn, make, n, growth)[0]
