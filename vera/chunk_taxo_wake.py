"""CHUNK-ASSOC-1113 D5: the ingest hand-off to the chunk-taxonomy background loop. One function, `wake()`: process-local, O(1), never raises,
no database write, no embedding on the caller's thread. It only marks the loop to start its next pass early; the loop debounces
(`job.wake_min_interval_s`, default `interval_s`) and does nothing while the loop is not running (flag off). Kept in its own module so the
acquisition write path carries no reference to the loop itself (a test pins that the ingest store names none)."""
from __future__ import annotations


def wake() -> None:
    try:
        from vera import chunk_taxo_job  # noqa: PLC0415
        chunk_taxo_job.wake()
    except Exception:  # noqa: BLE001
        pass
