"""STREAM-1031 T8: client disconnect cancels the stream ctx; in-flight work finishes. Offline."""
import asyncio
import threading

import main
from vera import progress, stream_ctx
from tests.test_stream_route_support import stream_env  # noqa: F401


def test_t8_generator_close_cancels_ctx_and_inflight_finishes():
    ctx = stream_ctx.StreamCtx(stall_s=300)
    sink = main._StreamSink(limit=8)
    sink.append("received", "done", {})
    release, finished = threading.Event(), threading.Event()

    def worker():  # stands for an in-flight evaluation that is allowed to finish
        release.wait(5)
        finished.set()
        sink.put_result(200, body={})

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    async def run():
        gen = main._stream_lines(sink, ctx, t, 0.05)
        first = await gen.__anext__()
        assert b'"received"' in first
        assert not ctx.cancelled
        await gen.aclose()  # what Starlette does when the client goes away
        return ctx.cancelled

    assert asyncio.run(run()) is True
    assert ctx.stop_reason() == stream_ctx.DISCONNECTED
    assert not finished.is_set()
    release.set()
    t.join(5)
    assert finished.is_set()  # in-flight finished after cancel


def test_completed_stream_does_not_cancel_ctx():
    ctx = stream_ctx.StreamCtx(stall_s=300)
    sink = main._StreamSink(limit=8)
    sink.put_result(200, body={"a": 1})

    async def run():
        return [x async for x in main._stream_lines(sink, ctx, None, 0.05)]

    out = asyncio.run(run())
    assert len(out) == 1 and not ctx.cancelled


def test_sink_never_drops_terminal_when_full():
    sink = main._StreamSink(limit=2)
    for _ in range(5):
        sink.append("search", "update", {})
    assert sink.items.qsize() == 2
    assert sink.append("done", "answered", {}) is True
    assert sink.append("done", "error", {}) is False  # second terminal ignored
