from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from vera.freshness import assess_freshness

NOW = datetime(2026, 10, 23, 12, 0, 0, tzinfo=timezone.utc)
CFG = {"freshness_max_age_hours": 24}


def one(chunk, cfg=CFG):
    return assess_freshness([chunk], now=NOW, cfg=cfg).chunks[0]


def iso(dt):
    return dt.isoformat()


def test_exactly_24h_fresh():
    assert one({"published": iso(NOW - timedelta(hours=24))}).status == "fresh"


def test_24h_plus_1s_stale():
    assert one({"published": iso(NOW - timedelta(hours=24, seconds=1))}).status == "stale"


def test_recently_ingested_old_publication_stale():
    c = one({"published": "2019-05-01", "retrieved_at": iso(NOW), "fetched_at": iso(NOW)})
    assert c.status == "stale" and c.acquired_at == iso(NOW)


def test_acquired_and_checked_never_make_fresh_or_give_date():
    c = one({"retrieved_at": iso(NOW), "last_checked_at": iso(NOW)})
    assert c.status == "unknown" and c.source_date is None and c.last_checked_at == iso(NOW)


def test_missing_unknown():
    assert one({}).status == "unknown"


def test_unparseable_unknown():
    assert one({"published": "not a date"}).status == "unknown"


def test_future_unknown_and_small_skew_ok():
    assert one({"published": iso(NOW + timedelta(days=2))}).status == "unknown"
    assert one({"published": iso(NOW + timedelta(seconds=60))}).status == "fresh"


def test_year_only_untrusted_default_and_trusted_option():
    assert one({"published": "2019"}).status == "unknown"
    assert one({"published": "2019"}, {"freshness_year_only_trusted": True}).status == "stale"


def test_chunk_date_preferred():
    c = one({"chunk_published": iso(NOW - timedelta(hours=1)), "published": "2019-01-01"})
    assert c.basis == "chunk" and c.status == "fresh"


def test_source_inherit_and_nested_object():
    c = one(SimpleNamespace(source={"updated": iso(NOW - timedelta(hours=2)), "published": "2019-01-01"}))
    assert c.basis == "source" and c.status == "fresh"


def test_bad_chunk_date_falls_back_to_source():
    c = one({"chunk_date": "garbage", "published": iso(NOW - timedelta(hours=3))})
    assert c.basis == "source" and c.status == "fresh"


def test_naive_and_z_suffix_treated_utc():
    assert one({"published": (NOW - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")}).status == "fresh"
    assert one({"published": (NOW - timedelta(hours=1)).replace(tzinfo=None).isoformat()}).status == "fresh"


def test_threshold_from_cfg():
    r = assess_freshness([{"published": iso(NOW - timedelta(hours=2))}], now=NOW, cfg={"freshness_max_age_hours": 1})
    assert r.chunks[0].status == "stale" and r.max_age_hours == 1
    assert assess_freshness([], now=NOW, cfg={}).max_age_hours == 24


def test_any_stale_or_unknown_logic():
    f = {"published": iso(NOW)}
    assert assess_freshness([f, f], now=NOW, cfg=CFG).any_stale_or_unknown is False
    assert assess_freshness([], now=NOW, cfg=CFG).any_stale_or_unknown is False
    r = assess_freshness([f, {}], now=NOW, cfg=CFG)
    assert r.any_stale_or_unknown and r.counts == {"fresh": 1, "stale": 0, "unknown": 1}
    assert assess_freshness([f, {"published": "2019-01-01"}], now=NOW, cfg=CFG).any_stale_or_unknown
