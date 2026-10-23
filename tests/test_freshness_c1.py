"""FRESH-C1, amended FRESH-AMEND-1023: corpus_added_at drives the 24 h trigger; source_authored_at only the older/recent label."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from vera.freshness import assess_freshness

NOW = datetime(2026, 10, 23, 12, 0, 0, tzinfo=timezone.utc)
CFG = {"freshness_max_age_hours": 24}
TRUST = {"freshness_max_age_hours": 24, "source_authored_partial_dates_trusted": True}


def one(chunk, cfg=CFG, now=NOW):
    return assess_freshness([chunk], now=now, cfg=cfg).chunks[0]


def iso(dt):
    return dt.isoformat()


def test_exactly_24h_fresh():
    assert one({"corpus_added_at": iso(NOW - timedelta(hours=24))}).status == "fresh"


def test_24h_plus_1s_stale():
    assert one({"corpus_added_at": iso(NOW - timedelta(hours=24, seconds=1))}).status == "stale"


def test_recently_added_old_publication_fresh_trigger_but_older_label():
    # was: test_recently_ingested_old_publication_stale. Owner clarification: the trigger uses the corpus-added time,
    # so a newly added 2019 paper does NOT trigger, but its authoring date labels it older (never recent).
    c = one({"published": "2019-05-01", "corpus_added_at": iso(NOW - timedelta(hours=1))})
    assert c.status == "fresh" and c.authored_recency == "older"
    assert c.source_authored_at == "2019-05-01" and c.authored_precision == "day"
    assert c.corpus_added_at == NOW - timedelta(hours=1)


def test_old_corpus_stamp_recent_publication_is_stale_but_recent():
    c = one({"published": iso(NOW - timedelta(hours=2)), "corpus_added_at": iso(NOW - timedelta(hours=30))})
    assert c.status == "stale" and c.authored_recency == "recent"


def test_authored_date_never_drives_trigger_and_checked_never_gives_date():
    # was: test_acquired_and_checked_never_make_fresh_or_give_date
    c = one({"published": iso(NOW), "last_checked_at": iso(NOW)})
    assert c.status == "unknown" and c.corpus_added_at is None and c.last_checked_at == iso(NOW)
    assert c.authored_recency == "recent"


def test_missing_unknown():
    c = one({})
    assert c.status == "unknown" and c.basis == "unknown" and c.authored_recency == "unknown"


def test_unparseable_unknown():
    assert one({"corpus_added_at": "not a date"}).status == "unknown"
    assert one({"published": "not a date", "corpus_added_at": iso(NOW)}).authored_recency == "unknown"


def test_future_unknown_and_small_skew_ok():
    assert one({"corpus_added_at": iso(NOW + timedelta(days=2))}).status == "unknown"
    assert one({"corpus_added_at": iso(NOW + timedelta(seconds=60))}).status == "fresh"
    assert one({"published": iso(NOW + timedelta(days=2))}).authored_recency == "unknown"


def test_partial_or_date_only_corpus_stamp_is_invalid_and_listed_for_backfill():
    # owner refinement: an imprecise corpus_added_at counts as invalid -> unknown + revised to now()
    for v in ("2026", "2026-09", "2026-10-23"):
        ch = {"corpus_added_at": v}
        c = one(ch)
        assert c.status == "unknown" and c.backfill == [ch], v
    ok = {"corpus_added_at": iso(NOW - timedelta(hours=1))}
    assert one(ok).backfill == []


def test_partial_authored_date_never_recent():
    # was: test_year_only_untrusted_default_and_trusted_option (now applies to authoring-date labelling only)
    assert one({"published": "2019"}).authored_recency == "unknown"
    assert one({"published": "2019"}, TRUST).authored_recency == "older"
    assert one({"published": "2019"}, TRUST).authored_precision == "year"
    # FRESH-VERIFY-1023 F1: within 24 h after a period ends, and before it ends, a partial date is never recent
    assert one({"published": "2026"}, TRUST, datetime(2027, 1, 1, 0, 30, tzinfo=timezone.utc)).authored_recency == "unknown"
    assert one({"published": "2026-09"}, TRUST, datetime(2026, 10, 1, 10, tzinfo=timezone.utc)).authored_recency == "unknown"
    assert one({"published": "2026"}, TRUST, datetime(2026, 12, 31, 23, 56, tzinfo=timezone.utc)).authored_recency == "unknown"
    assert one({"published": "2026-09"}, TRUST).authored_precision == "month"
    assert one({"published": "2026-09"}, TRUST).authored_recency == "older"


def test_chunk_stamps_preferred_oldest_decides():
    # was: test_chunk_date_preferred
    ch = {"corpus_chunks": [{"corpus_added_at": iso(NOW - timedelta(hours=1)), "basis": "chunk"},
                            {"corpus_added_at": iso(NOW - timedelta(hours=30)), "basis": "chunk"}],
          "corpus_added_at": iso(NOW)}
    c = one(ch)
    assert c.basis == "chunk" and c.status == "stale"
    c2 = one({"chunk_published": iso(NOW - timedelta(hours=1)), "published": "2019-01-01",
              "corpus_added_at": iso(NOW)})
    assert c2.authored_basis == "chunk" and c2.authored_recency == "recent"


def test_document_inherit_and_nested_object():
    # was: test_source_inherit_and_nested_object
    c = one(SimpleNamespace(source={"corpus_added_at": iso(NOW - timedelta(hours=2)), "updated": iso(NOW - timedelta(hours=2)),
                                    "published": "2019-01-01"}))
    assert c.basis == "document" and c.status == "fresh"
    assert c.authored_basis == "source" and c.authored_recency == "recent"  # update before publication


def test_bad_chunk_stamp_makes_unknown_and_is_backfill_target():
    # was: test_bad_chunk_date_falls_back_to_source (an invalid chunk stamp is now repaired, not skipped)
    bad = {"corpus_added_at": "garbage", "basis": "chunk"}
    good = {"corpus_added_at": iso(NOW - timedelta(hours=3)), "basis": "chunk"}
    c = one({"corpus_chunks": [good, bad]})
    assert c.status == "unknown" and c.backfill == [bad]
    c2 = one({"chunk_date": "garbage", "published": iso(NOW - timedelta(hours=3)), "corpus_added_at": iso(NOW)})
    assert c2.authored_basis == "source" and c2.authored_recency == "recent"


def test_naive_and_z_suffix_treated_utc():
    assert one({"corpus_added_at": (NOW - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")}).status == "fresh"
    assert one({"corpus_added_at": (NOW - timedelta(hours=1)).replace(tzinfo=None).isoformat()}).status == "fresh"


def test_threshold_from_cfg():
    r = assess_freshness([{"corpus_added_at": iso(NOW - timedelta(hours=2))}], now=NOW, cfg={"freshness_max_age_hours": 1})
    assert r.chunks[0].status == "stale" and r.max_age_hours == 1
    assert assess_freshness([], now=NOW, cfg={}).max_age_hours == 24


def test_any_stale_or_unknown_logic():
    f = {"corpus_added_at": iso(NOW)}
    assert assess_freshness([f, f], now=NOW, cfg=CFG).any_stale_or_unknown is False
    assert assess_freshness([], now=NOW, cfg=CFG).any_stale_or_unknown is False
    r = assess_freshness([f, {}], now=NOW, cfg=CFG)
    assert r.any_stale_or_unknown and r.counts == {"fresh": 1, "stale": 0, "unknown": 1}
    assert assess_freshness([f, {"corpus_added_at": iso(NOW - timedelta(days=3))}], now=NOW, cfg=CFG).any_stale_or_unknown
    # an old authoring date never makes a freshly added chunk stale
    assert not assess_freshness([dict(f, published="2019-01-01")], now=NOW, cfg=CFG).any_stale_or_unknown
