"""R3NORM-FIX-1028 (D1): read-only existence lookup of canonical_sources keys, for the write-only admission store.

Kept out of vera/corpus_admission.py on purpose: that module is write-only by contract (no SELECT; guarded by
tests/test_item79_corpus_admission.py). This one runs on the READ-ONLY engine (VERA_DB_URL_RO) and only asks whether any of
the given keys is already a canonical_url, so a legacy spelling of a stored source is never inserted a second time."""
from __future__ import annotations

from vera.corpus_admission import SCHEMA

SQL_ANY_KEY = "SELECT canonical_source_id FROM {S}.canonical_sources WHERE canonical_url = ANY(:ks) LIMIT 1"


class RoKeyLookup:
    def __init__(self, engine_ro):
        self.engine = engine_ro

    def any_known(self, keys) -> bool:
        from sqlalchemy import text
        keys = [str(k) for k in dict.fromkeys(keys) if k]
        if not keys:
            return False
        with self.engine.connect() as c:
            return c.execute(text(SQL_ANY_KEY.replace("{S}", SCHEMA)), {"ks": keys}).first() is not None
