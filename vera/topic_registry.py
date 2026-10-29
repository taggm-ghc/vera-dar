"""TOPICREG-1028: DB-backed topic registry (vera_vjay.topic_registry, db/migrations/pending/011-DRAFT, NOT applied).
Read-only (SELECT in a READ ONLY transaction through the existing rw engine); app code never writes the registry here.
vera.micro_topic.load_registry delegates here only when cfg["registry_table"]["read_from_db"] is boolean true (default off)."""
from __future__ import annotations

import logging
import re
import zlib
from pathlib import Path

class TopicRegistryError(RuntimeError):
    """The DB topic registry is enabled but unreadable or invalid (fail closed; deliberately not an OSError/ValueError,
    so key_for_provenance does not turn it into 'unassigned')."""


# TOPICREG-1028: the registry table (vera_vjay.topic_registry, db/migrations/pending/011-DRAFT, NOT applied). The JSON file is
# the SEED and the FALLBACK. Settings live in cfg["registry_table"]; only explicit boolean true enables a toggle.
TOPIC_STATES = ("draft", "active", "suspended", "retired")
TOPIC_SQL_EXISTS = "SELECT to_regclass('vera_vjay.topic_registry')"
TOPIC_SQL_ROWS = ("SELECT topic_key, version, label, field, anchor_text, parent_openalex_topic, origin, state "
                  "FROM vera_vjay.topic_registry ORDER BY topic_row_id")
# R1: the table owner is unrestricted by the guard trigger, so the owner must not be (or be a direct member of) an app role.
TOPIC_APP_ROLES = ("vera_vjay_wo", "vera_vjay_rw", "vera_vjay_ro")
TOPIC_SQL_OWNER = ("SELECT pg_catalog.pg_get_userbyid(c.relowner) AS owner, "
                   "(SELECT count(*) FROM pg_catalog.pg_auth_members m JOIN pg_catalog.pg_roles r ON r.oid = m.roleid "
                   "WHERE m.member = c.relowner AND r.rolname IN ('vera_vjay_wo', 'vera_vjay_rw', 'vera_vjay_ro')) AS app_memberships "
                   "FROM pg_catalog.pg_class c WHERE c.oid = to_regclass('vera_vjay.topic_registry')")
_TOPIC_KEY = re.compile(r"[a-z0-9][a-z0-9-]{1,63}")
_PRINTABLE = re.compile(r"[ -~]+")
_ROW_KEYS = {"topic_key", "version", "label", "field", "anchor_text", "parent_openalex_topic", "origin", "state"}
_TOPIC_LIMITS = {"label": (3, 120), "field": (2, 60), "anchor_text": (20, 1200)}


def registry_table_settings(cfg: dict | None) -> dict:
    raw = (cfg or {}).get("registry_table")
    return raw if isinstance(raw, dict) else {}


def proposal_state(settings: dict) -> str:
    """State of an app-proposed topic. Owner decision 2026-10-28: always 'draft' (the DB trigger enforces it too)."""
    st = settings.get("proposal_state", "draft")
    if st != "draft":
        raise TopicRegistryError(f"proposal_state must be 'draft', got {st!r}")
    return st


def normalize_topic_key(text) -> str:
    """Q7 normalized key: NFKC, casefold, non-alphanumerics to single hyphens; must fit the table's key pattern."""
    import unicodedata
    s = re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKC", str(text or "")).casefold()).strip("-")
    if not _TOPIC_KEY.fullmatch(s):
        raise ValueError("text does not normalize to a valid topic key")
    return s


def read_topic_rows(engine=None, *, env=None, engine_factory=None):
    """(table_exists, rows). Reads in a READ ONLY transaction through the existing rw engine (SELECT only). A missing
    table gives (False, []); every other failure raises TopicRegistryError (class name only, never a URL or driver text)."""
    try:
        from sqlalchemy import text
        if engine is None:
            from vera import db
            kw = {} if engine_factory is None else {"engine_factory": engine_factory}
            engine = db.get_role_engine("rw", env=env, **kw)
        with engine.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            if conn.execute(text(TOPIC_SQL_EXISTS)).scalar() is None:
                return False, []
            owner = conn.execute(text(TOPIC_SQL_OWNER)).mappings().first()
            if not owner or owner["owner"] in TOPIC_APP_ROLES or owner["app_memberships"]:
                raise TopicRegistryError("topic registry table is owned by an app role (or its owner is a member of one)")
            return True, [dict(r) for r in conn.execute(text(TOPIC_SQL_ROWS)).mappings().all()]
    except TopicRegistryError:
        raise
    except Exception as exc:  # noqa: BLE001 - fail closed; only the class is reported
        raise TopicRegistryError(f"topic registry read failed ({type(exc).__name__})") from None


def registry_data_from_rows(rows) -> dict:
    """Validate DB rows (fail closed) and return Registry data from the ACTIVE rows only."""
    if not isinstance(rows, list) or not rows:
        raise TopicRegistryError("topic registry table exists but has no rows")
    active, seen = {}, set()
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or set(r) != _ROW_KEYS:
            raise TopicRegistryError(f"topic registry row {i} has unexpected columns")
        if not isinstance(r["topic_key"], str) or not _TOPIC_KEY.fullmatch(r["topic_key"]):
            raise TopicRegistryError(f"topic registry row {i} has an invalid key")
        if isinstance(r["version"], bool) or not isinstance(r["version"], int) or r["version"] < 1:
            raise TopicRegistryError(f"topic registry row {i} has an invalid version")
        if r["state"] not in TOPIC_STATES or r["origin"] not in ("seed", "runtime"):
            raise TopicRegistryError(f"topic registry row {i} has an unknown state or origin")
        for k, (lo, hi) in _TOPIC_LIMITS.items():
            v = r[k]
            if not isinstance(v, str) or not _PRINTABLE.fullmatch(v) or not lo <= len(v) <= hi or not v.strip():
                raise TopicRegistryError(f"topic registry row {i} has an invalid {k}")
        if not isinstance(r["parent_openalex_topic"], str) or not re.fullmatch(r"[A-Za-z0-9:_./-]{1,64}", r["parent_openalex_topic"]):
            raise TopicRegistryError(f"topic registry row {i} has an invalid parent topic")
        if (r["topic_key"], r["version"]) in seen:
            raise TopicRegistryError(f"duplicate (key, version) for {r['topic_key']!r}")
        seen.add((r["topic_key"], r["version"]))
        if r["state"] == "active":
            if r["topic_key"] in active:
                raise TopicRegistryError(f"more than one active row for {r['topic_key']!r}")
            active[r["topic_key"]] = r
    if not active:
        raise TopicRegistryError("topic registry has no active row")
    entries = [{"id": k, "label": r["label"], "parent_openalex_topic": r["parent_openalex_topic"], "field": r["field"],
                "status": r["origin"], "anchor_text": r["anchor_text"]} for k, r in active.items()]
    return {"anchor_registry_version": "db:v%d" % max(r["version"] for r in rows), "entries": entries}


def load_active_registry_data(cfg, reader=None):
    """Registry data dict from the DB's active rows, or None when the table is absent (caller uses the JSON seed).
    Raises TopicRegistryError on any other problem (fail closed)."""
    exists, rows = (reader or read_topic_rows)()
    if not exists:
        logging.getLogger("vera").warning("topic registry table absent: using the JSON seed")
        return None
    return registry_data_from_rows(rows)


# --- runtime registration (OD-1028(9); owner 2026-10-28). Whole path OFF by default (registry_table.registration_enabled). ---
TOPIC_SQL_KEY_EXISTS = "SELECT 1 FROM vera_vjay.topic_registry WHERE topic_key = :k"
TOPIC_SQL_MATCHABLE = ("SELECT topic_key, anchor_text, state FROM vera_vjay.topic_registry "
                       "WHERE state IN ('active', 'retired') ORDER BY topic_row_id")
TOPIC_SQL_INSERT = ("INSERT INTO vera_vjay.topic_registry (topic_key, label, field, anchor_text, origin, state) "
                    "VALUES (:k, :label, :field, :anchor, 'runtime', 'draft')")
TOPIC_SQL_ACTIVATE = ("UPDATE vera_vjay.topic_registry SET state = 'active' "
                      "WHERE topic_key = :k AND version = 1 AND state = 'draft' AND origin = 'runtime'")
TOPIC_SQL_SUSPEND = ("UPDATE vera_vjay.topic_registry SET state = 'suspended' "
                     "WHERE topic_key = :k AND version = 1 AND state = 'draft' AND origin = 'runtime'")
TOPIC_SQL_LOCK = "SELECT pg_advisory_xact_lock(:n)"
TOPIC_LOCK_ID = zlib.crc32(b"vera_vjay.topic_registry")  # one lock id serialises every registration (D6)
SIMILARITY_MAX_DEFAULT = 0.50


class RegistrationRefused(RuntimeError):
    """register_proposed_topic rejected the proposal before writing anything."""


def lexical_similarity(a: str, b: str) -> float:
    """PROVISIONAL dissimilarity metric: stemmed-token Jaccard plus character-trigram Jaccard (the existing
    scope_router.description_similarity). It is LEXICAL, NOT SEMANTIC: two texts about the same thing in different words
    score low. To be replaced by a semantic metric; the OD-1028(9)(b) threshold is uncalibrated."""
    from vera.scope_router import description_similarity
    return description_similarity(a, b)


def _clean(name: str, value, lo: int, hi: int) -> str:
    if not isinstance(value, str) or not _PRINTABLE.fullmatch(value) or not lo <= len(value) <= hi or not value.strip():
        raise RegistrationRefused(f"{name} must be {lo}-{hi} printable ASCII characters")
    return value.strip()


def similarity_max(settings: dict) -> float:
    v = settings.get("similarity_max", SIMILARITY_MAX_DEFAULT)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= 1:
        raise RegistrationRefused("similarity_max must be a number in (0, 1]")
    return float(v)


def register_proposed_topic(label, field, anchor_text, *, settings: dict, engine=None, similarity=lexical_similarity) -> dict:
    """Register a NEW topic (OD-1028(9); owner 2026-10-28). Written as 'draft', then in the SAME transaction under
    pg_advisory_xact_lock its anchor text is scored (>= similarity_max = a match) against every ACTIVE and RETIRED topic
    (other drafts and suspended topics are ignored):
      similar to an active topic  -> draft -> 'suspended' (logged: match kind 'active' + the matched key)
      similar only to a retired   -> draft -> 'suspended' (logged: match kind 'retired' + the matched key)
      similar to neither          -> draft -> 'active'
      the check cannot run (the metric raises or returns a non-number) -> stays 'draft' (fail closed)
    A database error rolls the whole transaction back (nothing written). Rows are never deleted; logs carry no user text.
    `similarity(a, b)` is the pluggable metric (default: PROVISIONAL lexical). Returns {"key", "state"}."""
    if settings.get("registration_enabled") is not True:
        raise RegistrationRefused("runtime topic registration is disabled")
    proposal_state(settings)
    limit = similarity_max(settings)
    label = _clean("label", label, *_TOPIC_LIMITS["label"])
    field = _clean("field", field, *_TOPIC_LIMITS["field"])
    anchor = _clean("anchor_text", anchor_text, *_TOPIC_LIMITS["anchor_text"])
    try:
        key = normalize_topic_key(label)
    except ValueError:
        raise RegistrationRefused("label does not normalize to a valid topic key") from None
    from sqlalchemy import text
    if engine is None:
        from vera import db
        engine = db.get_role_engine("rw")
    log = logging.getLogger("vera")
    try:
        with engine.begin() as conn:
            conn.execute(text(TOPIC_SQL_LOCK), {"n": TOPIC_LOCK_ID})
            if conn.execute(text(TOPIC_SQL_KEY_EXISTS), {"k": key}).first() is not None:
                raise RegistrationRefused("topic key already exists")
            conn.execute(text(TOPIC_SQL_INSERT), {"k": key, "label": label, "field": field, "anchor": anchor})
            matched = {}  # kind -> first matched key; 'active' wins over 'retired'
            for r in conn.execute(text(TOPIC_SQL_MATCHABLE)).mappings().all():
                if r["topic_key"] == key:
                    continue
                try:
                    score = similarity(anchor, r["anchor_text"])
                except Exception:  # noqa: BLE001 - the check cannot run: fail closed, keep the draft
                    score = None
                if isinstance(score, bool) or not isinstance(score, (int, float)) or score != score:
                    log.warning("topic proposal kept as draft (similarity check could not run)")
                    return {"key": key, "state": "draft"}
                if score >= limit:
                    matched.setdefault(r["state"], r["topic_key"])
            kind = "active" if "active" in matched else ("retired" if "retired" in matched else None)
            sql, state = (TOPIC_SQL_SUSPEND, "suspended") if kind else (TOPIC_SQL_ACTIVATE, "active")
            if conn.execute(text(sql), {"k": key}).rowcount != 1:
                raise TopicRegistryError("topic transition affected an unexpected number of rows")
            if kind:
                log.info("topic proposal suspended: similar to a %s topic (matched key %s)", kind, matched[kind])
            else:
                log.info("topic proposal activated")
            return {"key": key, "state": state}
    except (RegistrationRefused, TopicRegistryError):
        raise
    except Exception as exc:  # noqa: BLE001 - class only, no driver text, no user text
        raise TopicRegistryError(f"topic registration failed ({type(exc).__name__})") from None
