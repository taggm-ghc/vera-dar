"""TOPICREG-1028: DB-backed topic registry (vera_vjay.topreg_topics, db/migrations/pending/011-DRAFT, NOT applied).
READ path (read_topic_rows, load_active_registry_data): SELECT only, in a READ ONLY transaction through the existing rw engine.
WRITE path (register_proposed_topic): off by default (registry_table.registration_enabled); inserts one runtime 'draft' row and
moves it to active or suspended. vera.micro_topic.load_registry delegates to the read path only, and only when
cfg["registry_table"]["read_from_db"] is boolean true (default off).
"Match" (duplicate) means SEMANTIC EQUIVALENCE or PARAPHRASE only (OD-1028(49)); narrower or broader non-equivalent topics are distinct."""
from __future__ import annotations

import functools
import logging
import re
import zlib
from pathlib import Path

class TopicRegistryError(RuntimeError):
    """The DB topic registry is enabled but unreadable or invalid (fail closed; deliberately not an OSError/ValueError,
    so key_for_provenance does not turn it into 'unassigned')."""


# TOPICREG-1028: the registry table (vera_vjay.topreg_topics, db/migrations/pending/011-DRAFT, NOT applied). The JSON file is
# the SEED and the FALLBACK. Settings live in cfg["registry_table"]; only explicit boolean true enables a toggle.
TOPIC_STATES = ("draft", "active", "suspended", "retired", "banned", "prohibited", "secrets")
# OD-1028(51)/(52): refused-topic states. Such rows are NEVER loaded as active and NEVER trigger acquisition (registry_data_from_rows
# builds entries from state == 'active' only). Only the owner/admin may move a row out of these states (DB trigger).
REFUSED_STATES = ("banned", "prohibited", "secrets")
CACHE_TTL_MAX = 3600  # reviewer LOW: registry_table.cache_ttl_seconds accepted range is 0..3600, else rejected at load
_CATEGORY_CODE = re.compile(r"[a-z][a-z_]{1,31}")
TOPIC_SQL_EXISTS = "SELECT to_regclass('vera_vjay.topreg_topics')"
TOPIC_SQL_ROWS = ("SELECT topic_key, version, label, field, anchor_text, parent_openalex_topic, parent_topic_key, origin, state "
                  "FROM vera_vjay.topreg_topics ORDER BY topic_row_id")
# R1 (both directions): the table owner is unrestricted by the guard trigger. (a) the owner must not be, or hold a USABLE
# (inherit_option OR set_option; PG16+ columns, ADMIN-only rows do not count) membership in, an app role; (b) the session
# (current_user and session_user) must not be able to reach the owner (pg_has_role MEMBER follows inheritance).
# D11: the group and login role names come from ONE config value, registry_table.app_roles (validated at load); these are the defaults
# (they equal the psql-variable defaults of the 011 draft and db/accounts/create_vera_accounts.sql). An -v override of the draft's
# app_*_login variables must be mirrored in config/auto_acquire.json.
DEFAULT_APP_ROLES = {"groups": {"wo": "vera_vjay_wo", "rw": "vera_vjay_rw", "ro": "vera_vjay_ro"},
                     "logins": {"wo": "vera_pipeline_wo", "rw": "vera_claude_code_rw", "ro": "vera_eval_ro"}}
_ROLE_NAME = re.compile(r"[a-z][a-z0-9_]{1,62}")
_ROLE_KINDS = ("wo", "rw", "ro")


def app_roles_config(settings: dict | None) -> dict:
    """Validated {"groups": {...}, "logins": {...}} from registry_table.app_roles (default DEFAULT_APP_ROLES when absent): exactly the
    kinds wo/rw/ro in each of groups and logins, well-formed role names, all six distinct. Anything else raises TopicRegistryError."""
    raw = (settings or {}).get("app_roles", DEFAULT_APP_ROLES)
    if not isinstance(raw, dict) or set(raw) != {"groups", "logins"}:
        raise TopicRegistryError("registry_table.app_roles must have exactly groups and logins")
    out, names = {}, []
    for part in ("groups", "logins"):
        d = raw[part]
        if not isinstance(d, dict) or set(d) != set(_ROLE_KINDS):
            raise TopicRegistryError(f"registry_table.app_roles.{part} must have exactly wo, rw and ro")
        for k in _ROLE_KINDS:
            v = d[k]
            if not isinstance(v, str) or not _ROLE_NAME.fullmatch(v):
                raise TopicRegistryError(f"registry_table.app_roles.{part}.{k} is not a valid role name")
            names.append(v)
        out[part] = {k: d[k] for k in _ROLE_KINDS}
    if len(set(names)) != 6:
        raise TopicRegistryError("registry_table.app_roles names must be six distinct roles")
    return out


def app_role_names(settings: dict | None = None) -> tuple:
    """The six role names (groups wo, rw, ro, then logins wo, rw, ro) from config, validated."""
    r = app_roles_config(settings)
    return tuple(r["groups"][k] for k in _ROLE_KINDS) + tuple(r["logins"][k] for k in _ROLE_KINDS)


TOPIC_SQL_OWNER = ("SELECT pg_catalog.pg_get_userbyid(c.relowner) AS owner, "
                   "(SELECT count(*) FROM pg_catalog.pg_auth_members m JOIN pg_catalog.pg_roles r ON r.oid = m.roleid "
                   "WHERE m.member = c.relowner AND (m.inherit_option OR m.set_option) "
                   "AND r.rolname = ANY (:roles)) AS app_memberships, "
                   "pg_catalog.pg_has_role(current_user, c.relowner, 'MEMBER') AS current_reaches_owner, "
                   "pg_catalog.pg_has_role(session_user, c.relowner, 'MEMBER') AS session_reaches_owner "
                   "FROM pg_catalog.pg_class c WHERE c.oid = to_regclass('vera_vjay.topreg_topics')")  # :roles = the six names from app_role_names()


TOPIC_APP_ROLES = app_role_names(None)
_TOPIC_KEY = re.compile(r"[a-z0-9][a-z0-9-]{1,63}")
_PRINTABLE = re.compile(r"[ -~]+")
_ROW_KEYS = {"topic_key", "version", "label", "field", "anchor_text", "parent_openalex_topic", "parent_topic_key", "origin", "state"}
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


def validate_registry_settings(settings: dict) -> None:
    from vera import topic_judge  # TOPICJUDGE-1028: a present judge block must be valid even while disabled
    try:
        topic_judge.validate_judge_settings(settings)
    except topic_judge.JudgeConfigError as exc:
        raise TopicRegistryError(str(exc)) from None
    """Fail closed on invalid registry_table settings (called at load and before any registration). cache_ttl_seconds, when
    present, must be a number in 0..3600; refused_state_map, when present, must be a valid mapping."""
    if "cache_ttl_seconds" in settings:
        v = settings["cache_ttl_seconds"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= CACHE_TTL_MAX:
            raise TopicRegistryError(f"registry_table.cache_ttl_seconds must be a number in 0..{CACHE_TTL_MAX}")
    app_roles_config(settings)
    if "refused_state_map" in settings:
        refused_state_mapping(settings)


def refusal_category_codes() -> frozenset:
    """Classifier category codes whose disposition is 'refuse' (vera.scope_router.CATEGORY_TABLE; the single source)."""
    from vera.scope_router import CATEGORY_TABLE
    return frozenset(c.code for c in CATEGORY_TABLE if c.disposition == "refuse")


def refused_state_mapping(settings: dict) -> dict:
    """The PROVISIONAL category code -> refused state mapping from config registry_table.refused_state_map, validated at load:
    a non-empty object whose keys are classifier categories with disposition 'refuse' and whose values are in REFUSED_STATES.
    `secrets` has NO code in the shipped map (admin-only) until a real secret-leak refusal code exists in the category table."""
    m = settings.get("refused_state_map")
    if not isinstance(m, dict) or not m:
        raise TopicRegistryError("registry_table.refused_state_map is missing or empty")
    known = refusal_category_codes()
    for code, st in m.items():
        if not isinstance(code, str) or not _CATEGORY_CODE.fullmatch(code) or code not in known or st not in REFUSED_STATES:
            raise TopicRegistryError("registry_table.refused_state_map has an invalid entry")
    return dict(m)


def refused_placeholder(category_code: str) -> dict:
    """The FIXED placeholder row content for a refusal category code (OD-1028(56)); mirrored by the DB's topreg_refused_chk."""
    if not isinstance(category_code, str) or not _CATEGORY_CODE.fullmatch(category_code):
        raise RegistrationRefused("category_code is not well-formed")
    return {"key": "refused-" + category_code.replace("_", "-"), "label": "refused " + category_code, "field": category_code,
            "anchor": "refused topic category " + category_code + ": fixed placeholder, no question text"}


def normalize_topic_key(text) -> str:
    """Q7 normalized key: NFKC, casefold, non-alphanumerics to single hyphens; must fit the table's key pattern."""
    import unicodedata
    s = re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKC", str(text or "")).casefold()).strip("-")
    if not _TOPIC_KEY.fullmatch(s):
        raise ValueError("text does not normalize to a valid topic key")
    return s


def read_topic_rows(engine=None, *, env=None, engine_factory=None, roles=None):
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
            role_names = TOPIC_APP_ROLES if roles is None else tuple(roles)
            owner = conn.execute(text(TOPIC_SQL_OWNER), {"roles": list(role_names)}).mappings().first()
            if not owner or owner["owner"] in role_names or owner["app_memberships"]:
                raise TopicRegistryError("topic registry table is owned by an app role (or its owner is a member of one)")
            if owner.get("current_reaches_owner") is not False or owner.get("session_reaches_owner") is not False:
                raise TopicRegistryError("the app session can reach the topic registry table owner (or this could not be verified)")
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
        pk = r["parent_topic_key"]
        if pk is not None and (not isinstance(pk, str) or not _TOPIC_KEY.fullmatch(pk) or pk == r["topic_key"]):
            raise TopicRegistryError(f"topic registry row {i} has an invalid parent topic key")
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
                "status": r["origin"], "anchor_text": r["anchor_text"], "parent_topic_key": r["parent_topic_key"]} for k, r in active.items()]
    return {"anchor_registry_version": "db:v%d" % max(r["version"] for r in rows), "entries": entries}


def load_active_registry_data(cfg, reader=None):
    """Registry data dict from the DB's active rows, or None when the table is absent (caller uses the JSON seed).
    Raises TopicRegistryError on any other problem (fail closed), including invalid registry_table settings."""
    validate_registry_settings(registry_table_settings(cfg))
    exists, rows = (reader or functools.partial(read_topic_rows, roles=app_role_names(registry_table_settings(cfg))))()
    if not exists:
        logging.getLogger("vera").warning("topic registry table absent: using the JSON seed")
        return None
    return registry_data_from_rows(rows)


# --- runtime registration (OD-1028(9); owner 2026-10-28). Whole path OFF by default (registry_table.registration_enabled). ---
TOPIC_SQL_KEY_EXISTS = "SELECT 1 FROM vera_vjay.topreg_topics WHERE topic_key = :k"
TOPIC_SQL_MATCHABLE = ("SELECT topic_key, anchor_text, state FROM vera_vjay.topreg_topics "
                       "WHERE state IN ('active', 'retired') ORDER BY topic_row_id")
TOPIC_SQL_MATCHABLE_JUDGE = ("SELECT topic_key, version, label, anchor_text, state, modified_at FROM vera_vjay.topreg_topics "
                             "WHERE state IN ('active', 'retired') ORDER BY topic_row_id")
TOPIC_SQL_JUDGE_DAY_COUNT = "SELECT count(*) FROM vera_vjay.topreg_judgments WHERE created_at > now() - interval '1 day'"
TOPIC_SQL_JUDGMENT_INSERT = ("INSERT INTO vera_vjay.topreg_judgments (draft_key, candidate_key, candidate_state, verdict, mode, model_id, latency_ms) "
                             "VALUES (:dk, :ck, :cs, :v, :m, :mid, :ms)")
# OD-1028(57) SHADOW mode: EXACT duplicate = same normalized topic_key, or the same normalized label (lower, whitespace-collapsed, as the
# unique label index/trigger) as an existing active, retired, suspended or draft topic.
TOPIC_SQL_DUPLICATE = ("SELECT topic_key FROM vera_vjay.topreg_topics WHERE topic_key = :k OR (state IN ('active', 'retired', 'suspended', 'draft') "
                       "AND lower(regexp_replace(btrim(label), '\\s+', ' ', 'g')) = :nl) ORDER BY topic_row_id LIMIT 1")
RESERVED_KEY_PREFIX = "refused-"  # OD-1028(57): reserved for the fixed refused-category placeholder rows; a proposal can never use it
TOPIC_SQL_PARENT_ACTIVE = "SELECT 1 FROM vera_vjay.topreg_topics WHERE topic_key = :p AND state = 'active'"
TOPIC_SQL_INSERT = ("INSERT INTO vera_vjay.topreg_topics (topic_key, label, field, anchor_text, parent_topic_key, origin, state) "
                    "VALUES (:k, :label, :field, :anchor, :parent, 'runtime', 'draft')")
TOPIC_SQL_INSERT_REFUSED = ("INSERT INTO vera_vjay.topreg_topics (topic_key, label, field, anchor_text, parent_topic_key, origin, state) "
                            "VALUES (:k, :label, :field, :anchor, NULL, 'runtime', :state)")
TOPIC_SQL_REFUSED_EXISTING = "SELECT state, field FROM vera_vjay.topreg_topics WHERE topic_key = :k AND version = 1"
TOPIC_SQL_ACTIVATE = ("UPDATE vera_vjay.topreg_topics SET state = 'active' "
                      "WHERE topic_key = :k AND version = 1 AND state = 'draft' AND origin = 'runtime'")
TOPIC_SQL_SUSPEND = ("UPDATE vera_vjay.topreg_topics SET state = 'suspended' "
                     "WHERE topic_key = :k AND version = 1 AND state = 'draft' AND origin = 'runtime'")
TOPIC_SQL_LOCK = "SELECT pg_advisory_xact_lock(:n)"
TOPIC_LOCK_ID = zlib.crc32(b"vera_vjay.topreg_topics")  # one lock id serialises every registration (D6)
SIMILARITY_MAX_DEFAULT = 0.50
MATCH_ORDER = ("active", "retired")  # refused rows are never text-matched (OD-1028(56)): category-level association only


class RegistrationRefused(RuntimeError):
    """register_proposed_topic rejected the proposal before writing anything."""


def lexical_similarity(a: str, b: str) -> float:
    """PROVISIONAL equivalence metric: stemmed-token Jaccard plus character-trigram Jaccard (the existing
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


def _sig(r) -> tuple:
    return (r["topic_key"], r.get("version"), r["state"], r.get("modified_at"))


def _judge_phase(engine, settings, key, label, anchor):
    """Runs BEFORE the advisory lock (no lock is held across a network call). Returns None when the judge is off, else
    {"mode", "sigs", "verdicts"}: a read-only snapshot of the candidate topics, the judge verdicts for EVERY candidate (unknown on any
    failure or an exhausted budget), and a best-effort log of the verdicts to topreg_judgments (own transaction; keys and states only, no
    question text; a logging failure never changes the outcome)."""
    from sqlalchemy import text
    from vera import topic_judge
    try:
        cfg = topic_judge.validate_judge_settings(settings)
    except topic_judge.JudgeConfigError as exc:
        raise TopicRegistryError(str(exc)) from None
    if cfg is None:
        return None
    log = logging.getLogger("vera")
    with engine.begin() as conn:
        rows = [dict(r) for r in conn.execute(text(TOPIC_SQL_MATCHABLE_JUDGE)).mappings().all() if r["topic_key"] != key]
        try:
            today = conn.execute(text(TOPIC_SQL_JUDGE_DAY_COUNT)).scalar()
        except Exception:  # noqa: BLE001 - class only; the table may be absent: the judge cannot run
            today = None
    over_day = isinstance(today, bool) or not isinstance(today, int) or today + len(rows) > cfg["max_judgments_per_day"]
    cands = [{"topic_key": r["topic_key"], "label": r["label"], "anchor_text": r["anchor_text"], "state": r["state"]} for r in rows]
    if over_day:
        log.warning("topic judge skipped (daily budget exhausted or the judgment log is unavailable)")
        verdicts = [topic_judge.Verdict(c["topic_key"], c["state"], "unknown") for c in cands]
    else:
        verdicts = topic_judge.judge_equivalence({"label": label, "anchor_text": anchor}, cands, settings=settings).verdicts
    try:
        if not over_day:
            with engine.begin() as conn:
                for v in verdicts:
                    conn.execute(text(TOPIC_SQL_JUDGMENT_INSERT), {"dk": key, "ck": v.candidate_key, "cs": v.candidate_state, "v": v.verdict,
                                                                 "m": cfg["mode"], "mid": v.model_id, "ms": v.latency_ms})
    except Exception as exc:  # noqa: BLE001
        log.warning("topic judgments could not be logged (%s)", type(exc).__name__)
    return {"mode": cfg["mode"], "sigs": {_sig(r) for r in rows}, "verdicts": verdicts}


def _judge_outcome(conn, plan, key, log):
    """ACTIVE mode, under the lock. (state, kind, matched_key) or None for 'stay draft'. Topics activated or changed since the judge
    call are NOT re-judged under the lock (no network call while it is held): the draft is kept (fail closed) and can be retried."""
    from sqlalchemy import text
    now = [dict(r) for r in conn.execute(text(TOPIC_SQL_MATCHABLE_JUDGE)).mappings().all() if r["topic_key"] != key]
    if {_sig(r) for r in now} != plan["sigs"]:
        log.warning("topic proposal kept as draft (candidate topics changed during the judge call)")
        return None
    matched = {}
    for v in plan["verdicts"]:
        if v.verdict == "yes":
            matched.setdefault(v.candidate_state, v.candidate_key)
    kind = next((k for k in MATCH_ORDER if k in matched), None)
    if kind:
        return "suspended", kind, matched[kind]
    if any(v.verdict != "no" for v in plan["verdicts"]):
        log.warning("topic proposal kept as draft (judge verdict unknown)")
        return None
    return "active", None, None


def register_proposed_topic(label, field, anchor_text, *, settings: dict, engine=None, similarity=lexical_similarity,
                            parent_key=None) -> dict:
    """Register a NEW topic (OD-1028(9), OD-1028(49); owner 2026-10-28). Written as 'draft', then in the SAME transaction under
    pg_advisory_xact_lock its anchor text is scored (>= similarity_max = a MATCH, meaning semantic equivalence or paraphrase
    ONLY) against every ACTIVE, RETIRED, BANNED, PROHIBITED and SECRETS topic (other drafts and suspended topics are ignored). A narrower topic inside an
    active topic, or a broader non-equivalent topic, is NOT a match: it is a distinct micro-topic and is activated.
      equivalent to an active topic  -> draft -> 'suspended' (logged: match kind 'active' + the matched key)
      equivalent only to a retired   -> draft -> 'suspended' (logged: match kind 'retired' + the matched key)
      equivalent to neither          -> draft -> 'active'
    `parent_key` (optional) is supplied by the caller when the proposal context identifies a parent topic (e.g. it arose under
    a known topic); it must be a well-formed key naming an ACTIVE topic (refused before any write otherwise; the DB trigger
    enforces it too) and is stored immutably as parent_topic_key. It is NEVER inferred from similarity; None = no parent.
    Acquisition caps and targets are unaffected.
    TOPICJUDGE-1028 (OD-1028(54)), only when registry_table.judge.enabled is true: a free Groq model judges equivalence against EVERY
    candidate BEFORE the lock (vera.topic_judge). SHADOW (default once enabled; OD-1028(57)): verdicts are logged to topreg_judgments and
    never decide; the draft goes to 'suspended' (nothing written) only if it is an EXACT duplicate (normalized key or label of an active,
    retired, suspended or draft topic), else to 'active'; the lexical metric is log-only. A key with the reserved 'refused-' prefix is refused. ACTIVE: any 'yes' -> suspended, all 'no' -> active, any 'unknown' (or candidates changed since the judge
    call) -> stays draft; the lexical metric is log-only. A verdict never causes a refusal.
      the check cannot run (the metric raises or returns a non-number) -> stays 'draft' (fail closed)
    A database error rolls the whole transaction back (nothing written). Rows are never deleted; logs carry no user text.
    `similarity(a, b)` is the pluggable metric (default: PROVISIONAL lexical). Returns {"key", "state"}."""
    if settings.get("registration_enabled") is not True:
        raise RegistrationRefused("runtime topic registration is disabled")
    proposal_state(settings)
    validate_registry_settings(settings)
    limit = similarity_max(settings)
    label = _clean("label", label, *_TOPIC_LIMITS["label"])
    field = _clean("field", field, *_TOPIC_LIMITS["field"])
    anchor = _clean("anchor_text", anchor_text, *_TOPIC_LIMITS["anchor_text"])
    try:
        key = normalize_topic_key(label)
    except ValueError:
        raise RegistrationRefused("label does not normalize to a valid topic key") from None
    if key.startswith(RESERVED_KEY_PREFIX):
        raise RegistrationRefused("topic key uses the reserved refused-category prefix")
    if parent_key is not None and (not isinstance(parent_key, str) or not _TOPIC_KEY.fullmatch(parent_key) or parent_key == key):
        raise RegistrationRefused("parent_key must be a well-formed topic key other than the new key")
    from sqlalchemy import text
    if engine is None:
        from vera import db
        engine = db.get_role_engine("rw")
    log = logging.getLogger("vera")
    try:
        plan = _judge_phase(engine, settings, key, label, anchor)  # TOPICJUDGE-1028: before the lock; None = judge off
        with engine.begin() as conn:
            conn.execute(text(TOPIC_SQL_LOCK), {"n": TOPIC_LOCK_ID})
            shadow = plan is not None and plan["mode"] == "shadow"
            if shadow:
                # OD-1028(57): only an EXACT duplicate is held back. The DB's unique key/label controls make a duplicate row unwritable, so
                # nothing is written: the outcome is 'suspended' (not recorded). Judge verdicts were already logged and never decide.
                nl = re.sub(r"\s+", " ", label.strip()).lower()
                dup = conn.execute(text(TOPIC_SQL_DUPLICATE), {"k": key, "nl": nl}).first()
                if dup is not None:
                    log.info("topic proposal suspended (shadow mode): exact duplicate; nothing written")
                    return {"key": key, "state": "suspended"}
            elif conn.execute(text(TOPIC_SQL_KEY_EXISTS), {"k": key}).first() is not None:
                raise RegistrationRefused("topic key already exists")
            if parent_key is not None and conn.execute(text(TOPIC_SQL_PARENT_ACTIVE), {"p": parent_key}).first() is None:
                raise RegistrationRefused("parent_key does not name an active topic")
            conn.execute(text(TOPIC_SQL_INSERT), {"k": key, "label": label, "field": field, "anchor": anchor, "parent": parent_key})
            if plan is not None and plan["mode"] == "active":
                # ACTIVE judge mode: the verdicts decide; the lexical stand-in is LOG-ONLY (its outcome is logged, never used).
                out = _judge_outcome(conn, plan, key, log)
                if out is None:
                    return {"key": key, "state": "draft"}
                state, kind, mkey = out
                if conn.execute(text(TOPIC_SQL_SUSPEND if kind else TOPIC_SQL_ACTIVATE), {"k": key}).rowcount != 1:
                    raise TopicRegistryError("topic transition affected an unexpected number of rows")
                try:
                    lex = [r["topic_key"] for r in conn.execute(text(TOPIC_SQL_MATCHABLE)).mappings().all()
                           if r["topic_key"] != key and similarity(anchor, r["anchor_text"]) >= limit]
                except Exception:  # noqa: BLE001
                    lex = None
                log.info("topic proposal %s by judge (%s); lexical stand-in (log only) matched %s", state,
                         "equivalent to a %s topic" % kind if kind else "no equivalent topic", "n/a" if lex is None else len(lex))
                return {"key": key, "state": state}
            if shadow:
                # SHADOW (OD-1028(57)): not an exact duplicate -> active whatever the judge said; the lexical metric is LOG-ONLY.
                try:
                    lex = [r["topic_key"] for r in conn.execute(text(TOPIC_SQL_MATCHABLE)).mappings().all()
                           if r["topic_key"] != key and similarity(anchor, r["anchor_text"]) >= limit]
                except Exception:  # noqa: BLE001
                    lex = None
                if conn.execute(text(TOPIC_SQL_ACTIVATE), {"k": key}).rowcount != 1:
                    raise TopicRegistryError("topic transition affected an unexpected number of rows")
                log.info("topic proposal activated (shadow mode; lexical stand-in, log only, matched %s)", "n/a" if lex is None else len(lex))
                return {"key": key, "state": "active"}
            # judge off: the lexical stand-in decides, as before
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
            kind = next((k for k in MATCH_ORDER if k in matched), None)
            sql, state = (TOPIC_SQL_SUSPEND, "suspended") if kind else (TOPIC_SQL_ACTIVATE, "active")
            if conn.execute(text(sql), {"k": key}).rowcount != 1:
                raise TopicRegistryError("topic transition affected an unexpected number of rows")
            if kind:
                log.info("topic proposal suspended: equivalent to a %s topic (matched key %s)", kind, matched[kind])
            else:
                log.info("topic proposal activated")
            return {"key": key, "state": state}
    except (RegistrationRefused, TopicRegistryError):
        raise
    except Exception as exc:  # noqa: BLE001 - class only, no driver text, no user text
        raise TopicRegistryError(f"topic registration failed ({type(exc).__name__})") from None


def ensure_refused_category(category_code, *, settings: dict, engine=None) -> dict:
    """OD-1028(56): make sure the FIXED placeholder row for a refusal category exists (one row per category code: key
    'refused-<code>', label 'refused <code>', anchor generated from the code only). Idempotent insert-if-absent under the advisory
    lock; it takes NO label, question text or any other text, and rejects anything but a configured refusal category code. The state
    comes from the PROVISIONAL config map registry_table.refused_state_map (harmful -> banned; advice, rule_change -> prohibited;
    `secrets` is admin-only until a real secret-leak refusal code exists). The row is never loaded as active, never triggers
    acquisition and is never text-matched. An existing row with another state or field is a TopicRegistryError (not overwritten).
    Returns {"key", "state", "created": bool}."""
    if settings.get("registration_enabled") is not True:
        raise RegistrationRefused("runtime topic registration is disabled")
    validate_registry_settings(settings)
    mapping = refused_state_mapping(settings)
    if not isinstance(category_code, str) or category_code not in mapping:
        raise RegistrationRefused("category_code is not a configured refusal category")
    state, ph = mapping[category_code], refused_placeholder(category_code)
    from sqlalchemy import text
    if engine is None:
        from vera import db
        engine = db.get_role_engine("rw")
    try:
        with engine.begin() as conn:
            conn.execute(text(TOPIC_SQL_LOCK), {"n": TOPIC_LOCK_ID})
            row = conn.execute(text(TOPIC_SQL_REFUSED_EXISTING), {"k": ph["key"]}).mappings().first()
            if row is not None:
                if row["state"] != state or row["field"] != category_code:
                    raise TopicRegistryError("refused category placeholder exists with a different state")
                return {"key": ph["key"], "state": state, "created": False}
            conn.execute(text(TOPIC_SQL_INSERT_REFUSED), {"k": ph["key"], "label": ph["label"], "field": ph["field"],
                                                          "anchor": ph["anchor"], "state": state})
        logging.getLogger("vera").info("refused category placeholder recorded: state %s", state)
        return {"key": ph["key"], "state": state, "created": True}
    except (RegistrationRefused, TopicRegistryError):
        raise
    except Exception as exc:  # noqa: BLE001 - class only
        raise TopicRegistryError(f"refused category registration failed ({type(exc).__name__})") from None


def scrutiny_flag(category_code, settings: dict) -> dict:
    """PURE (no I/O), CATEGORY LEVEL ONLY (OD-1028(56)): {"category_code": code or None, "state": mapped refused state or None} for a
    classifier category code named by an EXISTING refusal verdict, via registry_table.refused_state_map. There is no text matching
    and no similarity against refused rows (they hold only fixed placeholders). It is a request for STRICTER INTENT REVIEW by a future
    caller ("extra scrutiny, not automatic refusal", OD-1028(48)/(52)); it must NEVER by itself cause a refusal. Unmapped, invalid
    or non-string input, or an invalid map, gives None."""
    none = {"category_code": None, "state": None}
    if not isinstance(category_code, str):
        return none
    try:
        st = refused_state_mapping(settings).get(category_code)
    except TopicRegistryError:
        return none
    return none if st is None else {"category_code": category_code, "state": st}
