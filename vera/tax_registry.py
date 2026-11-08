"""TAXO-EXTENSIBLE-1108: extensible taxonomy registry on the migration-022 tables (vera_vjay.tax_anchors, tax_lineage,
tax_entity_events and the views tax_registry_active_v / tax_registry_eval_v). INERT: nothing imports this module from a request
path; every write path is behind a flag in config/tax_registry.json (ships false) AND refuses unless to_regclass('vera_vjay.tax_anchors')
is not NULL (022 applied). All SQL targets vera_vjay.tax_* only; SELECT, INSERT and UPDATE of state only; no DELETE, no DDL. Session
settings (vera.tax_actor, vera.tax_batch) are set only with set_config(name, value, true), i.e. transaction-local.
No taxonomy name is written in this file: names come from the induction manifest, the topreg registry or the database."""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "tax_registry.json"
FLAGS = ("read_from_tax", "write_proposals", "approve_enabled", "sync_enabled")

_SLUG = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
_TERM = re.compile(r"[a-z0-9][a-z0-9 -]{0,47}")
_ACTOR = re.compile(r"owner:[a-z0-9_.-]{1,60}")
_SHA = re.compile(r"[0-9a-f]{64}")
_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_ANCHOR = re.compile(r"[ -~]+")


class TaxRegistryError(RuntimeError):
    """Fail closed; messages carry a class name or a count, never a URL, credential or source text."""


class TaxRefused(TaxRegistryError):
    """A write path was refused (flag off, migration 022 absent, or a precondition failed)."""


# --------------------------------------------------------------------------------------------------------------------- SQL
SQL_PRESENT = "SELECT to_regclass('vera_vjay.tax_anchors')"
SQL_SET_CONFIG = "SELECT set_config(:n, :v, true)"
# Every SQL statement is a complete literal (no f-string assembly), so tests/sql_contract_support.py sees every table and column.
SQL_ROWS = {
    "active": 'SELECT kind, entity_id, slug, label, description, entity_state, anchor_id, anchor_text, core_terms, anchor_sha256, anchor_state, parent_slugs, domain_slugs, domain_labels, includes_proposed FROM vera_vjay.tax_registry_active_v ORDER BY kind DESC, entity_id',
    "proposed": 'SELECT kind, entity_id, slug, label, description, entity_state, anchor_id, anchor_text, core_terms, anchor_sha256, anchor_state, parent_slugs, domain_slugs, domain_labels, includes_proposed FROM vera_vjay.tax_registry_eval_v ORDER BY kind DESC, entity_id',
}
KINDS = {"domains": ("tax_domains", "domain_id"), "topics": ("tax_topics", "topic_id"), "subtopics": ("tax_subtopics", "subtopic_id")}
SQL_EXISTING = {
    'domains': 'SELECT slug, label, state FROM vera_vjay.tax_domains',
    'topics': 'SELECT slug, label, state FROM vera_vjay.tax_topics',
    'subtopics': 'SELECT slug, label, state FROM vera_vjay.tax_subtopics',
}
SQL_INSERT_ENTITY = {
    'domains': "INSERT INTO vera_vjay.tax_domains (slug, label, origin, origin_ref, state) VALUES (:slug, :label, 'induction', :oref, 'proposed') ON CONFLICT (slug) DO NOTHING",
    'topics': "INSERT INTO vera_vjay.tax_topics (slug, label, origin, origin_ref, state) VALUES (:slug, :label, 'induction', :oref, 'proposed') ON CONFLICT (slug) DO NOTHING",
    'subtopics': "INSERT INTO vera_vjay.tax_subtopics (slug, label, origin, origin_ref, state) VALUES (:slug, :label, 'induction', :oref, 'proposed') ON CONFLICT (slug) DO NOTHING",
}
SQL_INSERT_LINK = {
    'domain_topic': "INSERT INTO vera_vjay.tax_domain_topic (domain_id, topic_id, assigned_via, classifier_name, classifier_version) SELECT d.domain_id, t.topic_id, 'classifier', :cname, :cver FROM vera_vjay.tax_domains d, vera_vjay.tax_topics t WHERE d.slug = :a AND t.slug = :b ON CONFLICT DO NOTHING",
    'topic_subtopic': "INSERT INTO vera_vjay.tax_topic_subtopic (topic_id, subtopic_id, assigned_via, classifier_name, classifier_version) SELECT t.topic_id, s.subtopic_id, 'classifier', :cname, :cver FROM vera_vjay.tax_topics t, vera_vjay.tax_subtopics s WHERE t.slug = :a AND s.slug = :b ON CONFLICT DO NOTHING",
}
SQL_INSERT_ANCHOR = {
    'topics': "INSERT INTO vera_vjay.tax_anchors (topic_id, anchor_text, core_terms, origin, origin_ref, state) SELECT topic_id, :text, CAST(:terms AS text[]), 'induction', :oref, 'proposed' FROM vera_vjay.tax_topics WHERE slug = :slug ON CONFLICT DO NOTHING",
    'subtopics': "INSERT INTO vera_vjay.tax_anchors (subtopic_id, anchor_text, core_terms, origin, origin_ref, state) SELECT subtopic_id, :text, CAST(:terms AS text[]), 'induction', :oref, 'proposed' FROM vera_vjay.tax_subtopics WHERE slug = :slug ON CONFLICT DO NOTHING",
}
# approval: the stored proposed rows (entity + the newest proposed anchor's content hash, computed by the database)
SQL_STORED = {
    'domains': 'SELECT slug, label, state, origin, NULL AS anchor_sha FROM vera_vjay.tax_domains WHERE slug = ANY (CAST(:slugs AS text[]))',
    'topics': "SELECT x.slug, x.label, x.state, x.origin, (SELECT a.content_sha256 FROM vera_vjay.tax_anchors a WHERE a.topic_id = x.topic_id AND a.state = 'proposed' ORDER BY a.anchor_id DESC LIMIT 1) AS anchor_sha FROM vera_vjay.tax_topics x WHERE x.slug = ANY (CAST(:slugs AS text[]))",
    'subtopics': "SELECT x.slug, x.label, x.state, x.origin, (SELECT a.content_sha256 FROM vera_vjay.tax_anchors a WHERE a.subtopic_id = x.subtopic_id AND a.state = 'proposed' ORDER BY a.anchor_id DESC LIMIT 1) AS anchor_sha FROM vera_vjay.tax_subtopics x WHERE x.slug = ANY (CAST(:slugs AS text[]))",
}
SQL_DECIDE_ENTITY = {
    'domains': "UPDATE vera_vjay.tax_domains SET state = :ns WHERE slug = :slug AND state = 'proposed'",
    'topics': "UPDATE vera_vjay.tax_topics SET state = :ns WHERE slug = :slug AND state = 'proposed'",
    'subtopics': "UPDATE vera_vjay.tax_subtopics SET state = :ns WHERE slug = :slug AND state = 'proposed'",
}
SQL_DECIDE_ANCHOR = {
    'topics': "UPDATE vera_vjay.tax_anchors SET state = :ns WHERE state = 'proposed' AND content_sha256 = :sha AND topic_id = (SELECT topic_id FROM vera_vjay.tax_topics WHERE slug = :slug)",
    'subtopics': "UPDATE vera_vjay.tax_anchors SET state = :ns WHERE state = 'proposed' AND content_sha256 = :sha AND subtopic_id = (SELECT subtopic_id FROM vera_vjay.tax_subtopics WHERE slug = :slug)",
}
# sync: ACTIVE entities that came from the topreg registry, with their active anchor hash
SQL_SYNC_STATE = ("SELECT 'topics' AS kind, x.slug, x.origin_ref, a.content_sha256 AS anchor_sha FROM vera_vjay.tax_topics x LEFT JOIN vera_vjay.tax_anchors a ON a.topic_id = x.topic_id AND a.state = 'active' WHERE x.state = 'active' AND x.origin_ref LIKE 'topreg:%' "
                  "UNION ALL SELECT 'subtopics', y.slug, y.origin_ref, b.content_sha256 FROM vera_vjay.tax_subtopics y LEFT JOIN vera_vjay.tax_anchors b ON b.subtopic_id = y.subtopic_id AND b.state = 'active' WHERE y.state = 'active' AND y.origin_ref LIKE 'topreg:%'")
SQL_RETIRE_ANCHOR = {
    'topics': "UPDATE vera_vjay.tax_anchors SET state = 'retired' WHERE state = 'active' AND topic_id = (SELECT topic_id FROM vera_vjay.tax_topics WHERE slug = :slug)",
    'subtopics': "UPDATE vera_vjay.tax_anchors SET state = 'retired' WHERE state = 'active' AND subtopic_id = (SELECT subtopic_id FROM vera_vjay.tax_subtopics WHERE slug = :slug)",
}
SQL_SEED_ANCHOR = {
    'topics': "INSERT INTO vera_vjay.tax_anchors (topic_id, anchor_text, core_terms, origin, origin_ref, state) SELECT topic_id, :text, NULL, 'registry', :oref, 'active' FROM vera_vjay.tax_topics WHERE slug = :slug AND state = 'active'",
    'subtopics': "INSERT INTO vera_vjay.tax_anchors (subtopic_id, anchor_text, core_terms, origin, origin_ref, state) SELECT subtopic_id, :text, NULL, 'registry', :oref, 'active' FROM vera_vjay.tax_subtopics WHERE slug = :slug AND state = 'active'",
}


# ----------------------------------------------------------------------------------------------------------------- config
def load_config(path=None) -> dict:
    p = Path(path) if path else CONFIG_PATH
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TaxRegistryError(f"tax registry config unreadable ({type(exc).__name__})") from None
    if not isinstance(raw, dict):
        raise TaxRegistryError("tax registry config must be an object")
    for k in raw:
        if k not in FLAGS and not str(k).startswith("_"):
            raise TaxRegistryError(f"tax registry config has an unknown key {k!r}")
    out = {}
    for f in FLAGS:
        if f not in raw or raw[f] is not True and raw[f] is not False:
            raise TaxRegistryError(f"tax registry config {f} must be literally true or false")
        out[f] = raw[f]
    return out


def _flag(cfg, name) -> bool:
    return isinstance(cfg, dict) and cfg.get(name) is True


# ------------------------------------------------------------------------------------------------------- pure helpers
def anchor_sha256(anchor_text, core_terms) -> str:
    """The database's content_sha256: sha256(utf8(anchor_text + newline + '|'.join(core_terms or []))) as lowercase hex."""
    body = str(anchor_text) + "\n" + "|".join(core_terms or [])
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def core_terms_from(keyphrases):
    """Lowercased keyphrases that fit the DB pattern, de-duplicated in order, at most 60; None when none remain."""
    out, seen = [], set()
    for k in keyphrases or []:
        if not isinstance(k, str):
            continue
        t = k.strip().lower()
        if _TERM.fullmatch(t) and t not in seen:
            seen.add(t)
            out.append(t)
            if len(out) == 60:
                break
    return out or None


def slugify(text):
    """020 slug: NFKC, casefold, non-alphanumeric runs to one hyphen, trimmed; None unless ^[a-z0-9]+(-[a-z0-9]+)*$, 2-64."""
    s = re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKC", str(text or "")).casefold()).strip("-")
    return s if 2 <= len(s) <= 64 and _SLUG.fullmatch(s) else None


def fold_label(label) -> str:
    return re.sub(r"\s+", " ", str(label).strip()).lower()


def _valid_label(label) -> bool:
    return isinstance(label, str) and 2 <= len(label.strip()) <= 120 and not _CTRL.search(label)


def _valid_anchor(text) -> bool:
    return isinstance(text, str) and 20 <= len(text) <= 1200 and bool(_ANCHOR.fullmatch(text))


def registry_data_from_rows(rows, mode="active") -> dict:
    """Matcher registry data from view rows. One entry per row WITH an anchor; rows without one are listed in skipped."""
    if mode not in ("active", "proposed"):
        raise TaxRegistryError("mode must be active or proposed")
    entries, skipped = [], []
    for r in rows or []:
        if not r.get("anchor_text"):
            skipped.append({"id": r.get("slug"), "kind": r.get("kind"), "reason": "no_anchor"})
            continue
        labels = list(r.get("domain_labels") or [])
        parents = list(r.get("parent_slugs") or [])
        terms = r.get("core_terms")
        entries.append({"id": r["slug"], "label": r["label"], "field": labels[0] if labels else None, "fields": labels,
                        "domain_slugs": list(r.get("domain_slugs") or []), "parent_topic_key": parents[0] if parents else None,
                        "parents": parents, "anchor_text": r["anchor_text"], "core_terms": list(terms) if terms else None,
                        "kind": r.get("kind"), "state": r.get("entity_state"), "anchor_state": r.get("anchor_state"),
                        "anchor_sha256": r.get("anchor_sha256"), "includes_proposed": bool(r.get("includes_proposed"))})
    sig = hashlib.sha256(json.dumps(sorted((e["id"], str(e["anchor_sha256"])) for e in entries)).encode("utf-8")).hexdigest()[:12]
    return {"anchor_registry_version": f"tax-{mode}:{sig}", "entries": entries, "skipped": skipped}


def drift_report(topreg_data, tax_registry_data) -> dict:
    """Pure comparison of the topreg registry data with the tax registry data (entries by id; topreg ids are slugified)."""
    def keyed(data, conv):
        return {conv(e["id"]): e for e in (data or {}).get("entries") or [] if isinstance(e, dict) and e.get("id")}
    a = keyed(topreg_data, lambda i: slugify(i) or str(i))
    b = keyed(tax_registry_data, str)
    both = sorted(set(a) & set(b))
    return {"only_in_topreg": sorted(set(a) - set(b)), "only_in_tax": sorted(set(b) - set(a)),
            "label_differs": [k for k in both if fold_label(a[k].get("label")) != fold_label(b[k].get("label"))],
            "anchor_differs": [k for k in both if b[k].get("anchor_sha256") != anchor_sha256(a[k].get("anchor_text"), None)],
            "compared": len(both)}


# ------------------------------------------------------------------------------------------------------------ db access
@contextlib.contextmanager
def _read(engine):
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        yield conn


@contextlib.contextmanager
def _write(engine):
    with engine.begin() as conn:
        yield conn


def _text(sql):
    from sqlalchemy import text
    return text(sql)


def tax_present(conn) -> bool:
    return conn.execute(_text(SQL_PRESENT)).scalar() is not None


def _set(conn, name, value):
    conn.execute(_text(SQL_SET_CONFIG), {"n": name, "v": value})


def read_registry_rows(engine, mode="active"):
    """(present, rows): one SELECT from the active or the eval view in a READ ONLY transaction. present False when 022 is absent."""
    if mode not in SQL_ROWS:
        raise TaxRegistryError("mode must be active or proposed")
    try:
        with _read(engine) as conn:
            if not tax_present(conn):
                return False, []
            return True, [dict(r) for r in conn.execute(_text(SQL_ROWS[mode])).mappings().all()]
    except TaxRegistryError:
        raise
    except Exception as exc:  # noqa: BLE001 - fail closed; class name only
        raise TaxRegistryError(f"tax registry read failed ({type(exc).__name__})") from None


def read_existing(engine) -> dict:
    """{domains|topics|subtopics: {slug: {"label", "state"}}} (all states). READ ONLY."""
    try:
        with _read(engine) as conn:
            return {k: {str(r[0]): {"label": str(r[1]), "state": str(r[2])} for r in conn.execute(_text(q)).all()}
                    for k, q in SQL_EXISTING.items()}
    except Exception as exc:  # noqa: BLE001
        raise TaxRegistryError(f"tax existing read failed ({type(exc).__name__})") from None


def load_registry_data(cfg, engine=None, mode="active", reader=None):
    """Matcher registry data from the tax views, or None (mode active with read_from_tax not true, or 022 absent). Mode 'proposed'
    is for owner evaluation: it ignores read_from_tax but still needs 022. Any other failure raises (fail closed)."""
    if mode not in ("active", "proposed"):
        raise TaxRegistryError("mode must be active or proposed")
    if mode == "active" and not _flag(cfg, "read_from_tax"):
        return None
    if reader is None:
        if engine is None:
            from vera import db
            engine = db.get_role_engine("rw")
        present, rows = read_registry_rows(engine, mode)
    else:
        present, rows = reader()
    if not present:
        return None
    return registry_data_from_rows(rows, mode)


# --------------------------------------------------------------------------------------------------------------- proposals
def _batch(manifest) -> str:
    b = manifest.get("batch_sha256") if isinstance(manifest, dict) else None
    if not isinstance(b, str) or not _SHA.fullmatch(b):
        raise TaxRegistryError("manifest batch_sha256 missing or malformed")
    return b


def plan_proposals(manifest, existing) -> dict:
    """Pure plan from an induction manifest. Conflicting items are skipped and listed in plan['problems'] ({'key','reason'})."""
    batch = _batch(manifest)
    sha12 = batch[:12]
    oref, cver = f"induce:{sha12}", f"batch:{sha12}"
    have = {k: dict((existing or {}).get(k) or {}) for k in KINDS}
    live = {k: {fold_label(v["label"]): s for s, v in have[k].items() if v["state"] in ("proposed", "active")} for k in KINDS}
    plan = {"batch_sha256": batch, "origin_ref": oref, "classifier_name": "taxonomy_induction", "classifier_version": cver,
            "domains": [], "topics": [], "subtopics": [], "domain_topic": [], "topic_subtopic": [], "anchors": [], "problems": []}
    problems = plan["problems"]
    planned = {k: {} for k in KINDS}  # slug -> label planned in this batch

    def prob(key, reason):
        problems.append({"key": key, "reason": reason})

    def live_slug(kind, slug):
        s = have[kind].get(slug) or {}
        return s.get("state") in ("proposed", "active") or slug in planned[kind]

    def domain_for(field, key):
        """slug of the domain to link (existing live, or planned); None with a problem recorded."""
        if not _valid_label(field):
            prob(key, "invalid_domain_label")
            return None
        fl = fold_label(field)
        if fl in live["domains"]:
            return live["domains"][fl]
        for s, lbl in planned["domains"].items():
            if fold_label(lbl) == fl:
                return s
        ds = slugify(field)
        if ds is None:
            prob(key, "invalid_domain_slug")
            return None
        if ds in have["domains"]:
            prob(key, "domain_slug_exists_not_live")
            return None
        planned["domains"][ds] = field.strip()
        plan["domains"].append({"slug": ds, "label": field.strip(), "origin_ref": oref})
        return ds

    def check_entity(kind, p):
        key, label, slug = p.get("key"), p.get("label"), p.get("key")
        if not isinstance(key, str) or slugify(key) != key:
            prob(str(key), "invalid_slug")
            return False
        if not _valid_label(label):
            prob(key, "invalid_label")
            return False
        if not _valid_anchor(p.get("anchor_text")):
            prob(key, "invalid_anchor")
            return False
        if slug in have[kind] or slug in planned[kind]:
            prob(key, "slug_exists")
            return False
        fl = fold_label(label)
        if fl in live[kind] or any(fold_label(v) == fl for v in planned[kind].values()):
            prob(key, "label_clash")
            return False
        return True

    props = [p for p in (manifest.get("proposals") or []) if isinstance(p, dict)]
    for level, kind in (("topic", "topics"), ("subtopic", "subtopics")):
        for p in (q for q in props if q.get("level") == level):
            if not check_entity(kind, p):
                continue
            key = p["key"]
            if level == "topic":
                dslug = domain_for(p.get("field"), key)
                if dslug is None:
                    continue
                plan["domain_topic"].append({"domain": dslug, "topic": key})
            else:
                parent = slugify(p.get("parent"))
                if parent is None or not live_slug("topics", parent):
                    prob(key, "unknown_parent")
                    continue
                plan["topic_subtopic"].append({"topic": parent, "subtopic": key})
            planned[kind][key] = p["label"].strip()
            plan[kind].append({"slug": key, "label": p["label"].strip(), "origin_ref": oref})
            plan["anchors"].append({"kind": kind, "slug": key, "anchor_text": p["anchor_text"],
                                    "core_terms": core_terms_from(p.get("keyphrases")), "origin_ref": oref})
    for p in props:
        if p.get("level") not in ("topic", "subtopic"):
            prob(str(p.get("key")), "invalid_level")
    used = {d["domain"] for d in plan["domain_topic"]}
    plan["domains"] = [d for d in plan["domains"] if d["slug"] in used]
    plan["counts"] = {k: len(plan[k]) for k in ("domains", "topics", "subtopics", "domain_topic", "topic_subtopic", "anchors")}
    return plan


def _require_write(cfg, flag, conn):
    if not _flag(cfg, flag):
        raise TaxRefused(f"{flag} is not true")
    _require_present(conn)


def _require_present(conn):
    if not tax_present(conn):
        raise TaxRefused("migration 022 is not applied (vera_vjay.tax_anchors absent)")


def _gate(conn, cfg, flag, apply):
    """Before any tax_anchors access: apply needs the flag AND 022; a dry run needs 022 only."""
    if apply:
        _require_write(cfg, flag, conn)
    else:
        _require_present(conn)


def write_proposals(engine, plan, cfg, apply=False) -> dict:
    """Dry run reports counts. Apply (write_proposals true AND 022 present) inserts the plan as proposed rows in one transaction."""
    rep = {"applied": False, "counts": dict(plan.get("counts") or {}), "problems": list(plan.get("problems") or [])}
    if not apply:
        return rep
    if not _flag(cfg, "write_proposals"):
        raise TaxRefused("write_proposals is not true")
    try:
        with _write(engine) as conn:
            _require_write(cfg, "write_proposals", conn)
            _set(conn, "vera.tax_actor", "induction:" + plan["batch_sha256"][:12])
            _set(conn, "vera.tax_batch", plan["batch_sha256"])  # 022: an induction proposal needs both settings
            for kind in KINDS:
                for r in plan[kind]:
                    conn.execute(_text(SQL_INSERT_ENTITY[kind]), {"slug": r["slug"], "label": r["label"], "oref": r["origin_ref"]})
            cv = {"cname": plan["classifier_name"], "cver": plan["classifier_version"]}
            for l in plan["domain_topic"]:
                conn.execute(_text(SQL_INSERT_LINK["domain_topic"]), {"a": l["domain"], "b": l["topic"], **cv})
            for l in plan["topic_subtopic"]:
                conn.execute(_text(SQL_INSERT_LINK["topic_subtopic"]), {"a": l["topic"], "b": l["subtopic"], **cv})
            for a in plan["anchors"]:
                conn.execute(_text(SQL_INSERT_ANCHOR[a["kind"]]), {"slug": a["slug"], "text": a["anchor_text"],
                                                                      "terms": a["core_terms"], "oref": a["origin_ref"]})
    except TaxRegistryError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TaxRegistryError(f"write proposals failed and was rolled back ({type(exc).__name__})") from None
    rep["applied"] = True
    return rep


# ---------------------------------------------------------------------------------------------------------------- approval
def approve_batch(engine, manifest, approve, reject, actor, cfg, apply=False) -> dict:
    """Approve or reject manifest items (keys and 'domain:<slug>'). Only stored proposed induction rows whose label and anchor
    hash equal the manifest change. Apply needs approve_enabled true AND 022; one transaction, actor and batch transaction-local."""
    if not isinstance(actor, str) or not _ACTOR.fullmatch(actor):
        raise TaxRegistryError("actor must match owner:[a-z0-9_.-]{1,60}")
    batch = _batch(manifest)
    approve, reject = set(approve or ()), set(reject or ())
    rep = {"applied": False, "outcomes": {}, "warnings": [], "problems": []}
    if approve & reject:
        rep["problems"].append({"key": ",".join(sorted(approve & reject)), "reason": "in_both_approve_and_reject"})
        return rep
    props = {p["key"]: p for p in manifest.get("proposals") or [] if isinstance(p, dict) and isinstance(p.get("key"), str)}
    dom_label = {}
    for p in props.values():
        s = slugify(p.get("field"))
        if p.get("new_domain") and s:
            dom_label.setdefault(s, str(p["field"]).strip())  # stored stripped by plan_proposals
    items = []  # (order, kind, slug, label, sha, item)
    for item in sorted(approve | reject):
        if item.startswith("domain:"):
            s = item[7:]
            if s not in dom_label:
                rep["outcomes"][item] = "refused:unknown_item"
                continue
            items.append((0, "domains", s, dom_label[s], None, item))
        elif item in props:
            p = props[item]
            kind = "topics" if p.get("level") == "topic" else "subtopics" if p.get("level") == "subtopic" else None
            if kind is None:
                rep["outcomes"][item] = "refused:invalid_level"
                continue
            items.append((1 if kind == "topics" else 2, kind, item, str(p.get("label") or "").strip(),
                          anchor_sha256(p.get("anchor_text"), core_terms_from(p.get("keyphrases"))), item))
        else:
            rep["outcomes"][item] = "refused:unknown_item"
    items.sort(key=lambda t: (t[0], t[2]))

    def run(conn):
        _gate(conn, cfg, "approve_enabled", apply)
        stored = {}
        for kind in KINDS:
            slugs = sorted({i[2] for i in items if i[1] == kind} | ({s for s in (slugify(props[i[5]].get("field")) for i in items
                            if i[1] == "topics") if s} if kind == "domains" else set()))
            if slugs:
                stored[kind] = {r[0]: {"label": r[1], "state": r[2], "origin": r[3], "sha": r[4]}
                                for r in conn.execute(_text(SQL_STORED[kind]), {"slugs": slugs}).all()}
        ok = []
        for order, kind, slug, label, sha, item in items:
            row = (stored.get(kind) or {}).get(slug)
            if row is None:
                why = "not_found"
            elif row["state"] != "proposed":
                why = "state_not_proposed"
            elif row["origin"] != "induction":
                why = "origin_not_induction"
            elif row["label"] != label:
                why = "label_mismatch"
            elif kind != "domains" and row["sha"] != sha:
                why = "content_mismatch"
            else:
                why = None
            if why:
                rep["outcomes"][item] = "refused:" + why
                continue
            if item in approve and kind == "topics":
                ds = slugify(props[item].get("field"))
                d = (stored.get("domains") or {}).get(ds)
                if d and d["state"] == "proposed" and ("domain:" + ds) not in approve:
                    rep["warnings"].append({"key": item, "warning": "domain still proposed"})
            ok.append((order, kind, slug, sha, item))
        if not apply:
            for _o, _k, _s, _h, item in ok:
                rep["outcomes"][item] = "would_approve" if item in approve else "would_reject"
            return
        _set(conn, "vera.tax_actor", actor)
        _set(conn, "vera.tax_batch", batch)
        for _o, kind, slug, sha, item in ok:
            ns = "active" if item in approve else "rejected"
            if conn.execute(_text(SQL_DECIDE_ENTITY[kind]), {"ns": ns, "slug": slug}).rowcount != 1:
                raise TaxRefused(f"entity {item} changed during approval")
            if kind != "domains" and ns == "active":  # one active anchor per entity: retire any old one first (0 rows is fine)
                conn.execute(_text(SQL_RETIRE_ANCHOR[kind]), {"slug": slug})
            if kind != "domains" and conn.execute(_text(SQL_DECIDE_ANCHOR[kind]), {"ns": ns, "slug": slug, "sha": sha}).rowcount != 1:
                raise TaxRefused(f"anchor of {item} changed during approval")
            rep["outcomes"][item] = "approved" if ns == "active" else "rejected"

    if apply and not _flag(cfg, "approve_enabled"):
        raise TaxRefused("approve_enabled is not true")
    try:
        with (_write(engine) if apply else _read(engine)) as conn:
            run(conn)
    except TaxRegistryError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TaxRegistryError(f"approval failed and was rolled back ({type(exc).__name__})") from None
    rep["applied"] = bool(apply)
    return rep


# ------------------------------------------------------------------------------------------------------------------- sync
def sync_anchors(engine, registry_data, cfg, apply=False) -> dict:
    """Seed tax_anchors for ACTIVE tax entities with origin_ref 'topreg:<key>' from topreg registry data. Never creates entities."""
    want = {}
    rep = {"applied": False, "seed": [], "replace": [], "unchanged": 0, "missing_entity": [], "invalid": []}
    for e in (registry_data or {}).get("entries") or []:
        if isinstance(e, dict) and e.get("id"):
            if _valid_anchor(e.get("anchor_text")):
                want[str(e["id"])] = e["anchor_text"]
            else:
                rep["invalid"].append(str(e["id"]))
    work = []

    def run(conn):
        _gate(conn, cfg, "sync_enabled", apply)
        rows = conn.execute(_text(SQL_SYNC_STATE)).all()
        seen = set()
        for kind, slug, oref, sha in ((str(r[0]), str(r[1]), str(r[2]), r[3]) for r in rows):
            key = oref[len("topreg:"):]
            if key not in want:
                continue
            seen.add(key)
            new = anchor_sha256(want[key], None)
            if sha == new:
                rep["unchanged"] += 1
            elif sha is None:
                rep["seed"].append(key)
                work.append((kind, slug, oref, want[key], False))
            else:
                rep["replace"].append(key)
                work.append((kind, slug, oref, want[key], True))
        rep["missing_entity"] = sorted(set(want) - seen)
        if not apply:
            return
        for kind, slug, oref, text, retire in work:
            if retire:
                conn.execute(_text(SQL_RETIRE_ANCHOR[kind]), {"slug": slug})
            if conn.execute(_text(SQL_SEED_ANCHOR[kind]), {"slug": slug, "text": text, "oref": oref}).rowcount != 1:
                raise TaxRefused(f"anchor seed for {oref} did not insert")

    if apply and not _flag(cfg, "sync_enabled"):
        raise TaxRefused("sync_enabled is not true")
    try:
        with (_write(engine) if apply else _read(engine)) as conn:
            run(conn)
    except TaxRegistryError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TaxRegistryError(f"anchor sync failed and was rolled back ({type(exc).__name__})") from None
    rep["seed"].sort()
    rep["replace"].sort()
    rep["applied"] = bool(apply)
    return rep
