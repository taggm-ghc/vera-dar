"""TAXO-EXTENSIBLE-1108: vera/tax_registry.py and scripts/tax_registry_cli.py with FAKE engines (no DB, no network). Taxonomy names
below are invented and obviously synthetic."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

from vera import chunk_topics, micro_topic
from vera import tax_registry as tr

ROOT = Path(__file__).resolve().parent.parent
B = "ab" * 32
ANCHOR = "synthetic anchor text about synthetic things for testing"


class FakeResult:
    def __init__(self, rows=None, scalar=None, rowcount=1):
        self._rows, self._scalar, self.rowcount = rows or [], scalar, rowcount

    def scalar(self):
        return self._scalar

    def all(self):
        return list(self._rows)

    def mappings(self):
        return self


class FakeConn:
    def __init__(self, eng):
        self.eng = eng

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.eng.calls.append((sql, dict(params or {})))
        return self.eng.route(sql, dict(params or {}))


class FakeEngine:
    def __init__(self, route=None, present=True):
        self.calls, self.present, self._route, self.opened = [], present, route, []

    def connect(self):
        self.opened.append("connect")
        return FakeConn(self)

    def begin(self):
        self.opened.append("begin")
        return FakeConn(self)

    def route(self, sql, params):
        if "to_regclass" in sql:
            return FakeResult(scalar="vera_vjay.tax_anchors" if self.present else None)
        if self._route:
            r = self._route(sql, params)
            if r is not None:
                return r
        return FakeResult()

    def sqls(self):
        return [c[0] for c in self.calls]


CFG_OFF = {"read_from_tax": False, "write_proposals": False, "approve_enabled": False, "sync_enabled": False}


def cfg(**on):
    return {**CFG_OFF, **{k: True for k in on}}


def row(kind="topic", slug="synthetic-topic-a", label="Synthetic Topic A", anchor=ANCHOR, domains=("Synthetic Domain",), parents=None,
        state="active", terms=None, proposed=False):
    sha = tr.anchor_sha256(anchor, terms) if anchor else None
    return {"kind": kind, "entity_id": 1, "slug": slug, "label": label, "description": None, "entity_state": state, "anchor_id": 1 if anchor else None,
            "anchor_text": anchor, "core_terms": terms, "anchor_sha256": sha, "anchor_state": "active" if anchor else None,
            "parent_slugs": parents, "domain_slugs": [tr.slugify(d) for d in domains], "domain_labels": list(domains),
            "includes_proposed": proposed}


# ---------------------------------------------------------------------------------------------------------- config
def test_shipped_config_is_all_false_and_strict(tmp_path):
    c = tr.load_config()
    assert c == CFG_OFF
    raw = json.loads((ROOT / "config" / "tax_registry.json").read_text())
    assert set(raw) == {"_note", *tr.FLAGS} and all(raw[f] is False for f in tr.FLAGS)
    for bad in ({**CFG_OFF, "read_from_tax": "true"}, {**CFG_OFF, "sync_enabled": 1}, {"read_from_tax": False}, {**CFG_OFF, "cap": 5}, []):
        p = tmp_path / "c.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(tr.TaxRegistryError):
            tr.load_config(p)
    with pytest.raises(tr.TaxRegistryError):
        tr.load_config(tmp_path / "missing.json")


# ------------------------------------------------------------------------------------------------------- pure helpers
def test_anchor_sha256_matches_hand_computed_vector():
    assert tr.anchor_sha256("abc", ["x y", "z"]) == hashlib.sha256(b"abc\nx y|z").hexdigest()
    assert tr.anchor_sha256("abc", None) == hashlib.sha256(b"abc\n").hexdigest()
    assert tr.anchor_sha256("abc", []) == hashlib.sha256(b"abc\n").hexdigest()


def test_core_terms_from():
    assert tr.core_terms_from(["Alpha Beta", "alpha beta", " gamma ", "Bad!", "", None, "-x", "a" * 49]) == ["alpha beta", "gamma"]
    assert tr.core_terms_from([]) is None and tr.core_terms_from(["!!"]) is None and tr.core_terms_from(None) is None
    assert len(tr.core_terms_from([f"term {i}" for i in range(100)])) == 60
    assert tr.core_terms_from(["a" * 48]) == ["a" * 48]


def test_slugify_mirrors_020_check():
    assert tr.slugify("Synthetic  Topic: A!") == "synthetic-topic-a"
    assert tr.slugify("ＡＢ cd") == "ab-cd"
    assert tr.slugify("a") is None and tr.slugify("!!") is None and tr.slugify(None) is None
    assert tr.slugify("x" * 65) is None and tr.slugify("x" * 64) == "x" * 64


# -------------------------------------------------------------------------------------------------- entries and matcher
def test_entry_mapping_multi_domain_multi_parent_and_skipped():
    rows = [row(slug="synthetic-topic-a", domains=("Synthetic Domain", "Other Synthetic Domain")),
            row("subtopic", "synthetic-sub-a", "Synthetic Sub A", parents=["synthetic-topic-a", "synthetic-topic-b"], terms=["alpha"]),
            row("subtopic", "synthetic-sub-b", "Synthetic Sub B", anchor=None, parents=["synthetic-topic-a"]),
            row(slug="synthetic-topic-c", domains=())]
    d = tr.registry_data_from_rows(rows, "proposed")
    assert d["anchor_registry_version"].startswith("tax-proposed:") and len(d["anchor_registry_version"]) == len("tax-proposed:") + 12
    ids = [e["id"] for e in d["entries"]]
    assert ids == ["synthetic-topic-a", "synthetic-sub-a", "synthetic-topic-c"]
    a, s, c = d["entries"]
    assert a["field"] == "Synthetic Domain" and a["fields"] == ["Synthetic Domain", "Other Synthetic Domain"] and a["parent_topic_key"] is None
    assert s["parent_topic_key"] == "synthetic-topic-a" and s["parents"] == ["synthetic-topic-a", "synthetic-topic-b"]
    assert s["core_terms"] == ["alpha"] and s["kind"] == "subtopic" and s["state"] == "active"
    assert c["field"] is None and c["fields"] == []
    assert d["skipped"] == [{"id": "synthetic-sub-b", "kind": "subtopic", "reason": "no_anchor"}]
    d2 = tr.registry_data_from_rows(list(reversed(rows)), "proposed")
    assert d2["anchor_registry_version"] == d["anchor_registry_version"]
    assert tr.registry_data_from_rows(rows[:1], "active")["anchor_registry_version"].startswith("tax-active:")
    with pytest.raises(tr.TaxRegistryError):
        tr.registry_data_from_rows(rows, "x")


def test_matcher_accepts_entries_including_field_none():
    rows = [row(slug="synthetic-topic-a", anchor="zork quibble frobnicate alpha beta gamma delta"),
            row(slug="synthetic-topic-c", domains=(), anchor="plugh xyzzy wibble epsilon zeta eta theta"),
            row("subtopic", "synthetic-sub-a", parents=["synthetic-topic-a"], anchor="zork quibble sub alpha iota kappa lambda")]
    d = tr.registry_data_from_rows(rows)
    reg = micro_topic.Registry(d)
    assert reg.version == d["anchor_registry_version"] and set(reg.by_id) == {e["id"] for e in d["entries"]}
    sc, e = reg.best("plugh xyzzy wibble epsilon")
    assert e["id"] == "synthetic-topic-c" and sc > 0
    tree = chunk_topics.tree_of(d["entries"])
    assert chunk_topics.root_of(tree, "synthetic-sub-a") == "synthetic-topic-a"
    assert chunk_topics.root_of(tree, "synthetic-topic-c") == "synthetic-topic-c"
    idx = chunk_topics.ChunkIndex(d["entries"])
    assert len(idx.entries) == 3 and idx.domain[1] == ""


# ----------------------------------------------------------------------------------------------------------- reading
def test_read_mode_selects_view_in_read_only_txn_with_order():
    for mode, view in (("active", "tax_registry_active_v"), ("proposed", "tax_registry_eval_v")):
        eng = FakeEngine(lambda sql, p: FakeResult(rows=[row()]) if "tax_registry_" in sql else None)
        present, rows = tr.read_registry_rows(eng, mode)
        sqls = eng.sqls()
        assert present and len(rows) == 1 and eng.opened == ["connect"]
        assert sqls[0] == "SET TRANSACTION READ ONLY"
        sel = [s for s in sqls if "tax_registry_" in s]
        assert len(sel) == 1 and f"vera_vjay.{view}" in sel[0] and "ORDER BY kind DESC, entity_id" in sel[0]
        assert ("eval" in sel[0]) == (mode == "proposed")


def test_load_registry_data_flags_and_absence():
    eng = FakeEngine(lambda sql, p: FakeResult(rows=[row()]) if "tax_registry_" in sql else None)
    assert tr.load_registry_data(CFG_OFF, eng) is None and eng.calls == []  # flag off: no DB access at all
    d = tr.load_registry_data(cfg(read_from_tax=True), eng)
    assert d and d["anchor_registry_version"].startswith("tax-active:")
    # proposed mode ignores the flag
    assert tr.load_registry_data(CFG_OFF, FakeEngine(lambda s, p: FakeResult(rows=[row()]) if "tax_registry_" in s else None), "proposed")
    # 022 absent -> None in both modes, and the view is never queried
    absent = FakeEngine(present=False)
    assert tr.load_registry_data(cfg(read_from_tax=True), absent) is None
    assert tr.load_registry_data(CFG_OFF, absent, "proposed") is None
    assert not any("tax_registry_" in s for s in absent.sqls())
    # injected reader
    assert tr.load_registry_data(cfg(read_from_tax=True), reader=lambda: (False, [])) is None
    assert tr.load_registry_data(cfg(read_from_tax=True), reader=lambda: (True, [row()]))["entries"]
    # other failures fail closed
    def boom(sql, p):
        if "tax_registry_" in sql:
            raise OSError("secret-host")
    with pytest.raises(tr.TaxRegistryError) as ei:
        tr.load_registry_data(cfg(read_from_tax=True), FakeEngine(boom))
    assert "secret-host" not in str(ei.value)
    with pytest.raises(tr.TaxRegistryError):
        tr.load_registry_data(CFG_OFF, mode="bogus")


# ---------------------------------------------------------------------------------------------------------- proposals
def manifest(*props):
    return {"batch_sha256": B, "proposals": list(props)}


def prop(key, level="topic", label=None, field="Synthetic New Domain", parent=None, new_domain=True, kp=("alpha", "beta"), anchor=ANCHOR):
    return {"key": key, "label": label or key.replace("-", " ").title(), "field": field, "parent": parent, "anchor_text": anchor,
            "level": level, "new_domain": new_domain, "keyphrases": list(kp), "content_sha256": "0" * 64}


EXISTING = {"domains": {"synthetic-live-domain": {"label": "Synthetic Live Domain", "state": "active"},
                        "synthetic-gone-domain": {"label": "Synthetic Gone Domain", "state": "retired"}},
            "topics": {"synthetic-live-topic": {"label": "Synthetic Live Topic", "state": "active"},
                       "synthetic-old-topic": {"label": "Synthetic Old Topic", "state": "rejected"}},
            "subtopics": {}}


def test_plan_new_domain_topic_subtopic():
    m = manifest(prop("synthetic-topic-x"), prop("synthetic-topic-y", field="synthetic new  domain"),
                 prop("synthetic-sub-x", "subtopic", field="Synthetic Live Domain", parent="synthetic-live-topic", new_domain=False),
                 prop("synthetic-sub-y", "subtopic", parent="synthetic-topic-x", new_domain=False, kp=()),
                 prop("synthetic-topic-z", field="Synthetic Live Domain"))
    p = tr.plan_proposals(m, EXISTING)
    assert p["problems"] == []
    assert p["domains"] == [{"slug": "synthetic-new-domain", "label": "Synthetic New Domain", "origin_ref": "induce:" + B[:12]}]
    assert [t["slug"] for t in p["topics"]] == ["synthetic-topic-x", "synthetic-topic-y", "synthetic-topic-z"]
    assert {"domain": "synthetic-new-domain", "topic": "synthetic-topic-x"} in p["domain_topic"]
    assert {"domain": "synthetic-new-domain", "topic": "synthetic-topic-y"} in p["domain_topic"]
    assert {"domain": "synthetic-live-domain", "topic": "synthetic-topic-z"} in p["domain_topic"]
    assert p["topic_subtopic"] == [{"topic": "synthetic-live-topic", "subtopic": "synthetic-sub-x"},
                                   {"topic": "synthetic-topic-x", "subtopic": "synthetic-sub-y"}]
    assert p["classifier_name"] == "taxonomy_induction" and p["classifier_version"] == "batch:" + B[:12]
    anc = {a["slug"]: a for a in p["anchors"]}
    assert anc["synthetic-topic-x"]["core_terms"] == ["alpha", "beta"] and anc["synthetic-sub-y"]["core_terms"] is None
    assert anc["synthetic-sub-x"]["kind"] == "subtopics" and p["counts"]["anchors"] == 5


def test_plan_conflicts_are_reported_and_skipped():
    m = manifest(prop("synthetic-live-topic", field="Synthetic Live Domain"),       # slug exists (active)
                 prop("synthetic-old-topic", field="Synthetic Live Domain"),         # slug exists (rejected: never revived)
                 prop("synthetic-topic-q", label="synthetic  LIVE topic", field="Synthetic Live Domain"),  # live label clash (fold)
                 prop("synthetic-sub-q", "subtopic", parent="synthetic-unknown", new_domain=False),
                 prop("synthetic-topic-r", field="Synthetic Gone Domain"),           # domain slug exists but retired
                 prop("Bad Slug!", field="Synthetic Live Domain"),
                 prop("synthetic-topic-s", field="Synthetic Live Domain", anchor="short"),
                 prop("synthetic-topic-t", field="Synthetic Live Domain", label="x"),
                 prop("synthetic-topic-u", field="Synthetic New Domain", label="Dup"), prop("synthetic-topic-v", field="Synthetic New Domain", label="dup"))
    p = tr.plan_proposals(m, EXISTING)
    why = {x["key"]: x["reason"] for x in p["problems"]}
    assert why["synthetic-live-topic"] == "slug_exists" and why["synthetic-old-topic"] == "slug_exists"
    assert why["synthetic-topic-q"] == "label_clash" and why["synthetic-sub-q"] == "unknown_parent"
    assert why["synthetic-topic-r"] == "domain_slug_exists_not_live" and why["Bad Slug!"] == "invalid_slug"
    assert why["synthetic-topic-s"] == "invalid_anchor" and why["synthetic-topic-t"] == "invalid_label"
    assert why["synthetic-topic-v"] == "label_clash"
    assert [t["slug"] for t in p["topics"]] == ["synthetic-topic-u"]
    # a planned domain nobody uses is dropped
    p2 = tr.plan_proposals(manifest(prop("synthetic-live-topic", field="Synthetic Brand New")), EXISTING)
    assert p2["domains"] == [] and p2["topics"] == []
    with pytest.raises(tr.TaxRegistryError):
        tr.plan_proposals({"proposals": []}, EXISTING)


def test_write_proposals_dry_run_flag_and_absent():
    plan = tr.plan_proposals(manifest(prop("synthetic-topic-x")), EXISTING)
    eng = FakeEngine()
    r = tr.write_proposals(eng, plan, CFG_OFF)
    assert r["applied"] is False and r["counts"]["topics"] == 1 and eng.calls == []
    with pytest.raises(tr.TaxRefused):
        tr.write_proposals(eng, plan, CFG_OFF, apply=True)
    assert eng.calls == []  # flag off: the DB is never touched
    absent = FakeEngine(present=False)
    with pytest.raises(tr.TaxRefused):
        tr.write_proposals(absent, plan, cfg(write_proposals=True), apply=True)
    assert not any("INSERT" in s for s in absent.sqls())


def test_write_proposals_apply_sql_order_and_params():
    plan = tr.plan_proposals(manifest(prop("synthetic-topic-x"), prop("synthetic-sub-y", "subtopic", parent="synthetic-topic-x", new_domain=False)), EXISTING)
    eng = FakeEngine()
    r = tr.write_proposals(eng, plan, cfg(write_proposals=True), apply=True)
    assert r["applied"] and eng.opened == ["begin"]
    sqls = eng.sqls()
    first = [i for i, s in enumerate(sqls) if "set_config" in s]
    assert [eng.calls[i][1] for i in first] == [{"n": "vera.tax_actor", "v": "induction:" + B[:12]}, {"n": "vera.tax_batch", "v": B}]
    assert max(first) < min(i for i, x in enumerate(sqls) if x.startswith("INSERT"))
    ins = [s for s in sqls if s.startswith("INSERT")]
    names = [re.search(r"vera_vjay\.(\w+)", s).group(1) for s in ins]
    assert names == ["tax_domains", "tax_topics", "tax_subtopics", "tax_domain_topic", "tax_topic_subtopic", "tax_anchors", "tax_anchors"]
    assert all("'proposed'" in s for s in ins if s.startswith(("INSERT INTO vera_vjay.tax_anchors", "INSERT INTO vera_vjay.tax_domains")))
    anc = [c for c in eng.calls if c[0].startswith("INSERT INTO vera_vjay.tax_anchors")]
    assert anc[0][1]["terms"] == ["alpha", "beta"] and anc[0][1]["oref"] == "induce:" + B[:12]
    assert "state = 'proposed'" not in "".join(sqls) and not any(s.startswith(("UPDATE", "DELETE")) for s in sqls)


# ----------------------------------------------------------------------------------------------------------- approval
ACTOR = "owner:tester"


def stored_route(topic_sha=None, topic_state="proposed", label="Synthetic Topic X", dom_state="proposed", sub_sha=None, origin="induction"):
    m = manifest(prop("synthetic-topic-x", label=label), prop("synthetic-sub-y", "subtopic", parent="synthetic-topic-x", new_domain=False))
    ts = topic_sha or tr.anchor_sha256(ANCHOR, ["alpha", "beta"])
    ss = sub_sha or tr.anchor_sha256(ANCHOR, ["alpha", "beta"])

    def route(sql, p):
        if ("AS anchor_sha FROM" in sql) and "tax_domains" in sql:
            return FakeResult(rows=[("synthetic-new-domain", "Synthetic New Domain", dom_state, "induction", None)])
        if ("AS anchor_sha FROM" in sql) and "tax_topics" in sql:
            return FakeResult(rows=[("synthetic-topic-x", "Synthetic Topic X", topic_state, origin, ts)])
        if ("AS anchor_sha FROM" in sql) and "tax_subtopics" in sql:
            return FakeResult(rows=[("synthetic-sub-y", "Synthetic Sub Y", "proposed", "induction", ss)])
        return None
    return m, route


def test_approve_actor_and_flags():
    m, route = stored_route()
    with pytest.raises(tr.TaxRegistryError):
        tr.approve_batch(FakeEngine(route), m, {"synthetic-topic-x"}, set(), "tester", cfg(approve_enabled=True), apply=True)
    with pytest.raises(tr.TaxRegistryError):
        tr.approve_batch(FakeEngine(route), m, set(), set(), "owner:Bad Name", cfg(approve_enabled=True))
    eng = FakeEngine(route)
    with pytest.raises(tr.TaxRefused):
        tr.approve_batch(eng, m, {"synthetic-topic-x"}, set(), ACTOR, CFG_OFF, apply=True)
    assert eng.calls == []
    absent = FakeEngine(route, present=False)
    with pytest.raises(tr.TaxRefused):
        tr.approve_batch(absent, m, {"synthetic-topic-x"}, set(), ACTOR, cfg(approve_enabled=True), apply=True)
    assert not any(s.startswith("UPDATE") for s in absent.sqls())
    r = tr.approve_batch(FakeEngine(route), m, {"synthetic-topic-x"}, {"synthetic-topic-x"}, ACTOR, cfg(approve_enabled=True))
    assert r["problems"] and r["outcomes"] == {}


def test_approve_refuses_content_mismatch_and_wrong_state():
    m, route = stored_route(topic_sha="f" * 64)
    r = tr.approve_batch(FakeEngine(route), m, {"synthetic-topic-x", "synthetic-sub-y"}, set(), ACTOR, CFG_OFF)
    assert r["outcomes"]["synthetic-topic-x"] == "refused:content_mismatch" and r["outcomes"]["synthetic-sub-y"] == "would_approve"
    m, route = stored_route(topic_state="active")
    r = tr.approve_batch(FakeEngine(route), m, {"synthetic-topic-x"}, set(), ACTOR, CFG_OFF)
    assert r["outcomes"]["synthetic-topic-x"] == "refused:state_not_proposed"
    m, route = stored_route(origin="registry")
    assert tr.approve_batch(FakeEngine(route), m, {"synthetic-topic-x"}, set(), ACTOR, CFG_OFF)["outcomes"]["synthetic-topic-x"] == "refused:origin_not_induction"
    m, route = stored_route(label="Other")
    m["proposals"][0]["label"] = "Synthetic Topic X"
    r = tr.approve_batch(FakeEngine(route), m, {"synthetic-topic-x", "nope-key", "domain:nope-dom"}, set(), ACTOR, CFG_OFF)
    assert r["outcomes"]["nope-key"] == "refused:unknown_item" and r["outcomes"]["domain:nope-dom"] == "refused:unknown_item"
    # refused keys never reach an UPDATE
    m, route = stored_route(topic_sha="f" * 64)
    eng = FakeEngine(route)
    tr.approve_batch(eng, m, {"synthetic-topic-x"}, set(), ACTOR, cfg(approve_enabled=True), apply=True)
    assert not any(s.startswith("UPDATE") for s in eng.sqls())


def test_approve_apply_order_set_config_local_and_domain_warning():
    m, route = stored_route()
    eng = FakeEngine(route)
    r = tr.approve_batch(eng, m, {"domain:synthetic-new-domain", "synthetic-sub-y", "synthetic-topic-x"}, set(), ACTOR,
                         cfg(approve_enabled=True), apply=True)
    assert r["applied"] and eng.opened == ["begin"] and r["warnings"] == []
    assert set(r["outcomes"].values()) == {"approved"}
    ups = [(s, p) for s, p in eng.calls if s.startswith("UPDATE")]
    seq = [(re.search(r"vera_vjay\.(\w+)", s).group(1), p.get("slug")) for s, p in ups]
    assert seq == [("tax_domains", "synthetic-new-domain"), ("tax_topics", "synthetic-topic-x"), ("tax_anchors", "synthetic-topic-x"),
                   ("tax_anchors", "synthetic-topic-x"), ("tax_subtopics", "synthetic-sub-y"), ("tax_anchors", "synthetic-sub-y"),
                   ("tax_anchors", "synthetic-sub-y")]
    # (4) each anchor activation is preceded by retiring the old active anchor (unique index: one active per entity)
    kinds = [("retire" if "'retired'" in s else "activate") for s, _p in ups if "tax_anchors" in s]
    assert kinds == ["retire", "activate", "retire", "activate"]
    assert all(p["ns"] == "active" for s, p in ups if "'retired'" not in s)
    sets = [(i, p) for i, (s, p) in enumerate(eng.calls) if "set_config" in s]
    assert [p for _i, p in sets] == [{"n": "vera.tax_actor", "v": ACTOR}, {"n": "vera.tax_batch", "v": B}]
    first_update = min(i for i, (s, _p) in enumerate(eng.calls) if s.startswith("UPDATE"))
    assert max(i for i, _p in sets) < first_update
    # a topic approved while its new domain is still proposed: warning
    r = tr.approve_batch(FakeEngine(route), m, {"synthetic-topic-x"}, set(), ACTOR, CFG_OFF)
    assert r["warnings"] == [{"key": "synthetic-topic-x", "warning": "domain still proposed"}]
    # reject
    eng = FakeEngine(route)
    r = tr.approve_batch(eng, m, set(), {"synthetic-topic-x"}, ACTOR, cfg(approve_enabled=True), apply=True)
    assert r["outcomes"] == {"synthetic-topic-x": "rejected"}
    assert all(p["ns"] == "rejected" for s, p in eng.calls if s.startswith("UPDATE"))  # rejects never retire an active anchor


def test_approve_update_race_rolls_back():
    m, route = stored_route()

    def racing(sql, p):
        return FakeResult(rowcount=0) if sql.startswith("UPDATE vera_vjay.tax_topics") else route(sql, p)
    with pytest.raises(tr.TaxRefused):
        tr.approve_batch(FakeEngine(racing), m, {"synthetic-topic-x"}, set(), ACTOR, cfg(approve_enabled=True), apply=True)


# --------------------------------------------------------------------------------------------------------------- sync
def sync_route(sha, oref="topreg:synthetic-key-a", kind="topics"):
    return lambda sql, p: FakeResult(rows=[(kind, "synthetic-key-a", oref, sha)]) if "origin_ref LIKE" in sql else None


def test_sync_anchors_seed_replace_unchanged_missing():
    data = {"entries": [{"id": "synthetic-key-a", "anchor_text": ANCHOR}, {"id": "synthetic-key-b", "anchor_text": ANCHOR},
                        {"id": "synthetic-key-c", "anchor_text": "short"}]}
    r = tr.sync_anchors(FakeEngine(sync_route(None)), data, CFG_OFF)
    assert r["seed"] == ["synthetic-key-a"] and r["missing_entity"] == ["synthetic-key-b"] and r["invalid"] == ["synthetic-key-c"] and not r["applied"]
    r = tr.sync_anchors(FakeEngine(sync_route(tr.anchor_sha256(ANCHOR, None))), data, CFG_OFF)
    assert r["unchanged"] == 1 and r["seed"] == [] and r["replace"] == []
    r = tr.sync_anchors(FakeEngine(sync_route("0" * 64)), data, CFG_OFF)
    assert r["replace"] == ["synthetic-key-a"]
    with pytest.raises(tr.TaxRefused):
        tr.sync_anchors(FakeEngine(sync_route(None)), data, CFG_OFF, apply=True)
    with pytest.raises(tr.TaxRefused):
        tr.sync_anchors(FakeEngine(present=False), data, CFG_OFF)
    absent = FakeEngine(sync_route(None), present=False)
    with pytest.raises(tr.TaxRefused):
        tr.sync_anchors(absent, data, cfg(sync_enabled=True), apply=True)
    assert not any("tax_anchors a" in s or s.startswith(("INSERT", "UPDATE")) for s in absent.sqls())


def test_sync_anchors_apply_retires_before_insert_and_never_creates_entities():
    data = {"entries": [{"id": "synthetic-key-a", "anchor_text": ANCHOR}]}
    eng = FakeEngine(sync_route("0" * 64))
    r = tr.sync_anchors(eng, data, cfg(sync_enabled=True), apply=True)
    assert r["applied"] and r["replace"] == ["synthetic-key-a"]
    w = [s for s in eng.sqls() if s.startswith(("UPDATE", "INSERT"))]
    assert w[0].startswith("UPDATE vera_vjay.tax_anchors SET state = 'retired'") and w[1].startswith("INSERT INTO vera_vjay.tax_anchors")
    ins = [c for c in eng.calls if c[0].startswith("INSERT")][0]
    assert ins[1] == {"slug": "synthetic-key-a", "text": ANCHOR, "oref": "topreg:synthetic-key-a"}
    assert not any("INSERT INTO vera_vjay.tax_topics" in s or "tax_subtopics (" in s or "tax_domains (" in s for s in eng.sqls())
    eng = FakeEngine(sync_route(None))
    tr.sync_anchors(eng, data, cfg(sync_enabled=True), apply=True)
    assert not any(s.startswith("UPDATE") for s in eng.sqls())
    bad = lambda sql, p: FakeResult(rowcount=0) if sql.startswith("INSERT") else sync_route(None)(sql, p)
    with pytest.raises(tr.TaxRefused):
        tr.sync_anchors(FakeEngine(bad), data, cfg(sync_enabled=True), apply=True)


def test_drift_report():
    a = {"entries": [{"id": "synthetic-one", "label": "One", "anchor_text": ANCHOR}, {"id": "synthetic-two", "label": "Two", "anchor_text": ANCHOR},
                     {"id": "synthetic-only-a", "label": "X", "anchor_text": ANCHOR}]}
    b = tr.registry_data_from_rows([row(slug="synthetic-one", label="one "), row(slug="synthetic-two", label="Deux", anchor=ANCHOR + " changed"),
                                    row(slug="synthetic-only-b")])
    d = tr.drift_report(a, b)
    assert d == {"only_in_topreg": ["synthetic-only-a"], "only_in_tax": ["synthetic-only-b"], "label_differs": ["synthetic-two"],
                 "anchor_differs": ["synthetic-two"], "compared": 2}
    assert tr.drift_report(None, None)["compared"] == 0


# ----------------------------------------------------------------------------------------------------------- CLI
def load_cli():
    spec = importlib.util.spec_from_file_location("_t_tax_cli", ROOT / "scripts" / "tax_registry_cli.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_cli(argv, engine):
    cli, lines = load_cli(), []
    rc = cli.main(argv, engine=engine, out=lines.append)
    return rc, "\n".join(lines)


def test_cli_exit_codes(tmp_path):
    assert run_cli([], FakeEngine())[0] == 2
    assert run_cli(["show", "--mode", "bogus"], FakeEngine())[0] == 2
    assert run_cli(["approve", "--manifest", "m", "--actor", "owner:x", "--approve", "BAD KEY"], FakeEngine())[0] == 2
    rc, out = run_cli(["show", "--mode", "active"], FakeEngine(present=False))
    assert rc == 3 and "022" in out
    eng = FakeEngine(lambda sql, p: FakeResult(rows=[row()]) if "tax_registry_" in sql else None)
    rc, out = run_cli(["show", "--mode", "proposed"], eng)
    assert rc == 0 and "synthetic-topic-a" in out
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(manifest(prop("synthetic-topic-x"))))
    rc, out = run_cli(["propose", "--manifest", str(mp)], FakeEngine())
    assert rc == 0 and '"applied": false' in out
    rc, out = run_cli(["propose", "--manifest", str(mp), "--apply"], FakeEngine())  # shipped config: write_proposals false
    assert rc == 3 and "REFUSED" in out
    mp.write_text(json.dumps(manifest(prop("Bad Slug!"))))
    rc, out = run_cli(["propose", "--manifest", str(mp), "--apply"], FakeEngine())
    assert rc == 3 and "invalid_slug" in out
    rc, out = run_cli(["propose", "--manifest", str(tmp_path / "nope.json")], FakeEngine())
    assert rc == 4 and "CANNOT_VERIFY" in out
    m, route = stored_route()
    mp.write_text(json.dumps(m))
    rc, out = run_cli(["approve", "--manifest", str(mp), "--approve", "synthetic-topic-x", "--actor", "owner:t"], FakeEngine(route))
    assert rc == 0 and "would_approve" in out
    rc, out = run_cli(["approve", "--manifest", str(mp), "--approve", "synthetic-topic-x", "--actor", "owner:t", "--apply"], FakeEngine(route))
    assert rc == 3  # approve_enabled ships false


# ------------------------------------------------------------------------------------------------------ static guards
SRC = (ROOT / "vera" / "tax_registry.py").read_text(encoding="utf-8")


def _sql_constants():
    out = []
    for name in dir(tr):
        if name.startswith("SQL_"):
            v = getattr(tr, name)
            out.extend(v.values() if isinstance(v, dict) else [v])
    return out


def test_every_sql_constant_touches_only_tax_objects_and_never_deletes_or_ddl():
    consts = _sql_constants()
    assert len(consts) > 20
    for sql in consts:
        for obj in re.findall(r"vera_vjay\.(\w+)", sql):
            assert obj.startswith("tax_"), (obj, sql)
        assert not re.search(r"\b(DELETE|TRUNCATE|DROP|ALTER|CREATE|GRANT|REVOKE|COPY)\b", sql, re.I), sql
        assert re.match(r"(SELECT|INSERT INTO|UPDATE)\b", sql), sql
        if "set_config" in sql:
            assert re.fullmatch(r"SELECT set_config\(:n, :v, true\)", sql)
        if sql.startswith("UPDATE"):
            assert re.match(r"UPDATE vera_vjay\.tax_\w+ SET state = ", sql)
    code = SRC.split('"""', 2)[2]
    assert code.count("set_config(") == 1 and "SQL_SET_CONFIG" in code and ", false)" not in code
    assert not re.search(r"\.execute\(\s*['\"f]", code)  # no inline SQL: every statement is a named constant


def test_no_hard_coded_taxonomy_names_and_no_importers():
    seed = json.loads((ROOT / "config" / "micro_topic_anchors.json").read_text(encoding="utf-8"))["entries"]
    names = {str(e[k]).casefold() for e in seed for k in ("id", "label", "field") if e.get(k)}
    assert names
    for p in (ROOT / "vera" / "tax_registry.py", ROOT / "scripts" / "tax_registry_cli.py"):
        low = p.read_text(encoding="utf-8").casefold()
        for n in names:
            assert n not in low, (p.name, n)
    me = Path(__file__).resolve()
    importers = []
    for base in ("vera", "scripts", "tests", "safety", "db", "pages"):
        for p in (ROOT / base).rglob("*.py"):
            if p.name in ("tax_registry.py", "tax_registry_cli.py") or p.resolve() == me:
                continue
            if re.search(r"(from vera import[^\n]*\btax_registry\b|import vera\.tax_registry|from vera\.tax_registry|\btax_registry_cli\b)", p.read_text(encoding="utf-8", errors="ignore")):
                importers.append(str(p.relative_to(ROOT)))
    for p in ROOT.glob("*.py"):
        assert "tax_registry" not in p.read_text(encoding="utf-8", errors="ignore"), p.name
    assert importers == []


def test_no_three_part_column_names_in_vera():
    """CONTRACT-CHECK-1108: tests/sql_contract_support.py cannot see vera_vjay.<table>.<column>; SQL string literals in vera/
    must use table aliases. Docstrings and comments (prose) are not SQL and are skipped, as the contract extractor skips them."""
    import ast
    import re
    from pathlib import Path
    from tests import sql_contract_support as sc
    pat = re.compile(r"vera_vjay\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_]")
    root = Path(__file__).resolve().parent.parent / "vera"
    hits = []
    for p in sorted(root.rglob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        skip = sc._docstring_ids(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip and pat.search(n.value):
                hits.append(f"{p.relative_to(root)}:{n.lineno}")
    assert hits == []
