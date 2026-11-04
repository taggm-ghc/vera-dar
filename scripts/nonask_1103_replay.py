#!/usr/bin/env python3
"""NONASK-1103 WP-E: offline replay harness for per-flag calibration (remediation plan 5b).

Runs the EXISTING pure functions (vera.fulldoc_admission.admit, vera.source_type.classify,
vera.search_providers.ddg_identifier) over synthetic fixtures in tests/fixtures/nonask_1103/. No network, DB, LLM or
environment reads. Metrics carry 95% Wilson intervals; calibration and holdout halves are reported separately (split is
a fixed-seed hash recorded in the fixtures, fixed before any tuning). Output is an offline fixture baseline, not live.
SYNTHETIC FIXTURES; real-data calibration (plan 5b) still required before any flag flip.
Split = fixed-seed hash of the item's cluster (near-duplicate phrasings share a half); scenarios follow their question's cluster.
Items and passages marked "ambiguous": true are excluded from every metric and listed under "ambiguous" for the owner spot-check.

  python3 scripts/nonask_1103_replay.py [--split calibration|holdout|all] [--json] [--set admission.min_relevance=0.5]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIX = ROOT / "tests" / "fixtures" / "nonask_1103"
SEED = "nonask1103-v1"
Z = 1.959963984540054


CAVEAT = "synthetic fixtures; real-data calibration (plan 5b) still required before any flag flip"


def split_of(cluster_id: str) -> str:
    """Split of a cluster id (items outside a named cluster use their own id as the cluster)."""
    h = int(hashlib.sha256(f"{SEED}:{cluster_id}".encode()).hexdigest(), 16)
    return "calibration" if h % 2 == 0 else "holdout"


def wilson(k: int, n: int, z: float = Z) -> dict:
    """95% Wilson score interval for k successes in n trials (n == 0 -> rate and bounds None)."""
    if n <= 0:
        return {"k": k, "n": n, "rate": None, "lo": None, "hi": None}
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": k, "n": n, "rate": round(p, 4), "lo": round(max(0.0, c - h), 4), "hi": round(min(1.0, c + h), 4)}


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def _pick(items: list, split: str) -> list:
    return [i for i in items if split == "all" or i["split"] == split]


class FakeCorpus:
    """Duck-typed corpus for admit(): synthetic documents, every passage offered as a candidate span."""

    def __init__(self, scen: dict):
        self.status, self.source_kind = "ok", "bundle"
        self.texts, self.acquired, self.cands, self.labels, self.ambiguous = {}, [], [], {}, set()
        for d in scen["docs"]:
            text, ranges = "", []
            for n, p in enumerate(d["passages"]):
                start = len(text)
                text += p["text"]
                sid = f"{d['id']}:{n}"
                self.cands.append({"span_id": sid, "document_id": d["id"], "start": start, "end": len(text),
                                   "text": p["text"], "score": round(1.0 - 0.01 * len(self.cands), 3)})
                self.labels[sid] = bool(p["on_topic"])
                if p.get("ambiguous"):
                    self.ambiguous.add(sid)
                text += "\n\n"
            self.texts[d["id"]] = text
            self.acquired.append({
                "document_id": d["id"], "canonical_source_id": d["url"], "title": d["title"], "url": d["url"],
                "doi": None, "published": "2024", "authors": ["Synthetic, A."], "licence": "CC-BY",
                "licence_url": "https://creativecommons.org/licenses/by/4.0/", "licence_decision": "allow",
                "attribution": "synthetic", "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "chars": len(text), "evidence_ranges": [[0, len(text)]]})

    def text(self, did):
        return self.texts[did]

    def retrieve(self, q, k):
        return list(self.cands)

    def inspection(self):
        return {i: {"chunks_inspected": 3} for i in self.texts}


def load_fulldoc_cfg(overrides: list[str] | None = None) -> dict:
    cfg = json.loads((ROOT / "config" / "fulldoc_answer.json").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    for ov in overrides or ():
        key, _, raw = ov.partition("=")
        try:
            val = json.loads(raw)
        except ValueError:
            val = raw
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


def run_admission(scenarios: list, cfg: dict) -> dict:
    from vera.fulldoc_admission import admit
    tp = fp = fn = tn = 0
    per, leak, noans_ok, noans_n, amb = [], 0, 0, 0, []
    for s in scenarios:
        c = FakeCorpus(s)
        b = admit(s["question"], c, cfg)
        adm = {a["span_id"] for a in b.admitted}
        # merged passages report their member ids too
        for a in b.admitted:
            adm.update(a.get("merged_span_ids") or ())
        stp = sfp = sfn = stn = 0
        for sid, on in c.labels.items():
            if sid in c.ambiguous:
                amb.append({"scenario": s["id"], "span_id": sid, "admitted": sid in adm})
                continue
            if on and sid in adm:
                stp += 1
            elif on:
                sfn += 1
            elif sid in adm:
                sfp += 1
            else:
                stn += 1
        tp, fp, fn, tn = tp + stp, fp + sfp, fn + sfn, tn + stn
        has_on = any(v for k, v in c.labels.items() if k not in c.ambiguous)
        if not has_on:  # negative scenario: the right outcome is that nothing is admitted
            noans_n += 1
            noans_ok += 0 if adm else 1
        leak += 1 if sfp else 0
        per.append({"id": s["id"], "status": b.status, "tp": stp, "fp": sfp, "fn": sfn, "tn": stn})
    return {
        "span_precision": wilson(tp, tp + fp),
        "on_topic_recall": wilson(tp, tp + fn),
        "off_topic_admit_rate": wilson(fp, fp + tn),
        "scenarios_with_off_topic_admitted": wilson(leak, len(scenarios)),
        "no_relevant_passage_scenarios_admit_nothing": wilson(noans_ok, noans_n),
        "per_scenario": per, "ambiguous_excluded": amb}


def run_source_type(questions: list) -> dict:
    from vera import source_type
    acq = json.loads((ROOT / "config" / "auto_acquire.json").read_text(encoding="utf-8"))
    labels = ("technical", "scholarly", "general", "mixed")
    routes = ("technical", "scholarly", "general", "mixed")
    cm = {l: {r: 0 for r in routes} for l in labels}
    for q in (x for x in questions if not x.get("ambiguous")):
        r = source_type.classify(q["text"], acq)["route"]
        cm[q["label"]][r if r in routes else "mixed"] += 1
    n_t, n_s, n_g, n_m = (sum(cm[l].values()) for l in labels)
    return {
        "router_enabled_in_config": source_type.enabled(acq),
        "confusion_label_by_route": cm,
        "practitioner_recall_route_technical": wilson(cm["technical"]["technical"], n_t),
        "practitioner_not_general": wilson(n_t - cm["technical"]["general"], n_t),
        "scholarly_routed_general": wilson(cm["scholarly"]["general"], n_s),
        "scholarly_routed_scholarly": wilson(cm["scholarly"]["scholarly"], n_s),
        "general_routed_general": wilson(cm["general"]["general"], n_g),
        "mixed_routed_mixed": wilson(cm["mixed"]["mixed"], n_m)}


def run_identifiers(records: list) -> dict:
    from vera.search_providers import ddg_identifier
    ok_h = n_h = ok_o = n_o = 0
    miss = []
    for r in records:
        got = (ddg_identifier(r["url"]) or {}).get("kind")
        hit = got == r["expected_kind"]
        if r["expected_kind"] == "handle":
            n_h, ok_h = n_h + 1, ok_h + hit
        else:
            n_o, ok_o = n_o + 1, ok_o + hit
        if not hit:
            miss.append({"id": r["id"], "expected": r["expected_kind"], "got": got})
    return {"handle_recognised": wilson(ok_h, n_h), "non_handle_unchanged": wilson(ok_o, n_o), "misses": miss}


def run_subquery_search(scenarios: list) -> dict:
    """NONASK-1103 S3 (plan 6.2, CTQ-S8): the pre-fetch relevance tiers (vera.search_query.relevance_tiers) of each
    scenario's records, BEFORE = judged against the whole question's concepts as one pool (today), AFTER = judged
    against the record's own sub-query with the named entity central (S2/S3). The AFTER metric of a single-topic control
    is today's (no parts). Metrics: tier-0 recall of relevant records (must not drop), off-entity records kept at tier 0
    (must fall), and single-topic controls whose tiers changed (must be zero)."""
    from vera import search_query as sq
    cfg = sq._default_terms_cfg()
    gen = json.loads((ROOT / "config" / "auto_acquire.json").read_text(encoding="utf-8"))["search"]["openalex_profile"]["generic_terms"]
    cnt = {"b_rel0": 0, "a_rel0": 0, "rel": 0, "b_off0": 0, "a_off0": 0, "off": 0, "ctrl_changed": 0, "ctrl": 0}
    for s in scenarios:
        recs = s["records"]
        text = [r["title"] + " " + r.get("snippet", "") for r in recs]
        before = sq.relevance_tiers(sq.key_concepts(s["question"], cfg, extra_generic=gen), text)
        if s["parts"]:
            after = [None] * len(recs)
            for k, part in enumerate(s["parts"]):
                idx = [i for i, r in enumerate(recs) if r["part"] == k]
                kc = sq.key_concepts(part["text"], cfg, extra_generic=gen, central=part["entities"])
                for i, t in zip(idx, sq.relevance_tiers(kc, [text[i] for i in idx])):
                    after[i] = t
        else:
            after = list(before)
            cnt["ctrl"] += 1
            cnt["ctrl_changed"] += 1 if after != before else 0
        for r, b, a in zip(recs, before, after):
            key = "rel" if r["relevant"] else "off"
            if s["parts"]:
                cnt[key] += 1
                cnt["b_" + key + "0"] += b == 0
                cnt["a_" + key + "0"] += a == 0
    return {"relevant_tier0_before": wilson(cnt["b_rel0"], cnt["rel"]), "relevant_tier0_after": wilson(cnt["a_rel0"], cnt["rel"]),
            "off_entity_tier0_before": wilson(cnt["b_off0"], cnt["off"]), "off_entity_tier0_after": wilson(cnt["a_off0"], cnt["off"]),
            "single_topic_controls_changed": wilson(cnt["ctrl_changed"], cnt["ctrl"]),
            "regression": bool(cnt["a_rel0"] < cnt["b_rel0"] or cnt["ctrl_changed"])}


def run_subquery_adversarial(scenarios: list) -> dict:
    """NONASK-1103 S3, adversarial: parts come from the REAL vera.subquery.decompose (not supplied). BEFORE = today
    (question-level concepts, one pool). MID = the first S3 version (part concepts, entity central, no disambiguation).
    AFTER = shipped (part_tiers with registry context: full alias, or name/parent plus a context term, generic phrases
    removed). Records labelled ambiguous are excluded. Controls run the same code (no split -> unchanged by design;
    their n_parts is reported, a split of a generic-name question would be a decomposer defect)."""
    from vera import search_query as sq, subquery
    cfg = sq._default_terms_cfg()
    gen = json.loads((ROOT / "config" / "auto_acquire.json").read_text(encoding="utf-8"))["search"]["openalex_profile"]["generic_terms"]
    reg = subquery.load_entity_registry()
    ctx_for = lambda e: subquery.entity_context(e, reg)  # noqa: E731
    cnt = {k: 0 for k in ("rel", "off", "b_rel", "m_rel", "a_rel", "b_off", "m_off", "a_off")}
    parts_n, unmapped = {}, 0
    ctrl = {"rel_n": 0, "rel_t0": 0, "off_n": 0, "off_t0": 0, "same": 0}
    for s in scenarios:
        plan = subquery.decompose(s["question"])
        ents_of = [tuple(p.entities) or ((p.label,) if plan.rule == "entity_list" and p.label else ()) for p in plan.parts]
        parts = [(p.text, e) for p, e in zip(plan.parts, ents_of)]
        parts_n[s["id"]] = len(parts)
        recs, part_of = [], []
        for r in s["records"]:  # a record naming several entities (a list) is filed under each of them
            names = r["entity"] if isinstance(r.get("entity"), list) else [r.get("entity")]
            ks = [next((i for i, p in enumerate(plan.parts) if n and any(n.lower() in e.lower() or e.lower() in n.lower()
                                                                          for e in ents_of[i])), None) for n in names]
            for k in dict.fromkeys(ks):
                recs.append(r)
                part_of.append(k)
                unmapped += 1 if (len(parts) > 1 and k is None) else 0
        text = [r["title"] + " " + r.get("snippet", "") for r in recs]
        before = sq.relevance_tiers(sq.key_concepts(s["question"], cfg, extra_generic=gen), text)
        after = sq.part_tiers(s["question"], text, part_of, parts, cfg=cfg, extra_generic=gen, ctx_for=ctx_for)
        mid = list(before)
        if len(parts) > 1:
            mid = sq.part_tiers(s["question"], text, part_of, parts, cfg=cfg, extra_generic=gen, ctx_for=None)
        for r, b, m, a in zip(recs, before, mid, after):
            if r["label"] == "ambiguous":
                continue
            key = "rel" if r["label"] == "relevant" else "off"
            if s["kind"] == "control":  # no split: same code path both ways, reported apart (not a S3 effect)
                ctrl[key + "_n"] += 1
                ctrl[key + "_t0"] += a == 0
                ctrl["same"] += a == b
                continue
            cnt[key] += 1
            cnt["b_" + key] += b == 0
            cnt["m_" + key] += m == 0
            cnt["a_" + key] += a == 0
    w = lambda k, n: wilson(cnt[k], cnt[n])  # noqa: E731
    return {"relevant_tier0_before": w("b_rel", "rel"), "relevant_tier0_mid_no_disambiguation": w("m_rel", "rel"),
            "relevant_tier0_after": w("a_rel", "rel"), "off_tier0_before": w("b_off", "off"),
            "off_tier0_mid_no_disambiguation": w("m_off", "off"), "off_tier0_after": w("a_off", "off"),
            "controls_tier0_unchanged_by_design": {"relevant_tier0": f"{ctrl['rel_t0']}/{ctrl['rel_n']}", "off_tier0": f"{ctrl['off_t0']}/{ctrl['off_n']}", "records_with_same_tier": ctrl["same"]},
            "n_parts": parts_n, "records_not_mapped_to_a_part": unmapped,
            "regression": bool(cnt["a_rel"] < cnt["b_rel"] or cnt["a_off"] >= cnt["b_off"])}


def replay(split: str = "all", overrides: list[str] | None = None) -> dict:
    qs, sc, ids = _pick(_load("questions.json")["questions"], split), _pick(_load("scenarios.json")["scenarios"], split), \
        _pick(_load("identifiers.json")["records"], split)
    sqs = _pick(_load("subquery_search.json")["scenarios"], split)
    adv = _pick(_load("subquery_adversarial.json")["scenarios"], split)
    amb_q = [q["id"] for q in qs if q.get("ambiguous")]
    return {"label": "offline fixture baseline, not live", "caveat": CAVEAT, "split": split,
            "ambiguous": {"questions": amb_q, "passages": [a for a in run_admission(sc, load_fulldoc_cfg(overrides))["ambiguous_excluded"]]}, "overrides": overrides or [],
            "n": {"questions": len(qs), "scenarios": len(sc), "identifiers": len(ids), "subquery_scenarios": len(sqs), "subquery_adversarial": len(adv)},
            "flags": {"admission": run_admission(sc, load_fulldoc_cfg(overrides)),
                      "source_type_router": run_source_type(qs),
                      "handle_identifier": run_identifiers(ids),
                      "subquery_search": run_subquery_search(sqs),
                      "subquery_adversarial": run_subquery_adversarial(adv)}}


def _fmt(w: dict) -> str:
    return "n/a (n=0)" if w["rate"] is None else f"{w['rate']:.3f} [{w['lo']:.3f},{w['hi']:.3f}] {w['k']}/{w['n']}"


def render(rep: dict) -> str:
    out = [f"{rep['label']} | {rep['caveat']}", f"split={rep['split']} n={rep['n']} overrides={rep['overrides']}"]
    for flag, m in rep["flags"].items():
        out.append(f"[{flag}]")
        for k, v in m.items():
            if isinstance(v, dict) and "rate" in v:
                out.append(f"  {k}: {_fmt(v)}")
            elif k == "confusion_label_by_route":
                out.append(f"  {k}: {json.dumps(v)}")
            elif k not in ("per_scenario", "misses", "ambiguous_excluded", "n_parts"):
                out.append(f"  {k}: {v}")
    out.append(f"[ambiguous, excluded from metrics, for owner spot-check] {json.dumps(rep['ambiguous'])}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=("calibration", "holdout", "all"), default="all")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="a.b=JSON")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = replay(a.split, a.overrides)
    print(json.dumps(rep, indent=1) if a.json else render(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
