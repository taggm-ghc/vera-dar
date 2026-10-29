"""SEMSIM-RELEASE-1028: release-time attestation of the category descriptions (OD-1028(38) addendum, (47), (48)).

RELEASE/BUILD TIME ONLY, on the owner's machine or the build host. NEVER run in, or import from, the API (the Render
API has 512 MB and the local embedding model must never load there). Needs requirements-release.txt.

What it does: refuses unless HF_HUB_OFFLINE=1 and config/semantic_similarity.json carries a numeric threshold_category
(still null = uncalibrated -> refuses, writes nothing); loads the pinned model through vera.semantic_similarity.warm()
(manifest-hash verified); computes the pairwise semantic similarity of every category description; refuses if the
maximum is >= the threshold; else writes config/category_attestation.json (descriptions hash, model manifest hash,
threshold, max pairwise score, tool version, timestamp). The runtime only compares the descriptions hash
(vera.scope_router.verify_attestation) when classifier_categories.json sets attestation_required true.

Process control, not authentication: anyone can compute the descriptions hash without the model, so the hash alone is
not proof the model ran. The owner COMMITS config/category_attestation.json as part of the release; at runtime the
file's model_manifest_sha256 and threshold_category must also equal config/semantic_similarity.json (threshold non-null).
The tool parses the table with check_attestation=False (vera.category_table), so the owner can edit descriptions and
re-attest while attestation_required stays true; scope_router (the API import) would refuse the stale table, this tool
never imports it.

Usage: HF_HUB_OFFLINE=1 VERA_SEMSIM_MODEL_DIR=<dereferenced model dir> python scripts/attest_category_descriptions.py
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOL_VERSION = "1"
OUT_PATH = ROOT / "config" / "category_attestation.json"
CATEGORIES_FILE = ROOT / "config" / "classifier_categories.json"


class AttestationRefused(Exception):
    pass


def build_attestation(rows, cfg: dict, embedder, model_manifest_sha256: str, now: str | None = None) -> dict:
    """Pure core (testable with a fake embedder): max pairwise score over `rows`, gated by the category threshold."""
    from vera import category_table as sr
    from vera import semantic_similarity as ss

    thr = cfg.get("threshold_category")
    if isinstance(thr, bool) or not isinstance(thr, (int, float)) or not -1 <= thr <= 1:
        raise AttestationRefused("threshold_category is null/uncalibrated or invalid; refusing to attest")
    cap = ss._cap(cfg)
    if cap is None:
        raise AttestationRefused("invalid max_input_chars")
    best, pair = None, None
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            s = ss._similarity_score(a.description, b.description, embedder, cap)
            if s is None:
                raise AttestationRefused(f"no score for {a.code} vs {b.code}")
            if best is None or s > best:
                best, pair = s, [a.code, b.code]
    if best is None:
        raise AttestationRefused("need at least two categories")
    if best >= thr:
        raise AttestationRefused(f"max pairwise score {best} ({pair[0]} vs {pair[1]}) >= threshold {thr}")
    return {
        "schema": sr.ATTESTATION_SCHEMA,
        "descriptions_sha256": sr.descriptions_sha256(rows),
        "model_manifest_sha256": model_manifest_sha256,
        "model_revision": cfg.get("model_revision"),
        "threshold_category": thr,
        "max_pairwise_score": best,
        "max_pair": pair,
        "tool_version": TOOL_VERSION,
        "attested_at": now or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main(environ=os.environ, out_path: Path = OUT_PATH) -> int:
    try:
        if environ.get("HF_HUB_OFFLINE") != "1":
            raise AttestationRefused("HF_HUB_OFFLINE=1 is required")
        from vera import category_table as sr
        from vera import semantic_similarity as ss

        cfg = dict(ss._load_config())
        thr = cfg.get("threshold_category")
        if thr is None:
            raise AttestationRefused("threshold_category is null (uncalibrated); refusing to attest")
        cfg["enabled"] = True  # in-memory only: the tool must load the model that the shipped config keeps disabled
        if not ss.warm(cfg, environ):
            raise AttestationRefused("model did not load (pins, VERA_SEMSIM_MODEL_DIR, manifest hash)")
        ad = ss._adapter(cfg, environ)
        raw = json.loads(CATEGORIES_FILE.read_text(encoding="utf-8"))
        rows = sr.parse_categories(raw, check_attestation=False)  # D-D: re-attest even when the table is stale
        att = build_attestation(rows, cfg, ad, cfg["model_sha256"])
        out_path.write_text(json.dumps(att, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"attested: max pairwise {att['max_pairwise_score']} < threshold {att['threshold_category']}; wrote {out_path.name}")
        return 0
    except AttestationRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # fail verbosely, never write a partial attestation
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
