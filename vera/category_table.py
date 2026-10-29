"""Category table parser and release-time attestation check (split from scope_router, SEMSIM-RELEASE-1028 D-D).

Pure functions, no module-level load, no embedding library, no model. scope_router loads the shipped table at import
and enforces the attestation there; the release tool (scripts/attest_category_descriptions.py) imports THIS module and
parses with check_attestation=False so the owner can re-attest after editing descriptions.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


class ScopeConfigError(ValueError):
    """config/scope_router.json is missing, malformed or selects an unimplemented mode."""


DISPOSITIONS = ("in_scope", "off_topic", "refuse", "unsupported_use")  # the dispositions the code implements
REFUSING_DISPOSITIONS = frozenset({"refuse", "unsupported_use"})  # never answered, listed as normal research, or acquired
# Owner rule 2026-10-28 ("a new C entry must have a significantly dissimlar description from all other descriptions").
# Calibration (config description_similarity_note; measure = description_similarity): on the six shipped descriptions
# the pairwise max is 0.203 (harmful vs rule_change); 0.50 = max + 0.297 margin. Re-calibrate when a row is added.
DESCRIPTION_SIMILARITY_MAX = 0.50
_DESC_STOPWORDS = frozenset("a an the and or of to for in on at by with about any your you only is are be it this that "
                            "as from defined below".split())  # "defined below": a prompt cross-reference, not meaning
_DESC_SUFFIXES = ("ations", "ation", "ising", "ized", "ised", "ing", "ers", "ed", "es", "ly", "s")
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")
_CODE_RE = re.compile(r"[a-z][a-z_]{1,31}")


@dataclass(frozen=True)
class Category:
    code: str
    description: str
    disposition: str
    refusal_rank: int | None
    aliases: tuple[str, ...]


def _desc_stem(w: str) -> str:
    for suf in _DESC_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def _desc_tokens(t: str) -> frozenset:
    return frozenset(re.findall(r"[a-z]+", t.lower())) - _DESC_STOPWORDS


def _jaccard(x: frozenset, y: frozenset) -> float:
    return len(x & y) / len(x | y) if x | y else 1.0


def _trigrams(t: str) -> frozenset:
    s_ = " ".join(sorted(_desc_tokens(t)))
    return frozenset(s_[k:k + 3] for k in range(len(s_) - 2))


def description_similarity(a: str, b: str) -> float:
    """OOE-APRIME-C-1028 D1 (review): max of (a) Jaccard of suffix-stemmed content tokens and (b) character-trigram
    Jaccard over the sorted content tokens; lowercased, stopwords removed. Deterministic, stdlib only."""
    stem = _jaccard(frozenset(map(_desc_stem, _desc_tokens(a))), frozenset(map(_desc_stem, _desc_tokens(b))))
    return max(stem, _jaccard(_trigrams(a), _trigrams(b)))


ATTESTATION_SCHEMA = "vera-category-attestation-v1"


def descriptions_sha256(rows) -> str:
    """SEMSIM-RELEASE-1028: sha256 of the canonical "code TAB description" lines sorted by code. Shared by the release-time
    attestation tool (scripts/attest_category_descriptions.py) and the runtime check, so both hash the same bytes."""
    pairs = sorted((r.code, r.description) for r in rows)
    return hashlib.sha256("\n".join(f"{c}\t{d}" for c, d in pairs).encode("ascii")).hexdigest()


def verify_attestation(att, rows, sem_cfg=None) -> None:
    """Runtime check (no model, no embedding library): the attestation's description hash must equal the hash of the
    current descriptions. Any other fault is also a ScopeConfigError (fail closed). The score, threshold and model hash
    are release-time facts and are NOT recomputed here. D-C: it is a PROCESS control, not authentication: anyone can
    compute the hash without the model, so the owner commits the attestation file as part of the release and the hash
    alone is not proof the model ran. The file's model_manifest_sha256 and threshold_category must also equal the
    values in config/semantic_similarity.json (sem_cfg; the threshold must be non-null)."""
    where = "category_attestation.json"
    if not isinstance(att, dict) or att.get("schema") != ATTESTATION_SCHEMA:
        raise ScopeConfigError(f"{where}: missing or unknown schema")
    want = att.get("descriptions_sha256")
    if not isinstance(want, str) or want != descriptions_sha256(rows):
        raise ScopeConfigError(f"{where}: descriptions changed since attestation (re-run the release-time attestation tool)")
    thr = sem_cfg.get("threshold_category") if isinstance(sem_cfg, dict) else None
    if isinstance(thr, bool) or not isinstance(thr, (int, float)):
        raise ScopeConfigError(f"{where}: threshold_category in semantic_similarity.json is null or invalid")
    if att.get("threshold_category") != thr or isinstance(att.get("threshold_category"), bool):
        raise ScopeConfigError(f"{where}: threshold_category differs from semantic_similarity.json (re-attest)")
    msha = sem_cfg.get("model_sha256")
    if not isinstance(msha, str) or att.get("model_manifest_sha256") != msha:
        raise ScopeConfigError(f"{where}: model_manifest_sha256 differs from semantic_similarity.json (re-attest)")


def parse_categories(raw, attestation=None, sem_cfg=None, check_attestation: bool = True, *,
                     where: str = "classifier_categories.json", runtime_codes: frozenset = frozenset(),
                     rowcheck_only: bool = False) -> tuple[Category, ...]:
    """Validate the category table; any fault is a ScopeConfigError (fail closed). Returns an immutable tuple.
    SEMSIM-RELEASE-1028: when raw["attestation_required"] is literally true (default off), `attestation` must match the
    current descriptions and config (verify_attestation). A present but non-bool flag is a ScopeConfigError. The release
    tool passes check_attestation=False to re-attest after editing descriptions.
    `runtime_codes` (CATTAB-1028): codes of DB rows with origin=runtime; they are exempt from the one-row-per-disposition
    rule for in_scope / off_topic (a runtime row can never be refusing, see scope_router.parse_db_rows). `rowcheck_only`
    (CATTAB-1028): field-level validation of a single (non-active) DB row version. CONVERGE-1028: CATTAB-1028's
    parse_categories extensions ported here when SEMSIM-RELEASE-1028 moved the parser out of scope_router."""
    rows = raw.get("categories") if isinstance(raw, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ScopeConfigError(f"{where}: 'categories' must be a non-empty list")
    if "attestation_required" in raw and not isinstance(raw["attestation_required"], bool):
        raise ScopeConfigError(f"{where}: attestation_required must be a literal true or false")
    out, labels = [], set()
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or set(r) != {"code", "description", "disposition", "refusal_rank", "aliases"}:
            raise ScopeConfigError(f"{where}: row {i} must have exactly code, description, disposition, refusal_rank, aliases")
        code, desc, disp, rank, aliases = (r["code"], r["description"], r["disposition"], r["refusal_rank"], r["aliases"])
        if not isinstance(code, str) or not _CODE_RE.fullmatch(code):
            raise ScopeConfigError(f"{where}: row {i} code {code!r} must match {_CODE_RE.pattern}")
        if code in labels:
            raise ScopeConfigError(f"{where}: duplicate code or alias {code!r}")
        labels.add(code)
        if _CTRL_RE.search(code) or (isinstance(desc, str) and _CTRL_RE.search(desc)):
            raise ScopeConfigError(f"{where}: row {i}: control characters are not allowed in codes or descriptions")
        if not isinstance(desc, str) or not desc.strip() or not desc.isascii() or any(ch in desc for ch in '"{}'):
            raise ScopeConfigError(f"{where}: {code}: description must be non-empty ASCII text without quotes or braces")
        if disp not in DISPOSITIONS:
            raise ScopeConfigError(f"{where}: {code}: unknown disposition {disp!r} (known: {DISPOSITIONS})")
        if disp in REFUSING_DISPOSITIONS:
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise ScopeConfigError(f"{where}: {code}: a refusing row needs a positive integer refusal_rank")
        elif rank is not None:
            raise ScopeConfigError(f"{where}: {code}: refusal_rank must be null for disposition {disp!r}")
        if not isinstance(aliases, list) or not all(isinstance(a, str) and _CODE_RE.fullmatch(a) for a in aliases):
            raise ScopeConfigError(f"{where}: {code}: aliases must be a list of label strings")
        out.append(Category(code, desc.strip(), disp, rank, tuple(aliases)))
    if rowcheck_only:  # CATTAB-1028: field-level validation of a single (non-active) DB row version
        return tuple(out)
    for c in out:  # aliases are checked after all codes are known: never a code, never twice
        for a in c.aliases:
            if a in labels:
                raise ScopeConfigError(f"{where}: alias {a!r} of {c.code} collides with a code or another alias")
            labels.add(a)
    ranks = [c.refusal_rank for c in out if c.refusal_rank is not None]
    if len(ranks) != len(set(ranks)):
        raise ScopeConfigError(f"{where}: refusal_rank values must be unique")
    for k in ("in_scope", "off_topic", "unsupported_use"):  # the code paths keyed on these need exactly one row each
        if sum(c.disposition == k and c.code not in runtime_codes for c in out) != 1 or \
                (k == "unsupported_use" and sum(c.disposition == k for c in out) != 1):
            raise ScopeConfigError(f"{where}: exactly one row must have disposition {k!r}")
    for i, a in enumerate(out):
        for b in out[i + 1:]:
            score = description_similarity(a.description, b.description)
            if score >= DESCRIPTION_SIMILARITY_MAX:
                raise ScopeConfigError(f"{where}: descriptions of {a.code} and {b.code} are too similar "
                                       f"(score {score:.3f} >= {DESCRIPTION_SIMILARITY_MAX})")
    if raw.get("attestation_required") is True and check_attestation:
        verify_attestation(attestation, out, sem_cfg)
    return tuple(out)
