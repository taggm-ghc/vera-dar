"""VERA-DAR T1: in-memory research-domain registry (plan Part 0 s0.4.2; owner directive OD-K17bw s3, s4).

Pure Python: no DB, network or SQL. The reviewable persistence design lives in a separate, unexecuted migration;
nothing here touches it.

Semantics (directive s3):
  * Names are unique after normalization (NFKC, casefold, collapsed whitespace); registering a known name reuses
    the existing domain (created=False) rather than creating a near-duplicate.
  * A domain relates to many corpora (canonical_source_id) and many chunks (span_id); either asset may carry many
    domains. Corpus and chunk associations are stored and queried INDEPENDENTLY: labelling a corpus never implies
    labels on its chunks, a chunk may carry a domain its parent corpus lacks, and no parent link is consulted.
  * A duplicate association is idempotent: the call returns False and no second row is kept. Disassociating a
    missing association returns False.
  * Counts are DISTINCT per domain; distinct_totals() counts each asset once however many domains it carries.
  * UNCLASSIFIED means "no association recorded" (absent). It is NOT "irrelevant", "rejected" or "low quality";
    unclassified assets stay valid and must be included by any broader fallback.
  * Uncertain classification is kept as a proposal (record_uncertain); record_uncertain itself never registers a domain.
    The one auto-registration path is vera.domain_identify: a DERIVED domain (origin 'derived') is registered only after
    min_support documents share a candidate cluster, bounded by max_derived_domains (config; uncalibrated).

Deferred / defaults (Phase B RC-5, RC-8, RC-9): persistence (tables, wiring) and the sidebar/API exposure are
DEFERRED to later increments; the routing flag in config/domain_routing.json defaults OFF and its domain list is
empty. There are no module-level singletons: callers instantiate InMemoryDomainRegistry themselves.
Uniqueness key is Domain.name_norm = normalize_domain_name(name), matching the migration's name_norm UNIQUE column.

Config domain entries ({name, description}) may also carry an optional "terms" list; its values are set by the
owner (HD-DAR-4), not by this module, and this registry ignores it.

Domain membership is a classification aid only. It is NEVER an admission, credibility, quality or relevance signal
and must not be consulted by evidence admission or source qualification. Accordingly Domain carries no such field.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "domain_routing.json"
ORIGINS = ("config", "manual", "fixture", "derived")  # derived: content-identified by vera.domain_identify (R5)
UNCLASSIFIED = "unclassified"  # marker: no domain association recorded; absence, not irrelevance
CLASSIFIED = "classified"


def normalize_domain_name(name: str) -> str:
    """NFKC, casefold, collapse whitespace, strip. Empty or non-string raises ValueError."""
    if not isinstance(name, str):
        raise ValueError("domain name must be a string")
    out = " ".join(unicodedata.normalize("NFKC", name).casefold().split())
    if not out:
        raise ValueError("domain name is empty after normalization")
    return out


@dataclass(frozen=True)
class Domain:
    domain_id: int
    name: str
    name_norm: str  # uniqueness key (RC-5)
    description: str
    created_at: str
    origin: str  # one of ORIGINS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _need_actor(asserted_by: str) -> str:
    if not isinstance(asserted_by, str) or not asserted_by.strip():
        raise ValueError("asserted_by is required (who or what made the association)")
    return asserted_by.strip()


class InMemoryDomainRegistry:
    """Dict/set-backed registry. Association keys are (domain_id, asset_id); corpus and chunk maps never interact."""

    def __init__(self, clock: Callable[[], str] = _now):
        self._clock = clock
        self._next_id = 1
        self._by_id: dict[int, Domain] = {}
        self._by_norm: dict[str, int] = {}
        self._corpus: dict[tuple[int, int], str] = {}  # (domain_id, canonical_source_id) -> asserted_by
        self._chunk: dict[tuple[int, int], str] = {}   # (domain_id, span_id) -> asserted_by
        self._uncertain: list[dict] = []

    # --- registration and resolution ---
    def register(self, name: str, description: str = "", origin: str = "manual") -> tuple[Domain, bool]:
        """(domain, created). A normalized-name match is reused unchanged (first description and origin win)."""
        norm = normalize_domain_name(name)
        if origin not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}")
        existing = self._by_norm.get(norm)
        if existing is not None:
            return self._by_id[existing], False
        dom = Domain(self._next_id, " ".join(name.split()), norm, str(description or ""), self._clock(), origin)
        self._next_id += 1
        self._by_id[dom.domain_id] = dom
        self._by_norm[norm] = dom.domain_id
        return dom, True

    def get(self, ids_or_name: int | str) -> Domain | None:
        if isinstance(ids_or_name, bool):
            return None
        if isinstance(ids_or_name, int):
            return self._by_id.get(ids_or_name)
        if isinstance(ids_or_name, str):
            try:
                return self._by_id.get(self._by_norm.get(normalize_domain_name(ids_or_name), -1))
            except ValueError:
                return None
        return None

    def resolve(self, ids_or_names: Iterable[int | str]) -> tuple[list[Domain], list[str]]:
        """(known, unknown). Unknown inputs are preserved as text, never an error and never registered.
        Known domains are de-duplicated, in first-seen order."""
        known: list[Domain] = []
        unknown: list[str] = []
        for x in ids_or_names:
            d = self.get(x)
            if d is None:
                unknown.append(str(x))
            elif d not in known:
                known.append(d)
        return known, unknown

    def list_domains(self) -> list[Domain]:
        return [self._by_id[i] for i in sorted(self._by_id)]

    # --- associations (independent maps) ---
    def _need_domain(self, domain_id: int) -> None:
        if domain_id not in self._by_id:
            raise KeyError(f"unknown domain_id {domain_id!r}")

    def associate_corpus(self, domain_id: int, canonical_source_id: int, asserted_by: str) -> bool:
        """True if new; a duplicate returns False and keeps the original row. Never writes a chunk association."""
        self._need_domain(domain_id)
        key = (domain_id, canonical_source_id)
        if key in self._corpus:
            return False
        self._corpus[key] = _need_actor(asserted_by)
        return True

    def disassociate_corpus(self, domain_id: int, canonical_source_id: int) -> bool:
        return self._corpus.pop((domain_id, canonical_source_id), None) is not None

    def associate_chunk(self, domain_id: int, span_id: int, asserted_by: str) -> bool:
        """True if new; duplicate False. Never reads or writes corpus associations (no parent lookup)."""
        self._need_domain(domain_id)
        key = (domain_id, span_id)
        if key in self._chunk:
            return False
        self._chunk[key] = _need_actor(asserted_by)
        return True

    def disassociate_chunk(self, domain_id: int, span_id: int) -> bool:
        return self._chunk.pop((domain_id, span_id), None) is not None

    def domains_for_corpus(self, canonical_source_id: int) -> list[Domain]:
        return [self._by_id[d] for d, c in sorted(self._corpus) if c == canonical_source_id]

    def domains_for_chunk(self, span_id: int) -> list[Domain]:
        return [self._by_id[d] for d, s in sorted(self._chunk) if s == span_id]

    def corpora_for_domain(self, domain_id: int) -> list[int]:
        return sorted(c for d, c in self._corpus if d == domain_id)

    def chunks_for_domain(self, domain_id: int) -> list[int]:
        return sorted(s for d, s in self._chunk if d == domain_id)

    def association_actor(self, kind: str, domain_id: int, asset_id: int) -> str | None:
        """Who asserted an association (kind 'corpus' or 'chunk'); None if absent."""
        return {"corpus": self._corpus, "chunk": self._chunk}[kind].get((domain_id, asset_id))

    # --- unclassified marker (absence, not irrelevance) ---
    def corpus_state(self, canonical_source_id: int) -> str:
        return CLASSIFIED if self.domains_for_corpus(canonical_source_id) else UNCLASSIFIED

    def chunk_state(self, span_id: int) -> str:
        """Judged on the chunk's own associations only; a classified parent corpus does not matter."""
        return CLASSIFIED if self.domains_for_chunk(span_id) else UNCLASSIFIED

    # --- uncertain classification: kept as a proposal; record_uncertain never registers (derivation is vera.domain_identify's rule) ---
    def record_uncertain(self, label: str, note: str = "", candidates: Iterable[str] = ()) -> dict:
        """Keep an ambiguous proposal for later human or rule review. Does not create any domain."""
        if not isinstance(label, str) or not label.strip():
            raise ValueError("uncertain label is empty")
        rec = {"label": " ".join(label.split()), "note": str(note or ""),
               "candidates": [str(c) for c in candidates], "recorded_at": self._clock()}
        self._uncertain.append(rec)
        return dict(rec)

    def list_uncertain(self) -> list[dict]:
        return [dict(r) for r in self._uncertain]

    # --- coverage (DISTINCT counts) ---
    def coverage(self) -> list[dict]:
        """[{domain_id, name, description, corpus_count, chunk_count}]; [] for an empty registry.
        Counts are per-domain distinct assets. An asset in two domains counts once in each, so rows must not be
        summed for a total: use distinct_totals()."""
        return [{"domain_id": d.domain_id, "name": d.name, "description": d.description,
                 "corpus_count": len(self.corpora_for_domain(d.domain_id)),
                 "chunk_count": len(self.chunks_for_domain(d.domain_id))} for d in self.list_domains()]

    def distinct_totals(self) -> dict:
        """Each classified asset counted once across all domains, with the number of domains."""
        return {"domains": len(self._by_id),
                "corpora_classified": len({c for _, c in self._corpus}),
                "chunks_classified": len({s for _, s in self._chunk})}


def load_config(path: Path = CONFIG_PATH) -> dict:
    """The routing config; {} when missing or unreadable (fail closed)."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def routing_enabled(cfg: dict | None = None) -> bool:
    """Fail-closed flag: true only if the config value is the JSON boolean true."""
    cfg = load_config() if cfg is None else cfg
    return isinstance(cfg, dict) and cfg.get("enabled") is True


def load_registry(cfg: dict) -> InMemoryDomainRegistry:
    """Seed domains (origin 'config') from cfg['domains']; entries are a name or {name, description}.
    No associations are seeded. Malformed entries are skipped, not guessed."""
    reg = InMemoryDomainRegistry()
    raw = cfg.get("domains") if isinstance(cfg, dict) else None
    for item in raw if isinstance(raw, list) else []:
        name, desc = (item, "") if isinstance(item, str) else (
            (item.get("name"), item.get("description", "")) if isinstance(item, dict) else (None, ""))
        try:
            reg.register(name, desc or "", origin="config")  # type: ignore[arg-type]
        except ValueError:
            continue
    return reg
