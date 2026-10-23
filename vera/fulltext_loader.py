"""DARINT G1 + G2: load curated FULL-TEXT documents into VERA-DAR's corpus and hand them to DAR.

G1  Per manifest item: every rejection filter (below), then ONE transaction on the write-only account
    (vera_pipeline_wo): candidates -> canonical_sources -> sources, ids from INSERT ... RETURNING, provenance
    stored_text_kind "full_text". The transaction is rolled back unless the sources row is written, so no orphan
    candidate or canonical row is left. The DAR persist (spans, domain rows) runs INSIDE that same transaction (G2), so
    a failure there rolls the source back too; with --new-run the question/run rows join the first document's transaction,
    so a failed first document leaves no empty run. fetch_status is set on INSERT (no UPDATE). gate_a_decision is left NULL:
    Gate A (query fit) is an /ask-time gate and does not apply to a curated list (accepted divergence, INGEST-PLAN s4).
    The write-only account cannot read canonical_url or content, so existing rows are resolved through the READ-ONLY
    account: an existing canonical source (e.g. an abstract admitted by /ask) gets a NEW sources version with the full
    text; the same full text already stored (same content_hash) is "known".
G2  The new ids go into the doc for DarCorpus.ingest_document (domains inferred from CONTENT; the manifest's
    domain_hint is never read) and the resulting plan into dar_store.persist (INSERT-only DarWriter). The DarCorpus is
    first restored from the stored corpus (DarReader.load_snapshot) so identification sees existing state; stored
    full_text documents that have no evidence spans yet (an earlier run stopped after its sources commit, or ran
    --sources-only) are re-ingested and persisted (resume). One persist ctx, seeded from the snapshot, spans the run.

Filters (each refusal carries a reason code from config `reason_codes`): required manifest fields present and not a
placeholder; DOI well-formed; verifier_status VERIFIED; read_status a verbatim read; format accepted; stored kind
outside discovery_contract.json non_document_text_kinds; text file present, within size, sha256 equal to the
conversion record; conversion found a body; length, body length and body/abstract ratio above config minimums
(abstracts and snippets are refused); licence via vera.licence_gate.apply_licence_gate with the declared licence
as the CONTENT record (ND / no-educational-use / fee-only reject, missing licence held, plain NC allowed) plus the
arXiv rule on both URLs; a licence embedded in the document is checked too (rejecting marker = reject; a different
CC licence = hold). No item can skip a filter; there is no bypass flag.

No env reads, no URL handling, no engine construction here: engines are injected (scripts/ingest_fulltext.py).
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from vera import answer_contract
from vera.corpus_admission import LICENCE_OBS_INSERT, SCHEMA, WoCorpusStore, canonical_url, licence_observation_rows
from vera.dar_store import DarWriter
from vera.licence_gate import apply_licence_gate, licence_decision, reject_reason

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "fulltext_ingest.json"
ROUTING_CONFIG_PATH = ROOT / "config" / "domain_routing.json"


class Refused(Exception):
    """An item failed a filter. code is the configured reason code; str(self) is the reason."""

    def __init__(self, code: str, reason: str):
        super().__init__(reason)
        self.code = code


class IngestError(RuntimeError):
    """A write-path failure (DB state, preflight, persist accounting). Stops the run."""


def load_config(path: Path = CONFIG_PATH) -> dict:
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:  # fail verbosely: a loader without its rules must not run
        raise RuntimeError(f"full-text ingest config unreadable at {path}: {e!r}") from e
    if not isinstance(cfg, dict):
        raise RuntimeError(f"full-text ingest config at {path} is not an object")
    return cfg


def _refuse(cfg: dict, key: str, reason: str):
    raise Refused(cfg["reason_codes"][key], reason)


# --------------------------------------------------------------------------------------------- filters
def non_doi_identifier_ok(item: dict, cfg: dict) -> bool:
    """PLAN-NONDOI-1022: True only when doi is None AND item["identifier"] is an allowlisted kind (config
    non_doi_identifier_kinds, set ONLY by the acquisition overlay; the base config lists none) whose id fully matches
    the kind's anchored pattern and whose url matches url_pattern. Waives the doi requirement and doi_pattern only;
    every other filter still runs."""
    kinds = cfg.get("non_doi_identifier_kinds")
    ident = item.get("identifier")
    if item.get("doi") is not None or not isinstance(kinds, dict) or not isinstance(ident, dict):
        return False
    pat = kinds.get(str(ident.get("kind") or ""))
    return (isinstance(pat, str) and pat.startswith("^") and pat.endswith("$")
            and bool(re.fullmatch(pat, str(ident.get("id") or "")))
            and bool(re.match(cfg["url_pattern"], str(ident.get("url") or "").strip())))


def check_item(item: dict, cfg: dict, contract_path: Path = answer_contract.CONFIG_PATH) -> None:
    ph = re.compile(cfg["placeholder_pattern"])
    non_doi = non_doi_identifier_ok(item, cfg)  # PLAN-NONDOI-1022: waives ONLY the doi requirement + doi_pattern
    for f in cfg["required_item_fields"]:
        if f == "doi" and non_doi:
            continue
        v = item.get(f)
        if v is None or (isinstance(v, (list, dict)) and not v):
            _refuse(cfg, "missing_field", f"required field {f!r} is missing")
        if isinstance(v, str) and ph.search(v):
            _refuse(cfg, "placeholder", f"field {f!r} is a placeholder ({v!r})")
    for f in cfg["url_fields"]:
        if not re.match(cfg["url_pattern"], str(item.get(f) or "").strip()):
            _refuse(cfg, "bad_url", f"{f} is not a single https URL")
    if not non_doi and not re.match(cfg["doi_pattern"], str(item["doi"]).strip()):
        _refuse(cfg, "bad_doi", f"doi {item['doi']!r} is not well formed")
    if item.get("verifier_status") != cfg["verified_status"]:
        _refuse(cfg, "not_verified", f"verifier_status is {item.get('verifier_status')!r}, not {cfg['verified_status']}")
    if item.get("read_status") not in cfg["verbatim_read_statuses"]:
        _refuse(cfg, "not_verbatim", f"read_status {item.get('read_status')!r} is not a verbatim full read")
    if item.get("fulltext_format") not in cfg["accepted_formats"]:
        _refuse(cfg, "bad_format", f"fulltext_format {item.get('fulltext_format')!r} not accepted")
    kind = cfg["stored_text_kind"]
    if answer_contract.acquisition_state({"stored_text_kind": kind, "admission_passed": True}, contract_path) \
            != answer_contract.ACQ_ACQUIRED:
        _refuse(cfg, "non_document_kind", f"stored_text_kind {kind!r} is a non-document kind in the answer contract")


def _non_document_kinds(contract_path: Path = answer_contract.CONFIG_PATH) -> set:
    acq = answer_contract.load_config(contract_path).get("acquisition") or {}
    return {str(k) for k in (acq.get("non_document_text_kinds") or [])}


def check_declared_kind(declared: dict, cfg: dict, where: str, contract_path: Path = answer_contract.CONFIG_PATH) -> None:
    """Fail closed (FDC-1020 item 5), allowlist: a text_kind / stored_text_kind declared on the item or its
    conversion record must be a string equal (case-insensitively) to the configured full-document kind; anything
    else (non-string, a non-document kind, an unknown label such as 'abstract_only') is refused. Absent/None passes."""
    non_doc = _non_document_kinds(contract_path)
    allowed = str(cfg["stored_text_kind"]).strip().lower()
    for f in ("text_kind", "stored_text_kind"):
        v = declared.get(f)
        if v is None:
            continue
        if not isinstance(v, str) or v.strip().lower() in non_doc or v.strip().lower() != allowed:
            _refuse(cfg, "non_document_kind", f"{where} declares {f} {v!r}, not the full-document kind {allowed!r}")


def load_text(item: dict, base_dir: Path, cfg: dict) -> tuple[str, dict]:
    """texts/<id>.txt and its conversion record texts/<id>.meta.json (written by `ingest_fulltext.py convert`)."""
    base = Path(base_dir).resolve()
    tp = (base / item["text_file"]).resolve()
    if base not in tp.parents:
        _refuse(cfg, "text_missing", f"text_file {item['text_file']!r} resolves outside the manifest directory")
    mp = tp.with_suffix(".meta.json")
    if not tp.is_file() or not mp.is_file():
        _refuse(cfg, "text_missing", f"text file or conversion record missing for {item['document_id']}")
    raw = tp.read_bytes()
    if len(raw) > int(cfg["max_text_bytes"]):
        _refuse(cfg, "too_large", f"text is {len(raw)} bytes, above max_text_bytes {cfg['max_text_bytes']}")
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
        text = raw.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as e:
        _refuse(cfg, "text_missing", f"text or conversion record unreadable: {e!r}")
    if hashlib.sha256(raw).hexdigest() != meta.get("text_sha256"):
        _refuse(cfg, "text_hash", "text sha256 differs from the conversion record (edited after conversion?)")
    return text, meta


def check_content(text: str, meta: dict, cfg: dict) -> None:
    if not meta.get("has_body"):
        _refuse(cfg, "abstract_only", "conversion found no body: abstract/metadata only")
    if len(text) < int(cfg["min_fulltext_chars"]):
        _refuse(cfg, "too_short", f"text has {len(text)} chars, below min_fulltext_chars {cfg['min_fulltext_chars']}")
    body, abstract = int(meta.get("body_chars") or 0), int(meta.get("abstract_chars") or 0)
    if body < int(cfg["min_body_chars"]):
        _refuse(cfg, "abstract_only", f"body has {body} chars, below min_body_chars {cfg['min_body_chars']}")
    if abstract and body < float(cfg["min_body_to_abstract_ratio"]) * abstract:
        _refuse(cfg, "abstract_only", f"body ({body}) is not more than {cfg['min_body_to_abstract_ratio']}x the "
                                      f"abstract ({abstract})")


def _cc_family(url_or_id, cfg: dict) -> str | None:
    pats = cfg["cc_licence_patterns"]
    low = str(url_or_id or "").lower()
    m = re.search(pats["url"], low)
    if m:
        return f"cc-{m.group(1)}-{m.group(2)}"
    m = re.search(pats["id"], low)
    if not m:
        return None
    return "cc-" + re.sub(r"\s+", "-", m.group(1).strip()) + "-" + m.group(2)


def _public_domain_zero(rec: dict) -> bool:
    s = f"{rec.get('id') or ''} {rec.get('url') or ''}".lower()
    return "publicdomain/zero" in s or bool(re.search(r"\bcc0\b", s))


def check_licence(item: dict, meta: dict, cfg: dict, *, arxiv_evidence: dict | None = None) -> dict:
    """Returns the licence record for provenance; raises Refused on reject/hold.
    arxiv_evidence (ARX-SECREV-1023 R1): the version-bound licence evidence built ONLY by vera.arxiv_fulltext (route
    write path; read path: arxiv_fulltext.stored_evidence). It is passed through unchanged and is never read from or
    synthesised out of the item, the conversion record, a manifest or a discovery record."""
    lic = {"content": {"id": item.get("licence"), "url": item.get("licence_url"), "source": "manifest declared"}}
    if arxiv_evidence is not None:
        lic["arxiv_evidence"] = dict(arxiv_evidence)
    cand = {"url": item["fulltext_url"], "licence": lic, "stored_text_kind": cfg["stored_text_kind"],
            "gate_a_decision": None, "gate_a_rationale": ""}
    apply_licence_gate([cand])
    d = cand["licence_decision"]
    for u in (item.get("landing_url"),):
        d2, why2 = licence_decision({"url": u, "licence": lic, "stored_text_kind": cfg["stored_text_kind"]})
        if d2 == "reject":
            _refuse(cfg, "licence_reject", f"landing_url: {why2}")
    if d == "reject":
        _refuse(cfg, "licence_reject", cand["gate_a_rationale"].strip())
    if d != "allow":
        _refuse(cfg, "licence_hold", cand["gate_a_rationale"].strip())
    emb = (meta.get("meta") or {}).get("licence") or {}
    if emb.get("id") or emb.get("url"):
        why = reject_reason(emb.get("id"), emb.get("url"))
        if why:
            _refuse(cfg, "licence_reject", f"licence embedded in the document is {why}")
        fam_doc = _cc_family(emb.get("url"), cfg) or _cc_family(emb.get("id"), cfg)
        fam_man = _cc_family(item.get("licence_url"), cfg) or _cc_family(item.get("licence"), cfg)
        if fam_doc and fam_man and fam_doc != fam_man:
            _refuse(cfg, "licence_mismatch", f"document licence {fam_doc} differs from the declared {fam_man}: held")
        # WED-P17-VER item 4 ("unclear = hold"): with embedded_licence_unknown_holds (set ONLY by the automatic
        # acquisition overlay, vera/acq_config.ingest_cfg), an embedded licence whose family is not recognised (e.g. a
        # publisher user licence) is held. The curated base config does not set it, so the human-verified DAR-9 items
        # whose JATS embeds a non-CC marker ("open-access": eyring2016, lehner2020) keep their recorded decision.
        if cfg.get("embedded_licence_unknown_holds") is True and not fam_doc and not _public_domain_zero(emb):
            _refuse(cfg, "licence_hold", f"licence embedded in the document is not a recognised licence: held")
        lic["embedded"] = {"id": emb.get("id"), "url": emb.get("url"), "statement": emb.get("statement"),
                           "source": "document"}
    return {"licence": lic, "licence_decision": d, "licence_rationale": cand["gate_a_rationale"].strip()}


def check_item_declared(item: dict, cfg: dict) -> None:
    """The manifest-item level filters (pure; no I/O). Shared by prepare() and the DB-backed reader (R3)."""
    check_item(item, cfg)
    check_declared_kind(item, cfg, "manifest item")


def check_text_record(item: dict, text: str, meta: dict, cfg: dict, *, arxiv_evidence: dict | None = None) -> dict:
    """The text-level filters after the text and its conversion record are in hand (pure; no I/O): declared kind of the
    conversion record, body/length checks and the licence gate. Returns the licence record. Raises Refused.
    Shared by prepare() and vera.fulldoc_db (R3: the DB path re-applies the same filters, never ingest-time judgement)."""
    check_declared_kind(meta, cfg, "conversion record")
    check_content(text, meta, cfg)
    return check_licence(item, meta, cfg, arxiv_evidence=arxiv_evidence)


def prepare(item: dict, base_dir: Path, cfg: dict, now: str | None = None) -> dict:
    """All filters; returns the record to store. Raises Refused. domain_hint is never copied."""
    check_item_declared(item, cfg)
    text, meta = load_text(item, base_dir, cfg)
    return prepare_from_parts(item, text, meta, cfg, now, _checked=True)


def prepare_from_parts(item: dict, text: str, meta: dict, cfg: dict, now: str | None = None, _checked: bool = False,
                       *, arxiv_evidence: dict | None = None) -> dict:
    """prepare() without file I/O: the same filters and the same record from an in-memory item, text and conversion record."""
    if not _checked:
        check_item_declared(item, cfg)
    lic = check_text_record(item, text, meta, cfg, arxiv_evidence=arxiv_evidence)
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    src = {"doi": item.get("doi"), "url": item["landing_url"]}
    cu = canonical_url(src)
    provenance = {
        "url": item["landing_url"], "fulltext_url": item["fulltext_url"],
        "doi": str(item["doi"]).strip().lower() if item.get("doi") is not None else None,  # PLAN-NONDOI-1022
        "title": item["title"], "authors": item["authors"], "published": str(item["published_year"]),
        "source_type": cfg["source_type"], "licence": lic["licence"], "licence_decision": lic["licence_decision"],
        "attribution": item.get("attribution"), "stored_text_kind": cfg["stored_text_kind"],
        "fulltext_format": item["fulltext_format"], "converter": meta.get("converter"),
        "raw_sha256": meta.get("raw_sha256"), "fetched_at": item["fetched_at"],
        "verifier_status": item["verifier_status"], "read_status": item["read_status"],
        "admitted_by": cfg["admitted_by"], "retrieved_at": now, "document_id": item["document_id"],
        "sections": meta.get("sections") or [], "citations": meta.get("citations") or [],
        "references": meta.get("references") or []}
    return {"document_id": item["document_id"], "canonical_url": cu, "url": item["landing_url"],
            "title": item["title"], "text": text, "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "provenance": provenance, "published": str(item["published_year"]),
            "rationale": f"{cfg['candidate']['rationale_prefix']} Licence: {lic['licence_rationale']}"}


def licence_observations_for(rec: dict, cfg: dict, run_id, candidate_id=None, now=None) -> list[dict]:
    """The /ask path's licence observation rows (vera.corpus_admission.licence_observation_rows: one metadata row,
    one content row; same table, columns and semantics). Full text => the CONTENT scope governs. spdx is the declared
    licence id verbatim; detected_via names where it came from with the verbatim licence URL(s) (the manifest's and,
    when the JATS carries one, the document's). The table has no statement column, so the verbatim JATS licence
    statement stays in sources.provenance.licence.embedded.statement. native_decision is NULL (Gate A not applied).
    candidate_id is set (the /ask path leaves it NULL because it writes before the id is known)."""
    p = rec["provenance"]
    lic = p["licence"]
    content = lic.get("content") or {}
    via = f"{content.get('source')}: {content.get('url')}"
    emb = lic.get("embedded")
    if emb:
        via += f"; document: {emb.get('url') or emb.get('id')}"
    cand = {"url": p["url"], "doi": p["doi"], "licence_decision": p["licence_decision"],
            "stored_text_kind": cfg["stored_text_kind"], "gate_a_decision": None,
            "licence": {"metadata": {}, "content": {"id": content.get("id"), "source": via}}}
    rows = licence_observation_rows(cand, run_id, now=now)
    for r in rows:
        r["candidate_id"] = None if candidate_id is None else str(candidate_id)
    return rows


def doc_for_dar(rec: dict, source_id, canonical_source_id, cfg: dict) -> dict:
    """The DarCorpus ingest doc. Built only from stored fields; domain_hint never appears."""
    p = rec["provenance"]
    return {"source_key": rec["canonical_url"], "url": p["url"], "doi": p["doi"], "title": p["title"],
            "published": p["published"], "source_type": p["source_type"], "retrieved_at": p["retrieved_at"],
            "text": rec["text"], "stored_text_kind": cfg["stored_text_kind"], "admission_passed": True,
            "db_source_id": source_id, "db_canonical_source_id": canonical_source_id}


def ingest_routing_config(path: Path = ROUTING_CONFIG_PATH, cfg: dict | None = None) -> dict:
    """A deep copy of the shipped routing config with enabled=true for this ingest only (file never written)."""
    from vera import domain_registry
    shipped = domain_registry.load_config(path)
    if not shipped:
        raise RuntimeError(f"routing config unreadable at {path}")
    out = copy.deepcopy(shipped)
    if (cfg or {}).get("dar", {}).get("enable_for_ingest") is True:
        out["enabled"] = True
    return out


# --------------------------------------------------------------------------------------------- DB (injected engines)
def _sql(s: str) -> str:
    return s.replace("{S}", SCHEMA)


class RoLookup:
    """SELECT only, on the read-only account."""

    def __init__(self, engine_ro):
        self.engine = engine_ro

    def rows(self, sql: str, params: dict | None = None) -> list:
        from sqlalchemy import text
        with self.engine.connect() as c:
            return list(c.execute(text(_sql(sql)), params or {}).fetchall())

    def tables_present(self, names) -> list[str]:
        """Names from `names` that are NOT visible in the schema."""
        missing = []
        for n in names:
            r = self.rows("SELECT to_regclass(:t)", {"t": f"{SCHEMA}.{n}"})
            if not r or r[0][0] is None:
                missing.append(n)
        return missing

    def canonical(self, url: str):
        r = self.rows("SELECT canonical_source_id FROM {S}.canonical_sources WHERE canonical_url = :u", {"u": url})
        return r[0][0] if r else None

    def max_version(self, csid: int) -> int:
        r = self.rows("SELECT coalesce(max(version), 0) FROM {S}.sources WHERE canonical_source_id = :c", {"c": csid})
        return int(r[0][0]) if r else 0

    def same_hash(self, csid: int, h: str):
        r = self.rows("SELECT source_id, provenance->>'stored_text_kind' FROM {S}.sources "
                      "WHERE canonical_source_id = :c AND content_hash = :h", {"c": csid, "h": h})
        return (r[0][0], r[0][1]) if r else None

    def hash_any_canonical(self, h: str):
        """(source_id, canonical_source_id, stored_text_kind) of the oldest source with this content hash under ANY
        canonical URL, or None. Uses the existing index sources_content_hash_idx (db/migrations/002, no schema change)."""
        r = self.rows("SELECT source_id, canonical_source_id, provenance->>'stored_text_kind' FROM {S}.sources "
                      "WHERE content_hash = :h ORDER BY source_id LIMIT 1", {"h": h})
        return (r[0][0], r[0][1], r[0][2]) if r else None

    def span_counts(self, source_ids) -> dict:
        ids = [int(x) for x in source_ids]
        if not ids:
            return {}
        r = self.rows("SELECT source_id, count(*) FROM {S}.evidence_spans WHERE source_id = ANY(:ids) "
                      "GROUP BY source_id", {"ids": ids})
        return {int(a): int(b) for a, b in r}

    def spans(self, source_ids) -> list:
        ids = [int(x) for x in source_ids]
        if not ids:
            return []
        return self.rows("SELECT span_id, source_id, start_index, end_index FROM {S}.evidence_spans "
                         "WHERE source_id = ANY(:ids)", {"ids": ids})


class FulltextStore(WoCorpusStore):
    """WoCorpusStore pattern (INSERT ... RETURNING, write-only account) for full-text sources."""

    def __init__(self, engine_wo, question_text: str, lookup: RoLookup, cfg: dict, run_id: int | None = None):
        super().__init__(engine_wo, question_text)
        self.lookup, self.cfg = lookup, cfg
        if run_id is not None:
            self._run_id = int(run_id)

    def run_exists(self, run_id: int) -> bool:
        with self.engine.connect() as c:  # SELECT (run_id) is granted to the write-only role
            return self._one(c, "SELECT run_id FROM {S}.runs WHERE run_id = :r", {"r": int(run_id)}) is not None

    def missing_insert_privileges(self, tables) -> list[str]:
        """Tables the write-only role cannot INSERT into (checked for the current role before any write)."""
        out = []
        with self.engine.connect() as c:
            for t in tables:
                q = {"t": f"{SCHEMA}.{t}"}
                if self._one(c, "SELECT to_regclass(:t)", q) is None:  # absent table (e.g. migration 006 not applied)
                    out.append(t)
                elif self._one(c, "SELECT has_table_privilege(:t, 'INSERT')", q) is not True:
                    out.append(t)
        return out

    def admit_fulltext(self, rec: dict, in_tx=None) -> tuple[str, dict]:
        """("admitted", ids) or ("known", ids). Raises IngestError; the transaction is all-or-nothing.
        in_tx(conn, ids), when given, runs INSIDE the same transaction after the sources row (the DAR persist step):
        if it raises, the run (on --new-run), candidate, canonical, source, spans and domain rows all roll back.
        Its return value is put in ids["in_tx"]."""
        csid = self.lookup.canonical(rec["canonical_url"])
        version = 1
        if csid is not None:
            hit = self.lookup.same_hash(csid, rec["content_hash"])
            if hit:
                if hit[1] == self.cfg["stored_text_kind"]:
                    return "known", {"source_id": hit[0], "canonical_source_id": csid}
                raise IngestError(f"{rec['document_id']}: same content already stored as kind {hit[1]!r}")
            version = self.lookup.max_version(csid) + 1
        # D-057: durable identity is the content, not the URL. The same full text under a different canonical URL
        # (e.g. DOI missing, landing-page fallback) is "known": no run/candidate/canonical/source write, no alias row
        # (no schema change); the decision is returned for the run outcome. A new version of the SAME canonical URL
        # has a new hash, so it never matches here and re-versioning is unchanged.
        other = self.lookup.hash_any_canonical(rec["content_hash"])
        if other is not None:
            if other[2] != self.cfg["stored_text_kind"]:
                raise IngestError(f"{rec['document_id']}: same content already stored as kind {other[2]!r}")
            return "known", {"source_id": other[0], "canonical_source_id": other[1],
                             "decision": "known_same_content_other_canonical_url",
                             "duplicate_of_canonical_source_id": other[1]}
        rid = self._run_id
        cc = self.cfg["candidate"]
        with self.engine.begin() as c:  # one transaction: rolled back by any exception below
            if rid is None:  # --new-run: the question/run rows commit together with the first document, or not at all
                qid = self._one(c, "INSERT INTO {S}.questions (research_question) VALUES (:q) RETURNING question_id",
                                {"q": self.question_text})
                rid = self._one(c, "INSERT INTO {S}.runs (question_id, question) VALUES (:q, :t) RETURNING run_id",
                                {"q": qid, "t": self.question_text})
            cid = self._insert_new(c, "INSERT INTO {S}.candidates (run_id, source_url, title, snippet, gate_a_decision, "
                                      "gate_a_rationale, fetch_status) VALUES (:r,:u,:t,NULL,:d,:ra,:fs) "
                                      "RETURNING candidate_id",
                                   {"r": rid, "u": rec["url"], "t": rec["title"], "d": cc["gate_a_decision"],
                                    "ra": rec["rationale"], "fs": cc["fetch_status"]})
            if cid is None:
                raise IngestError(f"{rec['document_id']}: a candidate for this URL already exists in run {rid} "
                                  "but no matching full_text source does: use a new run")
            if csid is None:
                csid = self._insert_new(c, "INSERT INTO {S}.canonical_sources (canonical_url) VALUES (:u) "
                                           "RETURNING canonical_source_id", {"u": rec["canonical_url"]})
                if csid is None:
                    raise IngestError(f"{rec['document_id']}: canonical source appeared concurrently; rerun")
            sid = self._one(c, "INSERT INTO {S}.sources (canonical_source_id, candidate_id, version, content, "
                               "content_hash, provenance) VALUES (:cs,:cid,:v,:t,:h,CAST(:p AS jsonb)) "
                               "RETURNING source_id",
                            {"cs": csid, "cid": cid, "v": version, "t": rec["text"], "h": rec["content_hash"],
                             "p": json.dumps(rec["provenance"])})
            if sid is None:
                raise IngestError(f"{rec['document_id']}: sources INSERT returned no id")
            ids = {"source_id": sid, "canonical_source_id": csid, "candidate_id": cid, "version": version}
            if self.cfg.get("licence_observations_enabled") is True:  # item #97 parity, same transaction
                from sqlalchemy import text as _t
                rows = licence_observations_for(rec, self.cfg, rid, cid)
                for r in rows:
                    c.execute(_t(LICENCE_OBS_INSERT.replace("{S}", SCHEMA)), r)
                ids["licence_observations"] = len(rows)
            if in_tx is not None:
                ids["in_tx"] = in_tx(c, dict(ids))
        self._run_id = rid
        return "admitted", ids


def check_persist(res: dict, label: str) -> dict:
    bad = {k: res.get(k, 0) for k in ("unresolved", "promoted_skipped") if res.get(k, 0)}
    if res.get("status") != "persisted" or bad:
        raise IngestError(f"{label}: DAR persist incomplete: status={res.get('status')} reason={res.get('reason')} {bad}")
    return res


def prime_from_store(dar, lookup: RoLookup, reader, non_doc_kinds) -> tuple[dict, list[dict]]:
    """Restore the stored corpus into `dar` and seed one persist ctx. Returns (ctx, pending) where pending are stored
    documents with no evidence spans (to be re-ingested and persisted)."""
    snap = reader.load_snapshot(non_doc_kinds)
    docs = snap.get("documents") or []
    counts = lookup.span_counts([d["db_source_id"] for d in docs])
    pending = [d for d in docs if counts.get(int(d["db_source_id"]), 0) == 0]
    pend_keys = {d["source_key"] for d in pending}
    dar.restore({**snap, "documents": [d for d in docs if d["source_key"] not in pend_keys]})
    ctx = {"canonical": {d["source_key"]: d["db_canonical_source_id"] for d in docs}, "domains": {}, "spans": {}}
    for dom in dar.registry.list_domains():
        dbid = reader.domain_id_by_norm(dom.name_norm)
        if dbid is not None:
            ctx["domains"][dom.domain_id] = dbid
    key_of = {int(d["db_source_id"]): d["source_key"] for d in docs if d["source_key"] not in pend_keys}
    for span_id, source_id, s, e in lookup.spans(key_of):
        ctx["spans"][f"{key_of[int(source_id)]}:{s}-{e}"] = span_id
    return ctx, pending


class ConnWriter(DarWriter):
    """dar_store.DarWriter bound to ONE open transaction: each INSERT in a savepoint on that connection, so the DAR
    rows commit or roll back together with the document's sources row (DarWriter itself opens one transaction per
    INSERT). Duplicate detection is unchanged (IntegrityError in the savepoint -> None, then the RO reader)."""

    def __init__(self, conn, reader=None):
        super().__init__(None, reader=reader)
        self.conn = conn

    def _insert_new(self, sql: str, params: dict, returning: bool = True):
        from sqlalchemy.exc import IntegrityError
        sp = self.conn.begin_nested()
        try:
            res = self._exec(self.conn, sql, params)
            value = res.first()[0] if returning else True
            sp.commit()
            return value
        except IntegrityError:
            sp.rollback()
            return None


def _ctx_copy(ctx: dict) -> dict:
    return {k: dict(v) for k, v in ctx.items()}


def dar_step(dar, reader, ctx: dict, persist_fn, doc_builder, label: str):
    """Callback for FulltextStore.admit_fulltext(in_tx=...): DAR ingest + persist on the document's transaction.
    Persists into a COPY of ctx; the caller merges it only after the commit. (The in-memory DarCorpus already holds
    the document if the transaction then fails; the run stops at the first failure, so it is never reused.)"""
    def step(conn, ids):
        plan = dar.ingest_document(doc_builder(ids))
        if plan.get("status") != "indexed":
            raise IngestError(f"{label}: DAR ingest status {plan.get('status')} ({plan.get('reason')})")
        local = _ctx_copy(ctx)
        res = check_persist(persist_fn(plan, ConnWriter(conn, reader), ctx=local), label)
        return {"plan": plan, "persist": res, "ctx": local}
    return step


def resume_pending(dar, engine_wo, reader, pending: list[dict], ctx: dict, persist_fn) -> list[dict]:
    """Stored full-text documents without spans: ingest + persist, one transaction per document."""
    out = []
    for d in pending:
        d = dict(d, admission_passed=True)
        plan = dar.ingest_document(d)
        if plan.get("status") != "indexed":
            raise IngestError(f"resume {d['source_key']}: DAR ingest status {plan.get('status')} ({plan.get('reason')})")
        local = _ctx_copy(ctx)
        with engine_wo.begin() as conn:
            res = check_persist(persist_fn(plan, ConnWriter(conn, reader), ctx=local), f"resume {d['source_key']}")
        ctx.update(local)
        out.append(res)
    return out


def dar_summary(plan: dict) -> dict:
    return {"status": plan.get("status"), "chunks": len(plan.get("chunks") or []),
            "domains": len(plan.get("domains") or []), "doc_domains": len(plan.get("corpus_assoc") or []),
            "new_domains": sum(1 for d in plan.get("domains") or [] if d.get("created"))}
