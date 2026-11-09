"""H1B-PROMOTE-1107: repository-pinned manifest for stored ACQUIRED documents (owner H1(b), 2026-11-07).

Trust root: a reviewed repository commit of config/acquired_pins.json, never a DB row. The rw app account can write
every stored row and every hash recorded in it, so a hash the loader recomputes is only self-consistent; it proves
nothing unless it is compared with a value the rw path cannot write. This file is that value.

Modes (config `mode`, read only from the repository file; no env override):
  off     today's behaviour exactly: the manifest is not consulted (accepted risk through 2026-11-21, owner H1).
  shadow  served as today, except a document whose pin exists and differs is refused (tamper evidence); the counts of
          pinned / unpinned / mismatched stored documents are reported (FullDocCorpus.pin_report, one log line).
  on      a stored acquired document is served only when its text sha256, provenance sha256, canonical-URL sha256,
          licence id and licence decision equal its pin. Unpinned stored documents are QUARANTINED (not served).
Inline documents fetched within a request (rows built in memory from the fetch, tagged EPHEMERAL_KEY by the two
construction sites) are verified in memory and never read back from the DB; they are not subject to the manifest.

Residual risk (stated, not closed): promotion pins what is stored at export time. A row altered or forged BEFORE an
owner export passes the reader and would be pinned; the reviewer sees only hashes. Batches are frequent to keep that
window short; an independent re-fetch of the source to recompute text_sha256 is a deferred, separately authorised check.

A malformed manifest file fails CLOSED: it is read as mode "on" with no documents. A missing file is mode "off" (the
file is repository-controlled: removing it is a reviewed commit, not an rw-path action).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "acquired_pins.json"
MODES = ("off", "shadow", "on")
EPHEMERAL_KEY = "_inline_ephemeral"   # set ONLY on rows built in memory from a fetch (tests scan for other uses)
R_QUARANTINE = "R-ACQ-QUARANTINE"     # stored acquired document has no pin (mode on)
R_PINMISMATCH = "R-ACQ-PIN-MISMATCH"  # stored acquired document differs from its pin (shadow and on)
R_PINCFG = "R-ACQ-PIN-CONFIG"         # manifest unreadable or invalid (mode on: nothing stored is served)
PIN_FIELDS = ("text_sha256", "provenance_sha256", "canonical_url_sha256", "licence_id", "licence_decision")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_DOC_ID = re.compile(r"acq-[0-9a-f]{16}")
_BATCH_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_CACHE: dict = {}


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()


def provenance_sha256(prov: dict) -> str:
    """sha256 of the whole stored provenance object, canonical JSON. Stored provenance is never updated after write."""
    return sha256_text(canonical_json(prov))


def pin_entry(text: str, prov: dict, canonical_url: str, licence_id, licence_decision) -> dict:
    return {"text_sha256": sha256_text(text), "provenance_sha256": provenance_sha256(prov),
            "canonical_url_sha256": sha256_text(canonical_url), "licence_id": licence_id,
            "licence_decision": licence_decision}


def validate(m) -> str | None:
    """None when the manifest is valid, else a short reason (no values)."""
    if not isinstance(m, dict) or m.get("schema_version") != 1:
        return "not a schema_version 1 object"
    if m.get("mode") not in MODES:
        return "mode is not off, shadow or on"
    batches, docs = m.get("batches"), m.get("documents")
    if not isinstance(batches, list) or not isinstance(docs, dict):
        return "batches must be a list and documents an object"
    ids = set()
    for b in batches:
        if not isinstance(b, dict) or not isinstance(b.get("id"), str) or not _BATCH_ID.fullmatch(b["id"]) or b["id"] in ids:
            return "a batch has a missing, malformed or duplicate id"
        if not (isinstance(b.get("export_sha256"), str) and _HEX64.fullmatch(b["export_sha256"])):
            return f"batch {b['id']} has no export_sha256"
        ids.add(b["id"])
    seen_text = set()
    for did, e in docs.items():
        if not _DOC_ID.fullmatch(str(did)) or not isinstance(e, dict):
            return "a document id is malformed"
        for k in ("text_sha256", "provenance_sha256", "canonical_url_sha256"):
            if not (isinstance(e.get(k), str) and _HEX64.fullmatch(e[k])):
                return f"{did}: {k} is not a sha256"
        if not (isinstance(e.get("licence_id"), str) and e["licence_id"].strip()):
            return f"{did}: licence_id missing"
        if e.get("licence_decision") != "allow":
            return f"{did}: licence_decision is not allow"
        if e.get("batch") not in ids:
            return f"{did}: batch is not a listed batch"
        if e["text_sha256"] in seen_text:
            return f"{did}: text_sha256 pinned twice"
        seen_text.add(e["text_sha256"])
    return None


def _closed(reason: str) -> dict:
    return {"mode": "on", "documents": {}, "error": reason}


def load(path: Path | None = None) -> dict:
    """{"mode", "documents", "error"}. Cached on (mtime_ns, size). Never raises."""
    p = Path(path or CONFIG_PATH)
    try:
        st = p.stat()
    except FileNotFoundError:
        return {"mode": "off", "documents": {}, "error": None}
    except Exception as e:  # noqa: BLE001
        return _closed(f"manifest unreadable ({type(e).__name__})")
    key = (str(p), st.st_mtime_ns, st.st_size)
    hit = _CACHE.get("k") == key and _CACHE.get("v")
    if hit:
        return hit
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
        err = validate(m)
        v = _closed(f"manifest invalid: {err}") if err else {"mode": m["mode"], "documents": m["documents"], "error": None}
    except Exception as e:  # noqa: BLE001
        v = _closed(f"manifest unreadable ({type(e).__name__})")
    _CACHE.clear()
    _CACHE.update(k=key, v=v)
    return v


def status(manifest: dict, did, got: dict) -> str:
    """'pinned' | 'unpinned' | 'mismatch' for a stored document whose computed pin is `got`."""
    want = (manifest.get("documents") or {}).get(did)
    if want is None:
        return "unpinned"
    return "pinned" if all(got.get(k) is not None and got.get(k) == want.get(k) for k in PIN_FIELDS) else "mismatch"


def pinned_ids(manifest: dict) -> frozenset:
    return frozenset((manifest.get("documents") or {}).keys())


def row_document_id(row: dict):
    """document_id from a row's provenance (dict or JSON text), or None. Used only to PRE-SELECT rows for the serve
    caps in mode on; every selected row is still fully verified and compared with its pin."""
    p = row.get("provenance")
    try:
        p = json.loads(p) if isinstance(p, (str, bytes)) else p
    except Exception:  # noqa: BLE001
        return None
    return p.get("document_id") if isinstance(p, dict) else None


def export_records(sources: list, spans: list, fdcfg: dict, acq_cfg: dict) -> dict:
    """Hash-only promotion records for every stored acquired row given, verified by the acquired-path reader (no serve
    cap since OD-1109-NOCAP). {"documents": [{document_id, verdict, <PIN_FIELDS>, acquired_on}], "unusable": {code: n}}.
    verdict "verified" = passed every reader check except the pin comparison. No text, titles or URLs are returned."""
    import copy

    from vera import fulldoc_db
    a = copy.deepcopy(acq_cfg)
    b = a.setdefault("bounds", {})
    b.update(served_per_topic_max=None, served_topic_balanced=False)
    rows = [{k: v for k, v in r.items() if k != EPHEMERAL_KEY} for r in sources]  # stored rows only
    c = fulldoc_db.load_acquired_rows(rows, spans, fdcfg, a)  # OD-1109-NOCAP: the reader has no serve cap
    cands = getattr(c, "pin_candidates", None) or {}
    docs = []
    for did in sorted(cands):
        e = cands[did]
        docs.append({"document_id": did, "verdict": "verified", **{k: e[k] for k in PIN_FIELDS},
                     "acquired_on": e.get("acquired_on")})
    codes: dict = {}
    for u in c.unusable or ():
        code = u.get("code") if isinstance(u, dict) else None
        if code in (R_QUARANTINE, R_PINMISMATCH) and (u.get("document_id") in cands):
            continue  # verified apart from the pin: listed above
        codes[str(code)] = codes.get(str(code), 0) + 1
    return {"documents": docs, "unusable": dict(sorted(codes.items()))}
