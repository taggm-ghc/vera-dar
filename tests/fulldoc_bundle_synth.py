"""SYNTHETIC bundle writer for the D-055 full-document tests. Same layout as the real DAR-9 bundle:
manifest-filled.https.json, texts/<id>.txt, texts/<id>.meta.json, texts.SHA256SUMS (sha256sum format).
All content is invented (see tests/fulltext_synth.py). Writes only under the directory it is given."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from tests import fulltext_synth
from vera import fulltext_loader as fl
from vera import jats_text

MANIFEST_NAME = "manifest-filled.https.json"
SUMS_NAME = "texts.SHA256SUMS"


def sec(title: str, paras=(), label: str = "", children: str = "") -> str:
    """JATS <sec> XML for body_sections (C1 tests). `children` is nested <sec> XML placed after the paragraphs."""
    lab = f"<label>{label}</label>" if label else ""
    ps = "".join(f"<p>{p}</p>" for p in paras)
    return f"<sec>{lab}<title>{title}</title>{ps}{children}</sec>"


def write_bundle(root: Path, docs=fulltext_synth.DEFAULT_DOCS, items_over: dict | None = None,
                 tamper: dict | None = None, jats_over: dict | None = None,
                 body_sections: dict | None = None, text_kind: dict | str | None = None) -> Path:
    """items_over: {doc_id: manifest-item overrides}; jats_over: {doc_id: kwargs for fulltext_synth.jats};
    body_sections: {doc_id: JATS <sec> XML (see sec()) appended inside <body> after the standard sections};
    text_kind: None (default, unchanged output) | str (all docs) | {doc_id: str}: declares a `text_kind` label in the
    manifest item and the conversion record (e.g. "abstract"); it changes no text content (pair with jats_over
    {with_body: False} for abstract-only content);
    tamper: {doc_id: bytes|str appended to the .txt AFTER the sums were computed} (sum mismatch)."""
    root = Path(root)
    ing = fl.load_config()
    (root / "texts").mkdir(parents=True, exist_ok=True)
    items = []
    for i, (did, topic, doi) in enumerate(docs):
        kw = dict((jats_over or {}).get(did) or {})
        kw.setdefault("seed", i + 1)
        raw = fulltext_synth.jats(did, topic, doi, **kw)
        extra = (body_sections or {}).get(did)
        if extra:
            assert raw.count(b"</body>") == 1, "body_sections needs a document with a body"
            raw = raw.replace(b"</body>", extra.encode("utf-8") + b"</body>")
        conv = jats_text.convert(raw, "jats-xml", ing["converter"])
        item = fulltext_synth.item(did, doi, **((items_over or {}).get(did) or {}))
        out = root / f"texts/{did}.txt"  # file naming is fixed by document_id (items_over may point text_file elsewhere)
        data = conv["text"].encode("utf-8")
        out.write_bytes(data)
        meta = {k: v for k, v in conv.items() if k != "text"}
        meta.update({"document_id": did, "raw_sha256": hashlib.sha256(raw).hexdigest(),
                     "text_sha256": hashlib.sha256(data).hexdigest(), "raw_file": f"{did}.xml"})
        out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        tk = text_kind.get(did) if isinstance(text_kind, dict) else text_kind
        if tk:
            item["text_kind"] = tk
            meta["text_kind"] = tk
            out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        items.append(item)
    (root / MANIFEST_NAME).write_text(json.dumps({"_doc": "SYNTHETIC bundle manifest", "schema_version": 1,
                                                  "sources": items}, indent=1), encoding="utf-8")
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
             for p in sorted((root / "texts").iterdir()) if p.is_file()]
    (root / SUMS_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    for did, extra in (tamper or {}).items():
        with open(root / f"texts/{did}.txt", "ab") as f:
            f.write(extra if isinstance(extra, bytes) else str(extra).encode("utf-8"))
    return root


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def synth_cfg(pins: dict | None = None) -> dict:
    """The shipped fulldoc config with pins nulled (or set to `pins`)."""
    from vera import fulldoc_corpus
    cfg = copy.deepcopy(fulldoc_corpus.load_config())
    cfg["pins"] = {"manifest_sha256": None, "sums_sha256": None, **(pins or {})}
    cfg.setdefault("fulldoc", {})["enabled"] = True  # the shipped flag may be false; tests opt out explicitly
    return cfg
