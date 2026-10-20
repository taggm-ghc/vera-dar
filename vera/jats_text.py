"""DARINT G5: JATS XML / HTML article -> plain text with section structure and citation provenance (stdlib + defusedxml).

convert(raw, fmt, ccfg) returns
    {"text", "sections": [{"kind", "title", "level", "start", "end"}], "citations": [{"rids", "label", "start", "end"}],
     "references": [{"id", "label", "doi", "text"}], "meta": {"title", "authors", "doi", "journal", "year",
     "licence": {"id", "url"}}, "has_body", "abstract_chars", "body_chars", "converter"}
Offsets index into "text" (text[start:end]). Output is deterministic for the same input and config.

What goes into the text: the article title, every abstract, the body, and the back-matter elements named in config
(`back_keep`); the reference list is never part of the text (it is returned as `references`, the citation provenance).
Inline citations (JATS xref ref-type="bibr"; HTML links to a reference anchor) stay in the text as their visible label
and are recorded with offsets and target ids. Figures and tables keep their label and caption; table cells are kept as
rows; formulas become the configured placeholder; graphics and media are dropped.

XML safety: parsing uses defusedxml.ElementTree (expat underneath; it never resolves an external DTD or entity and
forbids entity declarations); the textual guards still run first: a document whose
text declares an entity (<!ENTITY) or a DOCTYPE internal subset is refused before parsing (entity-expansion guard);
input must be UTF-8 (checked on decoded text); expat must meet ccfg["min_expat_version"]; inputs larger than
ccfg["max_raw_bytes"] are refused; an undefined named entity is a parse error and is refused. A defusedxml refusal (DefusedXmlException) also maps to ConversionError. Every
refusal raises ConversionError with a reason; nothing is guessed.
All element lists, placeholders and limits come from the `converter` section of config/fulltext_ingest.json.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import defusedxml
import defusedxml.ElementTree as DET
from html.parser import HTMLParser

CONVERTER_ID = "vera.jats_text/1"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
_WS = re.compile(r"\s+")
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)


class ConversionError(ValueError):
    """The input cannot be converted safely or faithfully; the message is the reason."""


def _ln(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _flat(el) -> str:
    return _WS.sub(" ", "".join(el.itertext())).strip() if el is not None else ""


class _Builder:
    """Accumulates normalised text; whitespace is collapsed lazily so recorded offsets never move."""

    def __init__(self):
        self.parts: list[str] = []
        self.n = 0
        self.pending_space = False
        self.last = ""  # the last two characters written

    def _emit(self, s: str):
        self.parts.append(s)
        self.n += len(s)
        self.last = (self.last + s)[-2:]

    def text(self, s: str | None):
        if not s:
            return
        if s[0].isspace() and self.n and not self._at_break():
            self.pending_space = True
        words = s.split()
        for i, w in enumerate(words):
            if (i > 0 or self.pending_space) and self.n and not self._at_break():
                self._emit(" ")
            self.pending_space = False
            self._emit(w)
        if words and s[-1].isspace():
            self.pending_space = True

    def mark(self) -> int:
        """Offset where the next visible character will be written."""
        if self.pending_space and self.n and not self._at_break():
            self._emit(" ")
            self.pending_space = False
        return self.n

    def _at_break(self) -> bool:
        return self.last.endswith("\n")

    def line(self):
        if self.n and not self._at_break():
            self._emit("\n")
        self.pending_space = False

    def block(self):
        if self.n and self.last != "\n\n":
            self._emit("\n" if self._at_break() else "\n\n")
        self.pending_space = False

    def value(self) -> str:
        return "".join(self.parts).rstrip()


def _expat_ok(ccfg: dict):
    import pyexpat
    have = tuple(int(x) for x in pyexpat.version_info)
    need = tuple(int(x) for x in ccfg["min_expat_version"])
    if have < need:
        raise ConversionError(f"expat {have} is older than the configured minimum {need} (no amplification limits)")


def _decode_guard(raw: bytes, ccfg: dict, xml: bool) -> str:
    """Size cap, strict UTF-8 (a UTF-8 BOM is allowed), and for XML: no declared non-UTF-8 encoding, no DOCTYPE
    internal subset, no entity declaration. Checks run on DECODED text, so a UTF-16 or BOM trick cannot hide them."""
    if len(raw) > int(ccfg["max_raw_bytes"]):
        raise ConversionError(f"input is {len(raw)} bytes, above max_raw_bytes {ccfg['max_raw_bytes']}")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or raw[:2].count(b"\x00"):
        raise ConversionError("input is UTF-16/32 encoded: only UTF-8 is accepted")
    try:
        s = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ConversionError(f"input is not valid UTF-8: {e}") from e
    if not xml:
        return s
    head = s[: int(ccfg["prolog_scan_chars"])]
    decl = re.match(r"\s*<\?xml[^>]*encoding\s*=\s*[\"']([^\"']+)[\"']", head)
    if decl and decl.group(1).strip().lower().replace("_", "-") not in ("utf-8", "utf8"):
        raise ConversionError(f"declared encoding {decl.group(1)!r} is not UTF-8")
    if re.search(r"<!ENTITY", s, re.I):
        raise ConversionError("document declares an entity (<!ENTITY): refused (entity-expansion guard)")
    dt = re.search(r"<!DOCTYPE[^>\[]*\[", head, re.I)
    if dt:
        raise ConversionError("DOCTYPE has an internal subset: refused")
    _expat_ok(ccfg)
    return s


# ------------------------------------------------------------------------------------------------- JATS
class _Jats:
    def __init__(self, root, ccfg: dict):
        self.root, self.c = root, ccfg
        self.b = _Builder()
        self.sections: list[dict] = []
        self.citations: list[dict] = []
        self.drop = set(ccfg["jats_drop"])
        self.blocks = set(ccfg["jats_block"])
        self.cite_types = set(ccfg["jats_citation_ref_types"])
        el = ccfg["jats_elements"]
        self.formula, self.display_formula = set(el["formula"]), set(el["display_formula"])
        self.cite_el, self.sec_el, self.row_el = el["citation"], el["section"], set(el["table_row"])

    def _find(self, el, *names):
        for x in el.iter():
            if _ln(x.tag) in names:
                return x
        return None

    def _children(self, el, name):
        return [x for x in el if _ln(x.tag) == name]

    # inline/mixed content walker
    def walk(self, el):
        name = _ln(el.tag)
        if name in self.drop:
            self.b.text(el.tail)
            return
        if name in self.formula:
            if name in self.display_formula:
                self.b.block()
            self.b.text(" " + self.c["formula_placeholder"] + " ")
            if name in self.display_formula:
                self.b.block()
            self.b.text(el.tail)
            return
        if name == self.cite_el and el.get("ref-type") in self.cite_types:
            start = self.b.mark()
            self.b.text(el.text)
            for ch in el:
                self.walk(ch)
            end = self.b.n
            if end > start:
                self.citations.append({"rids": (el.get("rid") or "").split(), "label": "", "start": start,
                                       "end": end})  # label filled from the final text in _finish
            self.b.text(el.tail)
            return
        if name == self.sec_el:  # a section nested in a non-section container (e.g. boxed-text)
            self.section(el, "body", 2)
            self.b.text(el.tail)
            return
        if name in self.row_el:
            self.b.line()
            first = True
            for cell in el:
                if not first:
                    self.b.text(" " + self.c["table_cell_separator"] + " ")
                first = False
                self.walk_inline(cell)
            self.b.line()
            self.b.text(el.tail)
            return
        block = name in self.blocks
        if block:
            self.b.block()
        self.b.text(el.text)
        for ch in el:
            self.walk(ch)
        if block:
            self.b.block()
        self.b.text(el.tail)

    def walk_inline(self, el):
        self.b.text(el.text)
        for ch in el:
            self.walk(ch)

    def heading(self, el) -> str:
        label = self._children(el, "label")
        title = self._children(el, "title")
        parts = [_flat(label[0]) if label else "", _flat(title[0]) if title else ""]
        return " ".join(p for p in parts if p)

    def _has_content(self, el) -> bool:
        """True when the element holds visible text outside its own label/title (and outside dropped elements)."""
        if (el.text or "").strip():
            return True
        for ch in el:
            nm = _ln(ch.tag)
            if (ch.tail or "").strip():
                return True
            if nm in ("label", "title") or nm in self.drop:
                continue
            if nm in self.formula or self._has_content(ch):
                return True
        return False

    def section(self, el, kind: str, level: int, default_title: str = ""):
        if not self._has_content(el):  # e.g. an empty <app-group>: no heading without content
            return
        title = self.heading(el) or default_title
        self.b.block()
        start = self.b.mark()
        rec = {"kind": kind, "title": title, "level": level, "start": start, "end": start}
        self.sections.append(rec)
        if title:
            self.b.text(title)
            self.b.block()
        for ch in el:
            if _ln(ch.tag) in ("label", "title"):
                continue
            if _ln(ch.tag) == self.sec_el:
                self.section(ch, kind, level + 1)
                continue
            self.walk(ch)
        self.b.block()
        rec["end"] = len(self.b.value())

    def meta(self) -> dict:
        am = self._find(self.root, "article-meta")
        jm = self._find(self.root, "journal-meta")
        out = {"title": None, "authors": [], "doi": None, "journal": None, "year": None,
               "licence": {"id": None, "url": None}}
        if am is None:
            return out
        t = self._find(am, "article-title")
        out["title"] = _flat(t) or None
        for contrib in am.iter():
            if _ln(contrib.tag) == "contrib" and contrib.get("contrib-type", "author") == "author":
                sn, gn = self._find(contrib, "surname"), self._find(contrib, "given-names")
                coll = self._find(contrib, "collab")
                nm = " ".join(x for x in (_flat(gn), _flat(sn)) if x) or _flat(coll)
                if nm:
                    out["authors"].append(nm)
        for aid in am.iter():
            if _ln(aid.tag) == "article-id" and aid.get("pub-id-type") == "doi":
                out["doi"] = _flat(aid).lower() or None
                break
        for pd in am.iter():
            if _ln(pd.tag) == "pub-date":
                y = self._find(pd, "year")
                if y is not None and _flat(y).isdigit():
                    out["year"] = int(_flat(y))
                    break
        if jm is not None:
            jt = self._find(jm, "journal-title")
            out["journal"] = _flat(jt) or None
        lic = self._find(am, "license")
        if lic is not None:
            url = lic.get(XLINK_HREF) or lic.get("href")
            ref = self._find(lic, "license_ref")
            url = url or (_flat(ref) if ref is not None else None)
            lp = self._find(lic, "license-p")
            out["licence"] = {"id": lic.get("license-type") or None, "url": url or None,
                              "statement": _flat(lp)[: int(self.c["licence_statement_chars"])] or None}
        return out

    def references(self) -> list[dict]:
        refs = []
        for rl in self.root.iter():
            if _ln(rl.tag) != "ref-list":
                continue
            for ref in rl.iter():
                if _ln(ref.tag) not in self.c["reference_elements"]:  # never <ref-html> (Copernicus duplicate)
                    continue
                lab = self._children(ref, "label")
                doi = None
                for pid in ref.iter():
                    if _ln(pid.tag) == "pub-id" and pid.get("pub-id-type") == "doi":
                        doi = _flat(pid).lower()
                        break
                body = " ".join(_flat(x) for x in ref if _ln(x.tag) != "label")
                if doi is None:
                    m = _DOI.search(body)
                    doi = m.group(1).rstrip(".,;").lower() if m else None
                sn = self._find(ref, "surname")
                first = _flat(sn) if sn is not None else (body.split(",", 1)[0].split() or [""])[-1]
                yel = self._find(ref, "year")
                ym = re.search(self.c["author_year_citations"]["reference_year_pattern"],
                               _flat(yel) if yel is not None else body)
                refs.append({"id": ref.get("id"), "label": _flat(lab[0]) if lab else None, "doi": doi,
                             "first_author": first or None, "year": ym.group(0) if ym else None,
                             "text": body[: int(self.c["reference_text_chars"])]})
        return refs

    def author_year(self, text: str, refs: list[dict]) -> list[dict]:
        """Fallback for JATS that marks no citations (e.g. Copernicus: plain-text "Hawkins and Sutton, 2009"): a
        citation is recorded only when the in-text (surname, year) matches exactly one reference's first author and
        year. Recorded with method "author-year" (inferred), never mixed with xref citations."""
        ay = self.c["author_year_citations"]
        if not ay["enabled"]:
            return []
        index: dict = {}
        for r in refs:
            if r.get("id") and r.get("first_author") and r.get("year"):
                index.setdefault((r["first_author"].lower(), r["year"].lower()), []).append(r["id"])
        out = []
        for m in re.finditer(ay["pattern"], text):
            hit = index.get((m.group("surname").lower(), m.group("year").lower()))
            if hit and len(hit) == 1:
                out.append({"rids": hit, "label": "", "start": m.start(), "end": m.end(), "method": "author-year"})
        return out

    def run(self) -> dict:
        meta = self.meta()
        am = self._find(self.root, "article-meta")
        if meta["title"]:
            start = self.b.mark()
            self.b.text(meta["title"])
            self.sections.append({"kind": "title", "title": meta["title"], "level": 0, "start": start,
                                  "end": self.b.n})
            self.b.block()
        if am is not None:
            for ab in [x for x in am if _ln(x.tag) == "abstract"]:
                kind = ab.get("abstract-type") or ""
                default = self.c["abstract_titles"].get(kind, self.c["abstract_titles"][""])
                self.section(ab, "abstract", 1, default)
        body = self._find(self.root, "body")
        if body is not None:
            for ch in body:
                if _ln(ch.tag) == self.sec_el:
                    self.section(ch, "body", 1)
                else:
                    pre_start = self.b.mark()
                    self.walk(ch)
                    if len(self.b.value()) > pre_start:
                        self.sections.append({"kind": "body", "title": "", "level": 1, "start": pre_start,
                                              "end": len(self.b.value())})
        back = self._find(self.root, "back")
        if back is not None:
            for ch in back:
                nm = _ln(ch.tag)
                if nm in self.c["back_keep"]:
                    self.section(ch, "back", 1, self.c["back_titles"].get(nm, ""))
        for c in self.citations:
            c["method"] = "xref"
        refs = self.references()
        out = _finish(self.b, self.sections, self.citations, refs, meta, "jats-xml")
        if not out["citations"]:
            out["citations"] = self.author_year(out["text"], refs)
            for c in out["citations"]:
                c["label"] = out["text"][c["start"]:c["end"]]
        return out


def _finish(b: _Builder, sections, citations, references, meta, fmt) -> dict:
    text = b.value()
    for s in sections:
        s["end"] = min(s["end"], len(text))
        while s["end"] > s["start"] and text[s["end"] - 1].isspace():
            s["end"] -= 1
    for c in citations:
        c["label"] = text[c["start"]:c["end"]]
    sections = [s for s in sections if s["end"] > s["start"]]

    def chars(kind):
        top = [s for s in sections if s["kind"] == kind and s["level"] == 1]
        return sum(s["end"] - s["start"] for s in top)
    body = chars("body")
    return {"text": text, "sections": sections, "citations": citations, "references": references, "meta": meta,
            "has_body": body > 0, "abstract_chars": chars("abstract"), "body_chars": body,
            "converter": CONVERTER_ID, "format": fmt}


def _depth_guard(root, limit: int) -> None:
    """Iterative depth check BEFORE the recursive walk, so a deeply nested file is refused, never a RecursionError."""
    stack = [(root, 1)]
    while stack:
        el, d = stack.pop()
        if d > limit:
            raise ConversionError(f"element nesting deeper than max_element_depth {limit}: refused")
        stack.extend((ch, d + 1) for ch in el)


def jats_to_text(raw: bytes, ccfg: dict) -> dict:
    s = _decode_guard(raw, ccfg, xml=True)
    s = re.sub(r"^\s*<\?xml[^>]*\?>", "", s, count=1)  # declaration already checked; parse the decoded text
    try:
        root = DET.fromstring(s)
    except ET.ParseError as e:
        raise ConversionError(f"XML parse error (undefined entities are refused, never guessed): {e}") from e
    except defusedxml.DefusedXmlException as e:  # fixed reason: the exception may carry untrusted DTD/system ids
        raise ConversionError(f"XML refused by defusedxml ({type(e).__name__})") from None
    _depth_guard(root, int(ccfg["max_element_depth"]))
    if _ln(root.tag) in ccfg["jats_wrapper_roots"]:  # e.g. NCBI efetch db=pmc: <pmc-articleset><article>...
        arts = [ch for ch in root if _ln(ch.tag) == "article"]
        if len(arts) != 1:
            raise ConversionError(f"<{_ln(root.tag)}> holds {len(arts)} <article> elements; exactly one is required")
        root = arts[0]
    if _ln(root.tag) != "article":
        raise ConversionError(f"root element is <{_ln(root.tag)}>, not a JATS <article>")
    return _Jats(root, ccfg).run()


# ------------------------------------------------------------------------------------------------- HTML
class _Html(HTMLParser):
    def __init__(self, ccfg: dict):
        super().__init__(convert_charrefs=True)
        self.c = ccfg
        self.b = _Builder()
        self.skip_tags = set(ccfg["html_skip"])
        self.void = set(ccfg["html_void"])
        self.containers = set(ccfg["html_body_containers"])
        self.meta_names = {k: [x.lower() for x in v] for k, v in ccfg["html_meta_names"].items()}
        self.max_depth = int(ccfg["max_element_depth"])
        self.implicit = {k: set(v) for k, v in ccfg["html_implicit_close"].items() if not k.startswith("_")}
        self.block_tags = set(ccfg["html_block"])
        self.ref_heads = [h.lower() for h in ccfg["references_headings"]]
        self.abs_heads = [h.lower() for h in ccfg["abstract_headings"]]
        self.skip_depth = 0
        self.stack: list[str] = []
        self.heading: list[str] | None = None
        self.heading_level = 0
        self.mode = "body"           # body | abstract | refs
        self.refs_level = 0
        self.sections: list[dict] = []
        self.refs: list[dict] = []
        self.cur_ref: dict | None = None
        self.anchor: dict | None = None
        self.anchors: list[dict] = []
        self.meta = {"title": None, "authors": [], "doi": None, "journal": None, "year": None,
                     "licence": {"id": None, "url": None}}
        self.in_title = False
        self.title_parts: list[str] = []

    def _close_section(self):
        if self.sections and self.sections[-1].get("_open"):
            s = self.sections[-1]
            s["end"] = len(self.b.value())
            del s["_open"]

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "meta":
            self._meta(a)
            return
        if tag == "link" and (a.get("rel") or "").lower() == self.c["html_licence_rel"]:
            self.meta["licence"]["url"] = a.get("href")
            return
        if tag in self.void:
            if tag in self.c["html_line_break"] and not self.skip_depth:
                self.b.line()
            return
        if not self.skip_depth:  # HTML implicit close (e.g. <p><p>, <li><li>): pop before counting depth
            while self.stack and tag in self.implicit.get(self.stack[-1], ()):
                self.stack.pop()
        self.stack.append(tag)
        if len(self.stack) > self.max_depth:
            raise ConversionError(f"element nesting deeper than max_element_depth {self.max_depth}: refused")
        if tag == "title" and not self.meta["title"]:
            self.in_title = True
        if self.skip_depth or tag in self.skip_tags:
            self.skip_depth += 1
            return
        if re.fullmatch(r"h[1-6]", tag):
            self.heading, self.heading_level = [], int(tag[1])
            return
        if self.mode == "refs":
            if tag == "li":
                self.cur_ref = {"id": a.get("id"), "label": None, "doi": None, "_t": []}
            return
        if tag == "a" and (a.get("href") or "").startswith("#"):
            self.anchor = {"target": a["href"][1:], "start": self.b.mark()}
        if tag in self.block_tags:
            self.b.block()

    def handle_endtag(self, tag):
        if tag in self.void:
            return
        if tag in self.stack:
            while self.stack and self.stack.pop() != tag:
                pass
        if tag == "title":
            self.in_title = False
        if self.skip_depth:
            self.skip_depth -= 1
            return
        if self.heading is not None and re.fullmatch(r"h[1-6]", tag):
            self._heading_done(_WS.sub(" ", "".join(self.heading)).strip())
            return
        if self.mode == "refs":
            if tag == "li" and self.cur_ref is not None:
                t = _WS.sub(" ", "".join(self.cur_ref.pop("_t"))).strip()
                m = _DOI.search(t)
                self.cur_ref["doi"] = m.group(1).rstrip(".,;").lower() if m else None
                self.cur_ref["text"] = t[: int(self.c["reference_text_chars"])]
                self.refs.append(self.cur_ref)
                self.cur_ref = None
            return
        if tag == "a" and self.anchor is not None:
            end = self.b.n
            if end > self.anchor["start"]:
                self.anchors.append({**self.anchor, "end": end})
            self.anchor = None
        if tag in self.block_tags:
            self.b.block()

    def _heading_done(self, title: str):
        level, self.heading = self.heading_level, None
        low = title.lower()
        if self.mode == "refs" and level > self.refs_level:
            return
        self._close_section()
        if any(low == h or low.startswith(h + " ") for h in self.ref_heads):
            self.mode, self.refs_level = "refs", level
            return
        self.mode = "abstract" if any(low == h for h in self.abs_heads) else "body"
        if level == 1 and not self.sections and (self.meta["title"] is None or low == (self.meta["title"] or "").lower()):
            self.b.block()
            start = self.b.mark()
            self.b.text(title)
            self.sections.append({"kind": "title", "title": title, "level": 0, "start": start, "end": self.b.n})
            self.b.block()
            self.meta["title"] = self.meta["title"] or title
            return
        self.b.block()
        start = self.b.mark()
        self.sections.append({"kind": self.mode, "title": title, "level": 1 if self.mode == "abstract" else
                              max(1, level - 1), "start": start, "end": start, "_open": True})
        self.b.text(title)
        self.b.block()

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        if self.skip_depth:
            return
        if self.heading is not None:
            self.heading.append(data)
            return
        if self.mode == "refs":
            if self.cur_ref is not None:
                self.cur_ref["_t"].append(data)
            return
        if not self.stack or self.containers.intersection(self.stack):
            self.b.text(data)

    def _meta(self, a):
        name = (a.get("name") or a.get("property") or "").lower()
        val = (a.get("content") or "").strip()
        if not val:
            return
        m, names = self.meta, self.meta_names
        if name in names["title"]:
            m["title"] = val
        elif name in names["authors"]:
            m["authors"].append(val)
        elif name in names["doi"]:
            m["doi"] = val.lower().removeprefix("doi:")
        elif name in names["journal"]:
            m["journal"] = val
        elif name in names["year"] and re.match(r"\d{4}", val):
            m["year"] = int(val[:4])
        elif name in names["licence"]:
            m["licence"]["id"] = val

    def result(self) -> dict:
        self._close_section()
        ids = {r["id"] for r in self.refs if r.get("id")}
        cites = [{"rids": [x["target"]], "label": "", "start": x["start"], "end": x["end"]}
                 for x in self.anchors if x["target"] in ids]
        if self.meta["title"] is None and self.title_parts:
            self.meta["title"] = _WS.sub(" ", "".join(self.title_parts)).strip() or None
        return _finish(self.b, self.sections, cites, self.refs, self.meta, "html")


def html_to_text(raw: bytes, ccfg: dict) -> dict:
    s = _decode_guard(raw, ccfg, xml=False)
    p = _Html(ccfg)
    p.feed(s)
    p.close()
    out = p.result()
    if not out["sections"]:
        raise ConversionError("no headed sections found in HTML: cannot tell body from abstract")
    return out


def convert(raw: bytes, fmt: str, ccfg: dict) -> dict:
    """Dispatch on the manifest's fulltext_format. Unknown formats are refused."""
    if fmt == "jats-xml":
        return jats_to_text(raw, ccfg)
    if fmt == "html":
        return html_to_text(raw, ccfg)
    raise ConversionError(f"unsupported fulltext_format {fmt!r}")
