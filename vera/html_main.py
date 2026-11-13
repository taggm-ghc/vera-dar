"""ALLDOMAINS-WPG-IMPL-1111 (USFED-ROUTE-1111 steps 6a-6e, host-neutral): generic main-content extraction for any
permitted-original HTML route. No per-site code: every parameter is config (config/permitted_originals.json "extract"),
the same pattern for every host. stdlib html.parser only (no script execution, no external loads).

  parse(data)                       -> Node tree (depth and node bounds; raises HtmlRefused)
  visible_text(doc)                 -> the FULL visible text of the page (licence and marker checks run on THIS, before
                                       any pruning, so pruning never hides a copyright, third-party or contractor marker)
  capture_provenance(doc, pcfg)     -> byline / credit / licence statement / licence_href from the full page (6e);
                                       pruning applies to the stored body only, never to this record
  extract_main(raw, xcfg, host, boilerplate) -> {"html", "root", "stats", "doc"}: 6a root, 6b prune, 6c host
                                       boilerplate drop; the result goes to jats_text.html_to_text (6d, by the caller)
  block_hashes(raw, xcfg, host)     -> hashes of the page's leaf-block lines (used by the catalogue refresh to learn 6c)

Headings (h1-h6) are never pruned by token, density or the boilerplate set (FM-11). All numeric values are priors given in
config and are UNCALIBRATED until the per-row C0 (calibration note in the config).
"""
from __future__ import annotations

import hashlib
import re
from html import escape
from html.parser import HTMLParser

VOID = frozenset({"br", "img", "hr", "meta", "link", "input", "source", "wbr", "col", "area", "base", "embed",
                  "param", "track"})
HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_WS = re.compile(r"\s+")
# G-C MAJOR-1 (USFED-SHADOW-1113): every repetition is bounded so search() is linear on a long token without "@"
# (the unbounded form was quadratic: 80k chars 3.7 s). A longer prefix is still found via its last 64 characters.
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,190}\.[A-Za-z]{2,24}")
_NEVER_VISIBLE = frozenset({"script", "style", "noscript", "template", "head", "svg"})
_BLOCK = frozenset({"p", "li", "div", "td", "th", "dd", "dt", "blockquote", "pre", "section", "article", "ul", "ol",
                    "table", "tr", "dl", "figure", "figcaption", "main", "aside", "nav", "header", "footer", "form"})
_LEAF_BLOCK = frozenset({"p", "li", "div", "td", "th", "dd", "dt", "blockquote", "pre", "figcaption"})


class HtmlRefused(ValueError):
    """The page is not parsed (too deep or too many nodes). Carries no page text."""


class Node:
    __slots__ = ("tag", "attrs", "kids", "parent")

    def __init__(self, tag: str, attrs, parent):
        self.tag = tag
        self.attrs = {k: (v or "") for k, v in attrs}
        self.kids: list = []
        self.parent = parent


class _Parser(HTMLParser):
    def __init__(self, max_depth: int, max_nodes: int):
        super().__init__(convert_charrefs=True)
        self.root = Node("#doc", [], None)
        self.cur = self.root
        self.depth = 0
        self.n = 0
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    def _count(self):
        self.n += 1
        if self.n > self.max_nodes:
            raise HtmlRefused("too many nodes")

    def handle_starttag(self, tag, attrs):
        self._count()
        node = Node(tag, attrs, self.cur)
        self.cur.kids.append(node)
        if tag not in VOID:
            self.cur = node
            self.depth += 1
            if self.depth > self.max_depth:
                raise HtmlRefused("too deep")

    def handle_startendtag(self, tag, attrs):
        self._count()
        self.cur.kids.append(Node(tag, attrs, self.cur))

    def handle_endtag(self, tag):
        x, d = self.cur, self.depth
        while x is not self.root and x.tag != tag:
            x, d = x.parent, d - 1
        if x is not self.root:
            self.cur, self.depth = x.parent, d - 1

    def handle_data(self, data):
        self.cur.kids.append(data)


def parse(data, max_depth: int = 200, max_nodes: int = 400000) -> Node:
    s = data.decode("utf-8", "replace") if isinstance(data, (bytes, bytearray)) else str(data)
    p = _Parser(int(max_depth), int(max_nodes))
    p.feed(s)
    p.close()
    return p.root


def walk(n):
    yield n
    for k in n.kids:
        if isinstance(k, Node):
            yield from walk(k)


def norm_line(s: str) -> str:
    return _WS.sub(" ", s or "").strip().lower()


def line_hash(s: str) -> str:
    return hashlib.sha256(norm_line(s).encode("utf-8")).hexdigest()


def _text(n: Node) -> str:
    out = []

    def rec(x):
        for k in x.kids:
            if isinstance(k, str):
                out.append(k)
            elif k.tag not in _NEVER_VISIBLE:
                rec(k)
    rec(n)
    return _WS.sub(" ", "".join(out)).strip()


def _tot_link(n: Node, inlink: bool = False) -> tuple:
    tot = lk = 0
    for k in n.kids:
        if isinstance(k, str):
            ln = len(k.strip())
            tot += ln
            lk += ln if inlink else 0
        elif k.tag not in _NEVER_VISIBLE:
            a, b = _tot_link(k, inlink or k.tag == "a")
            tot += a
            lk += b
    return tot, lk


def visible_text(doc: Node) -> str:
    """The full visible text of the page, in document order (block boundaries become line breaks)."""
    out: list = []

    def rec(x):
        for k in x.kids:
            if isinstance(k, str):
                out.append(k)
            elif k.tag in _NEVER_VISIBLE:
                continue
            else:
                brk = k.tag in _BLOCK or k.tag in HEADINGS or k.tag == "br"
                if brk:
                    out.append("\n")
                rec(k)
                if brk:
                    out.append("\n")
    rec(doc)
    lines = [_WS.sub(" ", ln).strip() for ln in "".join(out).split("\n")]
    return "\n".join(ln for ln in lines if ln)


def _title(doc: Node) -> str:
    for n in walk(doc):
        if n.tag == "title":
            return _WS.sub(" ", "".join(k for k in n.kids if isinstance(k, str))).strip()
    return ""


# ---------------------------------------------------------------------------------------------------- 6e provenance
def capture_provenance(doc: Node, pcfg: dict | None) -> dict:
    """Byline, credit and licence statements and the licence link, from the FULL page, bounded and short. A statement
    is a short block whose text matches `statement_re`; only that block's first `max_chars` characters are kept."""
    pcfg = pcfg or {}
    mx = int(pcfg.get("max_chars", 300))
    cap = int(pcfg.get("max_items", 12))
    meta_names = {str(x).lower() for x in pcfg.get("author_meta_names") or ()}
    byline_re = re.compile(str(pcfg.get("byline_re") or r"(?:^|[-_ ])(byline|author|credit)(?:$|[-_ ])"), re.I)
    stmt_re = re.compile(str(pcfg.get("statement_re") or r"(?i)credit|courtesy|©|copyright|licen[cs]e|terms of use|"
                                                         r"public domain|reuse"))
    pub_names = {str(x).lower() for x in pcfg.get("published_meta_names") or ()}
    published = ""
    authors, bylines, statements, href = [], [], [], ""
    for n in walk(doc):
        if n.tag == "meta":
            nm = (n.attrs.get("name") or n.attrs.get("property") or "").lower()
            if nm in pub_names and not published and re.match(r"\s*(?:19|20)\d{2}", n.attrs.get("content", "")):
                published = n.attrs["content"].strip()[:32]
            if nm in meta_names and n.attrs.get("content", "").strip():
                authors.append(_WS.sub(" ", n.attrs["content"]).strip()[:mx])
        elif n.tag in ("a", "link") and not href:
            rel = (n.attrs.get("rel") or "").lower().split()
            if "license" in rel and n.attrs.get("href"):
                href = n.attrs["href"].strip()[:500]
        if n.tag in _NEVER_VISIBLE or n.tag in ("html", "body", "#doc", "main"):
            continue
        toks = " ".join([n.attrs.get("class", ""), n.attrs.get("id", ""), n.attrs.get("itemprop", "")])
        if toks.strip() and byline_re.search(toks):
            t = _text(n)
            if t and len(t) <= 4 * mx:
                bylines.append(t[:mx])
        if n.tag in _LEAF_BLOCK or n.tag in ("small", "span", "footer", "address"):
            t = _text(n)
            if t and len(t) <= mx and stmt_re.search(t):
                statements.append(t)
    def uniq(xs):
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
            if len(out) >= cap:
                break
        return out
    return {"authors": uniq(authors), "bylines": uniq(bylines), "statements": uniq(statements), "licence_href": href,
            "published": published}


# ---------------------------------------------------------------------------------------------------- 6a-6c extraction
class _X:
    def __init__(self, xcfg: dict):
        self.drop_tags = {str(t) for t in xcfg.get("drop_tags") or ()}
        self.drop_roles = {str(t).lower() for t in xcfg.get("drop_roles") or ()}
        self.token = re.compile(str(xcfg.get("drop_token_re") or r"(?!x)x"), re.I)
        self.dmax = float(xcfg.get("link_density_max", 0.5))
        self.dchars = int(xcfg.get("link_density_block_max_chars", 3000))
        self.roots = [str(r) for r in xcfg.get("roots") or ("main", "[role=main]", "article")]
        self.dtags = {"div", "section", "ul", "ol", "li", "p", "table", "dl"}
        # MT-B EX-2 / EX-3 (USFED-MATCH-1113, plan 5.8): both keys fail closed (a bad value refuses the page).
        self.sec = None
        sr = xcfg.get("drop_section_heading_re")
        if sr is not None:
            try:
                if not isinstance(sr, str) or not sr:
                    raise ValueError("type")
                self.sec = re.compile(sr, re.I)
            except (re.error, ValueError):
                raise HtmlRefused("bad drop_section_heading_re") from None
        em = xcfg.get("drop_email_blocks")
        if em is not None and not isinstance(em, bool):
            raise HtmlRefused("bad drop_email_blocks")
        self.email = em is True
        self.drop_level = None  # heading level of the section being dropped (state of ONE extract call)

    def why(self, n: Node):
        if n.tag in self.drop_tags:
            return "tag"
        if (n.attrs.get("role") or "").lower() in self.drop_roles:
            return "role"
        if n.attrs.get("aria-hidden", "").lower() == "true" or "hidden" in n.attrs:
            return "hidden"
        if n.tag in HEADINGS:
            return None  # headings are never pruned by token or density
        for t in (n.attrs.get("class", "") + " " + n.attrs.get("id", "")).split():
            if self.token.search(t):
                return "token"
        if n.tag in self.dtags:
            tot, lk = _tot_link(n)
            if tot and tot <= self.dchars and lk / tot > self.dmax:
                return "linkdensity"
        return None


def _has_heading(n: Node) -> bool:
    return any(isinstance(x, Node) and x.tag in HEADINGS for x in walk(n))


def _is_leaf_block(n: Node) -> bool:
    if n.tag not in _LEAF_BLOCK:
        return False
    return not any(x is not n and (x.tag in _BLOCK or x.tag in HEADINGS) for x in walk(n))


def pick_root(doc: Node, xcfg: dict, host: str | None = None):
    """6a: first selector with non-empty text (largest text if several), else <body>, else the document."""
    ov = ((xcfg.get("host_overrides") or {}).get((host or "").lower()) or {}).get("root")
    sels = [str(ov)] if ov else [str(r) for r in xcfg.get("roots") or ("main", "[role=main]", "article")]
    for sel in sels:
        if sel.startswith("[role=") and sel.endswith("]"):
            want = sel[6:-1].lower()
            c = [n for n in walk(doc) if (n.attrs.get("role") or "").lower() == want]
        else:
            c = [n for n in walk(doc) if n.tag == sel]
        c = [n for n in c if _text(n)]
        if c:
            return max(c, key=lambda n: len(_text(n))), sel
    body = [n for n in walk(doc) if n.tag == "body"]
    return (body[0] if body else doc), "body"


def _ser(n, out: list, x: _X, stats: dict, bp: set | None) -> None:
    if isinstance(n, str):
        if x.drop_level is None:
            out.append(escape(n, quote=False))
        return
    if n.tag in HEADINGS:
        lvl = int(n.tag[1])
        if x.drop_level is not None and lvl <= x.drop_level:
            x.drop_level = None  # the next heading of the same or higher level ends the dropped section
        if x.drop_level is not None:  # a sub-heading inside a dropped section: no orphan heading text
            stats["section"] = stats.get("section", 0) + 1
            return
        if n.tag != "h1" and x.sec is not None and x.sec.fullmatch(_WS.sub(" ", _text(n)).strip()):
            x.drop_level = lvl
            stats["section"] = stats.get("section", 0) + 1
            return
    elif x.drop_level is not None and not _has_heading(n):
        stats["section_block"] = stats.get("section_block", 0) + 1
        return
    why = x.why(n)
    if why:
        stats[why] = stats.get(why, 0) + 1
        return
    if x.email and n.tag in _LEAF_BLOCK and _is_leaf_block(n) and not _has_heading(n):
        t = _text(n)
        if t and _EMAIL.search(t):  # EX-3: a contact line is not explanatory body; no length cap
            stats["contact_line"] = stats.get("contact_line", 0) + 1
            return
    if bp and n.tag in _LEAF_BLOCK and _is_leaf_block(n) and not _has_heading(n):
        t = _text(n)
        if t and line_hash(t) in bp:
            stats["boilerplate"] = stats.get("boilerplate", 0) + 1
            return
    out.append("<%s>" % n.tag)
    for k in n.kids:
        _ser(k, out, x, stats, bp)
    if n.tag not in VOID:
        out.append("</%s>" % n.tag)


def extract_main(raw, xcfg: dict, host: str | None = None, boilerplate=None, doc: Node | None = None) -> dict:
    """6a-6c over the page. `boilerplate` is the host's set of line hashes (hex sha256) from the snapshot, or None. The
    root element itself is kept; its children are pruned. Returns the pruned HTML (bytes) for the converter."""
    doc = doc if doc is not None else parse(raw, xcfg.get("max_depth", 200), xcfg.get("max_nodes", 400000))
    x = _X(xcfg)
    root, sel = pick_root(doc, xcfg, host)
    out: list = []
    stats: dict = {}
    bp = set(boilerplate) if boilerplate else None
    for k in root.kids:
        _ser(k, out, x, stats, bp)
    html = "<html><head><title>%s</title></head><body><article>%s</article></body></html>" % (
        escape(_title(doc), quote=False), "".join(out))
    return {"html": html.encode("utf-8"), "root": sel, "stats": stats, "doc": doc}


def block_hashes(raw, xcfg: dict, host: str | None = None) -> set:
    """Hashes of the leaf-block lines that survive 6a-6b (headings excluded): the input of the refresh script's per-host
    boilerplate count (a line on at least `boilerplate_min_pages` pages of the host is boilerplate)."""
    doc = parse(raw, xcfg.get("max_depth", 200), xcfg.get("max_nodes", 400000))
    x = _X(xcfg)
    root, _sel = pick_root(doc, xcfg, host)
    hashes: set = set()

    def rec(n):
        if x.why(n):
            return
        if _is_leaf_block(n) and not _has_heading(n):
            t = _text(n)
            if t:
                hashes.add(line_hash(t))
            return
        for k in n.kids:
            if isinstance(k, Node):
                rec(k)
    for k in root.kids:
        if isinstance(k, Node):
            rec(k)
    return hashes
