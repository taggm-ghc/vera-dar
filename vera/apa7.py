"""NONASK-1103 C1: APA 7 formatter (house variant: identifiers appended). Pure functions, no network except the
injectable public-URL check. Input is a CSL-JSON-shaped dict. Never invents metadata: missing author -> title moves,
missing date -> n.d.; a joined author string is ONE literal author (never split). Invalid identifiers are omitted,
never repaired (reason codes R-CIT-*). Preprint: Author. (Year). *Title*. Repository. (no "[Preprint]" label; APA
example, V-1b CONFIRMED-PRIMARY). Software: no "[Computer software]" label (HOUSE: guides split, V-1b). Every metadata string passes through _p() before markup is added.

CSL keys read: type, title, author [{family, given} | {literal}] | str (joined, literal), editor, issued
({"date-parts": [[y, m, d]]}), container_title, volume, issue, page, publisher, site_name, DOI, ISBN, URL, arxiv,
PMCID, PMID, handle, number, version, genre, accessed (date-parts), changing (bool), read_version ("arxiv" when the
arXiv text was read), foreign_ids (kinds that come from another provider record: never attached)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote, urlsplit

from vera.search_providers import ddg_identifier, normalise_doi
from vera.sources_sidebar import plain, plain_url

HOUSE_CAPTION = "APA 7 (VERA-DAR house variant: identifiers appended)"
STYLE_PATH = Path(__file__).resolve().parent.parent / "config" / "citation_style.json"
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
RESOLVER_HOSTS = {"doi": "doi.org", "handle": "hdl.handle.net", "arxiv": "arxiv.org",
                  "pmcid": "pmc.ncbi.nlm.nih.gov", "pmid": "pubmed.ncbi.nlm.nih.gov"}
MAX_NUMBER_LEN = 60  # display guard, not a decision threshold (plan section 7; provisional)
_UNSAFE_ID = re.compile(r"[()<>\"'`\s\\]")
_CTRL_WS = re.compile(r"[\x00-\x20\x7f]")
_ARXIV_NEW = re.compile(r"\d{4}\.\d{4,5}(v\d+)?")
_ARXIV_OLD = re.compile(r"[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?")
_HANDLE = re.compile(r"\d+(\.\d+)*/\S+")
_WEB_TYPES = {"webpage", "post-weblog", "document"}
_ITALIC_TITLE = {"book", "report", "dataset", "software", "webpage", "post-weblog", "document", "preprint", "thesis"}
LABELS = {"article-journal": "journal article", "preprint": "preprint", "article-newspaper": "news article",
          "webpage": "web page", "document": "official documentation", "dataset": "dataset", "software": "software"}


def citation_style(path: Path = STYLE_PATH) -> str:
    """Flag from config (seed/fallback); unreadable or unknown -> 'apa7'."""
    try:
        v = json.loads(Path(path).read_text(encoding="utf-8")).get("citation_style")
    except (OSError, ValueError, AttributeError):
        return "apa7"
    return v if v in ("apa7", "legacy") else "apa7"


def _p(text) -> str:
    """plain() for every field, then `\\<` / `\\>` become entities: ui_safety.safe_markdown turns raw < > into
    &lt; &gt;, which a preceding backslash would leave visible as literal "&lt;". All other escapes render invisibly."""
    return plain(text).replace("\\<", "&lt;").replace("\\>", "&gt;")


# ---------------------------------------------------------------- validation (omit, never repair)

def valid_isbn(v) -> str | None:
    s = re.sub(r"[-\s]", "", str(v or ""))
    if re.fullmatch(r"\d{13}", s):
        return s if sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(s)) % 10 == 0 else None
    if re.fullmatch(r"\d{9}[\dXx]", s):
        tot = sum((10 if c in "Xx" else int(c)) * (10 - i) for i, c in enumerate(s))
        return s.upper() if tot % 11 == 0 else None
    return None


def valid_doi(v, *, canonical: bool = False) -> str | None:
    raw = str(v or "").strip()
    d = normalise_doi(raw)
    if not d or _UNSAFE_ID.search(d):
        return None
    bare = re.sub(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)", "", raw, flags=re.I)
    return d if (canonical or bare.lower() == d) else None


def valid_arxiv(v) -> str | None:
    s = re.sub(r"^arxiv:", "", str(v or "").strip(), flags=re.I)
    return s if _ARXIV_NEW.fullmatch(s) or _ARXIV_OLD.fullmatch(s) else None


def valid_pmcid(v) -> str | None:
    s = str(v or "").strip().upper()
    return s if re.fullmatch(r"PMC\d{1,10}", s) else None


def valid_pmid(v) -> str | None:
    s = str(v or "").strip()
    return s if re.fullmatch(r"\d{1,8}", s) else None


def valid_handle(v) -> str | None:
    """Handle only from a hdl.handle.net https URL or an explicit field value (never a free-standing match)."""
    s = str(v or "").strip()
    try:
        sp = urlsplit(s)
        if sp.scheme in ("http", "https") and (sp.hostname or "").lower() == "hdl.handle.net":
            s = sp.path.lstrip("/")
    except ValueError:
        return None
    return s if _HANDLE.fullmatch(s) and not _UNSAFE_ID.search(s) else None


def _public(url: str) -> bool:
    """Display-time check, NO DNS (C2 finding: is_public_url resolves hostnames and resets the fetch pin slot, which a
    render path must not do; the link is only shown, never fetched by the server). Rejects loopback/private/link-local
    IP literals and local-only host names; all fetched URLs already passed is_public_url upstream."""
    import ipaddress
    try:
        host = (urlsplit(url).hostname or "").lower().rstrip(".")
        if not host or host == "localhost" or host.endswith((".local", ".localhost", ".internal", ".lan", ".home.arpa")) \
                or "." not in host and not host.startswith("["):
            return False
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return True
        return ip.is_global
    except Exception:  # noqa: BLE001  fail closed
        return False


def safe_source_url(url) -> str | None:
    """https only, default port, no userinfo, no control/whitespace, public host. None = reject."""
    u = str(url or "")
    if not u or _CTRL_WS.search(u) or re.search(r"[<>\"\\`^{}|]", u):
        return None
    try:
        sp = urlsplit(u)
        port = sp.port
    except ValueError:
        return None
    if sp.scheme != "https" or not sp.hostname or sp.username is not None or sp.password is not None \
            or port not in (None, 443):
        return None
    return u.replace("(", "%28").replace(")", "%29") if _public(u) else None


def safe_link(kind: str, value) -> str | None:
    """The only link builder. https, fixed resolver host per kind; None = no link (render plain text/omit)."""
    v = str(value or "")
    if kind == "url":
        return safe_source_url(v)
    host = RESOLVER_HOSTS.get(kind)
    if not host or not v:
        return None
    q = quote(v, safe="/.:-_")
    return {"doi": f"https://doi.org/{q}", "handle": f"https://hdl.handle.net/{q}",
            "arxiv": f"https://arxiv.org/abs/{q}", "pmcid": f"https://pmc.ncbi.nlm.nih.gov/articles/{q}/",
            "pmid": f"https://pubmed.ncbi.nlm.nih.gov/{q}/"}[kind]


# ---------------------------------------------------------------- identifiers

def identifiers(rec: dict, codes: list | None = None) -> dict:
    """kind -> validated value for kinds that pass (doi, handle, arxiv, pmcid, pmid, isbn). Invalid -> omitted."""
    codes = codes if codes is not None else []
    foreign = set(rec.get("foreign_ids") or ())
    out: dict = {}
    from_url = ddg_identifier(rec.get("URL") or "") or {}

    def take(kind, raw, check, **kw):
        if kind in foreign:
            return
        if raw not in (None, ""):
            val = check(raw, **kw)
            if val:
                out[kind] = val
            else:
                codes.append(f"R-CIT-BADID-{kind}")
    take("doi", rec.get("DOI"), valid_doi)
    take("arxiv", rec.get("arxiv"), valid_arxiv)
    take("pmcid", rec.get("PMCID"), valid_pmcid)
    take("pmid", rec.get("PMID"), valid_pmid)
    take("handle", rec.get("handle") or (rec.get("URL") if "hdl.handle.net" in str(rec.get("URL") or "").lower()
                                         else None), valid_handle)
    take("isbn", rec.get("ISBN"), lambda v: valid_isbn(v))
    k = from_url.get("kind")
    if k and k not in out and k not in foreign and not rec.get(
            {"doi": "DOI", "arxiv": "arxiv", "pmcid": "PMCID", "pmid": "PMID"}[k]):
        val = from_url["value"] + (from_url.get("version") or "") if k == "arxiv" else from_url["value"]
        val = valid_doi(val, canonical=True) if k == "doi" else val
        if val:
            out[k] = val
    return out


def select_identifiers(rec: dict, codes: list | None = None) -> list[tuple[str, str]]:
    """Ordered [(kind, value)] to append: version read first, else DOI > Handle > arXiv > PMCID > PMID > ISBN >
    number > URL; plus at most one secondary that reaches a different resource."""
    codes = codes if codes is not None else []
    ids = identifiers(rec, codes)
    typ = rec.get("type") or ""
    if typ not in ("book", "chapter"):
        ids.pop("isbn", None)
    url = safe_source_url(rec.get("URL")) if rec.get("URL") else None
    if rec.get("URL") and not url:
        codes.append("R-CIT-BADURL")
    if typ in _WEB_TYPES | {"article-newspaper", "dataset", "software"} and url:
        return [("url", str(rec["URL"]))]
    if rec.get("read_version") == "arxiv" and "arxiv" in ids:
        out = [("arxiv", ids["arxiv"])]
        if "doi" in ids:
            out.append(("doi", ids["doi"]))
        return out
    for first in ("doi", "handle", "arxiv", "pmcid", "pmid", "isbn"):
        if first in ids:
            out = [(first, ids[first])]
            if first == "doi" and "pmcid" in ids:
                out.append(("pmcid", ids["pmcid"]))
            elif first == "doi" and "isbn" in ids:
                out.append(("isbn", ids["isbn"]))
            return out
    return [("url", str(rec["URL"]))] if url else []


def _render_id(kind: str, value: str, links: bool) -> str:
    if kind == "url":
        link, text = safe_link("url", value), plain_url(value)
    else:
        link = safe_link(kind, value)
        text = {"doi": f"https://doi.org/{value}", "handle": f"https://hdl.handle.net/{value}",
                "arxiv": f"arXiv:{value}", "pmcid": f"PMCID: {value}", "pmid": f"PMID: {value}",
                "isbn": f"ISBN {value}"}[kind]
        text = plain_url(text) if kind in ("doi", "handle") else text
    return f"[{text}]({link})" if (links and link and kind != "isbn") else text


# ---------------------------------------------------------------- names, dates

def _initials(given: str) -> str:
    parts = [p for p in re.split(r"[\s.]+", str(given).strip()) if p]
    out = []
    for p in parts:
        out.append("-".join(_p(h[0]) + "." for h in p.split("-") if h))
    return " ".join(out)


def _name(a) -> tuple[str, str, bool]:
    """(display 'Family, I. I.' or literal, sort key, is_family_form) for one author."""
    if isinstance(a, str):
        return _p(a), a.lower(), False
    if a.get("literal") or not a.get("family"):
        lit = str(a.get("literal") or a.get("name") or "")
        return _p(lit), lit.lower(), False
    fam = _p(a["family"])
    giv = _initials(a["given"]) if a.get("given") else ""
    return (f"{fam}, {giv}" if giv else fam), str(a["family"]).lower(), True


def _authors(rec: dict) -> list:
    a = rec.get("author")
    if not a:
        return []
    if isinstance(a, str):
        return [a]  # joined string = one literal author, never split
    return [x for x in a if x]


def _author_str(auths: list) -> str:
    names = [_name(a)[0] for a in auths]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]}, & {names[1]}"
    if len(names) <= 20:
        return ", ".join(names[:-1]) + f", & {names[-1]}"
    return ", ".join(names[:19]) + ", . . . " + names[-1]


def _editors(rec: dict) -> str:
    eds = rec.get("editor") or []
    if isinstance(eds, str):
        eds = [eds]
    out = []
    for e in eds:
        if isinstance(e, str) or e.get("literal") or not e.get("family"):
            out.append(_name(e)[0])
        else:
            out.append((_initials(e["given"]) + " " if e.get("given") else "") + _p(e["family"]))
    if not out:
        return ""
    s = out[0] if len(out) == 1 else (f"{out[0]} & {out[1]}" if len(out) == 2 else ", ".join(out[:-1]) + f", & {out[-1]}")
    return f"{s} ({'Ed.' if len(out) == 1 else 'Eds.'})"


def _date_parts(d) -> list:
    try:
        parts = (d or {}).get("date-parts", [[]])[0]
        return [int(p) for p in parts if p is not None][:3]
    except (AttributeError, IndexError, TypeError, ValueError):
        return []


def _year(rec) -> str:
    p = _date_parts(rec.get("issued"))
    return str(p[0]) if p else "n.d."


def _fmt_date(parts: list, full: bool) -> str:
    if not parts:
        return "n.d."
    if full and len(parts) >= 2 and 1 <= parts[1] <= 12:
        return f"{parts[0]}, {MONTHS[parts[1] - 1]}" + (f" {parts[2]}" if len(parts) > 2 else "")
    return str(parts[0])


def _dot(s: str) -> str:
    return s if s.endswith((".", "?", "!")) else s + "."


def _title(rec: dict) -> str:
    return _p(rec.get("title") or "")


# ---------------------------------------------------------------- input normaliser

def normalise(rec: dict) -> dict:
    """Map real CSL-JSON spellings onto the frozen key list; values are never changed. container-title ->
    container_title; ids {handle, arxiv, pmcid, pmid} -> top-level; type article + genre preprint -> preprint;
    provider/fulltext_url naming arXiv -> read_version; _same_record False entries -> foreign_ids."""
    r = dict(rec)
    if "container-title" in r and "container_title" not in r:
        r["container_title"] = r["container-title"]
    for k, v in (r.get("ids") or {}).items():
        dst = {"pmcid": "PMCID", "pmid": "PMID"}.get(k, k)
        r.setdefault(dst, v)
    if r.get("type") == "article" and str(r.get("genre") or "").lower() == "preprint":
        r["type"] = "preprint"
    if "read_version" not in r and (str(r.get("provider") or "").lower() == "arxiv"
                                    or "arxiv.org" in str(r.get("fulltext_url") or "").lower()):
        r["read_version"] = "arxiv"
    sr = r.get("_same_record") or {}
    if sr and "foreign_ids" not in r:
        r["foreign_ids"] = [{"DOI": "doi", "PMCID": "pmcid", "PMID": "pmid"}.get(k, k.lower())
                            for k, v in sr.items() if v is False]
    return r


# ---------------------------------------------------------------- reference

def reference_ex(rec: dict, *, suffix: str = "", links: bool = True) -> tuple[str, list]:
    codes: list = []
    rec = normalise(rec)
    typ = rec.get("type") or ""
    full = typ in _WEB_TYPES | {"article-newspaper"}
    auths = _authors(rec)
    if any(isinstance(a, str) or (a.get("literal") and "," in a["literal"]) for a in auths):
        codes.append("R-CIT-AUTHLIT")
    date = _fmt_date(_date_parts(rec.get("issued")), full)
    date = date + suffix if date != "n.d." or not suffix else f"n.d.{('-' + suffix)}"
    title = _title(rec)
    t_it = f"*{title}*" if title and typ in _ITALIC_TITLE else title
    cont = _p(rec["container_title"]) if rec.get("container_title") else ""
    pub = _p(rec["publisher"]) if rec.get("publisher") else ""
    a_str = _author_str(auths)
    if pub and a_str and pub == a_str:
        pub = ""
    ids = select_identifiers(rec, codes)
    idtxt = " ".join(_render_id(k, v, links) for k, v in ids)
    num = str(rec["number"]) if rec.get("number") else ""
    if num and len(num) > MAX_NUMBER_LEN:
        codes.append("R-CIT-NUMLEN")
        num = ""
    num = _p(num) if num else ""
    ver = _p(rec["version"]) if rec.get("version") else ""
    lead, rest = "", ""  # lead = the title element (moves to the author position when there is no author)
    if typ == "article-journal":
        lead = _dot(title) if title else ""
        if cont:
            vol = f"*{_p(rec['volume'])}*" if rec.get("volume") else ""
            iss = f"({_p(rec['issue'])})" if rec.get("issue") else ""
            pg = f", {_p(str(rec['page']).replace('-', '–'))}" if rec.get("page") else ""
            rest = f"*{cont}*" + (f", {vol}{iss}" if vol or iss else "") + pg + "."
    elif typ == "preprint":
        lead = _dot(t_it) if t_it else ""  # APA example (V-1b, CONFIRMED-PRIMARY): no "[Preprint]" label
        repo = cont or pub or ("arXiv" if any(k == "arxiv" for k, _ in ids) else "")
        rest = _dot(repo) if repo else ""
    elif typ == "chapter":
        lead = _dot(title) if title else ""
        eds = _editors(rec)
        inner = ("In " + (eds + ", " if eds else "")) if (eds or cont) else ""
        if cont:
            pg = f" (pp. {_p(str(rec['page']).replace('-', '–'))})" if rec.get("page") else ""
            inner += f"*{cont}*{pg}."
        rest = (inner + (" " + _dot(pub) if pub else "")).strip()
    elif typ == "article-newspaper":
        lead = _dot(title) if title else ""
        rest = f"*{cont}*." if cont else ""
    elif typ == "dataset":
        lead = (t_it + (f" ({num})" if num else "") + " [Data set].") if t_it else ""
        rest = _dot(pub) if pub else ""
    elif typ == "software":
        lead = _dot(t_it + (f" (Version {ver})" if ver else "")) if t_it else ""
        rest = _dot(pub) if pub else ""
    elif typ in _WEB_TYPES:
        lead = _dot(t_it) if t_it else ""
        site = _p(rec.get("site_name") or rec.get("container_title") or "")
        rest = _dot(site) if site and site != a_str else ""
    else:  # book, report, thesis, stored rows of unknown type
        lead = _dot(t_it + (f" (Report No. {num})" if num and typ == "report" else "")) if t_it else ""
        rest = ((f"*{cont}*." if cont else "") + (" " + _dot(pub) if pub else "")).strip()
    if a_str:
        head = [_dot(a_str), f"({date}).", lead, rest]
    else:
        head = [lead, f"({date}).", rest]
    acc = _date_parts(rec.get("accessed"))
    if (rec.get("changing") or typ in _WEB_TYPES) and acc and not _date_parts(rec.get("issued")) and ids \
            and ids[0][0] == "url":
        idtxt = f"Retrieved {MONTHS[acc[1] - 1]} {acc[2]}, {acc[0]}, from {idtxt}" if len(acc) == 3 and 1 <= acc[1] <= 12 else idtxt
    out = " ".join(s.strip() for s in head if s and s.strip())
    if idtxt:
        out = f"{out} {idtxt}".strip()
    return out, codes


def reference(rec: dict, *, suffix: str = "", links: bool = True) -> str:
    return reference_ex(rec, suffix=suffix, links=links)[0]


# ---------------------------------------------------------------- lists, a/b, in-text

def _sort_key(rec: dict):
    a = _authors(rec)
    return (_name(a[0])[1] if a else (rec.get("title") or "").lower(), _year(rec), (rec.get("title") or "").lower())


def suffixes(records: list[dict]) -> list[str]:
    """a/b/c for records sharing first-author key and year (ordered by title); '' otherwise."""
    groups: dict = {}
    for i, r in enumerate(records):
        a = _authors(r)
        groups.setdefault((_name(a[0])[1] if a else None, _year(r), None if a else (r.get("title") or "")), []).append(i)
    out = [""] * len(records)
    for k, idx in groups.items():
        if len(idx) > 1 and k[0] is not None:
            for n, i in enumerate(sorted(idx, key=lambda j: (records[j].get("title") or "").lower())):
                out[i] = "abcdefghijklmnopqrstuvwxyz"[n % 26]
    return out


def reference_list(records: list[dict], *, links: bool = True) -> list[str]:
    sfx = suffixes(records)
    pairs = sorted(zip(records, sfx), key=lambda p: _sort_key(p[0]) + (p[1],))
    return [reference(r, suffix=s, links=links) for r, s in pairs]


def _in_text_core(rec: dict, suffix: str, narrative: bool) -> str:
    rec = normalise(rec)
    auths = _authors(rec)
    date = _year(rec) + suffix
    if not auths:
        title = _title(rec)
        if rec.get("type") in _ITALIC_TITLE:
            return f"(*{title}*, {date})" if not narrative else f"*{title}* ({date})"
        return f'("{title}," {date})' if not narrative else f'"{title}" ({date})'
    names = []
    for a in auths:
        names.append(_p(a) if isinstance(a, str) else
                     _p(a["family"]) if a.get("family") and not a.get("literal") else
                     _p(str(a.get("literal") or a.get("name") or "")))
    if len(names) == 1:
        who = names[0]
    elif len(names) == 2:
        who = f"{names[0]} and {names[1]}" if narrative else f"{names[0]} & {names[1]}"
    else:
        who = f"{names[0]} et al."
    return f"{who} ({date})" if narrative else f"({who}, {date})"


def in_text(rec, *, ns=None, narrative: bool = False, context: list | None = None) -> str:
    """One record or a list of records (alphabetical by first element, joined by '; '). `ns` = passage locators
    shown as ' [1, 3]'. `context` = all records of the answer, for a/b disambiguation."""
    recs = [normalise(x) for x in (rec if isinstance(rec, (list, tuple)) else [rec])]
    orig = list(rec) if isinstance(rec, (list, tuple)) else [rec]
    pool_o = list(context) if context else orig
    sfx = suffixes([normalise(x) for x in pool_o])
    smap = {id(o): s for o, s in zip(pool_o, sfx)}
    recs = sorted(zip(orig, recs), key=lambda p: _sort_key(p[1]) + (smap.get(id(p[0]), ""),))
    recs = [(smap.get(id(o), ""), r) for o, r in recs]
    if len(recs) == 1:
        s = _in_text_core(recs[0][1], recs[0][0], narrative)
    else:
        inner = [_in_text_core(r, sf, False)[1:-1] for sf, r in recs]
        s = "(" + "; ".join(inner) + ")"
    if ns:
        s += " [" + ", ".join(str(int(n)) for n in sorted(set(ns))) + "]"
    return s


def characterise(rec: dict) -> str:
    """'<Type label> · <outlet/publisher if supplied> · <date or n.d.>'; type from the input field only."""
    rec = normalise(rec)
    full = (rec.get("type") or "") in _WEB_TYPES | {"article-newspaper"}
    mid = rec.get("container_title") or rec.get("publisher") or ""
    if not mid and rec.get("type") in _WEB_TYPES and rec.get("URL"):
        try:
            mid = "site: " + (urlsplit(str(rec["URL"])).hostname or "")
        except ValueError:
            mid = ""
        mid = mid if mid != "site: " else ""
    parts = [LABELS.get(rec.get("type") or "", "Source")]
    if mid:
        parts.append(mid if mid.startswith("site: ") else _p(mid))
    parts.append(_fmt_date(_date_parts(rec.get("issued")), full))
    return " · ".join(parts)


_RUN = re.compile(r"(?:\[\d+(?:\s*,\s*\d+)*\])+")


def display_answer(answer: str, ns: dict) -> str:
    """Display-time substitution of runs of [n] markers by one APA in-text citation plus the locator, e.g.
    "[1][3]" -> "(Alpha, 2020; Zeta, 2018) [1, 3]". `ns` maps n -> record; passages of one document (same
    doc_key, else identical record) merge into one citation. Returns a new string; `answer` is never changed."""
    def doc(n):
        r = ns.get(n) or ns.get(str(n))
        return None if r is None else r.get("doc_key") or json.dumps(r, sort_keys=True, default=str)

    def sub(m):
        nums = sorted({int(x) for x in re.findall(r"\d+", m.group(0))})
        known = [n for n in nums if doc(n) is not None]
        if not known:
            return m.group(0)
        seen, recs = set(), []
        for n in known:
            if doc(n) not in seen:
                seen.add(doc(n))
                recs.append(ns.get(n) or ns.get(str(n)))
        return in_text(recs, ns=known)
    return _RUN.sub(sub, answer)
