"""CHUNK-VALUE-1110 WP1: pure chunk-value and known-bad-document rules (no I/O, no DB, no network, no env).

One shared module for the ingest filter (later, fulldoc_db.build_rows) and the known-bad audit. Every threshold, heading
list and word list is read from the config dict (config/chunk_value.json); the module holds only rule-id names.

Chunk level: evaluate_chunks(text, sections, cfg, chunking) -> keep/drop per live 400-character chunk with reason codes
and the dropped character ranges (offsets plus sha256 of the slice, never text). There is deliberately NO symbol or
alpha-ratio rule at chunk level (equations are evidence).
Document level: document_flags(doc, cfg, words, chunk_result) and find_duplicates(docs, cfg)."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from urllib.parse import urlparse

# rule ids (the only named constants)
C1, C2, C3, C4, C5, C6 = ("C1_section_kind", "C2_heading", "C3_boilerplate", "C4_reference_list", "C5_low_info",
                          "C6_near_dup")
D1, D2, D3, D4 = "D1_arxiv", "D2_incomprehensible", "D3_incomplete", "D4_duplicate"
CAP = "cap_exceeded"
MASK_CHAR = " "


def config_sha256(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ chunker
def chunk_spans(text: str, chunking: dict) -> list:
    """Same algorithm as DarCorpus._chunk_spans at c3aabb1; chunking = {chunk_chars, chunk_overlap, min_chunk_chars}."""
    size, ov, mn = int(chunking["chunk_chars"]), int(chunking["chunk_overlap"]), int(chunking["min_chunk_chars"])
    n, spans, pos = len(text), [], 0
    rx = re.compile(r"\S")
    while True:
        m = rx.search(text, pos)
        if not m:
            break
        start = m.start()
        end = n if start + size >= n else None
        if end is None:
            end = start + size
            ws = max((i for i in range(start + 1, end + 1) if text[i].isspace()), default=None)
            if ws is not None:
                end = ws
        reached = end >= n
        while end > start + 1 and text[end - 1].isspace():
            end -= 1
        spans.append((start, end))
        if reached:
            break
        nxt = max(end - ov, start + 1)
        while nxt < n and not text[nxt - 1].isspace():
            nxt += 1
        if nxt >= end:
            nxt = end
        pos = nxt
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < mn:
        last = spans.pop()
        spans[-1] = (spans[-1][0], last[1])
    return [(s, e) for s, e in spans if text[s:e].strip()]


# ------------------------------------------------------------------------------------------------ helpers
def _norm_heading(title, ccfg: dict) -> str:
    t = re.sub(ccfg["section_number_regex"], "", str(title or ""), count=1)
    t = re.sub(r"\s+", " ", t.casefold()).strip()
    return re.sub(r"[\W_]+$", "", t).strip()


def _valid_sections(sections, n: int) -> list:
    out = []
    for s in sections or ():
        try:
            if isinstance(s, dict) and 0 <= int(s["start"]) < int(s["end"]) <= n:
                out.append({"kind": s.get("kind"), "title": s.get("title") or "", "level": int(s.get("level") or 0),
                            "start": int(s["start"]), "end": int(s["end"])})
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out, key=lambda x: (x["start"], x["level"]))


def _merge(rs, gap: int = 0) -> list:
    out: list = []
    for a, b in sorted((int(a), int(b)) for a, b in rs):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _inside(ranges: list, a: int, b: int, gap: int) -> bool:
    return any(x <= a and b <= y + gap for x, y in ranges)


def _content_tokens(text: str, ccfg: dict, stop: set) -> list:
    rx = re.compile(ccfg["token_regex"])
    mn = int(ccfg["min_token_chars"])
    return [w for w in rx.findall(text.casefold()) if w not in stop and len(w) >= mn]


def _heading_ranges(secs: list, ccfg: dict) -> list:
    """C2 ranges: a section whose normalised heading equals a listed heading (exact), plus nested deeper sections of the
    same kind. back_only_headings apply to kind 'back' only."""
    h = ccfg["C2_heading"]
    kinds = set(h.get("heading_kinds") or ())
    any_h = {_norm_heading(x, ccfg) for x in h.get("drop_headings") or ()}
    back_h = {_norm_heading(x, ccfg) for x in h.get("back_only_headings") or ()}
    out = []
    for i, m in enumerate(secs):
        if m["kind"] not in kinds:
            continue
        nh = _norm_heading(m["title"], ccfg)
        if not nh or not (nh in any_h or (m["kind"] == "back" and nh in back_h)):
            continue
        end = m["end"]
        for s in secs[i + 1:]:
            if s["kind"] == m["kind"] and s["level"] > m["level"]:
                end = max(end, s["end"])
            else:
                break
        out.append((m["start"], end))
    return out


def _boilerplate_flags(text: str, ccfg: dict) -> list:
    """Per-line spans [(start, end, flagged)] by line features: URL share, a normalised line repeated >= N times."""
    b = ccfg["C3_boilerplate"]
    urlrx = re.compile(b["url_regex"])
    lines, pos = [], 0
    for ln in text.split("\n"):
        lines.append((pos, pos + len(ln), ln))
        pos += len(ln) + 1
    counts: dict = {}
    for _, _, ln in lines:
        k = re.sub(r"\s+", " ", ln.strip().casefold())
        if len(k) >= int(b["repeat_min_line_chars"]):
            counts[k] = counts.get(k, 0) + 1
    out = []
    for s, e, ln in lines:
        st = ln.strip()
        if not st:
            continue
        urlchars = sum(len(m.group(0)) for m in urlrx.finditer(st))
        k = re.sub(r"\s+", " ", st.casefold())
        flagged = (urlchars / len(st) >= float(b["url_share_max"])) or counts.get(k, 0) >= int(b["repeat_min_count"])
        out.append((s, e, flagged))
    return out


def _chunk_boiler(a: int, b: int, lines: list, share: float) -> bool:
    inl = [f for s, e, f in lines if s < b and e > a]
    return bool(inl) and sum(inl) / len(inl) >= share


# ------------------------------------------------------------------------------------------------ chunk rules
def evaluate_chunks(text: str, sections, cfg: dict, chunking: dict, fmt: str | None = None) -> dict:
    """Keep/drop decision per chunk. Returns:
      chunks: [{start, end, keep, rule}]  (rule None when kept; CAP rule is not stored per chunk, see `capped`)
      dropped: merged [{start, end, rule, sha256}] character ranges to mask (chunk spans minus kept-chunk coverage)
      kept: complement of `dropped` in [0, len(text)) (tiles with dropped exactly)
      by_rule: {rule: {chunks, chars}} ; substantive_chunks (kept by the rules before the cap)
      capped: True when the rules would drop more than max_evidence_drop_share of evidence-range characters; then the
              document is stored whole (dropped = [])."""
    cc = cfg["chunk"]
    n = len(text)
    gap = int(cc["section_gap_chars"])
    stop = set(cc["function_words"])
    secs = _valid_sections(sections, n)
    spans = chunk_spans(text, chunking)
    c1r = _merge([(s["start"], s["end"]) for s in secs if s["kind"] in set(cc["C1_section_kind"]["drop_section_kinds"])], gap) \
        if cc["C1_section_kind"].get("enabled") else []
    c2r = _merge(_heading_ranges(secs, cc), gap) if cc["C2_heading"].get("enabled") else []
    b3 = cc["C3_boilerplate"]
    lines = _boilerplate_flags(text, cc) if b3.get("enabled") and fmt in set(b3.get("formats") or ()) else None
    ref_rx = re.compile(cc["C4_reference_list"]["line_regex"], re.M)
    shn = int(cc["C6_near_dup"]["shingle_words"])
    seen: list = []  # shingle sets of kept chunks
    inv: dict = {}  # shingle -> [index into seen]
    reasons = []
    for a, b in spans:
        t = text[a:b]
        why = None
        if c1r and _inside(c1r, a, b, gap):
            why = C1
        elif c2r and _inside(c2r, a, b, gap):
            why = C2
        elif cc["C4_reference_list"].get("enabled") and len(ref_rx.findall(t)) >= int(cc["C4_reference_list"]["min_lines"]):
            why = C4
        elif lines is not None and _chunk_boiler(a, b, lines, float(b3["chunk_flagged_line_share"])):
            why = C3
        else:
            toks = _content_tokens(t, cc, stop)
            nsp = [ch for ch in t if not ch.isspace()]
            symdig = (sum(1 for ch in nsp if not ch.isalpha()) / len(nsp)) if nsp else 0.0
            exempt = symdig >= float(cc["C5_low_info"]["exempt_symbol_digit_share_min"])  # math/table chunk: C5 and C6 skip it
            if (cc["C5_low_info"].get("enabled") and not exempt
                    and len(set(toks)) < int(cc["C5_low_info"]["min_distinct_content_tokens"])):
                why = C5
            elif cc["C6_near_dup"].get("enabled") and not exempt:
                sh = {" ".join(toks[i:i + shn]) for i in range(max(len(toks) - shn + 1, 1))} if toks else set()
                if sh:
                    cand = {j for g in sh for j in inv.get(g, ())}
                    jm = float(cc["C6_near_dup"]["jaccard_min"])
                    if any(len(sh & seen[j]) / len(sh | seen[j]) >= jm for j in cand):
                        why = C6
                    else:
                        seen.append(sh)
                        for g in sh:
                            inv.setdefault(g, []).append(len(seen) - 1)
        reasons.append(why)
    rec = {"version": cfg.get("version"), "config_sha256": config_sha256(cfg), "chars": n,
           "chunks_before": len(spans), "capped": False}
    rec["substantive_chunks"] = sum(1 for w in reasons if w is None)
    rec["rule_chunks_pre_cap"] = {w: reasons.count(w) for w in set(reasons) if w}
    cover = bytearray(n)
    for (a, b), w in zip(spans, reasons):
        if w is None:
            cover[a:b] = b"\x01" * (b - a)
    dropped = []
    for (a, b), w in zip(spans, reasons):
        if w is None:
            continue
        i = a
        while i < b:
            if cover[i]:
                i += 1
                continue
            j = i
            while j < b and not cover[j]:
                j += 1
            if dropped and dropped[-1]["rule"] == w and dropped[-1]["end"] >= i:
                dropped[-1]["end"] = max(dropped[-1]["end"], j)
            else:
                dropped.append({"start": i, "end": j, "rule": w})
            i = j
    dropped.sort(key=lambda d: d["start"])
    # safety cap: counted rules inside evidence-kind sections vs evidence-range characters
    ev = _merge([(s["start"], s["end"]) for s in secs if s["kind"] in set(cc["evidence_section_kinds"])], gap)
    ev_chars = sum(y - x for x, y in ev)
    counted = set(cc["cap_counted_rules"])
    lost = 0
    for d in dropped:
        if d["rule"] in counted:
            for x, y in ev:
                lost += max(0, min(d["end"], y) - max(d["start"], x))
    rec["evidence_chars"], rec["evidence_chars_dropped"] = ev_chars, lost
    if ev_chars and lost / ev_chars > float(cc["max_evidence_drop_share"]):
        rec["capped"] = True
        dropped = []
        reasons_final = [None] * len(spans)
    else:
        reasons_final = reasons
    for d in dropped:
        d["sha256"] = _sha(text[d["start"]:d["end"]])
    kept, pos = [], 0
    for d in dropped:
        if d["start"] > pos:
            kept.append([pos, d["start"]])
        pos = d["end"]
    if pos < n:
        kept.append([pos, n])
    by_rule: dict = {}
    for (a, b), w in zip(spans, reasons_final):
        if w:
            r = by_rule.setdefault(w, {"chunks": 0, "chars": 0})
            r["chunks"] += 1
    for d in dropped:
        by_rule.setdefault(d["rule"], {"chunks": 0, "chars": 0})["chars"] += d["end"] - d["start"]
    rec.update({"chunks": [{"start": a, "end": b, "keep": w is None, "rule": w} for (a, b), w in zip(spans, reasons_final)],
                "dropped": dropped, "kept": kept, "by_rule": by_rule,
                "chunks_after": sum(1 for w in reasons_final if w is None),
                "chars_dropped": sum(d["end"] - d["start"] for d in dropped)})
    return rec


def mask_text(text: str, dropped) -> str:
    """The text with each dropped range replaced by MASK_CHAR (same length, offsets unchanged)."""
    out = list(text)
    for d in dropped:
        for i in range(d["start"], d["end"]):
            out[i] = MASK_CHAR
    return "".join(out)


# ------------------------------------------------------------------------------------------------ document rules
def _hosts_hit(urls, hosts: set) -> bool:
    for u in urls:
        try:
            h = (urlparse(str(u)).hostname or "").lower()
        except ValueError:
            continue
        if h in hosts:
            return True
    return False


def d1_arxiv(doc: dict, cfg: dict) -> list:
    c = cfg["document"][D1]
    if not c.get("enabled"):
        return []
    hosts = {h.lower() for h in c["hosts"]}
    urls = list(doc.get("urls") or ()) + [u for u in (doc.get("fulltext_url"), doc.get("orig_url")) if u]
    idents = [str(x).lower() for x in list(doc.get("identifiers") or ()) + [doc.get("doi")] + urls if x]
    repo = str(doc.get("repository") or "").lower()
    if repo and any(x.lower() in repo for x in c.get("repository_substrings", ())):
        return [D1 + ":repository"]
    if _hosts_hit(urls, hosts):
        return [D1 + ":host"]
    if any(sub.lower() in i for sub in c["identifier_substrings"] for i in idents):
        return [D1 + ":identifier"]
    if str(doc.get("resolver") or "").lower() in {r.lower() for r in c["resolvers"]}:
        return [D1 + ":resolver"]
    return []


def normalise_extraction(text: str) -> str:
    """Repair common extraction artefacts before the dictionary and non-alnum metrics: soft hyphens removed, ligatures
    and compatibility forms folded (NFKC), line-break hyphens joined ("compre-\\nhension" -> "comprehension")."""
    t = unicodedata.normalize("NFKC", text.replace("\u00ad", ""))
    return re.sub(r"(?<=[^\W\d_])-[ \t]*\n[ \t]*(?=[^\W\d_])", "", t)


def d2_metrics(text: str, cfg: dict, words=None) -> dict:
    """Garbled-extraction metrics at document level (pure numbers; thresholds are applied in d2_incomprehensible)."""
    c = cfg["document"][D2]
    n = max(len(text), 1)
    raw = text
    text = normalise_extraction(text)  # dictionary, language and non-alnum metrics see repaired text
    m: dict = {}
    nonws = [ch for ch in text if not ch.isspace()]
    m["non_alnum_share"] = (sum(1 for ch in nonws if not ch.isalnum()) / len(nonws)) if nonws else 1.0
    m["bad_chars_per_1000"] = 1000 * len(re.findall(c["bad_char_regex"], raw)) / n
    m["cid_per_1000"] = 1000 * len(re.findall(c["cid_regex"], raw)) / n
    m["hyphen_break_per_1000"] = 1000 * len(re.findall(c["hyphen_break_regex"], raw)) / n
    lines = [ln for ln in raw.split("\n") if ln.strip()]
    m["n_lines"] = len(lines)
    m["mean_line_chars"] = (sum(len(ln) for ln in lines) / len(lines)) if lines else 0.0
    cnt: dict = {}
    for ln in lines:
        k = re.sub(r"\s+", " ", ln.strip().casefold())
        if len(k) >= int(c["repeat_line_min_chars"]):
            cnt[k] = cnt.get(k, 0) + 1
    rep = sum(v for v in cnt.values() if v >= int(c["repeat_line_min_count"]))
    m["repeat_line_share"] = rep / len(lines) if lines else 0.0
    toks = re.findall(c["token_regex"], text)
    m["n_alpha_tokens"] = len(toks)
    allw = re.findall(r"[^\W\d_]+", text.casefold())
    mk = {w.casefold() for w in c["english_marker_words"]}
    m["english_marker_share"] = (sum(1 for w in allw if w in mk) / len(allw)) if allw else 0.0
    m["is_english"] = int(m["english_marker_share"] >= float(c["english_marker_share_min"]))
    m["math_heavy"] = int(m["non_alnum_share"] >= float(c["math_heavy_non_alnum_share_min"]))
    if words is not None and m["is_english"] and len(toks) >= int(c["min_tokens_for_dictionary"]):
        m["dictionary_word_share"] = sum(1 for t in toks if t.lower() in words) / len(toks)
    else:
        m["dictionary_word_share"] = None  # not evaluated (no word list, or too few tokens)
    return m


def d2_incomprehensible(text: str, cfg: dict, words=None) -> tuple:
    c = cfg["document"][D2]
    if not c.get("enabled"):
        return [], {}
    m = d2_metrics(text, cfg, words)
    hits = []
    if m["dictionary_word_share"] is not None and m["dictionary_word_share"] < float(c["dictionary_word_share_min"]):
        hits.append("dictionary_word_share")
    if m["non_alnum_share"] > float(c["non_alnum_share_max"]):
        hits.append("non_alnum_share")
    for k, lim in (("bad_chars_per_1000", "bad_chars_per_1000_max"), ("cid_per_1000", "cid_per_1000_max"),
                   ("hyphen_break_per_1000", "hyphen_break_per_1000_max"), ("repeat_line_share", "repeat_line_share_max")):
        if m[k] > float(c[lim]):
            hits.append(k)
    if m["n_lines"] >= int(c["min_lines_for_line_metrics"]) and m["mean_line_chars"] < float(c["mean_line_chars_min"]):
        hits.append("mean_line_chars")
    return [f"{D2}:{h}" for h in hits], m


def d2_severity(metrics: dict, hits=None) -> str:
    """'purge' only for detected-English, not math-heavy text; any other D2 hit is 'review' (never purged). A
    hyphen-break hit alone (readable text with PDF line-break hyphens) is review too."""
    if hits is not None and all(str(h).split(":")[-1] == "hyphen_break_per_1000" for h in hits):
        return "review"
    return "purge" if metrics.get("is_english") and not metrics.get("math_heavy") else "review"


def d3_code_severity(code: str, chunk_result: dict | None, cfg: dict) -> str:
    """missing_title alone is review (the body is intact); too_few follows d3_severity's C5 rule; the rest purge."""
    if code.endswith(":missing_title"):
        return "review"
    if code.endswith("too_few_substantive_chunks"):
        rc = (chunk_result or {}).get("rule_chunks_pre_cap") or {}
        dom = max(rc, key=rc.get) if rc else None
        return "review" if dom == cfg["document"][D3]["review_when_dominant_drop_rule"] else "purge"
    return "purge"


def d3_severity(code_list: list, chunk_result: dict | None, cfg: dict) -> str:
    return "purge" if any(d3_code_severity(x, chunk_result, cfg) == "purge" for x in code_list) else "review"



def d3_incomplete(doc: dict, cfg: dict, chunk_result: dict | None = None) -> list:
    c = cfg["document"][D3]
    if not c.get("enabled"):
        return []
    out = []
    text = doc.get("text") or ""
    prov = doc.get("provenance") or {}
    if doc.get("rebuild_ok") is False:
        out.append(D3 + ":rebuild_failed")
    for f in c["required_provenance_fields"]:
        if not prov.get(f):
            out.append(f"{D3}:missing_{f}")
    if c.get("check_sha") and prov.get("text_sha256") and text and _sha(text) != prov["text_sha256"]:
        out.append(D3 + ":sha_mismatch")
    if doc.get("has_body") is False:
        out.append(D3 + ":no_body")
    bc = doc.get("body_chars")
    if bc is not None and int(bc) < int(c["body_chars_min"]):
        out.append(D3 + ":body_chars_below_floor")
    if chunk_result is not None:
        sub, before = chunk_result["substantive_chunks"], chunk_result["chunks_before"]
        if sub < int(c["doc_min_chunks"]) or (before >= int(c["share_check_min_chunks"])
                                              and sub / before < float(c["doc_min_substantive_share"])):
            out.append(D3 + ":too_few_substantive_chunks")
    if c.get("truncation_check") and text and bc is not None and len(text) < int(bc):
        has_back = any(s.get("kind") in set(c["back_section_kinds"]) for s in doc.get("sections") or () if isinstance(s, dict))
        if not has_back and text.rstrip()[-1:] not in c["sentence_end_chars"]:
            out.append(D3 + ":truncated")
    return out


def document_flags(doc: dict, cfg: dict, words=None, chunk_result: dict | None = None) -> dict:
    """Known-bad predicates for one document (D1-D3; D4 needs the whole set: find_duplicates).
    doc: {text, sections, urls, identifiers, doi, resolver, provenance{licence_id, canonical_url, title, text_sha256},
          body_chars, has_body, rebuild_ok}. Returns {known_bad, reasons, metrics}."""
    reasons = d1_arxiv(doc, cfg)
    r2, metrics = d2_incomprehensible(doc.get("text") or "", cfg, words)
    r3 = d3_incomplete(doc, cfg, chunk_result)
    reasons += r2 + r3
    sev = {}
    for r in reasons:
        sev[r] = "purge" if r.startswith(D1) else (d2_severity(metrics, [r]) if r.startswith(D2) else d3_code_severity(r, chunk_result, cfg))
    return {"known_bad": bool(reasons), "reasons": reasons, "severity": sev,
            "purge": any(v == "purge" for v in sev.values()), "metrics": metrics}


def _sketch(text: str, dcfg: dict) -> tuple:
    toks = re.findall(dcfg["token_regex"], text.casefold())
    n = int(dcfg["shingle_words"])
    hs = {int.from_bytes(hashlib.blake2b(" ".join(toks[i:i + n]).encode("utf-8"), digest_size=8).digest(), "big")
          for i in range(max(len(toks) - n + 1, 1 if toks else 0))}
    return sorted(hs)[:int(dcfg["minhash_k"])], len(hs)


def _jaccard_est(a: list, b: list, k: int) -> float:
    sa, sb = set(a), set(b)
    union = sorted(sa | sb)[:k]
    if not union:
        return 0.0
    return sum(1 for h in union if h in sa and h in sb) / len(union)


def find_duplicates(docs: list, cfg: dict) -> dict:
    """D4. docs: [{id, text_sha256, doi, pmcid, text, acquired_at, licence_ok}] in any order. Duplicates by text hash,
    DOI/PMCID, or bottom-k MinHash Jaccard >= jaccard_min. Per group the keeper is the earliest acquired document with a
    passing licence (licence_ok not False), ties by input order; the rest map to {code, of, basis}."""
    c = cfg["document"][D4]
    if not c.get("enabled"):
        return {}
    ids = [d["id"] for d in docs]
    parent = {i: i for i in ids}
    basis: dict = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b, why):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            basis[frozenset((a, b))] = why

    for key, why in (("text_sha256", "text_sha256"), ("doi", "doi"), ("pmcid", "pmcid")):
        seen: dict = {}
        for d in docs:
            v = str(d.get(key) or "").strip().lower()
            if v:
                if v in seen:
                    union(seen[v], d["id"], why)
                else:
                    seen[v] = d["id"]
    k, jm = int(c["minhash_k"]), float(c["jaccard_min"])
    sk = {d["id"]: _sketch(d.get("text") or "", c) for d in docs if d.get("text")}
    keys = list(sk)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            na, nb = sk[a][1], sk[b][1]
            if not na or not nb or min(na, nb) / max(na, nb) < jm or find(a) == find(b):
                continue
            if _jaccard_est(sk[a][0], sk[b][0], k) >= jm:
                union(a, b, "minhash")
    order = {d["id"]: n for n, d in enumerate(docs)}
    byid = {d["id"]: d for d in docs}
    groups: dict = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    out = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        keeper = min(members, key=lambda i: (byid[i].get("licence_ok") is False, str(byid[i].get("acquired_at") or "9999"), order[i]))
        for i in members:
            if i != keeper:
                why = basis.get(frozenset((keeper, i))) or next((v for kk, v in basis.items() if i in kk), "minhash")
                out[i] = {"code": D4, "of": keeper, "basis": why}
    return out


# ------------------------------------------------------------------------------------------------ audit entry point
CONFIG = None  # callers (the audit) may set these two after loading the module; otherwise audit_document uses its args
WORDS = None   # optional lower-case word set for the D2 dictionary metric (skipped when None)


def _default_cfg() -> dict:
    """The only file read in this module, and only when no config was injected: config/chunk_value.json next to vera/."""
    from pathlib import Path
    return json.loads((Path(__file__).resolve().parent.parent / "config" / "chunk_value.json").read_text(encoding="utf-8"))


def audit_document(meta: dict, chunks, cfg: dict | None = None, words=None) -> dict:
    """Read-only audit entry. meta: {source_id, document_id, host, licence_id, doi, title_present, chars, text_sha256};
    chunks: [(span_id, start_index, end_index, text)] (stored partitions, contiguous). Sections are not available to the
    audit, so C1 and C2 (section rules) do not fire here; C3-C6 run on the rebuilt text with fmt=None.
    Returns {"d_flags": [{"rule": "D1|D2|D3", "metrics": {name: number}}], "dropped": {span_id: "C4"...}}: rule ids and
    numbers only, never text. A stored partition counts as dropped when every non-space character in it lies in a
    dropped range (rule = the rule covering most of it). D4 needs the whole set (find_duplicates) and is not returned."""
    cfg = cfg or CONFIG or _default_cfg()
    words = words if words is not None else WORDS
    parts = sorted(chunks, key=lambda c: int(c[1]))
    text = "".join(c[3] for c in parts)
    # offsets come from the text actually received, never from the declared indices (stored spans may disagree)
    mismatch = sum(1 for c in parts if int(c[2]) - int(c[1]) != len(c[3]))
    gaps = sum(1 for i in range(1, len(parts)) if int(parts[i][1]) != int(parts[i - 1][2]))
    base = int(parts[0][1]) if parts else 0
    cr = evaluate_chunks(text, None, cfg, cfg["audit"]["chunking"], fmt=None)
    d_flags = []
    doc = {"text": text, "urls": ["https://%s/" % meta["host"]] if meta.get("host") else [], "doi": meta.get("doi"),
           "fulltext_url": meta.get("fulltext_url"), "orig_url": meta.get("orig_url"), "repository": meta.get("repository"),
           "identifiers": [meta.get("identifier")] if meta.get("identifier") else [],
           "provenance": {"licence_id": meta.get("licence_id"), "canonical_url": "x", "title": "x" if meta.get("title_present") else None,
                          "text_sha256": meta.get("text_sha256")},
           "rebuild_ok": mismatch == 0 and gaps == 0 and base == 0}
    if d1_arxiv(doc, cfg):
        d_flags.append({"rule": "D1", "severity": "purge", "metrics": {"arxiv": 1}})
    hits, m = d2_incomprehensible(text, cfg, words)
    if hits:
        d_flags.append({"rule": "D2", "severity": d2_severity(m, hits),
                        "metrics": {k: float(m[k]) for k in [h.split(":", 1)[1] for h in hits] + ["is_english", "math_heavy"]}})
    r3 = d3_incomplete(doc, cfg, cr)
    if r3:
        d_flags.append({"rule": "D3", "severity": d3_severity(r3, cr, cfg), "metrics": {"substantive_chunks": cr["substantive_chunks"], "n_hits": len(r3),
                                                  "sha_mismatch": int(any(x.endswith("sha_mismatch") for x in r3)),
                                                  "rebuild_failed": int(any(x.endswith("rebuild_failed") for x in r3)),
                                                  "length_mismatch_spans": mismatch, "index_gaps": gaps}})
    dropped, lo = {}, 0
    for sid, _s, _e, t in parts:
        hi = lo + len(t)
        cover: dict = {}
        for d in cr["dropped"]:
            ov = min(d["end"], hi) - max(d["start"], lo)
            if ov > 0:
                cover[d["rule"]] = cover.get(d["rule"], 0) + ov
        if cover and all(any(d["start"] <= i < d["end"] for d in cr["dropped"])
                         for i in range(lo, hi) if not text[i].isspace()):
            dropped[sid] = max(cover, key=cover.get)
        lo = hi
    return {"d_flags": d_flags, "dropped": dropped}
