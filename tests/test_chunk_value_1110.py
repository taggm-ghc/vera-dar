"""CHUNK-VALUE-1110 WP1 tests for vera.chunk_value. Synthetic fixtures only; no DB, network or env."""
import copy
import hashlib
import json
from pathlib import Path

from vera import chunk_value as cv

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config" / "chunk_value.json").read_text(encoding="utf-8"))
CH = {"chunk_chars": 400, "chunk_overlap": 60, "min_chunk_chars": 40}
WORDS = {"the", "of", "and", "model", "forecast", "evidence", "accuracy", "interval", "scores", "results", "that", "were", "data"}


def para(seed, n=60):
    return " ".join(f"w{seed}x{i % 17}y{(i * seed) % 23}" for i in range(n))


def doc_with(back_title, back_text, body_paras=8):
    body = "\n\n".join(para(s) for s in range(1, body_paras + 1))
    title = "A study of things"
    text = title + "\n\n" + body + "\n\n" + back_title + "\n\n" + back_text
    t0 = 0
    b0 = len(title) + 2
    b1 = b0 + len(body)
    k0 = b1 + 2
    secs = [{"kind": "title", "title": title, "level": 0, "start": 0, "end": len(title)},
            {"kind": "body", "title": "Results", "level": 1, "start": b0, "end": b1},
            {"kind": "back", "title": back_title, "level": 1, "start": k0, "end": len(text)}]
    return text, secs


def test_back_heading_dropped_exact_only():
    text, secs = doc_with("Acknowledgements", para(99, 120))
    r = cv.evaluate_chunks(text, secs, CFG, CH)
    rules = {d["rule"] for d in r["dropped"]}
    assert rules == {cv.C2}
    d = r["dropped"][0]
    assert d["start"] >= secs[2]["start"] and d["sha256"] == hashlib.sha256(text[d["start"]:d["end"]].encode()).hexdigest()
    # prefix is not enough: "Acknowledgement of limitations" stays
    text2, secs2 = doc_with("Acknowledgement of limitations", para(99, 120))
    assert cv.evaluate_chunks(text2, secs2, CFG, CH)["dropped"] == []


def test_body_supplementary_kept_but_back_supplementary_dropped():
    t, s = doc_with("Supplementary materials", para(98, 120))
    assert [d["rule"] for d in cv.evaluate_chunks(t, s, CFG, CH)["dropped"]] == [cv.C2]
    s[2]["kind"] = "body"
    assert cv.evaluate_chunks(t, s, CFG, CH)["dropped"] == []


def test_tiling_and_mask_preserve_offsets():
    text, secs = doc_with("Funding", para(97, 120))
    r = cv.evaluate_chunks(text, secs, CFG, CH)
    pieces = sorted([(a, b, "k") for a, b in r["kept"]] + [(d["start"], d["end"], "d") for d in r["dropped"]])
    pos = 0
    for a, b, _ in pieces:
        assert a == pos
        pos = b
    assert pos == len(text)
    m = cv.mask_text(text, r["dropped"])
    assert len(m) == len(text)
    for d in r["dropped"]:
        assert m[d["start"]:d["end"]].strip() == ""
    for a, b in r["kept"]:
        assert m[a:b] == text[a:b]


def test_low_info_and_near_dup():
    good = "\n\n".join(para(s) for s in range(1, 6))
    low = "the of and the of and the of and " * 14
    text = good + "\n\n" + low.strip()
    r = cv.evaluate_chunks(text, None, CFG, CH)
    assert cv.C5 in {c["rule"] for c in r["chunks"]}
    def lw(seed, n=70):
        return " ".join("".join(chr(97 + (i * seed + j * 3) % 26) for j in range(5)) for i in range(n))
    text = lw(5, 400)  # periodic text: interior chunks repeat one another
    r = cv.evaluate_chunks(text, None, CFG, CH)
    assert cv.C6 in {c["rule"] for c in r["chunks"]}


def test_no_symbol_ratio_rule_equations_kept():
    eq = " ".join("where w%dalpha%s equals \\sum_{i=1}^{n}\\alpha_i^2+\\beta/%d" % (i, chr(97 + i % 26), i) for i in range(30))
    r = cv.evaluate_chunks(eq, None, CFG, CH)
    assert r["dropped"] == [] and r["chunks_after"] == r["chunks_before"]


def test_reference_list_chunk():
    refs = "\n".join(f"{i}. Smith J, Jones K. Title of paper number {i}. J Stuff. 2020;1:{i}." for i in range(1, 12))
    text = para(1) + "\n\n" + refs
    r = cv.evaluate_chunks(text, None, CFG, CH)
    assert cv.C4 in {c["rule"] for c in r["chunks"]}


def test_cap_stores_document_whole():
    cfg = copy.deepcopy(CFG)
    low = "the of and the of and the of and " * 40
    text = para(1, 40) + "\n\n" + low.strip()
    secs = [{"kind": "body", "title": "R", "level": 1, "start": 0, "end": len(text)}]
    r = cv.evaluate_chunks(text, secs, cfg, CH)
    assert r["capped"] is True and r["dropped"] == [] and r["kept"] == [[0, len(text)]]
    cfg["chunk"]["max_evidence_drop_share"] = 1.0
    assert cv.evaluate_chunks(text, secs, cfg, CH)["capped"] is False


def test_title_chunk_dropped_when_whole_chunk_is_title():
    text = "Short title only"
    secs = [{"kind": "title", "title": text, "level": 0, "start": 0, "end": len(text)}]
    r = cv.evaluate_chunks(text, secs, CFG, CH)
    assert [c["rule"] for c in r["chunks"]] == [cv.C1]


def test_pure_deterministic_and_config_sha():
    text, secs = doc_with("References", para(96, 100))
    a, b = cv.evaluate_chunks(text, secs, CFG, CH), cv.evaluate_chunks(text, secs, CFG, CH)
    assert a == b and a["config_sha256"] == cv.config_sha256(CFG)


def test_boilerplate_only_for_text_formats():
    nav = "\n".join(["Home | About | Contact | https://example.org/a/b/c/d"] * 12)
    text = para(1) + "\n\n" + nav
    assert cv.C3 not in {c["rule"] for c in cv.evaluate_chunks(text, None, CFG, CH, fmt="jats")["chunks"]}
    assert cv.C3 in {c["rule"] for c in cv.evaluate_chunks(text, None, CFG, CH, fmt="html")["chunks"]}


def good_doc():
    text, secs = doc_with("Data availability", para(95, 80), body_paras=12)
    return {"text": text, "sections": secs, "urls": ["https://europepmc.org/article/PMC1"], "doi": "10.1371/x",
            "provenance": {"licence_id": "CC BY 4.0", "canonical_url": "https://europepmc.org/x", "title": "T",
                           "text_sha256": hashlib.sha256(text.encode()).hexdigest()},
            "body_chars": 5000, "has_body": True}


def test_document_flags_good_and_arxiv():
    d = good_doc()
    f = cv.document_flags(d, CFG)
    assert f["known_bad"] is False, f
    for urls, ids in ((["https://arxiv.org/abs/2301.00001"], []), (["https://export.arxiv.org/x"], []),
                      ([], ["10.48550/arXiv.2301.00001"])):
        d2 = dict(d, urls=urls, identifiers=ids)
        assert cv.document_flags(d2, CFG)["reasons"][0].startswith(cv.D1)
    assert cv.document_flags(dict(d, urls=["https://notarxiv.org.evil.com/"]), CFG)["known_bad"] is False
    assert cv.document_flags(dict(d, resolver="arxiv"), CFG)["reasons"] == ["D1_arxiv:resolver"]


def test_document_flags_garbled():
    d = good_doc()
    junk = "�Ã© (cid:12) " * 400
    f = cv.document_flags(dict(d, text=junk, provenance=dict(d["provenance"], text_sha256=cv._sha(junk))), CFG)
    assert any(r.startswith(cv.D2) for r in f["reasons"])
    words_text = "model forecast evidence data " * 100
    d3 = dict(d, text=words_text, provenance=dict(d["provenance"], text_sha256=cv._sha(words_text)))
    assert not any(r.startswith(cv.D2) for r in cv.document_flags(d3, CFG, WORDS)["reasons"])
    gib = "the of and qzxv lkjw mnbp rtyu is that " * 100
    d4 = dict(d, text=gib, provenance=dict(d["provenance"], text_sha256=cv._sha(gib)))
    assert "D2_incomprehensible:dictionary_word_share" in cv.document_flags(d4, CFG, WORDS)["reasons"]
    assert cv.d2_metrics(gib, CFG, None)["dictionary_word_share"] is None


def test_document_flags_incomplete():
    d = good_doc()
    bad = dict(d, provenance=dict(d["provenance"], licence_id="", text_sha256="0" * 64), body_chars=100, has_body=False,
               rebuild_ok=False)
    r = cv.document_flags(bad, CFG)["reasons"]
    for code in ("missing_licence_id", "sha_mismatch", "body_chars_below_floor", "no_body", "rebuild_failed"):
        assert f"{cv.D3}:{code}" in r
    cr = cv.evaluate_chunks("x " * 30, None, CFG, CH)
    assert f"{cv.D3}:too_few_substantive_chunks" in cv.document_flags(d, CFG, None, cr)["reasons"]


def test_find_duplicates():
    a, b = para(1, 400), para(2, 400)
    near = a + " extra tail words here"
    docs = [{"id": "late", "text": a, "acquired_at": "2026-02", "text_sha256": "h1"},
            {"id": "early", "text": a, "acquired_at": "2026-01", "text_sha256": "h1"},
            {"id": "near", "text": near, "acquired_at": "2026-03"},
            {"id": "other", "text": b},
            {"id": "d1", "doi": "10.1/ABC"}, {"id": "d2", "doi": "10.1/abc", "licence_ok": False}]
    out = cv.find_duplicates(docs, CFG)
    assert set(out) == {"late", "near", "d2"}
    assert out["late"]["of"] == "early" and out["late"]["basis"] == "text_sha256"
    assert out["near"]["basis"] == "minhash" and out["d2"]["of"] == "d1"


def test_audit_document_contract_and_no_text():
    import re
    text, secs = doc_with("Funding", para(94, 100), body_paras=14)
    low = "the of and the of and the of and " * 130
    full = text + "\n\n" + low.strip()
    parts, pos, i = [], 0, 0
    while pos < len(full):
        e = min(pos + 1000, len(full))
        parts.append((f"s{i}", pos, e, full[pos:e]))
        pos, i = e, i + 1
    meta = {"source_id": 1, "document_id": "d", "host": "arxiv.org", "licence_id": "CC BY 4.0", "doi": None,
            "title_present": True, "chars": len(full), "text_sha256": hashlib.sha256(full.encode()).hexdigest()}
    out = cv.audit_document(meta, parts, CFG)
    assert [f["rule"] for f in out["d_flags"]][0] == "D1"
    rx = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,39}$")
    for f in out["d_flags"]:
        assert rx.match(f["rule"]) and all(isinstance(v, (int, float)) for v in f["metrics"].values())
    assert out["dropped"] and set(out["dropped"]) <= {p[0] for p in parts}
    assert set(out["dropped"].values()) == {cv.C5}
    assert "s0" not in out["dropped"]
    json.dumps(out)
    clean = cv.audit_document(dict(meta, host="europepmc.org"), parts, CFG)
    assert all(f["rule"] != "D1" for f in clean["d_flags"])


def test_audit_chunking_matches_routing_config():
    r = json.loads((ROOT / "config" / "domain_routing.json").read_text(encoding="utf-8"))["corpus"]
    assert CFG["audit"]["chunking"] == {k: r[k] for k in CH}


def test_audit_tolerates_length_mismatch_and_gaps():
    text = para(1, 80) + "\n\n" + ("the of and the of and the of and " * 80)
    parts = [("a", 0, 5, text[:1000]), ("b", 5, 2000, text[1000:2000]), ("c", 9999, 9999, text[2000:])]
    meta = {"host": "europepmc.org", "licence_id": "x", "title_present": True, "text_sha256": None}
    out = cv.audit_document(meta, parts, CFG)
    d3 = [f for f in out["d_flags"] if f["rule"] == "D3"][0]["metrics"]
    assert d3["length_mismatch_spans"] == 3 and d3["rebuild_failed"] == 1 and d3["index_gaps"] >= 1
    assert cv.audit_document(meta, [], CFG)["dropped"] == {}


def test_d3_chunk_thresholds_short_docs_not_flagged():
    d = good_doc()
    short = cv.evaluate_chunks(para(1, 40), None, CFG, CH)
    assert short["chunks_before"] < CFG["document"]["D3_incomplete"]["share_check_min_chunks"]
    assert f"{cv.D3}:too_few_substantive_chunks" not in cv.document_flags(d, CFG, None, short)["reasons"]
    many = {"substantive_chunks": 4, "chunks_before": 12}
    assert f"{cv.D3}:too_few_substantive_chunks" in cv.document_flags(d, CFG, None, many)["reasons"]


ES = ("La estimación del modelo se realizó con los datos de la población y los resultados muestran una variación importante "
      "entre las regiones durante el período considerado por el estudio ")
DE = ("Die Schätzung des Modells wurde mit den Daten der Bevölkerung durchgeführt und die Ergebnisse zeigen eine erhebliche "
      "Variation zwischen den Regionen im untersuchten Zeitraum der Studie ")
RU = ("Оценка модели была выполнена по данным о населении, и результаты показывают значительные различия между регионами "
      "в течение рассматриваемого периода исследования ")
MATH = " ".join("$\\sum_{i=1}^{%d}\\alpha_i^2+\\beta_{%d}/\\gamma$ (%d) = [%d]" % (i, i, i, i * 7) for i in range(400))
TABLE = "\n".join("| %d | %.2f | %.3f | %d%% | %.1f |" % (i, i / 7, i / 13, i % 100, i * 1.5) for i in range(500))


def varied(s, n=60):
    return " ".join(f"{s} r{i}x{i * 7}" for i in range(n))


def _plain(text):
    d = good_doc()
    d.update(text=text, provenance=dict(d["provenance"], text_sha256=cv._sha(text)), sections=None, body_chars=None)
    return d


def test_non_english_math_table_never_purge_flagged():
    words = {"the"}  # an English-only list: foreign text would fail it if the metric ran
    for name, txt in (("es", varied(ES)), ("de", varied(DE)), ("ru", varied(RU)), ("math", MATH), ("table", TABLE)):
        f = cv.document_flags(_plain(txt), CFG, words)
        assert f["purge"] is False, (name, f["reasons"], f["metrics"])
        assert f["metrics"].get("dictionary_word_share") is None or f["metrics"]["is_english"]
        meta = {"host": "europepmc.org", "licence_id": "x", "title_present": True, "text_sha256": cv._sha(txt)}
        parts = [(f"s{i}", i * 1000, min((i + 1) * 1000, len(txt)), txt[i * 1000:(i + 1) * 1000]) for i in range((len(txt) + 999) // 1000)]
        out = cv.audit_document(meta, parts, CFG, words)
        assert all(x["severity"] == "review" for x in out["d_flags"]) or not out["d_flags"], (name, out["d_flags"])


def test_c5_unicode_and_math_table_exempt():
    for txt in (varied(RU), varied(ES), MATH, TABLE):
        r = cv.evaluate_chunks(txt, None, CFG, CH)
        assert sum(1 for c in r["chunks"] if c["rule"] == cv.C5) <= 1, txt[:20]  # at most a short tail chunk
        assert r["chunks_after"] >= 0.9 * r["chunks_before"], txt[:20]
    assert cv.C5 in {c["rule"] for c in cv.evaluate_chunks(("the of and " * 60).strip(), None, CFG, CH)["chunks"]}


def test_d3_review_when_c5_dominant_and_english_purge_garble():
    low = ("the of and " * 400).strip()
    cr = cv.evaluate_chunks(low, None, CFG, CH)
    f = cv.document_flags(good_doc(), CFG, None, cr)
    assert f["severity"][f"{cv.D3}:too_few_substantive_chunks"] == "review" and f["purge"] is False
    cr2 = {"substantive_chunks": 0, "chunks_before": 20, "rule_chunks_pre_cap": {cv.C4: 20}}
    assert cv.document_flags(good_doc(), CFG, None, cr2)["purge"] is True
    en = ("the model of the data is that results are " * 120)
    junk = en + "\ufffd" * 400
    assert cv.document_flags(_plain(junk), CFG)["purge"] is True


def test_d1_arxiv_via_provenance_fields():
    d = good_doc()
    for k, v in (("fulltext_url", "https://export.arxiv.org/pdf/1"), ("repository", "arXiv"), ("identifiers", ["10.48550/arXiv.2301.1"])):
        r = cv.document_flags(dict(d, **{k: v}), CFG)
        assert r["reasons"][0].startswith(cv.D1) and r["severity"][r["reasons"][0]] == "purge", k
    meta = {"host": "europepmc.org", "licence_id": "x", "title_present": True, "text_sha256": None, "fulltext_url": "https://arxiv.org/pdf/1"}
    out = cv.audit_document(meta, [("a", 0, 5, "hello")], CFG)
    assert out["d_flags"][0] == {"rule": "D1", "severity": "purge", "metrics": {"arxiv": 1}}


def test_d2_repairs_pdf_hyphenation_and_ligatures():
    line = "the model of the forecast and the evidence that results were fore-\ncast accuracy of the data inter-\nval scores are ﬁne\n"
    txt = "".join(line.replace("evidence", "evidence q%d" % i).replace("data", "data r%d" % i).replace("\ufb01ne", "\ufb01ne s%d" % i) for i in range(40))
    d = _plain(txt)
    f = cv.document_flags(d, CFG, WORDS | {"fine", "interval", "forecast", "s", "q", "r"})
    assert "D2_incomprehensible:dictionary_word_share" not in f["reasons"]
    assert f["metrics"]["is_english"] == 1 and f["metrics"]["dictionary_word_share"] >= 0.8
    assert f["purge"] is False, f["reasons"]  # a hyphen-break hit alone is review
    assert cv.normalise_extraction("compre-\nhension ­ﬁx") == "comprehension fix"
    # unrepaired broken words would have failed the dictionary
    raw_share = sum(1 for w in __import__("re").findall("[A-Za-z]{3,}", txt) if w.lower() in WORDS) / len(__import__("re").findall("[A-Za-z]{3,}", txt))
    assert raw_share < f["metrics"]["dictionary_word_share"]


def test_d3_missing_title_alone_is_review():
    d = good_doc()
    d["provenance"] = dict(d["provenance"], title=None)
    f = cv.document_flags(d, CFG)
    assert f["reasons"] == [f"{cv.D3}:missing_title"] and f["purge"] is False
    d["provenance"] = dict(d["provenance"], licence_id=None)
    assert cv.document_flags(d, CFG)["purge"] is True
    meta = {"host": "europepmc.org", "licence_id": "x", "title_present": False, "text_sha256": None}
    txt = " ".join(f"alpha{chr(97 + i % 26)}{chr(97 + i // 26 % 26)}word sentence number" for i in range(400))
    out = cv.audit_document(dict(meta, text_sha256=cv._sha(txt)), [("a", 0, len(txt), txt)], CFG)
    assert [(f["rule"], f["severity"]) for f in out["d_flags"] if f["rule"] == "D3"] == [("D3", "review")]
    d2 = [f for f in out["d_flags"] if f["rule"] == "D2"]
    assert all({"is_english", "math_heavy"} <= set(f["metrics"]) for f in d2)


def _ru(a, b):  # distinct Russian-like docs sharing only the ASCII tokens "1 2 2020"
    return " ".join(f"{a}{i % 37}слово{b}{i % 11}" for i in range(300)) + " 1 2 2020"


def test_d4_non_english_unrelated_not_duplicates_and_true_dup_is():
    ru1 = "альфа бета гамма " * 5 + "1 2 2020"
    ru2 = "дельта эпсилон зета " * 5 + "1 2 2020"
    zh1 = "天气预报模型的准确性评估 1 2 2020"
    zh2 = "海洋生物多样性调查报告 1 2 2020"
    docs = [{"id": i, "text": t} for i, t in enumerate((ru1, ru2, zh1, zh2))]
    assert cv.find_duplicates(docs, CFG) == {}
    long_a = " ".join(f"слово{i}фраза{(i * 7) % 13}" for i in range(400))
    long_b = " ".join(f"другое{i}текст{(i * 5) % 17}" for i in range(400))
    out = cv.find_duplicates([{"id": "a", "text": long_a}, {"id": "b", "text": long_a + " хвост"}, {"id": "c", "text": long_b}], CFG)
    assert set(out) == {"b"} and out["b"]["of"] == "a" and out["b"]["basis"] == "minhash"
    assert cv.find_duplicates([{"id": "x", "text": ru1, "doi": "10.1/q"}, {"id": "y", "text": ru2, "doi": "10.1/Q"}], CFG)["y"]["basis"] == "doi"


def test_c4_numbered_protocol_kept_real_reference_list_matches():
    proto = "\n".join(f"{i}. Mix Buffer A with Solution B and incubate sample {i} for ten minutes." for i in range(1, 12))
    r = cv.evaluate_chunks(para(1) + "\n\n" + proto, None, CFG, CH)
    assert cv.C4 not in {c["rule"] for c in r["chunks"]}
    refs = "\n".join(f"{i}. Smith AB, Jones K. Title of paper {i}. J Stuff. 2020;1:{i}." for i in range(1, 12))
    assert cv.C4 in {c["rule"] for c in cv.evaluate_chunks(para(1) + "\n\n" + refs, None, CFG, CH)["chunks"]}
    doi = "\n".join(f"[{i}] Garcia-Lopez, M. Another title {i}. doi:10.1234/abc{i}" for i in range(1, 8))
    assert cv.C4 in {c["rule"] for c in cv.evaluate_chunks(para(1) + "\n\n" + doi, None, CFG, CH)["chunks"]}


def test_d2_dictionary_only_hit_is_review_and_shipped_disabled():
    assert CFG["enabled"] is False
    txt = "the of and cytokine phosphorylation kinase receptor ligand mitochondrial apoptosis " * 60
    f = cv.document_flags(_plain(txt), CFG, WORDS)
    assert f["reasons"] == ["D2_incomprehensible:dictionary_word_share"], f["reasons"]
    assert f["metrics"]["is_english"] == 1 and f["purge"] is False
    assert f["severity"]["D2_incomprehensible:dictionary_word_share"] == "review"
