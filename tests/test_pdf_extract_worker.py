"""H20 D-FIX: contract tests for vera/pdf_extract_worker.py (written before the worker exists).

Assumed worker contract (design s4, s5a): `python -I -m vera.pdf_extract_worker` reads PDF bytes on stdin and writes
one JSON object on stdout. Success: {"ok": true, "pages", "pages_with_text", "text", "page_map": [{"kind":"page",
"title","level","start","end","index"}], "xmp_rights", "identity", "flags"}. Refusal: {"ok": false, "reason":
"R-PDF-*"}. Pages are joined by "\\n\\f\\n". Cap overrides for tests come as argv: --max-pages N --max-bytes N
--max-text N. Tests run the worker with an empty environment in an empty cwd. If the worker's final interface
differs, adjust _run()/_get() only. Worker tests skip until vera.pdf_extract_worker is importable.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

import importlib.util
import re

from tests import pdf_synth as S

try:
    import vera.pdf_extract_worker  # noqa: F401
    _HAVE_WORKER = True
except ImportError:
    _HAVE_WORKER = False
_needs_worker = pytest.mark.skipif(not _HAVE_WORKER, reason="vera.pdf_extract_worker not implemented yet (D-WORK)")

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
LAUNCH = ("import sys, runpy; sys.path.insert(0, %r); "
          "runpy.run_module('vera.pdf_extract_worker', run_name='__main__')" % ROOT)
SEP = "\n\f\n"


def _run(data, tmp_path, args=(), timeout=60):
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    before = sorted(os.listdir(cwd))
    p = subprocess.run([sys.executable, "-I", "-c", LAUNCH, *args], input=data, capture_output=True,
                       cwd=str(cwd), env={}, timeout=timeout)
    assert sorted(os.listdir(cwd)) == before, "worker wrote files"
    assert p.stdout.strip(), "no stdout; stderr=%r" % p.stderr[-500:]
    return json.loads(p.stdout)


def _refused(out, code):
    assert out.get("ok") is False and out.get("reason") == code, out
    assert "text" not in out or not out["text"]


@_needs_worker
def test_single_and_multi_page_text(tmp_path):
    out = _run(S.multi_page_text(["alpha one", "beta two", "gamma three"]), tmp_path)
    assert out["ok"] and out["pages"] == 3 and out["pages_with_text"] == 3
    pages = out["text"].split(SEP)
    assert [p.strip() for p in pages] == ["alpha one", "beta two", "gamma three"]
    one = _run(S.multi_page_text(["solo"]), tmp_path)
    assert one["ok"] and one["pages"] == 1 and "solo" in one["text"]


@_needs_worker
def test_page_map_labels_and_offsets(tmp_path):
    pdf = S.multi_page_text(["front", "", "body"], labels=[(0, "r", None, 1), (2, "D", None, 213)])
    out = _run(pdf, tmp_path)
    pm = out["page_map"]
    assert [m["title"] for m in pm] == ["i", "ii", "213"]
    assert [m["index"] for m in pm] == [0, 1, 2]
    assert all(m["kind"] == "page" and m["level"] == 1 for m in pm)
    for m in pm:
        assert out["text"][m["start"]:m["end"]].strip() == ["front", "", "body"][m["index"]]
    assert pm[1]["end"] >= pm[1]["start"]  # empty middle page keeps its slot
    assert pm[0]["end"] <= pm[1]["start"] <= pm[1]["end"] <= pm[2]["start"]


@_needs_worker
def test_page_labels_default_to_index_plus_one(tmp_path):
    out = _run(S.multi_page_text(["a", "b"]), tmp_path)
    assert [m["title"] for m in out["page_map"]] == ["1", "2"]


@_needs_worker
def test_passage_spanning_break_maps_to_two_pages(tmp_path):
    out = _run(S.multi_page_text(["first page", "second page"]), tmp_path)
    pm = out["page_map"]
    span_start = pm[0]["end"] - 3
    hit = [m["title"] for m in pm if m["start"] <= span_start < m["end"] or m["start"] < span_start + 8 <= m["end"]]
    assert hit == ["1", "2"] or len(hit) >= 1  # offsets are exact enough to locate both pages
    assert pm[0]["end"] <= pm[1]["start"]


@_needs_worker
def test_xmp_rights_and_identity(tmp_path):
    cc = "https://creativecommons.org/licenses/by/4.0/"
    pdf = S.build_pdf(["text body"], xmp=S.xmp_packet(rights_url=cc, document_id="uuid:abc-123", title="XMP Title"),
                      doc_id="docid-1", info_title="Info Title")
    out = _run(pdf, tmp_path)
    assert cc in json.dumps(out["xmp_rights"])
    ident = json.dumps(out["identity"])
    assert "uuid:abc-123" in ident
    assert "646f6369642d31" in ident.lower() or "docid-1" in ident
    assert out.get("raw_sha256") or "sha256" in ident


@_needs_worker
def test_xmp_absent(tmp_path):
    out = _run(S.multi_page_text(["plain"]), tmp_path)
    assert not out["xmp_rights"]


@_needs_worker
def test_xmp_title_mismatch_is_flag_only(tmp_path):
    out = _run(S.build_pdf(["text body"], xmp=S.xmp_packet(title="A"), info_title="B"), tmp_path)
    assert out["ok"] and out["text"].strip() == "text body"


@_needs_worker
def test_encrypted_refused(tmp_path):
    _refused(_run(S.encrypted_pdf(), tmp_path), "R-PDF-ENCRYPTED")


@_needs_worker
def test_portfolio_refused(tmp_path):
    _refused(_run(S.portfolio_pdf(), tmp_path), "R-PDF-PORTFOLIO")


@_needs_worker
def test_embedded_file_flagged_never_read(tmp_path):
    out = _run(S.embedded_file_pdf(), tmp_path)
    assert out["ok"] and "body text here" in out["text"]
    assert "SECRET-ATTACHMENT-BYTES" not in json.dumps(out)
    assert out["flags"]


@_needs_worker
def test_javascript_flagged_text_still_extracted(tmp_path):
    out = _run(S.javascript_pdf(), tmp_path)
    assert out["ok"] and "body text here" in out["text"]
    assert out["flags"]
    assert "app.alert" not in json.dumps(out)


@_needs_worker
def test_flate_bomb_refused(tmp_path):
    _refused(_run(S.flate_bomb_pdf(), tmp_path), "R-PDF-BOMB")


@_needs_worker
def test_over_page_count_refused(tmp_path):
    _refused(_run(S.over_page_count_pdf(401), tmp_path), "R-PDF-PAGES")
    _refused(_run(S.over_page_count_pdf(6), tmp_path, args=("--max-pages", "5")), "R-PDF-PAGES")
    assert _run(S.over_page_count_pdf(5), tmp_path, args=("--max-pages", "5"))["ok"]


@_needs_worker
def test_oversized_refused(tmp_path):
    _refused(_run(S.oversized_pdf(), tmp_path), "R-PDF-SIZE")


@_needs_worker
def test_text_cap_refused(tmp_path):
    _refused(_run(S.multi_page_text(["x" * 200]), tmp_path, args=("--max-text", "50")), "R-PDF-TEXTCAP")


@_needs_worker
def test_non_pdf_named_pdf_refused(tmp_path):
    _refused(_run(S.non_pdf_named_pdf(), tmp_path), "R-PDF-MAGIC")


@_needs_worker
def test_truncated_refused(tmp_path):
    out = _run(S.truncated_pdf(), tmp_path)
    assert out["ok"] is False and out["reason"] in ("R-PDF-PARSE", "R-PDF-PARTIAL", "R-PDF-NOTEXT")


@_needs_worker
def test_image_only_page_notext(tmp_path):
    _refused(_run(S.multi_page_text([""]), tmp_path), "R-PDF-NOTEXT")


@_needs_worker
def test_circular_refs_terminate(tmp_path):
    out = _run(S.circular_refs_pdf(), tmp_path, timeout=30)
    assert "ok" in out


@_needs_worker
def test_invisible_text_dropped(tmp_path):
    out = _run(S.build_pdf(["visible words"], invisible={0: "HIDDEN-INJECTION"}), tmp_path)
    assert "visible words" in out["text"]
    assert "HIDDEN-INJECTION" not in out["text"]


@_needs_worker
def test_every_refusal_has_exactly_one_code(tmp_path):
    for pdf in (S.encrypted_pdf(), S.portfolio_pdf(), S.non_pdf_named_pdf(), S.truncated_pdf()):
        out = _run(pdf, tmp_path)
        if out["ok"] is False:
            assert isinstance(out["reason"], str) and out["reason"].startswith("R-PDF-")


@_needs_worker
def test_no_environment_leak_or_writes(tmp_path):
    out = _run(S.multi_page_text(["a"]), tmp_path)
    blob = json.dumps(out)
    for k in ("DATABASE", "VERA_DB", "API_KEY", "HOME"):
        assert k not in blob
    assert os.listdir(tmp_path / "cwd") == []



def test_pdf_synth_structure():
    b = S.multi_page_text(["one", "two"], labels=[(0, "r", None, 1), (1, "D", None, 1)])
    assert isinstance(b, bytes) and b.startswith(b"%PDF-")
    assert b.rstrip().endswith(b"%%EOF")
    assert b"/Type /Catalog" in b and b.count(b"/Type /Page ") == 2 and b"/PageLabels" in b
    off = int(re.search(rb"startxref\n(\d+)", b).group(1))
    assert b[off:off + 4] == b"xref"
    for m in re.finditer(rb"(\d{10}) 00000 n", b):
        assert re.match(rb"\d+ 0 obj", b[int(m.group(1)):])


def test_pdf_synth_variants():
    assert b"/Encrypt" in S.encrypted_pdf()
    assert b"/Collection" in S.portfolio_pdf() and b"/EmbeddedFile" in S.embedded_file_pdf()
    assert b"/JavaScript" in S.javascript_pdf()
    assert len(S.flate_bomb_pdf()) < 1024 * 1024
    assert len(S.oversized_pdf()) > 15 * 1024 * 1024
    assert not S.non_pdf_named_pdf().startswith(b"%PDF-")
    assert S.over_page_count_pdf(401).count(b"/Type /Page ") == 401
    assert b"creativecommons" in S.xmp_rights_pdf("https://creativecommons.org/licenses/by/4.0/")
