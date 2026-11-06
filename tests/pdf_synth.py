"""Synthetic PDF builder for H20 tests (task D-FIX). Pure Python, no PDF library, no binary fixtures.

Every function returns bytes. Nothing is written to disk.
"""
import zlib

PAGE_FONT = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"


def _esc(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _stream(data, flate=True, extra=b""):
    if flate:
        data = zlib.compress(data)
        extra = b"/Filter /FlateDecode " + extra
    return b"<< " + extra + b"/Length %d >>\nstream\n" % len(data) + data + b"\nendstream"


def _content(text, invisible=None):
    ops = []
    if text:
        ops.append("BT /F1 12 Tf 72 720 Td (%s) Tj ET" % _esc(text))
    if invisible:
        ops.append("BT /F1 12 Tf 3 Tr 72 600 Td (%s) Tj ET" % _esc(invisible))
    return "\n".join(ops).encode("latin-1")


def build_pdf(pages=("Hello",), labels=None, encrypt=False, embedded_file=False, portfolio=False,
              javascript=False, open_action=False, xmp=None, doc_id=None, info_title=None,
              invisible=None, flate_pages=True, extra_objects=None, header=b"%PDF-1.7\n"):
    """pages: list of str (None/'' = empty page). labels: list of (start_index, style, prefix, first) tuples;
    style in D, r, R, a, A. invisible: {page_index: hidden_text}. extra_objects: list of raw object bodies."""
    objs = {}  # num -> bytes
    n = len(pages)
    kids = [4 + i * 2 for i in range(n)]
    nxt = 4 + n * 2
    objs[3] = PAGE_FONT
    for i, t in enumerate(pages):
        inv = (invisible or {}).get(i)
        objs[4 + i * 2] = (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents %d 0 R "
                           b"/Resources << /Font << /F1 3 0 R >> >> >>" % (5 + i * 2))
        objs[5 + i * 2] = _stream(_content(t, inv), flate_pages)
    cat = b"/Type /Catalog /Pages 2 0 R"
    if labels:
        nums = b""
        for start, style, prefix, first in labels:
            d = b"<< "
            if style:
                d += b"/S /" + style.encode() + b" "
            if prefix:
                d += b"/P (" + _esc(prefix).encode() + b") "
            if first is not None:
                d += b"/St %d " % first
            nums += b"%d " % start + d + b">> "
        cat += b" /PageLabels << /Nums [ " + nums + b"] >>"
    if javascript:
        objs[nxt] = b"<< /S /JavaScript /JS (app.alert\\('x'\\);) >>"
        cat += b" /OpenAction %d 0 R /Names << /JavaScript << /Names [(a) %d 0 R] >> >>" % (nxt, nxt)
        nxt += 1
    elif open_action:
        cat += b" /OpenAction [4 0 R /Fit]"
    if embedded_file:
        objs[nxt] = _stream(b"SECRET-ATTACHMENT-BYTES", False, b"/Type /EmbeddedFile ")
        objs[nxt + 1] = b"<< /Type /Filespec /F (a.txt) /EF << /F %d 0 R >> >>" % nxt
        cat += b" /Names << /EmbeddedFiles << /Names [(a.txt) %d 0 R] >> >>" % (nxt + 1)
        nxt += 2
    if portfolio:
        cat += b" /Collection << /Type /Collection /View /D >>"
        if not embedded_file:
            objs[nxt] = _stream(b"SECRET-ATTACHMENT-BYTES", False, b"/Type /EmbeddedFile ")
            objs[nxt + 1] = b"<< /Type /Filespec /F (a.txt) /EF << /F %d 0 R >> >>" % nxt
            cat += b" /Names << /EmbeddedFiles << /Names [(a.txt) %d 0 R] >> >>" % (nxt + 1)
            nxt += 2
    if xmp:
        objs[nxt] = _stream(xmp.encode("utf-8"), False, b"/Type /Metadata /Subtype /XML ")
        cat += b" /Metadata %d 0 R" % nxt
        nxt += 1
    objs[1] = b"<< " + cat + b" >>"
    objs[2] = b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % k for k in kids) + b"] /Count %d >>" % n
    trailer_extra = b""
    if info_title:
        objs[nxt] = b"<< /Title (" + _esc(info_title).encode("latin-1") + b") >>"
        trailer_extra += b" /Info %d 0 R" % nxt
        nxt += 1
    for body in extra_objects or []:
        objs[nxt] = body
        nxt += 1
    if encrypt:
        objs[nxt] = (b"<< /Filter /Standard /V 1 /R 2 /O (" + b"0" * 32 + b") /U (" + b"0" * 32 +
                     b") /P -4 >>")
        trailer_extra += b" /Encrypt %d 0 R" % nxt
        nxt += 1
    if doc_id:
        h = doc_id.encode().hex().encode()
        trailer_extra += b" /ID [<" + h + b"> <" + h + b">]"
    out = bytearray(header)
    offs = {}
    for num in sorted(objs):
        offs[num] = len(out)
        out += b"%d 0 obj\n" % num + objs[num] + b"\nendobj\n"
    size = max(objs) + 1
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for num in range(1, size):
        out += b"%010d 00000 n \n" % offs[num]
    out += b"trailer\n<< /Size %d /Root 1 0 R%s >>\nstartxref\n%d\n%%%%EOF\n" % (size, trailer_extra, xref)
    return bytes(out)


def xmp_packet(rights_url=None, rights_text=None, document_id=None, title=None):
    parts = []
    if rights_url:
        parts.append('<xmpRights:WebStatement>%s</xmpRights:WebStatement>' % rights_url)
        parts.append('<cc:license rdf:resource="%s"/>' % rights_url)
    if rights_text:
        parts.append('<dc:rights><rdf:Alt><rdf:li xml:lang="x-default">%s</rdf:li></rdf:Alt></dc:rights>' % rights_text)
    if document_id:
        parts.append('<xmpMM:DocumentID>%s</xmpMM:DocumentID>' % document_id)
    if title:
        parts.append('<dc:title><rdf:Alt><rdf:li xml:lang="x-default">%s</rdf:li></rdf:Alt></dc:title>' % title)
    return ('<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?><x:xmpmeta xmlns:x="adobe:ns:meta/">'
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:xmpRights="http://ns.adobe.com/xap/1.0/rights/" xmlns:cc="http://creativecommons.org/ns#" '
            'xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/">' + "".join(parts) +
            '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>')


def multi_page_text(texts, labels=None):
    return build_pdf(pages=list(texts), labels=labels)


def encrypted_pdf():
    return build_pdf(pages=["secret"], encrypt=True)


def portfolio_pdf():
    return build_pdf(pages=["cover"], portfolio=True, embedded_file=True)


def embedded_file_pdf():
    return build_pdf(pages=["body text here"], embedded_file=True)


def javascript_pdf():
    return build_pdf(pages=["body text here"], javascript=True)


def flate_bomb_pdf(inflated_bytes=96 * 1024 * 1024):
    """Small file whose page content stream inflates to inflated_bytes of whitespace (default above the 64 MiB cap)."""
    co = zlib.compressobj(9)
    chunk = b" " * (1024 * 1024)
    data = b"BT /F1 12 Tf 72 720 Td (x) Tj ET\n"
    comp = co.compress(data)
    for _ in range(inflated_bytes // len(chunk)):
        comp += co.compress(chunk)
    comp += co.flush()
    objs_stream = b"<< /Filter /FlateDecode /Length %d >>\nstream\n" % len(comp) + comp + b"\nendstream"
    return _rebuild_with_content(objs_stream)


def _rebuild_with_content(stream_obj):
    out = bytearray(b"%PDF-1.7\n")
    objs = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        3: PAGE_FONT,
        4: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R /Resources << /Font << /F1 3 0 R >> >> >>",
        5: stream_obj,
    }
    offs = {}
    for k in sorted(objs):
        offs[k] = len(out)
        out += b"%d 0 obj\n" % k + objs[k] + b"\nendobj\n"
    x = len(out)
    out += b"xref\n0 6\n0000000000 65535 f \n"
    for k in range(1, 6):
        out += b"%010d 00000 n \n" % offs[k]
    out += b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % x
    return bytes(out)


def over_page_count_pdf(pages=401):
    return build_pdf(pages=["p%d" % i for i in range(pages)])


def oversized_pdf(size=15 * 1024 * 1024 + 1):
    base = build_pdf(pages=["small"])
    pad = size - len(base) - 1
    return base + b"%" + b"A" * max(pad, 0) if pad > 0 else base


def non_pdf_named_pdf():
    """HTML 'request a copy' page that a host serves under a .pdf name."""
    return b"<!DOCTYPE html><html><body>Request a copy of this item</body></html>"


def truncated_pdf():
    full = build_pdf(pages=["one", "two", "three"])
    return full[: len(full) // 2]


def circular_refs_pdf():
    return build_pdf(pages=["loop"], extra_objects=[b"<< /Next 99 0 R >>"]).replace(b"/Count 1", b"/Count 1 /Parent 2 0 R")


def xmp_rights_pdf(rights_url, document_id=None, title=None):
    return build_pdf(pages=["body text here"], xmp=xmp_packet(rights_url=rights_url, document_id=document_id, title=title))
