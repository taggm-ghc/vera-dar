"""P3b/T7: the defusedxml rejection branch in vera/jats_text.py and vera/search_providers.py. No DB, no network.

Reachability note: every construct defusedxml forbids by default (entity declarations) is already refused by the
earlier textual guards (<!ENTITY / DOCTYPE internal subset), so no input through the public helpers reaches the
`except defusedxml.DefusedXmlException` branch. The branch is therefore tested by (1) proving with real inputs that
defusedxml itself raises on a forbidden construct and expands nothing, (2) making the parser call raise the real
defusedxml exception (monkeypatching only the `DET.fromstring` call inside each module) and asserting the mapped
error type and the FIXED message, which must not echo untrusted ids carried by the exception.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import defusedxml
import defusedxml.ElementTree as DET
import pytest

from vera import jats_text as jt
from vera import search_providers as sp

CFG = json.loads((Path(__file__).resolve().parent.parent / "config" / "fulltext_ingest.json").read_text())["converter"]
SECRET = "http://untrusted.invalid/evil-sysid"
BOMB = ('<?xml version="1.0"?><!DOCTYPE a [<!ENTITY x "AAAA"><!ENTITY y "&x;&x;&x;&x;">]><a>&y;</a>')
JATS_OK = "<article><front><article-meta><title-group><article-title>T</article-title></title-group></article-meta></front></article>"


def _forbidden(cls):
    if cls is defusedxml.EntitiesForbidden:
        return cls("x", "AAAA", None, SECRET, SECRET, None)
    return cls("x", SECRET, SECRET) if cls is defusedxml.DTDForbidden else cls("x", None, SECRET, SECRET)


EXC = [defusedxml.EntitiesForbidden, defusedxml.DTDForbidden, defusedxml.ExternalReferenceForbidden]


def test_defusedxml_itself_rejects_entities_without_expanding():
    with pytest.raises(defusedxml.EntitiesForbidden):
        DET.fromstring(BOMB)
    with pytest.raises(defusedxml.DTDForbidden):
        DET.fromstring(BOMB, forbid_dtd=True)


def test_prior_text_guards_still_refuse_real_entity_input_first():
    with pytest.raises(jt.ConversionError, match="entity"):
        jt.convert(BOMB.encode(), "jats-xml", CFG)
    p = sp.ArxivProvider({"endpoint": "http://x.invalid", "max_response_bytes": 10**6, "min_interval_s": 0, "query": {}})
    with pytest.raises(sp.ProviderResponseError, match="DOCTYPE/ENTITY"):
        p.parse(SimpleNamespace(text=BOMB), 5)


@pytest.mark.parametrize("cls", EXC)
def test_jats_maps_defusedxml_exception_to_fixed_conversion_error(monkeypatch, cls):
    exc = _forbidden(cls)
    calls = []

    def boom(s, *a, **k):
        calls.append(s)
        raise exc

    monkeypatch.setattr(jt.DET, "fromstring", boom)
    with pytest.raises(jt.ConversionError) as ei:
        jt.convert(JATS_OK.encode(), "jats-xml", CFG)
    assert calls, "the defusedxml parse call was not reached"
    assert str(ei.value) == f"XML refused by defusedxml ({cls.__name__})"
    assert SECRET not in str(ei.value) and ei.value.__cause__ is None and ei.value.__suppress_context__


@pytest.mark.parametrize("cls", EXC)
def test_arxiv_maps_defusedxml_exception_to_fixed_provider_error(monkeypatch, cls):
    exc = _forbidden(cls)
    monkeypatch.setattr(sp.DET, "fromstring", lambda s, *a, **k: (_ for _ in ()).throw(exc))
    p = sp.ArxivProvider({"endpoint": "http://x.invalid", "max_response_bytes": 10**6, "min_interval_s": 0, "query": {}})
    with pytest.raises(sp.ProviderResponseError) as ei:
        p.parse(SimpleNamespace(text='<feed xmlns="http://www.w3.org/2005/Atom"/>'), 5)
    assert str(ei.value) == f"arxiv response refused by defusedxml ({cls.__name__})"
    assert SECRET not in str(ei.value) and ei.value.__cause__ is None and ei.value.__suppress_context__


def test_ordinary_parse_error_keeps_its_own_branch():
    with pytest.raises(jt.ConversionError, match="XML parse error"):
        jt.convert(b"<article><a></article>", "jats-xml", CFG)
