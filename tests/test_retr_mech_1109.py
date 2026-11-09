"""RETR-MECH-1109: BM25 light stemming and generic-filler dropping in the candidate query (offline)."""
import copy
import json
from pathlib import Path

import pytest

from vera.dar_corpus import DarCorpus, light_stem
from vera.fulldoc_corpus import _routing_cfg

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("a,b", [("agents", "agent"), ("agent", "agent"), ("forecasting", "forecast"),
                                 ("forecasted", "forecast"), ("policies", "policy"), ("classes", "class"),
                                 ("gans", "gan"), ("bus", "bus"), ("analysis", "analysis"), ("class", "class"),
                                 ("red", "red"), ("ai", "ai"), ("2024s", "2024s")])
def test_light_stem(a, b):
    assert light_stem(a) == b


def test_stem_conflates_plural_and_ing():
    assert light_stem("agents") == light_stem("agent") and light_stem("models") == light_stem("modeling") == "model"


def _corpus(on):
    cfg = copy.deepcopy(_routing_cfg())
    cfg["corpus"]["bm25_stem"] = on
    cfg["corpus"]["query_drop_generic"] = on
    return DarCorpus(cfg)


def _doc(key, text):
    return {"source_key": key, "url": key, "doi": None, "title": key, "published": "2026-01-01",
            "source_type": "preprint", "retrieved_at": "2026-11-09", "text": text,
            "stored_text_kind": "full_text", "admission_passed": True, "db_source_id": None,
            "db_canonical_source_id": None}


def test_stemmed_retrieval_matches_plural():
    texts = {"d1": "A planner agent decomposes goals into steps and delegates work to worker nodes. " * 3,
             "d2": "Rivers carry sediment toward the delta across many seasons of the year. " * 3}
    for on, expect in ((True, "d1"), (False, None)):
        c = _corpus(on)
        for k, t in texts.items():
            c.ingest_document(_doc(k, t))
        got = [x["source_key"] for x in c.retrieve("agents", 5)["candidates"]]
        assert (expect in got) if expect else ("d1" in got) is False


def test_query_drops_generic_idiom():
    c = _corpus(True)
    assert "facto" not in c._query_terms("What is the de facto standard for agents?")
    assert "facto" in _corpus(False)._query_terms("What is the de facto standard for agents?")
