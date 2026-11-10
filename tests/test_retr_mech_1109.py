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
                                 ("gans", "gans"), ("news", "news"), ("lens", "lens"), ("united", "united"), ("bus", "bus"), ("analysis", "analysis"), ("class", "class"),
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


def test_light_stem_keeps_unrelated_words_distinct():
    assert light_stem("news") != light_stem("new") and light_stem("united") != light_stem("unit")
    assert light_stem("lens") != light_stem("len") and light_stem("gans") != light_stem("gan")
    assert len(light_stem("tested")) >= 4 and light_stem("rated") == "rated"  # never a stem shorter than 4


def _kept(q):
    from vera import fulldoc_admission as fa
    from vera.fulldoc_corpus import load_config
    cfg = load_config()
    icfg = cfg["admission"]["relevance"]
    return " ".join(fa.retrieval_kept_words(q, icfg))


def test_retrieval_drop_matches_whole_words_not_prefixes():
    assert "commodity" in _kept("What drives commodity prices?")
    assert "imports" in _kept("How do tariffs affect imports?")
    assert "Commonwealth" in _kept("What is the British Commonwealth?")
    assert "majority" in _kept("How did the majority leader vote?")
    out = _kept("What are the de facto status quo best practices?")
    assert "facto" not in out and "best" not in out and "practices" not in out
    assert "important" not in _kept("What is important about tariffs?").split()


def test_filler_inside_proper_name_or_quotes_is_kept():  # FILLER-PROPER-1109
    assert "Major League Baseball" in _kept("Who runs Major League Baseball?")
    assert "Major" in _kept("Major League Baseball history").split()
    assert "Key" in _kept("When did the Key Bridge collapse?").split()
    assert "Going" in _kept("What is the Going Concern doctrine?").split()
    assert "Best" in _kept("Where is Best Buy headquartered?").split()
    for q, w in (("Is Bank of America open?", "Bank"), ("Who is Will Smith?", "Will"),
                 ("What is May Day?", "May")):
        assert w in _kept(q).replace("?", "").split()
    assert '"key signature"' in _kept('What does "key signature" mean?')
    out = _kept("Explain Photosynthesis in the main types of plants").split()
    assert "Explain" not in out and "main" not in out and "types" not in out and "Photosynthesis" in out
    assert "best" not in _kept("What are the best practices for pruning?").split()
