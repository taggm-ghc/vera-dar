"""WPQ-1111, OFFLINE: open/closed compound handling (generic rule, no word list) and reference-type demotion.
Record shapes follow vera.search_providers' OpenAlex output (doi, url, title, snippet, source_type). No network, no DB.
The "reference-entry" type string is a PLACEHOLDER for the demotion list under test: it is injected through the config
reader because no recorded OpenAlex work with that type exists in the repo yet (PASS C5 open)."""
import json
from pathlib import Path

import pytest

from tests import acq_support as A
from vera import acq_discovered, auto_acquire, query_variants_acq as qv, search_providers as sp, search_query as sq

ROOT = Path(__file__).resolve().parent.parent
QT = json.loads((ROOT / "config" / "vera_eval_run.json").read_text())["search"]["query_terms"]
PROFILE = A.cfg()["search"]["openalex_profile"]
DEMOTE = frozenset({"reference-entry"})


def _oc(monkeypatch, n=1, demote=frozenset()):
    monkeypatch.setattr(sq, "_qt_oc", lambda: (n, frozenset(demote)))
    monkeypatch.setattr(sq, "open_compound_max_pairs", lambda: n)


@pytest.fixture(autouse=True)
def _pairs_on(monkeypatch):
    """The shipped default is max_pairs = 0 (rule off, owner decision (b)); these tests turn it on explicitly."""
    monkeypatch.setattr(sq, "open_compound_max_pairs", lambda: 1)


def _all(q):
    return {"ask": sp.build_openalex_search(q, QT)[0], "relaxed": sp.build_openalex_search(q, QT, relaxed=True)[0],
            "acq": sp.build_openalex_search(q, QT, profile=PROFILE)[0], "arxiv": sp.build_arxiv_query(q, {}, QT),
            "variants": qv.variants(q, terms_cfg=QT)}


SS = '("super strings" OR superstrings)'


# --- 1, 24: positive, every query shape, no bare components
@pytest.mark.parametrize("q", ["What are super strings?", "what is super strings"])
def test_super_strings_every_query_shape_has_the_pair_unit(q):
    r = _all(q)
    assert r["ask"] == SS and r["relaxed"] == SS and r["acq"] == SS
    assert 'all:"super strings"' in r["arxiv"] and "all:superstrings" in r["arxiv"] and "all:super " not in r["arxiv"]
    assert SS not in r["variants"] and not any(v in ("super", "strings") for v in r["variants"])
    assert not any(" OR " in v and ("super OR" in v or "OR strings" in v) for v in r["variants"])


def test_and_terms_treats_the_pair_as_one_term():  # 24
    qt = sq.build_query_terms("what are super strings", QT)
    assert qt.pairs == (("super", "strings"),)
    assert sq.and_terms(qt, 4) == [SS]
    assert sq.and_terms(qt, 4, exclude=("super",)) == ["strings"]  # an excluded word dissolves the pair


# --- 2 generality without any config entry
def test_generality_probes_need_no_config_entry(monkeypatch):
    assert sp.build_openalex_search("what are nano particles?", QT)[0] == '("nano particles" OR nanoparticles)'
    assert sp.build_openalex_search("what is neuro science", QT)[0] == '("neuro science" OR neuroscience)'
    assert "compound_concepts" in json.loads((ROOT / "config" / "query_terms.json").read_text())
    assert "superstring" not in (ROOT / "config" / "query_terms.json").read_text().replace("_wpq1111_note", "")


# --- 3 relevance side
def test_pair_concept_matches_open_hyphenated_and_closed_titles():
    kc = sq.key_concepts("What are super strings?", QT, pairs=True)
    assert kc[0] == ("super-strings", "super", "strings") and kc[1]["super-strings"] == 2
    t = sq.relevance_tiers(kc, ["Superstring compactifications and their phenomenology", "Super-string theory and the Standard Model",
                                "super strings in AdS", "Strings", "super, n.¹⁰", "Superconductivity basics",
                                "Climate of the Alps in summer and winter"])
    assert t[:3] == [0, 0, 0]      # pair match = tier 0
    assert t[3] == 1 and t[4] == 1  # component-only: tier 1 ("Strings" alone is not the pair; the dictionary entry is weak)
    assert t[5] == 1               # 17: C6 pinned: prefix stem still lets "super" match "Superconductivity" (known error), but only as a weak match
    assert t[6] == 2


def test_component_only_record_is_weak_not_dropped_22():
    kc = sq.key_concepts("climate change effects", QT, pairs=True)
    t = sq.relevance_tiers(kc, ["Climate and weather of the Alps in summer", "Climate change effects on crop yields", "Quantum dots in photonic crystal devices"])
    assert t == [1, 0, 2]


def test_without_pairs_flag_key_concepts_is_unchanged():
    assert sq.key_concepts("What are super strings?", QT) == (("super", "strings"), {"super": 1, "strings": 1})


# --- 4, 5, 6: lookalikes and no-pair questions
@pytest.mark.parametrize("q,phrase", [("post office strings", "post office"), ("data base design", "data base"),
                                      ("super market prices", "super market"), ("life style changes", "life style"),
                                      ("super string quartet", "super string")])
def test_lookalikes_keep_the_open_phrase_and_the_component_words(q, phrase):
    a, w1, w2 = sp.build_openalex_search(q, QT)[0], *phrase.split()
    assert f'"{phrase}"' in a and f"{w1}{w2}" in a
    assert f"{w1} OR " not in a.replace(f'"{phrase}" OR ', "")  # the open phrase is kept, never split into a bare ORed word
    kc = sq.key_concepts(q, QT, pairs=True)
    assert f"{w1}-{w2}" in kc[0] and w1 in kc[0] and w2 in kc[0]
    assert sq.relevance_tiers(kc, [f"The {phrase} in the literature of today"])[0] == 0  # open phrase still tier 0


@pytest.mark.parametrize("q", ["super bowl sponsorship", "guitar super slinky strings"])
def test_non_adjacent_words_are_never_joined_to_superstring(q):
    r = _all(q)
    assert not any("superstring" in str(v) for v in r.values())


@pytest.mark.parametrize("q", ["what are quarks?", "M-theory", "superstrings", "multi-agent orchestration", "midterm grades"])
def test_single_word_and_configured_compound_questions_are_byte_identical(q, monkeypatch):
    on = _all(q)
    _oc(monkeypatch, 0)
    assert _all(q) == on
    assert sq.key_concepts(q, QT, pairs=True) == sq.key_concepts(q, QT)


# --- 8, 30: rollback knob, non vacuous
def test_max_pairs_zero_restores_todays_queries_and_the_default_differs(monkeypatch):
    on = _all("What are super strings?")
    _oc(monkeypatch, 0)
    off = _all("What are super strings?")
    assert on != off
    assert off["ask"] == "(super OR strings)" and off["relaxed"] == "(super OR strings)" and off["acq"] == "super AND strings"
    assert off["arxiv"] == "(all:super OR all:strings)"
    assert sq.key_concepts("What are super strings?", QT, pairs=True)[0] == ("super", "strings")
    assert sq.build_query_terms("What are super strings?", QT).pairs == ()


# --- 9 bound
def test_three_plus_content_words_expand_at_most_max_pairs(monkeypatch):
    q = "super strings dark energy theory"
    for n, want in ((1, 1), (2, 2), (0, 0)):
        _oc(monkeypatch, n)
        assert len(sq.build_query_terms(q, QT).pairs) == want
    _oc(monkeypatch, 1)
    assert sp.build_openalex_search(q, QT)[0] == '(("super strings" OR superstrings) OR dark OR energy OR theory)'


# --- 20, 21 entity and multi-part
def test_entity_central_tiers_keep_todays_order_for_the_entity():
    kc = sq.key_concepts("what did Einstein say about super strings", QT, central=("Einstein",), pairs=True)
    assert kc[1]["einstein"] == max(kc[1].values())
    t = sq.relevance_tiers(kc, ["Einstein and the quantum theory of light", "Superstring compactifications and their phenomenology", "Cooking pasta at home for families"])
    assert t[0] == 0 and t[2] == 2


def test_part_tiers_for_parts_without_a_pair_are_unchanged(monkeypatch):
    texts = ["Quarks and hadrons in the standard model", "Weather of the Alps and cooking"]
    q = "what are quarks? and what is cooking?"
    parts = [("what are quarks?", ()), ("what is cooking?", ())]
    assert sq.build_query_terms(q, QT).pairs == ()  # the question itself has no eligible anchor pair here
    on = sq.part_tiers(q, texts, [0, 1], parts, cfg=QT)
    _oc(monkeypatch, 0)  # base behaviour (rule off)
    base = sq.part_tiers(q, texts, [0, 1], parts, cfg=QT)
    assert on == base and 2 not in base[:1]
    # a part WITH a pair differs from base only by the pair weight (non-vacuous: pairs on vs off do differ)
    t2 = ["Superstring compactifications and their phenomenology", "Climate of the Alps in summer and winter"]
    p2 = [("What are super strings?", ()), ("what is cooking?", ())]
    _oc(monkeypatch, 1)
    with_pair = sq.part_tiers("What are super strings? and what is cooking?", t2 + ["Strings"], [0, 1, 0], p2, cfg=QT)
    _oc(monkeypatch, 0)
    no_pair = sq.part_tiers("What are super strings? and what is cooking?", t2 + ["Strings"], [0, 1, 0], p2, cfg=QT)
    assert with_pair != no_pair


def test_variants_round_never_emits_the_pair_components_alone():  # 24, 16 (variants round)
    v = qv.variants("post office strings", terms_cfg=QT)
    assert '("post office" OR postoffice)' not in v[:0] and any('("post office" OR postoffice)' in x for x in v)
    assert not any(x in ("post", "office") for x in v)
    assert not any(" post " in f" {x} " and '"post office"' not in x and "post office" not in x for x in v)


# --- 23 validation at load
@pytest.mark.parametrize("bad", ['"x"', "-1", "true", "1.5", "null"])
def test_invalid_max_pairs_turns_the_rule_off_logs_once_and_does_not_raise(bad, tmp_path, monkeypatch, caplog):
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    (cfgdir / "query_terms.json").write_text('{"open_compound": {"max_pairs": %s}, "relevance": {"demote_record_types": "oops"}}' % bad)
    monkeypatch.setattr(sq, "_QT_PATH", cfgdir / "query_terms.json")
    sq._qt_oc.cache_clear()
    try:
        with caplog.at_level("WARNING"):
            assert sq._qt_oc() == (0, frozenset())
        assert sum("max_pairs invalid" in r.message for r in caplog.records) == 1
        assert sq.record_demoted({"source_type": "oops"}) is False
    finally:
        monkeypatch.undo()
        sq._qt_oc.cache_clear()


def test_shipped_config_values(monkeypatch):
    sq._qt_oc.cache_clear()
    assert sq._qt_oc() == (0, frozenset())  # shipped: rule off, demotion list empty
    assert sq.build_query_terms("What are super strings?", QT).pairs == (("super", "strings"),)  # autouse fixture: on
    monkeypatch.undo()
    assert sq.build_query_terms("What are super strings?", QT).pairs == ()  # real shipped config: today's behaviour


# --- 18 known gaps unchanged: non-boolean providers keep today's text
def test_non_boolean_providers_keep_todays_text_for_the_pair():
    qt = sq.build_query_terms("What are super strings?", QT)
    assert list(qt.core) + list(qt.extra) == ["super", "strings"]  # DDG words: "super strings" (a phrase by adjacency)
    assert sp.build_arxiv_query("What are super strings?", {}, QT) .count("super strings") == 1


# --- demotion (P5-1, P6-1): ordering only
def _rec(i, typ, title):
    d = f"10.9999/w{i}"
    return {"provider": "openalex", "doi": d, "url": f"https://doi.org/{d}", "title": title, "snippet": "",
            "source_type": typ, "openalex_id": f"https://openalex.org/W{i}"}


Q = "What are super strings?"
OED = [_rec(i, "reference-entry", f"superstring, n. definition of the term {i}") for i in range(3)]
JHEP = [_rec(10 + i, "article", f"Superstring compactifications and moduli {i}") for i in range(3)]


def test_record_demoted_reads_the_one_list(monkeypatch):
    assert sq.record_demoted(OED[0]) is False  # shipped list is empty until the type is observed (C5)
    _oc(monkeypatch, 1, DEMOTE)
    assert sq.record_demoted(OED[0]) and not sq.record_demoted(JHEP[0]) and not sq.record_demoted({})
    assert sq.record_demoted({"type": "Reference-Entry"})  # `type` fallback, case-insensitive


def test_acq_discovered_sorts_reference_entries_after_research(monkeypatch):
    _oc(monkeypatch, 1, DEMOTE)
    cfg = A.cfg()
    cfg["inline"]["use_discovered"] = True
    cands, _h = acq_discovered.discovered_candidates(OED + JHEP, cfg, question=Q)
    got = [c["doi"] for c in cands]
    assert got[:3] == [r["doi"] for r in JHEP] and set(got[3:]) == {r["doi"] for r in OED}


def test_acq_discovered_slots_keep_all_research_records(monkeypatch):  # 14
    _oc(monkeypatch, 1, DEMOTE)
    cfg = A.cfg()
    n = acq_discovered.discovered_max(cfg)
    res = [_rec(100 + i, "article", f"Superstring theory part {i}") for i in range(n)]
    cands, held = acq_discovered.discovered_candidates(OED + res, cfg, question=Q)
    assert {c["doi"] for c in cands} == {r["doi"] for r in res}


def test_rank_candidates_sorts_reference_entries_after_research(monkeypatch):
    _oc(monkeypatch, 1, DEMOTE)
    acq = A.cfg()
    ranked = auto_acquire.rank_candidates(Q, OED + JHEP, acq, set(), auto_acquire._fd_cfg(), lambda c: None, drops={})
    dois = [r[1] for r in ranked]
    assert dois[:3] == [r["doi"] for r in JHEP], dois


def test_non_boolean_provider_strings_are_identical_with_pairs_on_and_off(monkeypatch):  # fix 4 (18): recorded known gap
    """DDG, INSPIRE and the Wikipedia title adapter do not use the pair unit. DDG words come from build_query_terms and
    are unchanged on/off; INSPIRE and Wikipedia build from wiki_ref_portal.build_query (independent of pairs), so the
    Wikipedia redirect gap (/wiki/Super_strings, not Superstring theory) is unchanged. Gap recorded, not fixed."""
    from vera import portal_inspire, reference_portal as rp
    from tests.test_reference_portal_1108 import shipped
    ddg_cfg = json.loads((ROOT / "config" / "vera_eval_run.json").read_text())["search"]["duckduckgo_lite"]
    qq = "What are super strings?"

    def snap():
        return (sp.build_ddg_queries(qq, QT, ddg_cfg), portal_inspire._query(qq, {"max_query_terms": 8}),
                rp.wikipedia_title_urls(qq, shipped()))
    _oc(monkeypatch, 1)
    on = snap()
    _oc(monkeypatch, 0)
    assert snap() == on
    assert "super strings" in on[1] and on[2][0].endswith("/wiki/Super_strings")


def test_empty_demotion_list_gives_exactly_the_base_order(monkeypatch):  # fix 5 (27)
    from vera import acq_discovered as ad
    cfg = A.cfg()
    cfg["inline"]["use_discovered"] = True
    pool = OED + JHEP
    _oc(monkeypatch, 0, frozenset())  # base: rule off, no demotion
    base_rank = [r[1] for r in auto_acquire.rank_candidates(Q, pool, cfg, set(), auto_acquire._fd_cfg(), lambda c: None, drops={})]
    base_disc = [c["doi"] for c in ad.discovered_candidates(pool, cfg, question=Q)[0]]
    _oc(monkeypatch, 0, frozenset())  # explicit empty list, same code path
    assert [r[1] for r in auto_acquire.rank_candidates(Q, pool, cfg, set(), auto_acquire._fd_cfg(), lambda c: None, drops={})] == base_rank
    assert [c["doi"] for c in ad.discovered_candidates(pool, cfg, question=Q)[0]] == base_disc
    # the sort key adds nothing when the list is empty: same order as the tier-only order (stable provider order)
    tiers = sq.relevance_tiers(sq.key_concepts(Q, pairs=True), [f"{r['title']} {r['snippet']}" for r in pool])
    assert base_disc == [r["doi"] for _t, r in sorted(zip(tiers, pool), key=lambda x: x[0]) if _t != 2]
    _oc(monkeypatch, 0, DEMOTE)  # non-vacuous: a non-empty list does change the order
    assert [c["doi"] for c in ad.discovered_candidates(pool, cfg, question=Q)[0]] != base_disc


def test_nonempty_demotion_list_changes_rank_candidates_order(monkeypatch):
    _oc(monkeypatch, 1, frozenset())
    acq = A.cfg()
    a = [r[1] for r in auto_acquire.rank_candidates(Q, OED + JHEP, acq, set(), auto_acquire._fd_cfg(), lambda c: None, drops={})]
    _oc(monkeypatch, 1, DEMOTE)
    b = [r[1] for r in auto_acquire.rank_candidates(Q, OED + JHEP, acq, set(), auto_acquire._fd_cfg(), lambda c: None, drops={})]
    assert a != b


def test_both_callers_read_one_demote_object(monkeypatch):  # 31
    import inspect
    for mod in (acq_discovered, auto_acquire):
        assert "search_query.record_demoted" in inspect.getsource(mod)
    seen = []
    monkeypatch.setattr(sq, "demote_record_types", lambda: seen.append(1) or DEMOTE)
    sq.record_demoted(OED[0])
    assert seen == [1] and sq.record_demoted(OED[0])


def test_role_and_portal_lists_are_untouched():  # 28
    hr = json.loads((ROOT / "config" / "host_roles.json").read_text())
    assert hr["record_type_portal"] == ["preprint", "posted-content"]
