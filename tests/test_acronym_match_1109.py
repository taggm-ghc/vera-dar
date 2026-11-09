"""ANSWER-SCOPE-1109 root cause: question acronyms match spelled-out forms (and back) in admission, mechanically.
SYNTHETIC passages shaped like the live case ("What are components of RAG?"); no network, DB or env."""
import copy
import json
from pathlib import Path

from vera import acronym_match as am
from vera import fulldoc_admission as fa
from vera import term_expansion as te

ROOT = Path(__file__).resolve().parent.parent
FD = json.loads((ROOT / "config" / "fulldoc_answer.json").read_text())
STOP = frozenset(FD["admission"]["relevance"]["stopwords"])
Q = "What are components of RAG?"
SPEC = ("Our VLM-guided RAG system has components for image query optimization and a structured knowledge tree "
        "for robot action prediction.")
GEN_HY = "The components of retrieval-augmented generation are a retriever, a generator and an index of documents."
GEN_SP = "Retrieval augmented generation systems have two main components: a dense retriever and a seq2seq generator."
DEFN = "Retrieval-Augmented Generation (RAG) grounds a language model in retrieved passages."
NOISE = "Components of a rag rug: cotton strips, a backing and a ripping awl for rag weaving."
COINCIDENCE = "Components were reviewed against average gain over the research agenda generally."


def doc(i, title):
    return {"document_id": i, "canonical_source_id": f"https://x/{i}", "title": title, "url": f"https://x/{i}",
            "doi": None, "published": "2020", "authors": ["A"], "licence": "CC-BY",
            "licence_url": "https://creativecommons.org/licenses/by/4.0/", "licence_decision": "allow",
            "attribution": "a", "text_sha256": "h", "chars": 100, "evidence_ranges": [[0, 10 ** 9]]}


class Corpus:
    status = "ok"

    def __init__(self, docs):
        self.acquired = [doc(i, t) for i, (t, _) in docs.items()]
        self.full, self.cands, self.queries = {}, [], []
        for i, (_t, spans) in docs.items():
            pos, txt = 0, ""
            for j, s in enumerate(spans):
                self.cands.append({"span_id": f"{i}:{j}", "document_id": i, "start": pos, "end": pos + len(s),
                                   "text": s, "score": 1.0 - 0.01 * j})
                txt += s + " "
                pos += len(s) + 1
            self.full[i] = txt

    def text(self, i):
        return self.full[i]

    def retrieve(self, q, k):
        self.queries.append(q)
        return list(self.cands)

    def inspection(self):
        return {}


def cfg(on=True, signal=None):
    c = copy.deepcopy(FD)
    c["admission"]["acronym_match"]["enabled"] = on
    if signal:
        c["admission"]["signal"] = signal
    return c


def docs_of(b):
    return {a["document_id"] for a in b.admitted}


# ------------------------------------------------------------------------------------------------- unit: detection
def test_question_acronyms_shape_only():
    assert am.question_acronyms(Q) == ["rag"]
    assert am.question_acronyms("How do LLMs hallucinate?") == ["llm"]
    assert am.question_acronyms("WHAT ARE COMPONENTS OF RAG?") == []  # shouting is not an acronym signal
    assert am.question_acronyms("What is a rag rug?") == []


def test_initial_runs_contiguous_hyphen_case_tolerant():
    assert am.initial_runs("x Retrieval-Augmented Generation y", "rag", STOP)
    assert am.initial_runs("retrieval augmented generation", "rag", STOP)
    assert am.initial_runs("large language models", "llm", STOP)  # plural last word
    assert not am.initial_runs("retrieval, augmented generation", "rag", STOP)  # punctuation breaks contiguity
    assert not am.initial_runs("retrieval of augmented generation", "rag", STOP)  # skipped word breaks it
    assert not am.initial_runs("results are good", "rag", STOP)  # stopword initials never spell
    assert not am.initial_runs("an ai", "ai", STOP)


def test_definitions_learned_from_corpus_only_when_initials_match():
    assert am.definitions([DEFN], "rag", STOP) == ["retrieval augmented generation"]
    assert am.definitions(["deoxyribonucleic acid (DNA) is long"], "dna", STOP) == []  # initials do not spell it
    assert am.definitions(["Average income (AI) rose."], "ai", STOP) == ["average income"]


# -------------------------------------------------------------------------------------------- admission end to end
def test_rag_question_admits_general_docs_that_spell_it_out():
    docs = {"spec": ("VLM-guided RAG for robot action prediction", [SPEC, DEFN]),
            "g1": ("Retrieval-augmented generation", [GEN_HY]),
            "g2": ("A survey", [GEN_SP])}
    for signal in ("legacy", "idf_weighted"):
        off = fa.admit(Q, Corpus(docs), cfg(False, signal))
        on = fa.admit(Q, Corpus(docs), cfg(True, signal))
        assert "g1" not in docs_of(off) and "g2" not in docs_of(off), signal  # the live failure, reproduced
        assert {"spec", "g1", "g2"} <= docs_of(on), signal


def test_no_corpus_definition_no_expansion_by_default():
    docs = {"spec": ("t", [SPEC]), "g1": ("t", [GEN_HY])}
    assert "g1" not in docs_of(fa.admit(Q, Corpus(docs), cfg(True)))
    c = cfg(True)
    c["admission"]["acronym_match"]["bare_initials"] = True  # opt-in bare initials (>= min_scan_len letters)
    assert "g1" in docs_of(fa.admit(Q, Corpus(docs), c))


def test_ambiguous_acronym_is_not_expanded():
    q = "What do raised MMA levels mean?"
    med = "Methylmalonic acid (MMA) levels are raised in deficiency."
    fight = ["Mixed martial arts (MMA) bouts last three rounds.", "Raised mixed martial arts levels mean more bouts."]
    docs = {"m": ("t", [med]), "f": ("t", fight)}
    on, off = fa.admit(q, Corpus(docs), cfg(True)), fa.admit(q, Corpus(docs), cfg(False))
    assert [a["span_id"] for a in on.admitted] == [a["span_id"] for a in off.admitted]  # nothing expanded
    assert am.groups(q, [med] + fight, {"enabled": True}, STOP) == ()


def test_lowercase_word_rag_never_matches_the_acronym():
    docs = {"rug": ("Rag rugs", [NOISE]), "g1": ("RAG", [GEN_HY, DEFN])}
    on = fa.admit(Q, Corpus(docs), cfg(True))
    assert "rug" not in docs_of(on) and "g1" in docs_of(on)


def test_coincidental_initials_never_match_without_bare_initials():
    # "research agenda generally" spells r-a-g contiguously; it matches only as a corpus-defined long form (never
    # here) or with the opt-in bare_initials switch.
    runs = am.initial_runs(COINCIDENCE, "rag", STOP)
    assert [COINCIDENCE[a:b] for a, b in runs] == ["research agenda generally"]
    docs = {"c": ("t", [COINCIDENCE]), "g1": ("t", [DEFN])}
    assert "c" not in docs_of(fa.admit(Q, Corpus(docs), cfg(True)))


def test_two_letter_acronym_needs_a_corpus_definition():
    c = cfg(True)
    q = "What are risks of AI?"
    plain = {"d1": ("x", ["Risks of artificial intelligence include bias and misuse in hiring."])}
    assert docs_of(fa.admit(q, Corpus(plain), c)) == set()  # no definition: 2-letter initials do not match
    defined = {"d1": ("x", ["Artificial intelligence (AI) systems carry risks.",
                            "Risks of artificial intelligence include bias and misuse in hiring."])}
    assert "d1" in docs_of(fa.admit(q, Corpus(defined), c))


def test_spelled_question_matches_acronym_only_with_corpus_definition():
    q = "What are components of retrieval augmented generation?"
    alias = "RAG components include an index and a ranker."
    undefined = {"d1": ("x", [alias])}
    assert docs_of(fa.admit(q, Corpus(undefined), cfg(True))) == set()
    defined = {"d1": ("x", [DEFN, alias])}
    assert "d1:1" in {a["span_id"] for a in fa.admit(q, Corpus(defined), cfg(True)).admitted}
    assert "d1:1" not in {a["span_id"] for a in fa.admit(q, Corpus(defined), cfg(False)).admitted}
    lms = {"d2": ("x", ["Learning management system (LMS) components: courses and grades."])}
    assert docs_of(fa.admit("What are components of language model safety?", Corpus(lms), cfg(True))) == set()


def test_retrieval_query_gets_learned_forms_only_when_on():
    docs = {"g1": ("x", [DEFN, GEN_HY])}
    c_on = Corpus(docs)
    fa.admit(Q, c_on, cfg(True))
    assert "retrieval augmented generation" in c_on.queries[0]
    c_off = Corpus(docs)
    fa.admit(Q, c_off, cfg(False))
    assert c_off.queries[0] == Q


def test_off_is_identical_to_absent_key():
    docs = {"spec": ("t", [SPEC]), "g1": ("t", [GEN_HY])}
    absent = copy.deepcopy(FD)
    absent["admission"].pop("acronym_match")
    b1, b2 = fa.admit(Q, Corpus(docs), cfg(False)), fa.admit(Q, Corpus(docs), absent)
    assert [a["span_id"] for a in b1.admitted] == [a["span_id"] for a in b2.admitted]
    assert b1.trace == b2.trace


def test_seed_groups_unchanged_and_token_injection_defused():
    g = te.Group("acrrag", (), (), initials="rag", upper="RAG", own="rag", stop=STOP)
    assert g.spelled == "rag"
    s = te.canonicalize("xgacrrag and RAG and retrieval-augmented generation", (g,))
    assert s.split().count("xgacrrag") == 2 and "xg_acrrag" in s


# -------------------------------------------------------------------------------------- verifier follow-ups (11-09)
RAG2 = "Recurrent Attention Gate (RAG) layers gate recurrent features."


def test_long_form_question_never_equates_two_senses():
    q = "What are components of retrieval augmented generation?"
    gs = am.groups(q, [DEFN, RAG2], {"enabled": True}, STOP)
    assert gs == ()  # two different long forms for RAG: no group at all, in either direction
    assert am.groups(Q, [DEFN, RAG2], {"enabled": True}, STOP) == ()
    gate = {"g": ("t", [RAG2, "Components of recurrent attention gate layers include sigmoid units."])}
    gen = {"d": ("t", [DEFN])}
    b = fa.admit(q, Corpus({**gen, **gate}), cfg(True))
    assert "g:1" not in {a["span_id"] for a in b.admitted}
    # one sense only: the phrase group holds only the question's own phrase
    one = am.groups(q, [DEFN], {"enabled": True}, STOP)
    assert len(one) == 1 and one[0].phrases == (("retrieval augmented generation", False),) and one[0].upper == "RAG"


def test_odd_definition_disables_only_that_acronym():
    odd = "Results are shown in the table (RAG)."
    llm = "Large language models (LLMs) generate text."
    q = "How do LLMs use RAG components?"
    gs = am.groups(q, [DEFN, odd, llm], {"enabled": True}, STOP)
    assert [g.upper for g in gs] == ["LLM"]  # RAG is ambiguous here; LLM still expands
    assert am.groups(Q, [DEFN, llm], {"enabled": True}, STOP)[0].upper == "RAG"


def test_df_cache_is_bounded_lru():
    docs = {f"d{i}": ("t", [f"Definition {i}: Alpha beta gamma (ABG{i % 1}) and more text {i}."]) for i in range(3)}
    corp = Corpus({"g1": ("t", [DEFN, GEN_HY]), "l1": ("t", ["Large language models (LLMs) generate text."]), **docs})
    c = cfg(True, "idf_weighted")
    c["admission"]["df_cache_max"] = 2
    for q in (Q, "How do LLMs work?", "What is ABG?", Q, "What are LLMs and RAG?"):
        fa.admit(q, corp, c)
        assert len(corp.__dict__["_fda_cache"]) <= 2
    assert len(corp.__dict__["_fda_cache"]) == 2  # it did fill (distinct group sets), then evicted
    assert FD["admission"]["df_cache_max"] >= 1


def test_singular_and_plural_long_forms_are_one_sense():
    texts = ["Large language models (LLMs) generate text.", "A large language model (LLM) predicts tokens."]
    assert am.definitions(texts, "llm", STOP) == ["large language models"]
    gs = am.groups("How do LLMs work?", texts, {"enabled": True}, STOP)
    assert len(gs) == 1 and gs[0].upper == "LLM"
    g2 = am.groups("How does a large language model work?", texts, {"enabled": True}, STOP)
    assert len(g2) == 1 and g2[0].phrases == (("large language model", False),)
    # still two senses when the long forms really differ
    assert am.groups("What is RAG?", [DEFN, RAG2], {"enabled": True}, STOP) == ()
