"""CORPUS-DIGEST-1104: bounded agentic corpus digest. Offline only: scripted model (tests/digest_fakes.py), fixture
corpus, fake engine. No network, no database."""
import json

import pytest

from vera import corpus_description as cd
from tests.digest_fakes import FakeLLM

TITLES = ["Influenza nowcasting with hospital admissions", "Ensemble forecasts of dengue incidence",
          "Rt estimation during epidemic waves", "Coral bleaching under marine heatwaves",
          "Soil carbon in temperate grasslands", "Containerised workflows for reproducible pipelines"]
OVERVIEW = {"source_count": 209, "corpus_updated_at": "2026-11-04T10:00:00+00:00"}
PROFILE = {"titles": TITLES, "excerpts": [""] * 6, "source_types": {"journal-article": 200, "preprint": 9},
           "discovered_via": {"europe_pmc": 209}, "text_kinds": {"abstract_metadata": 209},
           "published_years": ["2005", "2026"], "year_counts": {"2025": 120, "2026": 40},
           "added_last_30_days": 31}
CLUSTERS = [("epidemic forecasting", [1, 2, 3]), ("marine and soil ecology", [4, 5]), ("research reproducibility", [6])]
GOOD = "The sampled sources concern epidemic forecasting, ecology and reproducibility."  # MVP: no figures
CFG = {"read_db_url_env": "X", "max_titles": 6, "prompt_version": "v2", "retry_s": 60, "echo_retry_s": 3600}


def run(llm, profile=PROFILE, cfg=CFG, now=None):
    return cd.run_digest_pipeline(dict(OVERVIEW), dict(profile), llm, cfg, **({"now": now} if now else {}))


def test_pipeline_happy_path_steps_bounds_and_subject_topics():
    llm = FakeLLM(CLUSTERS, [GOOD], coverage="The collection currently covers epidemic forecasting, ecology and reproducibility.")
    out = run(llm)
    assert llm.steps() == ["cluster", "draft", "check"]  # at most 3 model calls
    assert [c["max_tokens"] for c in llm.calls] == [900, 700, 500]
    assert all(0 < c["timeout_s"] <= 25 for c in llm.calls)
    assert out["summary"] == GOOD and out["topics"] == ["epidemic forecasting", "marine and soil ecology", "research reproducibility"]
    assert out["coverage"] == ("The collection currently includes research on epidemic forecasting, marine and soil ecology "
                               "and research reproducibility.") and out["model"] == "stub-model"  # code-built, not the model's
    assert out["topic_shares"]["counts"] == "documents" and out["topic_shares"]["basis_documents"] == 6


def test_checker_unsupported_sentences_are_dropped_and_coverage_falls_back_to_clusters():
    llm = FakeLLM(CLUSTERS, [GOOD, "It also covers astrophysics."], coverage="It covers everything.", supported={1, 3, 4, 5})  # 3-5: the three topic labels
    out = run(llm)
    assert out["summary"] == GOOD
    assert out["coverage"] == cd.coverage_from_clusters(cd.parse_clusters(
        {"clusters": [{"label": a, "members": b} for a, b in CLUSTERS]}, 6, 8, TITLES), 6)
    assert "is not" not in out["coverage"] and "only" not in out["coverage"]  # a description, not a limit


def test_no_supported_sentence_fails_verbosely():
    with pytest.raises(cd.DigestError, match="verify") as e:
        run(FakeLLM(CLUSTERS, [GOOD], supported=set()))
    assert [t["step"] for t in e.value.trace] == ["cluster", "draft", "check", "verify"]


def test_cluster_members_are_validated_and_primary_only():
    obj = {"clusters": [{"label": "epidemic forecasting", "members": [1, 2, 2, 99, 0, True, "3"]},
                        {"label": "ecology", "members": [2, 4]}, {"label": "empty", "members": []}, "junk"]}
    cl = cd.parse_clusters(obj, 6, 8, TITLES)
    assert cl == [{"label": "epidemic forecasting", "members": [1, 2], "count": 2},
                  {"label": "ecology", "members": [4], "count": 1}]


def test_deadline_bounds_the_pipeline():
    t = [0.0]

    def now():
        t[0] += 20.0  # each look at the clock costs 20 s; deadline 25 s
        return t[0]
    llm = FakeLLM(CLUSTERS, [GOOD])
    with pytest.raises(cd.DigestError, match="time budget") as e:
        run(llm, now=now)
    # the cluster reply came back after the deadline: it is not used and no further step runs
    assert llm.steps() == ["cluster"] and e.value.trace[-1] == {"step": "cluster", "ok": False, "why": "deadline"}


def test_model_failure_names_the_step():
    with pytest.raises(cd.DigestError, match="draft: model call failed"):
        run(FakeLLM(CLUSTERS, [GOOD], fail_step="draft"))


def test_excerpts_only_reach_the_model_when_configured():
    prof = {**PROFILE, "excerpts": ["We study flu admissions."] + [""] * 5}
    llm = FakeLLM(CLUSTERS, [GOOD])
    run(llm, profile=prof)
    assert "flu admissions" not in llm.calls[0]["messages"][1]["content"]
    llm2 = FakeLLM(CLUSTERS, [GOOD])
    run(llm2, profile=prof, cfg={**CFG, "agentic": {"sample_excerpt_chars": 100}})
    assert "excerpt: We study flu admissions." in llm2.calls[0]["messages"][1]["content"]
    assert "untrusted data" in llm.calls[0]["messages"][0]["content"]


# --- 80% topic shares (deterministic, owner suggestion 2026-11-04) -----------------------------------------------
def C(*pairs):
    return [{"label": a, "count": n} for a, n in pairs]


def test_shares_exactly_80_percent_reaches_the_target():
    s = cd.topic_shares(C(("a", 5), ("b", 3), ("c", 2)), 10)
    assert [t["label"] for t in s["main"]] == ["a", "b"] and s["main_pct"] == 80 and s["target_reached"]
    assert s["other"] == {"topics": 1, "documents": 2, "pct": 20} and s["ungrouped"]["documents"] == 0


def test_shares_just_below_80_takes_the_next_topic():
    s = cd.topic_shares(C(("a", 50), ("b", 29), ("c", 21)), 100)
    assert [t["label"] for t in s["main"]] == ["a", "b", "c"] and s["other"]["topics"] == 0


def test_shares_single_dominant_topic():
    s = cd.topic_shares(C(("small", 1), ("big", 9)), 10)
    assert [t["label"] for t in s["main"]] == ["big"] and s["main"][0]["pct"] == 90
    assert s["other"] == {"topics": 1, "documents": 1, "pct": 10}


def test_shares_ties_break_by_label_deterministically():
    a = cd.topic_shares(C(("zeta", 4), ("alpha", 4), ("mid", 2)), 10)
    b = cd.topic_shares(C(("alpha", 4), ("mid", 2), ("zeta", 4)), 10)
    assert a == b and [t["label"] for t in a["main"]] == ["alpha", "zeta"]


def test_shares_empty_corpus():
    s = cd.topic_shares([], 0)
    assert s["main"] == [] and not s["target_reached"] and s["other"]["topics"] == 0
    z = cd.topic_shares(C(("a", 1)), 0)  # no division by zero
    assert z["main"] == [] and z["main_pct"] == 0


def test_shares_with_ungrouped_documents_sum_to_100():
    s = cd.topic_shares(C(("a", 6), ("b", 1)), 10)  # 3 sampled documents in no cluster
    assert [t["label"] for t in s["main"]] == ["a", "b"] and not s["target_reached"]
    assert s["main_pct"] + s["other"]["pct"] + s["ungrouped"]["pct"] == 100 and s["ungrouped"]["documents"] == 3


# --- describer: previous digest on failure, stored with shares and coverage ---------------------------------------
class _Describer(cd.CorpusDescriber):
    def __init__(self, monkeypatch, overview):
        super().__init__(CFG, engine_factory=lambda *a, **k: object(), env={"X": "stub://"}, clock=lambda: self.t)
        self.t, self.overview = 0.0, overview
        monkeypatch.setattr(cd, "stamp_and_profile", lambda e, n, ab=0: (dict(self.overview), dict(PROFILE)))


def test_failure_keeps_the_previous_digest_then_deterministic_without_one(monkeypatch):
    d = _Describer(monkeypatch, dict(OVERVIEW))
    first = d.current(FakeLLM(CLUSTERS, [GOOD]))
    assert first["summary_source"] == cd.CHECKED and first["topic_shares"]["main"]
    d.overview = {**OVERVIEW, "source_count": 210, "corpus_updated_at": "2026-11-05T00:00:00+00:00"}
    out = d.current(FakeLLM(CLUSTERS, [GOOD], fail_step="cluster"))
    assert out["summary"] == GOOD and out["summary_generated_at"] == first["summary_generated_at"]
    assert out["source_count"] == 210 and out["coverage"] == first["coverage"]
    assert d.last_trace == [{"step": "cluster", "ok": False, "why": "RuntimeError"}]
    fresh = _Describer(monkeypatch, dict(OVERVIEW))
    out = fresh.current(FakeLLM(CLUSTERS, [GOOD], fail_step="cluster"))
    assert out["summary_source"] == "deterministic_fallback" and out["coverage"] is None and out["topic_shares"] is None


def test_api_model_carries_coverage_and_shares():
    import main
    fields = main.CorpusDescription.model_fields
    assert "coverage" in fields and "topic_shares" in fields


# --- grounded facts: read-only SQL ---------------------------------------------------------------------------------
class _Res:
    def __init__(self, rows):
        self.rows = rows

    def one(self):
        return self.rows[0]

    def all(self):
        return self.rows

    def scalar(self):
        return self.rows[0][0]

    def __iter__(self):
        return iter(self.rows)


class _Conn:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        q = str(sql)
        self.log.append((q, params))
        if q.startswith("SELECT count(*), max(created_at)"):
            return _Res([(2, None)])
        if "md5(s.source_id::text)" in q:
            return _Res([("Title A", None), ("Title B", "abs")])
        if "min(substring" in q:
            return _Res([("2020", "2026")])
        if "interval '30 days'" in q:
            return _Res([(1,)])
        if "AS y" in q:
            return _Res([("2020", 1), ("2026", 1)])
        return _Res([("x", 2)])


class _Eng:
    def __init__(self):
        self.log = []

    def connect(self):
        return _Conn(self.log)


def test_facts_queries_are_select_only_and_default_to_titles_only():
    eng = _Eng()
    overview, profile = cd.stamp_and_profile(eng, 5)
    assert all(q.lstrip().upper().startswith("SELECT") for q, _ in eng.log)
    sample = [p for q, p in eng.log if "md5(s.source_id::text)" in q][0]
    assert sample["ab"] == 0 and sample["n"] == 5 and sample["fd_kind"] == cd.FULLDOC_CHUNKS_KIND
    assert profile["titles"] == ["Title A", "Title B"] and profile["added_last_30_days"] == 1
    assert profile["year_counts"] == {"2020": 1, "2026": 1} and overview["source_count"] == 2
    facts = cd.grounded_facts(overview, profile)
    assert "peer" not in json.dumps(facts).lower()


def test_json_with_chain_bounds_each_call_and_tries_entries_in_order():
    seen = []

    class Completions:
        def __init__(self, fail):
            self.fail = fail

        def create(self, **kw):
            seen.append(kw)
            if self.fail:
                raise RuntimeError("x")

            class R:
                choices = [type("C", (), {"message": type("M", (), {"content": '{"ok": 1}'})()})()]
            return R()

    class Entry:
        def __init__(self, label, fail):
            self.label, self.model, self.extra_body, self.fail = label, label, {}, fail

        def client(self):
            return type("Cl", (), {"chat": type("Ch", (), {"completions": Completions(self.fail)})()})()
    obj, label = cd.json_with_chain([Entry("free-a", True), Entry("free-b", False)])([{"role": "user", "content": "x"}],
                                                                                   max_tokens=123, timeout_s=7)
    assert obj == {"ok": 1} and label == "free-b" and [k["model"] for k in seen] == ["free-a", "free-b"]
    assert all(k["max_tokens"] == 123 and 0 < k["timeout"] <= 7 and k["temperature"] == 0 for k in seen)
    assert seen[1]["timeout"] <= seen[0]["timeout"]  # one shared budget, not a fresh timeout per entry


def test_json_with_chain_stops_when_the_shared_budget_is_spent(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(cd.time, "monotonic", lambda: clock[0])
    tried = []

    class Entry:
        def __init__(self, label):
            self.label, self.model, self.extra_body = label, label, {}

        def client(self):
            def create(**kw):
                tried.append(self.label)
                clock[0] += 10.0  # each attempt uses 10 s of a 15 s budget
                raise RuntimeError("slow")
            return type("Cl", (), {"chat": type("Ch", (), {"completions": type("Co", (), {"create": staticmethod(create)})()})()})()
    with pytest.raises(RuntimeError, match="TimeoutError"):
        cd.json_with_chain([Entry("a"), Entry("b"), Entry("c")])([], max_tokens=10, timeout_s=15)
    assert tried == ["a", "b"]


def test_late_reply_is_not_used():
    t = [0.0]

    def now():
        return t[0]

    def slow(messages, max_tokens=None, timeout_s=None):
        t[0] += 30.0  # past the 25 s deadline
        return {"clusters": []}, "m"
    with pytest.raises(cd.DigestError, match="after the time budget"):
        run(slow, now=now)


def test_numbers_from_titles_are_not_allowed():  # MVP: no number in model text at all
    prof = {**PROFILE, "titles": TITLES[:5] + ["COVID-19 waves and hospital load"]}
    out = run(FakeLLM(CLUSTERS, ["One sampled source concerns COVID-19 waves.", GOOD]), profile=prof)
    assert out["summary"] == GOOD


def test_main_topics_line_states_the_real_share():
    from vera import sources_sidebar as sb
    shares = cd.topic_shares(C(("dominant subject", 9), ("minor subject", 1)), 10)
    lines = sb._share_lines(shares)
    assert lines[0].startswith("**Main topics** (90% of documents; share of the 10 sampled sources")
    assert "dominant subject 90%" in lines[0] and lines[1] == "Other topics: 10% across 1 smaller topic."
    assert "full documents stored as passages are not in the sample" in lines[0]  # D7: an older digest (no key) keeps it
    fd = sb._share_lines(cd.topic_shares(C(("dominant subject", 9), ("minor subject", 1)), 10, full_documents_in_sample=3))
    assert "share of the 10 sampled sources, including 3 full documents stored as passages" in fd[0]  # DIGEST-IMPROVE-1105
    assert "not in the sample" not in fd[0]
    import inspect
    assert "<> :fd_kind" not in inspect.getsource(cd.stamp_and_profile)  # the sample query no longer excludes them


# --- CORPUS-DIGEST-1104-VERIFY adversarial cases (scratch/CORPUS-DIGEST-1104-VERIFY/adversarial.py) -----------------
ADV_TITLES = ["Influenza forecasting with mechanistic models in 2019", "Coral bleaching under marine heatwaves",
              "Seasonal influenza transmission dynamics", "Reef fish recovery after bleaching events",
              "Peer review practices in ecology", "Ninety-five percent intervals for epidemic curves 95"]
ADV_PROF = {"titles": ADV_TITLES, "excerpts": [""] * 6, "source_types": {"journal_article": 200},
            "discovered_via": {"europe_pmc": 150, "openalex": 59}, "text_kinds": {"abstract_metadata": 209},
            "published_years": ["2005", "2026"], "year_counts": {"2024": 10}, "added_last_30_days": 12}
ADV_CL = [("influenza epidemiology", [1, 3, 6]), ("coral reef ecology", [2, 4, 5])]
ADV_OK = "The sampled sources concern influenza transmission and forecasting, alongside coral reef bleaching."


def test_sentence_citing_no_valid_cluster_is_dropped():  # D6
    # clusters are ranked by size then label: C1 = coral reef ecology, C2 = influenza epidemiology
    llm = FakeLLM(ADV_CL, [("Influenza transmission is covered.", []), ("Coral reef bleaching is covered.", [9, True]),
                           ("Influenza forecasting is covered.", [2])])
    out = cd.run_digest_pipeline(dict(OVERVIEW), dict(ADV_PROF), llm, {})
    assert out["summary"] == "Influenza forecasting is covered."


@pytest.mark.parametrize("verdicts", [[], [{"n": 2, "supported": True}], [{"n": 1, "supported": "true"}],
                                      [{"n": True, "supported": True}], [{"n": 1.0, "supported": True}], ["x"]])
def test_malformed_verdicts_never_support_a_sentence(verdicts):  # D4: {"n": true} is not sentence 1
    class Chk(FakeLLM):
        def __call__(self, m, max_tokens=None, timeout_s=None, avoid=None):
            if self.step(m) == "check":
                self.calls.append({"step": "check"})
                return {"verdicts": verdicts}, "x"
            return super().__call__(m, max_tokens, timeout_s, avoid)
    with pytest.raises(cd.DigestError, match="verify"):
        cd.run_digest_pipeline(dict(OVERVIEW), dict(ADV_PROF), Chk(ADV_CL, [ADV_OK]), {})


def test_checker_asks_for_a_model_other_than_the_drafter_and_records_it():  # D6, owner rule: distinct skeptic models
    llm = FakeLLM(ADV_CL, [ADV_OK], labels={"draft": "groq:drafter", "check": "groq:checker"})
    out = cd.run_digest_pipeline(dict(OVERVIEW), dict(ADV_PROF), llm, {})
    assert [c["avoid"] for c in llm.calls] == [None, None, "groq:drafter"]
    check = [t for t in out["trace"] if t["step"] == "check"][0]
    assert check["model"] == "groq:checker" and check["distinct_from_drafter"] is True
    assert out["model"] == "groq:drafter" and out["check_model"] == "groq:checker"
    same = FakeLLM(ADV_CL, [ADV_OK])  # the chain offers no other model: recorded, not hidden
    out = cd.run_digest_pipeline(dict(OVERVIEW), dict(ADV_PROF), same, {})
    assert [t for t in out["trace"] if t["step"] == "check"][0]["distinct_from_drafter"] is False


def test_json_with_chain_tries_other_models_first_when_asked():
    seen = []

    class Entry:
        def __init__(self, label):
            self.label, self.model, self.extra_body = label, label, {}

        def client(self):
            def create(**kw):
                seen.append(kw["model"])

                class R:
                    choices = [type("C", (), {"message": type("M", (), {"content": '{"ok": 1}'})()})()]
                return R()
            return type("Cl", (), {"chat": type("Ch", (), {"completions": type("Co", (), {"create": staticmethod(create)})()})()})()
    llm = cd.json_with_chain([Entry("a"), Entry("b")])
    assert llm.supports_avoid and llm([], avoid="a") == ({"ok": 1}, "b") and seen == ["b"]
    assert llm([], avoid="zzz") == ({"ok": 1}, "a")



# --- release verification: coverage is code-built; boundary and document-type claims are the checker's to deny ------
@pytest.mark.parametrize("cov", ["VERA-DAR only answers questions about influenza.", "This collection is limited to epidemiology.",
                                 "It covers everything."])
def test_model_coverage_is_never_used(cov):
    llm = FakeLLM(CLUSTERS, [GOOD], coverage=cov)
    out = run(llm)
    assert out["coverage"].startswith("The collection currently includes research on ") and cov not in out["coverage"]
    check_user = [c for c in llm.calls if c["step"] == "check"][0]["messages"][1]["content"]
    assert cov not in check_user  # not even offered to the checker
    assert '"coverage"' not in cd.DRAFT_SYSTEM


def test_checker_prompt_requires_denying_boundary_and_document_type_claims():
    for phrase in ("what VERA-DAR answers, allows or is limited to", "what kind of documents the sources are"):
        assert phrase in cd.CHECK_SYSTEM


@pytest.mark.parametrize("bad", ["VERA-DAR only answers questions about epidemic forecasting.",
                                 "This collection is limited to epidemic forecasting.",
                                 "The epidemic forecasting sources are journal articles."])
def test_boundary_sentences_are_dropped_when_the_checker_follows_its_prompt(bad):
    out = run(FakeLLM(CLUSTERS, [bad, GOOD], supported={2, 3, 4, 5}))  # the checker denies sentence 1 as instructed
    assert out["summary"] == GOOD


def test_coverage_is_empty_when_no_label_is_approved():
    out = run(FakeLLM(CLUSTERS, [GOOD], supported={1}))
    assert out["coverage"] == "" and out["topics"] == []
