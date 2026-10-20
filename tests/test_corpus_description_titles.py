"""ON-1020-FIX-1: a model-written corpus description must never echo a sampled source title (Fix A), and a missing
fulldoc.enabled key is disabled (Fix B). Stubs only: no LLM, no network, no database."""
import pytest

from vera import corpus_description as cd
from vera import fulldoc_corpus

TITLES = ["Nowcasting and Ensemble Forecasting of Seasonal Respiratory Hospitalisations",
          "Estimating Rt in Real Time: Forecast Evaluation Across Epidemic Waves",
          "Workflow Capture and Provenance for Reproducible Analysis Pipelines"]
OVERVIEW = {"source_count": 3, "corpus_updated_at": "2026-10-19T00:00:00"}
PROFILE = {"titles": TITLES, "published_years": ["2020", "2025"]}
CLEAN = "A small collection of epidemic forecasting and reproducible analysis papers from 2020 to 2025."


def _describer(monkeypatch, summary, topics=()):
    monkeypatch.setattr(cd, "stamp_and_profile", lambda engine, n: (dict(OVERVIEW), dict(PROFILE)))
    d = cd.CorpusDescriber({"read_db_url_env": "X", "max_titles": 3, "prompt_version": "v1", "retry_s": 60},
                           engine_factory=lambda *a, **k: object(), env={"X": "stub://"}, clock=lambda: 1000.0)
    calls = []

    def gen(messages):
        calls.append(messages)
        return summary, list(topics), "stub-model"
    return d, gen, calls


def test_clean_summary_passes(monkeypatch):
    d, gen, calls = _describer(monkeypatch, CLEAN, ["Epidemic forecasting", "Short-term epidemic forecasts"])
    out = d.current(gen)
    assert out["summary_source"] == "agentic_model" and out["summary"] == CLEAN and len(out["topics"]) == 2 and len(calls) == 1


@pytest.mark.parametrize("summary", [
    "Covers " + TITLES[0] + ".",
    "covers   ESTIMATING rt in REAL time forecast evaluation across epidemic waves!",  # case/space/punctuation
])
def test_echoed_full_title_falls_back(monkeypatch, summary):
    d, gen, _ = _describer(monkeypatch, summary)
    out = d.current(gen)
    assert out["summary_source"] == "deterministic_fallback" and out["summary"] == cd.fallback_summary(OVERVIEW, PROFILE)
    assert out["topics"] == [] and d._cache is None


def test_echoed_long_ngram_falls_back(monkeypatch):
    d, gen, _ = _describer(monkeypatch, "Work on workflow capture and provenance for reproducibility of studies.")
    assert d.current(gen)["summary_source"] == "deterministic_fallback"


def test_echoed_ngram_in_a_topic_falls_back(monkeypatch):
    d, gen, _ = _describer(monkeypatch, CLEAN, ["Ensemble forecasting of seasonal respiratory"])
    out = d.current(gen)
    assert out["summary_source"] == "deterministic_fallback" and out["topics"] == []


def test_below_threshold_phrase_is_allowed():
    assert not cd.echoes_title("ensemble forecasting of seasonal", TITLES)  # 4 tokens
    assert cd.echoes_title("ensemble forecasting of seasonal respiratory", TITLES)  # 5 tokens
    assert not cd.echoes_title("", TITLES) and not cd.echoes_title(CLEAN, []) and not cd.echoes_title(CLEAN, [None, ""])


def test_short_title_rejected_only_when_it_is_the_whole_sentence_or_topic():
    assert cd.echoes_title("Deep Learning Survey", ["Deep Learning Survey"])
    assert cd.echoes_title("Covers methods. deep learning survey!", ["Deep Learning Survey"])
    assert not cd.echoes_title("A survey of deep learning methods", ["Deep Learning Survey"])


@pytest.mark.parametrize("text", [
    "Reporting Delays and the Short\u200b-Term Epidemic Forecasts: A Multi-Season Assessment",   # zero-width split
    "Reporting Delays and Short-Term Epidemic Forecasts: A Multi-Season Assessment".replace("e", "\u0435"),  # Cyrillic e
    "\uff32eporting \uff24elays and Short-Term Epidemic Forecasts A Multi-Season Assessment",  # fullwidth
    "reporting_delays_and_short_term_epidemic_forecasts",
    "Workflow capture and the provenance for reproducible analysis pipelines",  # inserted stopword
])
def test_evasions_are_caught(text):
    titles = ["Reporting Delays and Short-Term Epidemic Forecasts: A Multi-Season Assessment", TITLES[2]]
    assert cd.echoes_title(text, titles)


def test_title_split_across_summary_and_topics_falls_back(monkeypatch):
    d, gen, _ = _describer(monkeypatch, "Covers estimating rt in real", ["time forecast evaluation across epidemic waves"])
    assert d.current(gen)["summary_source"] == "deterministic_fallback"


def test_echo_uses_longer_cooldown_and_is_distinct(monkeypatch):
    now = [1000.0]
    d, gen, calls = _describer(monkeypatch, "Covers " + TITLES[0])
    d.clock = lambda: now[0]
    d.cfg["echo_retry_s"] = 3600
    d.current(gen)
    assert d._retry_at == 1000.0 + 3600 and issubclass(cd.TitleEchoError, ValueError)
    now[0] = 1000.0 + 600  # past retry_s (60) but inside the echo cooldown: no new model call
    d.current(gen)
    assert len(calls) == 1


def test_missing_fulldoc_enabled_key_is_disabled(monkeypatch, tmp_path):
    fulldoc_corpus.reset_for_tests()
    env = {"VERA_FULLDOC_BUNDLE_DIR": str(tmp_path)}
    base = {"bundle_dir_env": "VERA_FULLDOC_BUNDLE_DIR", "manifest_name": "m", "sums_name": "s"}
    for fd in ({}, {"other": 1}, None, "yes", [], {"enabled": False}, {"enabled": "false"}, {"enabled": "maybe"}, {"enabled": "disabled"}):
        cfg = {**base} if fd is None else {**base, "fulldoc": fd}
        assert fulldoc_corpus.get_corpus(cfg, env) is None
    sentinel = object()  # explicit true passes the flag; an empty dir is unreadable, so load_bundle (stubbed) is reached
    monkeypatch.setattr(fulldoc_corpus, "load_bundle", lambda base, cfg: sentinel)
    assert fulldoc_corpus.get_corpus({**base, "fulldoc": {"enabled": True}}, env) is sentinel
    fulldoc_corpus.reset_for_tests()
    assert fulldoc_corpus.get_corpus({**base, "fulldoc": {"enabled": "true"}}, env) is sentinel
