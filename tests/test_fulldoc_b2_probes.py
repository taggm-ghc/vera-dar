"""D-055 v2, review B2: accepted but off-corpus questions must never be grounded (insufficient evidence or a fallback).

Two layers. (1) SYNTHETIC mimics, always run: questions that share some words with a synthetic full document but ask
about something else. (2) The six named off-corpus probes against the REAL DAR-9 bundle, run only when the env var
VERA_FULLDOC_PROBE_BUNDLE names it (the sanctioned runner masks /home, so they skip there; no real text is in the repo).
Deterministic: no LLM, network or DB. Not tuned against any evaluation set."""
import os
from functools import lru_cache
from pathlib import Path

import pytest

from tests.fulldoc_bundle_synth import synth_cfg, write_bundle
from vera import fulldoc_admission, fulldoc_corpus

SYNTH_OFF = [
    "How do I forecast sales for my startup?",
    "How should I version control my wedding photos?",
    "What transmission interval suits my wifi router?",
    "How do I track an outbreak of mold in my bathroom?",
    "What serial interval should my podcast episodes follow?",
    "Which software version should my bakery workflow use?",
]
# Owner approval 2026-10-21 (WED-P16-RET, "i fully approve"): at admission.min_question_coverage 0.5 this synthetic
# off-corpus probe is ADMITTED (at least 3 of its 5 content stems occur in one synthetic document). Accepted as a known false admit;
# every other probe keeps the insufficient expectation.
# OD-1028(12) (owner 0.34 thresholds, test-data acceptance per repo commit eb48132): two more synthetic off-corpus probes are
# ADMITTED by the lexical gates (2 of 5 content stems in one passage). Owner-accepted residual risk; the claim check is the backstop.
ACCEPTED_FALSE_ADMITS = {"Which software version should my bakery workflow use?",
                         "What transmission interval suits my wifi router?",
                         "What serial interval should my podcast episodes follow?"}
REAL_OFF = [
    "How do I evaluate my startup's sales forecasts?",
    "How should I version control my wedding photos?",
    "What is the best interval for retraining a machine learning model?",
    "Are most nutrition studies false?",
    "How do I forecast my household electricity bill?",
    "What software should I use to archive my family recipes?",
]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = write_bundle(tmp_path_factory.mktemp("b2bundle"))
    cfg = synth_cfg()
    corpus = fulldoc_corpus.load_bundle(root, cfg)
    assert corpus.status == "ok" and len(corpus.acquired) == 3
    return corpus, cfg


@pytest.mark.parametrize("q", SYNTH_OFF)
def test_synthetic_off_corpus_probe_is_insufficient(synth, q):
    corpus, cfg = synth
    ev = fulldoc_admission.admit(q, corpus, cfg)
    if q in ACCEPTED_FALSE_ADMITS:  # owner-accepted residual risk; the claim check and "insufficient" are the backstop
        assert ev.status == "admitted" and ev.admitted
    else:
        assert ev.status == "insufficient" and not ev.grounded and ev.admitted == ()
    assert len(ev.examined_full_documents) == 3  # inspected, recorded, and still not grounded


def test_synthetic_on_corpus_control_is_admitted(synth):  # the tightening does not make admission impossible
    corpus, cfg = synth
    ev = fulldoc_admission.admit("What serial interval of epidemic transmission was compared?", corpus, cfg)
    assert ev.grounded and all(a["attribution"]["licence_url"] for a in ev.admitted)


_REAL = os.environ.get("VERA_FULLDOC_PROBE_BUNDLE") or os.environ.get("VERA_REAL_DAR9_BUNDLE")  # runner sets the latter


@lru_cache(maxsize=1)
def _real():
    cfg = fulldoc_corpus.load_config()  # shipped config, shipped pins
    return cfg, fulldoc_corpus.load_bundle(Path(_REAL), cfg)


@pytest.mark.skipif(not (_REAL and Path(_REAL).is_dir()), reason="real DAR-9 bundle not available (env unset)")
@pytest.mark.parametrize("q", REAL_OFF)
def test_real_bundle_off_corpus_probe_is_insufficient(q):
    cfg, corpus = _real()
    assert corpus.status == "ok" and len(corpus.acquired) == 9
    ev = fulldoc_admission.admit(q, corpus, cfg)
    assert ev.status == "insufficient" and ev.admitted == ()
