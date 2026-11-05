"""NEWS-LANE-1104: offline tests for vera/news_intent.py (deterministic, config-driven). No network."""
from datetime import date

import pytest

from vera import news_intent as ni

TODAY = date(2026, 11, 4)


@pytest.mark.parametrize("q", [
    "What make the news about the USA 2026 midterm elections?",  # PERF-MISS-1104 case B
    "What is newsworthy about rethe USA 2026 midterm elections?",  # case A (typo kept)
    "Latest headlines on the ceasefire",
    "What is in the news about OpenAI this week?",
    "Who won the 2025 referendum?",
    "media coverage of the hurricane",
])
def test_news_intent_positive(q):
    assert ni.news_intent(q, today=TODAY)


@pytest.mark.parametrize("q", [
    "How do transformers work?",
    "What caused the 1929 stock market crash?",
    "Explain the 2008 election of Obama",  # event noun but an old year
    "What is retrieval-augmented generation?",
    "",
    None,
])
def test_news_intent_negative(q):
    assert not ni.news_intent(q, today=TODAY)


def test_year_alone_or_noun_alone_is_not_enough():
    assert not ni.news_intent("transformer models in 2026", today=TODAY)
    assert not ni.news_intent("how do elections work", today=TODAY)
    assert ni.cues("elections in 2026", today=TODAY) == ["event:election+2026"] or \
        ni.cues("elections in 2026", today=TODAY) == ["event:elections+2026"]


def test_config_driven_and_never_raises():
    assert ni.news_intent("zebra parade", cfg={"news_intent": {"cues": ["zebra parade"]}})
    assert not ni.news_intent("news", cfg={"news_intent": {}})
    assert ni.cues("news", cfg={"news_intent": {"cues": 5}}) == []  # bad shape -> no intent, no raise
