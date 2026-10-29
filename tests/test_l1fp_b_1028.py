"""L1FP-1028 commit B: deny:legal pattern 1 ('is it legal to ... my/our/me/i/we') no longer catches academic-integrity
legality questions (cite/quote/attribute/paraphrase/reproduce in one's own academic work); first-person and
personalised legal requests stay declined. Offline, patterns only."""
import pytest

from vera import scope_router as sr

CFG = sr.load_config()


@pytest.mark.parametrize("q", [
    "Is it legal to quote a source in my own thesis?",
    "Is it lawful to cite a paper in my dissertation without permission?",
    "Is it legal to paraphrase an article in my essay?",
    "Is it legal to reproduce a figure from a journal article in my thesis?"])
def test_academic_legality_questions_reach_layer2(q):
    assert sr.layer1(q, CFG)[1] != "deny:legal", q


@pytest.mark.parametrize("q", [
    "Is it legal for me to quote a source in my thesis?",
    "Is it legal to sue my landlord?",
    "Is it legal for me to record my boss?",
    "Is it legal to quote a source in a thesis and record my boss?",
    "Is it legal to quote my thesis supervisor in an email to sue my landlord?",
    "Would it be legal to paraphrase an article in my essay when I cite it?",  # first-person 'I': conservative
    "Is it legal to quote a source in my thesis? Also, is it legal to record my boss?"])
def test_personalised_legal_still_declined(q):
    assert sr.layer1(q, CFG) == (sr.DECLINE, "deny:legal"), q


@pytest.mark.parametrize("q", [
    "Is it legal to quote a source in my thesis\u2026 record my boss",
    "Is it legal to quote a source in my thesis... record my boss",
    "Is it legal to quote a source in my thesis. Record my boss",
    "Is it legal to quote a source in my thesis. Sue my landlord."])
def test_ellipsis_and_period_do_not_hide_first_person(q):
    import unicodedata
    for form in (q, unicodedata.normalize("NFKC", q)):
        assert sr.layer1(form, CFG) == (sr.DECLINE, "deny:legal"), form
