"""Item #74 M2 / K2.I2: the one public no-profiling / limits disclosure. The text is composed here, at call time, from
the answer contract (vera.answer_contract), so tests can pin it and it cannot drift from the enabled layers."""
import json

from vera import answer_contract
from vera.memory_gate import CONFIG_KEY, CONFIG_PATH

_BASE = ("VERA-DAR is an AI system. For each in-scope question it searches published scholarly sources and the web and "
         "lists published documents that may be relevant, with links. VERA-DAR has not acquired or read these full "
         "documents, so it does not use abstracts or snippets as evidence and does not write an answer from them. "
         "The list is not neutral or complete and may be wrong. Check the sources and read them yourself before "
         "relying on any claim. Licence terms are as reported by the indexing services for the listed records and were "
         "not verified with the publisher. VERA-DAR keeps no profile of you and does not tailor answers to you.")
# Owner decision 2026-10-19 (Option C), exact wording. The first two sentences are about third parties; the last two
# are VERA-DAR's own storage claims and are emitted only while the shipped configuration backs them (_demo_no_storage).
_THIRD_PARTY = (" Your question is sent to third-party AI and search services to generate a response. These services "
                "have their own data-handling and retention policies.")
_NO_SAVE = " VERA-DAR does not save your question or information identifying you."
_DEMO_NO_STORAGE = (" During this demonstration, VERA-DAR does not save newly discovered documents or research notes from "
                    "your questions.")
_CORPUS_MAY_STORE = " VERA-DAR may add documents found for your question to its own collection; it does not store the question."
_EVIDENCE_FINDINGS = " Earlier findings were checked only against the abstract or index record they cite."


def confirm_required(path=CONFIG_PATH) -> bool:
    """True unless the memory config explicitly turns owner confirmation off (unreadable config -> True, the stricter text).
    Kept for callers; the text no longer varies with it (discovery notes always need owner review)."""
    try:
        return bool(json.loads(open(path).read())[CONFIG_KEY]["store"].get("confirm_required", True))
    except (OSError, ValueError, KeyError, TypeError):
        return True


def corpus_admission_enabled(chain_path=answer_contract.CHAIN_CONFIG_PATH) -> bool:
    """True unless the chain config explicitly sets corpus.enabled to false (unreadable -> True, the stricter text)."""
    try:
        return (json.loads(open(chain_path).read()).get("corpus") or {}).get("enabled") is not False
    except (OSError, ValueError, AttributeError):
        return True


def disclosure_text(path=CONFIG_PATH, contract_path=answer_contract.CONFIG_PATH,
                    chain_path=answer_contract.CHAIN_CONFIG_PATH) -> str:
    """Base text (L0 always), the third-party/no-save sentences, then either the demonstration no-storage sentence
    (corpus admission off and L2 off) or the matching storage sentences. `path` is accepted for compatibility."""
    out = _BASE + _THIRD_PARTY + _NO_SAVE
    corpus_on = corpus_admission_enabled(chain_path)
    if not corpus_on and "L2" not in answer_contract.layers(contract_path):
        out += _DEMO_NO_STORAGE
    elif corpus_on:
        out += _CORPUS_MAY_STORE
    if "L2" in answer_contract.layers(contract_path):
        out += (" VERA-DAR's memory may hold discovery notes: which documents VERA-DAR listed for a topic, taken from the "
                "document titles. A discovery note is not a finding and not evidence. "
                + answer_contract.text("discovery_note_submitted", contract_path))
        if answer_contract.evidence_recall_enabled(contract_path):
            out += _EVIDENCE_FINDINGS
    return out


DISCLOSURE_TEXT = disclosure_text()
DISCLOSURE_TEXT_UNCONFIRMED = DISCLOSURE_TEXT  # identical while evidence recall is off


def render_disclosure(st) -> None:
    st.caption(disclosure_text())
