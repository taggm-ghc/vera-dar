"""Item #74 M2 / K2.I2: the one public no-profiling / limits disclosure. The text is composed here, at call time, from
the answer contract (vera.answer_contract), so tests can pin it and it cannot drift from the enabled layers."""
import json
from pathlib import Path

from vera import answer_contract
from vera.memory_gate import CONFIG_KEY, CONFIG_PATH

_BASE = ("VERA-DAR is an AI system. For each in-scope question it searches published scholarly sources and the web and "
         "lists published documents that may be relevant, with links. VERA-DAR has not acquired or read these full "
         "documents, so it does not use abstracts or snippets as evidence and does not write an answer from them. "
         "The list is not neutral or complete and may be wrong. Check the sources and read them yourself before "
         "relying on any claim. For listed records, only the indexing services' metadata licence was checked; the articles' own "
         "licences were not verified with the publisher. VERA-DAR keeps no profile of you and does not tailor answers to you.")
# S1 / DISC-E2 (owner direction 2026-10-21; wording needs owner approval before public activation): used instead of
# _BASE when the full-document path is enabled (fulldoc.enabled in config/fulldoc_answer.json).
_BASE_FULLDOC = (
    "VERA-DAR is an AI system. For each in-scope question it first searches a small collection of full documents that "
    "its operator selected and licence-checked in advance. If passages in them address the question, it writes a short "
    "answer from up to six passages, cites them with brief quotes, and has an AI model check each cited sentence "
    "against its passage; unsupported sentences are removed. Otherwise it searches published scholarly sources and the "
    "web and lists documents that may be relevant, with links. It has not read those listed documents and does not use "
    "their abstracts or snippets as evidence. Answers and lists are not neutral or complete and may be wrong. Check the "
    "sources and read them yourself before relying on any claim. Licences for documents in its own collection were "
    "checked against the licence recorded in each document and VERA-DAR's licence rules, not with the publisher. "
    "For listed records, only the indexing services' metadata licence was checked; the articles' own licences were "
    "not verified with the publisher. VERA-DAR keeps no profile of you and does not tailor answers to you. When it answers from its own "
    "collection, your question, the selected passages with their titles and links, and the draft answer sentences are "
    "sent to third-party AI services.")
FULLDOC_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "fulldoc_answer.json"
# Owner decision 2026-10-19 (Option C), exact wording. The first two sentences are about third parties; the last two
# are VERA-DAR's own storage claims and are emitted only while the shipped configuration backs them (_demo_no_storage).
_THIRD_PARTY = (" Your question is sent to third-party AI and search services to generate a response. These services "
                "have their own data-handling and retention policies.")
_NO_SAVE = " VERA-DAR does not save your question or information identifying you."
_DEMO_NO_STORAGE = (" During this demonstration, VERA-DAR does not save newly discovered documents or research notes from "
                    "your questions.")
_CORPUS_MAY_STORE = " VERA-DAR may add documents found for your question to its own collection; it does not store the question."
# WED-ACQ2 / P1.7 (owner approval 2026-10-21 of "the new disclosure wording"; text chosen per plan D-2): replaces
# _DEMO_NO_STORAGE while automatic acquisition is switched on in config/auto_acquire.json. Accurate to what the code
# stores (document chunks, source and licence provenance, acquisition time and trigger kind; never the question).
_AUTO_ACQUIRE_MAY_STORE = (" Its collection also grows automatically: VERA-DAR may add open-access full-text documents "
                           "found while searching for your question, after automated licence and content checks, recording "
                           "each document's source, licence and the time it was added; it does not store your question.")
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


def fulldoc_enabled(path=FULLDOC_CONFIG_PATH) -> bool:
    """Runtime state for the wording. True only for an explicit true-like fulldoc.enabled (same allowlist as
    fulldoc_corpus.get_corpus). An unreadable or malformed config returns True: the text then over-discloses
    (passages may be sent) rather than under-discloses. A readable config without the key means off (path disabled)."""
    try:
        fd = json.loads(Path(path).read_text(encoding="utf-8")).get("fulldoc")
    except (OSError, ValueError, AttributeError):
        return True
    if not isinstance(fd, dict) or "enabled" not in fd:
        return False
    en = fd["enabled"]
    return en is True or str(en).strip().lower() in ("true", "1", "yes", "on")


_HELP_LIST = ("Ask a question. VERA-DAR lists sources that may be relevant, with links. It has not read them and does "
              "not write an answer.")
_HELP_FULLDOC = ("Ask a question. VERA-DAR answers only from the full documents it holds, with cited passages, or says "
                 "why it cannot; otherwise it lists sources it has not read.")


def auto_acquire_enabled() -> bool:
    """The release-config flag only (config/auto_acquire.json enabled is exactly true). Deliberately ignores the env
    gates: a service without the write account (the UI) over-discloses, never under-discloses. Unreadable = False is
    safe here because an unreadable acquisition config also switches the feature itself off (vera.acq_config)."""
    from vera import acq_config
    return acq_config.flag_on(acq_config.load_config())


def ask_help_text(fulldoc_path=FULLDOC_CONFIG_PATH) -> str:
    """Help text of the question box, chosen by the runtime fulldoc flag (S1)."""
    return _HELP_FULLDOC if fulldoc_enabled(fulldoc_path) else _HELP_LIST


_OFFTOPIC = (" For a question outside its scope VERA-DAR gives no answer, but it may still search published scholarly "
             "sources for the question and list documents it has not read.")


# DAR-1021 (wording needs owner approval, DISC-E2): used instead of _OFFTOPIC only while automatic acquisition is on
# AND scope_expansion.override_layer2_declines is true in config/auto_acquire.json.
_OFFTOPIC_ANSWERING = (" For a question outside its scope VERA-DAR first lists published scholarly documents it has not read "
                       "and searches for open-access full text on the question in the same request. If documents found "
                       "this way pass its licence, content and relevance checks, VERA-DAR answers only from those "
                       "documents, in the same way: brief quoted citations, each cited sentence "
                       "checked by an AI model, unsupported sentences removed. Such answers may be wrong.")


def _layer2_answering_on() -> bool:
    """Release-config flags only (over-disclose on a service without the DB env). Unreadable = True: over-disclose."""
    try:
        from vera import acq_config
        cfg = acq_config.load_config()
        return acq_config.flag_on(cfg) and acq_config.override_layer2_declines(cfg)
    except Exception:  # noqa: BLE001
        return True


def offtopic_sentence() -> str:
    """FIX1-1021: the sentence is emitted while the off-topic reading list is configured on (unreadable config -> emitted:
    over-disclose rather than under-disclose)."""
    try:
        from vera.scope_router import offtopic_enabled

        if not offtopic_enabled():
            return ""
        return _OFFTOPIC_ANSWERING if _layer2_answering_on() else _OFFTOPIC
    except Exception:  # noqa: BLE001
        return _OFFTOPIC


def disclosure_text(path=CONFIG_PATH, contract_path=answer_contract.CONFIG_PATH,
                    chain_path=answer_contract.CHAIN_CONFIG_PATH, fulldoc_path=FULLDOC_CONFIG_PATH) -> str:
    """Base text (the full-document variant when fulldoc is enabled) (L0 always), the third-party/no-save sentences, then either the demonstration no-storage sentence
    (corpus admission off and L2 off) or the matching storage sentences. `path` is accepted for compatibility."""
    out = (_BASE_FULLDOC if fulldoc_enabled(fulldoc_path) else _BASE) + offtopic_sentence() + _THIRD_PARTY + _NO_SAVE
    corpus_on = corpus_admission_enabled(chain_path)
    auto = auto_acquire_enabled()
    if not corpus_on and "L2" not in answer_contract.layers(contract_path):
        out += _AUTO_ACQUIRE_MAY_STORE if auto else _DEMO_NO_STORAGE
    else:
        if corpus_on:
            out += _CORPUS_MAY_STORE
        if auto:  # every state with automatic acquisition on says so (never under-discloses)
            out += _AUTO_ACQUIRE_MAY_STORE
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
