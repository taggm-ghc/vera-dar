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
# NONASK-1103 SD.1 (R-038): added at render time only while a lane flag is on (both ship off).
_LANE_SEARCH = (" Search terms derived from your question may also be sent to GDELT (a news index) and Hacker News "
                "search (Algolia).")
_LANE_PAGES = (" VERA-DAR may read public web pages from vendors and news outlets to answer; these pages are used only "
               "for your answer, and page text is not saved.")
_LANE_LICENCE = " Pages read for one answer are not licence-cleared or stored."
_LANE_GDELT_CREDIT = " News index: [GDELT](https://www.gdeltproject.org/)."


def reporting_lane_on() -> bool:
    try:
        from vera import reporting_lane
        return bool(reporting_lane.lane_enabled())
    except Exception:  # noqa: BLE001
        return False


def _official_on() -> bool:
    try:
        from vera import official_lane
        return bool(official_lane.lane_enabled())
    except Exception:  # noqa: BLE001
        return False


def lanes_disclosure_on() -> bool:
    """True when the Official or Current-reporting lane flag is on (any error -> False)."""
    return _official_on() or reporting_lane_on()


def lane_third_party() -> str:
    """Extra third-party sentences; empty while both lanes are off."""
    if not lanes_disclosure_on():
        return ""
    return (_LANE_SEARCH if reporting_lane_on() else "") + _LANE_PAGES


def third_party_text() -> str:
    return _THIRD_PARTY + lane_third_party()


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
    try:
        return acq_config.flag_on(acq_config.load_config())
    except acq_config.ConfigError:  # already logged at ERROR (paths only); startup raises, the request path degrades to OFF
        return False


def ask_help_text(fulldoc_path=FULLDOC_CONFIG_PATH) -> str:
    """Help text of the question box, chosen by the runtime fulldoc flag (S1)."""
    return _HELP_FULLDOC if fulldoc_enabled(fulldoc_path) else _HELP_LIST


_OFFTOPIC = (" If the collection VERA-DAR holds covers a question, it answers from that collection; if it does not, "
             "VERA-DAR may search published scholarly sources and give a reading list of documents it has not read, "
             "which is not evidence for an answer. Some requests are refused for safety reasons.")


# DAR-1021 (wording needs owner approval, DISC-E2): used instead of _OFFTOPIC only while automatic acquisition is on
# AND scope_expansion.override_layer2_declines is true in config/auto_acquire.json.
_OFFTOPIC_ANSWERING = (" If the collection VERA-DAR holds cannot answer a question faithfully, it searches for open-access full "
                       "text on the question in the same request. If documents found this way pass its licence, content and "
                       "relevance checks, VERA-DAR answers only from those documents, in the same way: brief quoted "
                       "citations, each cited sentence checked by an AI model, unsupported sentences removed. Otherwise it "
                       "gives a reading list of documents it has not read. Such answers may be wrong. Some requests are "
                       "refused for safety reasons.")


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


def _tail(path=CONFIG_PATH, contract_path=answer_contract.CONFIG_PATH,
          chain_path=answer_contract.CHAIN_CONFIG_PATH) -> str:
    """The storage / memory sentences that follow the third-party and no-save sentences (flag-driven; shared by the
    full disclosure and the Ask-page notice so the two cannot drift)."""
    out = ""
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


def disclosure_text(path=CONFIG_PATH, contract_path=answer_contract.CONFIG_PATH,
                    chain_path=answer_contract.CHAIN_CONFIG_PATH, fulldoc_path=FULLDOC_CONFIG_PATH) -> str:
    """Base text (the full-document variant when fulldoc is enabled) (L0 always), the third-party/no-save sentences, then either the demonstration no-storage sentence
    (corpus admission off and L2 off) or the matching storage sentences. `path` is accepted for compatibility."""
    return ((base_fulldoc() if fulldoc_enabled(fulldoc_path) else _BASE) + offtopic_sentence() + third_party_text() + _NO_SAVE
            + _tail(path, contract_path, chain_path))


# UI-ABOUT-1031 / OD-1029-UI: the full disclosure is split at sentence boundaries (no wording change) so the About page
# can show it under headings and the Ask page can keep the required groups. Split points are markers inside the
# existing owner-approved constants; nothing is reworded.
_REFUSED = "Some requests are refused for safety reasons."
_FD_MARKERS = ("Answers and lists are not", "Licences for documents in its own", "For listed records, only",
               "VERA-DAR keeps no profile", "When it answers from its own")


_LANE_COUNT_OLD = "answer from up to six passages, cites them"
_LANE_COUNT_NEW = ("answer from up to six passages, plus up to six passages from public vendor or news pages it reads for "
                   "that answer, cites them")
_LANE_SENT_OLD = "the selected passages with their titles and links"
_LANE_SENT_NEW = "the selected passages (including any from public pages read for that answer) with their titles and links"


def base_fulldoc() -> str:
    """NONASK-1103 SD.1: the full-document base text; the passage-count and passages-sent sentences also cover lane
    passages (their own slots, admission.max_spans_total) only while a lane flag is on. Off: _BASE_FULLDOC exactly."""
    if not lanes_disclosure_on():
        return _BASE_FULLDOC
    return _BASE_FULLDOC.replace(_LANE_COUNT_OLD, _LANE_COUNT_NEW).replace(_LANE_SENT_OLD, _LANE_SENT_NEW)


def _fulldoc_parts() -> tuple[str, str, str, str, str, str]:
    """(how answers are produced incl. the AI-system sentence, reliability, own-collection licences, listed-record
    licences, no profile, third-party passage sentence): consecutive slices of _BASE_FULLDOC whose join is _BASE_FULLDOC."""
    base = base_fulldoc()
    idx = [base.index(m) for m in _FD_MARKERS]
    cuts = [0] + idx + [len(base)]
    return tuple(base[a:b] for a, b in zip(cuts, cuts[1:]))  # type: ignore[return-value]


def required_notice_text(path=CONFIG_PATH, contract_path=answer_contract.CONFIG_PATH,
                         chain_path=answer_contract.CHAIN_CONFIG_PATH, fulldoc_path=FULLDOC_CONFIG_PATH) -> str:
    """OD-1029-UI: the mandatory Ask-page disclosure groups, verbatim and flag-driven: AI system, reliance caveat, no
    profile, third-party AI services, refusal for safety, third-party / no-save, storage. The mechanism (how answers are
    produced), licence and off-topic-answering sentences are on the About page instead."""
    if fulldoc_enabled(fulldoc_path):
        how, reliance, _own_lic, _list_lic, profile, passages = _fulldoc_parts()
        head = how.split(" For each", 1)[0] + " " + reliance.strip() + " " + profile.strip() + " " + passages.strip()
    else:
        head = _BASE
    return (head + (" " + _REFUSED if offtopic_sentence() else "") + third_party_text() + _NO_SAVE
            + _tail(path, contract_path, chain_path))


def disclosure_sections(path=CONFIG_PATH, contract_path=answer_contract.CONFIG_PATH,
                        chain_path=answer_contract.CHAIN_CONFIG_PATH, fulldoc_path=FULLDOC_CONFIG_PATH):
    """[(heading, text)] whose texts, joined in order, are exactly disclosure_text() (headings are the only addition)."""
    off = offtopic_sentence()
    tail = third_party_text() + _NO_SAVE + _tail(path, contract_path, chain_path)
    on = lanes_disclosure_on()
    if fulldoc_enabled(fulldoc_path):
        how, reliance, own_lic, list_lic, profile, passages = _fulldoc_parts()
        secs = [("What VERA-DAR is and how it answers", how), ("Reliability", reliance),
                ("Licences", own_lic + list_lic + (_LANE_LICENCE if on else "")),
                ("Privacy and data sent to third parties", profile + passages)]
    else:
        secs = [("What VERA-DAR is and how it answers", _BASE)]
    if off:
        secs.append(("Questions outside the collection, and refused requests", off))
    secs.append(("Your question and what is stored", tail + (_LANE_GDELT_CREDIT if reporting_lane_on() else "")))
    return [(h, t.strip()) for h, t in secs]


DISCLOSURE_TEXT = disclosure_text()
DISCLOSURE_TEXT_UNCONFIRMED = DISCLOSURE_TEXT  # identical while evidence recall is off


# R-031 / H8: the one short line shown on every page except About; the full disclosure lives on About only (N1: the
# third-party clause is settled and stays in the short line).
SHORT_NOTICE = ("VERA-DAR is an AI system and its answers may be wrong. Your question is sent to third-party AI and "
                "search services.")

def short_notice_text() -> str:
    """SHORT_NOTICE, plus the lane sentences while a lane flag is on (SD.1)."""
    return SHORT_NOTICE + lane_third_party()


ABOUT_LINKS = (("pages/about.py", "About: how VERA-DAR answers, licences, data"),
               ("pages/how_to_use.py", "How to use VERA-DAR"))


def render_disclosure(st) -> None:
    """R-031: the one short notice, then links to the About and How to use pages. The full disclosure is on About only."""
    st.caption(short_notice_text())
    for target, label in ABOUT_LINKS:
        try:
            st.page_link(target, label=label)
        except Exception:  # noqa: BLE001  (StreamlitPageNotFoundError outside a multipage run, e.g. AppTest of a pages/ file)
            pass
