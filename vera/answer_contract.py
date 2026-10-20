"""K2 answer contract (plan-K2-contract.md r2, leaves K2.I0/K2.I1).

Two result families: an evidentiary answer (State A, the L3 seam) and a discovery result (State B: reading list).
Abstracts, snippets and metadata are never evidence (design DEC-FDA1): `examined_full_documents` is non-empty only for
the D-055 full-document path (verified documents inspected, passages admitted), so nothing else can be an "answer".

Switches and texts live in config/discovery_contract.json (no hard-coded data). This module never imports the L1
modules (vera/discovery.py, vera/discovery_leak_check.py): L0 must work when they are absent (D5).
"""
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "discovery_contract.json"
CHAIN_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "ask-provider-chain.json"

RESULT_KINDS = ("answer", "discovery", "reading_list", "decline", "unspecified")
NON_ANSWER_KINDS = tuple(k for k in RESULT_KINDS if k != "answer")
DEFAULT_KIND = "unspecified"  # V2: the default is a non-answer
ACQ_NOT_ACQUIRED, ACQ_ACQUIRED = "not_acquired", "full_document_acquired"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """The contract config; {} when missing or unreadable (callers then fall back to the safest layer, L0)."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def layers(path: Path = CONFIG_PATH) -> list[str]:
    """Enabled layers, from config. L0 is the floor and is always present; unknown names are dropped."""
    raw = load_config(path).get("layers")
    known = ("L0", "L1", "L2", "L3")
    got = [x for x in raw if x in known] if isinstance(raw, list) else []
    return ["L0"] + [x for x in dict.fromkeys(got) if x != "L0"]


def text(key: str, path: Path = CONFIG_PATH) -> str:
    """A text from config `texts`; '' if absent (tests pin the shipped values)."""
    return str((load_config(path).get("texts") or {}).get(key) or "")


def evidence_recall_enabled(path: Path = CONFIG_PATH) -> bool:
    """V3: fail-closed switch for recalling evidence findings; true only if explicitly set to true."""
    return (load_config(path).get("memory") or {}).get("evidence_recall_enabled") is True


def write_on_l0(path: Path = CONFIG_PATH) -> bool:
    return (load_config(path).get("discovery_memory") or {}).get("write_on_l0") is True


def no_sources_text(path: Path = CONFIG_PATH, chain_path: Path = CHAIN_CONFIG_PATH) -> str:
    """The L0 empty-result text: the existing grounding no_sources_text if configured (read even if grounding is
    disabled, since grounding-None must never fall to the bare model), else the contract fallback."""
    try:
        t = (json.loads(Path(chain_path).read_text()).get("grounding") or {}).get("no_sources_text")
    except (OSError, ValueError, AttributeError):
        t = None
    return str(t) if t else text("no_sources_fallback", path)


def acquisition_state(candidate: dict, path: Path = CONFIG_PATH) -> str:
    """Per-candidate acquisition (D2). `full_document_acquired` only for a stored source whose text kind is outside
    the abstract/metadata kinds AND that passed admission. `fetch_status` is never the signal (known breach)."""
    non_doc = {str(k) for k in ((load_config(path).get("acquisition") or {}).get("non_document_text_kinds") or [])}
    kind = candidate.get("stored_text_kind")
    if kind and str(kind) not in non_doc and candidate.get("admission_passed") is True:
        return ACQ_ACQUIRED
    return ACQ_NOT_ACQUIRED


def representation(state: str, path: Path = CONFIG_PATH) -> str:
    return text("representation_acquired" if state == ACQ_ACQUIRED else "representation_not_acquired", path)


def not_read_line(candidates, path: Path = CONFIG_PATH) -> str:
    """The ratified fixed disclosure; the partial variant when at least one but not all candidates are acquired."""
    states = [c.get("acquisition") for c in candidates or ()]
    if ACQ_ACQUIRED in states and ACQ_NOT_ACQUIRED in states:
        return text("not_read_line_partial", path)
    return text("not_read_line", path)


def examined_full_documents(candidates) -> list:
    """L3 seam (state A), D-055 Option C. Non-empty ONLY for a vera.fulldoc_evidence.EvidenceBundle (a frozen object
    built by vera.fulldoc_admission.admit, never a dict) that is grounded: verified full documents were inspected AND at
    least one of their passages was admitted. Dict candidates (web or stored) never count. It reads no config and no
    environment: neither can make it non-empty."""
    from vera.fulldoc_evidence import EvidenceBundle

    return [d for c in candidates or () if isinstance(c, EvidenceBundle) and c.grounded
            for d in c.examined_full_documents]


def result_kind_for(candidates) -> str:
    """The single dispatch point: "answer" only if full documents were examined and passages admitted, else a reading
    list."""
    return "answer" if examined_full_documents(candidates) else "reading_list"


def reading_list_result(candidates, tokens_used: int, cost_usd: float, *, sources: tuple = (), accepted: bool = True,
                        fallback=None, empty_text: str | None = None, path: Path = CONFIG_PATH):
    """L0 result: fixed non-empty `answer` text (so answer_via_chain does not treat it as empty), no claims, no claim
    check. `sources` keeps the internal retrieved candidates so corpus admission still runs (V1)."""
    from vera.ask_service import AskResult

    cands = tuple(candidates or ())
    return AskResult(answer=not_read_line(cands, path) if cands else (empty_text or no_sources_text(path)),
                     tokens_used=tokens_used, cost_usd=cost_usd, sources=tuple(sources), claim_check=None,
                     accepted=accepted, claims=(), result_kind="reading_list", candidates=cands,
                     contract={"layers": layers(path), "fallback": fallback})
