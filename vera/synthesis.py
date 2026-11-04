"""NONASK-1103 L3: synthesis pass over verified claims plus admitted spans. FLAG-OFF (config/synthesis.json `enabled`).

NOT wired into ask_service, fulldoc_answer or streamlit_app; wiring follows calibration (evidence-policy plan L3,
owner-flip-packet "L3 synthesis"). Invariants (evidence-policy plan section 7, KE-7):

  I1  A verification status is bound to the EXACT sentence text plus its cited span ids. A synthesised sentence whose
      (text, cites) pair equals a verified claim keeps that claim's status; any other sentence (new, reworded, or the
      same words citing different spans) is `unevaluated` and is shown only if the injected checker passes it. Nothing
      inherits status.
  I2  Re-checks see only the admitted span text of the spans the sentence cites (claim-check source shape
      {"n", "snippet"[, "title"]}); never the synthesiser's draft, rationale or other claims.
  F6  A previously rejected claim (same proposition after normalising case, spacing, citation markers and final
      punctuation, alone or contained in a longer sentence) is blocked before any check and is never reintroduced,
      whatever its citations.
  I4  Flag off: the baseline text is returned unchanged (same object), with zero synthesiser and zero checker calls.

Fails closed to the baseline (deterministic assembly, option A) when the synthesiser or checker is missing or fails,
or when no sentence survives. Only "supported" verdicts are shown (stricter than the A path; partial is dropped).
The synthesiser and checker are injected (no model client is chosen here; when wired, the call must use the free-tier,
lowest-cost model per owner standing rules). Logs codes and counts only, never question, claim or span text.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable

from vera import claim_check

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "synthesis.json"
SUPPORTED = "supported"
_CITE_MARK = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


def load_cfg(path: Path | None = None) -> dict:
    try:
        cfg = json.loads((path or CONFIG_PATH).read_text(encoding="utf-8"))
        return cfg if isinstance(cfg, dict) else {"enabled": False}
    except Exception:  # noqa: BLE001  unreadable config = off
        return {"enabled": False}


def enabled(cfg: dict | None = None) -> bool:
    return (cfg if cfg is not None else load_cfg()).get("enabled") is True


def status_key(text: str, cites) -> tuple:
    """I1 binding key: the exact (stripped) sentence text plus its sorted cited span ids."""
    return (str(text).strip(), tuple(sorted(int(c) for c in cites or ())))


def proposition_key(text: str) -> str:
    """F6 block key: the proposition with citation markers, case, spacing and final punctuation normalised away."""
    t = _CITE_MARK.sub("", str(text or ""))
    t = " ".join(t.lower().split())
    return t.rstrip(" .!?;:")


def judge_checker(cc_cfg: dict, call=None) -> Callable:
    """The existing claim-check interface (claim_check.judge) as an injectable checker(claims, sources)."""
    def check(claims, sources):
        return claim_check.judge(claims, sources, cc_cfg, call)
    return check


def synthesise(baseline: str, verified_claims: list, admitted_spans: list, *, rejected_claims=(),
               synthesizer: Callable | None = None, checker: Callable | None = None, cfg: dict | None = None,
               max_claims: int | None = None) -> tuple:
    """(text, record).

    baseline         the deterministic assembly (option A) text, returned unchanged on flag off or fail-closed.
    verified_claims  [{"text", "cites", "verdict"}] from the claim check; only verdict "supported" carries status.
    admitted_spans   [{"n", "snippet"[, "title"]}] admitted span text, the only re-check evidence (I2).
    rejected_claims  [{"text", ...}] claims the check rejected; never reintroduced (F6).
    synthesizer      callable(verified_claims, admitted_spans) -> draft text with [n] citations (one call).
    checker          callable(claims, sources) -> (verdicts by claim id, label), e.g. judge_checker(cc_cfg).
    """
    cfg = cfg if cfg is not None else load_cfg()
    if not enabled(cfg):
        return baseline, {"enabled": False, "outcome": "disabled"}
    rec = {"enabled": True, "outcome": None, "kept_exact": 0, "rechecked": 0, "recheck_passed": 0,
           "dropped_unsupported": 0, "blocked_rejected": 0, "dropped_uncited": 0, "dropped_bad_span": 0,
           "sentences": []}

    def closed(outcome):
        rec["outcome"] = outcome
        logger.info("synthesis outcome=%s", outcome)
        return baseline, rec

    if synthesizer is None or checker is None:
        return closed("fail_closed_no_dependency")
    spans = {int(s["n"]): s for s in admitted_spans or () if isinstance(s, dict) and "n" in s}
    status = {status_key(c["text"], c.get("cites")): SUPPORTED for c in verified_claims or ()
              if isinstance(c, dict) and c.get("verdict") == SUPPORTED}
    blocked = {proposition_key(c.get("text", "")) for c in rejected_claims or () if isinstance(c, dict)}
    blocked.discard("")
    try:
        draft = synthesizer(list(verified_claims or ()), list(spans.values()))
    except Exception:  # noqa: BLE001
        return closed("fail_closed_synthesizer_error")
    if not isinstance(draft, str) or not draft.strip():
        return closed("fail_closed_empty_draft")
    limit = int(max_claims if max_claims is not None else cfg.get("max_claims", 1000))
    # every cited number is kept here so a sentence citing a non-admitted span is dropped, not silently re-cited
    cited = claim_check.split_claims(draft, 10 ** 6, limit)
    n_parts = sum(1 for p in claim_check._SENT.split(draft) if p.strip())
    rec["dropped_uncited"] = max(0, n_parts - len(cited))
    shown, to_check = [], []
    for c in cited:
        if any(n not in spans for n in c["cites"]):
            rec["dropped_bad_span"] += 1
            continue
        pk = proposition_key(c["text"])
        if any(b == pk or b in pk for b in blocked):  # also catches a rejected claim glued into a longer sentence
            rec["blocked_rejected"] += 1
            rec["sentences"].append({"key": status_key(c["text"], c["cites"]), "status": "blocked_rejected"})
            continue
        key = status_key(c["text"], c["cites"])
        if status.get(key) == SUPPORTED:
            rec["kept_exact"] += 1
            rec["sentences"].append({"key": key, "status": SUPPORTED, "source": "exact_verified"})
            shown.append((c["id"], c["text"]))
        else:
            to_check.append(c)
    if to_check:
        claims = [{"id": i, "text": c["text"], "cites": list(c["cites"])} for i, c in enumerate(to_check)]
        sources = [{k: spans[n][k] for k in ("n", "snippet", "title") if k in spans[n]}
                   for n in sorted({n for c in claims for n in c["cites"]})]
        rec["rechecked"] = len(claims)
        try:
            verdicts, _label = checker(claims, sources)
        except Exception:  # noqa: BLE001
            verdicts = None
        if not isinstance(verdicts, dict):
            return closed("fail_closed_checker_unavailable")
        for cl, c in zip(claims, to_check):
            v = (verdicts.get(cl["id"]) or {}).get("verdict", "unevaluated")
            key = status_key(c["text"], c["cites"])
            rec["sentences"].append({"key": key, "status": v, "source": "rechecked"})
            if v == SUPPORTED:
                rec["recheck_passed"] += 1
                shown.append((c["id"], c["text"]))
            else:
                rec["dropped_unsupported"] += 1
    if not shown:
        return closed("fail_closed_nothing_shown")
    text = " ".join(t for _i, t in sorted(shown))
    rec["outcome"] = "synthesised"
    logger.info("synthesis outcome=synthesised kept_exact=%d rechecked=%d passed=%d blocked=%d",
                rec["kept_exact"], rec["rechecked"], rec["recheck_passed"], rec["blocked_rejected"])
    return text, rec
