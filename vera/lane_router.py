"""SCOPE-GATE-1104 S4 / OFFICIAL-LANE-S4A: per-sub-query lane selection (additive; never a refusal).

No subject list. A lane is ADDED by a signal that is not a topic:
  official   the part names an ACTIVE entity-registry row (owner-curated data; no host is ever guessed)
  reporting  the part carries time/news language (vera.temporality lexicon cue, or vera.news_intent);
             a volatile registry entity alone does NOT add reporting (gates.md G11): it needs a time cue too
  research   always (scholarly providers, full-text acquisition)
  web        always when inline.web_read_enabled (general web route, answer-only)
No lane is ever removed by a signal, and no route can decline: the plan only adds evidence sources. Whether the
evidence is sufficient is decided AFTER retrieval (SCOPE-GATE-1104 A1.2), never here.

Pure: no network, DB or model call. The trace carries lane names and closed basis tokens only, never question text.
Switch: config/auto_acquire.json lane_router.enabled (ships false). Off -> plan() returns None and callers keep
today's behaviour exactly. reporting_cue() is the G11 rule for vera.lane_evidence.collect and applies whatever the
switch says: a vendor question reaches the news/GDELT lane only on a time or news cue, never because its registry row
is volatile.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LANES = ("research", "web", "official", "reporting")
BASIS = ("always", "web_route", "registry", "time_cue", "news_cue", "volatile_with_time_cue")


@dataclass(frozen=True)
class PartRoute:
    label: str
    lanes: tuple            # subset of LANES, in LANES order
    basis: tuple            # closed tokens from BASIS
    entities: tuple = ()    # registry entity names (data, not user text)


@dataclass(frozen=True)
class LanePlan:
    parts: tuple
    run_on_sufficient: tuple = field(default_factory=tuple)  # lanes that run even when the corpus is sufficient

    def lanes(self) -> set:
        return {ln for p in self.parts for ln in p.lanes}

    def trace(self) -> dict:
        """Closed tokens and counts only."""
        return {"n_parts": len(self.parts), "lanes": sorted(self.lanes()),
                "basis": sorted({b for p in self.parts for b in p.basis})}


def _block(acq) -> dict:
    b = (acq or {}).get("lane_router") if isinstance(acq, dict) else None
    return b if isinstance(b, dict) else {}


def enabled(acq) -> bool:
    return _block(acq).get("enabled") is True


def _time_cue(text: str, temporality, cfg_t=None) -> bool:
    # volatile_entity=False: a vendor name alone is not a time cue (G11)
    return bool(temporality.detect(text, volatile_entity=False, cfg=cfg_t).current)


def _news_cue(text: str, news_intent) -> bool:
    try:
        return bool(news_intent is not None and news_intent.news_intent(text))
    except Exception:  # noqa: BLE001  a failing detector adds nothing
        return False


def reporting_cue(part_text: str, question: str = "", *, temporality=None, news_intent=None, cfg_t=None) -> bool:
    """G11: True only on a time cue in the part (temporality lexicon, volatile_entity=False) or news intent in the part
    or the whole question. A volatile registry entity alone is never a cue. Never raises (False on error)."""
    try:
        if temporality is None:
            from vera import temporality as _t
            temporality = _t
        if news_intent is None:
            try:
                from vera import news_intent as _n
                news_intent = _n
            except ImportError:
                news_intent = None
        return (_time_cue(part_text or "", temporality, cfg_t) or _news_cue(part_text or "", news_intent)
                or (bool(question) and _news_cue(question, news_intent)))
    except Exception:  # noqa: BLE001
        return False


def route_part(part, *, registry_active: set, web_on: bool, temporality, news_intent=None, question: str = "",
               cfg_t=None) -> PartRoute:
    lanes, basis = ["research"], ["always"]
    if web_on:
        lanes.append("web")
        basis.append("web_route")
    ents = tuple(e for e in (getattr(part, "entities", ()) or ()) if e in registry_active)
    if ents:
        lanes.append("official")
        basis.append("registry")
    text = getattr(part, "text", "") or ""
    t = _time_cue(text, temporality, cfg_t)
    n = _news_cue(text, news_intent) or (bool(question) and _news_cue(question, news_intent))
    if t or n:
        lanes.append("reporting")
        basis.append("time_cue" if t else "news_cue")
        if ents and t:
            basis.append("volatile_with_time_cue")
    order = {ln: i for i, ln in enumerate(LANES)}
    return PartRoute(getattr(part, "label", "P1"), tuple(sorted(dict.fromkeys(lanes), key=order.get)),
                     tuple(dict.fromkeys(basis)), ents)


def plan(question: str, acq: dict | None, *, subq_plan=None, registry=None, web_on: bool | None = None,
         temporality=None, news_intent=None) -> LanePlan | None:
    """None when the switch is off or on any error (callers keep today's path). Never raises."""
    try:
        if not enabled(acq):
            return None
        if temporality is None:
            from vera import temporality as _t
            temporality = _t
        if news_intent is None:
            try:
                from vera import news_intent as _n
                news_intent = _n
            except ImportError:
                news_intent = None
        from vera import subquery
        reg = registry if registry is not None else subquery.load_entity_registry()
        active = {r["entity"] for r in reg if isinstance(r, dict) and r.get("state") == "active"}
        sp = subq_plan or subquery.decompose(question, registry=reg)
        parts = list(sp.parts) or [subquery.SubPart("P1", question)]
        if web_on is None:
            web_on = (((acq or {}).get("inline") or {}).get("web_read_enabled") is True)
        routes = tuple(route_part(p, registry_active=active, web_on=web_on, temporality=temporality,
                                  news_intent=news_intent, question=question) for p in parts)
        ros = tuple(x for x in (_block(acq).get("run_on_sufficient") or ()) if x in ("official", "reporting"))
        return LanePlan(routes, ros)
    except Exception:  # noqa: BLE001
        return None
