"""In-process ledger of billable calls made during one M2 pass.

M6 eval sums these. The runner persists the entries as JSON on the
search_iterations row, so cost survives the process without needing a
separate table (a durable queryable cost table is a deferred item).
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class CostEntry:
    kind: str  # "search" | "llm_gate_a" | "fetch"
    provider: str  # "arxiv", "openai:gpt-4.1-nano", "http"
    cost_usd: float
    cost_known: bool = True  # False when no price was available (cost_usd is 0, not free)
    units: dict = field(default_factory=dict)  # e.g. {"prompt_tokens": 10, "completion_tokens": 5}
    ok: bool = True
    detail: str = ""
    cost_basis: str = ""  # why cost_usd is what it is, e.g. "no_published_price_keyless"; "" = not stated
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CostLedger:
    def __init__(self) -> None:
        self.entries: list[CostEntry] = []

    def record(self, **kw) -> CostEntry:
        entry = CostEntry(**kw)
        self.entries.append(entry)
        return entry

    def total_usd(self) -> float:
        return round(sum(e.cost_usd for e in self.entries), 6)

    def unknown_cost_calls(self) -> int:
        return sum(1 for e in self.entries if not e.cost_known)

    def as_dicts(self) -> list[dict]:
        return [asdict(e) for e in self.entries]
