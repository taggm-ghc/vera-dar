"""Model selection and pricing configuration: shared schema and loaders used by
main.py (the coordinator) and the scripts/ pricing-maintenance tools
(specialized sub-scripts), so the schema is defined once instead of duplicated
across files.

Config is version-controlled JSON, not a database. See the Resource invariant
in docs/vera-design.md: Render's filesystem is ephemeral, so anything written
there at runtime would not survive a redeploy, while Git-tracked JSON is
restored on every deploy, diffable, and human-reviewable.
"""
import json
import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger("vera")

# This file lives at <repo>/vera/pricing/config.py, three levels below the
# repo root, so CONFIG_DIR must climb back up to <repo>/config/ rather than
# assuming a sibling directory.
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
MODEL_SELECTION_PATH = CONFIG_DIR / "model-selection.json"
MODEL_PRICING_PATH = CONFIG_DIR / "model-pricing.json"


class ModelSelection(BaseModel):
    selected_model: str
    selected_at: str
    rationale: str


class PricingRecord(BaseModel):
    provider: str
    model: str
    service_tier: str = "standard"
    # None means the model has a single price regardless of context length
    # (e.g. legacy models like gpt-4o-mini); "short"/"long" for models that
    # publish separate short- and long-context rates.
    context_length: str | None = None
    currency: str = "USD"
    unit: str = "per_1m_tokens"
    input: float
    cached_input: float | None = None
    cache_writes: float | None = None
    output: float
    source_url: str
    retrieved_at: str  # when this record was scraped/entered (ISO datetime)
    effective_from: str  # date the price is believed to have taken effect
    notes: str = ""


class ModelPricing(BaseModel):
    records: list[PricingRecord]


def _load_json_config(path: Path, model_cls):
    try:
        return model_cls(**json.loads(path.read_text()))
    except Exception:
        logger.exception("Failed to load or validate %s", path)
        return None


def load_model_selection() -> ModelSelection | None:
    return _load_json_config(MODEL_SELECTION_PATH, ModelSelection)


def load_model_pricing() -> ModelPricing | None:
    return _load_json_config(MODEL_PRICING_PATH, ModelPricing)


def latest_pricing_for(model: str, pricing: ModelPricing | None) -> PricingRecord | None:
    if pricing is None:
        return None
    matches = [r for r in pricing.records if r.model == model]
    return max(matches, key=lambda r: r.effective_from) if matches else None


def append_pricing_record(record: PricingRecord, path: Path = MODEL_PRICING_PATH) -> bool:
    """Append a validated pricing record unless a byte-identical one already
    exists. Never edits or removes an existing record — see the Provenance
    invariant in docs/vera-design.md."""
    data = json.loads(path.read_text()) if path.exists() else {"records": []}
    record_dict = record.model_dump()
    if record_dict in data["records"]:
        return False
    data["records"].append(record_dict)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return True
