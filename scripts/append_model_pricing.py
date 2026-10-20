#!/usr/bin/env python3
"""Append a new dated pricing record to config/model-pricing.json.

Records are append-only: this script never edits or removes an existing
entry, only adds a new one. main.py always uses the most recent record (by
effective_from) for the currently selected model, so re-running this whenever
a provider changes pricing is sufficient — no other file needs to change.

Schema and the append-only writer live in vera/pricing/config.py (shared with
main.py) so this script and main.py stay in sync. (The former third sharer,
scripts/refresh_openai_pricing.py, was deprecated 2026-09-12 and moved to
anti-pattern/refresh_openai_pricing.py — gitignored, not part of the shipped
tool set — after live testing proved its parsing strategies unsound. This
script, run by a human reading the pricing page, is now the sole way
config/model-pricing.json gets updated. See pricing-scraper-design-prv.md.)

Usage:
    python scripts/append_model_pricing.py \\
        --provider openai --model gpt-4o-mini \\
        --input-per-million 0.15 --output-per-million 0.60 \\
        --source-url "https://developers.openai.com/api/docs/pricing" \\
        --notes "optional context"

    # Flagship-model tables publish separate short/long-context rates plus a
    # cache-writes price; pass --context-length and --cache-writes-per-million
    # for those, and call once per context length:
    python scripts/append_model_pricing.py \\
        --provider openai --model gpt-6-astra --context-length short \\
        --input-per-million 10.00 --cached-input-per-million 1.00 \\
        --cache-writes-per-million 12.50 --output-per-million 50.00 \\
        --source-url "https://developers.openai.com/api/docs/pricing"
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vera.pricing.config import MODEL_PRICING_PATH, PricingRecord, append_pricing_record  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", required=True, help="Model identifier, e.g. gpt-4o-mini")
    parser.add_argument("--service-tier", default="standard")
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--unit", default="per_1m_tokens")
    parser.add_argument("--context-length", choices=["short", "long"], default=None,
                         help="Omit for models that publish a single price regardless of context length")
    parser.add_argument("--input-per-million", type=float, required=True)
    parser.add_argument("--cached-input-per-million", type=float, default=None)
    parser.add_argument("--cache-writes-per-million", type=float, default=None)
    parser.add_argument("--output-per-million", type=float, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--retrieved-at", default=dt.datetime.now(dt.timezone.utc).isoformat())
    parser.add_argument("--effective-from", default=dt.date.today().isoformat())
    args = parser.parse_args()

    record = PricingRecord(
        provider=args.provider,
        model=args.model,
        service_tier=args.service_tier,
        context_length=args.context_length,
        currency=args.currency,
        unit=args.unit,
        input=args.input_per_million,
        cached_input=args.cached_input_per_million,
        cache_writes=args.cache_writes_per_million,
        output=args.output_per_million,
        source_url=args.source_url,
        retrieved_at=args.retrieved_at,
        effective_from=args.effective_from,
        notes=args.notes,
    )

    if append_pricing_record(record):
        print(f"Appended pricing record for {args.model} dated {args.effective_from} to {MODEL_PRICING_PATH}.")
    else:
        print("Identical record already present; not appending a duplicate.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
