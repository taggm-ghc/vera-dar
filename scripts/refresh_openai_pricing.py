#!/usr/bin/env python3
"""Automated refresh of Standard/Flex pricing for non-flagship (single-rate,
no short/long context split) OpenAI models, extracted from the live page's
own TextTokenPricingTables data component.

This REPLACES anti-pattern/refresh_openai_pricing.py (deprecated 2026-09-12
— see pricing-scraper-design-prv.md). That script's fallback regex was
proven systemically unsafe: it substring-collided on model names
(gpt-4o matching inside gpt-4o-mini) and confused cached-input for output
whenever 3+ price columns appeared in sequence. This script avoids both
failure modes structurally, not by patching the old heuristics:

  - Each tier's data is a SEPARATE component instance that explicitly
    labels its own tier in its props (`"tier":[0,"standard"]`) — the tier
    is read, never inferred from document order or a position assumption.
  - Every model is a distinct JSON object key inside that instance's own
    "rows" array — there is no proximity-based text matching and therefore
    no substring-collision risk, and no "grab the nearest two dollar
    amounts" column confusion, because each value's role is a fixed
    position within a well-defined tuple, not a guess.

Deliberately out of scope (skipped, not guessed):
  - Flagship-family models with a short/long context split (gpt-6-astra,
    gpt-5.6-*, gpt-5.5*, gpt-5.4*) — they appear in this same data
    component too, but only as a single (short-context) value. Their
    authoritative short+long records already exist in
    config/model-pricing.json from manual transcription of the dedicated
    Flagship table; re-deriving them here under context_length=None would
    create a confusing, differently-tagged duplicate of already-correct
    data, not new information.
  - Batch and Fast mode tiers — out of scope per this project's standing
    scope constraint (see pricing-scraper-design-prv.md): strictly
    OpenAI's own page, Standard + Flex only.

Fails loudly (nonzero exit, nothing written) if the expected component or
tier labels aren't found — never guesses or falls back to a weaker
heuristic. This is a refresh/import step, run manually or on a schedule; it
is never invoked by main.py at request time (same Resource-invariant
boundary as its predecessor).

Usage:
    python scripts/refresh_openai_pricing.py [--dry-run]
"""
import argparse
import datetime as dt
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vera.pricing.config import (  # noqa: E402
    ModelPricing,
    PricingRecord,
    append_pricing_record,
    load_model_pricing,
)

PRICING_URL = "https://developers.openai.com/api/docs/pricing"
PROVIDER = "openai"
COMPONENT_MARKER = "TextTokenPricingTables"
ACCEPTED_TIERS = {"standard", "flex"}

FLAGSHIP_FAMILY = {
    "gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
    "gpt-5.5", "gpt-5.5-pro", "gpt-5.4", "gpt-5.4-pro",
}


def fetch_page(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "vera-pricing-refresh/2.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _extract_balanced(data: str, start: int) -> str:
    assert data[start] == "[", f"expected '[' at {start}, got {data[start:start + 20]!r}"
    depth = 0
    for i in range(start, len(data)):
        if data[i] == "[":
            depth += 1
        elif data[i] == "]":
            depth -= 1
            if depth == 0:
                return data[start:i + 1]
    raise ValueError("unbalanced brackets while extracting rows blob")


def find_component_instances(html_text: str):
    """Yield (tier, rows_dict) for every TextTokenPricingTables instance on
    the page, with tier read explicitly from that instance's own props —
    never inferred from document order or position."""
    for m in re.finditer(re.escape(COMPONENT_MARKER), html_text):
        pos = m.start()
        window_start = max(0, pos - 6000)
        window = html_text[window_start:pos]

        tier_match = re.search(r'&quot;tier&quot;:\[0,&quot;([a-z]+)&quot;\]', window)
        if not tier_match:
            continue  # not a component instance's own props — skip, don't guess
        tier = tier_match.group(1)

        rows_idx = window.rfind('&quot;rows&quot;:[')
        if rows_idx == -1:
            continue
        blob_start = window_start + rows_idx + len('&quot;rows&quot;:')
        raw = _extract_balanced(html_text, blob_start)
        unescaped = html.unescape(raw).replace('\\"', '"')
        try:
            parsed = json.loads(unescaped)
        except json.JSONDecodeError:
            continue

        rows = {}
        for row in parsed[1]:
            cells = row[1]
            name = cells[0][1]
            values = [c[1] for c in cells[1:]]
            rows[name] = values
        yield tier, rows


def _to_price(value):
    return None if value in ("-", None) else float(value)


def latest_for_model_and_tier(model: str, tier: str, pricing: ModelPricing | None) -> PricingRecord | None:
    """Like vera.pricing.config.latest_pricing_for, but tier-aware. That helper
    ignores service_tier entirely (it was written for main.py's single-tier
    lookup), so reusing it here would compare a model's Standard price
    against whichever tier happens to have the latest effective_from
    overall — wrong when Standard and Flex share the same date, as ours do.
    Found via a failed --dry-run before this script was ever used for real,
    not assumed correct."""
    if pricing is None:
        return None
    matches = [
        r for r in pricing.records
        if r.model == model and r.service_tier == tier and r.context_length is None
    ]
    return max(matches, key=lambda r: r.effective_from) if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Fetch and parse, but do not write config/model-pricing.json")
    args = parser.parse_args()

    try:
        page = fetch_page(PRICING_URL)
    except Exception as exc:
        print(f"FAILED to fetch {PRICING_URL}: {exc}", file=sys.stderr)
        sys.exit(1)

    instances = list(find_component_instances(page))
    if not instances:
        print(
            f"FAILED to find any {COMPONENT_MARKER} instance with an explicit "
            "tier label on the page. Page structure may have changed; no "
            "price was guessed or reused from a stale record.",
            file=sys.stderr,
        )
        sys.exit(1)

    found_tiers = {tier for tier, _ in instances}
    missing = ACCEPTED_TIERS - found_tiers
    if missing:
        print(
            f"FAILED: expected tiers {sorted(ACCEPTED_TIERS)} but only found "
            f"{sorted(found_tiers)}. Page structure may have changed; "
            "refusing to proceed with partial tier coverage.",
            file=sys.stderr,
        )
        sys.exit(1)

    retrieved_at = dt.datetime.now(dt.timezone.utc).isoformat()
    effective_from = dt.date.today().isoformat()
    existing = load_model_pricing()

    appended = 0
    skipped_unchanged = 0
    skipped_flagship = 0
    skipped_other_shape = 0
    for tier, rows in instances:
        if tier not in ACCEPTED_TIERS:
            continue
        for model, values in rows.items():
            if model in FLAGSHIP_FAMILY:
                skipped_flagship += 1
                continue
            if len(values) != 3:
                # Not the simple (input, cached, output) shape this script
                # is scoped to — skip rather than guess at a different layout.
                skipped_other_shape += 1
                continue

            input_price = _to_price(values[0])
            cached_price = _to_price(values[1])
            output_price = _to_price(values[2])
            if input_price is None or output_price is None:
                continue

            latest = latest_for_model_and_tier(model, tier, existing)
            if (
                latest
                and latest.input == input_price
                and latest.cached_input == cached_price
                and latest.output == output_price
            ):
                skipped_unchanged += 1
                continue

            record = PricingRecord(
                provider=PROVIDER,
                model=model,
                service_tier=tier,
                context_length=None,
                input=input_price,
                cached_input=cached_price,
                cache_writes=None,
                output=output_price,
                source_url=PRICING_URL,
                retrieved_at=retrieved_at,
                effective_from=effective_from,
                notes=(
                    f"Automated refresh via {COMPONENT_MARKER} component "
                    "(explicit tier label read from its own props, not "
                    "inferred by position or matched by regex)."
                ),
            )
            if args.dry_run:
                print(f"Would append: {record.model_dump_json()}")
            elif append_pricing_record(record):
                appended += 1

    print(
        f"Done. tiers found: {sorted(found_tiers)}; appended={appended}, "
        f"unchanged(skipped)={skipped_unchanged}, "
        f"flagship-family(skipped, handled elsewhere)={skipped_flagship}, "
        f"other-shape(skipped)={skipped_other_shape}"
    )


if __name__ == "__main__":
    main()
