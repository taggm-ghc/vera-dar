#!/usr/bin/env python3
"""Capture visual-verification screenshots of OpenAI's live pricing page
(Standard + Flex tiers, full expanded model table) for a human to glance at.

This is NOT a scraping/parsing script and does not write to
config/model-pricing.json — it produces PNGs for a human to review, per
pricing-scraper-design-prv.md's desired outcome #6 ("cheap human
verification... make that glance take seconds, not require reading raw
HTML"). Never invoked by main.py or any deployed code path.

Design, flow diagram, and pseudocode: pricing-scraper-design-prv.md,
"Capture-script flow and pseudocode" section. The mechanism below was
verified live before being written here, not assumed: the Flagship hero
table and the full "additional models" list share one segmented control and
one scoped container (confirmed 2026-09-12 — see that doc's "Completeness
pass" section for how this was established, including the two false starts
recorded there so they aren't repeated).

Requires Playwright + Chromium (see requirements-dev.txt; not in requirements.txt).

Usage:
    python scripts/capture_openai_pricing.py [--out-dir captures]
"""
import argparse
import time
from pathlib import Path

from playwright.sync_api import Locator, sync_playwright

PRICING_URL = "https://developers.openai.com/api/docs/pricing"
# Present in every tier; used only to confirm a tier switch actually took
# visible effect, not as a scoping anchor for the whole table.
ANCHOR_MODEL = "gpt-6-astra"
# Scope per pricing-scraper-design-prv.md's standing constraint: strictly
# OpenAI's own page, Standard + Flex only — Batch/Fast mode not requested.
TIERS = ["standard", "flex"]


def wait_until(condition_fn, timeout_ms: int = 5000, poll_ms: int = 100, label: str = "") -> None:
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        if condition_fn():
            return
        time.sleep(poll_ms / 1000)
    raise TimeoutError(f"wait_until({label}) timed out after {timeout_ms}ms")


def make_timestamped_output_dir(base: Path) -> Path:
    out = base / time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    out.mkdir(parents=True, exist_ok=True)
    return out


def anchor_row_visible(section: Locator) -> bool:
    rows = section.locator("tr", has_text=ANCHOR_MODEL)
    visible = sum(1 for i in range(rows.count()) if rows.nth(i).is_visible())
    return visible == 1


def scroll_table_fully_right(section: Locator) -> None:
    """The rendered table sits inside its own `overflow-x:auto` wrapper,
    independent of the outer page/section width — confirmed by trial:
    widening the browser viewport did not reduce the Long-context columns'
    cropping, but scrolling this specific inner container to its max
    scrollLeft did. Silently does nothing if no such wrapper is found
    (e.g. narrower tables that don't need one), rather than failing."""
    scroller = section.locator("div[style*='overflow-x']").first
    if scroller.count() > 0:
        scroller.evaluate("el => { el.scrollLeft = el.scrollWidth; }")


def capture_tier(section: Locator, tier: str, out_dir: Path) -> Path:
    radio = section.locator(f'button[role="radio"][data-value="{tier}"]')
    radio.click()

    # Verify the *visible* state changed, not just aria-checked — the page
    # pre-renders every tier's rows simultaneously and toggles CSS
    # visibility, so aria-checked flipping alone doesn't prove the right
    # rows are now showing.
    wait_until(lambda: radio.get_attribute("aria-checked") == "true", label=f"{tier}-checked")
    wait_until(lambda: anchor_row_visible(section), label=f"{tier}-visible")

    scroll_table_fully_right(section)

    out_path = out_dir / f"pricing_{tier}.png"
    section.screenshot(path=out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="captures", help="Base directory for timestamped screenshot output")
    args = parser.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            # Height must exceed the full expanded table's rendered height
            # (~1975px observed) — a shorter viewport forces Playwright to
            # scroll-stitch the element screenshot, and some rows render as
            # blank placeholders until actually scrolled into view first.
            # Widening the viewport does NOT fix the long-context-column
            # cropping below (confirmed by trial: 2000px-wide viewport
            # produced an identical crop to 1600px) — the real cause is a
            # separate inner `overflow-x:auto` wrapper around the table
            # itself, handled explicitly further down, not by viewport size.
            page = browser.new_page(viewport={"width": 1600, "height": 2200})
            page.goto(PRICING_URL, wait_until="networkidle")
            # Every live test during design/verification needed this settle
            # delay after networkidle for client-side hydration to catch up
            # — never proven strictly required, but never once run without
            # it either. Keep it rather than silently dropping an unverified
            # simplification.
            page.wait_for_timeout(1500)

            heading = page.get_by_role("heading", name="Flagship models").first
            section = heading.locator(
                "xpath=ancestor::div[.//button[@role='radio']][1]"
            ).first

            # Un-collapse the full model table. Confirmed 2026-09-12: the
            # Flagship-family rows and the full non-flagship list are the
            # same table, collapsed to a handful of rows by default; this
            # button (not a separate section/control) reveals the rest.
            show_all = section.get_by_text("All models", exact=True).first
            if show_all.count() > 0:
                show_all.click()
                page.wait_for_timeout(500)

            out_dir = make_timestamped_output_dir(Path(args.out_dir))
            saved = []
            for tier in TIERS:
                path = capture_tier(section, tier, out_dir)
                saved.append(path)
                print(f"Saved {tier}: {path}")

            print(f"Done. {len(saved)} screenshot(s) in {out_dir}")
        finally:
            # try/finally, not a bare call: an exception mid-loop must not
            # leak the Chromium process.
            browser.close()


if __name__ == "__main__":
    main()
