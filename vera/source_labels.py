"""Shared source-class labels (item #72 W3c / R72-f; origin item #71 R6-a).

One helper used by vera/trace_eval/capture.py (grounded-baseline source list) and vera/m4/context_builder.py
(reasoning context). The label is text for attribution and display. This module claims NO effect on model
behaviour: whether a label changes what a model says is unverified.
"""
from __future__ import annotations

NEWS_LABEL = "[news, grey literature: practice claims only, not effect evidence]"
VENDOR_LABEL = "[vendor claim: not independent evidence]"


def source_label(rec: dict) -> str:
    """'' for an unlabelled record, else the label text (no trailing space). Vendor claim wins over news."""
    if rec.get("source_class") == "vendor_claim":
        return VENDOR_LABEL
    return NEWS_LABEL if rec.get("source_type") == "news" else ""
