"""M3 -- Evidence extraction, critical appraisal, and Gate B admission.

Pipeline (see docs/vera-design.md, Gate B):
    M2 sources -> evidence_extractor -> appraisal_rubric -> gate_b -> m3_runner
Retrieval creates candidates; inspection and appraisal create evidence;
policy (Gate B) creates context.
"""
