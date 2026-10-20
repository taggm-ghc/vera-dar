"""M4: Gate C (evidence-set adequacy) + evidence relations + reasoning context.

Evidence corpus contract (produced by M3, consumed here):

    {
      "run_id": str,
      "question": str,
      "sub_questions": [{"id": "sq1", "text": str, "required": True}],   # optional; see DEFAULT_SUB_QUESTIONS
      "spans": [{
          "span_id": str, "source_id": str, "text": str,
          "sub_question_ids": ["sq1"],
          "source_title": str, "year": int, "source_type": "rct|field_experiment|observational|...",
          "independence_group": str,      # optional; sources sharing a group are NOT independent
          "derived_from": [source_id],    # optional; secondary reporting of other sources
          "appraisal": {"overall": 0..1}  # optional (Gate B output)
      }]
    }
"""
