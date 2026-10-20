"""Rule-to-test coverage map for the AGENTS.md critical rules (WP-1, phase 1).

Each entry names a test that checks part of a rule. The map is evidence of
partial coverage only. It does not prove a rule is kept. Rules 5-8 have no
mapped test yet (rule 8 has no check; release-gate section (r) is WP-5).
"""
RULE_TESTS = {
    "1": ["tests/test_memory_store.py::test_no_question_or_identity_stored"],
    "2": [
        "tests/test_memory_gate.py::test_denies_each_reason",
        "tests/test_memory_d6_operator.py::test_new_rows_are_pending_by_default_and_when_key_missing",
        "tests/test_memory_d6_operator.py::test_recall_reads_confirmed_only",
    ],
    "3": ["tests/test_memory_gate.py::test_fake_system_turn_and_zero_width_split_phrase"],
    "4": ["tests/test_memory_d6_operator.py::test_wo_sql_is_update_by_claim_id_only_and_ro_reads_are_select"],
}
