"""CHECKCONFIG-FIX-1105: legacy-literal scan ignores docstrings and deny-list modules; default scan dir is repo/.
Synthetic layout under tmp_path only; git child processes disabled; no connection, no env file read."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GLOBS = '(".env", ".env.*", "*.env", ".env-*", "*.env-*")'


def _load():
    spec = importlib.util.spec_from_file_location("_t_check_config_1105", ROOT / "scripts" / "check_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._GIT["enabled"] = False
    return mod


def _layout(tmp_path, files):
    """Synthetic root with repo/ (canonical) and files {repo-relative path: text}. Returns (root, repo)."""
    root = tmp_path / "R"
    repo = root / "repo"
    for d in ("scripts", "tests", "db/accounts", "vera", ".git", "config/env"):
        (repo / d).mkdir(parents=True, exist_ok=True)
    (repo / "scripts" / "vera_test_runner.py").write_text(f"CRED_MASK_GLOBS = {GLOBS}\n")
    (repo / "tests" / "vera_test_boundary.py").write_text(f"CRED_FILE_GLOBS = {GLOBS}\n")
    (repo / "db" / "accounts" / "create_vera_accounts.sql").write_text("-- roles only\n")
    for rel, text in files.items():
        (repo / rel).write_text(text)
    return root, repo


def _count(mod, root, repo):
    findings = []
    out = mod.layout_checks(root, repo, root / "config" / "env", findings)
    return out["legacy_literal_count"], findings


def test_true_positive_code_literal_still_caught(tmp_path):
    mod = _load()
    root, repo = _layout(tmp_path, {
        "vera/bad.py": '"""Docstring mentioning week-1v2 is prose and is not counted."""\n'
                       'PATH = "config/week-1v2/data.json"\n',
    })
    count, findings = _count(mod, root, repo)
    assert count == 1
    assert "legacy path literals present" in findings


def test_known_false_positives_not_counted(tmp_path):
    mod = _load()
    root, repo = _layout(tmp_path, {
        # docstring-only hits (memory_gate, api_triage, apply_migrations, migration_guard, vera_test_runner docstring)
        "vera/memory_gate.py": '"""Adapted from AI-Internship and week-1v2."""\nX = 1\n',
        "vera/api_triage.py": '"""Ported from AI-Internship api_client."""\n',
        "db/migration_guard.py": '"""Never imports week-1v2 db.py."""\n',
        # deny-list modules: protective entries, exempt by file
        "vera/env_precedence.py": '_FORBIDDEN_FILE_GLOBS = (".env.db-accounts",)\n',
        "scripts/vera_test_runner.py": f'"""Masks AI-Internship files."""\nCRED_MASK_GLOBS = {GLOBS}\n',
    })
    count, findings = _count(mod, root, repo)
    assert count == 0
    assert not any("legacy" in f for f in findings)


def test_real_literal_in_exempt_file_still_caught(tmp_path):
    """Exemption is per named constant, not per file: a real legacy path elsewhere in an exempted module counts."""
    mod = _load()
    root, repo = _layout(tmp_path, {
        "vera/env_precedence.py": '_FORBIDDEN_FILE_GLOBS = (".env.db-accounts",)\n'
                                  'LEGACY_ROOT = "/data/week-1v2/cache"\n',
        "scripts/vera_test_runner.py": f'CRED_MASK_GLOBS = {GLOBS}\n'
                                       '_EXTRA = ".env.db-accounts"\n',
    })
    count, findings = _count(mod, root, repo)
    assert count == 2   # one in env_precedence.py (LEGACY_ROOT), one in vera_test_runner.py (_EXTRA)
    assert "legacy path literals present" in findings


def test_default_scan_dir_is_repo_not_vera(tmp_path):
    mod = _load()
    root, _ = _layout(tmp_path, {"vera/bad.py": 'P = "AI-Internship/x"\n'})
    report, findings = mod.build_report(root, {}, git=False)   # no repo= argument: default root/"repo"
    assert report["layout"]["legacy_literal_count"] == 1
    assert "legacy path literals present" in findings
    # a legacy-literal file under a non-canonical root/vera/ must not be what is scanned
    (root / "vera").mkdir()
    (root / "vera" / "other.py").write_text('Q = "week-1v2"\n')
    report2, _ = mod.build_report(root, {}, git=False)
    assert report2["layout"]["legacy_literal_count"] == 1


def test_real_canonical_tree_has_no_false_positives():
    mod = _load()
    count, _ = _count(mod, ROOT, ROOT)
    assert count == 0
