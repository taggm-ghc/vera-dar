"""OD-1030-C3 regression: the sanctioned runner must not let pytest create `<name>current` convenience symlinks.
If pytest renames/changes `_pytest.pathlib._force_symlink`, the runner launcher silently stops working and this fails."""


def test_tmp_path_has_no_symlinks(tmp_path):
    assert [p.name for p in tmp_path.parent.iterdir() if p.is_symlink()] == []


def test_second_tmp_path_has_no_symlinks(tmp_path):
    assert [p.name for p in tmp_path.parent.iterdir() if p.is_symlink()] == []


def test_force_symlink_helper_exists():
    import _pytest.pathlib as p
    assert hasattr(p, "_force_symlink"), "pytest renamed _force_symlink: update the runner launcher"


def test_root_conftest_patched_helper_and_no_links(tmp_path):
    """G2: the ROOT conftest (loaded by every pytest, direct or via the runner) disables the helper itself, and
    make_numbered_dir then creates 0 symlinks. Fails loudly if the helper is missing (pytest renamed it)."""
    import _pytest.pathlib as p
    assert hasattr(p, "_force_symlink"), "pytest renamed _force_symlink: update repo/conftest.py (G2 guard)"
    assert p._force_symlink("x", "y") is None and p._force_symlink.__name__ == "<lambda>", "root conftest guard not active"
    base = tmp_path / "numbered"; base.mkdir()
    p.make_numbered_dir(base, prefix="t-")
    assert [q.name for q in base.iterdir() if q.is_symlink()] == []
