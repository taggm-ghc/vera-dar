"""LTD-DIGEST-HUNK-1105: corpus_description.load_numeric_words reads digest.numeric_words via list_tables. No DB."""
import json
from pathlib import Path

import pytest

from vera import corpus_description as cd
from vera import list_tables as lt

ROOT = Path(__file__).resolve().parents[1]
ENV = {"VERA_DB_URL_RO": "postgresql://x/y"}
NAME = "digest.numeric_words"


class Eng:
    def __init__(self, lists, items, boom=None):
        self.lists, self.items, self.boom = lists, items, boom

    def connect(self):
        if self.boom:
            raise self.boom
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql):
        self.rows = self.lists if "lists_v" in sql else self.items
        return self

    def fetchall(self):
        return self.rows


def _cfg():
    spec = json.loads((ROOT / "config" / "list_tables.json").read_text())["lists"][NAME]
    return {"read_from_db": True, "db_url_env": "VERA_DB_URL_RO", "cache_ttl_s": 300, "failure_retry_s": 60,
            "lists": {NAME: spec}}


@pytest.fixture(autouse=True)
def _reset():
    lt.reset_cache()
    yield
    lt.reset_cache()


def _seed():
    return json.loads(cd.NUMERIC_WORDS_PATH.read_text(encoding="utf-8"))["numeric_words"]["words"]


def _patch(monkeypatch, eng):
    real = lt.get
    monkeypatch.setattr(lt, "get", lambda n, seed: real(n, seed, cfg=_cfg(), env=ENV, engine_factory=lambda *a, **k: eng,
                                                        clock=lambda: 1000.0))


def test_flag_off_gives_config_list():
    assert lt._CFG["read_from_db"] is False
    assert cd.load_numeric_words() == {str(w).casefold() for w in _seed()}


def test_flag_on_fake_engine_gives_db_rows(monkeypatch):
    _patch(monkeypatch, Eng([(NAME, "strings", "replace", 1)], [(NAME, "Zork", 1, 1, {}), (NAME, "blarg", 2, 1, {})]))
    assert cd.load_numeric_words() == {"zork", "blarg"}


def test_db_error_falls_back_to_seed(monkeypatch):
    _patch(monkeypatch, Eng([], [], boom=RuntimeError("down")))
    assert cd.load_numeric_words() == {str(w).casefold() for w in _seed()}


def test_non_default_path_ignores_db(monkeypatch, tmp_path):
    _patch(monkeypatch, Eng([(NAME, "strings", "replace", 1)], [(NAME, "zork", 1, 1, {})]))
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"numeric_words": {"words": ["One"]}}))
    assert cd.load_numeric_words(p) == {"one"}
