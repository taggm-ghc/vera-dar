"""GAPTESTS-1023 helpers (offline). Built on the 7da574a fixtures and work/dar-pipeline-1022/tests helpers.
Config comes from the REAL config/auto_acquire.json text (shipped state: use_discovered true) written to a tmp file copy
and parsed by the REAL acq_config.load_config(path). Variants are text edits of that copy, never an in-memory dict edit."""
from __future__ import annotations

import re
from pathlib import Path

from tests._gap_support import REAL_LOAD_CONFIG, SHIPPED_TEXT, http_ask  # noqa: F401
from vera import acq_config


def variant_text(*, use_discovered=None, inline_enabled=None, enabled=None, override_layer2=True) -> str:
    t = SHIPPED_TEXT

    def sub(key, val, t, nth=1):
        pat = re.compile(r'("%s"\s*:\s*)(true|false)' % key)
        ms = list(pat.finditer(t))
        assert ms, key
        m = ms[nth - 1]
        return t[:m.start()] + m.group(1) + ("true" if val else "false") + t[m.end():]
    if use_discovered is not None:
        assert len(re.findall(r'"use_discovered"\s*:', t)) == 1
        t = sub("use_discovered", use_discovered, t)
    if override_layer2 is not None:
        t = sub("override_layer2_declines", override_layer2, t)
    if enabled is not None:      # first "enabled" key is the top-level feature flag
        t = sub("enabled", enabled, t)
    return t


def use_file(rig, monkeypatch, tmp_path, text=None) -> Path:
    """Write `text` (default: the shipped file text UNEDITED apart from override_layer2 which the rig needs) to a file and
    route every load_config() call through the real parser on that path."""
    p = tmp_path / "auto_acquire.json"
    p.write_text(variant_text() if text is None else text, encoding="utf-8")
    loaded = REAL_LOAD_CONFIG(p)
    assert loaded is not None
    monkeypatch.setattr(acq_config, "load_config", lambda *a, **k: REAL_LOAD_CONFIG(p))
    rig.cfg = loaded
    return p
