"""PRIVACY-1104 G9: Streamlit usage telemetry is off in .streamlit/config.toml as well as in start.sh."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_gather_usage_stats_disabled_in_config():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["browser"]["gatherUsageStats"] is False
