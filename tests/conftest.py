"""Keep application tests independent of saved desktop preferences."""

import pytest


@pytest.fixture(autouse=True)
def isolated_cockpit_settings(tmp_path, monkeypatch):
    config_root = tmp_path / "config"
    config_file = config_root / "ai-toolbox-cockpit" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_root))
