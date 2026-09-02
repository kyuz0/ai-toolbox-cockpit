from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def config_path() -> Path:
    root = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    return root / "ai-toolbox-cockpit" / "config.json"


def load_settings() -> dict[str, Any]:
    try:
        with config_path().open("r", encoding="utf-8") as source:
            data = json.load(source)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return _legacy_settings()


def save_settings(data: dict[str, Any]) -> bool:
    path = config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as target:
            json.dump(data, target, indent=2)
            target.write("\n")
        temporary.chmod(0o600)
        temporary.replace(path)
        return True
    except (OSError, ValueError):
        return False


def get_setting(key: str, default: Any = None) -> Any:
    return load_settings().get(key, default)


def set_setting(key: str, value: Any) -> bool:
    data = load_settings()
    data[key] = value
    return save_settings(data)


def get_backend_settings(backend_id: str) -> dict[str, Any]:
    backends = load_settings().get("backends", {})
    if not isinstance(backends, dict):
        return {}
    value = backends.get(backend_id, {})
    return dict(value) if isinstance(value, dict) else {}


def save_backend_settings(backend_id: str, values: dict[str, Any]) -> bool:
    data = load_settings()
    backends = data.setdefault("backends", {})
    if not isinstance(backends, dict):
        backends = {}
        data["backends"] = backends
    current = backends.setdefault(backend_id, {})
    if not isinstance(current, dict):
        current = {}
        backends[backend_id] = current
    current.update(values)
    return save_settings(data)


def load_default_toolbox(backend_id: str, platform_id: str, fallback: str = "") -> str:
    values = get_backend_settings(backend_id).get("default_toolboxes", {})
    if isinstance(values, dict):
        selected = values.get(platform_id)
        if isinstance(selected, str) and selected:
            return selected
    return fallback


def save_default_toolbox(backend_id: str, platform_id: str, toolbox_id: str) -> bool:
    settings = get_backend_settings(backend_id)
    defaults = settings.get("default_toolboxes", {})
    if not isinstance(defaults, dict):
        defaults = {}
    defaults[platform_id] = toolbox_id
    return save_backend_settings(backend_id, {"default_toolboxes": defaults})


def load_active_platform(default: str) -> str:
    value = get_setting("active_platform", default)
    return value if isinstance(value, str) and value else default


def save_active_platform(platform_id: str) -> bool:
    return set_setting("active_platform", platform_id)


def _legacy_settings() -> dict[str, Any]:
    """Read old cockpit settings once without modifying the old files."""
    migrated: dict[str, Any] = {}
    for path in (Path("~/.llama-cockpit.conf").expanduser(), Path("~/.ds4-cockpit.conf").expanduser()):
        try:
            with path.open("r", encoding="utf-8") as source:
                value = json.load(source)
        except (OSError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
        if path.name.startswith(".llama"):
            migrated.setdefault("active_platform", value.get("active_platform", "strix-halo"))
            migrated.setdefault("backends", {})["llama_cpp"] = value
        else:
            migrated.setdefault("backends", {})["ds4"] = value
    return migrated
