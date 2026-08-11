"""Persistent, measured ubatch selections shared by benchmarks and inference."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_PROFILE_PATH = Path(
    os.environ.get(
        "AI_TOOLBOX_COCKPIT_UBATCH_PROFILES",
        "~/.config/ai-toolbox-cockpit/llama-ubatch-calibrations.json",
    )
).expanduser()


def model_key(model_path: str) -> str:
    """Return a stable key for a GGUF, normalizing multipart shard suffixes."""
    name = Path(model_path).name
    if name.lower().endswith(".gguf"):
        name = name[:-5]
    return re.sub(r"-\d{5}-of-\d{5}$", "", name)


def backend_from_name(value: str) -> str | None:
    """Map a toolbox name or image to the backend family used for calibration."""
    lowered = value.lower()
    if "vulkan" in lowered:
        return "vulkan"
    if "rocm" in lowered or "therock" in lowered:
        return "rocm"
    return None


def load_profiles(path: Path | None = None) -> dict[str, Any]:
    profile_path = path or DEFAULT_PROFILE_PATH
    if not profile_path.is_file():
        return {"schema_version": SCHEMA_VERSION, "profiles": {}}
    with profile_path.open("r", encoding="utf-8") as source:
        data = json.load(source)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported ubatch calibration schema: {profile_path}")
    if not isinstance(data.get("profiles"), dict):
        raise ValueError(f"Invalid ubatch calibration profiles: {profile_path}")
    return data


def get_calibrated_ubatch(
    model_path: str,
    platform_id: str,
    backend: str,
    path: Path | None = None,
) -> int | None:
    data = load_profiles(path)
    profile = (
        data["profiles"]
        .get(platform_id, {})
        .get(backend, {})
        .get(model_key(model_path), {})
    )
    value = profile.get("selected_ubatch")
    return value if isinstance(value, int) and value > 0 else None


def save_profile(
    platform_id: str,
    backend: str,
    model_path: str,
    profile: dict[str, Any],
    path: Path | None = None,
) -> Path:
    """Atomically append a run and update the selection after a successful run."""
    profile_path = path or DEFAULT_PROFILE_PATH
    data = load_profiles(profile_path)
    models = (
        data["profiles"]
        .setdefault(platform_id, {})
        .setdefault(backend, {})
    )
    key = model_key(model_path)
    existing = models.get(key, {})
    runs = list(existing.get("runs", []))
    runs.append(profile)
    selected = profile.get("selected_ubatch")
    models[key] = {
        "selected_ubatch": (
            selected if isinstance(selected, int) and selected > 0
            else existing.get("selected_ubatch")
        ),
        "latest_run": profile,
        "runs": runs,
    }

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = profile_path.with_suffix(profile_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(data, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(profile_path)
    return profile_path
