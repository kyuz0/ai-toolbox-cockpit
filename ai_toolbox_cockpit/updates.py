"""Non-blocking GitHub tag check for pipx-installed application updates."""

from __future__ import annotations

import json
import re
import urllib.request
from importlib import metadata


PACKAGE_NAME = "ai-toolbox-cockpit"
TAGS_URL = "https://api.github.com/repos/kyuz0/ai-toolbox-cockpit/tags?per_page=20"
UPGRADE_COMMAND = ("pipx", "upgrade", PACKAGE_NAME)


def installed_version() -> str:
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return "0.0.0+source"


def version_key(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in re.findall(r"\d+", value))


def latest_version(timeout: float = 5.0) -> str | None:
    try:
        request = urllib.request.Request(TAGS_URL, headers={"User-Agent": PACKAGE_NAME})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, AttributeError):
        return None
    versions = [str(item.get("name", "")).lstrip("v") for item in payload if isinstance(item, dict)]
    versions = [value for value in versions if version_key(value)]
    return max(versions, key=version_key) if versions else None


def available_update(current: str | None = None) -> str | None:
    current = current or installed_version()
    latest = latest_version()
    return latest if latest and version_key(latest) > version_key(current) else None
