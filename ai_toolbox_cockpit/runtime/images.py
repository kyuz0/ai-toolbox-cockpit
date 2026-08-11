"""OCI image metadata helpers and side-effect-free command builders."""

import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

from .engines import ContainerEngine


@dataclass(frozen=True)
class ImageCommands:
    engine: ContainerEngine

    def pull(self, image: str) -> list[str]:
        return [self.engine.value, "pull", image]

    def inspect(self, image: str) -> list[str]:
        return [self.engine.value, "image", "inspect", image]

    def remove_container(self, name: str) -> list[str]:
        return [self.engine.value, "rm", "-f", name]


def docker_hub_tag_url(image: str) -> str | None:
    reference = image.removeprefix("docker.io/")
    if "@" in reference:
        return None
    repository, separator, tag = reference.rpartition(":")
    if not separator:
        repository, tag = reference, "latest"
    if "/" not in repository:
        repository = f"library/{repository}"
    return f"https://hub.docker.com/v2/repositories/{repository}/tags/{tag}"


def get_remote_image_date(image: str, timeout: float = 5.0) -> str | None:
    url = docker_hub_tag_url(image)
    if not url:
        return None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ai-toolbox-cockpit"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        value = payload.get("last_updated")
        return value if isinstance(value, str) else None
    except (OSError, ValueError, AttributeError):
        return None


def parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    normalized = re.sub(r"\s+[A-Za-z]{2,5}$", "", value.strip())
    normalized = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", normalized)
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def is_remote_image_newer(remote_updated: str, container_created: str) -> bool:
    remote = parse_timestamp(remote_updated)
    created = parse_timestamp(container_created)
    return bool(remote and created and remote > created)
