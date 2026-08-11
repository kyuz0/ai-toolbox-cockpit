import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum


class ContainerEngine(StrEnum):
    PODMAN = "podman"
    DOCKER = "docker"


def detect_container_engines() -> tuple[ContainerEngine, ...]:
    return tuple(engine for engine in ContainerEngine if shutil.which(engine.value))


@dataclass(frozen=True)
class ContainerRecord:
    name: str
    image: str
    status: str
    created: str
    engine: ContainerEngine


def inspect_containers(
    engines: tuple[ContainerEngine, ...] | None = None,
) -> tuple[ContainerRecord, ...]:
    """Read all local container metadata without starting or mutating anything."""
    records: list[ContainerRecord] = []
    for engine in engines if engines is not None else detect_container_engines():
        try:
            result = subprocess.run(
                [
                    engine.value,
                    "ps",
                    "-a",
                    "--format",
                    "{{.Names}}|{{.Image}}|{{.Status}}|{{.CreatedAt}}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        for line in result.stdout.splitlines():
            parts = line.split("|", 3)
            if len(parts) < 3:
                continue
            records.append(
                ContainerRecord(
                    name=parts[0].strip(),
                    image=parts[1].strip(),
                    status=parts[2].strip().replace("292 years ago", "Unknown Date"),
                    created=parts[3].strip() if len(parts) == 4 else "",
                    engine=engine,
                )
            )
    return tuple(records)
