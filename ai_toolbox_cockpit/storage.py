"""Filesystem capacity helpers for model download destinations."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiskSpace:
    total: int
    used: int
    free: int


def disk_space_for_path(path: str | Path) -> DiskSpace | None:
    """Return space for the filesystem containing path or its nearest parent."""
    candidate = Path(path).expanduser()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return None
        candidate = parent
    try:
        usage = shutil.disk_usage(candidate)
    except OSError:
        return None
    return DiskSpace(total=usage.total, used=usage.used, free=usage.free)


def format_bytes(size: int) -> str:
    """Format a byte count using compact decimal storage units."""
    value = float(max(size, 0))
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if value < 1000 or unit == "PB":
            precision = 0 if unit == "B" else 1
            return f"{value:.{precision}f} {unit}"
        value /= 1000
    return f"{value:.1f} PB"


def disk_space_text(path: str | Path) -> str:
    """Build the download-page capacity readout for a configured directory."""
    space = disk_space_for_path(path)
    if space is None:
        return "Available space: unavailable"
    return (
        f"Available space: {format_bytes(space.free)} free of "
        f"{format_bytes(space.total)} on the destination filesystem"
    )


def download_space_note(required: int | None, free: int | None) -> str:
    """Describe required/free capacity and warn when the estimate cannot fit."""
    if required is None or required <= 0:
        return ""
    if free is None:
        return f"Estimated download size: {format_bytes(required)}."
    summary = (
        f"Estimated download size: {format_bytes(required)}; "
        f"available space: {format_bytes(free)}."
    )
    if required > free:
        return (
            f"{summary}\n\nWARNING: This download is larger than the available "
            "space and is unlikely to fit."
        )
    return summary
