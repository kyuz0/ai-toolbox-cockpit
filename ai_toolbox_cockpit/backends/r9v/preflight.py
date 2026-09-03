"""Read-only host checks for the fixed dual-R9700 R9V profile."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


GFX1201_TARGET = 120001
REFERENCE_RAM_BYTES = 128_000_000_000


@dataclass(frozen=True)
class PreflightReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    details: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [*(f"PASS: {item}" for item in self.details)]
        lines.extend(f"WARN: {item}" for item in self.warnings)
        lines.extend(f"FAIL: {item}" for item in self.errors)
        return "\n".join(lines)


def _kfd_targets(sys_root: Path) -> list[tuple[int, int]]:
    targets: list[tuple[int, int]] = []
    root = sys_root / "class/kfd/kfd/topology/nodes"
    for properties in root.glob("*/properties"):
        values: dict[str, int] = {}
        try:
            lines = properties.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                values[parts[0]] = int(parts[1])
        if values.get("location_id", 0) and values.get("gfx_target_version", 0):
            node = int(properties.parent.name) if properties.parent.name.isdigit() else 9999
            targets.append((node, values["gfx_target_version"]))
    return sorted(targets)


def _memory_bytes(proc_root: Path) -> tuple[int | None, int | None]:
    values: dict[str, int] = {}
    try:
        lines = (proc_root / "meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None, None
    for line in lines:
        key, separator, raw = line.partition(":")
        fields = raw.split()
        if separator and fields and fields[0].isdigit():
            values[key] = int(fields[0]) * 1024
    return values.get("MemTotal"), values.get("MemAvailable")


def _storage_report(path: Path) -> tuple[str | None, str | None]:
    try:
        mount = subprocess.run(
            ["findmnt", "-n", "-o", "SOURCE", "-T", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None, "could not inspect the PLE backing storage"
    source = mount.stdout.strip().split("[", 1)[0]
    if mount.returncode or not source.startswith("/dev/"):
        return None, f"PLE backing media is not directly discoverable ({source or 'unknown'})"
    try:
        block = subprocess.run(
            ["lsblk", "-s", "-n", "-o", "TYPE,ROTA,TRAN", source],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None, "could not inspect whether the PLE storage is rotational"
    disks = [
        line.split()
        for raw_line in block.stdout.splitlines()
        if (line := raw_line.strip()).startswith("disk ")
    ]
    if block.returncode or not disks:
        return None, f"could not resolve a physical disk behind {source}"
    if any(len(fields) > 1 and fields[1] == "1" for fields in disks):
        return "PLE is backed by rotational storage; R9V requires SSD storage", None
    transports = {fields[2] for fields in disks if len(fields) > 2 and fields[2] != "-"}
    warning = None if transports == {"nvme"} else "PLE storage is non-rotating but not the qualified NVMe class"
    return None, warning


def inspect_host(
    visible_devices: str,
    ple: Path,
    *,
    sys_root: Path = Path("/sys"),
    proc_root: Path = Path("/proc"),
    dev_root: Path = Path("/dev"),
) -> PreflightReport:
    errors: list[str] = []
    warnings: list[str] = []
    details: list[str] = []

    for device in (dev_root / "kfd", dev_root / "dri"):
        if device.exists():
            details.append(f"device node available: {device}")
        else:
            errors.append(f"required device node is missing: {device}")

    selected = [value.strip() for value in visible_devices.split(",")]
    targets = _kfd_targets(sys_root)
    try:
        indices = [int(value) for value in selected]
    except ValueError:
        indices = []
    if len(indices) != 2 or len(set(indices)) != 2:
        errors.append("R9V requires exactly two distinct numeric HIP device indices")
    elif any(index < 0 or index >= len(targets) for index in indices):
        errors.append(f"HIP device order {visible_devices!r} is outside the KFD GPU inventory")
    else:
        chosen = [targets[index][1] for index in indices]
        if chosen != [GFX1201_TARGET, GFX1201_TARGET]:
            errors.append(f"HIP device order {visible_devices!r} resolves to {chosen}, not two gfx1201 GPUs")
        else:
            details.append(f"HIP device order {visible_devices} resolves to two gfx1201 GPUs")

    total, available = _memory_bytes(proc_root)
    if total is None or available is None:
        warnings.append("could not read host memory capacity")
    else:
        details.append(
            f"host memory: {total / 1024**3:.1f} GiB total, {available / 1024**3:.1f} GiB available"
        )
        if total < REFERENCE_RAM_BYTES:
            warnings.append(
                "host RAM is below R9V's qualified 128 GB reference; keep PLE residency on SSD"
            )

    storage_error, storage_warning = _storage_report(ple)
    if storage_error:
        errors.append(storage_error)
    elif storage_warning:
        warnings.append(storage_warning)
    else:
        details.append("PLE is backed by non-rotating NVMe storage")

    return PreflightReport(tuple(errors), tuple(warnings), tuple(details))
