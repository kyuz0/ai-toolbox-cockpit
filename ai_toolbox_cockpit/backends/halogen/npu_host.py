"""Read-only host probes for the Ryzen AI NPU requirements Halogen's entrypoint checks."""

from pathlib import Path

XRT_LIBRARIES = ("libxrt_coreutil.so.2", "libxrt_core.so.2", "libxrt_driver_xdna.so.2")
SYSTEM_XRT_DIRS = ("usr/lib/x86_64-linux-gnu", "usr/lib", "usr/lib64")


def npu_device_available(root: Path = Path("/")) -> bool:
    return (root / "dev" / "accel" / "accel0").exists()


def _usable_directory_xrt(xrt: Path) -> bool:
    """A whole-directory mount works unless its links point into the unmounted system libraries."""
    for name in XRT_LIBRARIES:
        path = xrt / "lib" / name
        if not path.is_file():
            return False
        if path.is_symlink() and not path.resolve().is_relative_to(xrt.resolve()):
            return False
    return True


def xrt_mount_arguments(root: Path = Path("/")) -> list[str]:
    """Engine arguments that expose the host's XRT, or a ValueError naming what is missing."""
    xrt = root / "opt" / "xilinx" / "xrt"
    if _usable_directory_xrt(xrt):
        return ["-v", f"{xrt}:/opt/xilinx/xrt:ro"]
    for relative in SYSTEM_XRT_DIRS:
        base = root / relative
        resolved = {name: base / name for name in XRT_LIBRARIES if (base / name).is_file()}
        if len(resolved) == len(XRT_LIBRARIES):
            arguments: list[str] = []
            for name in XRT_LIBRARIES:
                real = resolved[name].resolve()
                arguments.extend([
                    "-v", f"{real}:/opt/xilinx/xrt/lib/{name}:ro",
                    "-v", f"{real}:{base / name}:ro",
                ])
            return arguments
    raise ValueError(
        "No host XRT with its NPU plugin was found. Install AMD's XRT under /opt/xilinx/xrt, or the "
        "distribution's XRT packages (Ubuntu: libxrt2, libxrt-npu2; Arch: xrt, xrt-plugin-amdxdna)."
    )


def fabric_clock_held(root: Path = Path("/")) -> bool:
    """True when the top fabric clock is held, as the host unit or the entrypoint sets it."""
    for device in sorted((root / "sys" / "class" / "drm").glob("card*/device")):
        level = device / "power_dpm_force_performance_level"
        clock = device / "pp_dpm_fclk"
        if not level.is_file() or not clock.is_file():
            continue
        try:
            mode = level.read_text().strip()
            levels = [line for line in clock.read_text().splitlines() if line.strip()]
        except OSError:
            continue
        if mode == "high":
            return True
        starred = [line for line in levels if "*" in line]
        if mode == "manual" and len(starred) == 1 and starred[0] == levels[-1]:
            return True
    return False
