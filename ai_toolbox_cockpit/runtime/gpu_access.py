"""Read-only checks for local rootless Podman GPU device access."""

import grp
import json
import os
from pathlib import Path
import pwd
import shlex
import subprocess


def _device_sources(command: list[str]) -> list[str]:
    sources = []
    for index, argument in enumerate(command):
        if argument == "--device" and index + 1 < len(command):
            sources.append(command[index + 1].split(":", 1)[0])
        elif argument.startswith("--device="):
            sources.append(argument.partition("=")[2].split(":", 1)[0])
    return sources


def gpu_access_issue(command: list[str], engine: str) -> str | None:
    """Explain a proven host-access failure; never change permissions or groups.

    Check only the local rootless Podman path. Docker/rootful/remote daemons
    have a different identity from this process. access() honors device ACLs;
    missing group membership alone does not establish a permission failure.
    """
    sources = _device_sources(command)
    if (engine != "podman" or "/dev/kfd" not in sources
            or os.environ.get("CONTAINER_HOST") or os.environ.get("CONTAINER_CONNECTION")):
        return None
    try:
        info = subprocess.run([engine, "info", "--format", "json"],
                              capture_output=True, text=True, timeout=10)
        if info.returncode:
            return None
        host = json.loads(info.stdout)["host"]
        if host.get("serviceIsRemote") or host.get("security", {}).get("rootless") is not True:
            return None
        devices = [Path("/dev/kfd")]
        if "/dev/dri" in sources:
            render_nodes = sorted(Path("/dev/dri").glob("renderD*"))
            if not render_nodes:
                return ("Cannot start the GPU server: no render devices found in /dev/dri.\n"
                        "Check that the host AMD GPU driver is loaded and exposes /dev/dri/renderD*.\n"
                        "Changing user groups cannot create a missing device.")
            devices.extend(render_nodes)
        devices.extend(Path(source) for source in sources if source.startswith("/dev/dri/renderD"))
        denied = []
        for device in dict.fromkeys(devices):
            try:
                details = device.stat()
            except FileNotFoundError:
                return (f"Cannot start the GPU server: {device} is missing on the host.\n"
                        "Check that the host AMD GPU driver is loaded and supports this GPU.")
            if not os.access(device, os.R_OK | os.W_OK, effective_ids=True):
                denied.append((device, details.st_gid))
        if not denied:
            return None
        account = pwd.getpwuid(os.geteuid())
        active_groups = set(os.getgroups()) | {os.getegid()}
        configured_groups = set(os.getgrouplist(account.pw_name, account.pw_gid))
        missing = {}
        descriptions = []
        for device, gid in denied:
            try:
                group = grp.getgrgid(gid).gr_name
            except KeyError:
                group = str(gid)
            descriptions.append(f"  {device} (owning group: {group})")
            if gid not in active_groups:
                missing[gid] = group
        message = [f"Cannot start the GPU server: {account.pw_name}'s current session lacks "
                   "read/write access to:", *descriptions]
        new_groups = sorted(name for gid, name in missing.items()
                            if gid not in configured_groups and name in {"render", "video"})
        pending_groups = sorted(name for gid, name in missing.items() if gid in configured_groups)
        if new_groups:
            message.extend([
                "Ask an administrator to add your account to the device group(s), or run:",
                "  " + shlex.join(["sudo", "usermod", "-aG", ",".join(new_groups), account.pw_name]),
            ])
        if any(name not in {"render", "video"} and gid not in configured_groups
               for gid, name in missing.items()):
            message.append("The device uses a nonstandard group. Ask an administrator to review its ownership and ACLs before changing group membership.")
        if pending_groups:
            message.append("Already configured but absent from this session: " + ", ".join(pending_groups) + ".")
        if missing:
            message.extend([
                "Fully log out and back in, then reopen Cockpit and check `id`.",
                "For SSH, disconnect and reconnect; for tmux, use a fresh login outside the old tmux server.",
                "Podman's --group-add keep-groups preserves this session's groups; it cannot grant missing groups.",
            ])
        else:
            message.append("Your session already has the device groups. Ask an administrator to check device permissions and ACLs (`getfacl /dev/kfd /dev/dri/renderD*`).")
        return "\n".join(message)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        # An unavailable preflight must not invent a permissions diagnosis.
        return None
