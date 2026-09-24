"""Shared RDMA endpoint discovery and container passthrough.

Linux exposes both native InfiniBand and RoCE HCAs below
``/sys/class/infiniband``.  Discovery is deliberately read-only: callers can
use a valid verbs-device, physical-port and GID-table tuple without invoking
vendor tools or relying on host-specific interface names.  Container passthrough
remains presence-based, so hosts without ``/dev/infiniband`` get no extra flags.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from pathlib import Path

from .engines import ContainerEngine

RDMA_DEVICE_PATH = "/dev/infiniband"
RDMA_SYSFS_PATH = "/sys/class/infiniband"
NETWORK_SYSFS_PATH = "/sys/class/net"
RDMA_MEMLOCK_ULIMIT = "memlock=-1"


@dataclass(frozen=True)
class RDMAEndpoint:
    """One populated RDMA GID-table entry exposed by a Linux HCA."""

    device: str
    port: int
    gid_index: int
    link_layer: str
    gid_type: str
    state: str
    netdev: str | None = None
    netdev_mode: str = ""

    @property
    def active(self) -> bool:
        """Whether the kernel reports the physical port as ACTIVE."""
        state = self.state.strip()
        return state.casefold() == "active" or state.startswith("4:")

    @property
    def link_type(self) -> str:
        """Classify a usable GID as native InfiniBand, RoCEv2, or unknown."""
        link_layer = "".join(
            character for character in self.link_layer.casefold() if character.isalnum()
        )
        gid_type = "".join(
            character for character in self.gid_type.casefold() if character.isalnum()
        )
        if link_layer == "infiniband" and gid_type.startswith("ib"):
            return "infiniband"
        if link_layer in {"ethernet", "roce", "rocev2"} and "rocev2" in gid_type:
            return "roce"
        return "unknown"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return ""


def _is_usable_gid(value: str) -> bool:
    """Reject unset GID slots, including kernel's link-local ``fe80::`` slot."""
    if not value:
        return False
    try:
        address = ipaddress.IPv6Address(value)
    except ipaddress.AddressValueError:
        return False
    # The lower 64 bits are the interface identifier. Linux uses an all-zero
    # IID for an unused ``fe80::`` GID slot even though its prefix is populated.
    return any(address.packed[-8:])


def _netdev_for_device(device: Path, network_path: Path) -> tuple[str | None, str]:
    try:
        netdevs = sorted(network_path.iterdir())
    except OSError:
        return None, ""
    for netdev in netdevs:
        infiniband_link = netdev / "device" / "infiniband"
        if not infiniband_link.exists() and not infiniband_link.is_symlink():
            continue
        try:
            if infiniband_link.resolve() == device.resolve():
                return netdev.name, _read_text(netdev / "mode")
        except (OSError, RuntimeError):
            continue
    return None, ""


def discover_rdma_endpoints(
    infiniband_path: str | Path = RDMA_SYSFS_PATH,
    network_path: str | Path = NETWORK_SYSFS_PATH,
) -> list[RDMAEndpoint]:
    """Return populated RDMA GID entries from Linux sysfs.

    Unset GID slots are omitted; port state is retained on each endpoint so
    callers can distinguish active links. The function only reads sysfs, making
    it suitable for use while composing the Textual application.
    """
    root = Path(infiniband_path)
    netroot = Path(network_path)
    try:
        devices = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError:
        return []

    endpoints: list[RDMAEndpoint] = []
    for device_path in devices:
        try:
            port_paths = sorted(
                (
                    path
                    for path in (device_path / "ports").iterdir()
                    if path.is_dir() and path.name.isdigit()
                ),
                key=lambda path: int(path.name),
            )
        except OSError:
            continue
        for port_path in port_paths:
            link_layer = _read_text(port_path / "link_layer")
            state = _read_text(port_path / "state")
            try:
                gid_paths = sorted(
                    (path for path in (port_path / "gids").iterdir() if path.name.isdigit()),
                    key=lambda path: int(path.name),
                )
            except OSError:
                continue
            netdev, netdev_mode = _netdev_for_device(device_path, netroot)
            for gid_path in gid_paths:
                gid = _read_text(gid_path)
                if not _is_usable_gid(gid):
                    continue
                gid_index = int(gid_path.name)
                gid_type = _read_text(
                    port_path / "gid_attrs" / "types" / str(gid_index)
                )
                endpoints.append(
                    RDMAEndpoint(
                        device=device_path.name,
                        port=int(port_path.name),
                        gid_index=gid_index,
                        link_layer=link_layer,
                        gid_type=gid_type,
                        state=state,
                        netdev=netdev,
                        netdev_mode=netdev_mode,
                    )
                )
    return endpoints


def host_rdma_device_nodes(rdma_path: str | Path = RDMA_DEVICE_PATH) -> list[Path]:
    """Return the InfiniBand device nodes present on the host."""
    path = Path(rdma_path)
    if not path.is_dir():
        return []
    try:
        # Linux also exposes by-ibdev/ and by-path/ convenience directories;
        # neither the directory trees nor their nested aliases are device nodes
        # that Docker should receive as top-level --device arguments.
        return sorted(entry for entry in path.iterdir() if not entry.is_dir())
    except OSError:
        return []


def container_rdma_args(
    engine: str | ContainerEngine,
    rdma_path: str = RDMA_DEVICE_PATH,
) -> list[str]:
    """Return engine flags that expose the host InfiniBand devices to a container.

    Podman accepts the device directory, so it gets the directory plus the rdma
    supplementary group and an unlimited memlock ulimit. Docker cannot take a
    directory, so each device node is passed individually.
    """
    engine_name = engine.value if isinstance(engine, ContainerEngine) else engine
    if not Path(rdma_path).is_dir():
        return []

    if engine_name == ContainerEngine.PODMAN.value:
        return [
            "--device", rdma_path,
            "--group-add", "rdma",
            "--ulimit", RDMA_MEMLOCK_ULIMIT,
        ]

    if engine_name == ContainerEngine.DOCKER.value:
        args: list[str] = []
        for device in host_rdma_device_nodes(rdma_path):
            args.extend(["--device", str(device)])
        if args:
            args.extend(["--ulimit", RDMA_MEMLOCK_ULIMIT])
        return args

    return []
