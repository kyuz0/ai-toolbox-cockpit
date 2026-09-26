"""Host group translation for container engine flags."""

import grp

_DEVICE_GROUPS = ("video", "render")


def docker_host_group_ids(args: list[str]) -> list[str]:
    """Pass host GIDs to Docker instead of group names.

    Docker resolves ``--group-add`` names in the image ``/etc/group``. Images
    such as Halogen have ``video`` but not ``render``, so the name fails before
    the container starts. A name that exists in the image still has the image
    GID, which does not match the host device nodes. Podman's ``keep-groups``
    has no Docker equivalent and is expanded to the host video and render GIDs.
    """
    result: list[str] = []
    seen: set[str] = set()
    index = 0
    while index < len(args):
        if args[index] == "--group-add" and index + 1 < len(args):
            result.extend(_group_add_flags(args[index + 1], seen))
            index += 2
            continue
        if args[index].startswith("--group-add="):
            for value in _resolved_group_ids(args[index].split("=", 1)[1]):
                if value in seen:
                    continue
                seen.add(value)
                result.append(f"--group-add={value}")
            index += 1
            continue
        result.append(args[index])
        index += 1
    return result


def _group_add_flags(value: str, seen: set[str]) -> list[str]:
    flags: list[str] = []
    for resolved in _resolved_group_ids(value):
        if resolved in seen:
            continue
        seen.add(resolved)
        flags.extend(["--group-add", resolved])
    return flags


def _resolved_group_ids(value: str) -> list[str]:
    if value == "keep-groups":
        return [gid for name in _DEVICE_GROUPS if (gid := _host_gid(name))]
    if value.isdigit():
        return [value]
    return [_host_gid(value) or value]


def _host_gid(name: str) -> str | None:
    try:
        return str(grp.getgrnam(name).gr_gid)
    except KeyError:
        return None
