"""Host group translation for container engine flags."""

import grp
import os


def docker_host_group_ids(args: list[str]) -> list[str]:
    """Pass host GIDs to Docker instead of group names.

    Docker resolves ``--group-add`` names in the image ``/etc/group``. Images
    such as Halogen have ``video`` but not ``render``, so the name fails before
    the container starts. A name that exists in the image still has the image
    GID, which does not match the host device nodes. Podman's ``keep-groups``
    has no Docker equivalent and is expanded to the caller's supplementary GIDs.
    An unknown host group cannot be translated safely, so fail before launching
    Docker rather than passing it a name that may not exist in the image.
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
        return [str(gid) for gid in os.getgroups()]
    if value.isdigit():
        return [value]
    try:
        return [str(grp.getgrnam(value).gr_gid)]
    except KeyError as error:
        raise ValueError(f"Cannot add host group {value!r}: group does not exist on this host") from error
