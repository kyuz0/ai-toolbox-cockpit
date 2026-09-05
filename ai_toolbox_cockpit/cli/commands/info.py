"""Static catalogue report command."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ...catalog import CatalogError, ModelBackendCatalog, Toolbox, load_model_catalog, load_toolbox_catalog
from ...settings import get_backend_settings, load_active_platform
from ...updates import installed_version
from ..contracts import CLI_SCHEMA_VERSION, CommandError, CommandResult, EXIT_UNAVAILABLE, Invocation


def handle_info(invocation: Invocation) -> CommandResult | CommandError:
    from ..registry import command_capabilities

    try:
        return CommandResult(
            data=build_info_report(
                full=bool(invocation.arguments["full"]),
                capabilities=command_capabilities(),
            )
        )
    except (CatalogError, OSError, ValueError) as error:
        return CommandError(
            code="catalog_error",
            message=str(error),
            exit_code=EXIT_UNAVAILABLE,
        )


def build_info_report(full: bool, capabilities: Mapping[str, Any]) -> dict[str, Any]:
    toolbox_catalog = load_toolbox_catalog()
    model_catalog = load_model_catalog()
    fallback_platform = toolbox_catalog.platforms[0].id if toolbox_catalog.platforms else ""
    models_dir = get_backend_settings("llama_cpp").get("models_dir")

    catalog: dict[str, Any] = {
        "toolbox_schema_version": toolbox_catalog.schema_version,
        "model_schema_version": model_catalog.schema_version,
        "platforms": [
            {
                "id": platform.id,
                "name": platform.name,
                "description": platform.description,
                "toolbox_ids": list(platform.toolbox_ids),
                "defaults": platform.defaults,
            }
            for platform in toolbox_catalog.platforms
        ],
        "toolboxes": [
            _toolbox_report(toolbox, full)
            for toolbox in sorted(toolbox_catalog.toolboxes.values(), key=lambda item: item.id)
        ],
        "backends": {
            backend_id: _backend_report(backend, full)
            for backend_id, backend in sorted(model_catalog.backends.items())
        },
    }
    if full:
        catalog["runtime_profiles"] = {
            profile_id: {"engine_args": list(profile.engine_args)}
            for profile_id, profile in sorted(toolbox_catalog.runtime_profiles.items())
        }

    return {
        "full": full,
        "application": {
            "name": "ai-toolbox-cockpit",
            "version": installed_version(),
        },
        "configuration": {
            "active_platform": load_active_platform(fallback_platform),
            "llama_cpp_models_dir": models_dir if isinstance(models_dir, str) and models_dir else "~/models",
        },
        "capabilities": capabilities,
        "catalog": catalog,
    }


def render_info_text(data: Mapping[str, Any]) -> str:
    application = data["application"]
    configuration = data["configuration"]
    catalog = data["catalog"]
    lines = [
        f"AI Toolbox Cockpit {application['version']}",
        f"CLI schema: {CLI_SCHEMA_VERSION}",
        f"Active platform: {configuration['active_platform']}",
        f"llama.cpp models directory: {configuration['llama_cpp_models_dir']}",
        "",
        "Platforms:",
    ]
    lines.extend(
        f"- {platform['id']}: {platform['name']} ({len(platform['toolbox_ids'])} toolboxes)"
        for platform in catalog["platforms"]
    )
    lines.extend(("", "Backends:"))
    for backend in catalog["backends"].values():
        lines.append(f"- {backend['id']} [{backend['kind']}]: {backend['entry_count']} {backend['entries_key']}")
        for model in backend[backend["entries_key"]]:
            details = " — ".join(
                str(value)
                for key, value in model.items()
                if key in {"name", "repo", "filename"} and value
            )
            lines.append(f"  - {model['id']}{f': {details}' if details else ''}")
    lines.extend(("", "Toolboxes:"))
    lines.extend(
        f"- {toolbox['id']} [{toolbox['backend']}]: {toolbox['image']}"
        for toolbox in catalog["toolboxes"]
    )
    if data["full"]:
        lines.extend(("", "Full static catalogue:"))
        lines.append(json.dumps(catalog, indent=2, sort_keys=True))
    return "\n".join(lines) + "\n"


def _toolbox_report(toolbox: Toolbox, full: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "id": toolbox.id,
        "backend": toolbox.backend,
        "name": toolbox.name,
        "container_name": toolbox.container_name,
        "group": toolbox.group,
        "image": toolbox.image,
        "channel": toolbox.channel,
        "maturity": toolbox.maturity,
        "description": toolbox.description,
        "runtime_profile": toolbox.runtime_profile,
        "features": toolbox.features,
        "server_binary": toolbox.server_binary,
        "supports_load_mode": toolbox.supports_load_mode,
    }
    if full:
        report["backend_config"] = toolbox.backend_config or {}
    return report


def _backend_report(backend: ModelBackendCatalog, full: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "id": backend.id,
        "kind": backend.kind,
        "entries_key": backend.entries_key,
        "storage": backend.storage,
        "entry_count": len(backend.entries),
        backend.entries_key: [_entry_report(entry, full) for entry in backend.entries],
    }
    if full:
        report["config"] = backend.config
    return report


def _entry_report(entry: dict[str, Any], full: bool) -> dict[str, Any]:
    if full:
        return dict(entry)
    return {
        key: value
        for key, value in entry.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }
