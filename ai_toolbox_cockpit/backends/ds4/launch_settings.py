"""Saved DwarfStar (ds4) launch settings, kept per model file and per role.

Layout inside ``backends.ds4`` of the shared config file::

    "launch": {"<model file name>": {"<role>": {"context": "262144", ...}}}

Models are keyed by file name, not full path, so moving the models directory
keeps the saved values. Only plain strings and booleans are stored. API keys
are never part of the ds4 form and are not stored.
"""

from __future__ import annotations

from typing import Any

from ai_toolbox_cockpit.settings import get_backend_settings, save_backend_settings

BACKEND_ID = "ds4"
ROLES = ("Standalone", "Coordinator", "Worker")


def _clean(values: dict[str, Any]) -> dict[str, str | bool]:
    return {
        str(key): value
        for key, value in values.items()
        if isinstance(value, (str, bool))
    }


def load_launch_settings(model_name: str, role: str) -> dict[str, str | bool]:
    """Return the saved form values for a model and role, or an empty dict."""
    if not model_name:
        return {}
    launch = get_backend_settings(BACKEND_ID).get("launch", {})
    per_model = launch.get(model_name, {}) if isinstance(launch, dict) else {}
    saved = per_model.get(role, {}) if isinstance(per_model, dict) else {}
    return _clean(saved) if isinstance(saved, dict) else {}


def save_launch_settings(model_name: str, role: str, values: dict[str, Any]) -> bool:
    """Save the form values for a model and role, keeping all other entries.

    ``save_backend_settings`` replaces top-level keys, so the whole ``launch``
    tree is read, changed and written back.
    """
    if not model_name or role not in ROLES:
        return False
    launch = get_backend_settings(BACKEND_ID).get("launch", {})
    launch = dict(launch) if isinstance(launch, dict) else {}
    per_model = launch.get(model_name, {})
    per_model = dict(per_model) if isinstance(per_model, dict) else {}
    per_model[role] = _clean(values)
    launch[model_name] = per_model
    return save_backend_settings(BACKEND_ID, {"launch": launch})


def load_engine() -> str:
    value = get_backend_settings(BACKEND_ID).get("engine", "")
    return value if isinstance(value, str) else ""


def clear_launch_settings(model_name: str, role: str) -> bool:
    """Forget the saved form values for a model and role; other entries stay."""
    launch = get_backend_settings(BACKEND_ID).get("launch", {})
    if not isinstance(launch, dict) or model_name not in launch:
        return True
    launch = dict(launch)
    per_model = launch[model_name]
    per_model = dict(per_model) if isinstance(per_model, dict) else {}
    per_model.pop(role, None)
    if per_model:
        launch[model_name] = per_model
    else:
        del launch[model_name]
    return save_backend_settings(BACKEND_ID, {"launch": launch})


# Form values in the order they appear in the form, with the labels shown in it.
FIELD_LABELS = {
    "context": "Context length",
    "prefill": "Graph prefill chunk",
    "host": "Host",
    "port": "Port",
    "mxfp4_tile4": "MXFP4 tile4 kernels",
    "mxfp4_down_rgroup": "MXFP4 down-projection rgroup",
    "v41_bounded_replay": "SWA bounded replay",
    "kv_enabled": "Disk KV cache",
    "kv_dir": "KV cache directory",
    "kv_mb": "KV cache size (MB)",
    "ssd_enabled": "SSD streaming",
    "ssd_cold": "Cold preload",
    "ssd_experts": "Expert memory budget",
    "ssd_layers": "Full expert layers",
    "dspark_enabled": "DSpark",
    "dspark_confidence": "DSpark confidence",
    "mtp_enabled": "Embedded MTP",
    "mtp": "External MTP model",
    "vision": "Vision encoder",
    "layers": "Layer range",
    "peer": "Peer address",
    "dist_prefill": "Distributed prefill chunk",
    "dist_window": "Distributed prefill window",
    "tensor_parallel": "Tensor parallel",
    "transport": "Transport",
    "rdma_device": "RDMA device",
    "rdma_port": "RDMA port",
    "rdma_gid": "RDMA GID index",
    "extra_args": "Extra args",
}


def _show(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value) if value is not None else ""
    if not text:
        return "(empty)"
    return text if len(text) <= 60 else text[:57] + "..."


def describe_changes(current: dict[str, Any], target: dict[str, Any]) -> list[str]:
    """One "label: current -> target" line for every value that would change."""
    return [
        f"{label}: {_show(current.get(key))} -> {_show(target.get(key))}"
        for key, label in FIELD_LABELS.items()
        if current.get(key) != target.get(key)
    ]


def load_last_model() -> str:
    """File name of the model that was launched last, or an empty string."""
    value = get_backend_settings(BACKEND_ID).get("last_model", "")
    return value if isinstance(value, str) else ""


def save_last_launch(engine: str, model_name: str) -> bool:
    """Remember the container engine and model that were launched last."""
    return save_backend_settings(
        BACKEND_ID, {"engine": engine, "last_model": model_name}
    )
