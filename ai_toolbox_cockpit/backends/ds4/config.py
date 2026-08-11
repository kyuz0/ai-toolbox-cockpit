"""DS4 curated model policy."""

from pathlib import Path

from ai_toolbox_cockpit.catalog import load_model_catalog


def load_models() -> dict:
    backend = load_model_catalog().backends["ds4"]
    return {
        "repo": backend.config.get("default_repo", "antirez/deepseek-v4-gguf"),
        "families": dict(backend.config.get("families", {})),
        "default_server_defaults": dict(backend.config.get("default_server_defaults", {})),
        "models": [dict(entry) for entry in backend.entries],
    }


def get_model_server_defaults(model_path: str) -> dict:
    filename = Path(model_path).name
    data = load_models()
    families = data.get("families", {})
    result = dict(data.get("default_server_defaults", {}))
    for model in data.get("models", []):
        if model.get("filename") == filename:
            family = model.get("family")
            if family in families:
                result.update(families[family])
            result.update(model.get("server_defaults", {}))
            return result
    if "GLM" in filename.upper():
        result.update(families.get("glm-5.2", {
            "ssd_streaming": True,
            "coordinator_layers": "0:37",
            "worker_layers": "38:output",
        }))
    else:
        result.update(families.get("deepseek-v4", {
            "coordinator_layers": "0:21",
            "worker_layers": "22:output",
        }))
    return result
