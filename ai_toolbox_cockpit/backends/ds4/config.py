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


def get_model_artifact(model_path: str) -> dict:
    """Return the exact DS4 catalogue record for a local artifact, if known."""
    filename = Path(model_path).name
    for model in load_models().get("models", []):
        if model.get("filename") == filename:
            return model
    return {}


def get_artifact_role(model_path: str) -> str:
    """Classify main and auxiliary DS4 GGUFs without mixing their semantics."""
    model = get_model_artifact(model_path)
    if model:
        return str(model.get("artifact_role", "main"))

    filename = Path(model_path).name.lower()
    if "dspark-support" in filename:
        return "dspark_support"
    if "vision-encoder" in filename:
        return "vision_encoder"
    if "mtp" in filename:
        return "mtp"
    return "main"


def get_model_server_defaults(model_path: str) -> dict:
    filename = Path(model_path).name
    data = load_models()
    families = data.get("families", {})
    result = dict(data.get("default_server_defaults", {}))
    model = get_model_artifact(filename)
    if model:
        family = model.get("family")
        if family in families:
            result.update(families[family])
        result.update(model.get("server_defaults", {}))
        return result
    if "GLM-5.3-FLASH" in filename.upper():
        result.update(families.get("glm-5.3-flash", {
            "standalone_ctx": 262144,
        }))
    elif "GLM" in filename.upper():
        # Unknown GLM artifacts must not inherit removed model-family policy.
        return result
    else:
        result.update(families.get("deepseek-v4", {
            "coordinator_layers": "0:21",
            "worker_layers": "22:output",
        }))
    return result
