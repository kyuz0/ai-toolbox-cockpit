import json
from importlib.resources import files

from .schema import CatalogError, ModelCatalog, ToolboxCatalog


def _load_asset(filename: str) -> dict:
    asset = files("ai_toolbox_cockpit").joinpath("assets", filename)
    with asset.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{filename} must contain a JSON object")
    return value


def load_toolbox_catalog() -> ToolboxCatalog:
    return ToolboxCatalog.from_dict(_load_asset("toolboxes.json"))


def load_model_catalog() -> ModelCatalog:
    catalog = ModelCatalog.from_dict(_load_asset("models.json"))
    toolbox_catalog = load_toolbox_catalog()
    toolboxes = toolbox_catalog.toolboxes
    for record in catalog.backends["llama_cpp"].config.get("calibrated_ubatches", []):
        toolbox = toolboxes.get(record["toolbox_id"])
        if toolbox is None or toolbox.backend != "llama_cpp":
            raise CatalogError(
                "calibrated ubatch references unknown llama.cpp toolbox: "
                f"{record['toolbox_id']}"
            )
    llama_models = {
        entry["id"]: entry
        for entry in catalog.backends["llama_cpp"].entries
    }
    toolbox_platforms = {
        toolbox_id: platform.id
        for platform in toolbox_catalog.platforms
        for toolbox_id in platform.toolbox_ids
    }
    for toolbox in toolboxes.values():
        recommended = (toolbox.backend_config or {}).get("recommended_use")
        if toolbox.backend != "llama_cpp" or not recommended:
            continue
        model = llama_models.get(recommended["model_id"])
        if model is None:
            raise CatalogError(
                f"{toolbox.id} recommended_use references unknown llama.cpp model: "
                f"{recommended['model_id']}"
            )
        platform_id = toolbox_platforms.get(toolbox.id)
        if recommended["platform_id"] != platform_id:
            raise CatalogError(
                f"{toolbox.id} recommended_use platform does not match its catalogue assignment"
            )
        sidecar = recommended.get("sidecar")
        if not sidecar:
            continue
        downloads = model.get("auxiliary_downloads", [])
        if not any(
            download["role"] == "mtp"
            and download["repo"] == sidecar["repo"]
            and download["recommended_filename"] == sidecar["filename"]
            for download in downloads
        ):
            raise CatalogError(
                f"{toolbox.id} recommended_use sidecar is not declared by "
                f"{recommended['model_id']}"
            )
    return catalog
