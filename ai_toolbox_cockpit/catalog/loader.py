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
    toolboxes = load_toolbox_catalog().toolboxes
    for record in catalog.backends["llama_cpp"].config.get("calibrated_ubatches", []):
        toolbox = toolboxes.get(record["toolbox_id"])
        if toolbox is None or toolbox.backend != "llama_cpp":
            raise CatalogError(
                "calibrated ubatch references unknown llama.cpp toolbox: "
                f"{record['toolbox_id']}"
            )
    return catalog
