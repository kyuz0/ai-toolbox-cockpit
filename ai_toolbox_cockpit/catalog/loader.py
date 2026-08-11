import json
from importlib.resources import files

from .schema import ModelCatalog, ToolboxCatalog


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
    return ModelCatalog.from_dict(_load_asset("models.json"))

