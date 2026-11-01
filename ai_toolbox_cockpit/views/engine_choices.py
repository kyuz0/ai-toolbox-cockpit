"""Named engine choices share backend implementations without sharing navigation."""
from dataclasses import dataclass
from ai_toolbox_cockpit.backends import BACKENDS


@dataclass(frozen=True)
class EngineChoice:
    label: str
    backend_id: str
    toolbox_id: str = ""


def engine_choices(catalog, platform_id: str) -> dict[str, EngineChoice]:
    toolboxes = catalog.platform_toolboxes(platform_id)
    choices = {}
    definitions = BACKENDS.items()
    if platform_id == "r9700":
        # This platform also covers one-card hosts. Keep its existing opening
        # engine when the four-card Strata toolbox becomes available.
        definitions = sorted(definitions, key=lambda item: item[0] == "strata")
    for backend_id, definition in definitions:
        entries = [t for t in toolboxes if t.backend == backend_id]
        named = [t for t in entries if (t.backend_config or {}).get("engine_selector")]
        if named:
            for toolbox in named:
                selector = toolbox.backend_config["engine_selector"]
                choices[selector["id"]] = EngineChoice(selector["label"], backend_id, toolbox.id)
        elif entries:
            choices[backend_id] = EngineChoice(definition.label, backend_id)
    return choices
