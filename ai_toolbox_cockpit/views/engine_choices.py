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
    for backend_id, definition in BACKENDS.items():
        entries = [t for t in toolboxes if t.backend == backend_id]
        named = [t for t in entries if (t.backend_config or {}).get("engine_selector")]
        if named:
            for toolbox in named:
                selector = toolbox.backend_config["engine_selector"]
                choices[selector["id"]] = EngineChoice(selector["label"], backend_id, toolbox.id)
        elif entries:
            choices[backend_id] = EngineChoice(definition.label, backend_id)
    return choices
