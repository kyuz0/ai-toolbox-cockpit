from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import ContentSwitcher, Select, Static

from ai_toolbox_cockpit.backends import BACKENDS, backend_options
from ai_toolbox_cockpit.catalog import ModelCatalog


class ModelsView(Vertical):
    def __init__(self, catalog: ModelCatalog, **kwargs) -> None:
        super().__init__(**kwargs)
        self.catalog = catalog

    def compose(self) -> ComposeResult:
        yield Static(
            "Model catalogs are global and backend-specific: GGUF, exact DS4 artifacts, HF repositories, or workflow bundles. Launch availability follows the selected platform in Servers.",
            classes="view-note",
        )
        yield Select(
            backend_options(),
            value="llama_cpp",
            allow_blank=False,
            id="model-backend-select",
        )
        panels = [
            definition.model_panel(
                self.catalog.backends[backend_id],
                id=f"model-panel-{backend_id}",
            )
            for backend_id, definition in BACKENDS.items()
        ]
        yield ContentSwitcher(
            *panels,
            initial="model-panel-llama_cpp",
            id="model-content-switcher",
        )

    @on(Select.Changed, "#model-backend-select")
    def backend_changed(self, event: Select.Changed) -> None:
        self.query_one("#model-content-switcher", ContentSwitcher).current = (
            f"model-panel-{event.value}"
        )

    def set_platform(self, platform_id: str) -> None:
        for definition in BACKENDS.values():
            for panel in self.query(definition.model_panel):
                panel.set_platform(platform_id)
