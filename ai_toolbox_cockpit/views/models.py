from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import ContentSwitcher, Static

from ai_toolbox_cockpit.backends import BACKENDS, backend_options
from ai_toolbox_cockpit.catalog import ModelCatalog
from ai_toolbox_cockpit.widgets import SearchableSelect


class ModelsView(Vertical):
    def __init__(self, catalog: ModelCatalog, **kwargs) -> None:
        super().__init__(**kwargs)
        self.catalog = catalog

    def compose(self) -> ComposeResult:
        yield Static(
            "Model catalogs are global and backend-specific: GGUF, exact DS4 artifacts, HF repositories, or workflow bundles. Launch availability follows the selected platform in Server Mode.",
            classes="view-note",
        )
        yield SearchableSelect("Select model backend", id="model-backend-select")
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

    def on_mount(self) -> None:
        select = self.query_one("#model-backend-select", SearchableSelect)
        select.set_options(backend_options())
        select.value = "llama_cpp"

    @on(SearchableSelect.Changed, "#model-backend-select")
    def backend_changed(self, event: SearchableSelect.Changed) -> None:
        self.query_one("#model-content-switcher", ContentSwitcher).current = (
            f"model-panel-{event.value}"
        )

    def set_platform(self, platform_id: str) -> None:
        for definition in BACKENDS.values():
            for panel in self.query(definition.model_panel):
                panel.set_platform(platform_id)
