from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import ContentSwitcher, Label, Static

from ai_toolbox_cockpit.backends import BACKENDS, backend_options
from ai_toolbox_cockpit.backends.base import BackendModelPanel
from ai_toolbox_cockpit.catalog import ModelCatalog
from ai_toolbox_cockpit.widgets import SearchableSelect


class ModelsView(Vertical):
    def __init__(self, catalog: ModelCatalog, **kwargs) -> None:
        super().__init__(**kwargs)
        self.catalog = catalog

    def compose(self) -> ComposeResult:
        yield Static(
            "Model handling follows the selected backend: GGUF files, Halogen HGN bundles, Hugging Face repositories, or ComfyUI workflow bundles.",
            classes="model-view-copy",
        )
        with Horizontal(id="model-backend-row", classes="inline-row"):
            yield Label("Backend", id="model-backend-select-label", classes="inline-label")
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
        self.set_platform(self.app.active_platform_id)

    @on(SearchableSelect.Changed, "#model-backend-select")
    def backend_changed(self, event: SearchableSelect.Changed) -> None:
        backend_id = str(event.value)
        switcher = self.query_one("#model-content-switcher", ContentSwitcher)
        if backend_id not in BACKENDS:
            switcher.current = None
            return
        switcher.current = f"model-panel-{backend_id}"
        self.refresh_active_panel(backend_id)

    def refresh_active_panel(self, backend_id: str | None = None) -> None:
        if backend_id is None:
            backend_id = str(
                self.query_one("#model-backend-select", SearchableSelect).value
                or "llama_cpp"
            )
        panel = self.query_one(f"#model-panel-{backend_id}", BackendModelPanel)
        panel.refresh_inventory()

    def set_platform(self, platform_id: str) -> None:
        for definition in BACKENDS.values():
            for panel in self.query(definition.model_panel):
                panel.set_platform(platform_id)
        select = self.query_one("#model-backend-select", SearchableSelect)
        backend_ids = self.app.toolbox_catalog.platform_backend_ids(platform_id)
        select.set_options(backend_options(backend_ids))
        selected = select.value if select.value in backend_ids else (
            backend_ids[0] if backend_ids else ""
        )
        select.value = selected
