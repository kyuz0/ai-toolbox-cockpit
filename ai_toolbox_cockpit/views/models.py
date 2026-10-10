from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import ContentSwitcher, Label, Static

from ai_toolbox_cockpit.backends import BACKENDS
from .engine_choices import engine_choices
from ai_toolbox_cockpit.backends.base import BackendModelPanel
from ai_toolbox_cockpit.catalog import ModelCatalog
from ai_toolbox_cockpit.widgets import SearchableSelect


class ModelsView(Vertical):
    def __init__(self, catalog: ModelCatalog, **kwargs) -> None:
        super().__init__(**kwargs)
        self.catalog = catalog

    def compose(self) -> ComposeResult:
        yield Static(
            "Choose an engine to see its curated models, quantizations and required preparation.",
            classes="model-view-copy",
        )
        with Horizontal(id="model-backend-row", classes="inline-row"):
            yield Label("Inference engine", id="model-backend-select-label", classes="inline-label")
            yield SearchableSelect("Select inference engine", id="model-backend-select")
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
        choice = engine_choices(self.app.toolbox_catalog, self.app.active_platform_id).get(str(event.value))
        switcher = self.query_one("#model-content-switcher", ContentSwitcher)
        if not choice:
            switcher.current = None
            return
        switcher.current = f"model-panel-{choice.backend_id}"
        if choice.backend_id == "vllm":
            self.query_one("#model-panel-vllm").select_engine(choice.toolbox_id, choice.label)
        self.refresh_active_panel(str(event.value))

    def refresh_active_panel(self, backend_id: str | None = None) -> None:
        if backend_id is None:
            backend_id = str(
                self.query_one("#model-backend-select", SearchableSelect).value
                or "llama_cpp"
            )
        choice = engine_choices(self.app.toolbox_catalog, self.app.active_platform_id).get(backend_id)
        if choice is None:
            return
        panel = self.query_one(f"#model-panel-{choice.backend_id}", BackendModelPanel)
        panel.refresh_inventory()

    def set_platform(self, platform_id: str) -> None:
        for definition in BACKENDS.values():
            for panel in self.query(definition.model_panel):
                panel.set_platform(platform_id)
        select = self.query_one("#model-backend-select", SearchableSelect)
        choices = engine_choices(self.app.toolbox_catalog, platform_id)
        select.set_options([(choice.label, key) for key, choice in choices.items()])
        selected = select.value if select.value in choices else next(iter(choices), "")
        select.value = selected
