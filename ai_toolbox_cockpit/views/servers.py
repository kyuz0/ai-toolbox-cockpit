from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import ContentSwitcher, Label, Static

from ai_toolbox_cockpit.backends import BACKENDS, backend_options
from ai_toolbox_cockpit.backends.base import BackendServerPanel
from ai_toolbox_cockpit.widgets import SearchableSelect


class ServersView(Vertical):
    def compose(self) -> ComposeResult:
        yield Static(
            "Each backend owns its launch controls and command builder. Every launch shows the exact command for confirmation.",
            classes="view-note",
        )
        with Horizontal(id="server-backend-row", classes="inline-row"):
            yield Label("Inference engine", id="server-backend-select-label", classes="inline-label")
            yield SearchableSelect("Select inference engine", id="server-backend-select")
        panels = [
            definition.server_panel(id=f"server-panel-{backend_id}")
            for backend_id, definition in BACKENDS.items()
        ]
        yield ContentSwitcher(
            *panels,
            initial="server-panel-llama_cpp",
            id="server-content-switcher",
        )

    def on_mount(self) -> None:
        self.set_platform(self.app.active_platform_id)

    @on(SearchableSelect.Changed, "#server-backend-select")
    def backend_changed(self, event: SearchableSelect.Changed) -> None:
        backend_id = str(event.value)
        switcher = self.query_one("#server-content-switcher", ContentSwitcher)
        switcher.current = (
            f"server-panel-{backend_id}" if backend_id in BACKENDS else None
        )

    def set_platform(self, platform_id: str) -> None:
        for panel in self.query(BackendServerPanel):
            panel.set_platform(platform_id)
        select = self.query_one("#server-backend-select", SearchableSelect)
        backend_ids = self.app.toolbox_catalog.platform_backend_ids(platform_id)
        select.set_options(backend_options(backend_ids))
        selected = select.value if select.value in backend_ids else (
            backend_ids[0] if backend_ids else ""
        )
        select.value = selected

    def refresh_model_inventory(self, backend_id: str) -> None:
        definition = BACKENDS.get(backend_id)
        if definition is None:
            return
        panel = self.query_one(
            f"#server-panel-{backend_id}",
            definition.server_panel,
        )
        panel.refresh_model_inventory()
