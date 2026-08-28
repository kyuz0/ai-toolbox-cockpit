from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, Static

from ai_toolbox_cockpit.catalog import ModelBackendCatalog


class BackendServerPanel(Vertical):
    backend_label = "Backend"
    support_state = "Backend-owned"
    summary = ""
    next_gate = ""

    def compose(self) -> ComposeResult:
        yield Label(self.backend_label, classes="panel-title")
        yield Static(f"Status: {self.support_state}", classes="support-state")
        yield Static(self.summary, classes="panel-copy")
        yield Static(f"Next validation gate: {self.next_gate}", classes="panel-copy")

    def set_platform(self, platform_id: str) -> None:
        """Refresh platform-owned choices; concrete panels override as needed."""

    def refresh_model_inventory(self) -> None:
        """Refresh backend-owned model controls after its inventory changes."""


class BackendModelPanel(Vertical):
    backend_label = "Backend"
    summary = ""

    def __init__(self, catalog: ModelBackendCatalog, **kwargs) -> None:
        super().__init__(**kwargs)
        self.catalog = catalog

    def refresh_inventory(self) -> None:
        """Refresh backend-owned local/cache state when its Models panel is activated."""

    def set_platform(self, platform_id: str) -> None:
        """Refresh platform-owned choices; most model catalogs are global."""


@dataclass(frozen=True)
class BackendDefinition:
    id: str
    label: str
    server_panel: type[BackendServerPanel]
    model_panel: type[BackendModelPanel]
