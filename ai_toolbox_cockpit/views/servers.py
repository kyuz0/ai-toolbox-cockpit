from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import ContentSwitcher, Select, Static

from ai_toolbox_cockpit.backends import BACKENDS, backend_options
from ai_toolbox_cockpit.backends.base import BackendServerPanel


class ServersView(Vertical):
    def compose(self) -> ComposeResult:
        yield Static(
            "Each backend owns its launch controls and command builder. Every launch shows the exact command for confirmation.",
            classes="view-note",
        )
        yield Static("", id="server-platform-support", classes="support-state")
        yield Select(
            backend_options(),
            value="llama_cpp",
            allow_blank=False,
            id="server-backend-select",
        )
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

    @on(Select.Changed, "#server-backend-select")
    def backend_changed(self, event: Select.Changed) -> None:
        self.query_one("#server-content-switcher", ContentSwitcher).current = (
            f"server-panel-{event.value}"
        )

    def set_platform(self, platform_id: str) -> None:
        platform = self.app.toolbox_catalog.platform(platform_id)
        support: list[str] = []
        for backend_id, definition in BACKENDS.items():
            candidates = [
                toolbox
                for toolbox in self.app.toolbox_catalog.platform_toolboxes(platform_id)
                if toolbox.backend == backend_id
                and toolbox.feature_state("server") != "unavailable"
            ]
            if not candidates:
                state = "unavailable"
            elif any(toolbox.maturity == "stable" for toolbox in candidates):
                state = "supported"
            else:
                state = "experimental"
            support.append(f"{definition.label}: {state}")
        self.query_one("#server-platform-support", Static).update(
            f"{platform.name} — " + "  ·  ".join(support)
        )
        for panel in self.query(BackendServerPanel):
            panel.set_platform(platform_id)
