import shlex
import subprocess

import pyfiglet

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.theme import Theme
from textual.widgets import Button, Footer, Header, Label, Static, TabbedContent, TabPane

from .catalog import load_model_catalog, load_toolbox_catalog
from .settings import load_active_platform, save_active_platform
from .runtime.terminal import pause_after_failure
from .updates import (
    RELAUNCH_AFTER_UPDATE,
    UPGRADE_COMMAND,
    available_update,
    installed_version,
)
from .views import ModelsView, ServersView, ToolboxesView
from .widgets import ConfirmModal, SearchableSelect, SelectModal


def generate_banner(version: str) -> str:
    ascii_art = pyfiglet.figlet_format("AI Toolbox Cockpit", font="small").rstrip()
    return (
        f"[bold #f2b544]{ascii_art}[/]\n"
        f"[dim]Local AI toolboxes and model servers  ·  v{version}[/dim]"
    )


class AiToolboxCockpitApp(App):
    TITLE = "AI Toolbox Cockpit"
    SUB_TITLE = "Local AI toolboxes and model servers"

    CSS = """
    DataTable > .datatable--cursor {
        background: #34383d;
        color: auto;
        text-style: none;
    }

    DataTable.inactive-table > .datatable--cursor {
        background: transparent;
        text-style: none;
    }

    DataTable > .datatable--header {
        background: #34383d;
    }

    OptionList > .option-list--option-highlighted {
        background: transparent;
        color: #f2b544;
        text-style: bold;
    }

    Header {
        background: #f2b544;
        color: #171a1d;
    }

    Tab, Tab:hover, Tab:focus, Tab.-active {
        background: transparent !important;
    }

    Tab:focus {
        color: #f2b544 !important;
        text-style: bold;
    }

    Underline > .underline--active,
    Tabs .underline--active,
    Tabs:focus .underline--active {
        background: #f2b544 !important;
    }

    Tab.-active {
        color: #f2b544 !important;
    }

    #title-banner {
        text-align: center;
        margin-bottom: 1;
        padding: 0 1;
        height: auto;
        text-style: bold;
        color: #f2b544;
    }

    #platform-row {
        align: center middle;
        height: auto;
        margin: 0 2 1 2;
    }

    #platform-row Label {
        width: auto;
        margin-right: 2;
        text-style: bold;
        color: #f2b544;
    }

    #platform-select {
        width: 1fr;
        max-width: 60;
    }

    #application-update-row {
        display: none;
        height: auto;
        margin: 0 2 1 2;
        padding: 1 2;
        align: center middle;
        border: round #f2b544;
        background: #25292e;
    }

    #application-update-message {
        width: 1fr;
        height: auto;
        color: #f2b544;
        text-style: bold;
    }

    #application-update-run {
        width: auto;
        min-width: 14;
        margin-left: 2;
    }

    TabbedContent { height: 1fr; }
    TabPane { padding: 1 2; }

    .view-note {
        height: auto;
        margin-bottom: 1;
        padding: 1 2;
        background: $surface;
        border: round #f2b544;
        color: $text;
        text-style: bold;
        text-align: center;
    }

    .toolbox-help {
        height: 1;
        margin-bottom: 1;
        color: #8f969e;
    }

    .toolbox-filters {
        margin-top: 0;
        margin-bottom: 1;
    }

    .toolbox-action-row {
        margin: 0 0 1 0;
    }

    #toolbox-refresh, #toolbox-delete {
        margin-left: 2;
    }

    .action-row {
        margin: 1 1;
        height: auto;
        align: left middle;
    }

    Button {
        margin-right: 1;
        height: 1;
        border: none;
        min-width: 12;
    }

    .inline-row {
        height: auto;
        max-height: 5;
        margin: 1 1;
    }

    .inline-row .inline-label {
        width: auto;
        min-width: 12;
        text-style: bold;
        color: #f2b544;
        margin-right: 1;
        height: 1;
        content-align: left middle;
    }

    .inline-row SearchableSelect, .inline-row Input {
        width: 1fr;
        margin-right: 1;
    }

    .server-settings {
        height: auto;
        margin: 1 1;
        padding: 0 1 1 1;
    }

    .settings-title {
        height: 1;
        margin-bottom: 1;
        color: #f2b544;
        text-style: bold;
    }

    .compact-fields {
        height: auto;
        min-height: 2;
        margin: 1 1;
    }

    .compact-field {
        width: 1fr;
        height: 2;
        margin: 0 2 0 0;
    }

    .compact-field:last-child { margin-right: 0; }

    .compact-field .field-label {
        height: 1;
        color: #8f969e;
        text-style: bold;
    }

    .compact-field Input, .compact-field SearchableSelect {
        width: 100%;
        height: 1;
    }

    .options-row {
        height: auto;
        max-height: 3;
        margin: 1 1;
    }

    .options-row Checkbox {
        margin-right: 4;
    }

    #vllm-eager {
        padding: 0;
    }

    Input, Checkbox {
        margin: 0;
        height: 1;
        border: none;
    }

    .model-zone {
        height: auto;
        padding: 1 2;
        margin: 1 0;
        border: round #34383d;
        background: #25292e;
    }

    .model-zone:focus-within { border: round #f2b544; }
    .zone-title {
        height: auto;
        width: 100%;
        color: #f2b544;
        text-style: bold;
        background: transparent;
        margin: 0 0 1 0;
    }

    ConfirmModal, SelectModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }

    #confirm_dialog {
        width: 90%;
        max-width: 100;
        height: auto;
        border: solid #f2b544;
        background: #25292e;
        padding: 1 2;
    }

    #select_dialog {
        width: 90%;
        max-width: 100;
        height: 80%;
        border: solid #f2b544;
        background: #25292e;
        padding: 1 2;
    }

    #confirm_message, #select_title {
        text-align: center;
        text-style: bold;
        color: #f2b544;
        margin-bottom: 1;
        width: 100%;
    }

    #confirm_buttons, #select_buttons { height: auto; align: center middle; }
    #select_list {
        border: solid #f2b544;
        height: 1fr;
        min-height: 10;
        margin-bottom: 1;
    }

    DataTable {
        height: 1fr;
        border: none;
    }

    #toolbox-catalog-table {
        height: 1fr;
        margin-bottom: 1;
    }

    ContentSwitcher { height: 1fr; }

    .panel-title {
        height: auto;
        margin: 1 0;
        text-style: bold;
        color: #f2b544;
    }

    .panel-copy, .storage-copy {
        height: auto;
        margin-bottom: 1;
    }

    .support-state {
        height: auto;
        color: #8f969e;
        margin-bottom: 1;
    }

    #server-backend-row {
        margin: 0 0 1 0;
        align: left middle;
    }

    #server-backend-select {
        width: 32;
        max-width: 40;
    }

    .model-view-copy {
        height: auto;
        margin-bottom: 1;
        color: #8f969e;
    }

    #model-backend-row {
        margin: 0 0 1 0;
        align: left middle;
    }

    #model-backend-select {
        width: 32;
        max-width: 40;
    }

    #server-content-switcher, #model-content-switcher {
        padding: 0 2 1 2;
    }

    VerticalScroll { height: 1fr; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.toolbox_catalog = load_toolbox_catalog()
        self.model_catalog = load_model_catalog()
        fallback = self.toolbox_catalog.platforms[0].id
        configured = load_active_platform(fallback)
        available = {platform.id for platform in self.toolbox_catalog.platforms}
        self.active_platform_id = configured if configured in available else fallback
        self.version = installed_version()
        self._available_version = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(generate_banner(self.version), id="title-banner")
        with Horizontal(id="platform-row"):
            yield Label("Platform", id="platform-select-label")
            yield SearchableSelect("Select hardware platform", id="platform-select")
        with Horizontal(id="application-update-row"):
            yield Label("", id="application-update-message")
            yield Button("Upgrade now", id="application-update-run", variant="primary")
        with TabbedContent(initial="tab-toolboxes"):
            with TabPane("Toolboxes", id="tab-toolboxes"):
                yield ToolboxesView(
                    self.toolbox_catalog,
                    self.active_platform_id,
                    id="toolboxes-view",
                )
            with TabPane("Server Mode", id="tab-servers"):
                yield ServersView(id="servers-view")
            with TabPane("Models", id="tab-models"):
                yield ModelsView(self.model_catalog, id="models-view")
        yield Footer()

    def on_mount(self) -> None:
        theme = Theme(
            name="cockpit-gold",
            primary="#f2b544",
            secondary="#8a5a00",
            accent="#f2b544",
            foreground="#f1f3f5",
            background="#171a1d",
            surface="#25292e",
            panel="#34383d",
            warning="#f2b544",
            error="#b75b52",
            success="#4f8a62",
            dark=True,
        )
        self.register_theme(theme)
        self.theme = "cockpit-gold"
        platform_select = self.query_one("#platform-select", SearchableSelect)
        platform_select.set_options([
            (f"{platform.name} — {platform.description}", platform.id)
            for platform in self.toolbox_catalog.platforms
        ])
        platform_select.value = self.active_platform_id
        self.check_application_update()

    @work(thread=True, exclusive=True, group="application-update")
    def check_application_update(self) -> None:
        latest = available_update(self.version)
        if latest:
            self.app.call_from_thread(self._show_application_update, latest)

    def _show_application_update(self, latest: str) -> None:
        self._available_version = latest
        command = shlex.join(UPGRADE_COMMAND)
        self.query_one("#application-update-message", Label).update(
            f"AI Toolbox Cockpit v{latest} is available. Command: {command}"
        )
        button = self.query_one("#application-update-run", Button)
        button.label = "Upgrade now"
        button.disabled = False
        self.query_one("#application-update-row", Horizontal).styles.display = "block"
        self.notify(
            f"AI Toolbox Cockpit v{latest} is available. Choose Upgrade now or run: {command}",
            severity="warning",
            timeout=12,
        )

    @on(Button.Pressed, "#application-update-run")
    def application_update_pressed(self) -> None:
        command = shlex.join(UPGRADE_COMMAND)
        self.push_screen(
            ConfirmModal(
                f"Upgrade AI Toolbox Cockpit to v{self._available_version}?\n\n{command}",
                yes_text="Upgrade",
            ),
            self._application_update_confirmed,
        )

    def _application_update_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        button = self.query_one("#application-update-run", Button)
        message = self.query_one("#application-update-message", Label)
        button.disabled = True
        message.update(f"Updating to AI Toolbox Cockpit v{self._available_version}…")
        update_error: OSError | subprocess.SubprocessError | None = None
        with self.suspend():
            try:
                print(f"Running: {shlex.join(UPGRADE_COMMAND)}")
                subprocess.run(list(UPGRADE_COMMAND), check=True)
            except (OSError, subprocess.SubprocessError) as error:
                update_error = error
                pause_after_failure(f"Application update failed: {error}")
        if update_error is not None:
            button.disabled = False
            message.update(
                f"Update failed. Run manually: {shlex.join(UPGRADE_COMMAND)}"
            )
            self.notify(
                f"Application update failed: {update_error}",
                severity="error",
                timeout=8,
            )
        else:
            button.label = "Updated"
            message.update(
                f"AI Toolbox Cockpit v{self._available_version} installed. Relaunching…"
            )
            self.notify("Application update complete. Relaunching AI Toolbox Cockpit.")
            self.exit(result=RELAUNCH_AFTER_UPDATE)

    def refresh_server_model_inventory(self, backend_id: str) -> None:
        """Refresh backend-owned Server Mode model controls after inventory changes."""
        self.query_one("#servers-view", ServersView).refresh_model_inventory(backend_id)

    @on(SearchableSelect.Changed, "#platform-select")
    def platform_changed(self, event: SearchableSelect.Changed) -> None:
        platform_id = str(event.value)
        self.active_platform_id = platform_id
        save_active_platform(platform_id)
        self.query_one("#toolboxes-view", ToolboxesView).set_platform(platform_id)
        self.query_one("#servers-view", ServersView).set_platform(platform_id)
        self.query_one("#models-view", ModelsView).set_platform(platform_id)

    @on(TabbedContent.TabActivated)
    def tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.pane.id == "tab-models":
            self.query_one("#models-view", ModelsView).refresh_active_panel()
