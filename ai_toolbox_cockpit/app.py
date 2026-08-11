import pyfiglet

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.theme import Theme
from textual.widgets import Footer, Header, Label, Static, TabbedContent, TabPane

from .catalog import load_model_catalog, load_toolbox_catalog
from .settings import load_active_platform, save_active_platform
from .updates import available_update, installed_version
from .views import ModelsView, ServersView, ToolboxesView
from .widgets import ConfirmModal, SearchableSelect, SelectModal


def generate_banner(version: str) -> str:
    ascii_art = pyfiglet.figlet_format("AI Toolbox Cockpit", font="small").rstrip()
    return (
        f"[bold #e57373]{ascii_art}[/]\n"
        f"[dim]Local AI toolboxes and model servers  ·  v{version}[/dim]"
    )


class AiToolboxCockpitApp(App):
    TITLE = "AI Toolbox Cockpit"
    SUB_TITLE = "Local AI toolboxes and model servers"

    CSS = """
    DataTable > .datatable--cursor {
        background: #333333;
        color: auto;
        text-style: none;
    }

    DataTable.inactive-table > .datatable--cursor {
        background: transparent;
        text-style: none;
    }

    DataTable > .datatable--header {
        background: #2a2a2a;
    }

    OptionList > .option-list--option-highlighted {
        background: transparent;
        color: #e57373;
        text-style: bold;
    }

    Header {
        background: #d32f2f;
    }

    Tab, Tab:hover, Tab:focus, Tab.-active {
        background: transparent !important;
    }

    Tab:focus {
        color: #e57373 !important;
        text-style: bold;
    }

    Underline > .underline--active,
    Tabs .underline--active,
    Tabs:focus .underline--active {
        background: #d32f2f !important;
    }

    Tab.-active {
        color: #e57373 !important;
    }

    #title-banner {
        text-align: center;
        margin-bottom: 1;
        padding: 0 1;
        height: auto;
        text-style: bold;
        color: #e57373;
    }

    #platform-row {
        align: center middle;
        height: auto;
        margin-bottom: 1;
    }

    #platform-row Label {
        width: auto;
        margin-right: 2;
        text-style: bold;
        color: #e57373;
    }

    #platform-select {
        width: 1fr;
        max-width: 60;
    }

    TabbedContent { height: 1fr; }
    TabPane { padding: 1 2; }

    .view-note {
        height: auto;
        margin-bottom: 1;
        padding: 1 2;
        background: $surface;
        border: round #d32f2f;
        color: $text;
        text-style: bold;
        text-align: center;
    }

    .filter-row {
        height: auto;
        max-height: 3;
        margin: 1 0;
        align: left middle;
    }

    .filter-row SearchableSelect {
        width: 1fr;
        margin-right: 2;
    }

    .action-row {
        margin: 1 0;
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
        margin-top: 1;
    }

    .inline-row .inline-label {
        width: auto;
        min-width: 12;
        text-style: bold;
        color: #e57373;
        padding-right: 1;
        height: 1;
        content-align: left middle;
    }

    .inline-row SearchableSelect, .inline-row Input { width: 1fr; }

    .settings-row {
        height: auto;
        max-height: 3;
        margin-top: 1;
    }

    .settings-row Input, .settings-row SearchableSelect {
        width: 1fr;
        margin-right: 2;
    }

    .settings-row Switch {
        margin-right: 1;
    }

    .server-settings {
        height: auto;
        margin-top: 1;
    }

    .settings-title {
        height: 1;
        color: #e57373;
        text-style: bold;
    }

    .compact-fields {
        height: 2;
        margin-top: 1;
    }

    .compact-field {
        width: 1fr;
        height: 2;
        margin-right: 2;
    }

    .compact-field .field-label {
        height: 1;
        color: #bdbdbd;
        text-style: bold;
    }

    .compact-field Input, .compact-field SearchableSelect {
        width: 100%;
        height: 1;
    }

    .options-row {
        height: auto;
        max-height: 3;
        margin-top: 1;
    }

    .options-row Switch, .options-row Checkbox {
        margin-right: 4;
    }

    #vllm-eager {
        padding: 0;
    }

    Input, Checkbox, Switch {
        margin: 0;
        height: 1;
        border: none;
    }

    .model-zone {
        height: auto;
        padding: 1 2;
        margin: 1 0;
        border: round #333333;
        background: #1e1e1e;
    }

    .model-zone:focus-within { border: round #d32f2f; }
    .zone-title {
        height: auto;
        width: 100%;
        color: #e57373;
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
        border: solid #d32f2f;
        background: #1e1e1e;
        padding: 1 2;
    }

    #select_dialog {
        width: 90%;
        max-width: 100;
        height: 80%;
        border: solid #d32f2f;
        background: #1e1e1e;
        padding: 1 2;
    }

    #confirm_message, #select_title {
        text-align: center;
        text-style: bold;
        color: #e57373;
        margin-bottom: 1;
        width: 100%;
    }

    #confirm_buttons, #select_buttons { height: auto; align: center middle; }
    #select_list {
        border: solid #d32f2f;
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
        color: #e57373;
    }

    .panel-copy, .storage-copy {
        height: auto;
        margin-bottom: 1;
    }

    .support-state {
        height: auto;
        color: #9e9e9e;
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
        color: #bdbdbd;
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
        padding: 0 1;
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(generate_banner(self.version), id="title-banner")
        with Horizontal(id="platform-row"):
            yield Label("Platform")
            yield SearchableSelect("Select hardware platform", id="platform-select")
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
            name="cockpit-red",
            primary="#d32f2f",
            secondary="#b71c1c",
            accent="#e57373",
            foreground="#ffffff",
            background="#121212",
            surface="#1e1e1e",
            panel="#2a2a2a",
            warning="#ffa000",
            error="#d32f2f",
            success="#4caf50",
            dark=True,
        )
        self.register_theme(theme)
        self.theme = "cockpit-red"
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
            self.app.call_from_thread(
                self.notify,
                f"AI Toolbox Cockpit v{latest} is available. Run: pipx upgrade ai-toolbox-cockpit",
                severity="warning",
                timeout=12,
            )

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
