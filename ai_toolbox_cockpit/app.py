from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.theme import Theme
from textual.widgets import Footer, Header, Label, Select, Static, TabbedContent, TabPane

from .catalog import load_model_catalog, load_toolbox_catalog
from .settings import load_active_platform, save_active_platform
from .updates import available_update, installed_version
from .views import BenchmarksView, ModelsView, ServersView, ToolboxesView


class AiToolboxCockpitApp(App):
    TITLE = "AI Toolbox Cockpit"
    SUB_TITLE = "Local AI toolboxes and model servers"

    CSS = """
    Screen {
        background: #121212;
        color: #f5f5f5;
    }

    #title-banner {
        height: 3;
        padding: 1 2;
        text-style: bold;
        color: #ff8a80;
        background: #1e1e1e;
    }

    #platform-row {
        height: 3;
        padding: 0 2;
        align: left middle;
        background: #1e1e1e;
    }

    #platform-row Label {
        width: auto;
        margin-right: 1;
        text-style: bold;
    }

    #platform-select {
        width: 1fr;
        max-width: 60;
    }

    TabbedContent { height: 1fr; }
    TabPane { padding: 1 2; }
    Tab, Tab:hover, Tab:focus, Tab.-active { background: transparent !important; }

    .view-note {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        background: #242424;
        color: #d0d0d0;
    }

    .filter-row {
        height: 3;
        margin-bottom: 1;
    }

    .filter-row Select {
        width: 1fr;
        margin-right: 1;
    }

    .action-row, .inline-row, .settings-row, .options-row {
        height: auto;
        min-height: 3;
        margin-bottom: 1;
        align: left middle;
    }

    .action-row Button { margin-right: 1; }

    .inline-label {
        width: 18;
        min-width: 14;
        text-style: bold;
        color: #ff8a80;
        content-align: left middle;
    }

    .inline-row SearchableSelect, .inline-row Input { width: 1fr; }
    .inline-row Button { margin-left: 1; }
    .settings-row Input, .settings-row SearchableSelect { width: 1fr; margin-right: 1; }
    .settings-row Switch, .options-row Switch, .options-row Checkbox { margin-right: 1; }

    Input { height: 3; }
    SearchableSelect { height: 3; }

    .model-zone {
        height: auto;
        min-height: 5;
        padding: 1;
        margin-bottom: 1;
        border: round #333333;
        background: #1e1e1e;
    }

    .model-zone:focus-within { border: round #d32f2f; }
    .zone-title { height: auto; color: #ff8a80; text-style: bold; margin-bottom: 1; }

    ConfirmModal, SelectModal { align: center middle; background: rgba(0, 0, 0, 0.72); }
    #confirm_dialog, #select_dialog {
        width: 90%;
        max-width: 110;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: solid #d32f2f;
        background: #1e1e1e;
    }
    #select_dialog { height: 80%; }
    #confirm_message, #select_title { height: auto; margin-bottom: 1; }
    #confirm_buttons, #select_buttons { height: auto; align: center middle; }
    #select_list { height: 1fr; min-height: 10; }

    DataTable { height: 1fr; }
    ContentSwitcher { height: 1fr; }

    .panel-title {
        height: 2;
        text-style: bold;
        color: #ff8a80;
    }

    .panel-copy, .storage-copy {
        height: auto;
        margin-bottom: 1;
    }

    .split-row {
        height: 1fr;
    }

    .split-row > Vertical {
        width: 1fr;
        margin-right: 1;
    }

    VerticalScroll {
        height: 1fr;
    }
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
        yield Static(
            f"AI TOOLBOX COCKPIT  ·  local containers, models, and servers  ·  v{self.version}",
            id="title-banner",
        )
        with Horizontal(id="platform-row"):
            yield Label("Platform")
            yield Select(
                [
                    (f"{platform.name} — {platform.description}", platform.id)
                    for platform in self.toolbox_catalog.platforms
                ],
                value=self.active_platform_id,
                allow_blank=False,
                id="platform-select",
            )
        with TabbedContent(initial="tab-toolboxes"):
            with TabPane("Toolboxes", id="tab-toolboxes"):
                yield ToolboxesView(
                    self.toolbox_catalog,
                    self.active_platform_id,
                    id="toolboxes-view",
                )
            with TabPane("Servers", id="tab-servers"):
                yield ServersView(id="servers-view")
            with TabPane("Models", id="tab-models"):
                yield ModelsView(self.model_catalog, id="models-view")
            with TabPane("Benchmarks", id="tab-benchmarks"):
                yield BenchmarksView(self.active_platform_id, id="benchmarks-view")
        yield Footer()

    def on_mount(self) -> None:
        theme = Theme(
            name="cockpit-red",
            primary="#d32f2f",
            secondary="#b71c1c",
            accent="#ff8a80",
            foreground="#f5f5f5",
            background="#121212",
            surface="#1e1e1e",
            panel="#242424",
            warning="#ffca28",
            error="#ef5350",
            success="#66bb6a",
            dark=True,
        )
        self.register_theme(theme)
        self.theme = "cockpit-red"
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

    @on(Select.Changed, "#platform-select")
    def platform_changed(self, event: Select.Changed) -> None:
        platform_id = str(event.value)
        self.active_platform_id = platform_id
        save_active_platform(platform_id)
        self.query_one("#toolboxes-view", ToolboxesView).set_platform(platform_id)
        self.query_one("#servers-view", ServersView).set_platform(platform_id)
        self.query_one("#models-view", ModelsView).set_platform(platform_id)
        self.query_one("#benchmarks-view", BenchmarksView).set_platform(platform_id)
