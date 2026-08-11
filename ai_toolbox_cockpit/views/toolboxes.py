"""Unified toolbox catalogue and lifecycle view."""

import shlex
import subprocess

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Select, Static

from ai_toolbox_cockpit.backends import BACKENDS
from ai_toolbox_cockpit.catalog import ToolboxCatalog
from ai_toolbox_cockpit.catalog.schema import Toolbox
from ai_toolbox_cockpit.runtime.images import get_remote_image_date, is_remote_image_newer
from ai_toolbox_cockpit.runtime.interactive import (
    build_create_command,
    build_delete_command,
    build_pull_command,
    detect_interactive_backend,
    InteractiveRuntime,
    interactive_runtime_for_engine,
    runtime_environment,
)
from ai_toolbox_cockpit.runtime.toolboxes import (
    InstalledToolbox,
    create_toolbox,
    delete_toolbox,
    enter_toolbox,
    inspect_installed_toolboxes,
    run_in_toolbox_command,
)
from ai_toolbox_cockpit.settings import save_default_toolbox
from ai_toolbox_cockpit.widgets import ConfirmModal


class ToolboxesView(Vertical):
    """Manage every backend's interactive toolbox from one explicit list."""

    def __init__(self, catalog: ToolboxCatalog, platform_id: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.catalog = catalog
        self.platform_id = platform_id
        self.backend_filter = "all"
        self.channel_filter = "all"
        self.selected_toolboxes: set[str] = set()
        self.installed: dict[str, InstalledToolbox] = {}
        self.remote_dates: dict[str, str] = {}
        self._pending_delete: tuple[Toolbox, ...] = ()
        self._pending_update: tuple[Toolbox, ...] = ()
        self._pending_create: tuple[Toolbox, ...] = ()

    def compose(self) -> ComposeResult:
        yield Static(
            "Select rows with Enter/click. Create/Update pulls only selected images; updates recreate only when the registry build is newer.",
            classes="view-note",
        )
        with Horizontal(classes="filter-row"):
            yield Select(
                [("All backends", "all")]
                + [(definition.label, backend_id) for backend_id, definition in BACKENDS.items()],
                value="all",
                allow_blank=False,
                id="toolbox-backend-filter",
            )
            yield Select(
                [
                    ("All channels", "all"),
                    ("Stable", "stable"),
                    ("Development", "development"),
                    ("Experimental", "experimental"),
                ],
                value="all",
                allow_blank=False,
                id="toolbox-channel-filter",
            )
        yield DataTable(id="toolbox-catalog-table", cursor_type="row", zebra_stripes=True)
        with Horizontal(classes="action-row"):
            yield Button("Refresh", id="toolbox-refresh")
            yield Button("Check Updates", id="toolbox-check-updates")
            yield Button("Create / Update", id="toolbox-create-update", variant="warning")
            yield Button("Enter", id="toolbox-enter", variant="primary")
            yield Button("Set Default", id="toolbox-set-default")
            yield Button("Model Manager", id="toolbox-model-manager")
            yield Button("Delete", id="toolbox-delete", variant="error")

    def on_mount(self) -> None:
        table = self.query_one("#toolbox-catalog-table", DataTable)
        table.add_columns("", "Backend", "Toolbox", "Status", "Category", "Channel", "Created", "Remote", "Image")
        self.refresh_rows()
        self.refresh_installed()

    def set_platform(self, platform_id: str) -> None:
        self.platform_id = platform_id
        self.selected_toolboxes.clear()
        self.remote_dates.clear()
        if self.is_mounted:
            self.refresh_rows()
            self.refresh_installed()

    def visible_toolboxes(self) -> tuple[Toolbox, ...]:
        return tuple(
            toolbox
            for toolbox in self.catalog.platform_toolboxes(self.platform_id)
            if (self.backend_filter == "all" or toolbox.backend == self.backend_filter)
            and (self.channel_filter == "all" or toolbox.channel == self.channel_filter)
        )

    def selected(self) -> tuple[Toolbox, ...]:
        return tuple(
            self.catalog.toolboxes[toolbox_id]
            for toolbox_id in self.selected_toolboxes
            if toolbox_id in self.catalog.toolboxes
        )

    def runtime_for_toolbox(
        self, toolbox: Toolbox, fallback: InteractiveRuntime
    ) -> InteractiveRuntime | None:
        installed = self.installed.get(toolbox.container_name)
        return interactive_runtime_for_engine(installed.engine) if installed else fallback

    def refresh_rows(self) -> None:
        table = self.query_one("#toolbox-catalog-table", DataTable)
        table.clear()
        for toolbox in self.visible_toolboxes():
            installed = self.installed.get(toolbox.container_name)
            status = installed.status if installed else "Not Installed"
            remote = self.remote_dates.get(toolbox.id, "—")
            if installed and remote and remote != "—" and is_remote_image_newer(remote, installed.created):
                status = "[yellow]Needs Update[/yellow]"
            table.add_row(
                "[x]" if toolbox.id in self.selected_toolboxes else "[ ]",
                BACKENDS[toolbox.backend].label,
                toolbox.name,
                status,
                toolbox.group,
                toolbox.channel,
                installed.created[:10] if installed and installed.created else "—",
                remote[:10] if remote != "—" else remote,
                toolbox.image,
                key=toolbox.id,
            )

    @work(thread=True, exclusive=True, group="toolbox-inspection")
    def refresh_installed(self) -> None:
        installed = inspect_installed_toolboxes()
        mapped = {item.name: item for item in installed}
        self.app.call_from_thread(self._apply_installed, mapped)

    def _apply_installed(self, installed: dict[str, InstalledToolbox]) -> None:
        self.installed = installed
        self.refresh_rows()

    @on(DataTable.RowSelected, "#toolbox-catalog-table")
    def toggle_row(self, event: DataTable.RowSelected) -> None:
        toolbox_id = str(event.row_key.value)
        if toolbox_id in self.selected_toolboxes:
            self.selected_toolboxes.remove(toolbox_id)
        else:
            self.selected_toolboxes.add(toolbox_id)
        self.refresh_rows()

    @on(Select.Changed, "#toolbox-backend-filter")
    def backend_changed(self, event: Select.Changed) -> None:
        self.backend_filter = str(event.value)
        self.refresh_rows()

    @on(Select.Changed, "#toolbox-channel-filter")
    def channel_changed(self, event: Select.Changed) -> None:
        self.channel_filter = str(event.value)
        self.refresh_rows()

    @on(Button.Pressed, "#toolbox-refresh")
    def refresh_pressed(self) -> None:
        self.refresh_installed()

    @on(Button.Pressed, "#toolbox-check-updates")
    def check_updates_pressed(self) -> None:
        selected = self.selected()
        if not selected:
            self.notify("Select one or more toolboxes first.", severity="warning")
            return
        self.check_updates(selected)

    @work(thread=True, exclusive=True, group="toolbox-updates")
    def check_updates(self, toolboxes: tuple[Toolbox, ...]) -> None:
        dates = {toolbox.id: get_remote_image_date(toolbox.image) or "—" for toolbox in toolboxes}
        self.app.call_from_thread(self._apply_remote_dates, dates)

    def _apply_remote_dates(self, dates: dict[str, str]) -> None:
        self.remote_dates.update(dates)
        self.refresh_rows()
        self.notify("Image update check complete.")

    @on(Button.Pressed, "#toolbox-create-update")
    def create_update_pressed(self) -> None:
        selected = self.selected()
        if not selected:
            self.notify("Select one or more toolboxes first.", severity="warning")
            return
        to_create: list[Toolbox] = []
        to_update: list[Toolbox] = []
        for toolbox in selected:
            installed = self.installed.get(toolbox.container_name)
            if not installed:
                to_create.append(toolbox)
                continue
            remote = self.remote_dates.get(toolbox.id)
            if remote and remote != "—" and is_remote_image_newer(remote, installed.created):
                to_update.append(toolbox)
        if not to_create and not to_update:
            self.notify("Nothing selected needs to be created or updated. Run Check Updates first.")
            return
        runtime = detect_interactive_backend()
        if not runtime:
            self.notify("Install Podman with Toolbx, or Distrobox with Podman/Docker.", severity="error")
            return
        self._pending_create = tuple(to_create)
        self._pending_update = tuple(to_update)
        update_warning = ""
        if to_update:
            update_warning = (
                "\n\nThese installed toolboxes will be deleted and recreated; packages installed inside them will be lost:\n"
                + ", ".join(toolbox.container_name for toolbox in to_update)
            )
        names = ", ".join(toolbox.container_name for toolbox in (*to_create, *to_update))
        commands: list[list[str]] = []
        for toolbox in to_update:
            toolbox_runtime = self.runtime_for_toolbox(toolbox, runtime)
            if not toolbox_runtime:
                self.notify(f"No compatible wrapper for {toolbox.container_name}.", severity="error")
                return
            commands.append(build_delete_command(toolbox_runtime, toolbox.container_name))
        for toolbox in (*to_create, *to_update):
            toolbox_runtime = self.runtime_for_toolbox(toolbox, runtime)
            if not toolbox_runtime:
                self.notify(f"No compatible wrapper for {toolbox.container_name}.", severity="error")
                return
            profile = self.catalog.runtime_profiles[toolbox.runtime_profile]
            commands.append(build_pull_command(toolbox_runtime, toolbox.image))
            commands.append(
                build_create_command(
                    toolbox_runtime,
                    toolbox.container_name,
                    toolbox.image,
                    profile.engine_args,
                )
            )
        preview = "\n".join(shlex.join(command) for command in commands)
        self.app.push_screen(
            ConfirmModal(
                f"Pull images and create/update: {names}?{update_warning}\n\nCommands:\n{preview}",
                yes_text="Continue",
            ),
            self._create_update_confirmed,
        )

    def _create_update_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        runtime = detect_interactive_backend()
        if not runtime:
            self.notify("Install Podman with Toolbx, or Distrobox with Podman/Docker.", severity="error")
            return
        try:
            with self.app.suspend():
                for toolbox in self._pending_update:
                    toolbox_runtime = self.runtime_for_toolbox(toolbox, runtime)
                    if not toolbox_runtime:
                        raise RuntimeError(f"No compatible wrapper for {toolbox.container_name}")
                    print(f"Deleting {toolbox.container_name} before update...")
                    delete_toolbox(toolbox_runtime, toolbox.container_name)
                for toolbox in (*self._pending_create, *self._pending_update):
                    toolbox_runtime = self.runtime_for_toolbox(toolbox, runtime)
                    if not toolbox_runtime:
                        raise RuntimeError(f"No compatible wrapper for {toolbox.container_name}")
                    profile = self.catalog.runtime_profiles[toolbox.runtime_profile]
                    print(f"Pulling {toolbox.image} and creating {toolbox.container_name}...")
                    create_toolbox(toolbox_runtime, toolbox.container_name, toolbox.image, profile.engine_args)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            self.notify(f"Toolbox operation failed: {error}", severity="error", timeout=8)
        else:
            self.notify("Toolbox operation complete.", severity="information")
        self.selected_toolboxes.clear()
        self.refresh_installed()

    @on(Button.Pressed, "#toolbox-enter")
    def enter_pressed(self) -> None:
        selected = self.selected()
        if len(selected) != 1:
            self.notify("Select exactly one installed toolbox to enter.", severity="warning")
            return
        toolbox = selected[0]
        if toolbox.container_name not in self.installed:
            self.notify("That toolbox is not installed.", severity="warning")
            return
        runtime = interactive_runtime_for_engine(self.installed[toolbox.container_name].engine)
        if not runtime:
            self.notify("No compatible interactive container backend found.", severity="error")
            return
        with self.app.suspend():
            enter_toolbox(runtime, toolbox.container_name)
        self.refresh_installed()

    @on(Button.Pressed, "#toolbox-delete")
    def delete_pressed(self) -> None:
        installed = tuple(
            toolbox for toolbox in self.selected() if toolbox.container_name in self.installed
        )
        if not installed:
            self.notify("Select one or more installed toolboxes.", severity="warning")
            return
        self._pending_delete = installed
        runtime = detect_interactive_backend()
        if not runtime:
            self.notify("No compatible interactive container backend found.", severity="error")
            return
        names = ", ".join(toolbox.container_name for toolbox in installed)
        commands: list[list[str]] = []
        for toolbox in installed:
            toolbox_runtime = self.runtime_for_toolbox(toolbox, runtime)
            if not toolbox_runtime:
                self.notify(f"No compatible wrapper for {toolbox.container_name}.", severity="error")
                return
            commands.append(build_delete_command(toolbox_runtime, toolbox.container_name))
        preview = "\n".join(shlex.join(command) for command in commands)
        self.app.push_screen(
            ConfirmModal(f"Delete these toolboxes?\n{names}\n\nCommands:\n{preview}", yes_text="Delete"),
            self._delete_confirmed,
        )

    def _delete_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        try:
            with self.app.suspend():
                for toolbox in self._pending_delete:
                    installed = self.installed.get(toolbox.container_name)
                    runtime = interactive_runtime_for_engine(installed.engine) if installed else None
                    if not runtime:
                        raise RuntimeError(f"No compatible wrapper for {toolbox.container_name}")
                    print(f"Deleting {toolbox.container_name}...")
                    delete_toolbox(runtime, toolbox.container_name)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            self.notify(f"Delete failed: {error}", severity="error", timeout=8)
        self.selected_toolboxes.clear()
        self.refresh_installed()

    @on(Button.Pressed, "#toolbox-set-default")
    def set_default_pressed(self) -> None:
        selected = self.selected()
        if len(selected) != 1:
            self.notify("Select exactly one toolbox to make the backend default.", severity="warning")
            return
        toolbox = selected[0]
        if save_default_toolbox(toolbox.backend, self.platform_id, toolbox.id):
            self.notify(f"{toolbox.name} is now the {BACKENDS[toolbox.backend].label} default for this platform.")
            self.app.query_one("#servers-view").set_platform(self.platform_id)
        else:
            self.notify("Could not save the default toolbox.", severity="error")

    @on(Button.Pressed, "#toolbox-model-manager")
    def model_manager_pressed(self) -> None:
        selected = self.selected()
        if len(selected) != 1:
            self.notify("Select exactly one installed ComfyUI toolbox.", severity="warning")
            return
        toolbox = selected[0]
        if toolbox.backend != "comfyui" or toolbox.container_name not in self.installed:
            self.notify("The toolbox model manager is currently provided by installed ComfyUI toolboxes.", severity="warning")
            return
        runtime = interactive_runtime_for_engine(self.installed[toolbox.container_name].engine)
        if not runtime:
            self.notify("No compatible interactive container backend found.", severity="error")
            return
        command = run_in_toolbox_command(runtime, toolbox.container_name, ["model_manager"])
        with self.app.suspend():
            subprocess.call(command, env=runtime_environment(runtime))
