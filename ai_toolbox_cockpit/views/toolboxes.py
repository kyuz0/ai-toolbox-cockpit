"""Unified toolbox catalogue and lifecycle view."""

import shlex
import subprocess

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Label, Static

from ai_toolbox_cockpit.backends import BACKENDS
from ai_toolbox_cockpit.catalog import ToolboxCatalog
from ai_toolbox_cockpit.catalog.schema import Toolbox
from ai_toolbox_cockpit.runtime.images import get_remote_image_date, is_remote_image_newer
from ai_toolbox_cockpit.runtime.terminal import command_failed, pause_after_failure
from ai_toolbox_cockpit.runtime.interactive import (
    build_create_command,
    build_delete_command,
    build_pull_command,
    detect_interactive_backend,
    InteractiveRuntime,
    runtime_environment,
)
from ai_toolbox_cockpit.runtime.toolboxes import (
    InstalledToolbox,
    create_toolbox,
    delete_toolbox,
    enter_toolbox,
    inspect_installed_toolboxes,
    runtime_for_installed_toolbox,
    run_in_toolbox_command,
)
from ai_toolbox_cockpit.settings import save_default_toolbox
from ai_toolbox_cockpit.widgets import (
    ConfirmModal,
    SearchableSelect,
    SingleClickDataTable,
    selection_marker,
)


class ToolboxesView(Vertical):
    """Manage every backend's interactive toolbox from one explicit list."""

    def __init__(self, catalog: ToolboxCatalog, platform_id: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.catalog = catalog
        self.platform_id = platform_id
        self.backend_filter = "all"
        self.channel_filter = "stable"
        self.selected_toolboxes: set[str] = set()
        self.installed: dict[str, InstalledToolbox] = {}
        self.remote_dates: dict[str, str] = {}
        self._pending_delete: tuple[Toolbox, ...] = ()
        self._pending_update: tuple[Toolbox, ...] = ()
        self._pending_create: tuple[Toolbox, ...] = ()

    def compose(self) -> ComposeResult:
        yield Static(
            "Select rows with Enter or click. Create / Update checks installed images automatically.",
            classes="toolbox-help",
        )
        with Horizontal(classes="compact-fields toolbox-filters"):
            with Vertical(classes="compact-field"):
                yield Label("Backend", id="toolbox-backend-filter-label", classes="field-label")
                yield SearchableSelect("Filter backend", id="toolbox-backend-filter")
            with Vertical(classes="compact-field"):
                yield Label("Channel", id="toolbox-channel-filter-label", classes="field-label")
                yield SearchableSelect("Filter channel", id="toolbox-channel-filter")
        with Horizontal(classes="action-row toolbox-action-row"):
            yield Button("Create / Update", id="toolbox-create-update", variant="warning")
            yield Button("Enter", id="toolbox-enter", variant="primary")
            yield Button("Model Manager", id="toolbox-model-manager")
            yield Button("Set Default", id="toolbox-set-default")
            yield Button("Refresh Status", id="toolbox-refresh")
            yield Button("Delete", id="toolbox-delete", variant="error")
        yield SingleClickDataTable(
            id="toolbox-catalog-table",
            cursor_type="row",
            zebra_stripes=True,
        )

    def on_mount(self) -> None:
        backend_filter = self.query_one("#toolbox-backend-filter", SearchableSelect)
        backend_filter.set_options(
            [("All backends", "all")]
            + [(definition.label, backend_id) for backend_id, definition in BACKENDS.items()]
        )
        backend_filter.value = "all"
        channel_filter = self.query_one("#toolbox-channel-filter", SearchableSelect)
        channel_filter.set_options([
            ("All channels", "all"),
            ("Stable", "stable"),
            ("Development", "development"),
            ("Experimental", "experimental"),
        ])
        channel_filter.value = "stable"
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
        self, toolbox: Toolbox, fallback: InteractiveRuntime | None
    ) -> InteractiveRuntime | None:
        installed = self.installed.get(toolbox.container_name)
        return runtime_for_installed_toolbox(installed) if installed else fallback

    def missing_runtime_message(self, toolbox: Toolbox) -> str:
        installed = self.installed.get(toolbox.container_name)
        if installed:
            return (
                f"Could not identify whether {toolbox.container_name} is owned by "
                f"Toolbx or Distrobox ({installed.engine.value})."
            )
        return "Install Podman with Toolbx, or Distrobox with Podman/Docker."

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
                selection_marker(toolbox.id in self.selected_toolboxes),
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
        self._refresh_action_states()

    def _refresh_action_states(self) -> None:
        selected = self.selected()
        installed = tuple(
            toolbox for toolbox in selected if toolbox.container_name in self.installed
        )
        self.query_one("#toolbox-create-update", Button).disabled = not selected
        self.query_one("#toolbox-enter", Button).disabled = not (
            len(selected) == 1 and len(installed) == 1
        )
        self.query_one("#toolbox-model-manager", Button).disabled = not (
            len(selected) == 1
            and len(installed) == 1
            and selected[0].backend == "comfyui"
        )
        self.query_one("#toolbox-set-default", Button).disabled = len(selected) != 1
        self.query_one("#toolbox-delete", Button).disabled = not installed

    @work(thread=True, exclusive=True, group="toolbox-inspection")
    def refresh_installed(self) -> None:
        installed = inspect_installed_toolboxes()
        mapped: dict[str, InstalledToolbox] = {}
        for item in installed:
            previous = mapped.get(item.name)
            if previous is None or (item.runtime is not None and previous.runtime is None):
                mapped[item.name] = item
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

    @on(SearchableSelect.Changed, "#toolbox-backend-filter")
    def backend_changed(self, event: SearchableSelect.Changed) -> None:
        self.backend_filter = str(event.value)
        self.refresh_rows()

    @on(SearchableSelect.Changed, "#toolbox-channel-filter")
    def channel_changed(self, event: SearchableSelect.Changed) -> None:
        self.channel_filter = str(event.value)
        self.refresh_rows()

    @on(Button.Pressed, "#toolbox-refresh")
    def refresh_pressed(self) -> None:
        self.refresh_installed()

    @on(Button.Pressed, "#toolbox-create-update")
    def create_update_pressed(self) -> None:
        selected = self.selected()
        if not selected:
            self.notify("Select one or more toolboxes first.", severity="warning")
            return
        if any(toolbox.container_name in self.installed for toolbox in selected):
            self.check_updates_for_create(selected)
            return
        self._prepare_create_update(selected)

    @work(thread=True, exclusive=True, group="toolbox-updates")
    def check_updates_for_create(self, toolboxes: tuple[Toolbox, ...]) -> None:
        dates = {
            toolbox.id: get_remote_image_date(toolbox.image) or "—"
            for toolbox in toolboxes
            if toolbox.container_name in self.installed
        }
        self.app.call_from_thread(self._apply_create_update_dates, toolboxes, dates)

    def _apply_create_update_dates(
        self,
        toolboxes: tuple[Toolbox, ...],
        dates: dict[str, str],
    ) -> None:
        self.remote_dates.update(dates)
        self.refresh_rows()
        self._prepare_create_update(toolboxes)

    def _prepare_create_update(self, selected: tuple[Toolbox, ...]) -> None:
        to_create: list[Toolbox] = []
        to_update: list[Toolbox] = []
        check_failed: list[Toolbox] = []
        for toolbox in selected:
            installed = self.installed.get(toolbox.container_name)
            if not installed:
                to_create.append(toolbox)
                continue
            remote = self.remote_dates.get(toolbox.id)
            if not remote or remote == "—":
                check_failed.append(toolbox)
                continue
            if remote and remote != "—" and is_remote_image_newer(remote, installed.created):
                to_update.append(toolbox)
        if not to_create and not to_update:
            if check_failed:
                names = ", ".join(toolbox.container_name for toolbox in check_failed)
                self.notify(
                    f"Could not check registry updates for: {names}.",
                    severity="error",
                )
            else:
                self.notify("The selected installed toolboxes are already up to date.")
            return
        fallback = detect_interactive_backend()
        if to_create and not fallback:
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
        if check_failed:
            update_warning += (
                "\n\nUpdate status could not be checked for these installed toolboxes, so they will be skipped:\n"
                + ", ".join(toolbox.container_name for toolbox in check_failed)
            )
        names = ", ".join(toolbox.container_name for toolbox in (*to_create, *to_update))
        commands: list[list[str]] = []
        for toolbox in to_update:
            toolbox_runtime = self.runtime_for_toolbox(toolbox, fallback)
            if not toolbox_runtime:
                self.notify(self.missing_runtime_message(toolbox), severity="error")
                return
            commands.append(build_delete_command(toolbox_runtime, toolbox.container_name))
        for toolbox in (*to_create, *to_update):
            toolbox_runtime = self.runtime_for_toolbox(toolbox, fallback)
            if not toolbox_runtime:
                self.notify(self.missing_runtime_message(toolbox), severity="error")
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
        fallback = detect_interactive_backend()
        if self._pending_create and not fallback:
            self.notify("Install Podman with Toolbx, or Distrobox with Podman/Docker.", severity="error")
            return
        operation_error: OSError | RuntimeError | subprocess.SubprocessError | None = None
        with self.app.suspend():
            try:
                for toolbox in self._pending_update:
                    toolbox_runtime = self.runtime_for_toolbox(toolbox, fallback)
                    if not toolbox_runtime:
                        raise RuntimeError(self.missing_runtime_message(toolbox))
                    print(f"Deleting {toolbox.container_name} before update...")
                    delete_toolbox(toolbox_runtime, toolbox.container_name)
                for toolbox in (*self._pending_create, *self._pending_update):
                    toolbox_runtime = self.runtime_for_toolbox(toolbox, fallback)
                    if not toolbox_runtime:
                        raise RuntimeError(self.missing_runtime_message(toolbox))
                    profile = self.catalog.runtime_profiles[toolbox.runtime_profile]
                    print(f"Pulling {toolbox.image} and creating {toolbox.container_name}...")
                    create_toolbox(toolbox_runtime, toolbox.container_name, toolbox.image, profile.engine_args)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                operation_error = error
                pause_after_failure(f"Toolbox operation failed: {error}")
        if operation_error is not None:
            self.notify(
                f"Toolbox operation failed: {operation_error}",
                severity="error",
                timeout=8,
            )
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
        runtime = runtime_for_installed_toolbox(self.installed[toolbox.container_name])
        if not runtime:
            self.notify("No compatible interactive container backend found.", severity="error")
            return
        session_error = ""
        with self.app.suspend():
            try:
                return_code = enter_toolbox(runtime, toolbox.container_name)
            except OSError as error:
                return_code = 127
                session_error = str(error)
            if command_failed(return_code):
                pause_after_failure(
                    f"Toolbox session failed: {session_error}"
                    if session_error
                    else f"Toolbox session exited with status {return_code}."
                )
        if command_failed(return_code):
            self.notify("Toolbox session failed.", severity="error", timeout=8)
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
        fallback = detect_interactive_backend()
        names = ", ".join(toolbox.container_name for toolbox in installed)
        commands: list[list[str]] = []
        for toolbox in installed:
            toolbox_runtime = self.runtime_for_toolbox(toolbox, fallback)
            if not toolbox_runtime:
                self.notify(self.missing_runtime_message(toolbox), severity="error")
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
        operation_error: OSError | RuntimeError | subprocess.SubprocessError | None = None
        with self.app.suspend():
            try:
                for toolbox in self._pending_delete:
                    installed = self.installed.get(toolbox.container_name)
                    runtime = runtime_for_installed_toolbox(installed) if installed else None
                    if not runtime:
                        raise RuntimeError(self.missing_runtime_message(toolbox))
                    print(f"Deleting {toolbox.container_name}...")
                    delete_toolbox(runtime, toolbox.container_name)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                operation_error = error
                pause_after_failure(f"Delete failed: {error}")
        if operation_error is not None:
            self.notify(
                f"Delete failed: {operation_error}", severity="error", timeout=8
            )
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
        runtime = runtime_for_installed_toolbox(self.installed[toolbox.container_name])
        if not runtime:
            self.notify("No compatible interactive container backend found.", severity="error")
            return
        command = run_in_toolbox_command(runtime, toolbox.container_name, ["model_manager"])
        manager_error = ""
        with self.app.suspend():
            try:
                return_code = subprocess.call(
                    command, env=runtime_environment(runtime)
                )
            except OSError as error:
                return_code = 127
                manager_error = str(error)
            if command_failed(return_code):
                pause_after_failure(
                    f"Model manager failed: {manager_error}"
                    if manager_error
                    else f"Model manager exited with status {return_code}."
                )
        if command_failed(return_code):
            self.notify("Model manager failed.", severity="error", timeout=8)
