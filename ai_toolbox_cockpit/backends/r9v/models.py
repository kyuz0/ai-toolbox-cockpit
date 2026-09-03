"""R9V immutable package download, verification, and PLE preparation UI."""

import shlex
import subprocess

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Label, Static

from ai_toolbox_cockpit.backends.base import BackendModelPanel
from ai_toolbox_cockpit.huggingface import get_hf_token, huggingface_environment
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.terminal import pause_after_failure
from ai_toolbox_cockpit.storage import disk_space_for_path, download_space_note
from ai_toolbox_cockpit.widgets import ConfirmModal, SearchableSelect

from .model_manager import (
    build_prepare_ple_cmd,
    get_download_cmd,
    get_models_dir,
    invalid_artifacts,
    package_dir,
    package_is_complete,
    ple_is_complete,
    ple_path,
    save_models_dir,
    verify_package,
    verify_ple,
)


class R9vModelPanel(BackendModelPanel):
    backend_label = "R9V Models"

    def __init__(self, catalog, **kwargs) -> None:
        super().__init__(catalog, **kwargs)
        self.platform_id = ""
        self._pending_command: list[str] = []

    @property
    def model(self) -> dict:
        return dict(self.catalog.entries[0])

    def compose(self) -> ComposeResult:
        yield Static(
            "R9V uses one immutable multi-file Qwen3.8 package. Download requires explicit acceptance of the Qwen Community License 1.0; verification and PLE derivation are separate deliberate steps.",
            classes="panel-copy",
        )
        with Vertical(classes="model-zone"):
            yield Label("Qwen3.8 Flash Next R9V package", classes="zone-title")
            with Horizontal(classes="inline-row"):
                yield Label("Storage root", id="r9v-models-dir-label", classes="inline-label")
                yield Input(value=str(get_models_dir()), id="r9v-models-dir")
                yield Button("Save Path", id="r9v-save-models-dir")
                yield Button("Refresh", id="r9v-refresh-model", variant="primary")
            with Horizontal(classes="inline-row"):
                yield Label("Container engine", id="r9v-model-engine-label", classes="inline-label")
                yield SearchableSelect("Select Podman or Docker", id="r9v-model-engine")
            with Horizontal(classes="inline-row"):
                yield Label("Runtime image", id="r9v-model-image-label", classes="inline-label")
                yield SearchableSelect("Select the R9V image", id="r9v-model-image")
            yield DataTable(id="r9v-model-status", cursor_type="row", zebra_stripes=True)
            with Horizontal(classes="action-row"):
                yield Button("Accept License & Download", id="r9v-download", variant="success")
                yield Button("Verify SHA256", id="r9v-verify")
                yield Button("Prepare PLE", id="r9v-prepare-ple")

    def on_mount(self) -> None:
        self.platform_id = self.app.active_platform_id
        engines = [(engine.value, engine.value) for engine in detect_container_engines()]
        select = self.query_one("#r9v-model-engine", SearchableSelect)
        select.set_options(engines)
        if engines:
            select.value = engines[0][1]
        table = self.query_one("#r9v-model-status", DataTable)
        table.add_columns("Profile", "Package", "PLE", "Location")
        self.refresh_platform()
        self.refresh_inventory()

    def set_platform(self, platform_id: str) -> None:
        self.platform_id = platform_id
        if self.is_mounted:
            self.refresh_platform()

    def refresh_platform(self) -> None:
        toolboxes = [
            toolbox
            for toolbox in self.app.toolbox_catalog.platform_toolboxes(self.platform_id)
            if toolbox.backend == "r9v"
        ]
        select = self.query_one("#r9v-model-image", SearchableSelect)
        select.set_options([(toolbox.name, toolbox.id) for toolbox in toolboxes])
        select.value = toolboxes[0].id if toolboxes else ""

    def refresh_inventory(self) -> None:
        model = self.model
        table = self.query_one("#r9v-model-status", DataTable)
        table.clear()
        table.add_row(
            model["name"],
            "Complete" if package_is_complete(model) else "Missing/incomplete",
            "Ready" if ple_is_complete(model) else "Not prepared",
            str(package_dir(model)),
            key=model["id"],
        )

    def _refresh_all(self) -> None:
        self.refresh_inventory()
        self.app.refresh_server_model_inventory("r9v")

    @on(Button.Pressed, "#r9v-save-models-dir")
    def save_path_pressed(self) -> None:
        value = self.query_one("#r9v-models-dir", Input).value.strip()
        if value and save_models_dir(value):
            self._refresh_all()
            self.notify("R9V storage path saved.")
        else:
            self.notify("Could not create or save that directory.", severity="error")

    @on(Button.Pressed, "#r9v-refresh-model")
    def refresh_pressed(self) -> None:
        self._refresh_all()

    @on(Button.Pressed, "#r9v-download")
    def download_pressed(self) -> None:
        model = self.model
        command = get_download_cmd(model)
        space = disk_space_for_path(get_models_dir())
        required = int(float(model["required_storage_gb"]) * 1024**3)
        capacity_note = download_space_note(required, space.free if space else None)
        self._pending_command = command
        self.app.push_screen(
            ConfirmModal(
                f"By continuing you confirm that you have reviewed and accept {model['license']}. Download the pinned {model['size_gb']} GiB package?\n\n{model['license_url']}\n\n{capacity_note}\n\n{shlex.join(command)}",
                yes_text="Accept & Download",
            ),
            self._download_confirmed,
        )

    def _download_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        error = None
        with self.app.suspend():
            try:
                subprocess.run(
                    self._pending_command,
                    env=huggingface_environment(get_hf_token()),
                    check=True,
                )
            except (OSError, subprocess.SubprocessError) as caught:
                error = caught
                pause_after_failure(f"R9V package download failed: {caught}")
        self._refresh_all()
        self.notify(
            "R9V package download complete." if error is None else f"R9V package download failed: {error}",
            severity="information" if error is None else "error",
            timeout=8,
        )

    @on(Button.Pressed, "#r9v-verify")
    def verify_pressed(self) -> None:
        model = self.model
        failures: list[str] = []
        with self.app.suspend():
            try:
                failures = verify_package(model)
                if not failures and ple_path(model).exists():
                    ple_failure = verify_ple(model)
                    if ple_failure:
                        failures.append(ple_failure)
            except OSError as error:
                failures = [str(error)]
            if failures:
                pause_after_failure("R9V verification failed:\n" + "\n".join(failures))
        self._refresh_all()
        self.notify(
            "All present R9V package and PLE hashes match." if not failures else failures[0],
            severity="information" if not failures else "error",
            timeout=8,
        )

    @on(Button.Pressed, "#r9v-prepare-ple")
    def prepare_ple_pressed(self) -> None:
        model = self.model
        invalid = invalid_artifacts(model)
        engine = str(self.query_one("#r9v-model-engine", SearchableSelect).value)
        toolbox_id = str(self.query_one("#r9v-model-image", SearchableSelect).value)
        if invalid:
            self.notify(f"Package is incomplete: {invalid[0]}", severity="error")
            return
        if not engine or toolbox_id not in self.app.toolbox_catalog.toolboxes:
            self.notify("Select a container engine and R9V image.", severity="error")
            return
        image = self.app.toolbox_catalog.toolboxes[toolbox_id].image
        self._pending_command = build_prepare_ple_cmd(engine, image, model)
        self.app.push_screen(
            ConfirmModal(
                f"Derive the 26.82 GiB PLE payload from the verified GGUF shards?\n\n{shlex.join(self._pending_command)}",
                yes_text="Prepare PLE",
            ),
            self._prepare_ple_confirmed,
        )

    def _prepare_ple_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        error = None
        with self.app.suspend():
            try:
                ple_path(self.model).parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(self._pending_command, check=True)
            except (OSError, subprocess.SubprocessError) as caught:
                error = caught
                pause_after_failure(f"PLE preparation failed: {caught}")
        self._refresh_all()
        self.notify(
            "R9V PLE payload prepared." if error is None else f"PLE preparation failed: {error}",
            severity="information" if error is None else "error",
            timeout=8,
        )
