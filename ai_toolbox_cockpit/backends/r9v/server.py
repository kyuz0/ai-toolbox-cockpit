"""R9V dual-R9700 Qwen server panel."""

import shlex
import subprocess
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Label, Static

from ai_toolbox_cockpit.backends.base import BackendServerPanel
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.server_process import run_foreground_server
from ai_toolbox_cockpit.runtime.terminal import pause_after_failure
from ai_toolbox_cockpit.settings import (
    get_backend_settings,
    load_default_toolbox,
    save_backend_settings,
)
from ai_toolbox_cockpit.widgets import ConfirmModal, SearchableSelect

from .model_manager import invalid_artifacts, package_dir, ple_is_complete, ple_path
from .preflight import inspect_host
from .runner import CONTAINER_NAME, build_runtime_probe_cmd, build_server_cmd


class R9vServerPanel(BackendServerPanel):
    backend_label = "R9V Server"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.platform_id = ""
        self._pending_command: list[str] = []
        self._pending_probe: list[str] = []
        self._pending_devices = "0,1"

    @property
    def model(self) -> dict:
        return dict(self.app.model_catalog.backends["r9v"].entries[0])

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label(self.backend_label, classes="panel-title")
            yield Static(
                "Launch the pinned Qwen3.8 Flash Next R9V profile on exactly two R9700 GPUs. Device order is semantic: rank 1 owns the larger expert placement and LRU cache.",
                classes="panel-copy",
            )
            with Horizontal(classes="inline-row"):
                yield Label("Engine", id="r9v-engine-label", classes="inline-label")
                yield SearchableSelect("Select Podman or Docker", id="r9v-engine")
            with Horizontal(classes="inline-row"):
                yield Label("Image", id="r9v-image-label", classes="inline-label")
                yield SearchableSelect("Select R9V image", id="r9v-image")
            with Horizontal(classes="inline-row"):
                yield Label("Model package", id="r9v-package-label", classes="inline-label")
                yield Input(id="r9v-package", disabled=True)
            with Horizontal(classes="inline-row"):
                yield Label("PLE payload", id="r9v-ple-label", classes="inline-label")
                yield Input(id="r9v-ple", disabled=True)
            with Horizontal(classes="inline-row"):
                yield Label("Cache directory", id="r9v-cache-label", classes="inline-label")
                yield Input(id="r9v-cache")
            with Horizontal(classes="compact-fields"):
                with Vertical(classes="compact-field"):
                    yield Label("HIP device order", id="r9v-devices-label", classes="field-label")
                    yield Input(value="0,1", id="r9v-devices")
                with Vertical(classes="compact-field"):
                    yield Label("Host", id="r9v-host-label", classes="field-label")
                    yield Input(value="localhost", id="r9v-host")
                with Vertical(classes="compact-field"):
                    yield Label("Port", id="r9v-port-label", classes="field-label")
                    yield Input(value="8004", id="r9v-port")
            yield Static(
                "Fixed profile: TP2, MTP2 block-FP8, 128K context, one sequence, SSD PLE, reuse3v2 tiered experts, grouped-16 prefill, Ring/Simple RCCL, R4D disabled. Start runs host and ROCm/GPU preflight checks first.",
                classes="panel-copy",
            )
            with Horizontal(classes="action-row"):
                yield Button("Start R9V Server", id="r9v-start", variant="primary")

    def on_mount(self) -> None:
        self.platform_id = self.app.active_platform_id
        engines = [(engine.value, engine.value) for engine in detect_container_engines()]
        select = self.query_one("#r9v-engine", SearchableSelect)
        select.set_options(engines)
        if engines:
            select.value = engines[0][1]
        settings = get_backend_settings("r9v")
        self.query_one("#r9v-cache", Input).value = str(
            settings.get("cache_dir", "~/.cache/r9v")
        )
        self.refresh_platform()
        self.refresh_model_inventory()

    def set_platform(self, platform_id: str) -> None:
        self.platform_id = platform_id
        if self.is_mounted:
            self.refresh_platform()

    def refresh_platform(self) -> None:
        toolboxes = [
            toolbox
            for toolbox in self.app.toolbox_catalog.platform_toolboxes(self.platform_id)
            if toolbox.backend == "r9v" and toolbox.feature_state("server") != "unavailable"
        ]
        select = self.query_one("#r9v-image", SearchableSelect)
        select.set_options([(toolbox.name, toolbox.id) for toolbox in toolboxes])
        fallback = self.app.toolbox_catalog.platform(self.platform_id).defaults.get("r9v", "")
        default = load_default_toolbox("r9v", self.platform_id, fallback)
        ids = {toolbox.id for toolbox in toolboxes}
        select.value = default if default in ids else (toolboxes[0].id if toolboxes else "")

    def refresh_model_inventory(self) -> None:
        if not self.is_mounted:
            return
        self.query_one("#r9v-package", Input).value = str(package_dir(self.model))
        self.query_one("#r9v-ple", Input).value = str(ple_path(self.model))

    @on(Button.Pressed, "#r9v-start")
    def start_pressed(self) -> None:
        model = self.model
        invalid = invalid_artifacts(model)
        if invalid:
            self.notify(f"R9V package is incomplete: {invalid[0]}", severity="error")
            return
        if not ple_is_complete(model):
            self.notify("Prepare the verified R9V PLE payload in Models first.", severity="error")
            return
        engine = str(self.query_one("#r9v-engine", SearchableSelect).value)
        toolbox_id = str(self.query_one("#r9v-image", SearchableSelect).value)
        if not engine or toolbox_id not in self.app.toolbox_catalog.toolboxes:
            self.notify("Select a container engine and R9V image.", severity="error")
            return
        toolbox = self.app.toolbox_catalog.toolboxes[toolbox_id]
        profile = self.app.toolbox_catalog.runtime_profiles[toolbox.runtime_profile]
        try:
            port = int(self.query_one("#r9v-port", Input).value)
            cache_value = self.query_one("#r9v-cache", Input).value.strip()
            cache_dir = Path(cache_value).expanduser().resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            save_backend_settings("r9v", {"cache_dir": cache_value})
            self._pending_devices = self.query_one("#r9v-devices", Input).value
            self._pending_probe = build_runtime_probe_cmd(
                engine=engine,
                image=toolbox.image,
                engine_args=list(profile.engine_args),
                visible_devices=self._pending_devices,
            )
            self._pending_command = build_server_cmd(
                engine=engine,
                image=toolbox.image,
                engine_args=list(profile.engine_args),
                model_dir=package_dir(model),
                ple_path=ple_path(model),
                cache_dir=cache_dir,
                visible_devices=self._pending_devices,
                host=self.query_one("#r9v-host", Input).value,
                port=port,
            )
        except (OSError, ValueError) as error:
            self.notify(f"Invalid R9V setting: {error}", severity="error")
            return
        self.app.push_screen(
            ConfirmModal(
                f"Start the fixed dual-R9700 R9V profile?\n\n{shlex.join(self._pending_command)}",
                yes_text="Start",
            ),
            self._start_confirmed,
        )

    def _start_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        command = self._pending_command
        with self.app.suspend():
            report = inspect_host(self._pending_devices, ple_path(self.model))
            print(f"\nR9V host preflight:\n{report.render()}\n")
            if not report.ok:
                pause_after_failure("R9V host preflight failed; the server was not started.")
                return
            print(f"R9V runtime probe:\n{shlex.join(self._pending_probe)}\n")
            try:
                subprocess.run(self._pending_probe, check=True)
            except (OSError, subprocess.SubprocessError) as error:
                pause_after_failure(f"R9V ROCm/GPU runtime probe failed: {error}")
                return
            run_foreground_server(command, command[0], CONTAINER_NAME)
