"""ComfyUI direct-container server UI."""

import shlex
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Label, Static, Switch

from ai_toolbox_cockpit.backends.base import BackendServerPanel
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.server_process import run_foreground_server
from ai_toolbox_cockpit.settings import get_backend_settings, load_default_toolbox, save_backend_settings
from ai_toolbox_cockpit.widgets import ConfirmModal, SearchableSelect

from .runner import ComfyPaths, build_server_cmd, default_paths


class ComfyUiServerPanel(BackendServerPanel):
    backend_label = "ComfyUI Server"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.platform_id = ""
        self._pending_command: list[str] = []

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label(self.backend_label, classes="panel-title")
            yield Static(
                "Launch ComfyUI directly with persistent model, input, output, and user/workflow directories. Defaults match the toolbox's start_comfy_ui command.",
                classes="panel-copy",
            )
            with Horizontal(classes="inline-row"):
                yield Label("Engine", id="comfy-engine-label", classes="inline-label")
                yield SearchableSelect("Select Podman or Docker", id="comfy-engine")
            with Horizontal(classes="inline-row"):
                yield Label("Image", id="comfy-image-label", classes="inline-label")
                yield SearchableSelect("Search ComfyUI images", id="comfy-image")
            with Horizontal(classes="compact-fields"):
                with Vertical(classes="compact-field"):
                    yield Label("Host", id="comfy-host-label", classes="field-label")
                    yield Input(value="localhost", placeholder="Host", id="comfy-host")
                with Vertical(classes="compact-field"):
                    yield Label("Port", id="comfy-port-label", classes="field-label")
                    yield Input(value="8000", placeholder="Port", id="comfy-port")
            with Horizontal(classes="compact-fields"):
                with Vertical(classes="compact-field"):
                    yield Label("Models directory", id="comfy-models-path-label", classes="field-label")
                    yield Input(placeholder="Models directory", id="comfy-models-path")
                with Vertical(classes="compact-field"):
                    yield Label("Input directory", id="comfy-inputs-path-label", classes="field-label")
                    yield Input(placeholder="Input directory", id="comfy-inputs-path")
            with Horizontal(classes="compact-fields"):
                with Vertical(classes="compact-field"):
                    yield Label("Output directory", id="comfy-outputs-path-label", classes="field-label")
                    yield Input(placeholder="Output directory", id="comfy-outputs-path")
                with Vertical(classes="compact-field"):
                    yield Label("User/workflow directory", id="comfy-user-path-label", classes="field-label")
                    yield Input(placeholder="User/workflow directory", id="comfy-user-path")
            with Horizontal(classes="options-row"):
                yield Switch(value=True, id="comfy-disable-mmap")
                yield Label("Disable mmap", id="comfy-disable-mmap-label")
                yield Switch(value=True, id="comfy-gpu-only")
                yield Label("GPU only", id="comfy-gpu-only-label")
                yield Switch(value=True, id="comfy-disable-smart")
                yield Label("Disable smart memory", id="comfy-disable-smart-label")
            with Horizontal(classes="options-row"):
                yield Switch(value=True, id="comfy-cache-none")
                yield Label("No node cache", id="comfy-cache-none-label")
                yield Switch(value=True, id="comfy-bf16-vae")
                yield Label("BF16 VAE", id="comfy-bf16-vae-label")
            with Horizontal(classes="inline-row"):
                yield Label("Extra args", id="comfy-extra-args-label", classes="inline-label")
                yield Input(placeholder="Additional ComfyUI arguments", id="comfy-extra-args")
            with Horizontal(classes="action-row"):
                yield Button("Save Paths", id="comfy-save-paths")
                yield Button("Start ComfyUI", id="comfy-start", variant="primary")

    def on_mount(self) -> None:
        self.platform_id = self.app.active_platform_id
        engines = [(engine.value, engine.value) for engine in detect_container_engines()]
        select = self.query_one("#comfy-engine", SearchableSelect)
        select.set_options(engines)
        if engines:
            select.value = engines[0][1]
        defaults = default_paths()
        settings = get_backend_settings("comfyui")
        for field, key, fallback in (
            ("#comfy-models-path", "models_dir", defaults.models),
            ("#comfy-inputs-path", "inputs_dir", defaults.inputs),
            ("#comfy-outputs-path", "outputs_dir", defaults.outputs),
            ("#comfy-user-path", "user_dir", defaults.user),
        ):
            self.query_one(field, Input).value = str(settings.get(key, fallback))
        self.refresh_platform(self.platform_id)

    def set_platform(self, platform_id: str) -> None:
        self.platform_id = platform_id
        if self.is_mounted:
            self.refresh_platform(platform_id)

    def refresh_platform(self, platform_id: str) -> None:
        toolboxes = [
            toolbox
            for toolbox in self.app.toolbox_catalog.platform_toolboxes(platform_id)
            if toolbox.backend == "comfyui" and toolbox.feature_state("server") != "unavailable"
        ]
        select = self.query_one("#comfy-image", SearchableSelect)
        select.set_options([
            (
                f"{toolbox.name}{' [experimental]' if toolbox.feature_state('server') == 'experimental' else ''} — {toolbox.image}",
                toolbox.id,
            )
            for toolbox in toolboxes
        ])
        default = load_default_toolbox(
            "comfyui", platform_id,
            self.app.toolbox_catalog.platform(platform_id).defaults.get("comfyui", ""),
        )
        select.value = default if default in {toolbox.id for toolbox in toolboxes} else (toolboxes[0].id if toolboxes else "")

    def paths(self) -> ComfyPaths:
        return ComfyPaths(*(
            Path(self.query_one(field, Input).value).expanduser().resolve()
            for field in ("#comfy-models-path", "#comfy-inputs-path", "#comfy-outputs-path", "#comfy-user-path")
        ))

    @on(Button.Pressed, "#comfy-save-paths")
    def save_paths_pressed(self) -> None:
        try:
            paths = self.paths()
            for path in (paths.models, paths.inputs, paths.outputs, paths.user):
                path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.notify(f"Could not create ComfyUI path: {error}", severity="error")
            return
        saved = save_backend_settings("comfyui", {
            "models_dir": str(paths.models),
            "inputs_dir": str(paths.inputs),
            "outputs_dir": str(paths.outputs),
            "user_dir": str(paths.user),
        })
        self.notify("ComfyUI paths saved." if saved else "Could not save ComfyUI paths.", severity="information" if saved else "error")

    @on(Button.Pressed, "#comfy-start")
    def start_pressed(self) -> None:
        engine = self.query_one("#comfy-engine", SearchableSelect).value
        toolbox_id = self.query_one("#comfy-image", SearchableSelect).value
        if not engine or toolbox_id not in self.app.toolbox_catalog.toolboxes:
            self.notify("Select an engine and ComfyUI image.", severity="error")
            return
        try:
            port = int(self.query_one("#comfy-port", Input).value)
            paths = self.paths()
            for path in (paths.models, paths.inputs, paths.outputs, paths.user):
                path.mkdir(parents=True, exist_ok=True)
        except (ValueError, OSError) as error:
            self.notify(f"Invalid ComfyUI setting: {error}", severity="error")
            return
        toolbox = self.app.toolbox_catalog.toolboxes[toolbox_id]
        profile = self.app.toolbox_catalog.runtime_profiles[toolbox.runtime_profile]
        try:
            self._pending_command = build_server_cmd(
                engine=engine,
                image=toolbox.image,
                engine_args=list(profile.engine_args),
                paths=paths,
                host=self.query_one("#comfy-host", Input).value,
                port=port,
                disable_mmap=self.query_one("#comfy-disable-mmap", Switch).value,
                gpu_only=self.query_one("#comfy-gpu-only", Switch).value,
                disable_smart_memory=self.query_one("#comfy-disable-smart", Switch).value,
                cache_none=self.query_one("#comfy-cache-none", Switch).value,
                bf16_vae=self.query_one("#comfy-bf16-vae", Switch).value,
                extra_args=self.query_one("#comfy-extra-args", Input).value,
            )
        except ValueError as error:
            self.notify(str(error), severity="error")
            return
        self.app.push_screen(
            ConfirmModal(f"Start ComfyUI?\n\n{shlex.join(self._pending_command)}", yes_text="Start"),
            self._start_confirmed,
        )

    def _start_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        command = self._pending_command
        with self.app.suspend():
            run_foreground_server(
                command,
                command[0],
                "ai-toolbox-cockpit-comfyui-server",
            )
