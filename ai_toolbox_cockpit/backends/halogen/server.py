"""Halogen's Strix Halo server form."""

import shlex
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Checkbox, Input, Label, Static

from ai_toolbox_cockpit.backends.base import BackendServerPanel
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.server_process import run_foreground_server
from ai_toolbox_cockpit.settings import get_backend_settings, load_default_toolbox, save_backend_settings
from ai_toolbox_cockpit.widgets import CockpitCheckbox, ConfirmModal, SearchableSelect

from .model_manager import (
    get_models_dir, incomplete_files, incomplete_npu_files, load_bundles, load_npu_models,
    save_models_dir,
)
from .runner import CONTAINER_NAME, build_server_cmd


def _npu_checkbox_id(model_id: str) -> str:
    return "halogen-npu-" + model_id.replace(".", "-")


class HalogenServerPanel(BackendServerPanel):
    backend_label = "Halogen Flash Server"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.platform_id = ""
        self._pending_command: list[str] = []
        self._pending_settings: dict = {}

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label(self.backend_label, classes="panel-title")
            yield Static(
                "Strix Halo (gfx1151) only. Experimental integration; GPU validation pending. "
                "Download a Qwen3.8-Flash-Next bundle in Models, then start the server here. "
                "Choose a + vision bundle to accept images. "
                "This image runs directly in Podman/Docker and cannot be entered as a toolbox.",
                classes="panel-copy",
            )
            yield Static(
                "Use at your own risk: Halogen is a third-party closed-source project and container, "
                "so it is harder to audit than the open-source backends. Cockpit disables the "
                "container's network and mounts only the selected model/tokenizer files read-only, "
                "plus GPU devices (and the NPU device when NPU models are selected). A host relay "
                "provides API access. This limits risk; it does not "
                "remove it. The container still shares the host kernel and GPU driver.",
                classes="panel-copy",
            )
            for control, label in (("engine", "Engine"), ("image", "Image"), ("model", "Model / precision")):
                with Horizontal(classes="inline-row"):
                    yield Label(label, id=f"halogen-{control}-label", classes="inline-label")
                    yield SearchableSelect(f"Select {label.lower()}", id=f"halogen-{control}")
            with Horizontal(classes="inline-row"):
                yield Label("Models directory", id="halogen-server-dir-label", classes="inline-label")
                yield Input(value=str(get_models_dir()), id="halogen-server-dir")
                yield Button("Save Path", id="halogen-server-save-path")
                yield Button("Scan Local", id="halogen-server-scan")
            for fields in (
                (("host", "Host", "127.0.0.1"), ("port", "Port", "8731")),
                (("context", "Request context", "262144"), ("pool", "KV pool positions", "524288"),
                 ("slots", "Concurrent slots", "4"), ("max-tok", "Max output tokens", "32768")),
            ):
                with Horizontal(classes="compact-fields"):
                    for control, label, default in fields:
                        with Vertical(classes="compact-field"):
                            yield Label(label, id=f"halogen-{control}-label", classes="field-label")
                            yield Input(value=default, id=f"halogen-{control}")
            with Horizontal(classes="inline-row"):
                yield Label("Prompt cache", id="halogen-prompt-cache-label", classes="inline-label")
                yield SearchableSelect("Select prompt cache mode", id="halogen-prompt-cache")
            with Vertical(classes="model-zone"):
                yield Label("NPU models (Ryzen AI NPU)", classes="zone-title")
                yield Static(
                    "Optional small models served beside Flash on the Ryzen AI NPU, behind the same "
                    "port. Download them in Models first. The host needs the amdxdna driver, "
                    "/dev/accel/accel0, XRT with its NPU plugin, and the GPU fabric clock held; "
                    "launch is blocked until they are present. Models whose files are already local "
                    "are selected by default.",
                    classes="panel-copy",
                )
                npu_models = load_npu_models()
                for index in range(0, len(npu_models), 3):
                    with Horizontal(classes="options-row"):
                        for entry in npu_models[index:index + 3]:
                            yield CockpitCheckbox(entry["name"], value=False,
                                                  id=_npu_checkbox_id(entry["id"]))
            yield Static(
                "Native context: up to 262144 tokens. KV pool positions control memory use; "
                "slots control concurrency. Defaults follow release 0.5.4. Cold loading can take minutes. "
                "The API listens on the chosen host/port; Ctrl+C stops it and returns here.",
                classes="panel-copy",
            )
            yield Button("Start Halogen", id="halogen-start", variant="primary")

    def on_mount(self) -> None:
        settings = get_backend_settings("halogen")
        engines = [(engine.value, engine.value) for engine in detect_container_engines()]
        select = self.query_one("#halogen-engine", SearchableSelect)
        select.set_options(engines)
        select.value = settings.get("engine", "") if settings.get("engine") in dict(engines) else (engines[0][1] if engines else "")
        cache = self.query_one("#halogen-prompt-cache", SearchableSelect)
        cache.set_options([("Fast prefix reuse (default)", "2"), ("Exact repeat answers", "1"), ("Off", "0")])
        cache.value = str(settings.get("prompt_cache", "2"))
        for control in ("host", "port", "context", "pool", "slots"):
            if control in settings:
                self.query_one(f"#halogen-{control}", Input).value = str(settings[control])
        self.query_one("#halogen-max-tok", Input).value = str(settings.get("max_tok", "32768"))
        self.set_platform(self.app.active_platform_id)
        self.refresh_model_inventory()

    def set_platform(self, platform_id: str) -> None:
        self.platform_id = platform_id
        if not self.is_mounted:
            return
        toolboxes = [item for item in self.app.toolbox_catalog.platform_toolboxes(platform_id)
                    if item.backend == "halogen" and item.feature_state("server") != "unavailable"]
        select = self.query_one("#halogen-image", SearchableSelect)
        select.set_options([(f"{item.name} — {item.image}", item.id) for item in toolboxes])
        default = load_default_toolbox("halogen", platform_id,
                                      self.app.toolbox_catalog.platform(platform_id).defaults.get("halogen", ""))
        select.value = default if default in {item.id for item in toolboxes} else (toolboxes[0].id if toolboxes else "")
        self.query_one("#halogen-start", Button).disabled = not toolboxes

    def refresh_model_inventory(self) -> None:
        directory = get_models_dir()
        self.query_one("#halogen-server-dir", Input).value = str(directory)
        select = self.query_one("#halogen-model", SearchableSelect)
        previous = select.value or get_backend_settings("halogen").get("bundle_id", "")
        bundles = load_bundles()
        select.set_options([(f"{entry['name']} — " + ("download required" if incomplete_files(entry, directory) else "ready"), entry["id"])
                            for entry in bundles])
        select.value = previous if previous in {entry["id"] for entry in bundles} else next(entry["id"] for entry in bundles if entry.get("recommended"))
        saved = get_backend_settings("halogen").get("npu_models")
        for entry in load_npu_models():
            checkbox = self.query_one(f"#{_npu_checkbox_id(entry['id'])}", Checkbox)
            if isinstance(saved, list):
                checkbox.value = entry["id"] in saved
            else:
                checkbox.value = (bool(entry.get("default_enabled"))
                                  and not incomplete_npu_files(entry, directory))

    @on(Button.Pressed, "#halogen-server-save-path")
    def save_path_pressed(self) -> None:
        if save_models_dir(self.query_one("#halogen-server-dir", Input).value.strip()):
            self.refresh_model_inventory()
            self.app.query_one("#model-panel-halogen").refresh_inventory()
            self.notify("Halogen model directory saved.")
        else:
            self.notify("Could not create or save that directory.", severity="error")

    @on(Button.Pressed, "#halogen-server-scan")
    def scan_pressed(self) -> None:
        self.refresh_model_inventory()

    @on(Button.Pressed, "#halogen-start")
    def start_pressed(self) -> None:
        toolbox_id = self.query_one("#halogen-image", SearchableSelect).value
        toolbox = self.app.toolbox_catalog.toolboxes.get(toolbox_id)
        if not toolbox or toolbox.backend != "halogen" or toolbox_id not in self.app.toolbox_catalog.platform(self.platform_id).toolbox_ids:
            self.notify("Select a Halogen image for Strix Halo.", severity="error")
            return
        try:
            values = {key: self.query_one(f"#halogen-{key}", Input).value.strip()
                      for key in ("host", "port", "context", "pool", "slots")}
            max_tokens = self.query_one("#halogen-max-tok", Input).value.strip()
            if not max_tokens:
                raise ValueError("Enter the maximum output tokens.")
            directory = self.query_one("#halogen-server-dir", Input).value.strip()
            if not directory:
                raise ValueError("Enter a model directory.")
            engine = self.query_one("#halogen-engine", SearchableSelect).value
            bundle_id = self.query_one("#halogen-model", SearchableSelect).value
            cache = self.query_one("#halogen-prompt-cache", SearchableSelect).value
            npu_models = tuple(entry["id"] for entry in load_npu_models()
                               if self.query_one(f"#{_npu_checkbox_id(entry['id'])}", Checkbox).value)
            self._pending_command = build_server_cmd(
                engine=engine, image=toolbox.image, platform_id=self.platform_id,
                engine_args=list(self.app.toolbox_catalog.runtime_profiles[toolbox.runtime_profile].engine_args),
                models_dir=Path(directory), bundle_id=bundle_id, host=values["host"],
                port=int(values["port"]), context_size=int(values["context"]),
                kv_pool_positions=int(values["pool"]), kv_slots=int(values["slots"]), prompt_cache=cache,
                max_tokens=int(max_tokens),
                npu_models=npu_models,
            )
            self._pending_settings = {**values, "models_dir": str(Path(directory).expanduser().resolve()),
                                      "engine": engine, "bundle_id": bundle_id, "prompt_cache": cache,
                                      "max_tok": max_tokens,
                                      "npu_models": list(npu_models)}
        except (ValueError, OSError) as error:
            self.notify(str(error), severity="error", timeout=10)
            return
        npu_note = ("NPU models on the Ryzen AI NPU: " + ", ".join(self._pending_settings["npu_models"]) + ".\n"
                    if self._pending_settings["npu_models"] else "")
        self.app.push_screen(ConfirmModal(
            "Start third-party closed-source Halogen Flash? Use at your own risk.\n"
            "No direct container network; selected bundle files mounted read-only.\n"
            f"Host API relay: {values['host']}:{values['port']} -> container loopback.\n"
            f"{npu_note}\n"
            f"{shlex.join(self._pending_command)}", yes_text="Start",
            copy_text=shlex.join(self._pending_command),
        ), self._start_confirmed)

    def _start_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        if not save_backend_settings("halogen", self._pending_settings):
            self.notify("Could not save settings; using them for this session.", severity="warning")
        command = self._pending_command
        with self.app.suspend():
            run_foreground_server(
                command, command[0], CONTAINER_NAME,
                isolated_api=(str(self._pending_settings["host"]), int(self._pending_settings["port"])),
            )
