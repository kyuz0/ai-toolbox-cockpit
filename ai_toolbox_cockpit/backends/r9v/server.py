"""R9V's labeled form for the tested dual-R9700 configuration."""

import shlex
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Collapsible, Input, Label, Static, TextArea

from ai_toolbox_cockpit.backends.base import BackendServerPanel
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.server_process import redact_command, run_foreground_server
from ai_toolbox_cockpit.settings import get_backend_settings, load_default_toolbox, save_backend_settings
from ai_toolbox_cockpit.widgets import ConfirmModal, SearchableSelect
from .model_manager import get_paths, incomplete_files, load_packages, ple_ready, save_paths
from .runner import CONTAINER_NAME, DEFAULTS, build_server_cmd


class R9vServerPanel(BackendServerPanel):
    backend_label = "R9V — Qwen3.8 Flash Next"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.platform_id = ""
        self._pending_command: list[str] = []
        self._pending_settings: dict = {}

    def fields(self, controls) -> ComposeResult:
        with Horizontal(classes="compact-fields"):
            for control, label in controls:
                with Vertical(classes="compact-field"):
                    yield Label(label, id=f"r9v-{control}-label", classes="field-label")
                    yield Input(value=DEFAULTS[control], id=f"r9v-{control}")

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label(self.backend_label, classes="panel-title")
            yield Static(
                "Tested: Qwen3.8 Flash Next IQ4_XS + FP8 MTP + Q8 vision, on 2× R9700 32 GB, "
                "64 GB RAM and NVMe. TP2, MTP2 and SSD PLE are fixed. Download the exact package "
                "and Prepare PLE in Models first. Podman is the tested runtime; Docker is experimental.",
                classes="panel-copy", id="r9v-profile-note",
            )
            for control, label in (("engine", "Container engine"), ("image", "Toolbox image"), ("model", "Model package")):
                with Horizontal(classes="inline-row"):
                    yield Label(label, id=f"r9v-{control}-label", classes="inline-label")
                    yield SearchableSelect(f"Select {label.lower()}", id=f"r9v-{control}")
            for key, label in (("models_dir", "Package directory"), ("ple_dir", "PLE directory"), ("cache_dir", "Cache directory")):
                with Horizontal(classes="inline-row"):
                    yield Label(label, id=f"r9v-{key}-label", classes="inline-label")
                    yield Input(value=str(get_paths()[key]), id=f"r9v-{key}")
            with Horizontal(classes="inline-row"):
                yield Button("Save Paths", id="r9v-server-save-paths")
                yield Button("Refresh Inventory", id="r9v-server-scan")
            yield from self.fields((("devices", "GPU indices (two)"), ("context", "Context tokens"), ("batch", "Prefill batch tokens")))
            with Horizontal(classes="inline-row"):
                yield Button("Apply 128K settings", id="r9v-context-128k")
                yield Button("Apply 256K settings", id="r9v-context-256k")
            yield Static("Context includes input + output tokens. Use the buttons to set context, KV memory "
                         "and expert cache together. 256K text mode disables the dynamic expert cache.",
                         classes="panel-copy")
            yield from self.fields((("host", "Bind address"), ("port", "API port")))
            yield from self.fields((("served_model", "API model name"),))
            with Horizontal(classes="inline-row"):
                yield Label("API key", id="r9v-api-key-label", classes="inline-label")
                yield Input(password=True, placeholder="Optional; kept only for this session", id="r9v-api-key")
            with Collapsible(title="Advanced memory and server settings", collapsed=True):
                yield Static("Defaults are the validated 64 GB profile. More sequences, KV memory or changed "
                             "expert offload may exceed RAM/VRAM. Offload GB is logical weight accounting, "
                             "not physical RAM allocation.", classes="panel-copy")
                yield from self.fields((("sequences", "Concurrent sequences"), ("kv_bytes", "KV bytes per GPU")))
                yield from self.fields((("expert_cache_slots", "Dynamic expert cache slots"),))
                yield from self.fields((("offload", "Logical offload GB"), ("offload_devices", "Offload GB per GPU")))
                with Horizontal(classes="extra-args-row"):
                    yield Label("Extra vLLM args", id="r9v-extra-args-label", classes="inline-label")
                    yield TextArea(soft_wrap=True, show_line_numbers=False, id="r9v-extra-args")
                yield Static("Extra args cannot override the form or fixed TP2/MTP2/SSD profile. "
                             "Sampling temperature, max output tokens and thinking are request parameters.", classes="panel-copy")
            yield Static("First startup takes minutes. Ctrl+C stops this server and returns to Cockpit. "
                         "Vision was tested at 128K; strict JSON formatting is not guaranteed.", classes="panel-copy")
            yield Button("Start R9V", id="r9v-start", variant="primary")

    @on(Button.Pressed, "#r9v-context-128k")
    @on(Button.Pressed, "#r9v-context-256k")
    def apply_context_settings(self, event: Button.Pressed) -> None:
        large = event.button.id == "r9v-context-256k"
        values = {"context": "262144" if large else "131072",
                  "kv_bytes": "4160749568" if large else "2285670400",
                  "expert_cache_slots": "0" if large else "16",
                  "sequences": "1", "batch": "1024"}
        for key, value in values.items():
            self.query_one(f"#r9v-{key}", Input).value = value
        self.notify("256K: larger KV cache, dynamic expert cache disabled, one sequence."
                    if large else "128K defaults restored: one sequence, 16 dynamic expert cache slots.")

    def on_mount(self) -> None:
        settings = get_backend_settings("r9v")
        engines = [(x.value, x.value) for x in detect_container_engines()]
        select = self.query_one("#r9v-engine", SearchableSelect)
        select.set_options(engines)
        select.value = settings.get("engine", "") if settings.get("engine") in dict(engines) else (engines[0][1] if engines else "")
        for key in DEFAULTS:
            self.query_one(f"#r9v-{key}", Input).value = str(settings.get(key, DEFAULTS[key]))
        self.query_one("#r9v-extra-args", TextArea).text = str(settings.get("extra_args", ""))
        self.set_platform(self.app.active_platform_id)
        self.refresh_model_inventory()

    def set_platform(self, platform_id: str) -> None:
        self.platform_id = platform_id
        if not self.is_mounted:
            return
        items = [x for x in self.app.toolbox_catalog.platform_toolboxes(platform_id)
                 if x.backend == "r9v" and x.feature_state("server") != "unavailable"]
        select = self.query_one("#r9v-image", SearchableSelect)
        select.set_options([(f"{x.name} — {x.image}", x.id) for x in items])
        default = load_default_toolbox("r9v", platform_id, self.app.toolbox_catalog.platform(platform_id).defaults.get("r9v", ""))
        select.value = default if default in {x.id for x in items} else (items[0].id if items else "")
        self.query_one("#r9v-start", Button).disabled = not items

    def refresh_model_inventory(self) -> None:
        paths = get_paths()
        for key, path in paths.items():
            self.query_one(f"#r9v-{key}", Input).value = str(path)
        packages = load_packages()
        select = self.query_one("#r9v-model", SearchableSelect)
        previous = select.value or get_backend_settings("r9v").get("package_id", "")
        choices = []
        for entry in packages:
            status = "package incomplete" if incomplete_files(entry, paths["models_dir"]) else (
                "PLE required" if not ple_ready(entry, paths["ple_dir"]) else "ready (sizes checked)")
            choices.append((f"{entry['name']} — {status}", entry["id"]))
        select.set_options(choices)
        select.value = previous if previous in {entry["id"] for entry in packages} else packages[0]["id"]

    @on(Button.Pressed, "#r9v-server-save-paths")
    def save_paths_pressed(self) -> None:
        values = {key: self.query_one(f"#r9v-{key}", Input).value.strip() for key in get_paths()}
        if save_paths(values):
            self.refresh_model_inventory()
            self.app.query_one("#model-panel-r9v").refresh_inventory()
            self.notify("R9V paths saved.")
        else:
            self.notify("Could not create or save those paths.", severity="error")

    @on(Button.Pressed, "#r9v-server-scan")
    def scan_pressed(self) -> None:
        self.refresh_model_inventory()

    @on(Button.Pressed, "#r9v-start")
    def start_pressed(self) -> None:
        item = self.app.toolbox_catalog.toolboxes.get(self.query_one("#r9v-image", SearchableSelect).value)
        if not item or item.backend != "r9v" or item.id not in self.app.toolbox_catalog.platform(self.platform_id).toolbox_ids:
            self.notify("Select an R9V image for R9700.", severity="error")
            return
        try:
            values = {key: self.query_one(f"#r9v-{key}", Input).value.strip() for key in DEFAULTS}
            paths = {key: self.query_one(f"#r9v-{key}", Input).value.strip() for key in get_paths()}
            if not all(paths.values()):
                raise ValueError("Enter package, PLE and cache directories.")
            engine = self.query_one("#r9v-engine", SearchableSelect).value
            package_id = self.query_one("#r9v-model", SearchableSelect).value
            extra = self.query_one("#r9v-extra-args", TextArea).text
            self._pending_command = build_server_cmd(
                engine=engine, image=item.image, platform_id=self.platform_id,
                engine_args=list(self.app.toolbox_catalog.runtime_profiles[item.runtime_profile].engine_args),
                **{key: Path(value) for key, value in paths.items()}, package_id=package_id, values=values,
                api_key=self.query_one("#r9v-api-key", Input).value, extra_args=extra)
            self._pending_settings = {**values, **paths, "engine": engine, "package_id": package_id, "extra_args": extra}
        except (ValueError, OSError) as error:
            self.notify(str(error), severity="error", timeout=10)
            return
        self.app.push_screen(ConfirmModal(
            "Start R9V on two R9700 GPUs? Ensure both GPUs are free.\n\n"
            + shlex.join(redact_command(self._pending_command)), yes_text="Start"), self._start_confirmed)

    def _start_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        paths = {key: self._pending_settings[key] for key in get_paths()}
        if not save_paths(paths):
            self.notify("Could not create or save R9V directories.", severity="error")
            return
        if not save_backend_settings("r9v", self._pending_settings):
            self.notify("Could not save settings; using this session's values.", severity="warning")
        command = self._pending_command
        with self.app.suspend():
            run_foreground_server(command, command[0], CONTAINER_NAME, display_command=redact_command(command))
