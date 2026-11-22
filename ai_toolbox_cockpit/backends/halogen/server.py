"""Halogen's Strix Halo server form."""

import shlex
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Label, Static

from ai_toolbox_cockpit.backends.base import BackendServerPanel
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.server_process import run_foreground_server
from ai_toolbox_cockpit.settings import get_backend_settings, load_default_toolbox, save_backend_settings
from ai_toolbox_cockpit.widgets import ConfirmModal, SearchableSelect

from .model_manager import (
    fast_draft_download, get_bundle, get_models_dir, get_server_defaults, incomplete_files, load_bundles, save_models_dir,
)
from .runner import CONTAINER_NAME, build_server_cmd


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
                "Strix Halo (gfx1151) only. "
                "Download a Qwen3.8-Flash-Next bundle in Models, then start the server here. "
                "Choose a + vision bundle to accept images. "
                "This image runs directly in Podman/Docker and cannot be entered as a toolbox.",
                classes="panel-copy",
            )
            yield Static(
                "Use at your own risk: Halogen is a third-party closed-source project and container, "
                "so it is harder to audit than the open-source backends. Cockpit disables the "
                "container's network and mounts only the selected model/tokenizer files read-only, "
                "plus GPU devices. A host relay provides API access. This limits risk; it does not "
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
                (("context", "Request context", ""), ("pool", "KV pool positions", ""),
                 ("slots", "Concurrent requests", "")),
            ):
                with Horizontal(classes="compact-fields"):
                    for control, label, default in fields:
                        with Vertical(classes="compact-field"):
                            yield Label(label, id=f"halogen-{control}-label", classes="field-label")
                            yield Input(value=default, id=f"halogen-{control}")
            with Horizontal(classes="inline-row"):
                yield Label("Prompt cache", id="halogen-prompt-cache-label", classes="inline-label")
                yield SearchableSelect("Select prompt cache mode", id="halogen-prompt-cache")
            with Vertical(classes="model-zone", id="halogen-speculation-settings"):
                yield Label("Speculative decoding", classes="zone-title")
                for control, label in (("speculation", "Method"), ("draft-head", "MTP draft head")):
                    with Horizontal(classes="inline-row"):
                        yield Label(label, id=f"halogen-{control}-label", classes="inline-label")
                        yield SearchableSelect(f"Select {label.lower()}", id=f"halogen-{control}")
                with Horizontal(classes="compact-fields"):
                    for key, label in (("greedy", "Greedy draft tokens"), ("sampled", "Sampled draft tokens")):
                        with Vertical(classes="compact-field"):
                            yield Label(label, id=f"halogen-{key}-draft-tokens-label", classes="field-label")
                            yield Input(placeholder="Auto", id=f"halogen-{key}-draft-tokens")
                yield Static("", id="halogen-draft-head-status", classes="panel-copy")
            yield Static(
                "KV pool positions control memory use. Cold loading can take minutes. "
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
        speculation = self.query_one("#halogen-speculation", SearchableSelect)
        speculation.set_options([("MTP", "mtp"), ("Disabled", "disabled")])
        speculation.value = "disabled" if settings.get("speculation") == "disabled" else "mtp"
        head = self.query_one("#halogen-draft-head", SearchableSelect)
        head.set_options([("Built-in (default)", "builtin"), ("Faster", "fast")])
        head.value = "fast" if settings.get("draft_head") == "fast" else "builtin"
        for key in ("greedy", "sampled"):
            self.query_one(f"#halogen-{key}-draft-tokens", Input).value = str(
                settings.get(f"{key}_draft_tokens", "")
            )
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

    @on(SearchableSelect.Changed, "#halogen-model")
    def update_allocation_placeholders(self, event: SearchableSelect.Changed) -> None:
        if event.value:
            for field, default in get_server_defaults(get_bundle(event.value)).items():
                self.query_one(f"#halogen-{field}", Input).placeholder = str(default)

    @on(SearchableSelect.Changed, "#halogen-model")
    @on(SearchableSelect.Changed, "#halogen-speculation")
    @on(SearchableSelect.Changed, "#halogen-draft-head")
    @on(Input.Changed, "#halogen-server-dir")
    def update_speculation_controls(self) -> None:
        bundle_id = self.query_one("#halogen-model", SearchableSelect).value
        if not bundle_id:
            return
        bundle = get_bundle(bundle_id)
        supported = bool(bundle.get("fast_draft_head"))
        self.query_one("#halogen-speculation-settings").display = supported
        active = self.query_one("#halogen-speculation", SearchableSelect).value == "mtp"
        head = self.query_one("#halogen-draft-head", SearchableSelect)
        head.disabled = not active
        for key in ("greedy", "sampled"):
            self.query_one(f"#halogen-{key}-draft-tokens", Input).disabled = not active
        status = self.query_one("#halogen-draft-head-status", Static)
        status.display = supported and active and head.value == "fast"
        if status.display:
            directory = self.query_one("#halogen-server-dir", Input).value.strip()
            try:
                ready = directory and not incomplete_files(fast_draft_download(bundle), Path(directory))
            except (OSError, RuntimeError, ValueError):
                ready = False
            status.update(
                "Faster head ready. Requires Halogen 0.20.7 or newer." if ready
                else "Get the faster MTP head in Models (1.39 GiB, Halogen 0.20.7 or newer)."
            )

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
            directory = self.query_one("#halogen-server-dir", Input).value.strip()
            if not directory:
                raise ValueError("Enter a model directory.")
            engine = self.query_one("#halogen-engine", SearchableSelect).value
            bundle_id = self.query_one("#halogen-model", SearchableSelect).value
            cache = self.query_one("#halogen-prompt-cache", SearchableSelect).value
            allocations = {key: int(values[key]) if values[key] else None for key in ("context", "pool", "slots")}
            speculation_options = {}
            draft_limits = {}
            if get_bundle(bundle_id).get("fast_draft_head"):
                speculation_options = {
                    "speculation": self.query_one("#halogen-speculation", SearchableSelect).value,
                    "draft_head": self.query_one("#halogen-draft-head", SearchableSelect).value,
                    **{
                        f"{key}_draft_tokens": self.query_one(f"#halogen-{key}-draft-tokens", Input).value.strip()
                        for key in ("greedy", "sampled")
                    },
                }
                if speculation_options["speculation"] == "mtp":
                    for key in ("greedy_draft_tokens", "sampled_draft_tokens"):
                        value = speculation_options[key]
                        if value:
                            try:
                                draft_limits[key] = int(value)
                            except ValueError:
                                label = key.replace("_", " ").capitalize()
                                raise ValueError(f"{label} must be an integer from 1 to 7, or blank for Auto.") from None
            self._pending_command = build_server_cmd(
                engine=engine, image=toolbox.image, platform_id=self.platform_id,
                engine_args=list(self.app.toolbox_catalog.runtime_profiles[toolbox.runtime_profile].engine_args),
                models_dir=Path(directory), bundle_id=bundle_id, host=values["host"],
                port=int(values["port"]), context_size=allocations["context"],
                kv_pool_positions=allocations["pool"], kv_slots=allocations["slots"], prompt_cache=cache,
                speculation=speculation_options.get("speculation", "mtp"),
                draft_head=speculation_options.get("draft_head", "builtin"),
                **draft_limits,
            )
            self._pending_settings = {**values, **speculation_options, "models_dir": str(Path(directory).expanduser().resolve()),
                                      "engine": engine, "bundle_id": bundle_id, "prompt_cache": cache}
        except (ValueError, OSError) as error:
            self.notify(str(error), severity="error", timeout=10)
            return
        self.app.push_screen(ConfirmModal(
            "Start third-party closed-source Halogen Flash? Use at your own risk.\n"
            "No direct container network; selected bundle files mounted read-only.\n"
            f"Host API relay: {values['host']}:{values['port']} -> container loopback.\n\n"
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
