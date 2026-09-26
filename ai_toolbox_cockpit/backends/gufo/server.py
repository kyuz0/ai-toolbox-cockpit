"""Gufo Strix Halo server controls."""

import shlex

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Label, Static, TextArea

from ai_toolbox_cockpit.backends.base import BackendServerPanel
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.server_process import run_foreground_server
from ai_toolbox_cockpit.settings import (
    get_backend_settings,
    load_default_toolbox,
    save_backend_settings,
)
from ai_toolbox_cockpit.widgets import ConfirmModal, SearchableSelect

from .model_manager import get_model, load_models, model_status, resolved_files
from .server_runner import CONTAINER_NAME, build_server_cmd


class GufoServerPanel(BackendServerPanel):
    backend_label = "Gufo Server"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.platform_id = ""
        self._pending_command: list[str] = []
        self._pending_settings: dict = {}
        self._next_speculation_preference: str | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label(self.backend_label, classes="panel-title")
            yield Static(
                "Experimental Strix Halo backend for the source-pinned Gufo ROCm 10.0 image. "
                "The curated profiles cover Qwen3.8 Flash Next Q4 with optional MTP-7, "
                "Qwen3.8 27B Q4, and DeepSeek V4 Flash 0731 with optional DSpark.",
                classes="panel-copy",
            )
            for control, label in (
                ("engine", "Engine"),
                ("image", "Image"),
                ("model", "Model / quant"),
                ("speculation", "Speculative decoding"),
                ("think", "Thinking"),
            ):
                with Horizontal(classes="inline-row"):
                    yield Label(label, id=f"gufo-{control}-label", classes="inline-label")
                    yield SearchableSelect(f"Select {label.lower()}", id=f"gufo-{control}")
            for fields in (
                (("host", "Host", "127.0.0.1"), ("port", "Port", "18080")),
                (("context", "Context", "262144"), ("sessions", "Concurrent sessions", "1")),
                (("max-tokens", "Maximum output", "32768"), ("max-pending-per-client", "Queued requests / client", "4")),
                (("draft-tokens", "MTP draft cap", "7"),),
            ):
                with Horizontal(classes="compact-fields"):
                    for control, label, default in fields:
                        with Vertical(classes="compact-field"):
                            yield Label(label, id=f"gufo-{control}-label", classes="field-label")
                            yield Input(value=default, id=f"gufo-{control}")
            with Horizontal(classes="extra-args-row"):
                yield Label("Extra Gufo args", id="gufo-extra-args-label", classes="inline-label")
                yield TextArea(id="gufo-extra-args", soft_wrap=True)
            yield Static(
                "The server binds inside the container and publishes only the selected host/port. "
                "Target directories and selected sidecars are mounted read-only. Ctrl+C stops it.",
                classes="panel-copy",
            )
            yield Button("Start Gufo", id="gufo-start", variant="primary")

    def on_mount(self) -> None:
        settings = get_backend_settings("gufo")
        engines = [(engine.value, engine.value) for engine in detect_container_engines()]
        select = self.query_one("#gufo-engine", SearchableSelect)
        select.set_options(engines)
        select.value = (
            settings.get("engine", "")
            if settings.get("engine") in dict(engines)
            else (engines[0][1] if engines else "")
        )
        think = self.query_one("#gufo-think", SearchableSelect)
        think.set_options([
            ("Model default", "auto"),
            ("On", "on"),
            ("Off", "off"),
        ])
        think_mode = str(settings.get("think_mode", "auto"))
        think.value = think_mode if think_mode in {"auto", "on", "off"} else "auto"
        for control in (
            "host", "port", "context", "sessions", "max_tokens",
            "max_pending_per_client", "draft_tokens",
        ):
            if control in settings:
                widget_id = control.replace("_", "-")
                self.query_one(f"#gufo-{widget_id}", Input).value = str(settings[control])
        self.query_one("#gufo-extra-args", TextArea).text = str(settings.get("extra_args", ""))
        self.set_platform(self.app.active_platform_id)
        self.refresh_model_inventory()

    def set_platform(self, platform_id: str) -> None:
        self.platform_id = platform_id
        if not self.is_mounted:
            return
        toolboxes = [
            item for item in self.app.toolbox_catalog.platform_toolboxes(platform_id)
            if item.backend == "gufo" and item.feature_state("server") != "unavailable"
        ]
        select = self.query_one("#gufo-image", SearchableSelect)
        select.set_options([(f"{item.name} — {item.image}", item.id) for item in toolboxes])
        default = load_default_toolbox(
            "gufo",
            platform_id,
            self.app.toolbox_catalog.platform(platform_id).defaults.get("gufo", ""),
        )
        ids = {item.id for item in toolboxes}
        select.value = default if default in ids else (toolboxes[0].id if toolboxes else "")
        self.query_one("#gufo-start", Button).disabled = not toolboxes
        self.refresh_model_inventory()

    def refresh_model_inventory(self) -> None:
        if not self.is_mounted:
            return
        settings = get_backend_settings("gufo")
        models = load_models()
        select = self.query_one("#gufo-model", SearchableSelect)
        previous = select.value or settings.get("model_id", "")
        select.set_options([
            (f"{entry['name']} — {model_status(entry)}", entry["id"])
            for entry in models
        ])
        ids = {entry["id"] for entry in models}
        preference = settings.get("speculation_mode")
        self._next_speculation_preference = (
            str(preference) if preference in {"baseline", "mtp", "dspark"} else None
        )
        select.value = previous if previous in ids else next(
            entry["id"] for entry in models if entry.get("recommended")
        )

    def _refresh_speculation(self, preferred: str | None = None) -> None:
        model_id = self.query_one("#gufo-model", SearchableSelect).value
        try:
            model = get_model(model_id)
        except ValueError:
            model = {}
        speculation = model.get("speculation")
        options = [("Disabled (baseline)", "baseline")]
        default = "baseline"
        if speculation:
            mode = speculation["mode"]
            label = "MTP" if mode == "mtp" else "DSpark"
            sidecar_ready = resolved_files(model)["sidecar"] is not None
            options.append((
                f"{label} ({'sidecar ready' if sidecar_ready else 'sidecar missing'})",
                mode,
            ))
            if sidecar_ready:
                default = mode
            self.query_one("#gufo-context", Input).value = str(model["context_size"])
            if speculation.get("draft_tokens"):
                self.query_one("#gufo-draft-tokens", Input).value = str(speculation["draft_tokens"])
        select = self.query_one("#gufo-speculation", SearchableSelect)
        select.set_options(options)
        values = {value for _, value in options}
        select.value = preferred if preferred in values else default

    @on(SearchableSelect.Changed, "#gufo-model")
    def model_changed(self, event: SearchableSelect.Changed) -> None:
        preference = self._next_speculation_preference
        self._next_speculation_preference = None
        self._refresh_speculation(preference)

    @on(Button.Pressed, "#gufo-start")
    def start_pressed(self) -> None:
        toolbox_id = self.query_one("#gufo-image", SearchableSelect).value
        toolbox = self.app.toolbox_catalog.toolboxes.get(toolbox_id)
        if (
            not toolbox or toolbox.backend != "gufo"
            or toolbox_id not in self.app.toolbox_catalog.platform(self.platform_id).toolbox_ids
        ):
            self.notify("Select the Gufo image for Strix Halo.", severity="error")
            return
        try:
            values = {
                key: self.query_one(f"#gufo-{key.replace('_', '-')}", Input).value.strip()
                for key in (
                    "host", "port", "context", "sessions", "max_tokens",
                    "max_pending_per_client", "draft_tokens",
                )
            }
            engine = self.query_one("#gufo-engine", SearchableSelect).value
            model_id = self.query_one("#gufo-model", SearchableSelect).value
            speculation_mode = self.query_one("#gufo-speculation", SearchableSelect).value
            think_mode = self.query_one("#gufo-think", SearchableSelect).value
            extra_args = self.query_one("#gufo-extra-args", TextArea).text.strip()
            self._pending_command = build_server_cmd(
                engine=engine,
                image=toolbox.image,
                engine_args=list(
                    self.app.toolbox_catalog.runtime_profiles[toolbox.runtime_profile].engine_args
                ),
                platform_id=self.platform_id,
                model_id=model_id,
                speculation_mode=speculation_mode,
                host=values["host"],
                port=int(values["port"]),
                context_size=int(values["context"]),
                sessions=int(values["sessions"]),
                max_tokens=int(values["max_tokens"]),
                think_mode=think_mode,
                max_pending_per_client=int(values["max_pending_per_client"]),
                draft_tokens=int(values["draft_tokens"]),
                extra_args=extra_args,
            )
            self._pending_settings = {
                **values,
                "engine": engine,
                "model_id": model_id,
                "speculation_mode": speculation_mode,
                "think_mode": think_mode,
                "extra_args": extra_args,
            }
        except (ValueError, OSError) as error:
            self.notify(str(error), severity="error", timeout=10)
            return
        self.app.push_screen(
            ConfirmModal(
                "Start the experimental Gufo server?\n\n"
                f"{shlex.join(self._pending_command)}",
                yes_text="Start",
            ),
            self._start_confirmed,
        )

    def _start_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        if not save_backend_settings("gufo", self._pending_settings):
            self.notify("Could not save settings; using them for this session.", severity="warning")
        command = self._pending_command
        with self.app.suspend():
            run_foreground_server(command, command[0], CONTAINER_NAME)
