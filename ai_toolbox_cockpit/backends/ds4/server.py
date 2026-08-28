"""DS4 server UI, including distributed and SSD-streaming controls."""

import shlex
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Checkbox, Input, Label, Static, Switch

from ai_toolbox_cockpit.backends.base import BackendServerPanel
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.server_process import run_foreground_server
from ai_toolbox_cockpit.settings import load_default_toolbox
from ai_toolbox_cockpit.widgets import ConfirmModal, SearchableSelect

from .config import get_model_server_defaults
from .model_manager import scan_local_models
from .server_runner import build_server_cmd


class Ds4ServerPanel(BackendServerPanel):
    backend_label = "DS4 Server"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.platform_id = ""
        self._pending_command: list[str] = []

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label(self.backend_label, classes="panel-title")
            yield Static(
                "Launch ds4-server in standalone, coordinator, or worker mode. ROCm MXFP4 optimizations are enabled by default; disk KV cache and SSD expert streaming are opt-in host writes.",
                classes="panel-copy",
            )
            with Horizontal(classes="inline-row"):
                yield Label("Engine", id="ds4-engine-label", classes="inline-label")
                yield SearchableSelect("Select Podman or Docker", id="ds4-engine")
            with Horizontal(classes="inline-row"):
                yield Label("Image", id="ds4-image-label", classes="inline-label")
                yield SearchableSelect("Search DS4 images", id="ds4-image")
            with Horizontal(classes="inline-row"):
                yield Label("Model", id="ds4-model-label", classes="inline-label")
                yield SearchableSelect("Search local DS4 GGUFs", id="ds4-model")
                yield Button("Scan", id="ds4-scan-models")
            with Horizontal(classes="compact-fields"):
                with Vertical(classes="compact-field"):
                    yield Label("Context length", id="ds4-context-label", classes="field-label")
                    yield Input(value="126000", placeholder="Context", id="ds4-context")
                with Vertical(classes="compact-field"):
                    yield Label("Graph prefill chunk", id="ds4-prefill-label", classes="field-label")
                    yield Input(placeholder="Auto", id="ds4-prefill")
                with Vertical(classes="compact-field"):
                    yield Label("Host", id="ds4-host-label", classes="field-label")
                    yield Input(value="localhost", placeholder="Host", id="ds4-host")
                with Vertical(classes="compact-field"):
                    yield Label("Port", id="ds4-port-label", classes="field-label")
                    yield Input(value="8000", placeholder="Port", id="ds4-port")
            with Vertical(classes="server-settings"):
                yield Label("ROCm MXFP4 optimizations", classes="settings-title")
                with Horizontal(classes="options-row"):
                    yield Checkbox(
                        "Tile4 kernels — DS4_ROCM_ENABLE_MXFP4_TILE4=1",
                        value=True,
                        id="ds4-mxfp4-tile4-enabled",
                    )
                with Horizontal(classes="options-row"):
                    yield Checkbox(
                        "Down-projection rgroup 4 — DS4_ROCM_MXFP4_DOWN_RGROUP=4",
                        value=True,
                        id="ds4-mxfp4-down-rgroup-enabled",
                    )
            with Horizontal(classes="options-row"):
                yield Switch(value=False, id="ds4-kv-enabled")
                yield Label("Disk KV cache", id="ds4-kv-enabled-label")
            with Horizontal(classes="compact-fields"):
                with Vertical(classes="compact-field"):
                    yield Label("KV cache directory", id="ds4-kv-dir-label", classes="field-label")
                    yield Input(value="~/.cache/ds4-kv", disabled=True, id="ds4-kv-dir")
                with Vertical(classes="compact-field"):
                    yield Label("KV cache size (MB)", id="ds4-kv-mb-label", classes="field-label")
                    yield Input(value="8192", placeholder="Size in MB", disabled=True, id="ds4-kv-mb")
            with Horizontal(classes="options-row"):
                yield Switch(value=False, id="ds4-ssd-enabled")
                yield Label("SSD streaming", id="ds4-ssd-enabled-label")
                yield Switch(value=False, disabled=True, id="ds4-ssd-cold")
                yield Label("Cold preload", id="ds4-ssd-cold-label")
            with Horizontal(classes="compact-fields"):
                with Vertical(classes="compact-field"):
                    yield Label("Expert memory budget", id="ds4-ssd-experts-label", classes="field-label")
                    yield Input(placeholder="For example, 16GB", disabled=True, id="ds4-ssd-experts")
                with Vertical(classes="compact-field"):
                    yield Label("Full expert layers", id="ds4-ssd-layers-label", classes="field-label")
                    yield Input(placeholder="For example, 0", disabled=True, id="ds4-ssd-layers")
            with Horizontal(classes="inline-row"):
                yield Label("MTP model", id="ds4-mtp-label", classes="inline-label")
                yield SearchableSelect("Optional local MTP GGUF", id="ds4-mtp")
            with Horizontal(classes="compact-fields"):
                with Vertical(classes="compact-field"):
                    yield Label("Role", id="ds4-role-label", classes="field-label")
                    yield SearchableSelect("Standalone / Coordinator / Worker", id="ds4-role")
                with Vertical(classes="compact-field"):
                    yield Label("Layer range", id="ds4-layers-label", classes="field-label")
                    yield Input(placeholder="For example, 0:21", id="ds4-layers")
                with Vertical(classes="compact-field"):
                    yield Label("Peer address", id="ds4-peer-label", classes="field-label")
                    yield Input(placeholder="Listen/coordinator IP and port", id="ds4-peer")
            with Horizontal(classes="compact-fields"):
                with Vertical(classes="compact-field"):
                    yield Label("Distributed prefill chunk", id="ds4-dist-prefill-label", classes="field-label")
                    yield Input(placeholder="Auto", disabled=True, id="ds4-dist-prefill")
                with Vertical(classes="compact-field"):
                    yield Label("Distributed prefill window", id="ds4-dist-window-label", classes="field-label")
                    yield Input(placeholder="Auto", disabled=True, id="ds4-dist-window")
            with Horizontal(classes="inline-row"):
                yield Label("Extra args", id="ds4-extra-args-label", classes="inline-label")
                yield Input(id="ds4-extra-args")
            with Horizontal(classes="action-row"):
                yield Button("Start DS4 Server", id="ds4-start", variant="primary")

    def on_mount(self) -> None:
        self.platform_id = self.app.active_platform_id
        engines = [(engine.value, engine.value) for engine in detect_container_engines()]
        engine_select = self.query_one("#ds4-engine", SearchableSelect)
        engine_select.set_options(engines)
        if engines:
            engine_select.value = engines[0][1]
        role = self.query_one("#ds4-role", SearchableSelect)
        role.set_options([(value, value) for value in ("Standalone", "Coordinator", "Worker")])
        role.value = "Standalone"
        self.refresh_platform(self.platform_id)
        self.refresh_models()

    def set_platform(self, platform_id: str) -> None:
        self.platform_id = platform_id
        if self.is_mounted:
            self.refresh_platform(platform_id)

    def refresh_platform(self, platform_id: str) -> None:
        toolboxes = [
            toolbox
            for toolbox in self.app.toolbox_catalog.platform_toolboxes(platform_id)
            if toolbox.backend == "ds4" and toolbox.feature_state("server") != "unavailable"
        ]
        select = self.query_one("#ds4-image", SearchableSelect)
        select.set_options([
            (
                f"{toolbox.name}{' [experimental]' if toolbox.feature_state('server') == 'experimental' else ''} — {toolbox.image}",
                toolbox.id,
            )
            for toolbox in toolboxes
        ])
        default = load_default_toolbox(
            "ds4", platform_id,
            self.app.toolbox_catalog.platform(platform_id).defaults.get("ds4", ""),
        )
        select.value = default if default in {toolbox.id for toolbox in toolboxes} else (toolboxes[0].id if toolboxes else "")

    def refresh_models(self) -> None:
        models = scan_local_models()
        model_select = self.query_one("#ds4-model", SearchableSelect)
        model_select.set_options([(model["name"], model["path"]) for model in models])
        model_select.value = models[0]["path"] if models else ""
        mtp = self.query_one("#ds4-mtp", SearchableSelect)
        mtp_models = [model for model in models if "mtp" in model["name"].lower()]
        mtp.set_options([("None", "")] + [(model["name"], model["path"]) for model in mtp_models])
        mtp.value = ""

    def refresh_model_inventory(self) -> None:
        self.refresh_models()

    @on(Button.Pressed, "#ds4-scan-models")
    def scan_pressed(self) -> None:
        self.refresh_models()
        self.notify("Local DS4 model directory scanned.")

    @on(SearchableSelect.Changed, "#ds4-model")
    @on(SearchableSelect.Changed, "#ds4-role")
    def defaults_changed(self) -> None:
        model = self.query_one("#ds4-model", SearchableSelect).value
        role = self.query_one("#ds4-role", SearchableSelect).value or "Standalone"
        defaults = get_model_server_defaults(model)
        self.query_one("#ds4-prefill", Input).value = str(defaults.get("prefill_chunk", ""))
        coordinator = role == "Coordinator"
        if coordinator:
            self.query_one("#ds4-context", Input).value = str(defaults.get("distributed_ctx", 262144))
            self.query_one("#ds4-layers", Input).value = str(defaults.get("coordinator_layers", "0:21"))
            self.query_one("#ds4-peer", Input).placeholder = "Listen IP Port"
        elif role == "Worker":
            self.query_one("#ds4-context", Input).value = str(defaults.get("distributed_ctx", 262144))
            self.query_one("#ds4-layers", Input).value = str(defaults.get("worker_layers", "22:output"))
            self.query_one("#ds4-peer", Input).placeholder = "Coordinator IP Port"
        else:
            self.query_one("#ds4-context", Input).value = str(defaults.get("standalone_ctx", 126000))
            self.query_one("#ds4-layers", Input).value = ""
        dist_prefill = self.query_one("#ds4-dist-prefill", Input)
        dist_window = self.query_one("#ds4-dist-window", Input)
        dist_prefill.disabled = not coordinator
        dist_window.disabled = not coordinator
        dist_prefill.value = str(defaults.get("dist_prefill_chunk", "")) if coordinator else ""
        dist_window.value = str(defaults.get("dist_prefill_window", "")) if coordinator else ""
        streaming = bool(defaults.get("ssd_streaming", False) if role == "Standalone" else defaults.get("distributed_ssd_streaming", False))
        self.query_one("#ds4-ssd-enabled", Switch).value = streaming
        self.query_one("#ds4-ssd-experts", Input).value = str(defaults.get("ssd_experts", ""))
        self.query_one("#ds4-ssd-layers", Input).value = str(defaults.get("ssd_full_layers", ""))
        self.query_one("#ds4-ssd-cold", Switch).value = bool(defaults.get("ssd_cold", False))

    @on(Switch.Changed, "#ds4-kv-enabled")
    def kv_toggled(self, event: Switch.Changed) -> None:
        self.query_one("#ds4-kv-dir", Input).disabled = not event.value
        self.query_one("#ds4-kv-mb", Input).disabled = not event.value

    @on(Switch.Changed, "#ds4-ssd-enabled")
    def ssd_toggled(self, event: Switch.Changed) -> None:
        self.query_one("#ds4-ssd-experts", Input).disabled = not event.value
        self.query_one("#ds4-ssd-layers", Input).disabled = not event.value
        self.query_one("#ds4-ssd-cold", Switch).disabled = not event.value

    @on(SearchableSelect.Changed, "#ds4-mtp")
    def mtp_changed(self, event: SearchableSelect.Changed) -> None:
        field = self.query_one("#ds4-extra-args", Input)
        tokens = shlex.split(field.value) if field.value else []
        if event.value and "--mtp-draft" not in tokens:
            tokens.extend(["--mtp-draft", "1"])
        elif not event.value and "--mtp-draft" in tokens:
            index = tokens.index("--mtp-draft")
            del tokens[index:index + 2]
        field.value = shlex.join(tokens)

    @staticmethod
    def _optional_positive(value: str, label: str) -> int | None:
        if not value:
            return None
        if not value.isdigit() or int(value) <= 0:
            raise ValueError(f"{label} must be a positive integer or blank")
        return int(value)

    @on(Button.Pressed, "#ds4-start")
    def start_pressed(self) -> None:
        engine = self.query_one("#ds4-engine", SearchableSelect).value
        toolbox_id = self.query_one("#ds4-image", SearchableSelect).value
        model = self.query_one("#ds4-model", SearchableSelect).value
        context = self.query_one("#ds4-context", Input).value
        if not engine or toolbox_id not in self.app.toolbox_catalog.toolboxes or not model:
            self.notify("Select an engine, DS4 image, and local model.", severity="error")
            return
        if not context.isdigit() or int(context) <= 0:
            self.notify("Context must be a positive integer.", severity="error")
            return
        try:
            prefill = self._optional_positive(self.query_one("#ds4-prefill", Input).value.strip(), "Prefill chunk")
            dist_prefill = self._optional_positive(self.query_one("#ds4-dist-prefill", Input).value.strip(), "Distributed prefill chunk")
            dist_window = self._optional_positive(self.query_one("#ds4-dist-window", Input).value.strip(), "Distributed prefill window")
            kv_enabled = self.query_one("#ds4-kv-enabled", Switch).value
            kv_mb = self._optional_positive(self.query_one("#ds4-kv-mb", Input).value.strip(), "KV disk MB") if kv_enabled else None
            kv_dir = ""
            if kv_enabled:
                raw_dir = self.query_one("#ds4-kv-dir", Input).value.strip()
                if not raw_dir:
                    raise ValueError("Choose a host KV-cache directory")
                kv_path = Path(raw_dir).expanduser().resolve()
                kv_path.mkdir(parents=True, exist_ok=True)
                kv_dir = str(kv_path)
        except (ValueError, OSError) as error:
            self.notify(str(error), severity="error")
            return
        toolbox = self.app.toolbox_catalog.toolboxes[toolbox_id]
        profile = self.app.toolbox_catalog.runtime_profiles[toolbox.runtime_profile]
        role = self.query_one("#ds4-role", SearchableSelect).value or "Standalone"
        self._pending_command = build_server_cmd(
            engine, toolbox.image, model, int(context),
            self.query_one("#ds4-host", Input).value,
            self.query_one("#ds4-port", Input).value,
            kv_enabled, kv_dir, kv_mb or 0, prefill,
            self.query_one("#ds4-mtp", SearchableSelect).value,
            self.query_one("#ds4-extra-args", Input).value,
            role,
            self.query_one("#ds4-layers", Input).value,
            self.query_one("#ds4-peer", Input).value,
            {"args": list(profile.engine_args), "server_binary": toolbox.server_binary or "ds4-server"},
            self.query_one("#ds4-ssd-enabled", Switch).value,
            self.query_one("#ds4-ssd-experts", Input).value,
            self.query_one("#ds4-ssd-layers", Input).value,
            self.query_one("#ds4-ssd-cold", Switch).value,
            dist_prefill if role == "Coordinator" else None,
            dist_window if role == "Coordinator" else None,
            self.query_one("#ds4-mxfp4-tile4-enabled", Checkbox).value,
            self.query_one("#ds4-mxfp4-down-rgroup-enabled", Checkbox).value,
        )
        self.app.push_screen(
            ConfirmModal(f"Start DS4 server?\n\n{shlex.join(self._pending_command)}", yes_text="Start"),
            self._start_confirmed,
        )

    def _start_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        command = self._pending_command
        with self.app.suspend():
            run_foreground_server(command, command[0], "ds4-cockpit-server")
