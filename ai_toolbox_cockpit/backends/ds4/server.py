"""DS4 server UI, including distributed and SSD-streaming controls."""

import shlex
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.markup import escape
from textual.widgets import Button, Checkbox, Input, Label, Static, TextArea

from ai_toolbox_cockpit.backends.base import BackendServerPanel
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.server_process import run_foreground_server
from ai_toolbox_cockpit.settings import load_default_toolbox
from ai_toolbox_cockpit.widgets import CockpitCheckbox, ConfirmModal, SearchableSelect

from .config import (
    DEEPSEEK_V41_FAMILIES,
    get_artifact_role,
    get_model_artifact,
    get_model_server_defaults,
    is_tensor_parallel_cli_worker,
    resolve_server_binary,
)
from .launch_settings import (
    clear_launch_settings,
    describe_changes,
    load_engine,
    load_last_model,
    load_launch_settings,
    save_last_launch,
    save_launch_settings,
)
from .model_manager import scan_local_models
from .server_runner import build_server_cmd

# Form values that do not come from the model catalog.
DEFAULT_HOST = "localhost"
DEFAULT_PORT = "8000"
DEFAULT_KV_DIR = "~/.cache/ds4-kv"
DEFAULT_KV_MB = "8192"


class Ds4ServerPanel(BackendServerPanel):
    backend_label = "DwarfStar (ds4) Server"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.platform_id = ""
        self._current_model_path = ""
        self._dspark_support_models: list[dict[str, str]] = []
        self._vision_encoders: list[dict[str, str]] = []
        self._mtp_models: list[dict[str, str]] = []
        self._pending_command: list[str] = []
        self._pending_launch: tuple[str, str, str, dict] | None = None
        self._pending_reset: tuple[str, str] | None = None
        self._model_defaults: dict = {}
        self._tensor_parallel_available = False

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
                yield SearchableSelect("Search DwarfStar (ds4) images", id="ds4-image")
            with Horizontal(classes="inline-row"):
                yield Label("Model", id="ds4-model-label", classes="inline-label")
                yield SearchableSelect("Search local DwarfStar (ds4) GGUFs", id="ds4-model")
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
                    yield Input(value=DEFAULT_HOST, placeholder="Host", id="ds4-host")
                with Vertical(classes="compact-field"):
                    yield Label("Port", id="ds4-port-label", classes="field-label")
                    yield Input(value=DEFAULT_PORT, placeholder="Port", id="ds4-port")
            with Vertical(id="ds4-mxfp4-zone", classes="server-settings"):
                yield Label("ROCm MXFP4 optimizations", classes="settings-title")
                with Horizontal(classes="options-row"):
                    yield CockpitCheckbox(
                        "Tile4 kernels — DS4_ROCM_ENABLE_MXFP4_TILE4=1",
                        value=False,
                        id="ds4-mxfp4-tile4-enabled",
                    )
                with Horizontal(classes="options-row"):
                    yield CockpitCheckbox(
                        "Down-projection rgroup 4 — DS4_ROCM_MXFP4_DOWN_RGROUP=4",
                        value=False,
                        id="ds4-mxfp4-down-rgroup-enabled",
                    )
            with Vertical(id="ds4-v41-bounded-replay-zone", classes="server-settings"):
                yield Label("DeepSeek V4.1 decoder", classes="settings-title")
                with Horizontal(classes="options-row"):
                    yield CockpitCheckbox(
                        "SWA bounded replay",
                        value=False,
                        id="ds4-v41-bounded-replay-enabled",
                    )
                    yield Static("Faster approximate inference (n=128).")
            with Horizontal(classes="options-row"):
                yield CockpitCheckbox("Disk KV cache", value=False, id="ds4-kv-enabled")
            with Horizontal(classes="compact-fields"):
                with Vertical(classes="compact-field"):
                    yield Label("KV cache directory", id="ds4-kv-dir-label", classes="field-label")
                    yield Input(value=DEFAULT_KV_DIR, disabled=True, id="ds4-kv-dir")
                with Vertical(classes="compact-field"):
                    yield Label("KV cache size (MB)", id="ds4-kv-mb-label", classes="field-label")
                    yield Input(value=DEFAULT_KV_MB, placeholder="Size in MB", disabled=True, id="ds4-kv-mb")
            with Horizontal(classes="options-row"):
                yield CockpitCheckbox("SSD streaming", value=False, id="ds4-ssd-enabled")
                yield CockpitCheckbox("Cold preload", value=False, disabled=True, id="ds4-ssd-cold")
            with Horizontal(classes="compact-fields"):
                with Vertical(classes="compact-field"):
                    yield Label("Expert memory budget", id="ds4-ssd-experts-label", classes="field-label")
                    yield Input(placeholder="For example, 16GB", disabled=True, id="ds4-ssd-experts")
                with Vertical(classes="compact-field"):
                    yield Label("Full expert layers", id="ds4-ssd-layers-label", classes="field-label")
                    yield Input(placeholder="For example, 0", disabled=True, id="ds4-ssd-layers")
            with Vertical(id="ds4-dspark-zone", classes="model-zone"):
                yield Label("DSpark speculative decoding", classes="zone-title")
                yield CockpitCheckbox("Enable DSpark", value=False, id="ds4-dspark-enabled")
                with Horizontal(classes="inline-row"):
                    yield Label("Support model", id="ds4-dspark-model-label", classes="inline-label")
                    yield SearchableSelect("Select the DSpark support GGUF", id="ds4-dspark-model")
                with Horizontal(classes="compact-fields"):
                    with Vertical(classes="compact-field"):
                        yield Label("Confidence threshold", id="ds4-dspark-confidence-label", classes="field-label")
                        yield Input(value="0.7", placeholder="Auto", id="ds4-dspark-confidence")
                yield Static("", id="ds4-dspark-note")
            with Horizontal(classes="options-row"):
                yield CockpitCheckbox(
                    "MTP heads embedded in model (--mtp)", value=False, id="ds4-mtp-enabled"
                )
            with Horizontal(classes="inline-row"):
                yield Label("External MTP model", id="ds4-mtp-label", classes="inline-label")
                yield SearchableSelect("Optional local MTP GGUF", id="ds4-mtp")
            with Horizontal(classes="inline-row"):
                yield Label("Vision encoder", id="ds4-vision-label", classes="inline-label")
                yield SearchableSelect("Optional compatible vision encoder", id="ds4-vision")
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
            with Vertical(id="ds4-tp-zone", classes="model-zone"):
                yield Label("Tensor-parallel distributed", classes="zone-title")
                with Horizontal(classes="options-row"):
                    yield CockpitCheckbox(
                        "Tensor parallel (--tensor-parallel)", value=False, id="ds4-tensor-parallel"
                    )
                with Horizontal(classes="inline-row"):
                    yield Label("Transport", id="ds4-transport-label", classes="inline-label")
                    yield SearchableSelect("TCP or RoCE", id="ds4-transport")
                with Horizontal(classes="compact-fields"):
                    with Vertical(classes="compact-field"):
                        yield Label("RDMA device", id="ds4-rdma-device-label", classes="field-label")
                        yield Input(placeholder="For example, rocep194s0", disabled=True, id="ds4-rdma-device")
                    with Vertical(classes="compact-field"):
                        yield Label("RDMA port", id="ds4-rdma-port-label", classes="field-label")
                        yield Input(placeholder="For example, 1", disabled=True, id="ds4-rdma-port")
                    with Vertical(classes="compact-field"):
                        yield Label("RDMA GID index", id="ds4-rdma-gid-label", classes="field-label")
                        yield Input(placeholder="For example, 1", disabled=True, id="ds4-rdma-gid")
                yield Static("", id="ds4-tp-note", classes="panel-copy")
            with Horizontal(classes="extra-args-row"):
                yield Label("Extra args", id="ds4-extra-args-label", classes="inline-label")
                yield TextArea(
                    soft_wrap=True,
                    compact=True,
                    highlight_cursor_line=False,
                    id="ds4-extra-args",
                )
            with Horizontal(classes="action-row"):
                yield Button("Start DwarfStar (ds4) Server", id="ds4-start", variant="primary")
                yield Button("Reset to defaults", id="ds4-reset")

    def on_mount(self) -> None:
        self.platform_id = self.app.active_platform_id
        engines = [(engine.value, engine.value) for engine in detect_container_engines()]
        engine_select = self.query_one("#ds4-engine", SearchableSelect)
        engine_select.set_options(engines)
        if engines:
            saved_engine = load_engine()
            engine_select.value = saved_engine if saved_engine in dict(engines) else engines[0][1]
        role = self.query_one("#ds4-role", SearchableSelect)
        role.set_options([(value, value) for value in ("Standalone", "Coordinator", "Worker")])
        role.value = "Standalone"
        transport = self.query_one("#ds4-transport", SearchableSelect)
        transport.set_options([("TCP", "tcp"), ("RoCE", "rdma")])
        transport.value = "tcp"
        self.query_one("#ds4-tp-zone", Vertical).styles.display = "none"
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
        self._dspark_support_models = [
            model for model in models
            if get_artifact_role(model["path"]) == "dspark_support"
        ]
        self._vision_encoders = [
            model for model in models
            if get_artifact_role(model["path"]) == "vision_encoder"
        ]
        mtp_models = [
            model for model in models
            if get_artifact_role(model["path"]) == "mtp"
        ]
        self._mtp_models = mtp_models
        auxiliary_paths = {
            model["path"]
            for model in (*self._dspark_support_models, *self._vision_encoders, *mtp_models)
        }
        target_models = [model for model in models if model["path"] not in auxiliary_paths]
        model_select = self.query_one("#ds4-model", SearchableSelect)
        model_select.set_options([(model["name"], model["path"]) for model in target_models])
        recommended = next(
            (
                model
                for model in target_models
                if get_model_artifact(model["path"]).get("recommended")
            ),
            None,
        )
        last_name = load_last_model()
        last_launched = next(
            (model for model in target_models if model["name"] == last_name), None
        ) if last_name else None
        chosen = last_launched or recommended
        model_select.value = (
            chosen["path"]
            if chosen
            else (target_models[0]["path"] if target_models else "")
        )
        dspark = self.query_one("#ds4-dspark-model", SearchableSelect)
        dspark.set_options([(model["name"], model["path"]) for model in self._dspark_support_models])
        dspark.value = self._dspark_support_models[0]["path"] if self._dspark_support_models else ""
        mtp = self.query_one("#ds4-mtp", SearchableSelect)
        mtp.set_options([("None", "")] + [(model["name"], model["path"]) for model in mtp_models])
        mtp.value = ""
        vision = self.query_one("#ds4-vision", SearchableSelect)
        vision.set_options([("None", "")])
        vision.value = ""
        self.defaults_changed()

    def refresh_model_inventory(self) -> None:
        self.refresh_models()

    @on(Button.Pressed, "#ds4-scan-models")
    def scan_pressed(self) -> None:
        self.refresh_models()
        self.notify("Local DwarfStar (ds4) model directory scanned.")

    @on(SearchableSelect.Changed, "#ds4-model")
    @on(SearchableSelect.Changed, "#ds4-role")
    def defaults_changed(self) -> None:
        self._apply_form_defaults()

    def _apply_form_defaults(self, reset: bool = False) -> None:
        """Set the controls from the model catalog, then from saved settings.

        With ``reset`` the controls are treated as if the model had just been
        selected, and saved settings are not applied.
        """
        model = self.query_one("#ds4-model", SearchableSelect).value
        model_changed = reset or model != self._current_model_path
        self._current_model_path = model
        role = self.query_one("#ds4-role", SearchableSelect).value or "Standalone"
        defaults = get_model_server_defaults(model)
        self._model_defaults = defaults
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
        self.query_one("#ds4-ssd-enabled", CockpitCheckbox).value = streaming
        self.query_one("#ds4-ssd-experts", Input).value = str(defaults.get("ssd_experts", ""))
        self.query_one("#ds4-ssd-layers", Input).value = str(defaults.get("ssd_full_layers", ""))
        self.query_one("#ds4-ssd-cold", CockpitCheckbox).value = bool(defaults.get("ssd_cold", False)) if streaming else False
        self._refresh_mxfp4_controls(model, model_changed)
        self._refresh_v41_bounded_replay_control(model, model_changed)
        self._refresh_vision_control(model, model_changed)
        if model_changed:
            self.query_one("#ds4-mtp-enabled", Checkbox).value = False
        self._refresh_tensor_parallel_controls(defaults, role, model_changed)
        self._refresh_dspark_controls(defaults, role, model_changed)
        if not reset:
            self._apply_saved_launch_settings(model, role)

    def _collect_launch_settings(self) -> dict[str, str | bool]:
        """Read every user-editable launch control for saving."""

        def text(control_id: str) -> str:
            return self.query_one(f"#{control_id}", Input).value.strip()

        def flag(control_id: str) -> bool:
            return bool(self.query_one(f"#{control_id}", Checkbox).value)

        def file_name(control_id: str) -> str:
            value = self.query_one(f"#{control_id}", SearchableSelect).value
            return Path(value).name if value else ""

        return {
            "context": text("ds4-context"),
            "prefill": text("ds4-prefill"),
            "host": text("ds4-host"),
            "port": text("ds4-port"),
            "kv_enabled": flag("ds4-kv-enabled"),
            "kv_dir": text("ds4-kv-dir"),
            "kv_mb": text("ds4-kv-mb"),
            "ssd_enabled": flag("ds4-ssd-enabled"),
            "ssd_cold": flag("ds4-ssd-cold"),
            "ssd_experts": text("ds4-ssd-experts"),
            "ssd_layers": text("ds4-ssd-layers"),
            "mxfp4_tile4": flag("ds4-mxfp4-tile4-enabled"),
            "mxfp4_down_rgroup": flag("ds4-mxfp4-down-rgroup-enabled"),
            "v41_bounded_replay": flag("ds4-v41-bounded-replay-enabled"),
            "dspark_enabled": flag("ds4-dspark-enabled"),
            "dspark_confidence": text("ds4-dspark-confidence"),
            "mtp_enabled": flag("ds4-mtp-enabled"),
            "mtp": file_name("ds4-mtp"),
            "vision": file_name("ds4-vision"),
            "layers": text("ds4-layers"),
            "peer": text("ds4-peer"),
            "dist_prefill": text("ds4-dist-prefill"),
            "dist_window": text("ds4-dist-window"),
            "tensor_parallel": flag("ds4-tensor-parallel"),
            "transport": self.query_one("#ds4-transport", SearchableSelect).value or "",
            "rdma_device": text("ds4-rdma-device"),
            "rdma_port": text("ds4-rdma-port"),
            "rdma_gid": text("ds4-rdma-gid"),
            "extra_args": self.query_one("#ds4-extra-args", TextArea).text,
        }

    def _apply_saved_launch_settings(self, model_path: str, role: str) -> None:
        """Overlay the saved values for this model and role on the catalog defaults.

        Runs after the catalog defaults, so a missing key keeps its default.
        Controls that the current model, role or files do not allow (DSpark
        without its support file, tensor parallel in standalone mode, an
        encoder that is not installed) are left as they are.
        """
        saved = load_launch_settings(Path(model_path).name, role) if model_path else {}
        if saved:
            self._apply_launch_values(saved, model_path)

    def _apply_launch_values(self, saved: dict, model_path: str) -> None:
        """Put the given form values into the controls that currently allow them."""

        def text(key: str, control_id: str) -> None:
            value = saved.get(key)
            if isinstance(value, str):
                self.query_one(f"#{control_id}", Input).value = value

        def flag(key: str, control_id: str) -> None:
            value = saved.get(key)
            control = self.query_one(f"#{control_id}", Checkbox)
            if isinstance(value, bool) and not control.disabled:
                control.value = value

        def choose(key: str, control_id: str, candidates: list[dict[str, str]]) -> None:
            name = saved.get(key)
            control = self.query_one(f"#{control_id}", SearchableSelect)
            if not isinstance(name, str) or control.disabled:
                return
            path = next((item["path"] for item in candidates if item["name"] == name), "")
            if (path or not name) and control.value != path:
                # The extra args are restored as saved, so the MTP handler that
                # edits them must not run for this change.
                with control.prevent(SearchableSelect.Changed):
                    control.value = path

        for key, control_id in (
            ("context", "ds4-context"), ("prefill", "ds4-prefill"),
            ("host", "ds4-host"), ("port", "ds4-port"),
            ("layers", "ds4-layers"), ("peer", "ds4-peer"),
            ("dist_prefill", "ds4-dist-prefill"), ("dist_window", "ds4-dist-window"),
        ):
            text(key, control_id)

        kv = self.query_one("#ds4-kv-enabled", Checkbox)
        flag("kv_enabled", "ds4-kv-enabled")
        text("kv_dir", "ds4-kv-dir")
        text("kv_mb", "ds4-kv-mb")
        self.query_one("#ds4-kv-dir", Input).disabled = not kv.value
        self.query_one("#ds4-kv-mb", Input).disabled = not kv.value

        if "mxfp4" in Path(model_path).name.lower():
            flag("mxfp4_tile4", "ds4-mxfp4-tile4-enabled")
            flag("mxfp4_down_rgroup", "ds4-mxfp4-down-rgroup-enabled")
        flag("v41_bounded_replay", "ds4-v41-bounded-replay-enabled")

        # DSpark and embedded MTP exclude each other and SSD streaming. Set the
        # DSpark switch first and apply its rules at once, so the SSD and MTP
        # controls below see the state they will have after the change.
        flag("dspark_enabled", "ds4-dspark-enabled")
        text("dspark_confidence", "ds4-dspark-confidence")
        self._sync_dspark_controls()
        flag("mtp_enabled", "ds4-mtp-enabled")
        embedded_mtp = self.query_one("#ds4-mtp-enabled", Checkbox)
        if embedded_mtp.value:
            external_mtp = self.query_one("#ds4-mtp", SearchableSelect)
            if external_mtp.value:
                with external_mtp.prevent(SearchableSelect.Changed):
                    external_mtp.value = ""
        else:
            choose("mtp", "ds4-mtp", self._mtp_models)
        self._sync_dspark_controls()

        ssd = self.query_one("#ds4-ssd-enabled", Checkbox)
        flag("ssd_enabled", "ds4-ssd-enabled")
        text("ssd_experts", "ds4-ssd-experts")
        text("ssd_layers", "ds4-ssd-layers")
        cold = self.query_one("#ds4-ssd-cold", Checkbox)
        cold_saved = saved.get("ssd_cold")
        if isinstance(cold_saved, bool):
            cold.value = cold_saved and bool(ssd.value)
        cold.disabled = not ssd.value
        self.query_one("#ds4-ssd-experts", Input).disabled = not ssd.value
        self.query_one("#ds4-ssd-layers", Input).disabled = not ssd.value

        family = get_model_artifact(model_path).get("family")
        choose(
            "vision", "ds4-vision",
            [
                encoder for encoder in self._vision_encoders
                if family and get_model_artifact(encoder["path"]).get("family") == family
            ],
        )

        flag("tensor_parallel", "ds4-tensor-parallel")
        transport = saved.get("transport")
        if (
            self._tensor_parallel_available
            and transport in ("tcp", "rdma")
            and self.query_one("#ds4-transport", SearchableSelect).value != transport
        ):
            self.query_one("#ds4-transport", SearchableSelect).value = transport
        text("rdma_device", "ds4-rdma-device")
        text("rdma_port", "ds4-rdma-port")
        text("rdma_gid", "ds4-rdma-gid")
        self._sync_transport_controls()

        extra_args = saved.get("extra_args")
        if isinstance(extra_args, str):
            self.query_one("#ds4-extra-args", TextArea).text = extra_args

    def _refresh_mxfp4_controls(self, model_path: str, model_changed: bool) -> None:
        supported = "mxfp4" in Path(model_path).name.lower()
        self.query_one("#ds4-mxfp4-zone", Vertical).styles.display = (
            "block" if supported else "none"
        )
        tile4 = self.query_one("#ds4-mxfp4-tile4-enabled", CockpitCheckbox)
        rgroup = self.query_one(
            "#ds4-mxfp4-down-rgroup-enabled", CockpitCheckbox
        )
        if model_changed:
            tile4.value = supported
            rgroup.value = supported
        elif not supported:
            tile4.value = False
            rgroup.value = False

    def _refresh_v41_bounded_replay_control(
        self, model_path: str, model_changed: bool
    ) -> None:
        supported = get_model_artifact(model_path).get("family") in DEEPSEEK_V41_FAMILIES
        self.query_one("#ds4-v41-bounded-replay-zone", Vertical).styles.display = (
            "block" if supported else "none"
        )
        replay = self.query_one("#ds4-v41-bounded-replay-enabled", CockpitCheckbox)
        replay.disabled = not supported
        if model_changed or not supported:
            replay.value = supported and bool(
                self._model_defaults.get("v41_decoder_swa_bounded_replay", False)
            )

    def _refresh_vision_control(self, model_path: str, model_changed: bool) -> None:
        family = get_model_artifact(model_path).get("family")
        compatible = [
            encoder
            for encoder in self._vision_encoders
            if family and get_model_artifact(encoder["path"]).get("family") == family
        ]
        vision = self.query_one("#ds4-vision", SearchableSelect)
        options = [("None", "")] + [
            (encoder["name"], encoder["path"])
            for encoder in compatible
        ]
        previous = vision.value
        vision.set_options(options)
        valid_values = {value for _, value in options}
        vision.value = "" if model_changed or previous not in valid_values else previous
        vision.disabled = not compatible

    def _refresh_dspark_controls(self, defaults: dict, role: str, model_changed: bool) -> None:
        support_filename = str(defaults.get("dspark_support_filename", ""))
        matches = [
            model for model in self._dspark_support_models
            if model["name"] == support_filename
        ]
        zone = self.query_one("#ds4-dspark-zone", Vertical)
        enabled = self.query_one("#ds4-dspark-enabled", Checkbox)
        support = self.query_one("#ds4-dspark-model", SearchableSelect)
        confidence = self.query_one("#ds4-dspark-confidence", Input)
        supported = bool(self._current_model_path and support_filename)
        distributed = bool(defaults.get("dspark_distributed", False))
        available = bool(matches) and (role == "Standalone" or distributed)
        zone.styles.display = "block" if supported else "none"
        support.set_options([(model["name"], model["path"]) for model in matches])
        support.value = matches[0]["path"] if matches else ""
        enabled.disabled = not available
        if model_changed:
            enabled.value = bool(defaults.get("dspark_enabled", False)) and available
            confidence.value = (
                "" if defaults.get("dspark_confidence_auto", False)
                else str(defaults.get("dspark_confidence", 0.7))
            )
        elif not available:
            enabled.value = False
        note = ""
        if supported and not matches:
            note = f"Download {support_filename} to enable DSpark."
        elif supported and role != "Standalone" and not distributed:
            note = "DSpark is available only in standalone mode."
        elif supported:
            note = (
                "The support model uses a fixed five-token draft cap and the adaptive "
                "scheduler. Auto leaves confidence to the runtime; enter a value "
                "only to override it."
            )
        self.query_one("#ds4-dspark-note", Static).update(note)
        self._sync_dspark_controls()

    def _sync_dspark_controls(self) -> None:
        checkbox = self.query_one("#ds4-dspark-enabled", Checkbox)
        active = checkbox.value and not checkbox.disabled
        mtp = self.query_one("#ds4-mtp-enabled", Checkbox)
        mtp.disabled = active
        if active:
            mtp.value = False
        self.query_one("#ds4-mtp", SearchableSelect).disabled = active or mtp.value
        self.query_one("#ds4-dspark-model", SearchableSelect).disabled = not active
        self.query_one("#ds4-dspark-confidence", Input).disabled = not active
        ssd = self.query_one("#ds4-ssd-enabled", CockpitCheckbox)
        if active:
            ssd.value = False
        ssd.disabled = active

    def _refresh_tensor_parallel_controls(self, defaults: dict, role: str, model_changed: bool) -> None:
        supported = bool(defaults.get("tensor_parallel", False))
        self._tensor_parallel_available = supported and role != "Standalone"
        zone = self.query_one("#ds4-tp-zone", Vertical)
        zone.styles.display = "block" if self._tensor_parallel_available else "none"
        tensor = self.query_one("#ds4-tensor-parallel", Checkbox)
        tensor.disabled = not self._tensor_parallel_available
        if model_changed:
            tensor.value = supported
        elif not supported:
            tensor.value = False
        transport = self.query_one("#ds4-transport", SearchableSelect)
        if model_changed:
            transport.value = str(defaults.get("distributed_transport", "tcp"))
        device = self.query_one("#ds4-rdma-device", Input)
        port = self.query_one("#ds4-rdma-port", Input)
        gid = self.query_one("#ds4-rdma-gid", Input)
        if model_changed:
            device.value = str(defaults.get("rdma_device", ""))
            port.value = str(defaults.get("rdma_port", ""))
            gid.value = str(defaults.get("rdma_gid_index", ""))
        self._sync_transport_controls()

    def _sync_transport_controls(self) -> None:
        available = self._tensor_parallel_available
        tensor = self.query_one("#ds4-tensor-parallel", Checkbox)
        transport = self.query_one("#ds4-transport", SearchableSelect)
        tensor_active = available and tensor.value
        transport.disabled = not tensor_active
        rdma_active = tensor_active and transport.value == "rdma"
        for control_id in ("ds4-rdma-device", "ds4-rdma-port", "ds4-rdma-gid"):
            self.query_one(f"#{control_id}", Input).disabled = not rdma_active
        layers = self.query_one("#ds4-layers", Input)
        layers.disabled = tensor_active
        if tensor_active:
            layers.value = ""
        self._sync_worker_cli_controls(tensor_active)

    def _sync_worker_cli_controls(self, tensor_active: bool) -> None:
        role = self.query_one("#ds4-role", SearchableSelect).value or "Standalone"
        cli_worker = is_tensor_parallel_cli_worker(self._current_model_path, role, tensor_active)
        for control_id in ("ds4-host", "ds4-port"):
            self.query_one(f"#{control_id}", Input).disabled = cli_worker
        note = self.query_one("#ds4-tp-note", Static)
        if not cli_worker:
            note.update("")
            return
        binary = resolve_server_binary(
            self._current_model_path, role, True, self._toolbox_server_binary()
        )
        note.update(
            f"Tensor-parallel workers run {binary} instead of "
            f"{self._toolbox_server_binary()}. {binary} serves no HTTP API, so host "
            "and port are not passed."
        )

    def _toolbox_server_binary(self) -> str:
        toolbox = self.app.toolbox_catalog.toolboxes.get(
            self.query_one("#ds4-image", SearchableSelect).value
        )
        return (toolbox.server_binary if toolbox else "") or "ds4-server"

    @on(Checkbox.Changed, "#ds4-tensor-parallel")
    def tensor_parallel_toggled(self) -> None:
        self._sync_transport_controls()

    @on(SearchableSelect.Changed, "#ds4-transport")
    def transport_changed(self) -> None:
        self._sync_transport_controls()

    @on(Checkbox.Changed, "#ds4-kv-enabled")
    def kv_toggled(self, event: Checkbox.Changed) -> None:
        self.query_one("#ds4-kv-dir", Input).disabled = not event.value
        self.query_one("#ds4-kv-mb", Input).disabled = not event.value

    @on(Checkbox.Changed, "#ds4-ssd-enabled")
    def ssd_toggled(self, event: Checkbox.Changed) -> None:
        self.query_one("#ds4-ssd-experts", Input).disabled = not event.value
        self.query_one("#ds4-ssd-layers", Input).disabled = not event.value
        cold = self.query_one("#ds4-ssd-cold", CockpitCheckbox)
        if not event.value:
            cold.value = False
        cold.disabled = not event.value

    @on(Checkbox.Changed, "#ds4-dspark-enabled")
    def dspark_toggled(self) -> None:
        self._sync_dspark_controls()

    @on(Checkbox.Changed, "#ds4-mtp-enabled")
    def embedded_mtp_toggled(self, event: Checkbox.Changed) -> None:
        if event.value:
            self.query_one("#ds4-mtp", SearchableSelect).value = ""
        self._sync_dspark_controls()

    @on(SearchableSelect.Changed, "#ds4-mtp")
    def mtp_changed(self, event: SearchableSelect.Changed) -> None:
        field = self.query_one("#ds4-extra-args", TextArea)
        tokens = shlex.split(field.text) if field.text else []
        if event.value and "--mtp-draft" not in tokens:
            tokens.extend(["--mtp-draft", "1"])
        elif not event.value and "--mtp-draft" in tokens:
            index = tokens.index("--mtp-draft")
            del tokens[index:index + 2]
        field.text = shlex.join(tokens)

    @staticmethod
    def _optional_positive(value: str, label: str) -> int | None:
        if not value:
            return None
        if not value.isdigit() or int(value) <= 0:
            raise ValueError(f"{label} must be a positive integer or blank")
        return int(value)

    @staticmethod
    def _probability(value: str, label: str) -> float:
        try:
            result = float(value)
        except ValueError as error:
            raise ValueError(f"{label} must be between 0 and 1") from error
        if not 0.0 <= result <= 1.0:
            raise ValueError(f"{label} must be between 0 and 1")
        return result

    @on(Button.Pressed, "#ds4-start")
    def start_pressed(self) -> None:
        engine = self.query_one("#ds4-engine", SearchableSelect).value
        toolbox_id = self.query_one("#ds4-image", SearchableSelect).value
        model = self.query_one("#ds4-model", SearchableSelect).value
        context = self.query_one("#ds4-context", Input).value
        if not engine or toolbox_id not in self.app.toolbox_catalog.toolboxes or not model:
            self.notify("Select an engine, DwarfStar (ds4) image, and local model.", severity="error")
            return
        if not context.isdigit() or int(context) <= 0:
            self.notify("Context must be a positive integer.", severity="error")
            return
        try:
            prefill = self._optional_positive(self.query_one("#ds4-prefill", Input).value.strip(), "Prefill chunk")
            dist_prefill = self._optional_positive(self.query_one("#ds4-dist-prefill", Input).value.strip(), "Distributed prefill chunk")
            dist_window = self._optional_positive(self.query_one("#ds4-dist-window", Input).value.strip(), "Distributed prefill window")
            kv_enabled = self.query_one("#ds4-kv-enabled", CockpitCheckbox).value
            kv_mb = self._optional_positive(self.query_one("#ds4-kv-mb", Input).value.strip(), "KV disk MB") if kv_enabled else None
            kv_dir = ""
            if kv_enabled:
                raw_dir = self.query_one("#ds4-kv-dir", Input).value.strip()
                if not raw_dir:
                    raise ValueError("Choose a host KV-cache directory")
                kv_path = Path(raw_dir).expanduser().resolve()
                kv_path.mkdir(parents=True, exist_ok=True)
                kv_dir = str(kv_path)
            dspark_enabled = self.query_one("#ds4-dspark-enabled", Checkbox).value
            raw_dspark_confidence = self.query_one(
                "#ds4-dspark-confidence", Input
            ).value.strip()
            dspark_confidence = (
                self._probability(raw_dspark_confidence, "DSpark confidence")
                if dspark_enabled and raw_dspark_confidence
                else None
            )
        except (ValueError, OSError) as error:
            self.notify(str(error), severity="error")
            return
        toolbox = self.app.toolbox_catalog.toolboxes[toolbox_id]
        profile = self.app.toolbox_catalog.runtime_profiles[toolbox.runtime_profile]
        role = self.query_one("#ds4-role", SearchableSelect).value or "Standalone"
        tensor_parallel = (
            self._tensor_parallel_available
            and self.query_one("#ds4-tensor-parallel", Checkbox).value
        )
        transport = (
            self.query_one("#ds4-transport", SearchableSelect).value or ""
        ) if tensor_parallel else ""
        rdma_device = self.query_one("#ds4-rdma-device", Input).value.strip()
        rdma_port = self.query_one("#ds4-rdma-port", Input).value.strip()
        rdma_gid_index = self.query_one("#ds4-rdma-gid", Input).value.strip()
        if role != "Standalone" and transport == "rdma":
            if not rdma_device:
                self.notify("RoCE transport requires an RDMA device.", severity="error")
                return
            if rdma_port and not rdma_port.isdigit():
                self.notify("RDMA port must be a positive integer.", severity="error")
                return
            if rdma_gid_index and not rdma_gid_index.isdigit():
                self.notify("RDMA GID index must be a non-negative integer.", severity="error")
                return
        try:
            self._pending_command = build_server_cmd(
                engine, toolbox.image, model, int(context),
                self.query_one("#ds4-host", Input).value,
                self.query_one("#ds4-port", Input).value,
                kv_enabled, kv_dir, kv_mb or 0, prefill,
                self.query_one("#ds4-mtp", SearchableSelect).value,
                self.query_one("#ds4-extra-args", TextArea).text,
                role,
                self.query_one("#ds4-layers", Input).value,
                self.query_one("#ds4-peer", Input).value,
                {"args": list(profile.engine_args), "server_binary": toolbox.server_binary or "ds4-server"},
                self.query_one("#ds4-ssd-enabled", CockpitCheckbox).value,
                self.query_one("#ds4-ssd-experts", Input).value,
                self.query_one("#ds4-ssd-layers", Input).value,
                self.query_one("#ds4-ssd-cold", CockpitCheckbox).value,
                dist_prefill if role == "Coordinator" else None,
                dist_window if role == "Coordinator" else None,
                self.query_one("#ds4-mxfp4-tile4-enabled", Checkbox).value,
                self.query_one("#ds4-mxfp4-down-rgroup-enabled", Checkbox).value,
                v41_decoder_swa_bounded_replay_enabled=self.query_one(
                    "#ds4-v41-bounded-replay-enabled", Checkbox
                ).value,
                dspark_enabled=dspark_enabled,
                dspark_path=self.query_one("#ds4-dspark-model", SearchableSelect).value,
                dspark_confidence=dspark_confidence,
                dspark_distributed=bool(self._model_defaults.get("dspark_distributed", False)),
                vision_path=self.query_one("#ds4-vision", SearchableSelect).value,
                mtp_enabled=self.query_one("#ds4-mtp-enabled", Checkbox).value,
                tensor_parallel=tensor_parallel,
                transport=transport,
                rdma_device=rdma_device,
                rdma_port=rdma_port,
                rdma_gid_index=rdma_gid_index,
                peer_default_port=str(self._model_defaults.get("distributed_port", 8081)),
            )
        except ValueError as error:
            self.notify(str(error), severity="error")
            return
        self._pending_launch = (engine, Path(model).name, role, self._collect_launch_settings())
        self.app.push_screen(
            ConfirmModal(
                f"Start DwarfStar (ds4) server?\n\n{shlex.join(self._pending_command)}",
                yes_text="Start",
                copy_text=shlex.join(self._pending_command),
            ),
            self._start_confirmed,
        )

    def _start_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        if self._pending_launch is not None:
            engine, model_name, role, values = self._pending_launch
            if not (
                save_launch_settings(model_name, role, values)
                and save_last_launch(engine, model_name)
            ):
                self.notify("Could not save settings; using them for this session.", severity="warning")
        command = self._pending_command
        with self.app.suspend():
            run_foreground_server(command, command[0], "ds4-cockpit-server")

    def _reset_form(self) -> None:
        """Put every launch control back to its default for the current model and role."""
        self.query_one("#ds4-host", Input).value = DEFAULT_HOST
        self.query_one("#ds4-port", Input).value = DEFAULT_PORT
        self.query_one("#ds4-peer", Input).value = ""
        self.query_one("#ds4-kv-enabled", Checkbox).value = False
        kv_dir = self.query_one("#ds4-kv-dir", Input)
        kv_dir.value, kv_dir.disabled = DEFAULT_KV_DIR, True
        kv_mb = self.query_one("#ds4-kv-mb", Input)
        kv_mb.value, kv_mb.disabled = DEFAULT_KV_MB, True
        self.query_one("#ds4-extra-args", TextArea).text = ""
        external_mtp = self.query_one("#ds4-mtp", SearchableSelect)
        if external_mtp.value:
            with external_mtp.prevent(SearchableSelect.Changed):
                external_mtp.value = ""
        self._apply_form_defaults(reset=True)

    @on(Button.Pressed, "#ds4-reset")
    def reset_pressed(self) -> None:
        model = self.query_one("#ds4-model", SearchableSelect).value
        if not model:
            self.notify("Select a local model first.", severity="error")
            return
        model_name = Path(model).name
        role = self.query_one("#ds4-role", SearchableSelect).value or "Standalone"
        current = self._collect_launch_settings()
        # Apply the defaults to read them, then put the current values back
        # before anything is drawn. The popup shows the difference.
        self._reset_form()
        defaults = self._collect_launch_settings()
        self._apply_launch_values(current, model)
        changes = describe_changes(current, defaults)
        has_saved = bool(load_launch_settings(model_name, role))
        if not changes and not has_saved:
            self.notify("The form already matches the defaults.")
            return
        lines = [f"Reset {model_name} ({role}) to its defaults?", "", "Current -> default"]
        lines += [escape(f"  {line}") for line in changes] or ["  No value on the form changes."]
        lines += [
            "",
            "Defaults come from the model catalog. They are not checked against this machine's memory.",
        ]
        if has_saved:
            lines.append("The saved settings for this model and role are removed.")
        self._pending_reset = (model_name, role)
        self.app.push_screen(
            ConfirmModal("\n".join(lines), yes_text="Reset"), self._reset_confirmed
        )

    def _reset_confirmed(self, confirmed: bool) -> None:
        if not confirmed or self._pending_reset is None:
            return
        model_name, role = self._pending_reset
        self._reset_form()
        if clear_launch_settings(model_name, role):
            self.notify("Reset to the defaults.")
        else:
            self.notify("Reset, but the saved settings could not be removed.", severity="warning")
