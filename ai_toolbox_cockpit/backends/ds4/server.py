"""DS4 server UI, including distributed and SSD-streaming controls."""

import shlex
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Checkbox, Input, Label, Static, TextArea

from ai_toolbox_cockpit.backends.base import BackendServerPanel
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.server_process import run_foreground_server
from ai_toolbox_cockpit.settings import load_default_toolbox
from ai_toolbox_cockpit.widgets import CockpitCheckbox, ConfirmModal, SearchableSelect

from .config import get_artifact_role, get_model_artifact, get_model_server_defaults
from .model_manager import scan_local_models
from .server_runner import build_server_cmd


class Ds4ServerPanel(BackendServerPanel):
    backend_label = "DwarfStar (ds4) Server"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.platform_id = ""
        self._current_model_path = ""
        self._dspark_support_models: list[dict[str, str]] = []
        self._vision_encoders: list[dict[str, str]] = []
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
                    yield Input(value="localhost", placeholder="Host", id="ds4-host")
                with Vertical(classes="compact-field"):
                    yield Label("Port", id="ds4-port-label", classes="field-label")
                    yield Input(value="8000", placeholder="Port", id="ds4-port")
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
            with Horizontal(classes="options-row"):
                yield CockpitCheckbox("Disk KV cache", value=False, id="ds4-kv-enabled")
            with Horizontal(classes="compact-fields"):
                with Vertical(classes="compact-field"):
                    yield Label("KV cache directory", id="ds4-kv-dir-label", classes="field-label")
                    yield Input(value="~/.cache/ds4-kv", disabled=True, id="ds4-kv-dir")
                with Vertical(classes="compact-field"):
                    yield Label("KV cache size (MB)", id="ds4-kv-mb-label", classes="field-label")
                    yield Input(value="8192", placeholder="Size in MB", disabled=True, id="ds4-kv-mb")
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
                        yield Input(value="0", id="ds4-dspark-confidence")
                yield Static("", id="ds4-dspark-note")
            with Horizontal(classes="inline-row"):
                yield Label("MTP model", id="ds4-mtp-label", classes="inline-label")
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
        model_select.value = (
            recommended["path"]
            if recommended
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
        model = self.query_one("#ds4-model", SearchableSelect).value
        model_changed = model != self._current_model_path
        self._current_model_path = model
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
        self.query_one("#ds4-ssd-enabled", CockpitCheckbox).value = streaming
        self.query_one("#ds4-ssd-experts", Input).value = str(defaults.get("ssd_experts", ""))
        self.query_one("#ds4-ssd-layers", Input).value = str(defaults.get("ssd_full_layers", ""))
        self.query_one("#ds4-ssd-cold", CockpitCheckbox).value = bool(defaults.get("ssd_cold", False)) if streaming else False
        self._refresh_mxfp4_controls(model, model_changed)
        self._refresh_vision_control(model, model_changed)
        self._refresh_dspark_controls(defaults, role, model_changed)

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
        available = bool(matches) and role == "Standalone"
        zone.styles.display = "block" if supported else "none"
        support.set_options([(model["name"], model["path"]) for model in matches])
        support.value = matches[0]["path"] if matches else ""
        enabled.disabled = not available
        if model_changed:
            enabled.value = bool(defaults.get("dspark_enabled", False)) and available
            confidence.value = str(defaults.get("dspark_confidence", 0))
        elif not available:
            enabled.value = False
        note = ""
        if supported and not matches:
            note = f"Download {support_filename} to enable DSpark."
        elif supported and role != "Standalone":
            note = "DSpark is available only in standalone mode."
        elif supported:
            note = "Uses the gfx1151-optimized five-proposal path with confidence 0."
        self.query_one("#ds4-dspark-note", Static).update(note)
        self._sync_dspark_controls()

    def _sync_dspark_controls(self) -> None:
        checkbox = self.query_one("#ds4-dspark-enabled", Checkbox)
        active = checkbox.value and not checkbox.disabled
        self.query_one("#ds4-dspark-model", SearchableSelect).disabled = not active
        self.query_one("#ds4-dspark-confidence", Input).disabled = not active
        ssd = self.query_one("#ds4-ssd-enabled", CockpitCheckbox)
        if active:
            ssd.value = False
        ssd.disabled = active

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
            dspark_confidence = self._probability(
                self.query_one("#ds4-dspark-confidence", Input).value.strip(),
                "DSpark confidence",
            ) if dspark_enabled else 0.0
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
            dspark_enabled=dspark_enabled,
            dspark_path=self.query_one("#ds4-dspark-model", SearchableSelect).value,
            dspark_confidence=dspark_confidence,
            vision_path=self.query_one("#ds4-vision", SearchableSelect).value,
        )
        self.app.push_screen(
            ConfirmModal(f"Start DwarfStar (ds4) server?\n\n{shlex.join(self._pending_command)}", yes_text="Start"),
            self._start_confirmed,
        )

    def _start_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        command = self._pending_command
        with self.app.suspend():
            run_foreground_server(command, command[0], "ds4-cockpit-server")
