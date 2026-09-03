"""llama.cpp server UI and launch adapter."""

import os
import shlex

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Checkbox, Input, Label, Static

from ai_toolbox_cockpit.backends.base import BackendServerPanel
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.server_process import redact_command, run_foreground_server
from ai_toolbox_cockpit.settings import load_default_toolbox
from ai_toolbox_cockpit.widgets import CockpitCheckbox, ConfirmModal, SearchableSelect

from .config import (
    get_calibrated_ubatch_defaults,
    get_default_inference_profile,
    get_dspark_config,
    get_effective_mtp_config,
    get_inference_profiles,
    get_model_config,
    get_mtp_server_args,
    get_recommended_server_defaults,
    get_recommended_use,
    get_toolbox_defaults,
    get_vision_projector_config,
    recommended_use_matches_model,
)
from .model_manager import (
    get_local_dspark_models,
    get_local_mtp_models,
    get_local_vision_projectors,
    scan_local_models,
)
from .server_runner import build_server_cmd


KV_TYPES = ("q8_0", "q5_1", "q5_0", "q4_1", "q4_0")
LOAD_MODES = (
    ("Direct I/O (dio)", "dio"),
    ("Resident / no mmap (none)", "none"),
    ("Memory mapped (mmap)", "mmap"),
)


class LlamaCppServerPanel(BackendServerPanel):
    backend_label = "llama.cpp Server"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.platform_id = ""
        self._current_model_config: dict | None = None
        self._pending_command: list[str] = []
        self._expected_extra_args = "--jinja"
        self._active_recommended_toolbox_id = ""

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label(self.backend_label, classes="panel-title")
            yield Static(
                "Launch llama-server directly from a platform image. The selected local GGUF directory is mounted read-only.",
                classes="panel-copy",
            )
            with Horizontal(classes="inline-row"):
                yield Label("Engine", id="llama-engine-label", classes="inline-label")
                yield SearchableSelect("Select Podman or Docker", id="llama-engine")
            with Horizontal(classes="inline-row"):
                yield Label("Image", id="llama-image-label", classes="inline-label")
                yield SearchableSelect("Search platform llama.cpp images", id="llama-image")
            with Vertical(id="llama-toolbox-guidance", classes="model-zone"):
                yield Label("Purpose-built toolbox", classes="zone-title")
                yield Static("", id="llama-toolbox-guidance-message")
            with Horizontal(classes="inline-row"):
                yield Label("Model", id="llama-model-label", classes="inline-label")
                yield SearchableSelect("Search local GGUF models", id="llama-model")
                yield Button("Scan", id="llama-scan-models")

            with Vertical(id="llama-mtp-zone", classes="model-zone"):
                yield Label("MTP speculative decoding", classes="zone-title")
                yield CockpitCheckbox("Enable MTP", id="llama-mtp-enabled", value=True)
                with Horizontal(id="llama-mtp-model-row", classes="inline-row"):
                    yield Label("MTP model", id="llama-mtp-model-label", classes="inline-label")
                    yield SearchableSelect("Select downloaded external MTP model", id="llama-mtp-model")
                yield Static("", id="llama-mtp-note")
                with Horizontal(classes="compact-fields"):
                    with Vertical(classes="compact-field"):
                        yield Label("Draft tokens", id="llama-mtp-draft-label", classes="field-label")
                        yield Input(value="2", id="llama-mtp-draft")
                    with Vertical(classes="compact-field"):
                        yield Label("Parallel sequences", id="llama-mtp-np-label", classes="field-label")
                        yield Input(value="1", id="llama-mtp-np")

            with Vertical(id="llama-dspark-zone", classes="model-zone"):
                yield Label("DSpark speculative decoding", classes="zone-title")
                yield CockpitCheckbox("Enable DSpark", id="llama-dspark-enabled", value=True)
                with Horizontal(classes="inline-row"):
                    yield Label("Drafter model", id="llama-dspark-model-label", classes="inline-label")
                    yield SearchableSelect("Select downloaded DSpark drafter", id="llama-dspark-model")
                with Horizontal(classes="compact-fields"):
                    with Vertical(classes="compact-field"):
                        yield Label("Draft tokens", id="llama-dspark-draft-label", classes="field-label")
                        yield Input(value="3", id="llama-dspark-draft")
                    with Vertical(classes="compact-field"):
                        yield Label("Draft GPU layers", id="llama-dspark-ngl-label", classes="field-label")
                        yield Input(value="99", id="llama-dspark-ngl")
                yield Static("", id="llama-dspark-note")

            with Vertical(id="llama-projector-zone", classes="model-zone"):
                yield Label("Vision projector (optional)", classes="zone-title")
                with Horizontal(classes="inline-row"):
                    yield Label("Projector", id="llama-projector-label", classes="inline-label")
                    yield SearchableSelect("Select downloaded mmproj", id="llama-projector")
                yield Static("", id="llama-projector-note")

            with Vertical(id="llama-profile-zone", classes="model-zone"):
                yield Label("Inference profile", classes="zone-title")
                with Horizontal(classes="inline-row"):
                    yield Label("Profile", id="llama-profile-label", classes="inline-label")
                    yield SearchableSelect("Select a curated profile", id="llama-profile")
                yield Static("", id="llama-profile-note")

            with Horizontal(classes="compact-fields"):
                with Vertical(classes="compact-field"):
                    yield Label("Context", id="llama-context-label", classes="field-label")
                    yield Input(value="126976", id="llama-context")
                with Vertical(classes="compact-field"):
                    yield Label("GPU layers", id="llama-ngl-label", classes="field-label")
                    yield Input(value="999", id="llama-ngl")
                with Vertical(classes="compact-field"):
                    yield Label("Host", id="llama-host-label", classes="field-label")
                    yield Input(value="localhost", id="llama-host")
                with Vertical(classes="compact-field"):
                    yield Label("Port", id="llama-port-label", classes="field-label")
                    yield Input(value="8080", id="llama-port")
            with Horizontal(classes="compact-fields"):
                with Vertical(classes="compact-field"):
                    yield Label("Batch size", id="llama-batch-label", classes="field-label")
                    yield Input(placeholder="llama.cpp default", id="llama-batch")
                with Vertical(classes="compact-field"):
                    yield Label("Ubatch size", id="llama-ubatch-label", classes="field-label")
                    yield Input(placeholder="llama.cpp default", id="llama-ubatch")
                with Vertical(classes="compact-field"):
                    yield Label("Parallel sequences", id="llama-parallel-label", classes="field-label")
                    yield Input(placeholder="llama.cpp default", id="llama-parallel")
            with Horizontal(classes="options-row"):
                yield CockpitCheckbox("Flash Attention", id="llama-fa", value=True)
                yield CockpitCheckbox("No memory mapping", id="llama-no-mmap", value=True)
                yield CockpitCheckbox("Quantize KV cache", id="llama-kv-enabled")
            with Horizontal(id="llama-load-mode-row", classes="inline-row"):
                yield Label("Load mode", id="llama-load-mode-label", classes="inline-label")
                yield SearchableSelect("Select model load mode", id="llama-load-mode")
            with Horizontal(id="llama-kv-row", classes="inline-row"):
                yield Label("KV cache", id="llama-kv-type-label", classes="inline-label")
                yield SearchableSelect("Select KV type", id="llama-kv-type")
            with Horizontal(classes="inline-row"):
                yield Label("GPU devices", id="llama-devices-label", classes="inline-label")
                yield Input(placeholder="HIP or Level Zero device list", id="llama-devices")
            with Horizontal(classes="inline-row"):
                yield Label("API key", id="llama-api-key-label", classes="inline-label")
                yield Input(placeholder="Optional llama-server API key", password=True, id="llama-api-key")
            with Horizontal(classes="inline-row"):
                yield Label("Extra args", id="llama-extra-args-label", classes="inline-label")
                yield Input(value="--jinja", id="llama-extra-args")
            with Horizontal(classes="action-row"):
                yield Button("Start Server", id="llama-start", variant="primary")

    def on_mount(self) -> None:
        self.platform_id = self.app.active_platform_id
        engine = self.query_one("#llama-engine", SearchableSelect)
        options = [(item.value, item.value) for item in detect_container_engines()]
        engine.set_options(options)
        if options:
            engine.value = options[0][1]
        kv = self.query_one("#llama-kv-type", SearchableSelect)
        kv.set_options([(value, value) for value in KV_TYPES])
        kv.value = KV_TYPES[0]
        load_mode = self.query_one("#llama-load-mode", SearchableSelect)
        load_mode.set_options(list(LOAD_MODES))
        load_mode.value = "none"
        self.query_one("#llama-kv-row", Horizontal).styles.display = "none"
        self.query_one("#llama-load-mode-row", Horizontal).styles.display = "none"
        self.query_one("#llama-toolbox-guidance", Vertical).styles.display = "none"
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
            if toolbox.backend == "llama_cpp" and toolbox.feature_state("server") != "unavailable"
        ]
        select = self.query_one("#llama-image", SearchableSelect)
        select.set_options([
            (
                f"{toolbox.name}{' [experimental]' if toolbox.feature_state('server') == 'experimental' else ''} — {toolbox.image}",
                toolbox.id,
            )
            for toolbox in toolboxes
        ])
        default = load_default_toolbox(
            "llama_cpp", platform_id,
            self.app.toolbox_catalog.platform(platform_id).defaults.get("llama_cpp", ""),
        )
        select.value = default if default in {toolbox.id for toolbox in toolboxes} else (toolboxes[0].id if toolboxes else "")

    def refresh_models(self) -> None:
        models = scan_local_models()
        select = self.query_one("#llama-model", SearchableSelect)
        previous = select.value
        select.set_options([(model["name"], model["path"]) for model in models])
        paths = {model["path"] for model in models}
        select.value = previous if previous in paths else (models[0]["path"] if models else "")

    def _selected_toolbox(self):
        toolbox_id = self.query_one("#llama-image", SearchableSelect).value
        return self.app.toolbox_catalog.toolboxes.get(toolbox_id)

    def _effective_mtp_config(self, model_config: dict | None = None) -> dict | None:
        return get_effective_mtp_config(
            self._current_model_config if model_config is None else model_config,
            self._selected_toolbox(),
        )

    def _select_recommended_model_for_toolbox(self) -> bool:
        recommended = get_recommended_use(self._selected_toolbox())
        if not recommended:
            return False
        for model in scan_local_models():
            config = get_model_config(model["path"])
            if recommended_use_matches_model(
                recommended, config, model["path"], require_filename=True
            ):
                select = self.query_one("#llama-model", SearchableSelect)
                if select.value != model["path"]:
                    select.value = model["path"]
                    return True
                return False
        return False

    def _toolbox_pairing_warning(
        self, model_config: dict | None, model_path: str
    ) -> str:
        recommended = get_recommended_use(self._selected_toolbox())
        if not recommended:
            return ""
        if recommended.get("platform_id") != self.platform_id:
            return "This toolbox is supported only on AMD Strix Halo / gfx1151."
        if not recommended_use_matches_model(recommended, model_config):
            return (
                "This fork is validated for "
                f"{recommended['model_display_name']}, not the selected model."
            )
        if not recommended_use_matches_model(
            recommended, model_config, model_path, require_filename=True
        ):
            return (
                "This model family matches, but the tested quant is "
                f"{recommended['model_display_name']}."
            )
        return ""

    def _refresh_toolbox_guidance(self) -> None:
        zone = self.query_one("#llama-toolbox-guidance", Vertical)
        message = self.query_one("#llama-toolbox-guidance-message", Static)
        recommended = get_recommended_use(self._selected_toolbox())
        if not recommended:
            zone.styles.display = "none"
            message.update("")
            return
        zone.styles.display = "block"
        model_path = self.query_one("#llama-model", SearchableSelect).value
        warning = self._toolbox_pairing_warning(
            get_model_config(model_path), model_path
        )
        status = (
            f"WARNING: {warning}"
            if warning
            else "Recommended model and quant selected."
        )
        sidecar_status = ""
        sidecar = recommended.get("sidecar")
        if sidecar:
            sidecar_found = bool(
                get_local_mtp_models([sidecar["filename"]], sidecar["repo"])
            )
            sidecar_status = (
                f" MTP sidecar ready: {sidecar['filename']}."
                if sidecar_found
                else (
                    " MTP sidecar missing. Download "
                    f"{sidecar['repo']} / {sidecar['filename']} from Models."
                )
            )
        message.update(
            "\n".join([
                recommended["message"],
                f"{status}{sidecar_status}",
                *(f"Note: {note}" for note in recommended.get("notes", [])),
                f"Guide: {recommended['documentation_url']}",
            ])
        )

    def refresh_model_inventory(self) -> None:
        self.refresh_models()
        self._refresh_mtp_model_options(self._effective_mtp_config())
        self._refresh_toolbox_guidance()

    @on(Button.Pressed, "#llama-scan-models")
    def scan_pressed(self) -> None:
        self.refresh_models()
        self._refresh_mtp_model_options(self._effective_mtp_config())
        self._refresh_toolbox_guidance()
        self.notify("Local GGUF directory scanned.")

    def _refresh_mtp_controls(self, mtp: dict | None) -> None:
        mtp_zone = self.query_one("#llama-mtp-zone", Vertical)
        mtp_zone.styles.display = "block" if mtp else "none"
        if mtp:
            self.query_one("#llama-mtp-enabled", Checkbox).value = True
            self.query_one("#llama-mtp-draft", Input).value = str(
                mtp.get("default_draft_n", 2)
            )
            self.query_one("#llama-mtp-np", Input).value = str(
                mtp.get("default_np", 1)
            )
        else:
            self.query_one("#llama-mtp-enabled", Checkbox).value = False
        self._refresh_mtp_model_options(mtp)

    def _refresh_mtp_model_options(self, mtp: dict | None) -> None:
        mtp_model_row = self.query_one("#llama-mtp-model-row", Horizontal)
        mtp_select = self.query_one("#llama-mtp-model", SearchableSelect)
        external_mtp_filenames = list(mtp.get("draft_models", [])) if mtp else []
        mtp_model_row.styles.display = "block" if external_mtp_filenames else "none"
        if external_mtp_filenames:
            matches = get_local_mtp_models(
                external_mtp_filenames, str(mtp.get("sidecar_repo", ""))
            )
            self.query_one("#llama-mtp-enabled", Checkbox).value = bool(matches)
            mtp_select.set_options([(item.name, str(item)) for item in matches])
            mtp_select.value = str(matches[0]) if matches else ""
            self.query_one("#llama-mtp-note", Static).update(
                (
                    f"Using sidecar from {mtp['sidecar_repo']}."
                    if mtp.get("sidecar_repo") and matches
                    else "Select the external MTP GGUF."
                )
                if matches
                else (
                    f"Download the MTP sidecar from {mtp['sidecar_repo']} first."
                    if mtp.get("sidecar_repo")
                    else "No supported external MTP GGUF found under the models directory."
                )
            )
        else:
            mtp_select.set_options([])
            mtp_select.value = ""
            self.query_one("#llama-mtp-note", Static).update("")

    @on(SearchableSelect.Changed, "#llama-model")
    def model_changed(self, event: SearchableSelect.Changed) -> None:
        path = str(event.value or "")
        config = get_model_config(path)
        self._current_model_config = config
        base = "--no-jinja" if config and config.get("no_jinja") else "--jinja"
        self._expected_extra_args = base
        self.query_one("#llama-extra-args", Input).value = base

        mtp = self._effective_mtp_config(config)
        self._refresh_mtp_controls(mtp)

        dspark = get_dspark_config(config)
        dspark_zone = self.query_one("#llama-dspark-zone", Vertical)
        dspark_select = self.query_one("#llama-dspark-model", SearchableSelect)
        dspark_zone.styles.display = "block" if dspark else "none"
        if dspark:
            matches = get_local_dspark_models(
                dspark["patterns"], dspark.get("default_pattern", "")
            )
            dspark_select.set_options([(item.name, str(item)) for item in matches])
            dspark_select.value = str(matches[0]) if matches else ""
            self.query_one("#llama-dspark-draft", Input).value = str(
                dspark.get("default_draft_n", 3)
            )
            self.query_one("#llama-dspark-ngl", Input).value = str(dspark.get("default_ngl", 99))
            self.query_one("#llama-dspark-enabled", Checkbox).value = bool(matches)
            self.query_one("#llama-dspark-note", Static).update(
                "Using the official Unsloth 0731 drafter."
                if matches
                else "Download the official Unsloth 0731 Q8_0 or BF16 DSpark drafter first."
            )
        else:
            dspark_select.set_options([])
            dspark_select.value = ""
            self.query_one("#llama-dspark-enabled", Checkbox).value = False

        projector = get_vision_projector_config(config)
        projector_zone = self.query_one("#llama-projector-zone", Vertical)
        projector_select = self.query_one("#llama-projector", SearchableSelect)
        projector_zone.styles.display = "block" if projector else "none"
        if projector:
            matches = get_local_vision_projectors(path, projector["patterns"])
            projector_select.set_options([("None (text only)", "")] + [(item.name, str(item)) for item in matches])
            projector_select.value = ""
            self.query_one("#llama-projector-note", Static).update(
                "Select a projector to add --mmproj." if matches else "No matching projector is downloaded beside this model."
            )

        profiles = get_inference_profiles(config)
        profile_zone = self.query_one("#llama-profile-zone", Vertical)
        profile_select = self.query_one("#llama-profile", SearchableSelect)
        profile_zone.styles.display = "block" if profiles else "none"
        if profiles:
            names = list(profiles)
            profile_select.set_options([(name, name) for name in names] + [("Default (empty)", "Default (empty)"), ("Custom", "Custom")])
            profile_select.value = get_default_inference_profile(config) or names[0]
        self._apply_toolbox_defaults()
        self._rebuild_extra_args()
        self._refresh_toolbox_guidance()

    @on(SearchableSelect.Changed, "#llama-image")
    def image_changed(self) -> None:
        self._select_recommended_model_for_toolbox()
        model_path = self.query_one("#llama-model", SearchableSelect).value
        self._current_model_config = get_model_config(model_path)
        toolbox = self._selected_toolbox()
        recommended = get_recommended_use(toolbox)
        recommended_is_active = recommended_use_matches_model(
            recommended, self._current_model_config
        )
        toolbox_id = toolbox.id if toolbox else ""
        reset_mtp_defaults = bool(
            recommended_is_active or self._active_recommended_toolbox_id
        )
        self._active_recommended_toolbox_id = (
            toolbox_id if recommended_is_active else ""
        )
        if reset_mtp_defaults:
            self._refresh_mtp_controls(self._effective_mtp_config())
        self._apply_toolbox_defaults()
        self._rebuild_extra_args()
        self._refresh_toolbox_guidance()

    def _apply_toolbox_defaults(self) -> None:
        if not self.is_mounted:
            return
        toolbox_id = self.query_one("#llama-image", SearchableSelect).value
        toolbox = self.app.toolbox_catalog.toolboxes.get(toolbox_id)
        defaults = get_toolbox_defaults(self._current_model_config, toolbox_id)
        defaults.update(
            get_recommended_server_defaults(toolbox, self._current_model_config)
        )
        self.query_one("#llama-context", Input).value = str(
            defaults.get("context_size", 126976)
        )
        for control_id, key in (
            ("#llama-batch", "batch_size"),
            ("#llama-ubatch", "ubatch_size"),
            ("#llama-parallel", "parallel_sequences"),
        ):
            value = defaults.get(key)
            self.query_one(control_id, Input).value = str(value) if value is not None else ""
        gpu_layers = defaults.get("gpu_layers", 999)
        self.query_one("#llama-ngl", Input).value = (
            str(gpu_layers) if gpu_layers is not None else ""
        )
        kv_cache_type = str(defaults.get("kv_cache_type", ""))
        self.query_one("#llama-kv-enabled", Checkbox).value = bool(kv_cache_type)
        self.query_one("#llama-kv-type", SearchableSelect).value = kv_cache_type or KV_TYPES[0]
        self.query_one("#llama-fa", Checkbox).value = bool(
            defaults.get("flash_attention", True)
        )
        supports_load_mode = bool(toolbox and toolbox.supports_load_mode)
        self.query_one("#llama-load-mode-row", Horizontal).styles.display = (
            "block" if supports_load_mode else "none"
        )
        self.query_one("#llama-no-mmap", Checkbox).styles.display = (
            "none" if supports_load_mode else "block"
        )
        self.query_one("#llama-load-mode", SearchableSelect).value = str(
            defaults.get("load_mode", "none")
        )
        self._apply_calibrated_ubatch_defaults()

    def _current_serving_config(self, kv_cache_type: str) -> str:
        dspark = get_dspark_config(self._current_model_config)
        if dspark and self.query_one("#llama-dspark-enabled", Checkbox).value:
            if kv_cache_type == "q8_0":
                return "dspark-vulkan-kv-q8"
            if kv_cache_type == "q4_0":
                return "dspark-vulkan-kv-q4"
            return "dspark"
        mtp = self._effective_mtp_config()
        if mtp and self.query_one("#llama-mtp-enabled", Checkbox).value:
            draft = self.query_one("#llama-mtp-draft", Input).value.strip()
            return f"mtp-{draft}" if draft else "mtp"
        return "baseline"

    def _apply_calibrated_ubatch_defaults(self) -> None:
        """Apply only a calibration matching model file and the active launch recipe."""
        if not self.is_mounted:
            return
        toolbox_id = self.query_one("#llama-image", SearchableSelect).value
        model_path = self.query_one("#llama-model", SearchableSelect).value
        toolbox = self.app.toolbox_catalog.toolboxes.get(toolbox_id)
        base_defaults = get_toolbox_defaults(self._current_model_config, toolbox_id)
        base_defaults.update(
            get_recommended_server_defaults(toolbox, self._current_model_config)
        )
        kv_cache_type = (
            self.query_one("#llama-kv-type", SearchableSelect).value
            if self.query_one("#llama-kv-enabled", Checkbox).value
            else "default"
        )
        calibrated = get_calibrated_ubatch_defaults(
            self._current_model_config,
            model_path,
            toolbox_id,
            self._current_serving_config(kv_cache_type),
            kv_cache_type,
        )
        for control_id, key in (
            ("#llama-batch", "batch_size"),
            ("#llama-ubatch", "ubatch_size"),
        ):
            value = calibrated.get(key, base_defaults.get(key))
            self.query_one(control_id, Input).value = str(value) if value is not None else ""

    @on(SearchableSelect.Changed, "#llama-profile")
    @on(Checkbox.Changed, "#llama-mtp-enabled")
    @on(Input.Changed, "#llama-mtp-draft")
    @on(Input.Changed, "#llama-mtp-np")
    @on(Checkbox.Changed, "#llama-dspark-enabled")
    @on(Input.Changed, "#llama-dspark-draft")
    @on(Input.Changed, "#llama-dspark-ngl")
    def profile_inputs_changed(self) -> None:
        self._apply_calibrated_ubatch_defaults()
        self._rebuild_extra_args()

    def _rebuild_extra_args(self) -> None:
        if not self.is_mounted:
            return
        config = self._current_model_config
        base = "--no-jinja" if config and config.get("no_jinja") else "--jinja"
        profile_name = self.query_one("#llama-profile", SearchableSelect).value
        profiles = get_inference_profiles(config)
        toolbox = self._selected_toolbox()
        recommended_defaults = get_recommended_server_defaults(toolbox, config)
        args = " ".join(
            part for part in (base, recommended_defaults.get("extra_args", "")) if part
        )
        note = ""
        if profile_name in profiles:
            profile = profiles[profile_name]
            args = " ".join(
                part
                for part in (
                    base,
                    profile.get("args", ""),
                    recommended_defaults.get("extra_args", ""),
                )
                if part
            )
            note = profile.get("description", "")
        elif profile_name == "Default (empty)":
            note = "No curated sampling parameters."
        elif profile_name == "Custom":
            return
        mtp = self._effective_mtp_config(config)
        if mtp and self.query_one("#llama-mtp-enabled", Checkbox).value:
            draft = self.query_one("#llama-mtp-draft", Input).value or "2"
            sequences = self.query_one("#llama-mtp-np", Input).value or "1"
            args += f" {get_mtp_server_args(mtp, draft, sequences)}"
        dspark = get_dspark_config(config)
        if dspark and self.query_one("#llama-dspark-enabled", Checkbox).value:
            draft = self.query_one("#llama-dspark-draft", Input).value or "3"
            draft_ngl = self.query_one("#llama-dspark-ngl", Input).value or "99"
            fit = dspark.get("fit", "off")
            args += (
                f" --spec-type draft-dspark --spec-draft-n-max {draft}"
                f" --fit {fit} -ngld {draft_ngl}"
            )
        self._expected_extra_args = args
        self.query_one("#llama-extra-args", Input).value = args
        self.query_one("#llama-profile-note", Static).update(note)

    @on(Input.Changed, "#llama-extra-args")
    def extra_args_changed(self, event: Input.Changed) -> None:
        if event.value != self.query_one("#llama-extra-args", Input).value:
            return
        if event.value == self._expected_extra_args:
            return
        profile = self.query_one("#llama-profile", SearchableSelect)
        if get_inference_profiles(self._current_model_config) and profile.value != "Custom":
            profile.value = "Custom"

    @on(Checkbox.Changed, "#llama-kv-enabled")
    def kv_changed(self, event: Checkbox.Changed) -> None:
        self.query_one("#llama-kv-row", Horizontal).styles.display = "block" if event.value else "none"
        self._apply_calibrated_ubatch_defaults()

    @on(SearchableSelect.Changed, "#llama-kv-type")
    def kv_type_changed(self) -> None:
        self._apply_calibrated_ubatch_defaults()

    @on(Button.Pressed, "#llama-start")
    def start_pressed(self) -> None:
        engine = self.query_one("#llama-engine", SearchableSelect).value
        toolbox_id = self.query_one("#llama-image", SearchableSelect).value
        model = self.query_one("#llama-model", SearchableSelect).value
        context = self.query_one("#llama-context", Input).value
        if not engine or toolbox_id not in self.app.toolbox_catalog.toolboxes or not model:
            self.notify("Select an engine, platform image, and local model.", severity="error")
            return
        if not context.isdigit() or int(context) <= 0:
            self.notify("Context must be a positive integer.", severity="error")
            return
        optional_values: dict[str, int | None] = {}
        for key, control_id, label in (
            ("batch_size", "#llama-batch", "Batch size"),
            ("ubatch_size", "#llama-ubatch", "Ubatch size"),
            ("parallel_sequences", "#llama-parallel", "Parallel sequences"),
        ):
            value = self.query_one(control_id, Input).value.strip()
            if value and (not value.isdigit() or int(value) <= 0):
                self.notify(f"{label} must be a positive integer or empty.", severity="error")
                return
            optional_values[key] = int(value) if value else None
        toolbox = self.app.toolbox_catalog.toolboxes[toolbox_id]
        config = get_model_config(model)
        if config and config.get("compatible_toolboxes"):
            if not any(value.lower() in toolbox.image.lower() for value in config["compatible_toolboxes"]):
                self.notify("The selected model is not compatible with that image.", severity="error")
                return
        projector = self.query_one("#llama-projector", SearchableSelect).value
        if projector and not os.path.isfile(projector):
            self.notify("The selected vision projector is no longer available.", severity="error")
            return
        dspark = get_dspark_config(config)
        draft_model = ""
        if dspark and self.query_one("#llama-dspark-enabled", Checkbox).value:
            draft_model = self.query_one("#llama-dspark-model", SearchableSelect).value
            if not draft_model or not os.path.isfile(draft_model):
                self.notify("Select a downloaded DSpark drafter.", severity="error")
                return
        mtp_draft_model = ""
        mtp = get_effective_mtp_config(config, toolbox)
        if (
            mtp
            and mtp.get("draft_models")
            and self.query_one("#llama-mtp-enabled", Checkbox).value
        ):
            mtp_draft_model = self.query_one("#llama-mtp-model", SearchableSelect).value
            if not mtp_draft_model or not os.path.isfile(mtp_draft_model):
                self.notify(
                    "Select a downloaded external MTP model.",
                    severity="error",
                )
                return
        profile = self.app.toolbox_catalog.runtime_profiles[toolbox.runtime_profile]
        kv_type = self.query_one("#llama-kv-type", SearchableSelect).value if self.query_one("#llama-kv-enabled", Checkbox).value else ""
        ngl = self.query_one("#llama-ngl", Input).value.strip()
        if ngl and (not ngl.isdigit() or int(ngl) < 0):
            self.notify(
                "GPU layers must be a non-negative integer or empty.",
                severity="error",
            )
            return
        command = build_server_cmd(
            engine=engine,
            image=toolbox.image,
            model_path=model,
            context_size=int(context),
            use_fa=self.query_one("#llama-fa", Checkbox).value,
            use_no_mmap=self.query_one("#llama-no-mmap", Checkbox).value,
            custom_args=self.query_one("#llama-extra-args", Input).value,
            host=self.query_one("#llama-host", Input).value,
            port=self.query_one("#llama-port", Input).value,
            ngl=int(ngl) if ngl else None,
            hip_devices=self.query_one("#llama-devices", Input).value,
            platform_id=self.platform_id,
            engine_args=list(profile.engine_args),
            kv_cache_type=kv_type,
            supports_load_mode=toolbox.supports_load_mode,
            api_key=self.query_one("#llama-api-key", Input).value,
            vision_projector_path=projector,
            draft_model_path=draft_model,
            mtp_draft_model_path=mtp_draft_model,
            batch_size=optional_values["batch_size"],
            ubatch_size=optional_values["ubatch_size"],
            parallel_sequences=optional_values["parallel_sequences"],
            load_mode=(
                self.query_one("#llama-load-mode", SearchableSelect).value
                if toolbox.supports_load_mode
                else ""
            ),
        )
        self._pending_command = command
        preview = redact_command(command)
        warning = self._toolbox_pairing_warning(config, model)
        warning_text = f"\n\nWARNING: {warning}" if warning else ""
        self.app.push_screen(
            ConfirmModal(
                f"Start llama.cpp server?{warning_text}\n\n{shlex.join(preview)}",
                yes_text="Start",
            ),
            self._start_confirmed,
        )

    def _start_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        command = self._pending_command
        preview = redact_command(command)
        with self.app.suspend():
            run_foreground_server(
                command,
                command[0],
                "llama-cockpit-server",
                display_command=preview,
            )
