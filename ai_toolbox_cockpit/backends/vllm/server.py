"""vLLM direct-container server UI."""

import shlex
import subprocess
import json
import shutil
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Checkbox, Input, Label, Static, TextArea

from ai_toolbox_cockpit.backends.base import BackendServerPanel
from ai_toolbox_cockpit.huggingface import get_hf_token, save_hf_token
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.server_process import redact_command, run_foreground_server
from ai_toolbox_cockpit.settings import get_backend_settings, load_default_toolbox, save_backend_settings
from ai_toolbox_cockpit.widgets import (
    CockpitCheckbox,
    ConfirmModal,
    HfTokenModal,
    SearchableSelect,
)

from .model_manager import checkpoint_ready, requires_preparation
from .runner import (
    VllmCachePaths,
    apply_toolbox_policy_overrides,
    apply_gpu_profile,
    build_server_cmd,
    build_device_probe_cmd,
    default_cache_paths,
)


ATTENTION_BACKENDS = ("TRITON_ATTN", "ROCM_ATTN", "ROCM_AITER_UNIFIED_ATTN", "R4D")


def validate_compiled_cache_roots(caches: VllmCachePaths) -> tuple[Path, Path, Path]:
    """Return resettable cache roots, rejecting broad or mismatched paths."""
    roots = (
        ("vLLM", caches.vllm, "vllm"),
        ("Triton", caches.triton, "triton"),
        ("AITER", caches.aiter, "aiter"),
    )
    validated: list[Path] = []
    for label, root, marker in roots:
        resolved = root.expanduser().resolve()
        path_parts = {part.lower() for part in resolved.parts}
        if (
            resolved in (Path("/"), Path.home())
            or len(resolved.parts) < 3
            or not any(marker in part for part in path_parts)
        ):
            raise ValueError(
                f"Refusing unsafe {label} cache root: {resolved}. "
                f"The path must contain a '{marker}' directory component."
            )
        validated.append(resolved)
    return tuple(validated)  # type: ignore[return-value]


class VllmServerPanel(BackendServerPanel):
    backend_label = "vLLM Server"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.platform_id = ""
        self._pending_legacy_model = ""
        self._pending_gpu_profile = ""
        self._pending_command: list[str] = []
        self._pending_caches = default_cache_paths()
        self._policy_by_id: dict[str, dict] = {}
        self._hf_token = get_hf_token()
        self._hf_token_prompted = False

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label(self.backend_label, classes="panel-title")
            yield Static(
                "Launch a curated Hugging Face repository with maintained defaults for tensor parallelism, attention, eager mode, environment, and extra flags.",
                classes="panel-copy",
            )
            with Horizontal(classes="inline-row"):
                yield Label("Container engine", id="vllm-engine-label", classes="inline-label")
                yield SearchableSelect("Select Podman or Docker", id="vllm-engine")
            with Horizontal(classes="inline-row"):
                yield Label("Toolbox image", id="vllm-image-label", classes="inline-label")
                yield SearchableSelect("Search vLLM images", id="vllm-image")
            with Horizontal(classes="inline-row", id="vllm-gpu-profile-row"):
                yield Label("GPU build", classes="inline-label")
                yield SearchableSelect("Select card setup", id="vllm-gpu-profile")
                yield Button("Download / Update build", id="vllm-pull-build")
            yield Static("", id="vllm-gpu-profile-note", classes="panel-copy")
            with Horizontal(classes="inline-row"):
                yield Label("Curated model", id="vllm-model-label", classes="inline-label")
                yield SearchableSelect("Search maintained model defaults", id="vllm-model")
            with Horizontal(classes="inline-row"):
                yield Label("Custom HF repo", id="vllm-custom-model-label", classes="inline-label")
                yield Input(placeholder="Optional owner/model; uses generic defaults", id="vllm-custom-model")

            with Horizontal(classes="inline-row"):
                yield Label("Local model directory", id="vllm-local-model-label", classes="inline-label")
                yield Input(placeholder="Optional prepared snapshot; mounted read-only", id="vllm-local-model")

            with Horizontal(classes="inline-row"):
                yield Label("Speculative decoding", id="vllm-speculation-label", classes="inline-label")
                yield SearchableSelect("Select decoding mode", id="vllm-speculation")
            with Horizontal(classes="inline-row"):
                yield Label("Draft model directory", id="vllm-draft-label", classes="inline-label")
                yield Input(placeholder="Local DFlash2 snapshot; mounted read-only", id="vllm-draft")

            with Horizontal(classes="inline-row"):
                yield Label("Server allocation", id="vllm-allocation-label", classes="inline-label")
                yield SearchableSelect("Everyday or measured allocation", id="vllm-allocation")
            yield Static("", id="vllm-allocation-note", classes="panel-copy")
            with Horizontal(classes="inline-row"):
                yield Label("GPU indices", id="vllm-devices-label", classes="inline-label")
                yield Input(placeholder="Indices from device discovery", id="vllm-devices")
                yield Button("Discover GPUs", id="vllm-discover-devices")
            yield Static("", id="vllm-device-inventory", classes="panel-copy")
            with Vertical(classes="server-settings"):
                yield Label("Runtime limits", classes="settings-title")
                with Horizontal(classes="compact-fields"):
                    with Vertical(classes="compact-field"):
                        yield Label("Tensor parallel", id="vllm-tp-label", classes="field-label")
                        yield SearchableSelect("Select TP size", id="vllm-tp")
                    with Vertical(classes="compact-field"):
                        yield Label("Max sequences", id="vllm-seqs-label", classes="field-label")
                        yield Input(value="1", placeholder="Concurrent sequences", id="vllm-seqs")
                    with Vertical(classes="compact-field"):
                        yield Label("Context length", id="vllm-context-label", classes="field-label")
                        yield Input(value="auto", placeholder="Model default", id="vllm-context")
                    with Vertical(classes="compact-field"):
                        yield Label("GPU memory", id="vllm-util-label", classes="field-label")
                        yield Input(value="0.90", placeholder="Utilization 0-1", id="vllm-util")

            with Vertical(classes="server-settings"):
                yield Label("Network and execution", classes="settings-title")
                with Horizontal(classes="compact-fields"):
                    with Vertical(classes="compact-field"):
                        yield Label("Host", id="vllm-host-label", classes="field-label")
                        yield Input(value="localhost", placeholder="Bind host", id="vllm-host")
                    with Vertical(classes="compact-field"):
                        yield Label("Port", id="vllm-port-label", classes="field-label")
                        yield Input(value="8000", placeholder="API port", id="vllm-port")
                    with Vertical(classes="compact-field"):
                        yield Label("Data type", id="vllm-dtype-label", classes="field-label")
                        yield Input(value="auto", placeholder="auto, bf16, fp16", id="vllm-dtype")
                    with Vertical(classes="compact-field"):
                        yield Label("Attention backend", id="vllm-attention-label", classes="field-label")
                        yield SearchableSelect("Select attention backend", id="vllm-attention")
                with Horizontal(classes="options-row"):
                    yield CockpitCheckbox("Force eager mode", value=False, id="vllm-eager")

            with Vertical(classes="server-settings"):
                yield Label("Persistent cache paths", classes="settings-title")
                with Horizontal(classes="compact-fields"):
                    with Vertical(classes="compact-field"):
                        yield Label("Hugging Face cache", id="vllm-hf-cache-label", classes="field-label")
                        yield Input(id="vllm-hf-cache")
                    with Vertical(classes="compact-field"):
                        yield Label("vLLM cache", id="vllm-compile-cache-label", classes="field-label")
                        yield Input(id="vllm-compile-cache")
                with Horizontal(classes="compact-fields"):
                    with Vertical(classes="compact-field"):
                        yield Label("Triton cache", id="vllm-triton-cache-label", classes="field-label")
                        yield Input(id="vllm-triton-cache")
                    with Vertical(classes="compact-field"):
                        yield Label("AITER cache", id="vllm-aiter-cache-label", classes="field-label")
                        yield Input(id="vllm-aiter-cache")
                with Horizontal(classes="action-row"):
                    yield Button("Save Cache Paths", id="vllm-save-caches")
                    yield CockpitCheckbox(
                        "Reset compiled caches before launch",
                        value=False,
                        id="vllm-reset-caches",
                    )
            with Horizontal(classes="inline-row", id="vllm-offload-row"):
                yield Label("NVMe PLE/cache directory", id="vllm-offload-cache-label", classes="inline-label")
                yield Input(placeholder="tcclaviger persistent NVMe storage", id="vllm-offload-cache")
            with Horizontal(classes="inline-row"):
                yield Label("API key", id="vllm-api-key-label", classes="inline-label")
                yield Input(placeholder="Optional OpenAI-compatible API key", password=True, id="vllm-api-key")
            with Horizontal(classes="extra-args-row"):
                yield Label("Extra args", id="vllm-extra-args-label", classes="inline-label")
                yield TextArea(
                    soft_wrap=True,
                    compact=True,
                    highlight_cursor_line=False,
                    placeholder="Additional vllm serve flags",
                    id="vllm-extra-args",
                )
            with Horizontal(classes="action-row"):
                yield Button("Start vLLM Server", id="vllm-start", variant="primary")

    def on_mount(self) -> None:
        self.platform_id = self.app.active_platform_id
        engines = [(engine.value, engine.value) for engine in detect_container_engines()]
        engine_select = self.query_one("#vllm-engine", SearchableSelect)
        engine_select.set_options(engines)
        if engines:
            engine_select.value = engines[0][1]
        attention = self.query_one("#vllm-attention", SearchableSelect)
        attention.set_options([(value, value) for value in ATTENTION_BACKENDS])
        cache_defaults = default_cache_paths()
        settings = get_backend_settings("vllm")
        for field, key, fallback in (
            ("#vllm-hf-cache", "hf_cache", cache_defaults.huggingface),
            ("#vllm-compile-cache", "vllm_cache", cache_defaults.vllm),
            ("#vllm-triton-cache", "triton_cache", cache_defaults.triton),
            ("#vllm-aiter-cache", "aiter_cache", cache_defaults.aiter),
            ("#vllm-offload-cache", "offload_cache", cache_defaults.offload),
        ):
            self.query_one(field, Input).value = str(settings.get(key, fallback))
        entries = [entry for entry in self.app.model_catalog.backends["vllm"].entries if entry.get("artifact_role") != "draft"]
        self._policy_by_id = {str(entry["id"]): dict(entry) for entry in entries}
        model = self.query_one("#vllm-model", SearchableSelect)
        model.set_options([(f"{entry.get('name', entry['repo'])} — {entry['repo']}", entry["id"]) for entry in entries])
        if entries:
            model.value = str(entries[0]["id"])
        self.refresh_platform(self.platform_id)

    def set_platform(self, platform_id: str) -> None:
        self.platform_id = platform_id
        if self.is_mounted:
            self.refresh_platform(platform_id)

    def refresh_platform(self, platform_id: str) -> None:
        toolboxes = [
            toolbox
            for toolbox in self.app.toolbox_catalog.platform_toolboxes(platform_id)
            if toolbox.backend == "vllm" and toolbox.feature_state("server") != "unavailable"
        ]
        select = self.query_one("#vllm-image", SearchableSelect)
        select.set_options([
            (
                f"{toolbox.name}{' [experimental]' if toolbox.feature_state('server') == 'experimental' else ''} — {toolbox.image}",
                toolbox.id,
            )
            for toolbox in toolboxes
        ])
        default = load_default_toolbox(
            "vllm", platform_id,
            self.app.toolbox_catalog.platform(platform_id).defaults.get("vllm", ""),
        )
        resolved = self.app.toolbox_catalog.resolve_toolbox_id(default)
        selected = self.app.toolbox_catalog.toolboxes.get(resolved)
        self._pending_legacy_model = (selected.backend_config or {}).get("legacy_default_models", {}).get(default, "") if selected else ""
        self._pending_gpu_profile = default if selected and default in (selected.backend_config or {}).get("gpu_profiles", {}) else ""
        default = resolved
        select.value = default if default in {toolbox.id for toolbox in toolboxes} else (toolboxes[0].id if toolboxes else "")

    @on(SearchableSelect.Changed, "#vllm-model")
    def model_changed(self, event: SearchableSelect.Changed) -> None:
        self._apply_model_policy(str(event.value))

    @on(SearchableSelect.Changed, "#vllm-image")
    def image_changed(self) -> None:
        toolbox = self.app.toolbox_catalog.toolboxes.get(str(self.query_one("#vllm-image", SearchableSelect).value))
        profiles = (toolbox.backend_config or {}).get("gpu_profiles", {}) if toolbox else {}
        gpu_select = self.query_one("#vllm-gpu-profile", SearchableSelect)
        self.query_one("#vllm-gpu-profile-row").display = bool(profiles)
        previous_gpu = self._pending_gpu_profile or str(gpu_select.value)
        self._pending_gpu_profile = ""
        gpu_select.set_options([(profile["name"], key) for key, profile in profiles.items()])
        selected_gpu = previous_gpu if previous_gpu in profiles else next(iter(profiles), "")
        if gpu_select.value != selected_gpu:
            gpu_select.value = selected_gpu
        self._update_gpu_profile_note()
        supported = (toolbox.backend_config or {}).get("supported_model_ids") if toolbox else None
        entries = [entry for entry in self._policy_by_id.values()
                   if (not supported or entry["id"] in supported)
                   and (not entry.get("toolbox_ids") or (toolbox and toolbox.id in entry["toolbox_ids"]))]
        model = self.query_one("#vllm-model", SearchableSelect)
        previous = self._pending_legacy_model or str(model.value)
        self._pending_legacy_model = ""
        model.set_options([(f"{entry.get('name', entry['repo'])} — {entry['repo']}", entry["id"]) for entry in entries])
        model.value = previous if previous in {entry["id"] for entry in entries} else (entries[0]["id"] if entries else "")
        self._apply_model_policy(str(model.value))

    @on(SearchableSelect.Changed, "#vllm-allocation")
    def allocation_changed(self) -> None:
        toolbox = self.app.toolbox_catalog.toolboxes.get(str(self.query_one("#vllm-image", SearchableSelect).value))
        profiles = toolbox.backend_config.get("performance_profiles", {}) if toolbox else {}
        profile = profiles.get(str(self.query_one("#vllm-allocation", SearchableSelect).value), {})
        policy = self._effective_policy(self._policy_by_id.get(str(self.query_one("#vllm-model", SearchableSelect).value), {}))
        self.query_one("#vllm-seqs", Input).value = str(profile.get("server_defaults", {}).get("max_num_seqs", policy.get("default_max_num_seqs", 1)))
        self.query_one("#vllm-allocation-note", Static).update(profile.get("note", policy.get("profile_note", "One sequence for everyday serving.")))
        if profile:
            self.query_one("#vllm-speculation", SearchableSelect).value = "baseline"

    @on(Button.Pressed, "#vllm-discover-devices")
    def discover_devices(self) -> None:
        try:
            engine = str(self.query_one("#vllm-engine", SearchableSelect).value)
            toolbox = self.app.toolbox_catalog.toolboxes.get(str(self.query_one("#vllm-image", SearchableSelect).value))
            if not toolbox:
                raise ValueError("Select an installed image")
            arguments = list(self.app.toolbox_catalog.runtime_profiles[toolbox.runtime_profile].engine_args)
            toolbox = self._selected_toolbox()
            command = build_device_probe_cmd(engine, toolbox.image, arguments)
            with self.app.suspend():
                output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
            records = json.loads(output.splitlines()[-1])
            self.query_one("#vllm-device-inventory", Static).update("\n".join(f"{item['index']}: {item['name']} ({item['architecture']})" for item in records) or "No GPUs discovered.")
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.notify(f"Device discovery failed: {error}", severity="error")

    def _selected_toolbox(self, toolbox_id: str = ""):
        toolbox = self.app.toolbox_catalog.toolboxes.get(
            toolbox_id or str(self.query_one("#vllm-image", SearchableSelect).value)
        )
        if not toolbox:
            return None
        profiles = (toolbox.backend_config or {}).get("gpu_profiles", {})
        selected = str(self.query_one("#vllm-gpu-profile", SearchableSelect).value)
        return apply_gpu_profile(toolbox, selected if selected in profiles else "")

    def _update_gpu_profile_note(self) -> None:
        toolbox = self._selected_toolbox()
        note = self.query_one("#vllm-gpu-profile-note", Static)
        note.display = bool(toolbox and (toolbox.backend_config or {}).get("gpu_profiles"))
        note.update(f"Selected build: {toolbox.image}. Missing images download on first launch; use Download / Update build to refresh this channel." if toolbox else "")

    @on(SearchableSelect.Changed, "#vllm-gpu-profile")
    def gpu_profile_changed(self) -> None:
        self._update_gpu_profile_note()
        self._apply_model_policy(str(self.query_one("#vllm-model", SearchableSelect).value))
        # Do not carry a one-card device override into the dual-card profile.
        self.query_one("#vllm-devices", Input).value = ""

    @on(Button.Pressed, "#vllm-pull-build")
    def pull_build_pressed(self) -> None:
        toolbox = self._selected_toolbox()
        engine = str(self.query_one("#vllm-engine", SearchableSelect).value)
        if not toolbox or engine not in {"podman", "docker"}:
            self.notify("Select a container engine and GPU build.", severity="error")
            return
        command = [engine, "pull", toolbox.image]
        self.app.push_screen(ConfirmModal(f"Download / update this build?\n\n{shlex.join(command)}", yes_text="Download / Update"),
                             lambda confirmed: self._pull_build_confirmed(confirmed, command))

    def _pull_build_confirmed(self, confirmed: bool, command: list[str]) -> None:
        if not confirmed:
            return
        try:
            with self.app.suspend():
                subprocess.run(command, check=True)
        except (OSError, subprocess.SubprocessError) as error:
            self.notify(f"Build download failed: {error}", severity="error")
            return
        self.notify("Selected GPU build is ready.")

    def _effective_policy(self, policy: dict, toolbox_id: str = "") -> dict:
        toolbox = self._selected_toolbox(toolbox_id)
        return apply_toolbox_policy_overrides(policy, toolbox.backend_config if toolbox else None)

    def _apply_model_policy(self, model_id: str) -> None:
        policy = self._effective_policy(self._policy_by_id.get(model_id, {}))
        speculation = self.query_one("#vllm-speculation", SearchableSelect)
        self.query_one("#vllm-offload-row").display = policy.get("runtime_variant") == "tcclaviger"
        default_speculation = policy.get("default_speculation", "baseline")
        options = [] if default_speculation == "mtp" else [("Baseline", "baseline")]
        if "mtp" in policy.get("speculation", {}):
            options.append(("MTP · 3 draft tokens (embedded)", "mtp"))
        if "dflash2" in policy.get("speculation", {}):
            options.append(("DFlash2 · 7 draft tokens", "dflash2"))
        speculation.set_options(options)
        speculation.value = default_speculation
        self.query_one("#vllm-draft", Input).value = ""
        self.query_one("#vllm-draft", Input).disabled = True
        valid_tp = [int(value) for value in policy.get("valid_tp", [1])]
        tp = self.query_one("#vllm-tp", SearchableSelect)
        tp.set_options([(str(value), str(value)) for value in valid_tp])
        tp.value = str(valid_tp[0])
        toolbox = self.app.toolbox_catalog.toolboxes.get(str(self.query_one("#vllm-image", SearchableSelect).value))
        profiles = toolbox.backend_config.get("performance_profiles", {}) if toolbox else {}
        allocation = self.query_one("#vllm-allocation", SearchableSelect)
        allocation.set_options([("Measured · 64 GB / TP2" if default_speculation == "mtp" else "Everyday · one sequence", "everyday")] + [(profile["name"], key) for key, profile in profiles.items()])
        allocation.value = "everyday"
        self.query_one("#vllm-devices", Input).value = str(get_backend_settings("vllm").get("devices", ""))
        self.query_one("#vllm-seqs", Input).value = str(policy.get("default_max_num_seqs", 1))
        self.query_one("#vllm-allocation-note", Static).update(policy.get("profile_note", "One sequence for everyday serving."))
        paths = get_backend_settings("vllm").get("artifact_paths", {}).get(model_id, {})
        local_directory = str(paths.get("prepared" if requires_preparation(policy) else "source", policy.get("local_directory", "")))
        if local_directory and not policy.get("requires_local_model") and not checkpoint_ready(Path(local_directory).expanduser()):
            local_directory = ""
        self.query_one("#vllm-local-model", Input).value = local_directory
        self.query_one("#vllm-util", Input).value = str(policy.get("gpu_memory_utilization", 0.90))
        self.query_one("#vllm-context", Input).value = str(policy.get("ctx", "auto"))
        self.query_one("#vllm-eager", Checkbox).value = bool(policy.get("enforce_eager", False))
        configured_attention = policy.get("attention_backend", "TRITON_ATTN")
        attention = self.query_one("#vllm-attention", SearchableSelect)
        attention_label = self.query_one("#vllm-attention-label", Label)
        if configured_attention is None:
            required_attention = str(
                policy.get("attention_backend_label", "Model-specific implementation")
            ).removesuffix(" (model-specific)")
            attention.set_options([(required_attention, required_attention)])
            attention.value = required_attention
            attention.disabled = True
            attention_label.update("Required attention backend")
        else:
            attention.disabled = False
            attention.set_options([(value, value) for value in ATTENTION_BACKENDS])
            attention.value = str(configured_attention)
            attention_label.update("Attention backend")

    @on(SearchableSelect.Changed, "#vllm-speculation")
    def speculation_changed(self, event: SearchableSelect.Changed) -> None:
        policy = self._effective_policy(self._policy_by_id.get(str(self.query_one("#vllm-model", SearchableSelect).value), {}))
        recipe = policy.get("speculation", {}).get(str(event.value), {})
        draft = self.query_one("#vllm-draft", Input)
        draft.disabled = str(event.value) != "dflash2"
        draft.value = str(recipe.get("local_directory", ""))
        self.query_one("#vllm-util", Input).value = str(recipe.get("gpu_memory_utilization", policy.get("gpu_memory_utilization", 0.90)))
        if recipe and str(event.value) == "dflash2":
            self.query_one("#vllm-allocation", SearchableSelect).value = "everyday"
            self.query_one("#vllm-seqs", Input).value = "1"

    def cache_paths(self) -> VllmCachePaths:
        return VllmCachePaths(*(
            Path(self.query_one(field, Input).value).expanduser().resolve()
            for field in ("#vllm-hf-cache", "#vllm-compile-cache", "#vllm-triton-cache", "#vllm-aiter-cache", "#vllm-offload-cache")
        ))

    @on(Button.Pressed, "#vllm-save-caches")
    def save_caches_pressed(self) -> None:
        try:
            draft_value = self.query_one("#vllm-draft", Input).value.strip()
            draft_directory = Path(draft_value).expanduser().resolve() if draft_value else None
            if draft_directory is not None and not checkpoint_ready(draft_directory, draft=True):
                raise ValueError("Draft model directory must contain the complete DFlash2 checkpoint and config.json")
            caches = self.cache_paths()
            for path in (caches.huggingface, caches.vllm, caches.triton, caches.aiter):
                path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.notify(f"Could not create cache path: {error}", severity="error")
            return
        saved = save_backend_settings("vllm", {
            "hf_cache": str(caches.huggingface),
            "vllm_cache": str(caches.vllm),
            "triton_cache": str(caches.triton),
            "aiter_cache": str(caches.aiter),
            "offload_cache": str(caches.offload),
        })
        self.notify("vLLM cache paths saved." if saved else "Could not save vLLM cache paths.", severity="information" if saved else "error")

    @on(Button.Pressed, "#vllm-start")
    def start_pressed(self) -> None:
        self._prepare_start()

    def _hf_token_received(self, choice: tuple[str, bool] | None) -> None:
        if choice is None:
            return
        token, remember = choice
        self._hf_token = token
        self._hf_token_prompted = True
        if token and remember:
            if save_hf_token(token):
                self.notify("Hugging Face token saved to Cockpit configuration.")
            else:
                self.notify(
                    "Could not save the Hugging Face token; using it for this session.",
                    severity="warning",
                )
        self._prepare_start()

    def _prepare_start(self) -> None:
        engine = self.query_one("#vllm-engine", SearchableSelect).value
        toolbox_id = self.query_one("#vllm-image", SearchableSelect).value
        custom = self.query_one("#vllm-custom-model", Input).value.strip()
        policy = dict(self._policy_by_id.get(self.query_one("#vllm-model", SearchableSelect).value, {}))
        model_id = custom or str(policy.get("repo", ""))
        if custom:
            policy = {"valid_tp": [1, 2], "attention_backend": "TRITON_ATTN", "extra_flags": [], "env": {}}
        if not engine or toolbox_id not in self.app.toolbox_catalog.toolboxes or not model_id:
            self.notify("Select an engine, vLLM image, and model repository.", severity="error")
            return
        policy = self._effective_policy(policy, str(toolbox_id))
        toolbox = self.app.toolbox_catalog.toolboxes[toolbox_id]
        allowed_models = (toolbox.backend_config or {}).get("supported_model_ids")
        if allowed_models and policy.get("id") not in allowed_models:
            self.notify("Select a model supported by this toolbox image.", severity="error")
            return
        self._hf_token = self._hf_token or get_hf_token()
        if not self._hf_token and not self._hf_token_prompted and not self.query_one("#vllm-local-model", Input).value.strip():
            self.app.push_screen(HfTokenModal(), self._hf_token_received)
            return
        try:
            port = int(self.query_one("#vllm-port", Input).value)
            tp = int(self.query_one("#vllm-tp", SearchableSelect).value)
            sequences = int(self.query_one("#vllm-seqs", Input).value)
            utilization = float(self.query_one("#vllm-util", Input).value)
            local_value = self.query_one("#vllm-local-model", Input).value.strip()
            local_directory = Path(local_value).expanduser().resolve() if local_value else None
            if local_directory is not None and not checkpoint_ready(local_directory):
                raise ValueError("Local model directory must contain the complete prepared checkpoint and config.json")
            draft_value = self.query_one("#vllm-draft", Input).value.strip()
            draft_directory = Path(draft_value).expanduser().resolve() if draft_value else None
            if draft_directory is not None and not checkpoint_ready(draft_directory, draft=True):
                raise ValueError("Draft model directory must contain the complete DFlash2 checkpoint and config.json")
            devices = self.query_one("#vllm-devices", Input).value.strip()
            if devices:
                import re
                if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", devices) or len(set(devices.split(","))) != len(devices.split(",")) or len(devices.split(",")) != tp:
                    raise ValueError("Choose distinct GPU indices matching the tensor parallel size")
                device_key = "ROCR_VISIBLE_DEVICES" if policy.get("runtime_variant") == "tcclaviger" else "HIP_VISIBLE_DEVICES"
                policy = dict(policy, env={**policy.get("env", {}), device_key: devices})
                save_backend_settings("vllm", {"devices": devices})
            caches = self.cache_paths()
            if policy.get("runtime_variant") == "tcclaviger":
                for suffix in ("", "ple", "tunableop", "lru_store"):
                    (caches.offload / suffix).mkdir(parents=True, exist_ok=True)
                if self.query_one("#vllm-reset-caches", CockpitCheckbox).value:
                    raise ValueError("Disable compiled-cache reset for the persistent tcclaviger PLE profile")
            for path in (caches.huggingface, caches.vllm, caches.triton, caches.aiter):
                path.mkdir(parents=True, exist_ok=True)
        except (ValueError, OSError) as error:
            self.notify(f"Invalid vLLM setting: {error}", severity="error")
            return
        toolbox = self._selected_toolbox(str(toolbox_id))
        profile = self.app.toolbox_catalog.runtime_profiles[toolbox.runtime_profile]
        try:
            self._pending_command = build_server_cmd(
                engine=engine,
                image=toolbox.image,
                engine_args=list(profile.engine_args),
                model_id=model_id,
                model_directory=local_directory,
                speculation=str(self.query_one("#vllm-speculation", SearchableSelect).value),
                draft_directory=draft_directory,
                policy=policy,
                host=self.query_one("#vllm-host", Input).value,
                port=port,
                tensor_parallel=tp,
                max_num_seqs=sequences,
                max_model_len=self.query_one("#vllm-context", Input).value,
                gpu_memory_utilization=utilization,
                attention_backend=self.query_one("#vllm-attention", SearchableSelect).value or None,
                enforce_eager=self.query_one("#vllm-eager", Checkbox).value,
                dtype=self.query_one("#vllm-dtype", Input).value or "auto",
                api_key=self.query_one("#vllm-api-key", Input).value,
                hf_token=self._hf_token,
                extra_args=self.query_one("#vllm-extra-args", TextArea).text,
                cache_paths=caches,
            )
        except ValueError as error:
            self.notify(str(error), severity="error")
            return
        self._pending_caches = caches
        reset = self.query_one("#vllm-reset-caches", CockpitCheckbox).value
        if reset:
            try:
                validate_compiled_cache_roots(caches)
            except ValueError as error:
                self.notify(str(error), severity="error")
                return
        reset_text = "\n\nThe vLLM, Triton, and AITER compiled cache contents will be permanently removed first." if reset else ""
        preview = redact_command(self._pending_command)
        self.app.push_screen(
            ConfirmModal(
                f"Start vLLM server?{reset_text}\n\n{shlex.join(preview)}",
                yes_text="Start",
                copy_text=shlex.join(preview),
            ),
            self._start_confirmed,
        )

    def _clear_compiled_caches(self) -> None:
        for resolved in validate_compiled_cache_roots(self._pending_caches):
            if not resolved.exists():
                resolved.mkdir(parents=True, exist_ok=True)
                continue
            for entry in resolved.iterdir():
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()

    def _start_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        command = self._pending_command
        preview = redact_command(command)
        with self.app.suspend():
            if self.query_one("#vllm-reset-caches", CockpitCheckbox).value:
                self._clear_compiled_caches()
            run_foreground_server(
                command,
                command[0],
                "ai-toolbox-cockpit-vllm-server",
                display_command=preview,
            )
