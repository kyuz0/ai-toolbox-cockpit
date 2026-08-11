"""vLLM direct-container server UI."""

import shlex
import shutil
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Label, Static, Switch

from ai_toolbox_cockpit.backends.base import BackendServerPanel
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.server_process import redact_command, run_foreground_server
from ai_toolbox_cockpit.settings import get_backend_settings, load_default_toolbox, save_backend_settings
from ai_toolbox_cockpit.widgets import ConfirmModal, SearchableSelect

from .runner import VllmCachePaths, build_server_cmd, default_cache_paths


ATTENTION_BACKENDS = ("TRITON_ATTN", "ROCM_ATTN", "ROCM_AITER_UNIFIED_ATTN")


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
        self._pending_command: list[str] = []
        self._pending_caches = default_cache_paths()
        self._policy_by_id: dict[str, dict] = {}

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label(self.backend_label, classes="panel-title")
            yield Static(
                "Launch a curated Hugging Face repository with the exact toolbox policy for tensor parallelism, attention, eager mode, environment, and extra flags.",
                classes="panel-copy",
            )
            with Horizontal(classes="inline-row"):
                yield Label("Container engine", classes="inline-label")
                yield SearchableSelect("Select Podman or Docker", id="vllm-engine")
            with Horizontal(classes="inline-row"):
                yield Label("Toolbox image", classes="inline-label")
                yield SearchableSelect("Search vLLM images", id="vllm-image")
            with Horizontal(classes="inline-row"):
                yield Label("Curated model", classes="inline-label")
                yield SearchableSelect("Search maintained model policy", id="vllm-model")
            with Horizontal(classes="inline-row"):
                yield Label("Custom HF repo", classes="inline-label")
                yield Input(placeholder="Optional owner/model; bypasses curated policy", id="vllm-custom-model")

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
                    yield Label("Force eager mode", classes="option-label")
                    yield Switch(value=False, id="vllm-eager")
            yield Static("", id="vllm-policy-note", classes="panel-copy")

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
                    yield Switch(value=False, id="vllm-reset-caches")
                    yield Label("Reset compiled caches before launch")
            with Horizontal(classes="inline-row"):
                yield Label("API key", classes="inline-label")
                yield Input(placeholder="Optional OpenAI-compatible API key", password=True, id="vllm-api-key")
            with Horizontal(classes="inline-row"):
                yield Label("Extra args", classes="inline-label")
                yield Input(placeholder="Additional vllm serve flags", id="vllm-extra-args")
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
        ):
            self.query_one(field, Input).value = str(settings.get(key, fallback))
        entries = self.app.model_catalog.backends["vllm"].entries
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
        select.value = default if default in {toolbox.id for toolbox in toolboxes} else (toolboxes[0].id if toolboxes else "")

    @on(SearchableSelect.Changed, "#vllm-model")
    def model_changed(self, event: SearchableSelect.Changed) -> None:
        policy = self._policy_by_id.get(str(event.value), {})
        valid_tp = [int(value) for value in policy.get("valid_tp", [1])]
        tp = self.query_one("#vllm-tp", SearchableSelect)
        tp.set_options([(str(value), str(value)) for value in valid_tp])
        tp.value = str(valid_tp[0])
        self.query_one("#vllm-seqs", Input).value = "1"
        self.query_one("#vllm-context", Input).value = str(policy.get("ctx", "auto"))
        self.query_one("#vllm-eager", Switch).value = bool(policy.get("enforce_eager", False))
        configured_attention = policy.get("attention_backend", "TRITON_ATTN")
        attention = self.query_one("#vllm-attention", SearchableSelect)
        if configured_attention is None:
            attention.disabled = True
            attention.value = ""
            attention_text = policy.get("attention_backend_label", "model-specific")
        else:
            attention.disabled = False
            attention.value = str(configured_attention)
            attention_text = str(configured_attention)
        environment = ", ".join(f"{key}={value}" for key, value in policy.get("env", {}).items()) or "none"
        extras = shlex.join([str(value) for value in policy.get("extra_flags", [])]) or "none"
        self.query_one("#vllm-policy-note", Static).update(
            f"Policy: attention {attention_text}; environment {environment}; flags {extras}"
        )

    def cache_paths(self) -> VllmCachePaths:
        return VllmCachePaths(*(
            Path(self.query_one(field, Input).value).expanduser().resolve()
            for field in ("#vllm-hf-cache", "#vllm-compile-cache", "#vllm-triton-cache", "#vllm-aiter-cache")
        ))

    @on(Button.Pressed, "#vllm-save-caches")
    def save_caches_pressed(self) -> None:
        try:
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
        })
        self.notify("vLLM cache paths saved." if saved else "Could not save vLLM cache paths.", severity="information" if saved else "error")

    @on(Button.Pressed, "#vllm-start")
    def start_pressed(self) -> None:
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
        try:
            port = int(self.query_one("#vllm-port", Input).value)
            tp = int(self.query_one("#vllm-tp", SearchableSelect).value)
            sequences = int(self.query_one("#vllm-seqs", Input).value)
            utilization = float(self.query_one("#vllm-util", Input).value)
            caches = self.cache_paths()
            for path in (caches.huggingface, caches.vllm, caches.triton, caches.aiter):
                path.mkdir(parents=True, exist_ok=True)
        except (ValueError, OSError) as error:
            self.notify(f"Invalid vLLM setting: {error}", severity="error")
            return
        toolbox = self.app.toolbox_catalog.toolboxes[toolbox_id]
        profile = self.app.toolbox_catalog.runtime_profiles[toolbox.runtime_profile]
        try:
            self._pending_command = build_server_cmd(
                engine=engine,
                image=toolbox.image,
                engine_args=list(profile.engine_args),
                model_id=model_id,
                policy=policy,
                host=self.query_one("#vllm-host", Input).value,
                port=port,
                tensor_parallel=tp,
                max_num_seqs=sequences,
                max_model_len=self.query_one("#vllm-context", Input).value,
                gpu_memory_utilization=utilization,
                attention_backend=self.query_one("#vllm-attention", SearchableSelect).value or None,
                enforce_eager=self.query_one("#vllm-eager", Switch).value,
                dtype=self.query_one("#vllm-dtype", Input).value or "auto",
                api_key=self.query_one("#vllm-api-key", Input).value,
                extra_args=self.query_one("#vllm-extra-args", Input).value,
                cache_paths=caches,
            )
        except ValueError as error:
            self.notify(str(error), severity="error")
            return
        self._pending_caches = caches
        reset = self.query_one("#vllm-reset-caches", Switch).value
        if reset:
            try:
                validate_compiled_cache_roots(caches)
            except ValueError as error:
                self.notify(str(error), severity="error")
                return
        reset_text = "\n\nThe vLLM, Triton, and AITER compiled cache contents will be permanently removed first." if reset else ""
        preview = redact_command(self._pending_command)
        self.app.push_screen(
            ConfirmModal(f"Start vLLM server?{reset_text}\n\n{shlex.join(preview)}", yes_text="Start"),
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
            if self.query_one("#vllm-reset-caches", Switch).value:
                self._clear_compiled_caches()
            run_foreground_server(
                command,
                command[0],
                "ai-toolbox-cockpit-vllm-server",
                display_command=preview,
            )
