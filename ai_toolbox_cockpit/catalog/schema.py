from dataclasses import dataclass
from typing import Any


BACKEND_IDS = frozenset({"llama_cpp", "vllm", "comfyui", "ds4"})
FEATURE_STATES = frozenset({"supported", "experimental", "unavailable"})
FEATURE_IDS = frozenset({"interactive", "models", "server"})
CHANNELS = frozenset({"stable", "development", "experimental"})
MATURITY_STATES = frozenset({"stable", "experimental"})
LLAMA_KV_CACHE_TYPES = frozenset({"default", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"})
LLAMA_LOAD_MODES = frozenset({"none", "mmap", "dio"})
MODEL_KINDS = {
    "llama_cpp": "gguf",
    "ds4": "gguf_file",
    "vllm": "hf_repository",
    "comfyui": "workflow_bundle",
}


class CatalogError(ValueError):
    """Raised when a shipped catalogue is internally inconsistent."""


def _required_string(data: dict, key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{context}.{key} must be a non-empty string")
    return value


def _required_string_list(data: dict, key: str, context: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise CatalogError(f"{context}.{key} must be a non-empty array of strings")
    return value


def _validate_model_entry(backend_id: str, entry: dict[str, Any], context: str) -> None:
    _required_string(entry, "name", context)
    if backend_id == "llama_cpp":
        _required_string(entry, "repo", context)
        profiles = entry.get("inference_profiles", {})
        if not isinstance(profiles, dict):
            raise CatalogError(f"{context}.inference_profiles must be an object")
        toolbox_defaults = entry.get("toolbox_defaults", {})
        if not isinstance(toolbox_defaults, dict):
            raise CatalogError(f"{context}.toolbox_defaults must be an object")
        auxiliary_downloads = entry.get("auxiliary_downloads", [])
        if not isinstance(auxiliary_downloads, list):
            raise CatalogError(f"{context}.auxiliary_downloads must be an array")
        for index, download in enumerate(auxiliary_downloads):
            download_context = f"{context}.auxiliary_downloads[{index}]"
            if not isinstance(download, dict):
                raise CatalogError(f"{download_context} must be an object")
            for key in ("name", "repo", "role", "recommended_filename", "description"):
                _required_string(download, key, download_context)
            if download["role"] not in {"mtp", "vision_projector", "dspark"}:
                raise CatalogError(f"{download_context}.role is unsupported")
        for toolbox_id, defaults in toolbox_defaults.items():
            defaults_context = f"{context}.toolbox_defaults.{toolbox_id}"
            if not isinstance(toolbox_id, str) or not toolbox_id.strip():
                raise CatalogError(f"{context}.toolbox_defaults keys must be toolbox IDs")
            if not isinstance(defaults, dict):
                raise CatalogError(f"{defaults_context} must be an object")
            unknown = set(defaults).difference({
                "batch_size", "ubatch_size", "parallel_sequences", "gpu_layers",
                "kv_cache_type",
            })
            if unknown:
                raise CatalogError(
                    f"{defaults_context} has unsupported settings: {', '.join(sorted(unknown))}"
                )
            for key in ("batch_size", "ubatch_size", "parallel_sequences"):
                value = defaults.get(key)
                if value is not None and (not isinstance(value, int) or value <= 0):
                    raise CatalogError(f"{defaults_context}.{key} must be a positive integer")
            gpu_layers = defaults.get("gpu_layers")
            if gpu_layers is not None and (
                not isinstance(gpu_layers, int) or gpu_layers < 0
            ):
                raise CatalogError(
                    f"{defaults_context}.gpu_layers must be a non-negative integer or null"
                )
            kv_cache_type = defaults.get("kv_cache_type")
            if kv_cache_type is not None and kv_cache_type not in LLAMA_KV_CACHE_TYPES.difference({"default"}):
                raise CatalogError(f"{defaults_context}.kv_cache_type is unsupported")
        mtp = entry.get("mtp")
        if mtp is not None:
            if not isinstance(mtp, dict):
                raise CatalogError(f"{context}.mtp must be an object")
            if mtp.get("supported") is not True:
                raise CatalogError(f"{context}.mtp.supported must be true")
            for key in ("default_draft_n", "default_np"):
                value = mtp.get(key)
                if not isinstance(value, int) or value <= 0:
                    raise CatalogError(f"{context}.mtp.{key} must be a positive integer")
            if "draft_models" in mtp:
                _required_string_list(mtp, "draft_models", f"{context}.mtp")
        dspark = entry.get("dspark")
        if dspark is not None:
            if not isinstance(dspark, dict):
                raise CatalogError(f"{context}.dspark must be an object")
            if dspark.get("supported") is not True:
                raise CatalogError(f"{context}.dspark.supported must be true")
            patterns = _required_string_list(dspark, "patterns", f"{context}.dspark")
            default_pattern = _required_string(dspark, "default_pattern", f"{context}.dspark")
            if default_pattern not in patterns:
                raise CatalogError(f"{context}.dspark.default_pattern must be listed in patterns")
            for key in ("default_draft_n", "default_ngl"):
                value = dspark.get(key)
                if not isinstance(value, int) or value <= 0:
                    raise CatalogError(f"{context}.dspark.{key} must be a positive integer")
            if dspark.get("fit") not in {"on", "off"}:
                raise CatalogError(f"{context}.dspark.fit must be 'on' or 'off'")
    elif backend_id == "ds4":
        for key in ("repo", "filename", "family"):
            _required_string(entry, key, context)
        artifact_role = entry.get("artifact_role", "main")
        if artifact_role not in {"main", "vision_encoder", "dspark_support", "mtp"}:
            raise CatalogError(f"{context}.artifact_role is unsupported")
        size = entry.get("size_gb")
        if not isinstance(size, (int, float)) or size <= 0:
            raise CatalogError(f"{context}.size_gb must be positive")
    elif backend_id == "vllm":
        _required_string(entry, "repo", context)
        valid_tp = entry.get("valid_tp")
        if not isinstance(valid_tp, list) or not valid_tp or not all(isinstance(value, int) and value > 0 for value in valid_tp):
            raise CatalogError(f"{context}.valid_tp must contain positive integers")
        _required_string_list(entry, "extra_flags", context) if entry.get("extra_flags") else None
        environment = entry.get("env", {})
        if not isinstance(environment, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in environment.items()
        ):
            raise CatalogError(f"{context}.env must map strings to strings")
    else:
        for key in ("recipe_id", "script"):
            _required_string(entry, key, context)
        _required_string_list(entry, "keywords", context)
        variants = entry.get("variants")
        if not isinstance(variants, list) or not variants:
            raise CatalogError(f"{context}.variants must be a non-empty array")
        for variant_index, variant in enumerate(variants):
            variant_context = f"{context}.variants[{variant_index}]"
            if not isinstance(variant, dict):
                raise CatalogError(f"{variant_context} must be an object")
            _required_string(variant, "name", variant_context)
            _required_string_list(variant, "args", variant_context)


def _validate_calibrated_ubatches(
    config: dict[str, Any], model_ids: set[str], context: str
) -> None:
    records = config.get("calibrated_ubatches", [])
    if not isinstance(records, list):
        raise CatalogError(f"{context}.calibrated_ubatches must be an array")

    selectors: set[tuple[str, str, str, str, str]] = set()
    for index, record in enumerate(records):
        record_context = f"{context}.calibrated_ubatches[{index}]"
        if not isinstance(record, dict):
            raise CatalogError(f"{record_context} must be an object")
        allowed = {
            "model_id", "toolbox_id", "filename_pattern", "serving_config",
            "kv_cache_type", "batch_size", "ubatch_size", "source_job",
            "source_job_status",
        }
        unknown = set(record).difference(allowed)
        if unknown:
            raise CatalogError(
                f"{record_context} has unsupported fields: {', '.join(sorted(unknown))}"
            )

        model_id = _required_string(record, "model_id", record_context)
        toolbox_id = _required_string(record, "toolbox_id", record_context)
        filename_pattern = _required_string(record, "filename_pattern", record_context)
        serving_config = _required_string(record, "serving_config", record_context)
        kv_cache_type = _required_string(record, "kv_cache_type", record_context)
        if model_id not in model_ids:
            raise CatalogError(f"{record_context}.model_id references unknown llama.cpp model")
        if kv_cache_type not in LLAMA_KV_CACHE_TYPES:
            raise CatalogError(f"{record_context}.kv_cache_type is unsupported")
        for key in ("batch_size", "ubatch_size"):
            value = record.get(key)
            if not isinstance(value, int) or value <= 0:
                raise CatalogError(f"{record_context}.{key} must be a positive integer")

        selector = (model_id, toolbox_id, filename_pattern, serving_config, kv_cache_type)
        if selector in selectors:
            raise CatalogError(f"{record_context} duplicates a calibrated ubatch selector")
        selectors.add(selector)

        _required_string(record, "source_job", record_context)
        source_job_status = _required_string(record, "source_job_status", record_context)
        if source_job_status not in {"complete", "partial", "failed"}:
            raise CatalogError(f"{record_context}.source_job_status is unsupported")


def _validate_llama_toolbox_backend_config(
    config: dict[str, Any], context: str
) -> None:
    """Validate optional llama.cpp policy owned by one toolbox/fork."""
    recommended = config.get("recommended_use")
    if recommended is None:
        return
    if not isinstance(recommended, dict):
        raise CatalogError(f"{context}.recommended_use must be an object")
    for key in (
        "platform_id",
        "model_id",
        "model_filename_pattern",
        "model_display_name",
        "message",
        "documentation_url",
    ):
        _required_string(recommended, key, f"{context}.recommended_use")

    notes = recommended.get("notes", [])
    if not isinstance(notes, list) or not all(
        isinstance(note, str) and note.strip() for note in notes
    ):
        raise CatalogError(
            f"{context}.recommended_use.notes must be an array of non-empty strings"
        )

    sidecar = recommended.get("sidecar")
    if sidecar is not None:
        if not isinstance(sidecar, dict):
            raise CatalogError(f"{context}.recommended_use.sidecar must be an object")
        for key in ("repo", "filename"):
            _required_string(sidecar, key, f"{context}.recommended_use.sidecar")

    defaults = recommended.get("server_defaults")
    if not isinstance(defaults, dict):
        raise CatalogError(
            f"{context}.recommended_use.server_defaults must be an object"
        )
    unknown = set(defaults).difference({
        "context_size", "batch_size", "ubatch_size", "gpu_layers",
        "parallel_sequences", "kv_cache_type", "load_mode",
        "flash_attention", "extra_args", "mtp",
    })
    if unknown:
        raise CatalogError(
            f"{context}.recommended_use.server_defaults has unsupported settings: "
            f"{', '.join(sorted(unknown))}"
        )
    for key in ("context_size", "batch_size", "ubatch_size", "parallel_sequences"):
        value = defaults.get(key)
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise CatalogError(
                f"{context}.recommended_use.server_defaults.{key} must be a positive integer"
            )
    gpu_layers = defaults.get("gpu_layers")
    if gpu_layers is not None and (
        not isinstance(gpu_layers, int) or gpu_layers < 0
    ):
        raise CatalogError(
            f"{context}.recommended_use.server_defaults.gpu_layers must be a non-negative integer"
        )
    if (
        "load_mode" in defaults
        and defaults["load_mode"] not in LLAMA_LOAD_MODES
    ):
        raise CatalogError(
            f"{context}.recommended_use.server_defaults.load_mode is unsupported"
        )
    if "flash_attention" in defaults and not isinstance(
        defaults["flash_attention"], bool
    ):
        raise CatalogError(
            f"{context}.recommended_use.server_defaults.flash_attention must be boolean"
        )
    if "kv_cache_type" in defaults and defaults["kv_cache_type"] not in LLAMA_KV_CACHE_TYPES.difference({"default"}):
        raise CatalogError(
            f"{context}.recommended_use.server_defaults.kv_cache_type is unsupported"
        )
    if "extra_args" in defaults and not isinstance(defaults["extra_args"], str):
        raise CatalogError(
            f"{context}.recommended_use.server_defaults.extra_args must be a string"
        )

    mtp = defaults.get("mtp")
    if mtp is None:
        return
    if not isinstance(mtp, dict):
        raise CatalogError(f"{context}.recommended_use.server_defaults.mtp must be an object")
    for key in ("default_draft_n", "default_np"):
        value = mtp.get(key)
        if not isinstance(value, int) or value <= 0:
            raise CatalogError(
                f"{context}.recommended_use.server_defaults.mtp.{key} must be a positive integer"
            )
    spec_types = _required_string_list(
        mtp, "spec_types", f"{context}.recommended_use.server_defaults.mtp"
    )
    if set(spec_types).difference({"draft-mtp", "ngram-mod"}) or "draft-mtp" not in spec_types:
        raise CatalogError(
            f"{context}.recommended_use.server_defaults.mtp.spec_types is unsupported"
        )
    for key in ("ngram_mod_n_max", "ngram_mod_n_match"):
        value = mtp.get(key)
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise CatalogError(
                f"{context}.recommended_use.server_defaults.mtp.{key} must be a positive integer"
            )
    p_min = mtp.get("spec_draft_p_min")
    if p_min is not None and (
        not isinstance(p_min, (int, float)) or isinstance(p_min, bool)
        or not 0 <= p_min <= 1
    ):
        raise CatalogError(
            f"{context}.recommended_use.server_defaults.mtp.spec_draft_p_min must be between 0 and 1"
        )


@dataclass(frozen=True)
class RuntimeProfile:
    id: str
    engine_args: tuple[str, ...]


@dataclass(frozen=True)
class Toolbox:
    id: str
    backend: str
    name: str
    container_name: str
    group: str
    image: str
    channel: str
    maturity: str
    description: str
    runtime_profile: str
    features: dict[str, str]
    server_binary: str = ""
    supports_load_mode: bool = False
    backend_config: dict[str, Any] | None = None

    def feature_state(self, feature: str) -> str:
        return self.features.get(feature, "unavailable")


@dataclass(frozen=True)
class Platform:
    id: str
    name: str
    description: str
    toolbox_ids: tuple[str, ...]
    defaults: dict[str, str]


@dataclass(frozen=True)
class ToolboxCatalog:
    schema_version: int
    runtime_profiles: dict[str, RuntimeProfile]
    toolboxes: dict[str, Toolbox]
    platforms: tuple[Platform, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolboxCatalog":
        if data.get("schema_version") != 3:
            raise CatalogError("toolboxes.json schema_version must be 3")

        raw_profiles = data.get("runtime_profiles")
        if not isinstance(raw_profiles, dict):
            raise CatalogError("runtime_profiles must be an object")
        profiles: dict[str, RuntimeProfile] = {}
        for profile_id, raw in raw_profiles.items():
            if not isinstance(raw, dict):
                raise CatalogError(f"runtime profile {profile_id} must be an object")
            args = raw.get("engine_args", [])
            if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
                raise CatalogError(f"runtime profile {profile_id}.engine_args must be strings")
            profiles[profile_id] = RuntimeProfile(profile_id, tuple(args))

        raw_toolboxes = data.get("toolboxes")
        if not isinstance(raw_toolboxes, list):
            raise CatalogError("toolboxes must be an array")
        toolboxes: dict[str, Toolbox] = {}
        container_names: set[str] = set()
        for index, raw in enumerate(raw_toolboxes):
            context = f"toolboxes[{index}]"
            if not isinstance(raw, dict):
                raise CatalogError(f"{context} must be an object")
            toolbox_id = _required_string(raw, "id", context)
            if toolbox_id in toolboxes:
                raise CatalogError(f"duplicate toolbox id: {toolbox_id}")
            backend = _required_string(raw, "backend", context)
            if backend not in BACKEND_IDS:
                raise CatalogError(f"unknown backend id {backend!r} in {context}")
            image = _required_string(raw, "image", context)
            if "/" not in image or (":" not in image and "@" not in image):
                raise CatalogError(f"{context}.image must be a complete tagged OCI reference")
            profile_id = _required_string(raw, "runtime_profile", context)
            if profile_id not in profiles:
                raise CatalogError(f"unknown runtime profile {profile_id!r} in {context}")
            features = raw.get("features", {})
            if not isinstance(features, dict):
                raise CatalogError(f"{context}.features must be an object")
            unknown_features = set(features).difference(FEATURE_IDS)
            missing_features = FEATURE_IDS.difference(features)
            if unknown_features or missing_features:
                raise CatalogError(
                    f"{context}.features must declare exactly: {', '.join(sorted(FEATURE_IDS))}"
                )
            for feature, state in features.items():
                if state not in FEATURE_STATES:
                    raise CatalogError(f"invalid feature state {state!r} for {toolbox_id}.{feature}")
            container_name = _required_string(raw, "container_name", context)
            if container_name in container_names:
                raise CatalogError(f"duplicate toolbox container_name: {container_name}")
            container_names.add(container_name)
            channel = _required_string(raw, "channel", context)
            maturity = _required_string(raw, "maturity", context)
            if channel not in CHANNELS:
                raise CatalogError(f"invalid channel {channel!r} in {context}")
            if maturity not in MATURITY_STATES:
                raise CatalogError(f"invalid maturity {maturity!r} in {context}")
            backend_config = raw.get("backend_config", {})
            if not isinstance(backend_config, dict):
                raise CatalogError(f"{context}.backend_config must be an object")
            if backend == "llama_cpp":
                _validate_llama_toolbox_backend_config(
                    backend_config, f"{context}.backend_config"
                )
            toolboxes[toolbox_id] = Toolbox(
                id=toolbox_id,
                backend=backend,
                name=_required_string(raw, "name", context),
                container_name=container_name,
                group=_required_string(raw, "group", context),
                image=image,
                channel=channel,
                maturity=maturity,
                description=str(raw.get("description", "")),
                runtime_profile=profile_id,
                features=dict(features),
                server_binary=str(raw.get("server_binary", "")),
                supports_load_mode=bool(raw.get("supports_load_mode", False)),
                backend_config=dict(backend_config),
            )

        raw_platforms = data.get("platforms")
        if not isinstance(raw_platforms, list):
            raise CatalogError("platforms must be an array")
        platforms: list[Platform] = []
        platform_ids: set[str] = set()
        assigned_toolboxes: dict[str, str] = {}
        for index, raw in enumerate(raw_platforms):
            context = f"platforms[{index}]"
            if not isinstance(raw, dict):
                raise CatalogError(f"{context} must be an object")
            platform_id = _required_string(raw, "id", context)
            if platform_id in platform_ids:
                raise CatalogError(f"duplicate platform id: {platform_id}")
            platform_ids.add(platform_id)
            ids = raw.get("toolbox_ids", [])
            if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
                raise CatalogError(f"{context}.toolbox_ids must be strings")
            missing = [toolbox_id for toolbox_id in ids if toolbox_id not in toolboxes]
            if missing:
                raise CatalogError(f"{context} references missing toolboxes: {', '.join(missing)}")
            for toolbox_id in ids:
                previous = assigned_toolboxes.get(toolbox_id)
                if previous:
                    raise CatalogError(
                        f"toolbox {toolbox_id!r} is assigned to both {previous!r} and {platform_id!r}"
                    )
                assigned_toolboxes[toolbox_id] = platform_id
            defaults = raw.get("defaults", {})
            if not isinstance(defaults, dict):
                raise CatalogError(f"{context}.defaults must be an object")
            for backend, toolbox_id in defaults.items():
                if backend not in BACKEND_IDS:
                    raise CatalogError(f"unknown default backend {backend!r} in {context}")
                if toolbox_id not in ids:
                    raise CatalogError(f"default toolbox {toolbox_id!r} is not assigned to {platform_id}")
                if toolboxes[toolbox_id].backend != backend:
                    raise CatalogError(f"default toolbox {toolbox_id!r} does not use backend {backend}")
            platforms.append(Platform(
                id=platform_id,
                name=_required_string(raw, "name", context),
                description=str(raw.get("description", "")),
                toolbox_ids=tuple(ids),
                defaults=dict(defaults),
            ))

        unassigned = set(toolboxes).difference(assigned_toolboxes)
        if unassigned:
            raise CatalogError(f"toolboxes are not assigned to a platform: {', '.join(sorted(unassigned))}")

        return cls(int(data["schema_version"]), profiles, toolboxes, tuple(platforms))

    def platform(self, platform_id: str) -> Platform:
        for platform in self.platforms:
            if platform.id == platform_id:
                return platform
        raise KeyError(platform_id)

    def platform_toolboxes(self, platform_id: str) -> tuple[Toolbox, ...]:
        platform = self.platform(platform_id)
        return tuple(self.toolboxes[toolbox_id] for toolbox_id in platform.toolbox_ids)


@dataclass(frozen=True)
class ModelBackendCatalog:
    id: str
    kind: str
    storage: dict[str, Any]
    config: dict[str, Any]
    entries_key: str
    entries: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ModelCatalog:
    schema_version: int
    backends: dict[str, ModelBackendCatalog]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelCatalog":
        if data.get("schema_version") != 2:
            raise CatalogError("models.json schema_version must be 2")
        raw_backends = data.get("backends")
        if not isinstance(raw_backends, dict):
            raise CatalogError("models.json backends must be an object")

        backends: dict[str, ModelBackendCatalog] = {}
        for backend_id, raw in raw_backends.items():
            context = f"backends.{backend_id}"
            if backend_id not in BACKEND_IDS:
                raise CatalogError(f"unknown model backend {backend_id!r}")
            if not isinstance(raw, dict):
                raise CatalogError(f"{context} must be an object")
            kind = _required_string(raw, "kind", context)
            if kind != MODEL_KINDS[backend_id]:
                raise CatalogError(
                    f"{context}.kind must be {MODEL_KINDS[backend_id]!r}, got {kind!r}"
                )
            storage = raw.get("storage", {})
            if not isinstance(storage, dict):
                raise CatalogError(f"{context}.storage must be an object")
            _required_string(storage, "config_key", f"{context}.storage")
            _required_string(storage, "default", f"{context}.storage")
            config = raw.get("config", {})
            if not isinstance(config, dict):
                raise CatalogError(f"{context}.config must be an object")
            entries_key = "bundles" if backend_id == "comfyui" else "models"
            entries = raw.get(entries_key, [])
            if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
                raise CatalogError(f"{context}.{entries_key} must contain objects")
            seen: set[str] = set()
            for index, entry in enumerate(entries):
                entry_id = _required_string(entry, "id", f"{context}.{entries_key}[{index}]")
                if entry_id in seen:
                    raise CatalogError(f"duplicate {backend_id} model/bundle id: {entry_id}")
                seen.add(entry_id)
                _validate_model_entry(backend_id, entry, f"{context}.{entries_key}[{index}]")
            if backend_id == "llama_cpp":
                _validate_calibrated_ubatches(config, seen, f"{context}.config")
            backends[backend_id] = ModelBackendCatalog(
                backend_id, kind, dict(storage), dict(config), entries_key, tuple(entries)
            )
        missing = BACKEND_IDS.difference(backends)
        if missing:
            raise CatalogError(f"models.json is missing backends: {', '.join(sorted(missing))}")
        return cls(int(data["schema_version"]), backends)
