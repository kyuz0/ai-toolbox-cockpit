# Architecture and backend extension contract

AI Toolbox Cockpit separates shared workstation operations from backend-owned model and server behavior. A backend is not a command template in JSON; it is a Python package with explicit validation and a pure command builder.

## Layers

- `app.py` owns the theme, global platform selector, tabs, version display, and background application-update notice.
- `views/` owns the three top-level workflows: toolboxes, servers, and models.
- `runtime/` owns engine/wrapper detection, image metadata, toolbox lifecycle commands, foreground server cleanup, and subprocess environment selection.
- `catalog/` loads and validates static JSON. Invalid or ambiguous shipped data stops startup with a specific `CatalogError`.
- `backends/<id>/` owns one backend's model panel, server panel, and pure command builders.

The registered backend IDs are `llama_cpp`, `ds4`, `vllm`, and `comfyui`. `ai_toolbox_cockpit/backends/__init__.py` is the only Python registry the shared views consume.

## `toolboxes.json`

Schema version 3 has three top-level sections:

- `runtime_profiles`: reusable native container-engine arguments;
- `toolboxes`: one unambiguous OCI image/container definition per backend and platform;
- `platforms`: the toolbox IDs visible for that hardware and each backend's default.

Every toolbox must declare:

- a unique `id` and unique runtime `container_name`;
- one of the registered backend IDs;
- a full tagged or digest-pinned OCI `image` reference;
- `channel`, `maturity`, display `group`, and `runtime_profile`;
- explicit `interactive`, `models`, and `server` feature states.

Feature states are `supported`, `experimental`, or `unavailable`. A toolbox must be assigned to exactly one platform. Reusing an image on another platform requires a separate toolbox record, keeping defaults, runtime names, and feature maturity explicit.

## `models.json`

Schema version 2 deliberately has four different record types:

- `llama_cpp`: GGUF repository records with optional profiles, MTP, vision-projector, and compatibility metadata;
- `ds4`: exact repository/filename artifacts with family, size, and optional server defaults;
- `vllm`: Hugging Face repositories with tensor-parallel, environment, attention, eager, context, parser, and extra-flag policy;
- `comfyui`: workflow bundles with maintained script/recipe IDs, matching keywords, and model-manager variants.

Each section declares its storage key and default. Backend-specific required fields and value types are validated in `catalog/schema.py`; the UI never guesses a missing backend policy.

The llama.cpp backend may also contain `config.calibrated_ubatches`. Each record
matches one cockpit model ID, toolbox ID, GGUF filename pattern, serving
configuration, and KV-cache type. Only an exact match supplies calibrated batch
and ubatch values. The source benchmark job ID remains in the record so its
engine revision and calibration evidence can be audited.

The import scripts under `scripts/` regenerate source-derived sections from the maintained toolbox projects. Review their JSON diff before committing because source launchers remain the policy authority.

## Adding a backend

1. Choose a stable backend ID and add it to `BACKEND_IDS` and `MODEL_KINDS` in `catalog/schema.py`.
2. Add backend-specific model validation in `_validate_model_entry`.
3. Create `backends/<id>/models.py`, `server.py`, and a pure command-builder module such as `runner.py`.
4. Register the model and server panel classes in `backends/__init__.py`.
5. Add its model section, toolbox records, platform assignments, defaults, and all three feature states.
6. Add command-policy tests that assert exact flags, mounts, environment variables, and rejected combinations without executing a container.
7. Add one known-good remote validation case before changing server maturity from experimental to supported.

Do not add backend conditionals to `app.py`. Shared behavior belongs in `runtime/`; model semantics and launch policy remain in the backend package.

## Operation boundaries

- Container mutations and server launches show generated commands and require confirmation.
- Interactive shells, model downloads, model managers, and servers suspend Textual while they own the terminal.
- API keys are password inputs, are not persisted, and are redacted from confirmation and terminal command display.
- vLLM cache reset excludes the Hugging Face model cache and rejects broad/mismatched cache paths before deletion.
- JSON contains data only and is never evaluated as shell code. User-provided extra arguments are parsed with `shlex` and appended to an argument vector.
