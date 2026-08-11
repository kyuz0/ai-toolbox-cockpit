# Implementation notes

## 2026-08-11 — redundant Server Mode support row removed

The Server Mode line listing the selected platform followed by `llama.cpp: supported`, `vLLM: supported`, `ComfyUI: supported`, and `DS4: supported` duplicated the available inference-engine choices and consumed vertical space. The widget and its status-building code have been removed. Platform changes still propagate to every backend panel so image filtering and backend-specific availability continue to work.

## 2026-08-11 — vLLM LFM2.5 and Muse Glimmer catalog sync

The cockpit vLLM catalog now includes `LiquidAI/LFM2.5-1.2B-Instruct` and `meta-models/Muse-Glimmer-30B`, copied from the checked-out vLLM toolbox's `scripts/models.py`. Both preserve the source `ROCM_AITER_UNIFIED_ATTN` selection, TP 1/2 support, and disabled broad AITER toggles. LFM uses its native Hugging Face repository directly with a `128000` context and no GGUF tokenizer/config workaround; Muse keeps its `131072` context plus `--model-impl transformers`. Catalog and pure command tests verify the exact records without downloading either model or starting vLLM.

## 2026-08-11 — vLLM launch-setting source of truth

The vLLM form rendered a `Policy:` sentence built only when the curated model changed. It duplicated live controls and immediately became incorrect when the user selected another attention backend. The sentence has been removed; the controls are now the only visible source of launch settings, and command construction continues to consume their current values. Remaining user-facing `policy` terminology was replaced with `maintained launch defaults` or `model launch recipe`. For models such as DeepSeek V4 that supply their own attention implementation and must not receive `--attention-backend`, the disabled field now shows the model-specific implementation under the explicit label `Required attention backend`. Headless UI coverage verifies user overrides, required-model state, and restoration when changing models; command coverage verifies that `ROCM_ATTN` overrides a model's `TRITON_ATTN` default.

## 2026-08-11 — compact vLLM eager control

The vLLM `Force eager mode` boolean was rendered as a Textual `Switch`, whose multi-cell track appeared as unexplained grey padding beside the label. It now uses the same compact labelled `Checkbox` pattern as the llama.cpp Server Mode options, with its inherited horizontal padding explicitly removed. Policy application and command construction both read the checkbox value directly, and a headless interaction test verifies its one-row compact geometry and off-to-on toggle behavior.

## 2026-08-11 — labelled vLLM server settings

The initial vLLM Server Mode form placed twelve numeric, policy, network, and cache controls into anonymous `settings-row` containers. Values and placeholders were incorrectly doing the work of persistent field labels. The form now groups settings into `Runtime limits`, `Network and execution`, and `Persistent cache paths`. Every control has a visible label directly above it: Tensor parallel, Max sequences, Context length, GPU memory, Host, Port, Data type, Attention backend, Hugging Face cache, vLLM cache, Triton cache, and AITER cache. The top fields are also clarified as `Container engine` and `Toolbox image`. A headless 180×45 test verifies each label's text, parent ownership, and position above its control; separate first-viewport and scrolled SVG checks cover runtime/network and cache/action sections.

## 2026-08-11 — explicit DeepSeek llama.cpp profile default

The server previously inferred a model's default inference profile from JSON object order, which made DeepSeek V4 Flash 0731 auto-select `Thinking (Effort: Max)`. The llama.cpp model schema now supports `default_inference_profile`; DeepSeek explicitly declares `Thinking (Effort: High)`. Server Mode validates that configured profile against the model's profile map and falls back to the first profile only for legacy entries without an explicit valid default. Catalog and headless UI tests verify that DeepSeek selects High and emits `reasoning_effort: high`, never Max, on initial model selection.

## 2026-08-11 — Server Mode selector and profile spacing correction

The backend selector in Server Mode was emitted as a bare `SearchableSelect`, so shared sizing expanded it to nearly the entire viewport without explaining what `llama.cpp`, `vLLM`, `ComfyUI`, and `DS4` represented. It now sits in a labelled `Inference engine` row and is capped at 32 columns. Dynamically displayed `.model-zone` panels previously had only bottom margin, allowing the inference-profile border to begin on the same row where the model control ended. Model zones now have symmetric one-row vertical margins. A 180×45 headless geometry test holds the selector to at most 40 columns and requires at least one blank row between the model row and profile panel.

## 2026-08-11 — model table ownership and local inventory correction

`BackendModelPanel.on_mount()` was dispatched by Textual in addition to each concrete backend's own `on_mount()` handler. Its unscoped `query_one(DataTable)` selected the first table in every concrete panel, appended generic catalog columns, and inserted curated entries into tables that were explicitly labelled as local filesystem inventories. This produced nonexistent llama.cpp and DS4 "local" models and duplicated the vLLM and ComfyUI catalogs. The generic base mount handler has been removed because every registered backend owns a concrete model panel. The base class now exposes only a `refresh_inventory()` lifecycle contract. Opening the Models tab or changing its backend calls that contract; llama.cpp and DS4 rescan disk, while vLLM recomputes cache state. Regression coverage verifies exact per-backend column/row ownership and proves every llama.cpp local row maps to a real fixture file.

## 2026-08-11 — toolbox selection marker rendering fix

The toolbox table initially passed raw `[x]` strings to Textual's `DataTable`. Textual 8 formats string cells with `rich.text.Text.from_markup`, so `[x]` was parsed as a zero-width Rich tag and disappeared as soon as a row became selected. The working llama cockpit avoided this by escaping the opening bracket. The shared cockpit now uses one `selection_marker()` helper that returns literal Rich `Text` for both `[ ]` and `[x]`. A headless Textual interaction test selects the exact `strix-vllm-dev` row and verifies that the rendered cell remains a literal `[x]` while selection state contains the matching toolbox ID.

## 2026-08-11 — baseline and scope correction

The initial project was an architectural scaffold, not a functional port. That is not an acceptable interpretation of the requested cockpit. The working definition of completion is now functional parity with the existing llama.cpp cockpit plus functional backend implementations for DS4, vLLM, and ComfyUI.

The source projects remain read-only inputs:

- `/home/kyuz0/Documents/Projects/llama-toolboxes-cockpit`
- `/home/kyuz0/Documents/Projects/strix-halo-ds4-toolbox/ds4-strix-halo-cockpit`
- `/home/kyuz0/Documents/Projects/amd-strix-halo-vllm-toolboxes`
- `/home/kyuz0/Documents/Projects/amd-strix-halo-comfyui-toolboxes`

Implementation order is deliberately dependency-first: shared runtime, toolbox management, llama.cpp parity, DS4 parity, vLLM, ComfyUI, integration, packaging, and documentation.

### Validation limits

Local tests may inspect files, build commands, instantiate the Textual app, build packages, and mock subprocess/network responses. They must not pull images, invoke container lifecycle commands, start servers, download models, or run GPU workloads. Those paths will receive explicit remote test cases rather than false claims of local validation.

### Packaging addition

The project repository is `https://github.com/kyuz0/ai-toolbox-cockpit`. Release parity includes installability with:

```bash
pipx install git+https://github.com/kyuz0/ai-toolbox-cockpit.git
```

The application update check must use that repository and instruct installed users to run `pipx upgrade ai-toolbox-cockpit`.

## 2026-08-11 — functional port completed locally

Implemented surfaces:

- shared Podman/Docker and Toolbx/Distrobox discovery, engine-bound installed-container operations, full toolbox lifecycle, command previews, Docker Hub dates, explicit row selection, backend/channel filters, and per-platform defaults;
- llama.cpp GGUF inventory/download planning, profiles, MTP, vision projectors, and direct server mode;
- DS4 exact-artifact manager, model defaults, standalone/distributed server roles, disk KV, SSD streaming, MTP, and distributed prefill controls;
- vLLM curated/custom Hugging Face explorer, local cache inventory, policy-aware direct server mode, persistent compiler/model caches, API key/dtype controls, and guarded compiled-cache reset;
- ComfyUI bundle browser, maintained toolbox `model_manager` bridge, persistent host directories, and direct server mode;
- GitHub-tag update notice, installed package version, pipx entry points, package data, and source-catalog import scripts.

Shared foreground process cleanup now lives in `runtime/server_process.py`; each backend keeps its own UI and command builder. API keys are passed only to the launched server process and redacted from both confirmation and terminal command display.

### Local evidence

The local validation is intentionally non-runtime:

- catalog import/validation and all command-policy tests;
- Textual mount, 80x24 layout, and keyboard-focus smoke test with container inspection/update calls mocked;
- bytecode compilation with output redirected under `/tmp`;
- wheel and source archive creation plus inspection of package assets and console entry points.

No container was pulled, created, started, entered, updated, or deleted. No model/workflow was downloaded. No GPU server was run.

### Remaining external gates

The code is locally complete, but actual backend startup is not honestly testable on this machine. Follow `docs/REMOTE_VALIDATION.md` and record outcomes one known-good image/model at a time. vLLM, ComfyUI, DS4 distributed mode, and platform-specific experimental entries must remain described as hardware-validation pending until those runs pass.

## 2026-08-11 — compact visual parity correction

The original `llama-toolboxes-cockpit` is the visual source of truth. AI Toolbox Cockpit now uses its exact red theme values and compact Textual rules for tabs, table cursors/headers, one-row buttons, one-row inputs, searchable selectors, model zones, notices, and modal dialogs. Tall native Textual `Select` controls were replaced by the shared one-row `SearchableSelect` for platform, backend, and channel choices.

The 80x24 Textual smoke test asserts that the platform selector, both toolbox filters, and toolbox action buttons remain exactly one terminal row high. The complete suite remains at 59 passing tests, and a fresh wheel builds successfully without network access.

The visible tab is named **Server Mode**, matching the original cockpit. The top title is a five-line `small`-font ASCII banner that fits within 80 columns. Compact controls remain one row high, while vertical spacing and padding were restored around the banner, platform row, notices, filters, action rows, backend panels, and model zones so sections are visually separated rather than packed together.

## 2026-08-11 — automatic CalVer workflow correction

The original cockpit's `.github/workflows/auto-version.yml` was initially omitted. It is now ported into this repository with the same contract: every non-`.github` push to `main` generates `YYYY.M.D.HHMM` in UTC, replaces the single project version in `pyproject.toml`, commits as `github-actions[bot]`, creates `v<version>`, and pushes `main` plus tags. A regression test verifies the required workflow operations and confirms the same version replacement leaves `pyproject.toml` valid.

## 2026-08-11 — functional port

### Shared foundation and toolbox control plane

- Added schema-versioned, typed catalogs with complete OCI references, separate display/container names, explicit feature states, runtime profiles, platform assignments, and backend defaults.
- Added Podman/Docker discovery and Toolbx/Distrobox command construction, including host runtime selection, Podman supplementary-group handling, Docker group-ID translation, and RDMA device handling.
- Added installed-container inspection, explicit checkbox selection, backend/channel filters, Docker Hub tag dates, create/update/enter/delete/default/model-manager actions, and exact mutation previews.
- Kept every interactive terminal takeover inside Textual suspension.

### Backend ports

- llama.cpp: 17 toolbox images, 23 curated repositories, local GGUF/projector discovery, confirmed HF downloads, inference profiles, MTP, load controls, API-key redaction, and server launch.
- DS4: 5 toolbox images, 5 exact artifacts, standalone/coordinator/worker launch, graph and distributed prefill, disk KV cache, SSD expert streaming, MTP, and exact-file HF downloads.
- vLLM: 2 toolbox images and 13 curated Hub policies, non-downloading Hub explorer/cache inventory, policy-aware direct launch, persistent HF/vLLM/Triton/AITER mounts, and guarded compiled-cache reset.
- ComfyUI: 2 toolbox images, 26 workflow/model families, 34 workflow files, in-toolbox model-manager bridge, persistent data mounts, and direct launch with the toolbox alias flags.

Backend server forms and pure command builders live in separate backend source files. vLLM and ComfyUI were not forced through llama.cpp model or server assumptions.

### Packaging and updates

- Distribution name and console command are `ai-toolbox-cockpit`.
- JSON catalogs are included as package data.
- Startup displays the installed package version and checks `kyuz0/ai-toolbox-cockpit` tags in a background worker.
- The update notice gives the exact `pipx upgrade ai-toolbox-cockpit` command.

### Local evidence

- Catalog validation: 26 toolboxes; model sections 23 llama.cpp, 5 DS4, 13 vLLM, and 26 ComfyUI.
- Unit/Textual suite: 54 tests passing after the canonical catalog refresh.
- `compileall`: passing for application and tests.
- No container engine lifecycle command, image pull, model download, backend server, or GPU workload was executed.

- Built `ai_toolbox_cockpit-2026.8.11.1-py3-none-any.whl` and the matching sdist without network or dependency resolution.
- Inspected the wheel: 53 files, both JSON assets, both console entry points, and no stale modules from earlier builds.
- Inspected the sdist: 96 files including architecture/remote-validation docs, TODO, implementation notes, tests, and catalog refresh scripts.
- Installed the wheel with `--no-deps` into an isolated `/tmp` target; package version, both entry points, typed catalogs, and all shipped catalog counts loaded successfully.
- Synced the completed source tree to `/home/kyuz0/Documents/Projects/ai-toolbox-cockpit` while preserving its existing `.git` directory and excluding generated build/cache artifacts.

### Remaining hardware validation

The code paths are implemented but real GPU behavior is not claimed as locally validated. Follow `docs/REMOTE_VALIDATION.md` for the pipx/update test, shared toolbox lifecycle matrix, and incremental llama.cpp, DS4, vLLM, and ComfyUI server/model tests. Start with one known-good image and already-present small model per backend; do not begin with multi-node DS4, DeepSeek-scale vLLM, or a large ComfyUI bundle.
