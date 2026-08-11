# AI Toolbox Cockpit implementation plan

This file is the execution checklist for the functional port. Items are checked only after code and safe validation are complete. Local implementation is complete; hardware/runtime outcomes remain governed by `docs/REMOTE_VALIDATION.md`.

## Safety boundary

- [x] Do not pull or download container images in this development environment.
- [x] Do not create, run, enter, update, or delete containers while developing or testing.
- [x] Do not download models or workflow bundles.
- [x] Do not execute GPU workloads or backend servers.
- [x] Record remote-machine test procedures for operations that cannot be exercised here.

## 1. Audit and architecture

- [x] Audit the existing Llama Toolboxes Cockpit implementation and tests.
- [x] Audit the DS4 cockpit implementation and model defaults.
- [x] Audit the vLLM toolbox launch scripts and curated model policy.
- [x] Audit the ComfyUI toolbox launch scripts and toolbox model manager.
- [x] Establish explicit backend IDs: `llama_cpp`, `vllm`, `comfyui`, and `ds4`.
- [x] Keep the app shell thin and put backend-specific server/model behavior in its backend package.
- [x] Extend the catalog schema so runtime container names, toolbox implementation, image references, and feature support are unambiguous.

## 2. Shared application foundation

- [x] Port searchable selectors and confirmation/progress widgets.
- [x] Port persistent settings and add settings needed by every backend.
- [x] Implement container-engine discovery without invoking containers.
- [x] Implement common Podman/Docker and Toolbx/Distrobox capability discovery.
- [x] Implement safe subprocess/result abstractions and command previews.
- [x] Implement Docker Hub image metadata/update checks.
- [x] Add common notifications, background workers, and error handling.

## 3. Unified toolbox management

- [x] Display platform toolboxes with backend/channel filtering.
- [x] Detect installed toolbox containers across supported engines.
- [x] Preserve explicit checkbox selection independent of the table cursor.
- [x] Create selected toolboxes with compatible Toolbx/Distrobox and engine combinations.
- [x] Compare local creation time against registry image update time.
- [x] Confirm and recreate only toolboxes that need an update.
- [x] Enter exactly one selected toolbox interactively while the TUI is suspended.
- [x] Confirm and delete selected toolboxes.
- [x] Support opening each toolbox's backend-provided model manager where applicable.
- [x] Unit-test command construction and state decisions without executing container commands.

## 4. llama.cpp backend parity

- [x] Port the complete curated GGUF model catalog and metadata.
- [x] Port local GGUF/projector discovery and filtering.
- [x] Port Hugging Face GGUF download planning and user flow.
- [x] Port vision-projector handling.
- [x] Port inference profiles and MTP controls.
- [x] Port server command construction, all current launch controls, and load modes.
- [x] Port API-key handling and persistence behavior.
- [x] Port the benchmark matrix UI and command runner.
- [x] Port u-batch calibration, persistent profiles, and calibration CLI.
- [x] Port and expand existing llama.cpp unit tests.

## 5. DS4 backend parity

- [x] Port the DS4 model catalog and local model discovery.
- [x] Port server defaults including prefill cap/chunk and optional disk KV cache.
- [x] Port single-node and multi-node command construction.
- [x] Port DS4 toolbox lifecycle details not covered by the common runtime.
- [x] Port and expand DS4 command tests.

## 6. vLLM backend

- [x] Import the toolbox's curated Hugging Face model policy into `models.json`.
- [x] Implement a searchable Hugging Face model/catalog panel without pre-downloading models.
- [x] Show local Hugging Face cache status without mutating it.
- [x] Implement vLLM server controls and command construction.
- [x] Preserve per-model environment variables, attention backend, context, sequence, eager, and extra-arg policy.
- [x] Persist Hugging Face, Triton, and AITER caches through explicit mounts.
- [x] Unit-test representative default, AWQ, Gemma, GPT-OSS, Qwen, and DeepSeek launch commands.
- [x] Mark remote GPU validation as required until exercised on supported hardware.

## 7. ComfyUI backend

- [x] Import workflow/model-bundle metadata into `models.json`.
- [x] Implement the workflow bundle browser.
- [x] Bridge bundle installation to the toolbox-provided `model_manager` command.
- [x] Implement ComfyUI server controls and direct container command construction.
- [x] Preserve model, input, output, and user-directory mounts.
- [x] Port required launch flags from the toolbox startup alias.
- [x] Unit-test server and model-manager commands without execution.
- [x] Mark remote GPU/workflow validation as required until exercised on supported hardware.

## 8. Unified UI integration

- [x] Make platform selection global and refresh every view consistently.
- [x] Keep toolbox, model, server, and benchmark views focused and backend-filterable.
- [x] Show only supported/experimental features and explain unavailable ones clearly.
- [x] Add command previews before mutating or launching operations.
- [x] Verify narrow terminal behavior and keyboard navigation with Textual pilot tests.

## 9. Packaging, installation, and application updates

- [x] Package all JSON assets and expose the `ai-toolbox-cockpit` CLI.
- [x] Expose the u-batch calibration CLI under the new package name.
- [x] Support `pipx install git+https://github.com/kyuz0/ai-toolbox-cockpit.git`.
- [x] Display the installed package version in the UI.
- [x] Check GitHub releases/tags for application updates at startup without blocking launch.
- [x] Give the exact `pipx upgrade ai-toolbox-cockpit` command when an update is available.
- [x] Add packaging smoke tests using a locally built wheel; do not access the network.

## 10. Documentation and release validation

- [x] Rewrite README around actual implemented capabilities and workflows.
- [x] Document architecture, catalog schema, and backend extension contract.
- [x] Document local safe tests and remote container/GPU test matrix.
- [x] Run formatting/static checks where configured.
- [x] Run all unit and Textual UI tests.
- [x] Build and inspect wheel/sdist contents locally.
- [x] Sync the completed implementation to `/home/kyuz0/Documents/Projects/ai-toolbox-cockpit`.
