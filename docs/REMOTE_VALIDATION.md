# Remote container and GPU validation

The local suite validates catalogs, command construction, package contents, and TUI mounting only. It does not prove that an image starts on a particular driver/kernel/firmware combination. Run this matrix on the intended machine one backend at a time and stop at the first failed gate.

Use existing local models and images for the first pass. Do not combine toolbox recreation, model download, and server validation into one test.

## 1. Installation and non-mutating smoke test

1. Install from GitHub with `pipx install git+https://github.com/kyuz0/ai-toolbox-cockpit.git`.
2. Run `ai-toolbox-cockpit` and confirm the installed version appears in the title.
3. Switch through each platform and backend filter. Confirm incompatible images disappear and experimental server choices are labelled.
4. Select an installed toolbox and run **Check Updates**. Confirm the remote date appears without recreating it.
5. Exit and reopen the cockpit. Confirm the selected platform and backend path settings persist.

Record the host OS, kernel, GPU, driver/runtime, Podman/Docker version, Toolbx/Distrobox version, cockpit version, and selected image tag.

## 2. Toolbox lifecycle

Use one disposable stable toolbox with no manual packages inside it.

1. Select it and choose **Create / Update**. Read the complete pull/create command in the confirmation dialog before continuing.
2. Confirm it appears as installed after refresh.
3. Enter it and run only lightweight identity checks such as `id` and the backend binary's `--help`; then exit.
4. Run **Check Updates**. If no newer image exists, confirm **Create / Update** reports that nothing needs changing.
5. Select **Delete**, verify the exact target/command, and confirm only that toolbox is removed.

Repeat separately for Podman+Toolbx and Docker+Distrobox where both are supported. When both engines exist, verify an installed Docker toolbox is entered/deleted through Docker rather than the default Podman engine.

## 3. llama.cpp

Prerequisite: one already-downloaded, known-good GGUF; add a matching `mmproj` only for the later vision pass.

1. Select the intended platform image and local GGUF. Start with default profile, moderate context, no KV quantization, and localhost binding.
2. Verify the command preview, start the server, and query `/health` or `/v1/models` from another terminal.
3. Stop with Ctrl+C and confirm the named server container is gone.
4. Repeat with an API key and verify unauthenticated requests fail while authenticated requests work. Confirm the key never appears in cockpit command output.
5. For a catalogued vision model, select its projector and verify `--mmproj` points inside the read-only model mount.
6. For an MTP model, verify the curated draft/parallel flags and run one short request.
7. Run u-batch calibration for exactly one toolbox/model. Confirm the profile is reused by server and benchmark commands.
8. Run a deliberately tiny benchmark depth curve first; inspect JSONL, stderr, and CSV before scheduling a full curve.

## 4. DS4

Prerequisite: one exact catalogued DS4 GGUF already present in the configured DS4 directory.

1. Start standalone mode with the catalogued defaults and no disk KV/SSD streaming.
2. Verify health/inference, stop with Ctrl+C, and confirm cleanup.
3. Enable disk KV with a dedicated empty `ds4-kv` directory and a small test budget. Confirm writes stay under that directory.
4. Validate SSD streaming separately, first with the model's maintained defaults.
5. Only after standalone passes, use two hosts for coordinator/worker mode. Verify host networking, role, layer ranges, peer address, and distributed prefill settings in both previews before launch.

Do not treat distributed mode as validated from command construction alone.

## 5. vLLM

Prerequisite: a repository already present in the selected Hugging Face cache for the first pass; export `HF_TOKEN` when the repository requires it.

1. Start with `meta-llama/Meta-Llama-3.1-8B-Instruct` and its curated policy. Verify TP, attention, tool-parser, context, dtype, and cache mounts in the preview.
2. Query `/v1/models`, then make one short OpenAI-compatible completion/chat request.
3. Repeat API-key validation and confirm redaction.
4. Stop and restart. Confirm Hugging Face, vLLM, Triton, and AITER paths persist and no model data is written inside the ephemeral container.
5. Validate one policy-specific model at a time: FP8/AWQ, GPT-OSS, Qwen unified attention, then DeepSeek's locked sparse-MLA policy.
6. Test **Reset compiled caches** only with dedicated paths whose directory components contain `vllm`, `triton`, and `aiter`. Confirm the Hugging Face model cache is untouched.

## 6. ComfyUI

Prerequisite: an installed ComfyUI toolbox and one already-installed workflow bundle.

1. Open **Model Manager** and confirm the toolbox's maintained manager starts; exit without downloading during the smoke pass.
2. Verify the bundle table matches available workflow names and variants.
3. Start ComfyUI with the default model/input/output/user directories and toolbox-equivalent flags.
4. Open the web UI, load the known workflow, queue one small generation, and confirm output lands in the configured host output directory.
5. Stop with Ctrl+C, restart, and confirm models, workflows/user data, inputs, and outputs remain available.

After each backend passes, record the exact image digest, model/repository or workflow ID, command preview, health result, cleanup result, and any required deviation from catalog defaults.
