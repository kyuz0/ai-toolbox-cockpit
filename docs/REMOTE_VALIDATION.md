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

## 4. DS4

Prerequisite: one exact catalogued DS4 GGUF already present in the configured DS4 directory.

1. Start standalone mode with the catalogued defaults and no disk KV/SSD streaming.
2. Verify health/inference, stop with Ctrl+C, and confirm cleanup.
3. Enable disk KV with a dedicated empty `ds4-kv` directory and a small test budget. Confirm writes stay under that directory.
4. Validate SSD streaming separately, first with the model's maintained defaults.
5. Only after standalone passes, use two hosts for coordinator/worker mode. Verify host networking, role, layer ranges, peer address, and distributed prefill settings in both previews before launch.
6. For the catalogued DeepSeek V4.1 Flash Q2, validate the tensor-parallel profile separately. Confirm the standalone default enables SSD streaming with a `92GB` routed-expert cache, the vision encoder is offered, and clearing SSD streaming leaves experts resident. Then select Coordinator/Worker and verify `--tensor-parallel`, TCP transport, the `9911` link port, and, for RoCE, `--rdma-device`, `--rdma-port` and `--rdma-gid-index` in both previews. Confirm the Worker preview runs `ds4` without `--host` or `--port` while the Coordinator preview runs `ds4-server` with both, and that neither preview emits `--layers` or SSD flags. Confirm RoCE without a device is rejected.
7. Verify InfiniBand passthrough on a host that has `/dev/infiniband`: the Podman preview should carry `--device /dev/infiniband` and `--ulimit memlock=-1`, and the Docker preview one `--device` per node under that directory. On a host without InfiniBand, confirm neither appears. Confirm RoCE traffic actually uses the NIC inside the container.

Do not treat distributed mode as validated from command construction alone.

## 5. vLLM

Prerequisite: a repository already present in the selected Hugging Face cache for the first pass; export `HF_TOKEN` when the repository requires it.

1. Start with `meta-llama/Meta-Llama-3.1-8B-Instruct` and its maintained launch defaults. Verify TP, attention, tool-parser, context, dtype, and cache mounts in the preview.
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

## 7. Gufo (Strix Halo only; experimental)

Prerequisite: the published source-pinned Gufo ROCm 10.0 image and one complete catalogued GGUF bundle.

1. Select Gufo under Strix Halo and create or update the experimental toolbox. Enter it, verify `gufo --version` and `gufo --help`, then exit. Confirm the toolbox uses the Gufo image and the Strix Halo ROCm runtime profile.
2. In Models, select one catalogued profile. Confirm the download preview pins the repository revision and every target shard plus its tested MTP or DSpark sidecar. Interrupt and resume a disposable download once, then confirm readiness requires every file's expected size.
3. In Server Mode, confirm a ready MTP or DSpark sidecar selects that speculative mode by default. Select **Disabled (baseline)** and start it on localhost. Confirm the preview runs `gufo serve`, mounts the target directory read-only, passes the advertised context, concurrent sessions, output limit, thinking effort and per-client queue limit, and contains no speculative flags.
4. Query `/v1/models`, `/v1/completions`, and `/v1/chat/completions`, including one streamed request. Stop with Ctrl+C and confirm `gufo-cockpit-server` is removed.
5. Repeat Qwen3.8 Flash Next with MTP-7 and confirm `--speculative mtp`, the read-only `--mtp-model` path, and `--draft-tokens 7`. Repeat DeepSeek V4 Flash with DSpark and confirm the read-only `--dspark-model` path.
6. Record the image digest, Gufo revision, exact target and sidecar revisions, model response, cleanup result, and watcher profile recovery.

## 8. Halogen Flash (Strix Halo only; validation pending)

Use the upstream `latest` image / Qwen3.8-Flash-Next W4B quality pair as the first
case. Record the resolved image digest and server version for each validation
run. Run these phases separately on the user's gfx1151 GPU host. Do not run them
on the development machine.

1. **Image lifecycle:** select Halogen under Strix Halo and confirm Create /
   Update previews only `podman pull` or `docker pull`. Pull it, refresh, and
   confirm Image ready appears. Enter must direct the user to Server Mode;
   no Toolbx/Distrobox container should be created. Repeat Create / Update to
   verify an already-pulled `ghcr.io/peonist-ai/halogen-flash-server:latest` image
   can be refreshed.
2. **Model preparation:** if the quality bundle is not already available,
   download it from Models as a separate operation. Use `~/halogen-models` or
   save a dedicated path. Confirm the preview includes the pinned revision,
   checkpoint, quality overlay, and tokenizer files. Verify readiness after
   downloading; confirm the saved directory also appears in Server Mode.
   Test an interrupted download/resume separately if needed.
3. **Serving:** select the ready quality bundle and default settings with
   localhost binding. Verify individual read-only selected bundle file mounts
   under `/models`, GPU devices, `memlock`/IPC settings, selected overlay,
   `--network=none`, dropped NET_ADMIN/NET_RAW, `no-new-privileges`, and the host
   API relay address in the preview. No `-p` publishing should appear. Podman
   uses `keep-groups`; Docker passes the host GIDs of `video` and `render`.
   The image's entrypoint must remain intact. Confirm `--pull=always` appears
   in the launch preview and the engine checks the registry even with a cached
   image. A failed pull must stop the launch. Start and allow the cold load to finish.
4. From another terminal, query `http://127.0.0.1:8731/health` and
   `/v1/models`, then send one short chat request to `/v1/chat/completions`
   using the returned model ID. Record the startup precision message and
   successful output. Also test streamed chat responses, concurrent requests,
   client disconnects, and a large image request when validating vision later.
   Only the host relay should listen on 8731; neither container port should be
   published. Confirm the image provides `python3` for the exec stream helper.
   Inspect effective mounts and network configuration. Inside the running
   container, verify only loopback is available and attempted connections to
   controlled Internet, LAN, host, and DNS endpoints fail over IPv4 and IPv6.
   Verify a dedicated test file outside the selected bundle is not visible and
   selected mounts reject writes. Test Podman and Docker separately. An occupied
   host API port must fail startup without leaving a server container running.
5. Stop with Ctrl+C, including during a streaming request. Confirm
   `ai-toolbox-cockpit-halogen-server` is removed, the relay listener closes,
   and its exec clients terminate. Also check cleanup after startup failure,
   reopen Cockpit, and confirm path/server settings persist. Restart and verify
   the same bundle serves without a download. Test the speed overlay only
   after the quality pair passes, as a separate case.
6. If testing Delete, confirm its target is the image, not a toolbox or models
   directory. Model files must remain. Image removal may refuse while other
   containers use it; Cockpit does not force removal.

Record the image digest, model revision, OS/kernel/GPU, engine version, command
preview, startup time, health/chat results, and Ctrl+C cleanup. Do not mark the
integration supported from local command/UI tests alone.
