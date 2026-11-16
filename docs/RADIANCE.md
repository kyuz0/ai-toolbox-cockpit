# Radiance on four R9700s

Select **AMD Radeon AI PRO R9700 → Radiance** in Toolboxes and install or update the ROCm 10.1 image. Radiance 2.1 uses native YAML configuration; Cockpit generates and checks it before starting the server.

## Model

The current model is **Qwen3.8-Flash-Next IQR-V4** on four R9700 GPUs, with exact inter-GPU transport, i8 KV cache and native MTP speculative decoding.

| Artifact | Download | Pinned revision |
| --- | ---: | --- |
| [qwen3.8-next-flash-iqr-v4.rad](https://huggingface.co/StillDeadcode/qwen3.8-next-flash-iqr-v4) | 121,678,372,864 bytes | `679c943216e5a33da5f458b5da378586219e874b` |

The `.rad` file includes the tokenizer, draft tensors and vision tower. Model Manager downloads into `~/models/radiance`; **Download / Repair** uses the saved Hugging Face authentication and **Scan Local** checks the complete file size. Acquisition integrity comes from the Hugging Face client. A complete file can also sit directly in the selected models directory.

## Serving profiles

| Profile | Context tokens | Concurrent requests | Placement |
| --- | ---: | ---: | --- |
| 256K, disk n-grams | 262,144 | 1 by default; up to 16 | GPU expert weights, disk n-grams, 144 host-memory recurrent snapshots |
| 256K, RAM n-grams | 262,144 | 1 by default; up to 16 | GPU expert weights and embeddings, RAM n-grams, 48 tip-only recurrent snapshots |
| 1M, one request | 1,000,000 | 1 configured slot | GPU expert weights, disk n-grams, 144 host-memory recurrent snapshots, YaRN `4:262144` |

The disk profile reserves about 15.93 GiB of host memory for recurrent snapshots. It left approximately 37 GiB available on the tested 64 GB host. The model file's download size is not its VRAM requirement. The RAM profile holds about 47.7 GiB of n-grams and 5.31 GiB of snapshots in host memory; measured available memory fell to about 1 GiB. Choose disk n-grams if other applications need that RAM.

With MTP3 and 16 configured slots, the native profile retained all histories across the complete 0–64K speed curves at 1, 4, 8, 10 and 16 simultaneous requests. At 64K, four requests reached 459 aggregate generation tokens/s, eight reached 523 and sixteen reached 574. Both prompt-processing and generation curves are documented in the [R9700 toolbox guide](https://github.com/kyuz0/amd-r9700-ai-toolboxes/blob/main/docs/radiance-rocm-10.1.md).

Full-window capacity differs from those speed curves. At 262,144 tokens per session, four active sessions retained all history; eight, ten and sixteen completed with substantial recomputation and longer waits. Five through seven active sessions have not been tested. Keep the native profile's 16 configured slots when reproducing its cache allocation; the number of active requests is controlled by clients.

The single-user profile processed 999,744 input tokens and generated 256 output tokens using the model owner's [documented YaRN extension](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/README.md). This demonstrates input capacity, not million-token answer quality or concurrent 1M sessions.

The RAM profile was faster on the measured 0–64K curves. At 64K, aggregate prompt processing / generation reached 7,238 / 332 tokens/s for one request, 7,043 / 523 for four, 7,052 / 546 for eight, 7,092 / 561 for ten and 7,101 / 588 for sixteen. It retained every measured history through 64K. At the full native window, four active sessions retained all history, but eight retained only four histories per batch and reached a 139-second longest first-token wait, compared with 106 seconds for disk. These profiles differ in snapshot policy and embedding placement as well as n-grams.

## Start and stop

Choose the model and **Serving profile** in Server Mode. **Concurrent requests** defaults to one; increase it when serving multiple clients. **Context tokens** includes prompt and generated tokens. **Context scaling** is `none` for native context and `yarn:4:262144` for the tested extension. **Profile default** restores the selected profile's value; explicit saved values are preserved.

**Speculative decoding** offers **Disabled** and **MTP**. MTP defaults to three **Draft tokens**; **Model default** restores that value. Choose **Image input** and **Thinking effort** for the intended workload. The measured text recipes kept the vision tower enabled.

**Start Radiance** shows the complete command for review, mounts the selected model file read-only and binds the API to loopback by default. Ctrl+C stops the foreground server and returns to Cockpit. API endpoints include `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/tokenize` and `/metrics`.

The [R9700 toolbox guide](https://github.com/kyuz0/amd-r9700-ai-toolboxes/blob/main/docs/radiance-rocm-10.1.md) provides the exact source revision, launch examples, paired performance measurements and build workflow.

The API model ID for the IQR-V4 profiles is `radiance-qwen38-flash-iqrv4-679c9432`.
