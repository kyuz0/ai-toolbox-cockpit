# DeepSeek on four Radeon AI PRO R9700 GPUs

Select **R9700** as the platform, install the engine toolbox, then choose that engine in **Models** and **Server Mode**. Each engine uses its own pinned artifact set. The server screen shows the model, decoding mode, context and exact launch command before starting. Press Ctrl+C to stop the server and return to Cockpit.

| Engine | Curated model | Format | Four-GPU decoding profile |
| --- | --- | --- | --- |
| Affinity | DeepSeek-V4-Flash-0731, custom 2.875 bpw | Assembled `.aff` target and separate `.dspark.aff` | Target-only or DSpark; one active request; target experts in VRAM |
| hipfire | DeepSeek-V4-Flash-0731, MQ2R | Native `.mq2r` | Target-only; one active request; F32 KV cache |
| Strata | DeepSeek-V4.1-Flash-1017, GSQ-RCO IQ2_XXS 2.00 bpw | Four GGUF backbone shards, Engram and prepared native metadata | Target-only or DSpark; one active request; expert offload to RAM/SSD |

These are text profiles. Their server APIs and native timing protocols differ; use a benchmark's model, quantization, request count and cache methodology when comparing results. Low-bit model compression also affects quality. A generation rate alone does not establish quality equivalence.

## Affinity

**Models → Affinity → Download / Repair** downloads three ordered parts from [the 2.875 bpw repository](https://huggingface.co/neuromaniacmd/DeepSeek-V4-Flash-0731-affinity-2.875bpw), assembles `model-0731.aff`, and downloads the matched DSpark sidecar and [0731 tokenizer](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731). Downloads are pinned to exact revisions. A complete assembled target is reused without downloading its parts again. Interrupted preparation leaves an existing target intact; downloaded parts remain available for another attempt.

The assembled target is 108,506,644,480 bytes and the DSpark sidecar is 7,549,747,200 bytes. A fresh download and assembly need approximately 209.2 GiB for retained parts, target, drafter and tokenizer, plus the container and optional growing prefix cache. The confirmation shows the additional space needed for the files currently missing.

**Server Mode → Affinity** uses all four cards, static expert placement, the native HTTP-v2 interface and one sequence. Select target-only or DSpark. Context starts at 8,192 tokens; the allocation includes 64 additional native KV positions. This profile does not enable concurrent slots or RAM/SSD target-expert streaming. Confirm full expert residency in startup output before using a larger context.

Prefix snapshots live under `.prefix-cache` in the selected Affinity model directory, with a 64 GiB limit. This cache stores attention state on disk. Model weights are mounted read-only and the cache has a separate writable mount. The API binds to localhost by default; the server's displayed port is published from the container.

## hipfire

**Models → hipfire → Download / Repair** downloads the pinned [DeepSeek-V4-Flash-0731 MQ2R](https://huggingface.co/hipfire-models/deepseek-v4-flash-0731) file, 82,191,359,851 bytes. This profile needs no GGUF conversion or assembled target.

**Server Mode → hipfire** launches the native `hipfire serve` command with four GPUs, the legacy F32 KV backend, a single batching lane and speculation disabled. Context starts at 8,192 tokens. The model is mounted read-only; `.hipfire-state` in the selected model directory holds writable runtime and kernel-cache files. The configured local model is used without fetching a remote model registry.

The four-GPU profile does not expose a DSpark selector. Its native full-prompt benchmark measurements should retain that methodology when presented alongside standard cached-history HTTP results.

## Strata

**Models → Strata → Download / Repair** downloads the pinned [GSQ-RCO backbone and Engram](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF), plus the matched DSpark drafter and small metadata sidecar from [the Unsloth repository](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF). The complete selected set is 256,816,889,805 bytes, approximately 239.2 GiB. All seven files are included in the displayed download total.

Then choose the R9700 Strata image and **Prepare for Strata**. Preparation creates tokenizer files, Engram lookup metadata, tensor offsets and the 2.00 bpw expert profile under `.prepared` in the model directory. It checks the resulting pack with the native executable. Original GGUF weights are mounted read-only; only the preparation output is writable. Preparation refuses to overwrite an existing pack.

**Server Mode → Strata** selects target-only or DSpark and defaults to an 8,192-token context. The HTTP wrapper uses all four GPUs, full six-expert routing, a single sequence and the upstream DeepSeek V4.1 chat/template implementation. The engine's `--tp 1` setting is part of this four-GPU placement recipe; it does not mean only one GPU is used. This model exceeds aggregate VRAM, so expert offload remains part of the configuration. Keep the pack and its source model directories together under the configured model root.

Choosing the Intel B70 platform instead restores Strata's separate Flash Next AtomicChat model, preparation and MTP controls.
