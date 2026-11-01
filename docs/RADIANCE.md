# Standalone Radiance on R9700

Select **AMD Radeon AI PRO R9700 → Standalone Radiance** in Toolboxes and install the ROCm 10.1 image. Models and Server Mode use Radiance’s native `.rad` format, including its tokenizer, vision tower and bundled drafter. The separate **vLLM Radiance (legacy)** entry keeps its earlier checkpoint format and launch controls.

## Models and GPU layouts

| Native artifact | Download | GPU layout | Bundled decoding |
| --- | ---: | --- | --- |
| Qwen3.8-27B FP8 | 30.78 GB | 2 or 4 R9700s | DFlash2, 7 draft tokens |
| Qwen3.8-27B MXFP4 | 19.38 GB | 1 or 4 R9700s | DFlash2, 7 draft tokens |
| Qwen3.8-Flash-Next IQ4R | 121.97 GB | 4 R9700s | MTP, 3 draft tokens |

The model manager pins each repository revision and downloads into `~/models/radiance`. **Download / Repair** uses the saved Hugging Face authentication. **Scan Local** checks complete file sizes; acquisition integrity comes from the Hugging Face client. A model can also sit directly in the chosen models directory.

Flash Next keeps its 47.68 GiB n-gram table in pinned host RAM. Its selected profile passes `--ngram-placement ram`; allow about 54 GiB of available host RAM at startup, including loading headroom. The main weights and experts use the four GPU pools; a small token-embedding pool also uses pinned host memory. The download size is not the VRAM requirement.

## Start and stop

Choose the native model and **Serving profile** in Server Mode. The default profile retains the original GPU layout. Both Qwen3.8-27B artifacts also offer **4 R9700 GPUs · up to 16 sequences**, using the same downloaded file. Cockpit selects the qualified GPU count, FP8 KV cache, exact inter-GPU communication and GPU weight placement. Start with the default 8,192-token context and a server sequence capacity of one. The context field includes prompt and output tokens and is bounded at 67,840 for these profiles.

The four-GPU Qwen27 profile reserves 144 host-memory state checkpoints (`--checkpoint-slots 144`), requiring about 20.65 GiB of host RAM in addition to other process memory. Model weights remain on the GPUs. Select a server sequence capacity from 1 to 16 and choose Target only or Bundled draft for baseline or DFlash2-7. Client concurrency is separate from server sequence capacity; the checkpoint budget does not guarantee full history reuse at every concurrency/depth.

Flash Next also supports server capacities up to 16. Client concurrency is a separate client setting. The complete C4/C8/C16 speed curves used a server capacity of 16, FP8 KV and contexts through 64K history. With MTP3 at 64K, four active requests reached 477.78 aggregate tokens/s and retained all history; eight reached 186.26 and sixteen 41.61, with increasing history replay. Use one or four active requests for deep-context work; more short-context requests can improve aggregate throughput. Sampled C8/C16 checks also verified concurrent answers and accepted draft tokens on every request.

Select **Target only** or the model’s **Bundled draft** mode. Choose image input and thinking effort explicitly. For Qwen sampled thinking requests, use temperature 1.0, top-p 0.95, top-k 20 and min-p 0. Non-thinking requests use temperature 0.7, top-p 0.8, top-k 20, min-p 0 and presence penalty 1.5. Client request settings take precedence over server defaults.

**Start Radiance** shows the complete command for review, mounts only the selected model file read-only and binds the API to loopback by default. Ctrl+C stops the foreground server and returns to Cockpit. The API includes `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/tokenize` and `/metrics`.

The source-built image uses the supplied ROCm compiler. See the [R9700 toolbox guide](https://github.com/kyuz0/amd-r9700-ai-toolboxes/blob/main/docs/radiance-rocm-10.1.md) for the pinned source, build workflow and measured profiles.

## Frozen-continuation quality

The published image completed short100 and long12 panels for each artifact with target-only scoring. NLL is per local target token; lower is better. Hosted Qwen precision is undisclosed and the Flash endpoint is described as based on Flash Next, so these scores do not establish original-weight logit parity.

| Native artifact | Short100 NLL | Long12 NLL | Local target tokens, short / long |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B FP8 | 0.281940 | 0.099176 | 5,836 / 764 |
| Qwen3.8-27B MXFP4 | 0.318670 | 0.147677 | 5,836 / 764 |
| Flash Next FP8/IQ4R | 0.288974 | 0.130511 | 5,696 / 766 |
