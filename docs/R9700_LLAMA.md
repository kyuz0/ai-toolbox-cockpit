# R9700 llama.cpp profiles

For native Affinity, hipfire and Strata models, see [DeepSeek on four R9700 GPUs](R9700_DEEPSEEK.md).

Install **llama.cpp · ROCm 10.0** on the R9700 platform. Select the model in Server Mode, then choose its matching launch profile. Each profile selects its tested GPU devices, layer placement, F16 KV cache and target-only decoding. The four-card profiles start with an 8,192-token context and one sequence.

| Model / quant | GPUs | Model repository revision | Matching projector |
| --- | ---: | --- | --- |
| Qwen3.8-27B UD-Q4_K_XL | 1 or 4 | `unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502` | `mmproj-BF16.gguf` |
| Qwen3.8-Flash-Next UD-Q4_K_XL | 4 | `unsloth/Qwen3.8-Flash-Next-GGUF@766911a6b7369840a91dbcd95f9f997acaab6cd6` | `mmproj-BF16.gguf` |
| DeepSeek V4 Flash Vision Exp UD-Q2_K_XL | 4 | `unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF@87ce15810931a270f165e5e50439cd446f6b4ba5` | `mmproj-BF16.gguf` |

Keep every decoder shard and download the projector separately from the same repository revision. Cockpit finds a shared projector beside the quantization directory as well as beside a decoder file. Select it explicitly in the **Projector** control; leaving that control empty starts text-only serving. The Flash Next set has four decoder shards and DeepSeek Vision has three. These profiles were qualified with llama.cpp build 11872, source `344be0a60074e6395da3c4a4f300f7361df5a67d`.

The four-card Qwen27 Q4XL profile uses calibrated ubatch 256, batch 2048, F16 KV and equal layer splitting across devices 0,1,2,3. Its everyday allocation starts one sequence at 8,192 tokens. The allocation selector also offers one, four, eight or sixteen sequences with 68,096 context tokens per sequence, reserving 68,096, 272,384, 544,768 or 1,089,536 tokens in total. Select the allocation that matches the intended simultaneous requests; the client sets request concurrency separately. These choices use the same Qwen27 decoder and BF16 projector download.

The Flash loader also maps a 26.82 GiB per-layer token-embedding table from disk; OS page-cache residency varies with workload. These configurations were qualified on a 64 GB host.

All three models passed sampled text, thinking, image input and required tool calls. DeepSeek Vision also passed two-image input with reversed image order. For Qwen, the sampled thinking profile uses temperature 1.0, top-p 0.95, top-k 20 and min-p 0; direct answers use temperature 0.7, top-p 0.8 and presence penalty 1.5. DeepSeek's sampled profiles use temperature 1.0, top-p 0.95, top-k 0 and min-p 0, with explicit maximum-effort thinking or disabled reasoning. Server defaults and client request settings are separate; clients can override sampling.

## Measured depth profiles

The standard raw-completion workload appends 2,048 new tokens, generates exactly 256 tokens and measures two repeats at each of nine history depths through 64K. These deterministic speed measurements are separate from the sampled feature checks. Rates are tokens/s on a Threadripper 9970X / 64 GB host with four R9700s installed, using verified throughput-performance and automatic fans under an explicit benchmark-policy exception.

| Model / GPUs | Calibrated ubatch | PP / TG at 0 | PP / TG at 64K |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B Q4XL / 1 | 512 | 1086.35 / 29.46 | 690.19 / 24.28 |
| Flash Next Q4XL / 4 | 2048 | 1617.46 / 34.10 | 890.61 / 29.64 |
| DeepSeek Vision Q2XL / 4 | 512 | 659.02 / 26.71 | 148.34 / 20.89 |

All three curves use batch 2048, F16 KV and one sequence. The measured total context allocation is 68,096 tokens. Flash Next and DeepSeek Vision expose this as a separate measured allocation; their normal launches start at 8,192 tokens. The earlier Qwen27 episode allocations remain available alongside the current calibrated one-card default. See the [complete speed viewer](https://local-llm-benchmarks.dev/speed.html) for all depths and repeat variation.

The Qwen target-only frozen panels completed all 100 short and 12 long-context cases for each artifact. NLL is per local target token; lower is better.

| Artifact | Short100 NLL | Long12 NLL | Short / long target tokens |
| --- | ---: | ---: | ---: |
| Qwen27 Q4XL | 0.279769 | 0.099542 | 5,836 / 764 |
| Flash Next Q4XL | 0.284752 | 0.126600 | 5,696 / 766 |

Hosted whole-continuation comparisons include 99 Qwen27 short cases, 98 Flash short cases and all 12 long cases for both. The excluded hosted token bytes do not reconstruct their saved response text; local panels remain complete. Provider Qwen precision is undisclosed, and the Flash endpoint is described as based on Flash Next. These scores do not establish original-weight logit parity.
