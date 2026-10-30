# Qwen3.8 on Intel Arc Pro B70

Select **Intel Arc B70** as the platform. These profiles target 32 GB Arc Pro B70 cards on Linux with a working Intel GPU driver and Podman/Toolbx or Docker/Distrobox. The engine libraries are packaged in the images; no host oneAPI installation is needed.

## Install and choose an engine

Upgrade Cockpit with `pipx upgrade ai-toolbox-cockpit`, then open **Toolboxes** and use **Create / Update** for the engines you want. Source recipes and build workflows are maintained in [intel-b70-ai-toolboxes](https://github.com/kyuz0/intel-b70-ai-toolboxes).

| Engine in Cockpit | Model / quantization | B70 cards | Context allocation | Decoding |
| --- | --- | --- | --- | --- |
| B70 upstream vLLM | RedHatAI Qwen3.8 27B INT4 | 1 or 2 | 67,840 | Baseline or embedded MTP3; graphs |
| B70 upstream vLLM | Official Qwen3.8 27B FP8 | 2 | 67,840 | Baseline or embedded MTP3; eager |
| b70-llama-vulkan | Unsloth Qwen3.8 27B UD-Q4_K_XL | 1 | 67,840 | Baseline or embedded MTP3; F16 KV |
| b70-llama-sycl | Unsloth Qwen3.8 27B UD-Q4_K_XL | 1 | 67,840 | Baseline or embedded MTP3; F16 KV; optimized weight path disabled |
| B70 Intel LLM Scaler | RedHatAI Qwen3.8 27B INT4 | 1 | 67,840 | Baseline, eager |
| B70 EXL3 XPU | turboderp Qwen3.8 27B EXL3 4.00 bpw | 1 | 67,840 | Embedded MTP3; FP8 KV; prefix caching off |
| B70 Optimization Lab FP8 | Official Qwen3.8 27B FP8 | 1 | 32,768 | MTP5 with draft-only INT4 shortlist; CPU embeddings; prefix caching off |
| llama.cpp SYCL Flash Next (patched) | AtomicChat Flash Next AD4.27 | 2 | 68,096 | Baseline or separate Q4 MTP3; Q8 K / Q4 V cache |
| Strata B70 | AtomicChat Flash Next AD4.27 native pack | 2 | 68,096 | Baseline or separate official MTP converted to Q2 |

All presets use one active sequence. Context allocation includes the prompt and generated answer; 67,840 positions leave room beyond a 65,536-token conversation history. The single-card Optimization Lab profile has a smaller context limit and allocates 97.5% of GPU memory. Close other GPU workloads before starting it.

EXL3, Optimization Lab, Strata and the patched Flash Next engine are explicitly experimental. EXL3 and Optimization Lab reprocess conversation history because their profiles disable prefix caching. Their speed should not be interpreted as a working-cache replacement for the upstream profile. Optimization Lab packages a pinned publisher binary and its matching Python overlays.

## Qwen3.8 27B

1. In **Models**, select the named engine and download its curated checkpoint. Upstream and Scaler share the RedHatAI INT4 download; upstream and Optimization Lab share the official FP8 download. Revisions and required files are pinned in the catalogue.
2. For llama.cpp, choose **Tested B70 models**, then the Unsloth 27B **UD-Q4_K_XL** quant. This GGUF already contains the MTP head.
3. In **Server Mode**, select the same engine and model. Use **Discover GPUs** and choose the B70 indices reported by that engine. Vulkan and SYCL indices can differ; do not assume the sample device number is correct on another computer.
4. For upstream vLLM, choose tensor parallel size **1** or **2** for INT4, or **2** for official FP8, and select the matching number of Intel device indices. Choose **MTP** for the speculative configuration shown in the episode. Scaler exposes the baseline configuration.
5. Start the server. Cockpit shows the generated command before launch. The endpoint binds to localhost by default; Ctrl+C stops the server and returns to Cockpit.

The episode’s two-card llama.cpp baseline controls use the same 27B GGUF and F16 KV cache. Disable MTP, select both B70 device indices, and add `--split-mode layer --tensor-split 1,1` in **Extra args**. Keep the other 27B defaults. Two-card MTP is not part of those controls.

For the upstream INT4 throughput configurations, select TP **2**, disable MTP, and set **Max sequences** to **8** or **16**. The normal interactive preset remains one sequence.

## Flash Next with patched llama.cpp

Install **llama.cpp SYCL Flash Next (patched)**. In **Models → llama.cpp → Tested B70 models**, download the AtomicChat **AD-4.27bpw-Q4_K_M-M64** quant and the separate **Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf** sidecar. The main model has 33 shards, totalling about 94.5 GB; the sidecar is about 2.6 GB.

In Server Mode, select the AtomicChat model and patched engine. The profile uses both B70s with layer splitting and places the draft on the second SYCL device. Select the Q4 MTP sidecar for MTP3, or uncheck MTP for the baseline. This profile is distinct from the standard 27B SYCL image.

## Flash Next with Strata

Install **Strata B70**. In **Models → Strata**, choose a models directory, select **Download / Repair**, then **Prepare for Strata**. Preparation builds the native pack, downloads the pinned official MTP tensors (about 5.2 GB), and creates the Q2 draft runtime. The original GGUF shards remain in place. Allow additional disk space for the generated files; the dual-card Flash Next configuration was tested with 64 GB of host RAM.

When the status says **Ready for Strata**, open **Server Mode → Strata** and select baseline or MTP. Strata uses both B70 cards. Its native pack and draft runtime are separate from llama.cpp's Q4 draft file.

To use both Flash Next engines without downloading the main shards twice, download them through Strata with its models directory set to the same parent directory used by llama.cpp (normally `~/models`). Then use **Scan Local** in llama.cpp and select those existing AtomicChat shards; download only the separate Q4 MTP sidecar there.

## Sources

The release combines [upstream vLLM](https://github.com/vllm-project/vllm), [llama.cpp](https://github.com/ggml-org/llama.cpp), [EXL3 XPU](https://github.com/sybil-solutions/exl3xpu), and the [B70 Optimization Lab](https://github.com/steveseguin/b70-optimization-lab) with the pinned build recipes in the [B70 toolbox repository](https://github.com/kyuz0/intel-b70-ai-toolboxes/tree/main/toolboxes). Each recipe records source or publisher-image provenance. The model selector supplies the compatible quantization and sidecars for each engine.
