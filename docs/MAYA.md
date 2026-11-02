# Maya on Strix Halo

Select **AMD Strix Halo → Maya** in Toolboxes. The experimental ROCm 10.0 image follows stable Project Maya releases. Install or Update uses the maintained Strix Halo image; `maya-version` reports its exact source revision.

## Models

| Model | Weight bundle | Native speculation |
| --- | --- | --- |
| GLM-5.3-Flash Maya-S24 | 94.7 GB; IQ2/IQ3 routed experts, Q4_K attention/shared experts, selected Q6_K/Q8_0/F32 tensors | Included MTP layer, up to three draft tokens |
| DeepSeek-V4.1-Flash-1017 Maya-S | 339.5 GB, including 161.3 GB Q6_K Engram tables; IQ2/IQ3 routed experts and Q8_0 dense weights | Separate matched DSpark files, about 8 GB; up to five draft tokens |

In **Models**, select the model and destination, choose **Download / Repair**, then **Prepare for Maya**. Downloads pin the tested repository revision and complete shard/sidecar set. Preparation writes a native pack next to the original weights; keep the source files. Packs reference absolute source paths, so prepare again after moving the model directory.

Allow space for the complete bundle and generated metadata/dense tensors. DeepSeek-S exceeds 128 GB of physical memory and uses NVMe-backed Engram and expert reads. Its storage requirement does not imply all weights are resident in RAM.

## Serving

In **Server Mode**, select the model and **MTP** or **DSpark**. **Target only** disables speculation. This profile serves one sequence at a time, with 2,048–68,096 context tokens and localhost binding by default. GLM uses temperature 1, top-p 0.95 and top-k 64; DeepSeek uses temperature 1, top-p 1 and top-k 64. Both select high reasoning effort. Clients may override supported request settings.

The preparation and server commands mount the model directory at the same absolute path. Server Mode starts an inspectable foreground container; Ctrl+C stops it. Updating the toolbox preserves downloaded models and generated packs. Recorded benchmarks identify their measured Maya version; a rolling-image update does not change those historical results.
