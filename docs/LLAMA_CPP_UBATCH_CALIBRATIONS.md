# Importing llama.cpp ubatch calibrations

Use only the `local-llm-benchmarks` checkout named by the user. Do not inspect
or import vLLM, DS4, or other backend results.

## Identity rules

An importable result must have both:

- `stack.engine_name == "llama.cpp"`;
- `tuning.ubatch.status == "calibrated"`.

Match the result using the benchmark model artifact, not its tokenizer. The
tokenizer source and revision make benchmark token counts reproducible, but do
not identify the GGUF selected by cockpit.

The complete launch identity is:

1. cockpit llama.cpp model ID;
2. exact cockpit toolbox ID;
3. GGUF quant, represented by `filename_pattern`;
4. benchmark `serving.config_id`;
5. benchmark `tuning.kv_cache_type`, normalized to `default` when absent.

Do not import the engine's `default_ubatch`, a `user-specified` ubatch, or an
ubatch inferred from a similarly named model. Preserve the exact benchmark job
ID and enclosing job status. The referenced `job.json` remains the source for
the engine revision, timestamp, tokenizer provenance, candidate scores, and
failure evidence.

## Quick procedure

1. Read `AGENTS.md` in both repositories and keep the benchmark checkout
   read-only.
2. Run the extractor from this cockpit checkout:

   ```text
   python scripts/extract_llmb_ubatches.py \
     --benchmarks /path/to/local-llm-benchmarks \
     --output /tmp/llama-cpp-ubatches.json
   ```

3. Review every reported mapping. Resolve an `unmapped:` item only with an
   explicit model-ID alias or an exact toolbox container mapping.
4. Copy the reviewed `calibrated_ubatches` array into
   `backends.llama_cpp.config` in `ai_toolbox_cockpit/assets/models.json`.
5. Run `uv run --with pytest python -m pytest tests/test_catalog.py
   tests/test_llama_calibrations.py tests/test_llmb_ubatch_extractor.py
   tests/test_app.py`, then the full test suite.

The extractor never edits either repository. If multiple jobs have the same
complete identity, it emits the newest calibration and retains its job status.
The enclosing job can be `partial` or `failed` because work after calibration
failed; review that status separately while still requiring the ubatch profile
itself to say `calibrated`.

## Cockpit behavior

Cockpit applies a calibration only when the selected GGUF filename, toolbox,
serving configuration, and KV-cache type all match. A baseline calibration does
not silently become an MTP or DSpark calibration. If no exact calibration
exists, cockpit falls back to the model's existing `toolbox_defaults` or leaves
batch and ubatch empty for llama.cpp to choose.
