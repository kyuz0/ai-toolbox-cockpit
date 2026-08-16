import json
import tempfile
import unittest
from pathlib import Path

from scripts.extract_llmb_ubatches import extract


class LlmbUbatchExtractorTests(unittest.TestCase):
    def test_extracts_only_calibrated_llama_cpp_and_uses_model_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmarks = root / "local-llm-benchmarks"
            (benchmarks / "catalog").mkdir(parents=True)
            (benchmarks / "results" / "calibrated").mkdir(parents=True)
            (benchmarks / "results" / "forced").mkdir()
            (benchmarks / "catalog/catalog.toml").write_text(
                '[engines.strix-llama-rocm]\nfamily = "llama.cpp"\n'
                'container = "llama-rocm-7.14"\n',
                encoding="utf-8",
            )
            models = root / "models.json"
            models.write_text(
                json.dumps({
                    "backends": {"llama_cpp": {"models": [{
                        "id": "llama-unsloth-lfm2-5-1-2b-instruct-gguf",
                        "repo": "unsloth/LFM2.5-1.2B-Instruct-GGUF",
                    }]}}
                }),
                encoding="utf-8",
            )
            toolboxes = root / "toolboxes.json"
            toolboxes.write_text(
                json.dumps({"toolboxes": [{
                    "id": "strix-halo-llama-rocm-7-14",
                    "backend": "llama_cpp",
                    "container_name": "llama-rocm-7.14",
                }]}),
                encoding="utf-8",
            )
            job = {
                "id": "lfm-calibrated",
                "status": "complete",
                "created_at": "2026-08-09T17:33:09+00:00",
                "subject": {
                    "model_id": "lfm25-1p2b-instruct-bf16",
                    "model_source": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
                    "tokenizer_source": "LiquidAI/LFM2.5-1.2B-Instruct",
                    "quant": "BF16",
                },
                "stack": {
                    "engine_id": "strix-llama-rocm",
                    "engine_name": "llama.cpp",
                },
                "serving": {"config_id": "baseline"},
                "tuning": {
                    "kv_cache_type": "default",
                    "ubatch": {
                        "status": "calibrated",
                        "batch_size": 2048,
                        "selected_ubatch": 2048,
                    },
                },
            }
            (benchmarks / "results/calibrated/job.json").write_text(
                json.dumps(job), encoding="utf-8"
            )
            job["id"] = "lfm-forced"
            job["tuning"]["ubatch"]["status"] = "user-specified"
            (benchmarks / "results/forced/job.json").write_text(
                json.dumps(job), encoding="utf-8"
            )

            records, unmapped = extract(benchmarks, models, toolboxes)

            self.assertEqual(len(records), 1)
            self.assertEqual(
                records[0]["model_id"],
                "llama-unsloth-lfm2-5-1-2b-instruct-gguf",
            )
            self.assertEqual(records[0]["filename_pattern"], "*BF16*.gguf")
            self.assertEqual(records[0]["source_job"], "lfm-calibrated")
            self.assertEqual(unmapped, [])


if __name__ == "__main__":
    unittest.main()
