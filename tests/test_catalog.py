import json
import unittest
from importlib.resources import files
from unittest.mock import patch

from ai_toolbox_cockpit.catalog import load_model_catalog, load_toolbox_catalog
from ai_toolbox_cockpit.catalog.schema import CatalogError, ModelCatalog, ToolboxCatalog


class CatalogTests(unittest.TestCase):
    @staticmethod
    def asset(name: str) -> dict:
        return json.loads(files("ai_toolbox_cockpit.assets").joinpath(name).read_text(encoding="utf-8"))

    def test_toolbox_catalog_loads_and_has_full_image_references(self) -> None:
        catalog = load_toolbox_catalog()
        self.assertEqual(catalog.schema_version, 3)
        self.assertEqual({platform.id for platform in catalog.platforms}, {"strix-halo", "r9700", "gb10", "intel-b70"})
        for toolbox in catalog.toolboxes.values():
            self.assertIn("/", toolbox.image)
            self.assertTrue(":" in toolbox.image or "@" in toolbox.image)

    def test_strix_halo_spans_multiple_image_repositories(self) -> None:
        catalog = load_toolbox_catalog()
        repositories = {
            toolbox.image.rsplit(":", 1)[0]
            for toolbox in catalog.platform_toolboxes("strix-halo")
        }
        self.assertGreaterEqual(len(repositories), 4)

    def test_model_catalog_preserves_backend_semantics(self) -> None:
        catalog = load_model_catalog()
        self.assertEqual(catalog.backends["llama_cpp"].kind, "gguf")
        self.assertEqual(catalog.backends["vllm"].kind, "hf_repository")
        self.assertEqual(catalog.backends["comfyui"].entries_key, "bundles")
        self.assertEqual(catalog.backends["ds4"].kind, "gguf_file")

    def test_ds4_catalog_contains_requested_deepseek_v4_artifacts(self) -> None:
        entries = {
            entry["filename"]: entry
            for entry in load_model_catalog().backends["ds4"].entries
        }
        filenames = {
            "DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf",
            "DeepSeek-V4-Flash-DSpark-support-0731.gguf",
        }

        self.assertLessEqual(filenames, entries.keys())
        for filename in filenames:
            self.assertEqual(entries[filename]["repo"], "antirez/deepseek-v4-gguf")
        self.assertEqual(
            entries["DeepSeek-V4-Flash-DSpark-support-0731.gguf"]["size_gb"],
            5.99,
        )

    def test_vllm_catalog_contains_current_toolbox_models(self) -> None:
        entries = {
            entry["repo"]: entry
            for entry in load_model_catalog().backends["vllm"].entries
        }
        lfm = entries["LiquidAI/LFM2.5-1.2B-Instruct"]
        muse = entries["meta-models/Muse-Glimmer-30B"]

        self.assertEqual(lfm["id"], "vllm-liquidai-lfm2-5-1-2b-instruct")
        self.assertEqual(lfm["name"], "LFM2.5-1.2B-Instruct")

        for entry in (lfm, muse):
            self.assertEqual(entry["attention_backend"], "ROCM_AITER_UNIFIED_ATTN")
            self.assertEqual(entry["env"]["VLLM_ROCM_USE_AITER"], "0")
            self.assertEqual(entry["env"]["VLLM_ROCM_USE_AITER_LINEAR"], "0")
            self.assertEqual(entry["valid_tp"], [1, 2])

        self.assertEqual(lfm["ctx"], "128000")
        self.assertNotIn("extra_flags", lfm)
        self.assertEqual(muse["ctx"], "131072")
        self.assertEqual(muse["extra_flags"], ["--model-impl", "transformers"])

    def test_toolbox_catalog_rejects_ambiguous_container_names(self) -> None:
        data = self.asset("toolboxes.json")
        data["toolboxes"][1]["container_name"] = data["toolboxes"][0]["container_name"]
        with self.assertRaisesRegex(CatalogError, "duplicate toolbox container_name"):
            ToolboxCatalog.from_dict(data)

    def test_toolbox_catalog_requires_explicit_feature_states(self) -> None:
        data = self.asset("toolboxes.json")
        del data["toolboxes"][0]["features"]["server"]
        with self.assertRaisesRegex(CatalogError, "must declare exactly"):
            ToolboxCatalog.from_dict(data)

    def test_model_catalog_rejects_invalid_backend_policy(self) -> None:
        data = self.asset("models.json")
        data["backends"]["vllm"]["models"][0]["valid_tp"] = []
        with self.assertRaisesRegex(CatalogError, "valid_tp"):
            ModelCatalog.from_dict(data)

    def test_model_catalog_rejects_invalid_dspark_defaults(self) -> None:
        data = self.asset("models.json")
        data["backends"]["llama_cpp"]["models"][0]["dspark"]["default_draft_n"] = 0
        with self.assertRaisesRegex(CatalogError, "default_draft_n"):
            ModelCatalog.from_dict(data)

    def test_model_catalog_rejects_invalid_external_mtp_model(self) -> None:
        data = self.asset("models.json")
        model = next(
            entry
            for entry in data["backends"]["llama_cpp"]["models"]
            if entry["id"] == "llama-kingjones777-qwen3-8-27b-rocmfp4-strix-mtp-gguf"
        )
        model["mtp"]["draft_models"] = []
        with self.assertRaisesRegex(CatalogError, "draft_models"):
            ModelCatalog.from_dict(data)

    def test_model_catalog_rejects_invalid_toolbox_defaults(self) -> None:
        data = self.asset("models.json")
        defaults = data["backends"]["llama_cpp"]["models"][0]["toolbox_defaults"]
        defaults["strix-halo-llama-vulkan-radv-performance"]["batch_size"] = 0
        with self.assertRaisesRegex(CatalogError, "batch_size"):
            ModelCatalog.from_dict(data)

    def test_model_catalog_rejects_duplicate_calibrated_ubatch_selector(self) -> None:
        data = self.asset("models.json")
        records = data["backends"]["llama_cpp"]["config"]["calibrated_ubatches"]
        records.append(dict(records[0]))
        with self.assertRaisesRegex(CatalogError, "duplicates a calibrated ubatch selector"):
            ModelCatalog.from_dict(data)

    def test_model_catalog_rejects_unknown_calibrated_ubatch_model(self) -> None:
        data = self.asset("models.json")
        record = data["backends"]["llama_cpp"]["config"]["calibrated_ubatches"][0]
        record["model_id"] = "missing-model"
        with self.assertRaisesRegex(CatalogError, "unknown llama.cpp model"):
            ModelCatalog.from_dict(data)

    def test_loaded_catalog_rejects_unknown_calibrated_ubatch_toolbox(self) -> None:
        data = self.asset("models.json")
        toolbox_catalog = load_toolbox_catalog()
        record = data["backends"]["llama_cpp"]["config"]["calibrated_ubatches"][0]
        record["toolbox_id"] = "missing-toolbox"
        with (
            patch("ai_toolbox_cockpit.catalog.loader._load_asset", return_value=data),
            patch(
                "ai_toolbox_cockpit.catalog.loader.load_toolbox_catalog",
                return_value=toolbox_catalog,
            ),
            self.assertRaisesRegex(CatalogError, "unknown llama.cpp toolbox"),
        ):
            load_model_catalog()


if __name__ == "__main__":
    unittest.main()
