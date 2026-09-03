import json
import unittest
from importlib.resources import files
from unittest.mock import patch

from ai_toolbox_cockpit.backends import BACKENDS
from ai_toolbox_cockpit.backends.llama_cpp.model_manager import get_hf_quants
from ai_toolbox_cockpit.catalog import load_model_catalog, load_toolbox_catalog
from ai_toolbox_cockpit.catalog.schema import CatalogError, ModelCatalog, ToolboxCatalog


class CatalogTests(unittest.TestCase):
    def test_gb10_catalog_has_all_initial_toolboxes(self) -> None:
        catalog = load_toolbox_catalog()
        platform = catalog.platform("gb10")
        self.assertEqual(
            set(platform.toolbox_ids),
            {
                "gb10-llama-cuda13",
                "gb10-ds4-cuda13",
                "gb10-vllm-cu130",
            },
        )
        self.assertEqual(platform.defaults["llama_cpp"], "gb10-llama-cuda13")
        self.assertEqual(platform.defaults["ds4"], "gb10-ds4-cuda13")
        self.assertEqual(platform.defaults["vllm"], "gb10-vllm-cu130")
        for toolbox_id in platform.toolbox_ids:
            self.assertEqual(
                catalog.toolboxes[toolbox_id].runtime_profile,
                "nvidia-gb10",
            )

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

    def test_strix_halo_experimental_rocm_llama_toolboxes_are_catalogued(self) -> None:
        catalog = load_toolbox_catalog()
        expected = {
            "strix-halo-llama-rocm-10-0-performance": (
                "llama-rocm-10.0-performance",
                "docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-10.0-performance",
            ),
            "strix-halo-llama-rocm-10-0-qwen-3-8-flash-next": (
                "llama-rocm-10.0-qwen-3.8-flash-next",
                "docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-10.0-qwen-3.8-flash-next",
            ),
            "strix-halo-llama-rocm-10-0-engramhalo": (
                "llama-rocm-10.0-engramhalo",
                "docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-10.0-engramhalo",
            ),
        }

        for toolbox_id, (container_name, image) in expected.items():
            toolbox = catalog.toolboxes[toolbox_id]
            self.assertEqual(toolbox.container_name, container_name)
            self.assertEqual(toolbox.image, image)
            self.assertEqual(toolbox.channel, "experimental")
            self.assertIn(toolbox.runtime_profile, {"amd-rocm", "amd-rocm-hipblaslt"})

    def test_r9700_toolboxes_match_the_active_source_images(self) -> None:
        catalog = load_toolbox_catalog()
        expected = {
            "r9700-llama-rocm-10-0": "docker.io/kyuz0/amd-r9700-toolboxes:rocm-10.0",
            "r9700-llama-therock-nightly": "docker.io/kyuz0/amd-r9700-toolboxes:therock-nightly",
            "r9700-llama-vulkan-radv": "docker.io/kyuz0/amd-r9700-toolboxes:vulkan-radv",
            "r9700-llama-vulkan-rocmfpx": "docker.io/kyuz0/amd-r9700-toolboxes:vulkan-rocmfpx",
            "r9700-r9v-qwen38-rocm-10-0": "docker.io/kyuz0/amd-r9700-toolboxes:r9v-qwen38-rocm-10.0",
        }

        toolboxes = {
            toolbox.id: toolbox
            for toolbox in catalog.platform_toolboxes("r9700")
        }
        self.assertEqual(set(toolboxes), set(expected))
        for toolbox_id, image in expected.items():
            self.assertEqual(toolboxes[toolbox_id].image, image)
        for toolbox_id in (
            "r9700-llama-rocm-10-0",
            "r9700-llama-vulkan-radv",
        ):
            self.assertTrue(toolboxes[toolbox_id].supports_load_mode)
        for toolbox_id in (
            "r9700-llama-therock-nightly",
            "r9700-llama-vulkan-rocmfpx",
            "r9700-r9v-qwen38-rocm-10-0",
        ):
            self.assertFalse(toolboxes[toolbox_id].supports_load_mode)
        self.assertEqual(
            catalog.platform("r9700").defaults["llama_cpp"],
            "r9700-llama-rocm-10-0",
        )
        self.assertEqual(
            catalog.platform("r9700").defaults["r9v"],
            "r9700-r9v-qwen38-rocm-10-0",
        )
        self.assertEqual(
            toolboxes["r9700-r9v-qwen38-rocm-10-0"].backend,
            "r9v",
        )
        self.assertEqual(
            toolboxes["r9700-r9v-qwen38-rocm-10-0"].runtime_profile,
            "amd-rocm-ipc",
        )

    def test_model_catalog_preserves_backend_semantics(self) -> None:
        catalog = load_model_catalog()
        self.assertEqual(catalog.backends["llama_cpp"].kind, "gguf")
        self.assertEqual(catalog.backends["vllm"].kind, "hf_repository")
        self.assertEqual(catalog.backends["comfyui"].entries_key, "bundles")
        self.assertEqual(catalog.backends["ds4"].kind, "gguf_file")
        self.assertEqual(catalog.backends["r9v"].kind, "immutable_profile")

    def test_r9v_catalog_is_one_pinned_qwen_profile(self) -> None:
        backend = load_model_catalog().backends["r9v"]

        self.assertEqual(len(backend.entries), 1)
        model = backend.entries[0]
        self.assertEqual(model["repo"], "Dyluhn/Qwen3.8-Flash-Next-R9V-IQ4_XS")
        self.assertEqual(
            model["revision"],
            "bf836f0c20b6c92fcad4226ad3115eb8a19f7582",
        )
        self.assertEqual(model["license"], "Qwen Community License 1.0")
        self.assertEqual(len(model["artifacts"]), 19)
        self.assertEqual(model["ple"]["bytes"], 28800138240)
        self.assertEqual(
            model["ple"]["sha256"],
            "dd55c28902f38cd88134b2a569c51282c5ffce30080487e1a645740115c56cc3",
        )

    def test_ds4_uses_dwarfstar_display_name(self) -> None:
        self.assertEqual(BACKENDS["ds4"].label, "DwarfStar (ds4)")

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
        defaults = load_model_catalog().backends["ds4"].config["families"]["deepseek-v4"]
        self.assertTrue(defaults["dspark_enabled"])
        self.assertEqual(defaults["dspark_support_filename"], "DeepSeek-V4-Flash-DSpark-support-0731.gguf")
        self.assertEqual(defaults["dspark_confidence"], 0)

    def test_ds4_catalog_contains_deepseek_vision_artifacts(self) -> None:
        entries = {
            entry["filename"]: entry
            for entry in load_model_catalog().backends["ds4"].entries
        }
        main_filenames = {
            "DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf",
            "DeepSeek-V4-Flash-Vision-Exp-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf",
            "DeepSeek-V4-Flash-Vision-Exp-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out.gguf",
        }
        encoder_filename = "DeepSeek-V4-Flash-Vision-Encoder.gguf"
        dspark_filename = "DeepSeek-V4-Flash-Vision-Exp-DSpark-support.gguf"

        self.assertLessEqual(main_filenames | {encoder_filename, dspark_filename}, entries.keys())
        for filename in main_filenames | {encoder_filename, dspark_filename}:
            self.assertEqual(entries[filename]["repo"], "antirez/deepseek-v4-gguf")
            self.assertEqual(entries[filename]["family"], "deepseek-v4-vision")
            self.assertEqual(len(entries[filename]["sha256"]), 64)
        self.assertEqual(entries[encoder_filename]["artifact_role"], "vision_encoder")
        self.assertEqual(entries[dspark_filename]["artifact_role"], "dspark_support")
        recommended = [
            entry["filename"]
            for entry in entries.values()
            if entry["family"].startswith("deepseek-v4") and entry.get("recommended")
        ]
        self.assertEqual(
            recommended,
            ["DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf"],
        )

        defaults = load_model_catalog().backends["ds4"].config["families"]["deepseek-v4-vision"]
        self.assertTrue(defaults["dspark_enabled"])
        self.assertEqual(defaults["dspark_support_filename"], dspark_filename)

    def test_ds4_catalog_contains_supported_glm53_flash_artifacts(self) -> None:
        entries = {
            entry["filename"]: entry
            for entry in load_model_catalog().backends["ds4"].entries
        }

        q2 = entries["GLM-5.3-Flash-Q2.gguf"]
        q4 = entries["GLM-5.3-Flash-Q4_K.gguf"]
        vision = entries["GLM-5.3-Flash-Vision-Encoder.gguf"]

        for model in (q2, q4, vision):
            self.assertEqual(model["repo"], "antirez/glm-5.3-flash-gguf")
            self.assertEqual(model["family"], "glm-5.3-flash")
        self.assertEqual(q2["server_defaults"]["standalone_ctx"], 262144)
        self.assertNotIn("ssd_streaming", q2["server_defaults"])
        self.assertNotIn("ssd_experts", q2["server_defaults"])
        self.assertEqual(q4["server_defaults"]["standalone_ctx"], 4096)
        self.assertNotIn("ssd_streaming", q4["server_defaults"])
        self.assertEqual(vision["artifact_role"], "vision_encoder")
        self.assertEqual(
            vision["sha256"],
            "ae23e14c6979e889051b2e4a39351abcdafb161e18e606fae4d8c40095a4bf3a",
        )
        self.assertNotIn("GLM-5.3-Flash-FP8.gguf", entries)

    def test_ds4_catalog_does_not_offer_glm52(self) -> None:
        entries = load_model_catalog().backends["ds4"].entries

        self.assertFalse(any(entry.get("family") == "glm-5.2" for entry in entries))
        self.assertNotIn(
            "glm-5.2",
            load_model_catalog().backends["ds4"].config["families"],
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

    def test_llama_catalog_contains_qwen38_flash_next(self) -> None:
        entries = {
            entry["repo"]: entry
            for entry in load_model_catalog().backends["llama_cpp"].entries
        }
        model = entries["unsloth/Qwen3.8-Flash-Next-GGUF"]

        self.assertEqual(model["id"], "llama-unsloth-qwen3-8-flash-next-gguf")
        self.assertEqual(model["vision_projector"]["patterns"], ["mmproj-*.gguf"])
        self.assertEqual(model["mtp"]["default_draft_n"], 2)
        self.assertEqual(
            model["auxiliary_downloads"],
            [{
                "name": "drluoto MTP sidecars",
                "repo": "drluoto/Qwen3.8-Flash-Next-MTP-GGUF",
                "role": "mtp",
                "recommended_filename": "mtp-Qwen3.8-Flash-Next-Q8_0.gguf",
                "description": (
                    "External MTP heads for the Strix Halo Flash Next fork. "
                    "Q8_0 is the ROCm-validated default; Q4_K_M and bf16 remain "
                    "available as alternatives."
                ),
            }, {
                "name": "EngramHalo Strix Halo MTP sidecar",
                "repo": "EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF",
                "role": "mtp",
                "recommended_filename": "mtp-Qwen3.8-Flash-Next-Q8_0.gguf",
                "description": (
                    "Dedicated Q8_0 MTP sidecar used by "
                    "Aristo94/EngramHalo.cpp's measured Strix Halo configuration."
                ),
            }],
        )
        self.assertEqual(model["default_inference_profile"], "Thinking (Effort: XHigh)")
        self.assertEqual(
            model["toolbox_defaults"],
            {
                "r9700-llama-rocm-10-0": {"gpu_layers": None},
                "r9700-llama-vulkan-radv": {"gpu_layers": None},
            },
        )

    def test_llama_downloads_offer_every_qwen38_rocmfpx_gguf(self) -> None:
        repo = "julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF"
        entries = {
            entry["repo"]: entry
            for entry in load_model_catalog().backends["llama_cpp"].entries
        }
        model = entries[repo]
        files = [
            "Qwen3.8-27B-Q3_K_M.gguf",
            "Qwen3.8-27B-Q3_K_S.gguf",
            "Qwen3.8-27B-ROCmFP2.gguf",
            "Qwen3.8-27B-ROCmFP4-FAST.gguf",
            "Qwen3.8-27B-ROCmFP4-STRIX_LEAN.gguf",
            "Qwen3.8-27B-ROCmFP8.gguf",
        ]

        self.assertEqual(
            model["id"],
            "llama-julianmb-qwen-3-8-27b-rocmfp4-fast-gguf",
        )
        self.assertEqual(model["compatible_toolboxes"], ["rocmfpx"])
        self.assertEqual(model["mtp"]["default_draft_n"], 4)
        with patch(
            "ai_toolbox_cockpit.backends.llama_cpp.model_manager.HfApi.list_repo_files",
            return_value=files,
        ) as list_repo_files:
            self.assertEqual(get_hf_quants(repo), files)
        list_repo_files.assert_called_once_with(repo_id=repo, repo_type="model")

    def test_llama_catalog_contains_glm53_flash_with_unsloth_profiles(self) -> None:
        entries = {
            entry["repo"]: entry
            for entry in load_model_catalog().backends["llama_cpp"].entries
        }
        model = entries["unsloth/GLM-5.3-Flash-GGUF"]

        self.assertEqual(model["id"], "llama-unsloth-glm-5-3-flash-gguf")
        self.assertEqual(model["vision_projector"]["patterns"], ["mmproj-*.gguf"])
        self.assertEqual(model["default_inference_profile"], "Thinking (Effort: Max)")
        self.assertEqual(
            model["inference_profiles"],
            {
                "Thinking (Effort: Max)": {
                    "description": "Unsloth default; maximum reasoning for complicated tasks",
                    "args": "--chat-template-kwargs '{\"reasoning_effort\":\"max\"}' --temp 1.0 --top-p 0.95",
                },
                "Thinking (Effort: High)": {
                    "description": "High reasoning effort with Unsloth's default sampling",
                    "args": "--chat-template-kwargs '{\"reasoning_effort\":\"high\"}' --temp 1.0 --top-p 0.95",
                },
                "Thinking (Effort: Low)": {
                    "description": "Low reasoning effort with Unsloth's default sampling",
                    "args": "--chat-template-kwargs '{\"reasoning_effort\":\"low\"}' --temp 1.0 --top-p 0.95",
                },
                "DeepSWE": {
                    "description": "Unsloth's DeepSWE evaluation sampling settings",
                    "args": "--temp 0.95 --top-p 1.0",
                },
            },
        )

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

    def test_flash_next_toolbox_recommended_use_is_strix_only_and_structured(self) -> None:
        catalog = load_toolbox_catalog()
        toolbox_id = "strix-halo-llama-rocm-10-0-qwen-3-8-flash-next"
        toolbox = catalog.toolboxes[toolbox_id]
        recommended = toolbox.backend_config["recommended_use"]

        self.assertIn(toolbox_id, catalog.platform("strix-halo").toolbox_ids)
        for platform_id in ("r9700", "gb10", "intel-b70"):
            self.assertNotIn(toolbox_id, catalog.platform(platform_id).toolbox_ids)
        self.assertEqual(recommended["platform_id"], "strix-halo")
        self.assertEqual(recommended["model_filename_pattern"], "*UD-IQ4_XS*.gguf")
        self.assertEqual(recommended["server_defaults"]["load_mode"], "dio")
        self.assertEqual(
            recommended["server_defaults"]["mtp"]["spec_types"],
            ["draft-mtp", "ngram-mod"],
        )

    def test_toolbox_catalog_rejects_invalid_recommended_use_load_mode(self) -> None:
        data = self.asset("toolboxes.json")
        toolbox = next(
            entry
            for entry in data["toolboxes"]
            if entry["id"] == "strix-halo-llama-rocm-10-0-qwen-3-8-flash-next"
        )
        toolbox["backend_config"]["recommended_use"]["server_defaults"][
            "load_mode"
        ] = "magic"
        with self.assertRaisesRegex(CatalogError, "load_mode is unsupported"):
            ToolboxCatalog.from_dict(data)

    def test_engramhalo_profile_records_the_validated_limits(self) -> None:
        catalog = load_toolbox_catalog()
        toolbox_id = "strix-halo-llama-rocm-10-0-engramhalo"
        toolbox = catalog.toolboxes[toolbox_id]
        recommended = toolbox.backend_config["recommended_use"]
        defaults = recommended["server_defaults"]

        self.assertIn(toolbox_id, catalog.platform("strix-halo").toolbox_ids)
        self.assertEqual(
            recommended["model_filename_patterns"],
            ["*UD-IQ3_XXS*.gguf", "*UD-IQ4_XS*.gguf"],
        )
        self.assertEqual(
            recommended["sidecar"]["repo"],
            "EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF",
        )
        self.assertEqual(defaults["context_size"], 163840)
        self.assertEqual(defaults["load_mode"], "mmap")
        self.assertEqual(defaults["kv_cache_type"], "q8_0")
        self.assertEqual(defaults["mtp"]["spec_draft_p_min"], 0.75)
        self.assertTrue(any("262144" in note for note in recommended["notes"]))

    def test_recommended_use_rejects_ambiguous_filename_matchers(self) -> None:
        data = self.asset("toolboxes.json")
        toolbox = next(
            entry
            for entry in data["toolboxes"]
            if entry["id"] == "strix-halo-llama-rocm-10-0-engramhalo"
        )
        recommended = toolbox["backend_config"]["recommended_use"]
        recommended["model_filename_pattern"] = "*.gguf"

        with self.assertRaisesRegex(CatalogError, "exactly one"):
            ToolboxCatalog.from_dict(data)

    def test_recommended_use_can_describe_a_fork_without_a_sidecar(self) -> None:
        data = self.asset("toolboxes.json")
        toolbox = next(
            entry
            for entry in data["toolboxes"]
            if entry["id"] == "strix-halo-llama-rocm-10-0-qwen-3-8-flash-next"
        )
        recommended = toolbox["backend_config"]["recommended_use"]
        recommended.pop("sidecar")
        recommended["server_defaults"].pop("mtp")

        catalog = ToolboxCatalog.from_dict(data)

        self.assertEqual(
            catalog.toolboxes[toolbox["id"]].backend_config["recommended_use"][
                "model_id"
            ],
            "llama-unsloth-qwen3-8-flash-next-gguf",
        )

    def test_model_catalog_rejects_invalid_backend_policy(self) -> None:
        data = self.asset("models.json")
        data["backends"]["vllm"]["models"][0]["valid_tp"] = []
        with self.assertRaisesRegex(CatalogError, "valid_tp"):
            ModelCatalog.from_dict(data)

    def test_model_catalog_rejects_unknown_ds4_artifact_role(self) -> None:
        data = self.asset("models.json")
        data["backends"]["ds4"]["models"][0]["artifact_role"] = "generic"
        with self.assertRaisesRegex(CatalogError, "artifact_role"):
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

        data = self.asset("models.json")
        model = next(
            entry
            for entry in data["backends"]["llama_cpp"]["models"]
            if entry["id"] == "llama-unsloth-qwen3-8-flash-next-gguf"
        )
        model["toolbox_defaults"]["r9700-llama-rocm-10-0"]["gpu_layers"] = -1
        with self.assertRaisesRegex(CatalogError, "gpu_layers"):
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
