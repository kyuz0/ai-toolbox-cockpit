import json
import unittest
from importlib.resources import files

from ai_toolbox_cockpit.catalog import load_model_catalog, load_toolbox_catalog
from ai_toolbox_cockpit.catalog.schema import CatalogError, ModelCatalog, ToolboxCatalog


class CatalogTests(unittest.TestCase):
    @staticmethod
    def asset(name: str) -> dict:
        return json.loads(files("ai_toolbox_cockpit.assets").joinpath(name).read_text(encoding="utf-8"))

    def test_toolbox_catalog_loads_and_has_full_image_references(self) -> None:
        catalog = load_toolbox_catalog()
        self.assertEqual(catalog.schema_version, 2)
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


if __name__ == "__main__":
    unittest.main()
