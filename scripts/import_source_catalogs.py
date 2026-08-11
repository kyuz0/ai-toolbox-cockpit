#!/usr/bin/env python3
"""Import backend-owned source catalogs into the cockpit's unified models.json.

This maintainer utility performs no network access. It exists so curated policy
can be refreshed from checked-out toolbox repositories without hand-copying it.
"""

from __future__ import annotations

import argparse
import json
import re
import runpy
from pathlib import Path
from typing import Any


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-models", type=Path, required=True)
    parser.add_argument("--ds4-models", type=Path, required=True)
    parser.add_argument("--vllm-models", type=Path, required=True)
    parser.add_argument("--comfy-manager", type=Path, required=True)
    parser.add_argument("--comfy-workflows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    llama_entries = load_json(args.llama_models)
    for entry in llama_entries:
        entry["id"] = f"llama-{slug(entry['repo'])}"

    ds4_source = load_json(args.ds4_models)
    ds4_entries = []
    for entry in ds4_source["models"]:
        record = dict(entry)
        record["id"] = f"ds4-{slug(entry['filename'])}"
        record["repo"] = entry.get("repo", ds4_source["repo"])
        family = ds4_source.get("families", {}).get(entry.get("family"), {})
        record["server_defaults"] = {**family, **entry.get("server_defaults", {})}
        ds4_entries.append(record)

    vllm_source = runpy.run_path(str(args.vllm_models))
    vllm_entries = []
    for repo, policy in vllm_source["MODEL_TABLE"].items():
        vllm_entries.append({
            "id": f"vllm-{slug(repo)}",
            "name": repo.split("/")[-1],
            "repo": repo,
            **policy,
        })

    comfy_source = runpy.run_path(str(args.comfy_manager))
    comfy_entries = []
    for family in comfy_source["MODEL_FAMILIES"]:
        comfy_entries.append({
            "id": f"comfy-{slug(family['name'])}",
            "name": family["name"],
            "keywords": family.get("keywords", []),
            "exclude_keywords": family.get("exclude_keywords", []),
            "script": family["script"],
            "variants": family["variants"],
        })

    output = {
        "schema_version": 2,
        "backends": {
            "llama_cpp": {
                "kind": "gguf",
                "storage": {"config_key": "models_dir", "default": "~/models"},
                "models": llama_entries,
            },
            "ds4": {
                "kind": "gguf_file",
                "storage": {"config_key": "models_dir", "default": "~/ds4"},
                "config": {
                    "default_repo": ds4_source["repo"],
                    "families": ds4_source.get("families", {}),
                    "default_server_defaults": ds4_source.get("default_server_defaults", {}),
                },
                "models": ds4_entries,
            },
            "vllm": {
                "kind": "hf_repository",
                "storage": {"config_key": "hf_home", "default": "~/.cache/huggingface"},
                "config": {
                    "policy_source": str(args.vllm_models),
                    "default_attention_backend": "TRITON_ATTN",
                    "attention_backends": ["TRITON_ATTN", "ROCM_ATTN", "ROCM_AITER_UNIFIED_ATTN"],
                    "default_gpu_memory_utilization": "0.90",
                },
                "models": vllm_entries,
            },
            "comfyui": {
                "kind": "workflow_bundle",
                "storage": {"config_key": "models_dir", "default": "~/comfy-models"},
                "config": {
                    "policy_source": str(args.comfy_manager),
                    "model_manager_command": ["model_manager"],
                    "workflows": sorted(path.name for path in args.comfy_workflows.glob("*.json")),
                },
                "bundles": comfy_entries,
            },
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
