#!/usr/bin/env python3
"""Import Llama and DS4 toolbox catalogs and retain first-class backend images."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def profile_for(engine_args: list[str], platform_id: str) -> str:
    if platform_id == "intel-b70":
        return "intel-level-zero"
    if "/dev/kfd" in engine_args:
        return "amd-rocm"
    return "vulkan"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--ds4", type=Path, required=True)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    llama = load(args.llama)
    ds4 = load(args.ds4)
    existing = load(args.existing)
    retained = [
        item for item in existing["toolboxes"]
        if item["backend"] in {"vllm", "comfyui"}
    ]
    toolboxes = []
    assignments: dict[str, list[str]] = {"strix-halo": [], "r9700": [], "gb10": [], "intel-b70": []}
    defaults: dict[str, dict[str, str]] = {key: {} for key in assignments}

    for platform in llama["platforms"]:
        platform_id = platform["id"]
        for group in platform["groups"]:
            stable = "official" in group["name"].lower()
            for source in group["toolboxes"]:
                toolbox_id = f"{platform_id}-{slug(source['name'])}"
                image = f"{platform['registry']}:{source.get('tag', 'latest')}"
                toolboxes.append({
                    "id": toolbox_id,
                    "backend": "llama_cpp",
                    "name": source["name"],
                    "container_name": source["name"],
                    "group": group["name"],
                    "image": image,
                    "channel": "stable" if stable else "experimental",
                    "maturity": "stable" if stable else "experimental",
                    "description": source.get("description", ""),
                    "runtime_profile": profile_for(source.get("engine_args", []), platform_id),
                    "supports_load_mode": bool(source.get("supports_load_mode", False)),
                    "features": {
                        "interactive": "supported", "server": "supported",
                        "models": "supported", "benchmark": "supported",
                    },
                })
                assignments[platform_id].append(toolbox_id)
                if source.get("tag") == platform.get("default_toolbox_tag"):
                    defaults[platform_id]["llama_cpp"] = toolbox_id
        if "llama_cpp" not in defaults[platform_id] and assignments[platform_id]:
            defaults[platform_id]["llama_cpp"] = assignments[platform_id][0]

    for group in ds4["groups"]:
        for source in group["toolboxes"]:
            platform_id = "r9700" if "gfx1201" in source["name"] else "strix-halo"
            toolbox_id = f"{platform_id}-{slug(source['name'])}"
            channel = "experimental" if "nightly" in source["name"] or "gfx1201" in source["name"] else "stable"
            toolboxes.append({
                "id": toolbox_id,
                "backend": "ds4",
                "name": source["name"],
                "container_name": source["name"],
                "group": group["name"],
                "image": f"{ds4['registry']}:{source.get('tag', 'latest')}",
                "channel": channel,
                "maturity": "experimental" if channel == "experimental" else "stable",
                "description": source.get("description", ""),
                "runtime_profile": "amd-rocm",
                "server_binary": source.get("server_binary", "ds4-server"),
                "features": {
                    "interactive": "supported", "server": "supported",
                    "models": "supported", "benchmark": "unavailable",
                },
            })
            assignments[platform_id].append(toolbox_id)
            if source["name"] == "ds4-rocm-7.14":
                defaults[platform_id]["ds4"] = toolbox_id
    if "ds4" not in defaults["r9700"]:
        ds4_r9700 = [item for item in assignments["r9700"] if "ds4" in item]
        if ds4_r9700:
            defaults["r9700"]["ds4"] = ds4_r9700[0]

    for source in retained:
        toolboxes.append(source)
        platform_id = "strix-halo"
        assignments[platform_id].append(source["id"])
        if source["channel"] == "stable" and source["backend"] not in defaults[platform_id]:
            defaults[platform_id][source["backend"]] = source["id"]

    platform_meta = {
        "strix-halo": ("AMD Strix Halo", "Ryzen AI Max, gfx1151"),
        "r9700": ("AMD Radeon AI PRO R9700", "RDNA 4, gfx1201"),
        "gb10": ("NVIDIA GB10", "DGX Spark / GB10 catalogue pending"),
        "intel-b70": ("Intel Arc B70", "Battlemage / Level Zero"),
    }
    output = {
        "schema_version": 2,
        "runtime_profiles": existing["runtime_profiles"],
        "toolboxes": toolboxes,
        "platforms": [
            {
                "id": platform_id,
                "name": platform_meta[platform_id][0],
                "description": platform_meta[platform_id][1],
                "toolbox_ids": assignments[platform_id],
                "defaults": defaults[platform_id],
            }
            for platform_id in ("strix-halo", "r9700", "gb10", "intel-b70")
        ],
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
