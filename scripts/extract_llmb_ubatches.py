#!/usr/bin/env python3
"""Emit llama.cpp ubatch calibrations that map to this cockpit catalogue.

This utility is deliberately read-only. It reads a local-llm-benchmarks
checkout and prints a reviewed models.json config fragment; it never edits
either repository.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any


MODEL_ID_ALIASES = {
    "lfm25-1p2b-instruct-bf16": "llama-unsloth-lfm2-5-1-2b-instruct-gguf",
}
SOURCE_ALIASES = {
    "LiquidAI/LFM2.5-1.2B-Instruct-GGUF": "unsloth/LFM2.5-1.2B-Instruct-GGUF",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def cockpit_model_id(subject: dict[str, Any], models: list[dict[str, Any]]) -> str | None:
    by_id = {model["id"]: model for model in models}
    alias = MODEL_ID_ALIASES.get(str(subject.get("model_id", "")))
    if alias in by_id:
        return alias

    source = str(subject.get("model_source", "")).strip()
    source = SOURCE_ALIASES.get(source, source)
    matches = [model["id"] for model in models if model["repo"] == source]
    return matches[0] if len(matches) == 1 else None


def extract(
    benchmark_root: Path,
    cockpit_models_path: Path,
    cockpit_toolboxes_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    benchmark_catalog = tomllib.loads(
        (benchmark_root / "catalog/catalog.toml").read_text(encoding="utf-8")
    )
    cockpit_models = load_json(cockpit_models_path)["backends"]["llama_cpp"]["models"]
    cockpit_toolboxes = load_json(cockpit_toolboxes_path)["toolboxes"]
    toolbox_by_container = {
        toolbox["container_name"]: toolbox["id"]
        for toolbox in cockpit_toolboxes
        if toolbox["backend"] == "llama_cpp"
    }
    engine_to_toolbox = {
        engine_id: toolbox_by_container[engine["container"]]
        for engine_id, engine in benchmark_catalog.get("engines", {}).items()
        if engine.get("family") == "llama.cpp"
        and engine.get("container") in toolbox_by_container
    }

    selected: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    unmapped: set[str] = set()
    for job_path in sorted((benchmark_root / "results").glob("**/job.json")):
        job = load_json(job_path)
        ubatch = job.get("tuning", {}).get("ubatch", {})
        stack = job.get("stack", {})
        if stack.get("engine_name") != "llama.cpp" or ubatch.get("status") != "calibrated":
            continue

        engine_id = str(stack.get("engine_id", ""))
        toolbox_id = engine_to_toolbox.get(engine_id)
        model_id = cockpit_model_id(job.get("subject", {}), cockpit_models)
        if not toolbox_id or not model_id:
            unmapped.add(
                f"{job.get('subject', {}).get('model_id', '?')} / {engine_id or '?'}"
            )
            continue

        quant = str(job.get("subject", {}).get("quant", "")).strip()
        serving_config = str(job.get("serving", {}).get("config_id", "")).strip()
        kv_cache_type = str(job.get("tuning", {}).get("kv_cache_type") or "default")
        if not quant or not serving_config:
            unmapped.add(f"{job.get('id', job_path.name)} / incomplete identity")
            continue

        record = {
            "model_id": model_id,
            "toolbox_id": toolbox_id,
            "filename_pattern": f"*{quant}*.gguf",
            "serving_config": serving_config,
            "kv_cache_type": kv_cache_type,
            "batch_size": int(ubatch["batch_size"]),
            "ubatch_size": int(ubatch["selected_ubatch"]),
            "source_job": str(job["id"]),
            "source_job_status": str(job["status"]),
        }
        identity = (
            model_id, toolbox_id, record["filename_pattern"], serving_config,
            kv_cache_type,
        )
        previous = selected.get(identity)
        if previous is None or str(job["created_at"]) > previous["_calibrated_at"]:
            record["_calibrated_at"] = str(job["created_at"])
            selected[identity] = record

    records = sorted(
        ({key: value for key, value in record.items() if key != "_calibrated_at"}
         for record in selected.values()),
        key=lambda record: (
            record["model_id"], record["toolbox_id"], record["filename_pattern"],
            record["serving_config"], record["kv_cache_type"],
        ),
    )
    return records, sorted(unmapped)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument(
        "--models",
        type=Path,
        default=project_root / "ai_toolbox_cockpit/assets/models.json",
    )
    parser.add_argument(
        "--toolboxes",
        type=Path,
        default=project_root / "ai_toolbox_cockpit/assets/toolboxes.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records, unmapped = extract(args.benchmarks, args.models, args.toolboxes)
    rendered = json.dumps({"calibrated_ubatches": records}, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    for identity in unmapped:
        print(f"unmapped: {identity}", file=sys.stderr)


if __name__ == "__main__":
    main()
