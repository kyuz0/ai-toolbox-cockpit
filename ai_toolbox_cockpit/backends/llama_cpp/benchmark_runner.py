import csv
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import get_preferred_benchmark_ubatch
from .model_manager import resolve_model_path


@dataclass(frozen=True)
class BenchmarkSettings:
    max_context: int = 65536
    context_step: int = 8192
    prefill: int = 2048
    generation: int = 128
    repetitions: int = 3
    delay: int = 10
    flash_attention: bool = True
    use_mmap: bool = False
    kv_cache_type: str = ""
    platform_id: str = ""
    rocm_ubatch: int | None = None
    vulkan_ubatch: int | None = None
    extra_args: str = ""
    load_mode_toolboxes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class BenchmarkJob:
    toolbox_name: str
    model_path: str
    series: str
    command: tuple[str, ...]
    output_path: Path
    stderr_path: Path


def context_depths(settings: BenchmarkSettings) -> tuple[int, ...]:
    if settings.prefill <= 0 or settings.context_step <= 0:
        raise ValueError("Prefill and context step must be positive.")
    if settings.max_context < settings.context_step:
        raise ValueError("Maximum context must be at least the context step.")
    if settings.max_context % settings.context_step:
        raise ValueError("Maximum context must be divisible by the context step.")
    return (0, *range(settings.context_step, settings.max_context + 1, settings.context_step))


def toolbox_prefix(toolbox_command: str, toolbox_name: str) -> list[str]:
    if os.path.basename(toolbox_command) == "toolbox":
        return [toolbox_command, "run", "-c", toolbox_name, "--"]
    return [toolbox_command, "enter", toolbox_name, "--"]


def safe_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "benchmark"


def build_benchmark_jobs(
    toolbox_command: str,
    toolbox_names: list[str],
    model_paths: list[str],
    results_dir: Path,
    settings: BenchmarkSettings,
) -> list[BenchmarkJob]:
    jobs = []
    extra_args = shlex.split(settings.extra_args) if settings.extra_args else []
    depth_values = ",".join(str(depth) for depth in context_depths(settings))

    for model_pattern in model_paths:
        model_path = resolve_model_path(model_pattern)
        model_name = safe_filename_part(Path(model_path).stem)

        for toolbox_name in toolbox_names:
            toolbox_part = safe_filename_part(toolbox_name)
            is_vulkan = "vulkan" in toolbox_name.lower()
            backend = "vulkan" if is_vulkan else "rocm"
            override = settings.vulkan_ubatch if is_vulkan else settings.rocm_ubatch
            ubatch = override or get_preferred_benchmark_ubatch(
                model_path, settings.platform_id, backend
            )

            for series, prompt_tokens, generation_tokens, depths in (
                ("prefill", settings.prefill, 0, depth_values),
                ("generation", 0, settings.generation, depth_values),
            ):
                command = toolbox_prefix(toolbox_command, toolbox_name)
                command.extend([
                    "llama-bench",
                    "-ngl", "99",
                ])
                if toolbox_name in settings.load_mode_toolboxes:
                    command.extend(["--load-mode", "mmap" if settings.use_mmap else "none"])
                else:
                    command.extend(["-mmp", "1" if settings.use_mmap else "0"])
                command.extend([
                    "-m", model_path,
                    "-fa", "1" if settings.flash_attention else "0",
                    "-p", str(prompt_tokens),
                    "-n", str(generation_tokens),
                    "-d", depths,
                    "-b", str(settings.prefill),
                    "-r", str(settings.repetitions),
                    "-o", "jsonl",
                ])

                suffix = f"__curve-{series}"
                if settings.flash_attention:
                    suffix += "__fa1"
                if settings.use_mmap:
                    suffix += "__mmap1"
                if settings.kv_cache_type:
                    suffix += f"__kv-{safe_filename_part(settings.kv_cache_type)}"
                if ubatch:
                    command.extend(["-ub", str(ubatch)])
                    suffix += f"__ub{ubatch}"

                if settings.kv_cache_type:
                    command.extend([
                        "--cache-type-k", settings.kv_cache_type,
                        "--cache-type-v", settings.kv_cache_type,
                    ])

                command.extend(extra_args)
                output_path = results_dir / f"{model_name}__{toolbox_part}{suffix}.jsonl"
                jobs.append(BenchmarkJob(
                    toolbox_name=toolbox_name,
                    model_path=model_path,
                    series=series,
                    command=tuple(command),
                    output_path=output_path,
                    stderr_path=output_path.with_suffix(".stderr.log"),
                ))

    return jobs


def run_benchmark_job(job: BenchmarkJob) -> tuple[str, int | None]:
    if job.output_path.is_file() and job.output_path.stat().st_size > 0:
        return "skipped", None

    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            open(job.output_path, "w", encoding="utf-8") as output,
            open(job.stderr_path, "w", encoding="utf-8") as error_output,
        ):
            result = subprocess.run(
                list(job.command),
                stdout=output,
                stderr=error_output,
                text=True,
            )
            if result.returncode != 0:
                error_output.write(f"\nBenchmark exited with code {result.returncode}\n")
                return "failed", result.returncode
    except KeyboardInterrupt:
        if job.output_path.exists():
            job.output_path.unlink()
        if job.stderr_path.exists():
            job.stderr_path.unlink()
        raise
    except Exception:
        if job.output_path.exists() and job.output_path.stat().st_size == 0:
            job.output_path.unlink()
        if job.stderr_path.exists() and job.stderr_path.stat().st_size == 0:
            job.stderr_path.unlink()
        raise

    return "completed", 0


def write_curve_summary(jobs: list[BenchmarkJob], output_path: Path) -> int:
    """Combine JSONL using unambiguous starting-depth and ending-context fields."""
    rows = []
    for job in jobs:
        if not job.output_path.is_file():
            continue
        with open(job.output_path, "r", encoding="utf-8") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                depth = int(record.get("n_depth", 0))
                prompt = int(record.get("n_prompt", 0))
                generated = int(record.get("n_gen", 0))
                rows.append({
                    "model": Path(job.model_path).stem,
                    "toolbox": job.toolbox_name,
                    "series": job.series,
                    "starting_depth": depth,
                    "ending_context": depth + prompt + generated,
                    "n_prompt": prompt,
                    "n_gen": generated,
                    "n_batch": int(record.get("n_batch", 0)),
                    "n_ubatch": int(record.get("n_ubatch", 0)),
                    "avg_ts": float(record.get("avg_ts", 0)),
                    "stddev_ts": float(record.get("stddev_ts", 0)),
                    "samples_ts": json.dumps(record.get("samples_ts", [])),
                    "build_commit": record.get("build_commit", ""),
                    "gpu_info": record.get("gpu_info", ""),
                })

    rows.sort(key=lambda row: (
        row["toolbox"], row["model"], row["series"], row["starting_depth"]
    ))
    fields = [
        "model", "toolbox", "series", "starting_depth", "ending_context",
        "n_prompt", "n_gen", "n_batch", "n_ubatch", "avg_ts",
        "stddev_ts", "samples_ts", "build_commit", "gpu_info",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
