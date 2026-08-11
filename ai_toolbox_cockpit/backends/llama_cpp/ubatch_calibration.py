"""Calibrate one model/backend ubatch across the standard depth curve."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .benchmark_runner import (
    BenchmarkJob,
    run_benchmark_job,
    safe_filename_part,
    toolbox_prefix,
)
from .config import get_platform
from .model_manager import get_active_platform, get_benchmark_results_dir, resolve_model_path
from ai_toolbox_cockpit.runtime.toolboxes import get_os_toolbox_cmd
from .ubatch_profiles import backend_from_name, model_key, save_profile


DEFAULT_UBATCHES = (256, 512, 1024, 2048)
DEFAULT_DEPTHS = tuple(range(0, 65537, 8192))


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def toolbox_supports_load_mode(platform_id: str, toolbox_name: str) -> bool:
    platform = get_platform(platform_id) or {}
    for group in platform.get("groups", []):
        for toolbox in group.get("toolboxes", []):
            if toolbox.get("name") == toolbox_name:
                return bool(toolbox.get("supports_load_mode", False))
    return False


def build_calibration_jobs(
    toolbox_command: str,
    toolbox_name: str,
    model_path: str,
    output_dir: Path,
    ubatches: tuple[int, ...],
    depths: tuple[int, ...] = DEFAULT_DEPTHS,
    prefill: int = 2048,
    repetitions: int = 3,
    supports_load_mode: bool = True,
) -> list[BenchmarkJob]:
    depth_values = ",".join(str(depth) for depth in depths)
    model_part = safe_filename_part(Path(model_path).stem)
    toolbox_part = safe_filename_part(toolbox_name)
    jobs = []

    for ubatch in ubatches:
        command = toolbox_prefix(toolbox_command, toolbox_name)
        command.extend(["llama-bench", "-ngl", "99"])
        if supports_load_mode:
            command.extend(["--load-mode", "none"])
        else:
            command.extend(["-mmp", "0"])
        command.extend([
            "-m", model_path,
            "-fa", "1",
            "-p", str(prefill),
            "-n", "0",
            "-d", depth_values,
            "-b", str(prefill),
            "-ub", str(ubatch),
            "-r", str(repetitions),
            "-o", "jsonl",
        ])
        output_path = output_dir / (
            f"{model_part}__{toolbox_part}__curve-prefill__fa1__ub{ubatch}.jsonl"
        )
        jobs.append(BenchmarkJob(
            toolbox_name=toolbox_name,
            model_path=model_path,
            series="prefill",
            command=tuple(command),
            output_path=output_path,
            stderr_path=output_path.with_suffix(".stderr.log"),
        ))
    return jobs


def read_candidate(job: BenchmarkJob, expected_depths: tuple[int, ...]) -> dict:
    points = []
    if job.output_path.is_file():
        for line in job.output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            points.append({
                "depth": int(record.get("n_depth", -1)),
                "avg_ts": float(record.get("avg_ts", 0)),
                "stddev_ts": float(record.get("stddev_ts", 0)),
                "samples_ts": record.get("samples_ts", []),
            })

    ubatch_index = job.command.index("-ub") + 1
    ubatch = int(job.command[ubatch_index])
    complete = (
        tuple(point["depth"] for point in points) == expected_depths
        and all(point["avg_ts"] > 0 and len(point["samples_ts"]) >= 3 for point in points)
    )
    return {
        "ubatch": ubatch,
        "complete": complete,
        "points": points,
        "raw_jsonl": str(job.output_path),
        "stderr_log": str(job.stderr_path),
    }


def select_ubatch(candidates: list[dict], depths: tuple[int, ...]) -> int | None:
    complete = [candidate for candidate in candidates if candidate["complete"]]
    if not complete:
        return None

    best_at_depth = {
        depth: max(
            point["avg_ts"]
            for candidate in complete
            for point in candidate["points"]
            if point["depth"] == depth
        )
        for depth in depths
    }
    for candidate in complete:
        candidate["mean_relative_throughput"] = sum(
            point["avg_ts"] / best_at_depth[point["depth"]]
            for point in candidate["points"]
        ) / len(depths)

    return max(
        complete,
        key=lambda candidate: (candidate["mean_relative_throughput"], -candidate["ubatch"]),
    )["ubatch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate ubatch for one model/backend across all standard depths."
    )
    parser.add_argument("--toolbox", required=True, help="Installed toolbox name")
    parser.add_argument("--model", required=True, help="GGUF path or shard glob")
    parser.add_argument("--platform", default=get_active_platform())
    parser.add_argument("--backend", choices=("rocm", "vulkan"))
    parser.add_argument("--ubatches", type=parse_int_list, default=DEFAULT_UBATCHES)
    parser.add_argument("--prefill", type=int, default=2048)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cooldown", type=int, default=10)
    parser.add_argument("--toolbox-command", default="")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=get_benchmark_results_dir() / "ubatch_calibration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model)
    if not Path(model_path).is_file():
        raise SystemExit(f"Model not found: {args.model}")
    backend = args.backend or backend_from_name(args.toolbox)
    if not backend:
        raise SystemExit("Could not infer backend; pass --backend rocm or --backend vulkan.")
    toolbox_command = args.toolbox_command or get_os_toolbox_cmd()
    if not toolbox_command:
        raise SystemExit("No Toolbx or Distrobox command is available.")
    if args.prefill <= 0 or args.repetitions <= 0 or args.cooldown < 0:
        raise SystemExit("Prefill/repetitions must be positive and cooldown cannot be negative.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.results_dir.expanduser()
        / args.platform
        / backend
        / model_key(model_path)
        / safe_filename_part(args.toolbox)
        / run_id
    )
    jobs = build_calibration_jobs(
        toolbox_command=toolbox_command,
        toolbox_name=args.toolbox,
        model_path=model_path,
        output_dir=output_dir,
        ubatches=args.ubatches,
        prefill=args.prefill,
        repetitions=args.repetitions,
        supports_load_mode=toolbox_supports_load_mode(args.platform, args.toolbox),
    )

    print(f"Calibrating {model_key(model_path)} on {backend} ({args.toolbox})")
    print(f"Raw results: {output_dir}")
    statuses = {}
    for index, job in enumerate(jobs, start=1):
        ubatch = int(job.command[job.command.index("-ub") + 1])
        print(f"[{index}/{len(jobs)}] ubatch {ubatch}")
        status, return_code = run_benchmark_job(job)
        statuses[ubatch] = {"status": status, "return_code": return_code}
        print(f"  {status}")
        if index < len(jobs) and status != "skipped" and args.cooldown:
            time.sleep(args.cooldown)

    candidates = [read_candidate(job, DEFAULT_DEPTHS) for job in jobs]
    for candidate in candidates:
        candidate.update(statuses[candidate["ubatch"]])
        if candidate["status"] not in {"completed", "skipped"}:
            candidate["complete"] = False
    selected = select_ubatch(candidates, DEFAULT_DEPTHS)
    profile = {
        "selected_ubatch": selected,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "toolbox": args.toolbox,
        "depths": list(DEFAULT_DEPTHS),
        "prefill": args.prefill,
        "repetitions": args.repetitions,
        "candidates": candidates,
    }
    profile_path = save_profile(
        args.platform, backend, model_path, profile
    )

    if selected is None:
        raise SystemExit(f"No ubatch completed the full curve. Results recorded in {profile_path}")
    print(f"Selected ubatch: {selected}")
    print(f"Calibration profile: {profile_path}")


if __name__ == "__main__":
    main()
