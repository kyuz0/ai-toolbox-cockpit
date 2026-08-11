import json
import tempfile
import unittest
from pathlib import Path

from ai_toolbox_cockpit.backends.llama_cpp.benchmark_runner import BenchmarkJob
from ai_toolbox_cockpit.backends.llama_cpp.ubatch_calibration import (
    DEFAULT_DEPTHS,
    build_calibration_jobs,
    read_candidate,
    select_ubatch,
)


def candidate(ubatch: int, values: list[float]) -> dict:
    return {
        "ubatch": ubatch,
        "complete": True,
        "points": [
            {
                "depth": depth,
                "avg_ts": value,
                "stddev_ts": 0.1,
                "samples_ts": [value, value, value],
            }
            for depth, value in zip(DEFAULT_DEPTHS, values)
        ],
    }


class UbatchCalibrationTest(unittest.TestCase):
    def test_jobs_cover_every_depth_with_prefill_only(self):
        jobs = build_calibration_jobs(
            toolbox_command="toolbox",
            toolbox_name="llama-vulkan-radv-performance",
            model_path="/models/model.gguf",
            output_dir=Path("/tmp/results"),
            ubatches=(256, 512, 1024, 2048),
        )
        self.assertEqual(len(jobs), 4)
        expected_depths = ",".join(str(depth) for depth in DEFAULT_DEPTHS)
        for job, expected_ubatch in zip(jobs, (256, 512, 1024, 2048)):
            self.assertEqual(job.command[job.command.index("-p") + 1], "2048")
            self.assertEqual(job.command[job.command.index("-n") + 1], "0")
            self.assertEqual(job.command[job.command.index("-d") + 1], expected_depths)
            self.assertEqual(
                job.command[job.command.index("-ub") + 1], str(expected_ubatch)
            )

    def test_selection_weights_each_depth_equally(self):
        shallow_specialist = candidate(256, [1000.0] + [50.0] * 8)
        consistent = candidate(512, [900.0] + [100.0] * 8)
        self.assertEqual(
            select_ubatch([shallow_specialist, consistent], DEFAULT_DEPTHS),
            512,
        )

    def test_incomplete_candidate_is_not_selected(self):
        incomplete = candidate(2048, [200.0] * 9)
        incomplete["complete"] = False
        complete = candidate(1024, [100.0] * 9)
        self.assertEqual(select_ubatch([incomplete, complete], DEFAULT_DEPTHS), 1024)

    def test_read_candidate_requires_all_depths_and_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate.jsonl"
            records = [
                {
                    "n_depth": depth,
                    "n_ubatch": 256,
                    "avg_ts": 10.0,
                    "stddev_ts": 0.1,
                    "samples_ts": [10.0, 10.0, 10.0],
                }
                for depth in DEFAULT_DEPTHS
            ]
            output.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            job = BenchmarkJob(
                toolbox_name="llama-rocm-7.14",
                model_path="/models/model.gguf",
                series="prefill",
                command=("llama-bench", "-ub", "256"),
                output_path=output,
                stderr_path=output.with_suffix(".stderr.log"),
            )
            self.assertTrue(read_candidate(job, DEFAULT_DEPTHS)["complete"])


if __name__ == "__main__":
    unittest.main()
