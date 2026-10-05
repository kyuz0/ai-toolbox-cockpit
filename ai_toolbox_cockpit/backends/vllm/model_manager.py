"""Pinned HF artifacts and CPU checkpoint preparation for vLLM."""
import json
import os
from pathlib import Path
import struct
import sys


def incomplete_files(entry: dict, directory: Path) -> list[str]:
    return [item["path"] for item in entry["download"]["files"]
            if not (directory / item["path"]).is_file()
            or (directory / item["path"]).stat().st_size != item["size_bytes"]]


def checkpoint_ready(directory: Path, *, draft: bool = False) -> bool:
    """Check config, all indexed shards and safetensors payload boundaries."""
    try:
        json.loads((directory / "config.json").read_text())
        if not draft:
            for name in ("tokenizer.json", "tokenizer_config.json"):
                json.loads((directory / name).read_text())
        index_path = directory / "model.safetensors.index.json"
        if index_path.is_file():
            files = set(json.loads(index_path.read_text())["weight_map"].values())
        else:
            files = {"model.safetensors"}
        if not files:
            return False
        for name in files:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                return False
            weight = directory / path
            with weight.open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                if not 0 < length <= 64 * 1024 * 1024:
                    return False
                header = json.loads(stream.read(length))
            ends = [value["data_offsets"][1] for key, value in header.items() if key != "__metadata__"]
            if not ends or weight.stat().st_size != 8 + length + max(ends):
                return False
        return True
    except (OSError, ValueError, KeyError, IndexError, TypeError, struct.error):
        return False


def get_download_cmd(entry: dict, directory: Path) -> list[str]:
    hf = Path(sys.executable).parent / "hf"
    executable = str(hf) if hf.is_file() else "hf"
    return [executable, "download", entry["repo"], "--revision", entry["revision"],
            "--local-dir", str(directory), *[item["path"] for item in entry["download"]["files"]]]


def build_prepare_cmd(engine: str, image: str, source: Path, destination: Path) -> list[str]:
    source, destination = source.expanduser().resolve(), destination.expanduser().resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Original and prepared checkpoints must be separate directories")
    for directory in (source, destination):
        if any(character in str(directory) for character in (":", "\n", "\r", "\0")):
            raise ValueError("Checkpoint paths cannot contain colons or control characters")
    if engine not in {"podman", "docker"}:
        raise ValueError("Choose Podman or Docker")
    command = [engine, "run", "--rm", "--network=none", "--memory=24g"]
    if engine == "podman":
        command += ["--userns=keep-id"]
    else:
        command += ["--user", f"{os.getuid()}:{os.getgid()}"]
    return command + ["--entrypoint", "/opt/vllm/bin/python", "-v", f"{source}:/models/source:ro",
                      "-v", f"{destination}:/models/prepared", image, "/opt/ggz14/fp8_mtp.py",
                      "/models/source", "/models/prepared"]
