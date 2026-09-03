import os
import re
import glob
import fnmatch
from huggingface_hub import HfApi
from pathlib import Path
from ai_toolbox_cockpit.settings import get_backend_settings, save_backend_settings

def get_models_dir() -> Path:
    configured = get_backend_settings("llama_cpp").get("models_dir")
    if isinstance(configured, str) and configured:
        return Path(configured).expanduser()
    return Path(os.path.expanduser("~/models"))

def save_models_dir(path_str: str) -> bool:
    try:
        new_dir = Path(os.path.expanduser(path_str))
        new_dir.mkdir(parents=True, exist_ok=True)
        return save_backend_settings("llama_cpp", {"models_dir": path_str})
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def scan_local_models() -> list[dict]:
    models_dir = get_models_dir()
    if not models_dir.exists():
        return []
    
    found = set()
    for root, dirs, files in os.walk(models_dir):
        for f in files:
            if f.endswith(".gguf"):
                # Projector files are selected separately in Server Mode; they
                # cannot be used as the main llama-server model.
                # Speculative drafters are auxiliary models and are not
                # standalone chat models.
                if f.lower().startswith(("mmproj", "dspark-", "mtp-")):
                    continue
                path = Path(root) / f
                rel_path = path.relative_to(models_dir)
                
                # Check for sharded models (-0000X-of-0000Y)
                if "-000" in f and "-of-000" in f:
                    grouped_pattern = re.sub(r"-000\d+-of-000\d+\.gguf$", "-*-of-*.gguf", str(rel_path))
                    found.add(grouped_pattern)
                else:
                    found.add(str(rel_path))
                    
    return [{"name": m, "path": str(models_dir / m)} for m in sorted(list(found))]


def get_local_dspark_models(patterns: list[str], default_pattern: str = "") -> list[Path]:
    """Find curated DSpark drafter files beneath the llama.cpp models directory."""
    if not patterns:
        return []

    models_dir = get_models_dir()
    if not models_dir.exists():
        return []

    matches: dict[str, Path] = {}
    for pattern in patterns:
        for candidate in models_dir.glob(f"**/{pattern}"):
            if candidate.is_file():
                matches[str(candidate.resolve())] = candidate

    def sort_key(path: Path) -> tuple[int, str]:
        preferred = bool(default_pattern) and path.as_posix().endswith(default_pattern)
        return (0 if preferred else 1, path.as_posix().lower())

    return sorted(matches.values(), key=sort_key)


def get_local_vision_projectors(
    model_path: str, patterns: list[str]
) -> list[Path]:
    """Find projector GGUFs downloaded alongside the selected model."""
    if not model_path or not patterns:
        return []

    model_file = Path(resolve_model_path(model_path))
    if not model_file.is_file():
        return []

    projectors = {
        candidate
        for pattern in patterns
        for candidate in model_file.parent.glob(pattern)
        if candidate.is_file()
    }
    return sorted(projectors, key=lambda path: path.name.lower())


def get_local_mtp_models(filenames: list[str], repo: str = "") -> list[Path]:
    """Find external MTP GGUFs, optionally restricted to one Hub repository."""
    if not filenames:
        return []

    models_dir = get_models_dir()
    if not models_dir.exists():
        return []
    search_root = models_dir / repo.split("/")[-1] if repo else models_dir
    if not search_root.exists():
        return []
    matches = {
        candidate
        for filename in filenames
        for candidate in search_root.glob(f"**/{filename}")
        if candidate.is_file()
    }
    return sorted(
        matches,
        key=lambda path: path.as_posix().lower(),
    )


def is_quant_downloaded(repo: str, quant: str) -> bool:
    models_dir = get_models_dir()
    if not models_dir.exists():
        return False
        
    repo_base = repo.split('/')[-1].replace('-GGUF', '').lower()
    # Normalized form strips hyphens/underscores for flexible comparison
    repo_norm = repo_base.replace('-', '').replace('_', '')
    
    # 1. Exact path match based on standard download dir
    standard_dir = models_dir / repo.split('/')[-1]
    if (standard_dir / quant).exists():
        return True
    
    def _dir_matches_repo(dirpath: str) -> bool:
        """Check if a directory path is related to this specific repo."""
        rel = os.path.relpath(dirpath, models_dir).lower()
        for part in rel.split(os.sep):
            if part == '.':
                continue
            part_norm = part.replace('-', '').replace('_', '')
            # Require repo_base to be IN the dir name (not the reverse),
            # so "qwen3635ba3b" won't match a dir for "qwen3635ba3bmtp"
            if repo_norm in part_norm:
                return True
        return False
        
    # 2. Fuzzy scan across models_dir
    for root, dirs, files in os.walk(models_dir):
        if quant.endswith(".gguf"):
            # Only match files in directories related to this repo
            if not _dir_matches_repo(root):
                continue
            if "*" in quant:
                for f in files:
                    if fnmatch.fnmatch(f, quant):
                        return True
            else:
                if quant in files:
                    return True
        else:
            # quant is a folder name like "BF16"
            if quant in dirs:
                if _dir_matches_repo(root):
                    return True
                try:
                    for f in os.listdir(os.path.join(root, quant)):
                        if repo_base in f.lower():
                            return True
                except OSError:
                    pass
                        
    return False

def resolve_model_path(pattern_path: str) -> str:
    """Resolves a pattern like *-of-*.gguf to the first actual file."""
    actual_files = glob.glob(pattern_path)
    if actual_files:
        actual_files.sort()
        return actual_files[0]
    return pattern_path

def get_hf_quants(repo: str, token: str = "") -> list[str]:
    api = HfApi(token=token or None)
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="model")
    except Exception:
        return []

    quants = set()
    for f in files:
        if f.endswith(".gguf"):
            parts = f.split('/')
            if len(parts) > 1:
                # It's in a subfolder (e.g., "BF16")
                quants.add(parts[0])
            else:
                # Top level file: Check if it's a shard
                if "-000" in f and "-of-000" in f:
                    grouped_pattern = re.sub(r"-000\d+-of-000\d+\.gguf$", "-*-of-*.gguf", f)
                    quants.add(grouped_pattern)
                else:
                    quants.add(f)
    return sorted(list(quants))


def get_hf_quants_with_sizes(
    repo: str, token: str = ""
) -> tuple[list[str], dict[str, int]]:
    """Return downloadable GGUF groups and their summed Hub file sizes."""
    api = HfApi(token=token or None)
    try:
        entries = api.list_repo_tree(
            repo_id=repo,
            repo_type="model",
            recursive=True,
        )
        files = [
            (entry.path, size)
            for entry in entries
            if isinstance(size := getattr(entry, "size", None), int)
        ]
    except Exception:
        return [], {}

    sizes: dict[str, int] = {}
    directory_quants: set[str] = set()
    for filename, size in files:
        if not filename.endswith(".gguf"):
            continue
        parts = filename.split("/")
        if len(parts) > 1:
            quant = parts[0]
            directory_quants.add(quant)
        elif "-000" in filename and "-of-000" in filename:
            quant = re.sub(
                r"-000\d+-of-000\d+\.gguf$",
                "-*-of-*.gguf",
                filename,
            )
        else:
            quant = filename
        sizes[quant] = sizes.get(quant, 0) + size
    for quant in directory_quants:
        sizes[quant] = sum(
            size for filename, size in files if filename.startswith(f"{quant}/")
        )
    return sorted(sizes), sizes

import sys

def get_download_cmd(repo: str, quant_pattern: str) -> list[str]:
    final_dir = str(get_models_dir() / repo.split('/')[-1])
    
    # Use the hf executable from the current Python environment
    hf_bin = os.path.join(os.path.dirname(sys.executable), "hf")
    if not os.path.exists(hf_bin):
        hf_bin = "hf" # Fallback to PATH if not found
    
    cmd = [
        hf_bin, "download",
        repo,
        "--local-dir", final_dir
    ]
    
    if quant_pattern.endswith(".gguf"):
        # Single file or shard glob: use --include for patterns, positional for exact
        if "*" in quant_pattern:
            cmd.extend(["--include", quant_pattern])
        else:
            cmd.append(quant_pattern)
    else:
        # Folder-based quant (e.g., "BF16", "UD-IQ2_M"): use --include with glob
        cmd.extend(["--include", f"{quant_pattern}/*"])
    
    return cmd
