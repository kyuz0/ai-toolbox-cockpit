from fnmatch import fnmatchcase
from pathlib import Path

from ai_toolbox_cockpit.catalog import load_model_catalog, load_toolbox_catalog

def load_models() -> list[dict]:
    return [dict(entry) for entry in load_model_catalog().backends["llama_cpp"].entries]

def load_toolboxes() -> dict:
    catalog = load_toolbox_catalog()
    platforms = []
    for platform in catalog.platforms:
        groups: dict[str, list[dict]] = {}
        repositories: set[str] = set()
        for toolbox in catalog.platform_toolboxes(platform.id):
            if toolbox.backend != "llama_cpp":
                continue
            repositories.add(toolbox.image.rsplit(":", 1)[0])
            groups.setdefault(toolbox.group, []).append({
                "name": toolbox.container_name,
                "tag": toolbox.image.rsplit(":", 1)[-1],
                "description": toolbox.description,
                "supports_load_mode": toolbox.supports_load_mode,
                "engine_args": list(catalog.runtime_profiles[toolbox.runtime_profile].engine_args),
            })
        platforms.append({
            "id": platform.id,
            "name": platform.name,
            "description": platform.description,
            "registry": next(iter(repositories)) if len(repositories) == 1 else "",
            "groups": [
                {"name": group_name, "toolboxes": toolboxes}
                for group_name, toolboxes in groups.items()
            ],
        })
    return {"platforms": platforms}

def get_platforms() -> list[dict]:
    """Returns the list of platform definitions from toolboxes.json."""
    data = load_toolboxes()
    return data.get("platforms", [])

def get_platform(platform_id: str) -> dict | None:
    """Returns a single platform dict by its ID, or None if not found."""
    for p in get_platforms():
        if p.get("id") == platform_id:
            return p
    return None

def get_platform_registry(platform_id: str) -> str:
    """Compatibility helper; unified platforms can span image repositories."""
    catalog = load_toolbox_catalog()
    repositories = {
        toolbox.image.rsplit(":", 1)[0]
        for toolbox in catalog.platform_toolboxes(platform_id)
        if toolbox.backend == "llama_cpp"
    }
    return next(iter(repositories)) if len(repositories) == 1 else ""


def get_model_config(selected_path: str) -> dict | None:
    """Look up a curated model entry by fuzzy-matching repo basename against a local file path."""
    if not selected_path:
        return None
    curated = load_models()
    path_lower = selected_path.lower()
    path_norm = path_lower.replace("-", "").replace("_", "")
    
    candidates = []
    for m in curated:
        repo_basename = m["repo"].split("/")[-1].lower()
        
        # 1. Exact repo_basename in path (e.g. folder name match)
        if repo_basename in path_lower:
            candidates.append((len(repo_basename), 3, m))
            continue
            
        # Clean suffix/infixes like -gguf, -gguf-mtp
        clean_basename = repo_basename
        if clean_basename.endswith("-gguf"):
            clean_basename = clean_basename[:-5]
        elif "-gguf-" in clean_basename:
            clean_basename = clean_basename.replace("-gguf-", "-")
            
        # 2. Cleaned basename in path
        if clean_basename in path_lower:
            candidates.append((len(clean_basename), 2, m))
            continue
            
        # 3. Normalized matching (ignoring hyphens and underscores)
        clean_norm = clean_basename.replace("-", "").replace("_", "")
        if clean_norm in path_norm:
            candidates.append((len(clean_norm), 1, m))
            
    if not candidates:
        return None
        
    # Sort candidates by:
    # - Strategy priority (3 = exact, 2 = clean, 1 = normalized) descending
    # - Match length descending (longer/more specific match wins)
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return candidates[0][2]



def get_inference_profiles(model_config: dict) -> dict:
    """Returns the inference_profiles dict for a model, or empty dict if none."""
    if not model_config:
        return {}
    return model_config.get("inference_profiles", {})


def get_default_inference_profile(model_config: dict) -> str | None:
    """Return an explicit valid default profile, then the first profile for legacy entries."""
    profiles = get_inference_profiles(model_config)
    if not profiles:
        return None
    configured = model_config.get("default_inference_profile")
    if configured in profiles:
        return str(configured)
    return next(iter(profiles))


def get_toolbox_defaults(model_config: dict, toolbox_id: str) -> dict:
    """Return typed defaults for an exact curated model/toolbox combination."""
    if not model_config or not toolbox_id:
        return {}
    return dict(model_config.get("toolbox_defaults", {}).get(toolbox_id, {}))


def get_recommended_use(toolbox) -> dict | None:
    """Return backend-owned guidance for a purpose-built llama.cpp toolbox."""
    if toolbox is None:
        return None
    recommended = (toolbox.backend_config or {}).get("recommended_use")
    return dict(recommended) if recommended else None


def recommended_use_matches_model(
    recommended: dict | None,
    model_config: dict | None,
    selected_path: str = "",
    *,
    require_filename: bool = False,
) -> bool:
    """Check the curated model, and optionally its tested quant filename."""
    if not recommended or not model_config:
        return False
    if model_config.get("id") != recommended.get("model_id"):
        return False
    if not require_filename:
        return True
    return bool(selected_path) and fnmatchcase(
        Path(selected_path).name.lower(),
        str(recommended.get("model_filename_pattern", "")).lower(),
    )


def get_recommended_server_defaults(
    toolbox, model_config: dict | None
) -> dict:
    """Resolve defaults only for the model family targeted by the toolbox."""
    recommended = get_recommended_use(toolbox)
    if not recommended_use_matches_model(recommended, model_config):
        return {}
    return dict(recommended.get("server_defaults", {}))


def get_calibrated_ubatch_defaults(
    model_config: dict,
    selected_path: str,
    toolbox_id: str,
    serving_config: str,
    kv_cache_type: str,
) -> dict:
    """Resolve one exact llama.cpp calibration for the selected launch identity."""
    if not model_config or not selected_path or not toolbox_id:
        return {}
    records = load_model_catalog().backends["llama_cpp"].config.get(
        "calibrated_ubatches", []
    )
    filename = Path(selected_path).name.lower()
    matches = [
        record
        for record in records
        if record["model_id"] == model_config.get("id")
        and record["toolbox_id"] == toolbox_id
        and fnmatchcase(filename, record["filename_pattern"].lower())
        and record["serving_config"] == serving_config
        and record["kv_cache_type"] == kv_cache_type
    ]
    if len(matches) > 1:
        raise ValueError(
            "Ambiguous calibrated ubatch for "
            f"{model_config.get('id')} / {toolbox_id} / {filename}"
        )
    if not matches:
        return {}
    return {
        "batch_size": matches[0]["batch_size"],
        "ubatch_size": matches[0]["ubatch_size"],
    }


def get_mtp_config(model_config: dict) -> dict | None:
    """Returns the mtp config dict for a model, or None if MTP is not supported."""
    if not model_config:
        return None
    mtp = model_config.get("mtp")
    if mtp and mtp.get("supported"):
        return mtp
    return None


def get_effective_mtp_config(model_config: dict | None, toolbox=None) -> dict | None:
    """Overlay a fork's structured MTP recipe on the model's base metadata."""
    base = get_mtp_config(model_config)
    recommended = get_recommended_use(toolbox)
    if not recommended_use_matches_model(recommended, model_config):
        return base
    defaults = recommended.get("server_defaults", {})
    mtp_defaults = defaults.get("mtp")
    if not mtp_defaults:
        return base
    effective = dict(base or {"supported": True})
    effective.update(mtp_defaults)
    sidecar = recommended.get("sidecar")
    if sidecar:
        effective["draft_models"] = [sidecar["filename"]]
        effective["sidecar_repo"] = sidecar["repo"]
    return effective


def get_mtp_server_args(mtp: dict, draft: str, sequences: str) -> str:
    """Build MTP flags, including structured multi-drafter recipes."""
    spec_types = mtp.get("spec_types")
    if spec_types:
        args = [
            "--spec-type", ",".join(spec_types),
            "--spec-draft-n-max", draft,
        ]
        if "spec_draft_p_min" in mtp:
            args.extend(["--spec-draft-p-min", str(mtp["spec_draft_p_min"])])
        if "ngram-mod" in spec_types:
            if "ngram_mod_n_max" in mtp:
                args.extend([
                    "--spec-ngram-mod-n-max", str(mtp["ngram_mod_n_max"]),
                ])
            if "ngram_mod_n_match" in mtp:
                args.extend([
                    "--spec-ngram-mod-n-match", str(mtp["ngram_mod_n_match"]),
                ])
        return " ".join(args)
    if mtp.get("draft_models"):
        return (
            "--spec-type draft-mtp"
            " --spec-draft-ngl 99"
            " --spec-draft-device ROCm0"
            f" --spec-draft-n-max {draft}"
            " --spec-draft-n-min 0"
            " --spec-draft-p-min 0.0"
            f" -fit off --parallel {sequences} -dev ROCm0"
        )
    return f"--spec-type draft-mtp --spec-draft-n-max {draft} -np {sequences}"


def get_dspark_config(model_config: dict) -> dict | None:
    """Return the DSpark config for a curated model, if supported."""
    if not model_config:
        return None
    dspark = model_config.get("dspark")
    if dspark and dspark.get("supported"):
        return dspark
    return None


def get_vision_projector_config(model_config: dict) -> dict | None:
    """Return the opt-in vision-projector config for a curated model."""
    if not model_config:
        return None
    config = model_config.get("vision_projector")
    if not config or not config.get("patterns"):
        return None
    return config
