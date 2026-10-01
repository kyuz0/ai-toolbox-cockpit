"""DS4 curated model policy."""

from pathlib import Path

from ai_toolbox_cockpit.catalog import load_model_catalog

# Tensor-parallel workers for these families are served by a different CLI than the
# toolbox default. DeepSeek V4.1 Flash workers run `ds4`; every other role and
# family keeps the binary declared by the toolbox profile.
TENSOR_PARALLEL_WORKER_BINARIES: dict[str, str] = {"deepseek-v4.1-flash": "ds4"}
DEFAULT_SERVER_BINARY = "ds4-server"
DISTRIBUTED_TRANSPORT_TCP = "tcp"
DISTRIBUTED_TRANSPORT_ROCE = "roce"
DISTRIBUTED_TRANSPORT_INFINIBAND = "infiniband"
_RDMA_UI_TRANSPORTS = {
    "rdma",  # Backward-compatible generic CLI value.
    "roce",
    "rocev2",
    "ib",
    "infiniband",
}


def normalize_distributed_transport(transport: str) -> str:
    """Map a cockpit fabric choice to the value accepted by the DS4 CLI.

    DS4 exposes one wire-level ``rdma`` transport for both RoCE and native
    InfiniBand.  The cockpit keeps the link types distinct so its RDMA endpoint
    controls and guidance do not imply that a RoCE GID is valid on IPoIB.
    """
    value = str(transport).strip().casefold()
    if value in {"", DISTRIBUTED_TRANSPORT_TCP}:
        return value
    if value in _RDMA_UI_TRANSPORTS:
        return "rdma"
    raise ValueError("Transport must be TCP, RoCEv2, or InfiniBand")


def is_rdma_transport(transport: str) -> bool:
    """Whether a cockpit transport choice uses the DS4 RDMA transport."""
    try:
        return normalize_distributed_transport(transport) == "rdma"
    except ValueError:
        return False


def load_models() -> dict:
    backend = load_model_catalog().backends["ds4"]
    return {
        "repo": backend.config.get("default_repo", "antirez/deepseek-v4-gguf"),
        "families": dict(backend.config.get("families", {})),
        "default_server_defaults": dict(backend.config.get("default_server_defaults", {})),
        "models": [dict(entry) for entry in backend.entries],
    }


def get_model_artifact(model_path: str) -> dict:
    """Return the exact DS4 catalogue record for a local artifact, if known."""
    filename = Path(model_path).name
    for model in load_models().get("models", []):
        if model.get("filename") == filename:
            return model
    return {}


def get_model_family(model_path: str) -> str:
    """Return the curated family for a local artifact, or an empty string."""
    return str(get_model_artifact(model_path).get("family", ""))


def is_tensor_parallel_cli_worker(model_path: str, role: str, tensor_parallel: bool) -> bool:
    """True when the role runs the plain ds4 CLI instead of the ds4-server daemon.

    The ds4 CLI has no HTTP listener, so it takes no --host/--port options.
    """
    return (
        bool(tensor_parallel)
        and str(role).lower() == "worker"
        and get_model_family(model_path) in TENSOR_PARALLEL_WORKER_BINARIES
    )


def resolve_server_binary(
    model_path: str,
    role: str,
    tensor_parallel: bool,
    default_binary: str = DEFAULT_SERVER_BINARY,
) -> str:
    """Return the server binary for a role, honouring family-specific TP workers."""
    if is_tensor_parallel_cli_worker(model_path, role, tensor_parallel):
        return TENSOR_PARALLEL_WORKER_BINARIES[get_model_family(model_path)]
    return default_binary


def get_artifact_role(model_path: str) -> str:
    """Classify main and auxiliary DS4 GGUFs without mixing their semantics."""
    model = get_model_artifact(model_path)
    if model:
        return str(model.get("artifact_role", "main"))

    filename = Path(model_path).name.lower()
    if "dspark-support" in filename:
        return "dspark_support"
    if "vision-encoder" in filename:
        return "vision_encoder"
    if "mtp" in filename:
        return "mtp"
    return "main"


def get_model_server_defaults(model_path: str) -> dict:
    filename = Path(model_path).name
    data = load_models()
    families = data.get("families", {})
    result = dict(data.get("default_server_defaults", {}))
    model = get_model_artifact(filename)
    if model:
        family = model.get("family")
        if family in families:
            result.update(families[family])
        result.update(model.get("server_defaults", {}))
        return result
    if "DEEPSEEK-V4.1-FLASH" in filename.upper():
        result.update(families.get("deepseek-v4.1-flash", {}))
    elif "GLM-5.3-FLASH" in filename.upper():
        result.update(families.get("glm-5.3-flash", {
            "standalone_ctx": 262144,
        }))
    elif "GLM" in filename.upper():
        # Unknown GLM artifacts must not inherit removed model-family policy.
        return result
    else:
        result.update(families.get("deepseek-v4", {
            "coordinator_layers": "0:21",
            "worker_layers": "22:output",
        }))
    return result
