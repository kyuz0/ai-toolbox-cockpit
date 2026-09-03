from .base import BackendDefinition
from .comfyui.models import ComfyUiModelPanel
from .comfyui.server import ComfyUiServerPanel
from .ds4.models import Ds4ModelPanel
from .ds4.server import Ds4ServerPanel
from .llama_cpp.models import LlamaCppModelPanel
from .llama_cpp.server import LlamaCppServerPanel
from .vllm.models import VllmModelPanel
from .vllm.server import VllmServerPanel


BACKENDS: dict[str, BackendDefinition] = {
    "llama_cpp": BackendDefinition("llama_cpp", "llama.cpp", LlamaCppServerPanel, LlamaCppModelPanel),
    "vllm": BackendDefinition("vllm", "vLLM", VllmServerPanel, VllmModelPanel),
    "comfyui": BackendDefinition("comfyui", "ComfyUI", ComfyUiServerPanel, ComfyUiModelPanel),
    "ds4": BackendDefinition("ds4", "DwarfStar (ds4)", Ds4ServerPanel, Ds4ModelPanel),
}


def backend_options() -> list[tuple[str, str]]:
    return [(definition.label, backend_id) for backend_id, definition in BACKENDS.items()]
