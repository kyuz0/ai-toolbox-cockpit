# AI Agent Context: AI Toolbox Cockpit

## Scope

This project is a Python/Textual TUI for managing local AI toolbox containers
across hardware platforms and software backends.

## Architecture rules

- Keep `app.py` as an application shell; backend behavior belongs under
  `ai_toolbox_cockpit/backends/<backend>/`.
- Platform is global state. Backend is a filter or backend-owned panel choice.
- Every toolbox catalogue record uses a complete OCI image reference.
- Backend IDs are explicit and registered in Python. Do not dynamically execute
  code or shell templates from JSON.
- Each backend owns its Server UI and model semantics. Do not create a universal
  model abstraction that treats GGUF files, Hugging Face repositories, and
  ComfyUI workflow bundles as identical.
- Shared Podman/Docker and Toolbx/Distrobox logic belongs under `runtime/`.
- Batch selection must use an explicit selected-ID set, never cursor highlight.
- Every dropdown or select control must have a persistent visible label stating
  what it controls. Placeholder text, the selected value, and surrounding section
  headings do not count as labels.
- Interactive subprocesses must run while the Textual app is suspended.

## Testing

- Pure catalogue and command builders may be tested locally.
- Do not start containers, GPU servers, model downloads, or toolbox sessions on
  the local development machine.
- Runtime behavior must be tested on the user's remote GPU systems, one backend
  and one known-good model at a time.
