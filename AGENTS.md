# AI Agent Context: AI Toolbox Cockpit

## Scope

This project is a Python/Textual TUI for managing local AI toolbox containers
across hardware platforms and software backends.

## Architecture rules

- HARD GATE — speculative decoding controls (user, 14 November 2026): Use a clearly named **Speculative decoding** section with **Disabled** and standard supported method names such as **MTP**, **DFlash2** or **DSpark**. The terms “Target only” and “Bundled draft” are forbidden in user-facing controls, help and documentation. Expose understandable configurable settings, including draft-token count where supported. Default to the configuration actually recommended by recorded testing; preserve explicit user choices. Never substitute opaque invented terms or hide supported controls. Enforce this rule in UI, command-generation tests, visible-copy checks and release review. Do not claim an untested setting is recommended.

- HARD GATE — product UI copy (user, 14 November 2026): Never put internal policy, agent commentary, implementation rationale, qualification-process narration, or explanations of a previous mistake into Cockpit screens. Do not add random commentary after a fix. UI text must directly help the user operate the current control or understand an actionable error. Keep internal decisions and research in project records. Review the complete visible-copy diff before publishing; remove unsolicited commentary rather than replacing it with another explanation.

- Keep `app.py` as an application shell; backend behavior belongs under
  `ai_toolbox_cockpit/backends/<backend>/`.
- Platform is global state. Backend is a filter or backend-owned panel choice.
- Every toolbox catalogue record uses a complete OCI image reference.
- A toolbox entry represents an engine installation. Hardware-specific builds of the same engine belong in a labelled Server Mode selector. Model, quantization, GPU count, KV cache and serving allocations belong in Server Mode profiles; never duplicate an image into separately installable toolbox entries for those settings.
- User-facing engine selectors must expose named engine variants consistently in Server Mode and Models. Sharing a backend implementation must not hide an engine behind its parent family; curated downloads must match the selected engine.
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

## Git workflow

- Stage all completed task changes before handing work back to the user. Never
  leave completed changes unstaged.
- Adding a speculative-decoding sidecar, catalogue record, selector option, or
  direct CLI flags to an existing backend is a minor integration change. Do not
  classify it as substantial solely because it enables MTP, DFlash, DSpark, or
  another speculative mode.

## Context policy — mandatory gate (14 November 2026)

- Before qualifying a model or creating/updating a serving profile, find the model owner's published native maximum context and any documented extension (with source URL and revision). Exercise that context on the intended engine, hardware, quantization, decoding mode and concurrency. If it fails, establish and record the highest usable context actually tested and the observed reason for the lower result. A short speed curve or a server configured with a large window does not establish full-window input support.
- Set the profile's default to the highest successfully tested usable context for that configuration. Preserve explicit user overrides. Record published context, configured context, actual input/output capacity exercised and tested limits separately; never invent qualification or treat a benchmark allocation as the model's maximum.
- HARD GATE: Cockpit must allow user-entered context lengths above every catalogue/default/tested/published value. No UI, command builder, model schema or hidden clamp may enforce an upper context ceiling. Defaults and published/tested values are information, never permission limits. Validate integer syntax and positivity, and retain independently required allocation relationships; let the engine report its actual capacity constraints. This prohibition applies to every backend and future integrations.
- Context changes must pass regression coverage that larger explicit values reach the generated command unchanged and saved overrides survive UI refresh. Never reintroduce the former 67,840-token profile ceiling or generic 8K Radiance default.
