# Container and toolbox security review

Reviewed 2026-09-16 at commit `cc8dbe3e95a136ebdfbdd6cb39fa41c6e96b212c`.

Halogen follow-up, 2026-09-18: Cockpit now launches this server with
`--network=none` and individual read-only mounts for the selected bundle files.
The other findings below describe the original review unless stated otherwise.

This is a source and command-construction review, assuming a compromised image,
model repository, or backend dependency. It is not evidence of an actual
compromise or a demonstrated kernel escape. No containers, GPU servers, model
downloads, or toolbox sessions were started. Installed runtime versions,
rootless status, image contents/entrypoints, daemon configuration, firewall
rules, and effective permissions on the remote GPU hosts remain unverified.

The central finding is that interactive Toolbx/Distrobox environments are trusted
host-integrated development environments. Server Mode creates separate direct
Podman/Docker containers with substantially narrower host mounts, but its current
defaults are not suitable as a strong boundary against hostile image code.

## Findings, ordered by priority

### 1. High; potentially critical with rootful Docker: interactive host access

`runtime/interactive.py:201–223` builds plain `toolbox create --image IMAGE NAME`
or `distrobox create ... --additional-flags ...`. Toolbx ignores Cockpit's
`engine_args` entirely; Distrobox receives them but no unsharing options.
Changing the catalogue alone therefore cannot isolate Toolbx.

Upstream Toolbx currently creates privileged containers with host network, PID
and IPC namespaces, SELinux separation disabled, and mounts including the host
root at `/run/host`, `/dev`, home, runtime directory and D-Bus. This describes
upstream defaults, not an inspection of the user's installed version.
[Toolbx source](https://github.com/containers/toolbox/blob/main/src/cmd/create.go).

Distrobox also intentionally shares home and host integration resources; its
documentation explicitly warns about host modification with rootful engines.
[Distrobox security implications](https://distrobox.it/#security-implications).

Consequences of a malicious image include reading SSH/cloud/registry credentials
and Cockpit's saved HF token, changing shell startup files or user services,
and poisoning models or code in the user's home. These attacks require no kernel
breakout. SSH-agent and desktop/D-Bus socket access adds delegated host authority
where available. Read/write access still depends on host permissions and user
namespace mappings; a rootless container's root is not automatically host root.

Cockpit detects executable availability, not daemon privilege mode
(`runtime/engines.py:12`, `runtime/interactive.py:45`). Running the Docker CLI
without sudo does not prove a rootless daemon. If a shared host/runtime path
exposes an accessible rootful engine socket, that provides another route to host
root. Socket availability and permissions must be inspected, not assumed.
[Docker daemon security](https://docs.docker.com/engine/security/).

Recommendation: describe interactive sessions as trusted host access; use direct
rootless containers for routine inference. Prefer a dedicated service account
with no personal credentials or Docker administration access. For genuinely
untrusted GPU workloads, consider a dedicated machine or VM with supported GPU
passthrough, since containers still share the kernel and GPU driver.

### 2. High: DS4 multi-node ignores the API Host setting

`backends/ds4/server_runner.py:92–118` switches to `--network=host` for any
non-standalone role, removes port publishing, and still passes the server
`--host 0.0.0.0`. The UI defaults Host to `localhost`.

Thus selecting Coordinator/Worker can configure an API listener on all host IPv4
interfaces despite a localhost Host field. Actual reachability depends on the
backend role starting that listener and host firewall rules. The builder adds
no API authentication itself. This is a concrete mismatch in command
construction, separate from general outbound-network exposure.

Recommendation: honor the API bind address in host-network mode and separate it
from the inter-node transport address. Require explicit distributed networking
configuration and restrict peers/ports on the host firewall. Keep standalone
mode as the ordinary isolated path.

### 3. High: vLLM combines credentials, writable caches and remote code

`backends/vllm/runner.py:81–124` mounts the complete default
`~/.cache/huggingface`, `~/.cache/vllm`, `~/.cache/triton`, and `~/.aiter`
read/write. It forwards `HF_TOKEN`, including ambient `HF_TOKEN` when no explicit
token is supplied. Nine of fifteen current curated policies enable
`--trust-remote-code`; the builder does not pin model/code revisions by default.

The HF root can also contain a saved login token at
`~/.cache/huggingface/token`. Removing the environment variable alone would not
remove this exposure. [HF token storage](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hf_token_path).

A malicious image can read those credentials and overwrite shared model or
compiled-code caches. Repository code enabled by `trust_remote` runs with the
same container access. Cache changes survive `--rm` and may affect later runs or
host tools that consume them. Console redaction exists, but does not hide secrets
from container code, engine metadata, or every process-inspection mechanism.

Recommendation: download pinned snapshots separately; mount only the selected
snapshot and its required blobs read-only; keep inference token-free; give each
backend dedicated writable compilation caches. Require explicit acceptance of
repository code and pin its revision as well as model weights.

### 4. High: server profiles weaken several independent protections

The following are the explicit Cockpit defaults. Runtime/image configuration can
add further access, so this table is not an effective-permissions audit.

| Server | Explicit host mounts | Additional exposure |
| --- | --- | --- |
| llama.cpp | Entire configured model root read-only | AMD/Intel profiles disable seccomp; Podman disables SELinux separation and uses keep-id; RDMA devices auto-added on Strix Halo when present |
| DS4 | Model root read-only; optional KV directory read/write | Host IPC, `SYS_PTRACE`, profile device/seccomp settings; Podman label disable/keep-id; multi-node host networking |
| vLLM | HF, vLLM, Triton, AITER caches read/write | Host IPC, `SYS_PTRACE`, HF token, optional remote code; Podman label disable/keep-id; Docker explicit host UID/GID |
| ComfyUI | Models, input, output and user directories all read/write | Host IPC, `SYS_PTRACE`, profile device/seccomp settings; Podman label disable/keep-id; Docker explicit host UID/GID |
| Gufo | Selected target directory and optional speculative sidecar directory read-only | Host IPC, profile ROCm devices and unconfined seccomp; Podman label disable/keep-id |
| R9V | Model root and PLE file read-only; cache read/write | Explicit `--user 0:0`, host IPC, label disable, ROCm devices/unconfined seccomp |
| Halogen (updated 2026-09-18) | Selected checkpoint, overlay, tokenizer files and optional vision weights read-only | No container network; inbound API relay; ROCm devices, unconfined seccomp, host IPC, unlimited memlock; user inherited from image; always-pull profile |

Evidence: `assets/toolboxes.json:3–93`; `backends/llama_cpp/server_runner.py:75–113`;
`backends/ds4/server_runner.py:72–107`; `backends/vllm/runner.py:81–108`;
`backends/comfyui/runner.py:60–76`; `backends/gufo/server_runner.py`; `backends/r9v/runner.py:108–114`;
`backends/halogen/runner.py` (updated since the original review).

Disabling seccomp removes syscall filtering. `label=disable` removes SELinux
container separation where SELinux applies. Host IPC shares an otherwise useful
boundary. `SYS_PTRACE` enlarges debugging authority, but does not by itself prove
that a server can inspect arbitrary host processes: PID/user namespaces and
permissions still matter. These direct server builders do not explicitly request
`--privileged`, host PID, full home, or engine-socket mounts.
[Podman run security options](https://docs.podman.io/en/latest/markdown/podman-run.1.html).

R9V explicitly runs as container root; Docker llama.cpp, DS4 and Halogen leave
the user to the image. Rootful Docker without user remapping makes container UID
0 host UID 0, though remaining container restrictions still apply. Rootless
Podman avoids that identity equivalence. Neither engine's mode is enforced here.

GPU devices expose host kernel drivers, and visibility environment variables are
not device-access controls. `/dev/dri` exposes more than one render node where
present; NVIDIA requests all GPUs. Podman's `keep-groups` preserves supplementary
host groups, not just video/render. Distrobox automatically gets RDMA devices
when present (`runtime/interactive.py:166`), even without a distributed workload.
Treat RDMA as a separate networking/device exposure requiring validation.

Recommendation: test default seccomp, private IPC with adequate shared memory,
no `SYS_PTRACE`, retained LSM isolation, explicit non-root identity,
`no-new-privileges`, dropped capabilities, narrow device nodes, read-only root
with dedicated writable cache/tmp paths, and appropriate resource limits.
Maintain only demonstrated backend-specific exceptions. Existing comments about
ROCm requirements are not validation that every backend needs these privileges.

### 5. High impact if compromised: mutable images and permissive profiles

All 27 catalogue image records use mutable tags; none pins a digest.
`runtime/toolboxes.py:109–111` pulls before creation; Halogen additionally uses
`--pull=always` with a `:latest` image. Ordinary server builders also reference
tags, so launch identity can change when local tags are updated or missing.
Cockpit adds no image signer/provenance check; an independently configured
engine trust policy may still apply. Image update dates are not authenticity.

`catalog/schema.py:469–472` validates engine arguments only as strings. Catalogue
data can therefore request new mounts or privileges through these arguments.
The catalogue is bundled package data, not a discovered unauthenticated remote
feed (`catalog/loader.py:7–17`), so this is a hardening gap, not a demonstrated
remote injection path. Argument-list subprocess execution and backend commands
placed after the image are positive controls against shell/engine injection.

Recommendation: distribute reviewed image digests, verify expected publishers
and build provenance, record actual image identity, and use explicit updates.
Enforce security policy around permitted engine options instead of allowing
arbitrary JSON options to weaken it. Digest pinning provides stable identity,
not proof that an image is benign.

Model integrity is backend-specific: R9V pins revisions and implements SHA256
verification; Gufo and Halogen pin revisions but their readiness checks verify sizes.
Do not describe all model downloads as cryptographically verified.

## Network isolation: practical options

### Implemented for Halogen

Halogen is a third-party closed-source project and container. **Use at your own
risk:** unlike the open-source backends, its implementation is not readily
available for inspection and audit. See the [upstream project](https://github.com/peonist-ai/halogen-flash-server).

The Halogen builder enforces `--network=none`, drops `NET_ADMIN`/`NET_RAW`, and
sets `no-new-privileges`. Only explicitly reviewed GPU profile arguments are
accepted; additional network, mount, privilege, or environment arguments fail
the launch. It mounts individual selected model and tokenizer files read-only,
including the vision tower only for a vision bundle. It does not mount the
whole model directory, home, credentials, or engine sockets. GPU device access
and the existing profile's host IPC, memlock, and seccomp exceptions remain.

`runtime/isolated_api.py` accepts connections on the configured host API address
and uses engine exec stdin/stdout to reach a fixed loopback API port inside the
container. The image's Python interpreter runs the stream helper. There is no
published container port, shared host socket mount, or general outbound proxy.
The listener defaults to localhost, supports streaming, and is closed with its
exec clients when serving ends. Binding a public interface still exposes the
API; this change adds no API authentication. Image pulls and model downloads
use the host's network separately from the running container.

This limits direct network access and readable host data; it does **not** make
the closed-source image safe. The host kernel/GPU driver remain shared, the
image tag is mutable, and a malicious server can return accessible data in an
allowed API response. Local command and mocked lifecycle checks do not prove
effective isolation. Validate both engines on the remote GPU host as described
in `REMOTE_VALIDATION.md`, including IPv4/IPv6 egress and streaming API access.

### Other workloads and alternatives

Local models make offline inference plausible, but Cockpit does not yet download
everything outside containers. vLLM downloads repositories at server startup
(`backends/vllm/models.py:23–26`); ComfyUI invokes `model_manager` inside the
host-integrated toolbox (`backends/comfyui/models.py:100–105`). These preparation
paths need changes or separate controlled provisioning.

| Workload | Proposed network policy | Effort / limitation |
| --- | --- | --- |
| Batch inference or model conversion | Direct rootless container with `--network=none` | Small command change once artifacts are local; remove port publishing. R9V PLE preparation already does this. |
| HTTP server accepting local requests | Dedicated private network, tightly controlled ingress and denied new outbound connections | Moderate runtime work; verify implementation separately for Docker and rootless Podman. |
| Distributed inference | Explicit allowlist of peer addresses and ports | Greater effort; isolate API binding from peer traffic and account for RDMA. |
| Interactive Toolbx | Use direct-container isolation for untrusted work | Toolbx's host integration defeats a simple profile-level network flag. |
| Interactive Distrobox | Unshare network plus engine network policy | Namespace separation alone does not block egress; shared host resources remain an escape from the intended restriction. |

`--network=none` leaves container loopback only; normal host `-p` access to a
server will not work. It is a good batch policy, not a drop-in replacement for
an HTTP server. [Docker none networking](https://docs.docker.com/engine/network/drivers/none/).

For HTTP, use an engine-tested internal bridge with a constrained API relay, or
a stateful policy allowing API requests/replies while blocking container-initiated
traffic. An internal network is only a starting point: Docker explicitly allows
gateway/host communication and communication with peers on that network.
Rootless Podman differs in implementation, so do not promise that an identical
`--internal` plus `-p` recipe behaves identically across engines. Block access to
host services, other containers, LAN, Internet and unwanted DNS, including IPv6.
[Docker internal networks](https://docs.docker.com/reference/cli/docker/network/create/#internal),
[Podman internal networks](https://docs.podman.io/en/latest/markdown/podman-network-create.1.html#internal).

Distrobox offers `--unshare-netns` and `--unshare-all`, but even `--unshare-all`
retains home, basic sockets and host filesystem access. `--home` changes the
container's home location without removing the original host home mount. These
are insufficient as a malicious-code sandbox.
[Distrobox creation options](https://distrobox.it/usage/distrobox-create/).

Use HF offline settings for predictable local artifact loading; they are
cooperative library settings, not a firewall. Pulling images happens through the
host engine and is separate from the container's runtime network policy.
Even blocked outbound connections cannot prevent a malicious server returning
accessible secrets in an allowed API response. Mount/credential minimization
must accompany network restrictions.

## Suggested implementation order and validation

1. Fix DS4's Host mismatch and expose effective security/network settings in the
   launch preview, including engine rootless status and image digest.
2. Introduce a direct rootless server policy, starting with standalone llama.cpp:
   local read-only model, no credentials, controlled API ingress and blocked egress.
3. Separate vLLM provisioning from inference; isolate writable caches and stop
   mounting the credential-bearing HF root. Make ComfyUI models read-only and
   restrict writable inputs/user/output according to actual backend needs.
4. Remove broad privilege defaults one backend at a time, keeping explicit,
   validated compatibility exceptions. Make RDMA opt-in.
5. Pin/verify images and repository code; present interactive wrappers as trusted
   development environments with a separate risk profile.

Local validation: the 54 existing `*commands.py` tests passed. A separate pure
builder check reproduced DS4's localhost-to-all-interfaces mismatch and counted
27 unpinned image records and 9/15 remote-code vLLM policies. These checks do not
prove runtime isolation.

On the user's remote GPU systems, follow `REMOTE_VALIDATION.md`: one backend and
one already-downloaded known-good model at a time. Record effective mounts,
read/write status, UID mappings, groups, capabilities, seccomp/LSM state,
namespaces, devices and image digest without dumping credentials. Verify API
reachability from allowed and disallowed locations and attempted outbound access
to controlled Internet, LAN, host and DNS endpoints over IPv4/IPv6. Verify denied
writes to a dedicated test model mount, absence of home/agent/engine sockets,
cache scope, GPU operation and cleanup. Test distributed/RDMA separately.

The original review changed no production behavior. The Halogen follow-up
implements the network and mount restrictions described above; other backends
are unchanged.
