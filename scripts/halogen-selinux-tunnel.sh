#!/usr/bin/env bash
# Scoped SELinux allow for SSH-tunneled access to the Halogen Flash API port.
#
# Why: the API is published loopback-only (-p 127.0.0.1:8731:8731). An
# `ssh -L 8731:127.0.0.1:8731` tunnel makes sshd-session connect out to that
# port; on targeted/enforcing SELinux the port is unreserved_port_t and the
# connect is denied, so the tunnel fails while local curl works. Niche issue for
# SELinux-enforcing distros only.
#
# Grants (scoped to one port, enforcing stays on): a dedicated halogen_api_port_t
# type on tcp/PORT; sshd_session_t may name_connect it; the container network
# binder (pasta/slirp4netns) may name_bind it. Nothing else is widened.
#
# Usage: scripts/halogen-selinux-tunnel.sh [apply|revert|status] [PORT]
#   apply   (default) install the module and label the port.
#   revert  remove the port label, then the module (safe order).
#   status  show mode, module, port label, binder. Read-only.
# PORT defaults to 8731. Run 'apply' AFTER starting the Halogen server so the
# binder is live. Privileged commands are echoed before they run.

set -euo pipefail

ACTION="${1:-apply}"
PORT="${2:-8731}"
MODULE="halogen_ssh_tunnel"
TYPE="halogen_api_port_t"

say() { printf '%s\n' "$*"; }

# run <cmd...>: silent on success; on failure print the command + output and abort.
run() {
    local out
    if out="$("$@" 2>&1)"; then
        return 0
    fi
    printf 'FAILED: %s\n' "$*" >&2
    [ -n "$out" ] && printf '%s\n' "$out" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: scripts/halogen-selinux-tunnel.sh [apply|revert|status] [PORT]
  apply   (default) label tcp/PORT as halogen_api_port_t; allow sshd connect + binder bind.
  revert  remove the port label, then the module (safe order).
  status  show mode, module, port label, binder. Read-only.
PORT defaults to 8731. Run 'apply' after starting the Halogen server.
EOF
}

require_enforcing() {
    if [ "$(getenforce 2>/dev/null || echo Disabled)" = "Disabled" ]; then
        say "SELinux not enforcing; nothing to do"
        exit 0
    fi
}

require_tools() {
    local t
    for t in checkmodule semodule_package semodule semanage; do
        command -v "$t" >/dev/null || { say "missing '$t' (selinux-policy-devel, policycoreutils)" >&2; exit 1; }
    done
}

# Forwarder PID matched by comm, so a 'podman run --network=...' parent is not used.
binder_pid() {
    ps -eo pid=,comm= 2>/dev/null | awk '$2 ~ /^(pasta|slirp4netns)/ {print $1; exit}' || true
}

detect_binder() {
    local pid
    pid="$(binder_pid)"
    [ -n "$pid" ] || { say "binder not running; start the Halogen server first" >&2; exit 1; }
    BINDER="$(ps -o context= -p "$pid" 2>/dev/null | cut -d: -f3 || true)"
    [ -n "$BINDER" ] || { say "cannot read binder context" >&2; exit 1; }
    say "binder: $BINDER"
}

# Raw policy module (no m4 macros, so 'checkmodule -M' is fine).
write_policy() {
    cat > "$WORKDIR/$MODULE.te" <<EOF
module $MODULE 1.0;

require {
    type sshd_session_t;
    type $BINDER;
    attribute port_type;
    class tcp_socket { name_connect name_bind };
}

type $TYPE;
typeattribute $TYPE port_type;

allow sshd_session_t $TYPE:tcp_socket name_connect;
allow $BINDER $TYPE:tcp_socket name_bind;
EOF
}

apply() {
    require_enforcing
    require_tools
    detect_binder

    WORKDIR="$(mktemp -d)"
    trap 'rm -rf "$WORKDIR"' EXIT

    write_policy
    run checkmodule -M -m -o "$WORKDIR/$MODULE.mod" "$WORKDIR/$MODULE.te"
    run semodule_package -o "$WORKDIR/$MODULE.pp" -m "$WORKDIR/$MODULE.mod"
    printf '$ sudo semodule -i %s\n' "$WORKDIR/$MODULE.pp"
    run sudo semodule -i "$WORKDIR/$MODULE.pp"

    printf '$ sudo semanage port -a -t %s -p tcp %s\n' "$TYPE" "$PORT"
    if ! sudo semanage port -a -t "$TYPE" -p tcp "$PORT" >/dev/null 2>&1; then
        printf '$ sudo semanage port -m -t %s -p tcp %s\n' "$TYPE" "$PORT"
        run sudo semanage port -m -t "$TYPE" -p tcp "$PORT"
    fi
    say "applied: tcp/$PORT -> $TYPE"
}

revert() {
    printf '$ sudo semanage port -d -t %s -p tcp %s\n' "$TYPE" "$PORT"
    sudo semanage port -d -t "$TYPE" -p tcp "$PORT" >/dev/null 2>&1 || say "(no $TYPE label on tcp/$PORT)"
    printf '$ sudo semodule -r %s\n' "$MODULE"
    sudo semodule -r "$MODULE" >/dev/null 2>&1 || say "(module $MODULE not installed)"
    say "reverted"
}

status() {
    say "mode:   $(getenforce 2>/dev/null || echo unknown)"
    if sudo semodule -l 2>/dev/null | grep -qw "$MODULE"; then
        say "module: installed"
    else
        say "module: not installed"
    fi
    local cur
    cur="$(sudo semanage port -l 2>/dev/null | awk -v p="$PORT" '$2=="tcp" && $4==p {print $1; exit}' || true)"
    say "port:   tcp/$PORT -> ${cur:-(none)}"
    local pid comm
    pid="$(binder_pid)"
    comm="$( [ -n "$pid" ] && ps -o comm= -p "$pid" 2>/dev/null || true )"
    say "binder: ${pid:-none}${comm:+ $comm}"
}

case "$ACTION" in
    apply)  apply ;;
    revert) revert ;;
    status) status ;;
    -h|--help|help) usage ;;
    *) say "unknown action: $ACTION" >&2; usage >&2; exit 2 ;;
esac
