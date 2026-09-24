"""Foreground server lifecycle shared by backend server panels."""

import shlex
import signal
import subprocess
from uuid import uuid4
from contextlib import nullcontext

from .isolated_api import IsolatedAPIRelay
from .terminal import command_failed, pause_after_failure


def redact_command(
    command: list[str],
    secret_options: tuple[str, ...] = ("--api-key",),
    secret_environment: tuple[str, ...] = ("HF_TOKEN",),
) -> list[str]:
    """Redact every separate or equals-form value for sensitive CLI options."""
    redacted = list(command)
    index = 0
    while index < len(redacted):
        argument = redacted[index]
        if argument in secret_options and index + 1 < len(redacted):
            redacted[index + 1] = "<redacted>"
            index += 2
            continue
        for option in secret_options:
            if argument.startswith(f"{option}="):
                redacted[index] = f"{option}=<redacted>"
                break
        for variable in secret_environment:
            if argument.startswith(f"{variable}="):
                redacted[index] = f"{variable}=<redacted>"
                break
        index += 1
    return redacted


def _existing_servers(engine: str, base_name: str) -> list[str]:
    result = subprocess.run(
        [engine, "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise OSError(f"Could not inspect existing containers: {result.stderr.strip()}")
    return [name for name in result.stdout.splitlines()
            if name == base_name or name.startswith(f"{base_name}-")]


def _with_container_name(command: list[str], old_name: str, new_name: str) -> list[str]:
    updated = list(command)
    try:
        index = updated.index("--name") + 1
    except ValueError as error:
        raise ValueError("Server command has no container name") from error
    if updated[index] != old_name:
        raise ValueError("Server command container name does not match the backend")
    updated[index] = new_name
    if old_name == "ds4-cockpit-server":
        updated[index + 1:index + 1] = ["--env", f"DS4_LOCK_FILE=/tmp/{new_name}.lock"]
    return updated


def _ask_alongside_port(command: list[str], isolated_api: tuple[str, int] | None) -> int | None:
    if "-p" in command:
        mapping = command[command.index("-p") + 1].rsplit(":", 2)
        if len(mapping) < 2:
            raise ValueError("Cannot identify the server's published port")
        old_port = int(mapping[-2])
    elif isolated_api is not None:
        old_port = isolated_api[1]
    else:
        raise ValueError("This server uses host networking, so the cockpit cannot choose separate ports automatically; launch it separately with distinct service ports")
    while True:
        try:
            answer = input(f"Host port for the additional server (different from {old_port}; Enter cancels): ").strip()
        except EOFError:
            return None
        if not answer:
            return None
        if answer.isdecimal() and 1 <= int(answer) <= 65535 and int(answer) != old_port:
            return int(answer)
        print("Enter a valid port from 1 to 65535, different from the selected port.")


def _with_published_port(command: list[str], new_port: int) -> list[str]:
    updated = list(command)
    if "-p" in updated:
        index = updated.index("-p") + 1
        mapping = updated[index].rsplit(":", 2)
        mapping[-2] = str(new_port)
        updated[index] = ":".join(mapping)
    return updated


def run_foreground_server(
    command: list[str],
    engine: str,
    container_name: str,
    *,
    display_command: list[str] | None = None,
    isolated_api: tuple[str, int] | None = None,
) -> int:
    """Run a server, preserving existing containers unless replacement is chosen."""
    try:
        existing = _existing_servers(engine, container_name)
    except OSError as error:
        pause_after_failure(str(error))
        return 127
    action = "new"
    alongside_port = None
    if existing:
        print(f"Existing {container_name} container(s): {', '.join(existing)}")
        while True:
            try:
                choice = input("Replace these servers [r], run alongside [a], or cancel [Enter]? ").strip().lower()
            except EOFError:
                return 0
            if choice in {"", "r", "a"}:
                break
            print("Enter r, a, or press Enter to cancel.")
        if not choice:
            return 0
        action = "replace" if choice == "r" else "alongside"
        if action == "alongside":
            try:
                alongside_port = _ask_alongside_port(command, isolated_api)
            except ValueError as error:
                print(error)
                return 0
            if alongside_port is None:
                return 0
    new_name = f"{container_name}-{uuid4().hex[:8]}"
    try:
        prepared = list(command)
        preview = list(display_command if display_command is not None else command)
        if alongside_port is not None:
            prepared = _with_published_port(prepared, alongside_port)
            preview = _with_published_port(preview, alongside_port)
        prepared = _with_container_name(prepared, container_name, new_name)
        preview = _with_container_name(preview, container_name, new_name)
    except (ValueError, IndexError) as error:
        pause_after_failure(str(error))
        return 127
    if action == "replace":
        for old_name in existing:
            try:
                result = subprocess.run([engine, "rm", "-f", old_name], capture_output=True, text=True)
            except OSError as error:
                pause_after_failure(f"Could not stop {old_name}: {error}")
                return 127
            if result.returncode:
                pause_after_failure(f"Could not stop {old_name}: {result.stderr.strip()}")
                return 127
    print(f"\nStarting server:\n{shlex.join(preview)}\n")
    print(
        "Press Ctrl+C to stop the server and return to the cockpit. "
        "If startup fails, press Enter after reviewing the error.\n"
    )
    old_handler = signal.signal(signal.SIGINT, signal.default_int_handler)
    process: subprocess.Popen | None = None
    try:
        if isolated_api is None:
            relay = nullcontext()
        elif alongside_port is None:
            relay = IsolatedAPIRelay(engine, new_name, *isolated_api)
        else:
            relay = IsolatedAPIRelay(engine, new_name, isolated_api[0], alongside_port,
                                     container_port=isolated_api[1])
        with relay:
            if isolated_api is not None:
                print(f"Isolated API relay: {isolated_api[0]}:{alongside_port or isolated_api[1]} "
                      "-> container loopback (no container network).\n")
            process = subprocess.Popen(prepared)
            return_code = process.wait()
        if command_failed(return_code):
            pause_after_failure(f"Server exited with status {return_code}.")
        return return_code
    except OSError as error:
        pause_after_failure(f"Could not start the server command: {error}")
        return 127
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if process is not None:
            subprocess.run([engine, "rm", "-f", new_name], capture_output=True)
            process.kill()
            process.wait()
        return 130
    finally:
        try:
            if isolated_api is not None and process is not None:
                subprocess.run([engine, "rm", "-f", new_name], capture_output=True)
        except OSError as error:
            print(f"Could not remove the server container: {error}")
        finally:
            signal.signal(signal.SIGINT, old_handler)
