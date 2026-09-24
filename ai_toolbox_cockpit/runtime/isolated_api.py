"""Inbound TCP access to a networkless container through engine exec streams.

The host only accepts connections. Each accepted socket becomes stdin/stdout of
an engine exec client; the helper connects to a fixed container loopback port.
No host socket, network namespace, or writable directory is mounted inside.
"""

import ipaddress
import socket
import socketserver
import subprocess
import threading
import time


# Python is supplied by the serving image. No shell, PTY, downloaded helper, or
# destination supplied by an API client is involved. Separate directions retain
# streaming, backpressure, and client write-half-close semantics.
LOOPBACK_HELPER = """
import os, socket, sys, threading
s = socket.create_connection(('127.0.0.1', int(sys.argv[1])), timeout=10)
s.settimeout(None)
def upload():
    try:
        while True:
            data = os.read(0, 65536)
            if not data:
                break
            s.sendall(data)
    except OSError:
        pass
    finally:
        try:
            s.shutdown(socket.SHUT_WR)
        except OSError:
            pass
threading.Thread(target=upload, daemon=True).start()
try:
    while True:
        data = s.recv(65536)
        if not data:
            break
        pending = memoryview(data)
        while pending:
            pending = pending[sys.stdout.buffer.write(pending):]
        sys.stdout.buffer.flush()
except OSError:
    pass
finally:
    s.close()
"""


def build_loopback_exec_cmd(engine: str, container_name: str, port: int) -> list[str]:
    if engine not in {"podman", "docker"} or not 1 <= port <= 65535:
        raise ValueError("The isolated API requires Podman/Docker and a valid port.")
    return [engine, "exec", "-i", container_name,
            "python3", "-I", "-u", "-c", LOOPBACK_HELPER, str(port)]


class _RelayServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class IsolatedAPIRelay:
    """Own the listener and exec clients for the lifetime of a foreground server."""

    def __init__(self, engine: str, container_name: str, host: str, port: int,
                 *, container_port: int | None = None):
        self.command = build_loopback_exec_cmd(engine, container_name, container_port or port)
        self.host = str(ipaddress.ip_address("127.0.0.1" if host == "localhost" else host))
        self.port = port
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(64)
        self._clients: dict[subprocess.Popen, socket.socket] = {}
        self._closing = False

    def __enter__(self):
        relay = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                relay._handle(self.request)

        class Server(_RelayServer):
            address_family = socket.AF_INET6 if ":" in relay.host else socket.AF_INET

            def server_bind(self):
                if self.address_family == socket.AF_INET6:
                    self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                super().server_bind()

        self._server = Server((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self._server.server_close()
            raise
        return self

    def _handle(self, client: socket.socket):
        if not self._slots.acquire(blocking=False):
            return
        process = None
        try:
            with self._lock:
                if self._closing:
                    return
                process = subprocess.Popen(
                    self.command, stdin=client, stdout=client, start_new_session=True,
                )
                self._clients[process] = client
            process.wait()
        except OSError as error:
            print(f"Isolated API relay failed: {error}")
        finally:
            with self._lock:
                if process is not None:
                    self._clients.pop(process, None)
            self._slots.release()

    def __exit__(self, *_):
        with self._lock:
            self._closing = True
            clients = list(self._clients.items())
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
        for process, client in clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 5
        for process, _ in clients:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
