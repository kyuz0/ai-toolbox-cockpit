"""Relay unit tests: no containers, GPU servers, or network listeners started."""

import io
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from ai_toolbox_cockpit.runtime.isolated_api import (
    IsolatedAPIRelay, LOOPBACK_HELPER, build_loopback_exec_cmd,
)


class IsolatedAPITests(TestCase):
    def test_exec_uses_fixed_loopback_helper_without_shell_or_tty(self):
        for engine in ("podman", "docker"):
            command = build_loopback_exec_cmd(engine, "server", 9000)
            self.assertEqual(command, [engine, "exec", "-i", "server", "python3",
                                       "-I", "-u", "-c", LOOPBACK_HELPER, "9000"])
        for engine, port in (("sh", 9000), ("podman", 0), ("docker", 65536)):
            with self.assertRaises(ValueError):
                build_loopback_exec_cmd(engine, "server", port)

    def test_helper_streams_binary_data_and_half_closes_upload(self):
        connection = Mock()
        response = b"HTTP/1.1 200 OK\r\n\r\ndata: \x00\xff\n\n"
        connection.recv.side_effect = [response[:20], response[20:], b""]
        class ShortWriter(io.BytesIO):
            def write(self, data):
                return super().write(data[:3])

        output = ShortWriter()
        fake_sys = SimpleNamespace(argv=["-c", "9000"], stdout=SimpleNamespace(buffer=output))
        scope = {}
        # Execute the actual helper using fake I/O. Its upload thread is invoked
        # explicitly below, so this test performs no networking or background work.
        with patch.dict("sys.modules", {"sys": fake_sys}), \
             patch("socket.create_connection", return_value=connection) as connect, \
             patch("threading.Thread") as thread, \
             patch("os.read", side_effect=[b"POST /\r\n\x00\xff", b""]):
            exec(compile(LOOPBACK_HELPER, "loopback-helper", "exec"), scope)
            thread.call_args.kwargs["target"]()
        connect.assert_called_once_with(("127.0.0.1", 9000), timeout=10)
        self.assertEqual(output.getvalue(), response)
        connection.sendall.assert_called_once_with(b"POST /\r\n\x00\xff")
        connection.shutdown.assert_called_once_with(socket.SHUT_WR)
        connection.close.assert_called_once()

    def test_accepted_socket_is_only_exec_input_and_output(self):
        relay = IsolatedAPIRelay("podman", "server", "localhost", 9000)
        client = Mock()
        process = Mock()
        with patch("ai_toolbox_cockpit.runtime.isolated_api.subprocess.Popen", return_value=process) as launch:
            relay._handle(client)
        launch.assert_called_once_with(relay.command, stdin=client, stdout=client, start_new_session=True)
        process.wait.assert_called_once_with()
        self.assertEqual(relay._clients, {})

    def test_shutdown_and_connection_limit_prevent_new_exec_clients(self):
        relay = IsolatedAPIRelay("docker", "server", "::1", 9000)
        with patch("ai_toolbox_cockpit.runtime.isolated_api.subprocess.Popen") as launch:
            for _ in range(64):
                self.assertTrue(relay._slots.acquire(blocking=False))
            relay._handle(Mock())
            relay._slots.release()
            relay._closing = True
            relay._handle(Mock())
        launch.assert_not_called()

    def test_helper_reports_a_closed_api_without_a_traceback(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        result = subprocess.run([sys.executable, "-I", "-u", "-c", LOOPBACK_HELPER, str(port)],
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn("not listening yet", result.stderr)

    def test_exec_launch_failure_releases_slot(self):
        relay = IsolatedAPIRelay("podman", "server", "localhost", 9000)
        with patch("ai_toolbox_cockpit.runtime.isolated_api.subprocess.Popen", side_effect=OSError("missing")), \
             patch("builtins.print") as output:
            relay._handle(Mock())
        self.assertIn("missing", str(output.call_args))
        self.assertEqual(relay._clients, {})
        for _ in range(64):
            self.assertTrue(relay._slots.acquire(blocking=False))

    def test_cleanup_closes_listener_and_clients_and_reaps_stuck_exec(self):
        relay = IsolatedAPIRelay("podman", "server", "localhost", 9000)
        relay._server = Mock()
        relay._thread = Mock()
        process, client = Mock(), Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("exec", 5), 0]
        relay._clients[process] = client
        relay.__exit__(None, None, None)
        relay._server.shutdown.assert_called_once()
        relay._server.server_close.assert_called_once()
        relay._thread.join.assert_called_once()
        client.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)
