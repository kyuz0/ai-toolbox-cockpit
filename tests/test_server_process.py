import unittest
from unittest.mock import Mock, patch

from ai_toolbox_cockpit.runtime.server_process import redact_command, run_foreground_server


class ServerProcessTests(unittest.TestCase):
    @staticmethod
    def ps(names=""):
        return Mock(returncode=0, stdout=names, stderr="")

    @staticmethod
    def port(mapping="8000/tcp -> 127.0.0.1:8000\n"):
        return Mock(returncode=0, stdout=mapping, stderr="")

    def test_new_server_gets_unique_name_without_deleting_any_container(self):
        process = Mock()
        process.wait.return_value = 0
        command = ["podman", "run", "--name", "ds4-cockpit-server", "-p", "127.0.0.1:8000:8000", "image"]
        with patch("ai_toolbox_cockpit.runtime.server_process.subprocess.run", return_value=self.ps()) as run, \
             patch("ai_toolbox_cockpit.runtime.server_process.subprocess.Popen", return_value=process) as launch, \
             patch("ai_toolbox_cockpit.runtime.server_process.signal.signal"):
            self.assertEqual(run_foreground_server(command, "podman", "ds4-cockpit-server"), 0)
        self.assertEqual(len(run.call_args_list), 1)
        self.assertEqual(run.call_args.args[0], ["podman", "ps", "--format", "{{.Names}}"])
        launched = launch.call_args.args[0]
        name = launched[launched.index("--name") + 1]
        self.assertTrue(name.startswith("ds4-cockpit-server-"))
        self.assertIn(f"DS4_LOCK_FILE=/tmp/{name}.lock", launched)
        self.assertEqual(command[command.index("--name") + 1], "ds4-cockpit-server")

    def test_existing_server_cancel_leaves_it_untouched(self):
        with patch("ai_toolbox_cockpit.runtime.server_process.subprocess.run", return_value=self.ps("ds4-cockpit-server\n")) as run, \
             patch("ai_toolbox_cockpit.runtime.server_process.subprocess.Popen") as launch, \
             patch("builtins.input", return_value=""):
            self.assertEqual(run_foreground_server(["podman", "run", "--name", "ds4-cockpit-server"],
                                                   "podman", "ds4-cockpit-server"), 0)
        launch.assert_not_called()
        self.assertEqual(run.call_count, 1)

    def test_alongside_changes_only_new_name_host_port_and_ds4_lock(self):
        process = Mock()
        process.wait.return_value = 0
        command = ["podman", "run", "--name", "ds4-cockpit-server", "-p", "127.0.0.1:8000:8000", "image"]
        with patch("ai_toolbox_cockpit.runtime.server_process.subprocess.run", side_effect=[self.ps("ds4-cockpit-server\n"), self.port()]) as run, \
             patch("ai_toolbox_cockpit.runtime.server_process._host_port_available", return_value=True), \
             patch("ai_toolbox_cockpit.runtime.server_process.subprocess.Popen", return_value=process) as launch, \
             patch("ai_toolbox_cockpit.runtime.server_process.signal.signal"), \
             patch("builtins.input", side_effect=["a", "8001"]):
            self.assertEqual(run_foreground_server(command, "podman", "ds4-cockpit-server"), 0)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[1].args[0], ["podman", "port", "ds4-cockpit-server"])
        launched = launch.call_args.args[0]
        self.assertEqual(launched[launched.index("-p") + 1], "127.0.0.1:8001:8000")
        self.assertTrue(launched[launched.index("--name") + 1].startswith("ds4-cockpit-server-"))
        self.assertEqual(command[command.index("-p") + 1], "127.0.0.1:8000:8000")

    def test_alongside_keeps_already_selected_free_port(self):
        process = Mock()
        process.wait.return_value = 0
        command = ["podman", "run", "--name", "ds4-cockpit-server", "-p", "127.0.0.1:8731:8731", "image"]
        with patch("ai_toolbox_cockpit.runtime.server_process.subprocess.run", side_effect=[self.ps("ds4-cockpit-server-1d859986\n"), self.port()]) as run, \
             patch("ai_toolbox_cockpit.runtime.server_process._host_port_available", return_value=True) as available, \
             patch("ai_toolbox_cockpit.runtime.server_process.subprocess.Popen", return_value=process) as launch, \
             patch("ai_toolbox_cockpit.runtime.server_process.signal.signal"), \
             patch("builtins.input", return_value="a") as answer:
            self.assertEqual(run_foreground_server(command, "podman", "ds4-cockpit-server"), 0)
        self.assertEqual(run.call_count, 2)
        available.assert_called_once_with("127.0.0.1", 8731)
        answer.assert_called_once()
        launched = launch.call_args.args[0]
        self.assertEqual(launched[launched.index("-p") + 1], "127.0.0.1:8731:8731")

    def test_replace_removes_only_listed_existing_container_after_choice(self):
        process = Mock()
        process.wait.return_value = 0
        with patch("ai_toolbox_cockpit.runtime.server_process.subprocess.run", side_effect=[self.ps("llama-cockpit-server\n"), self.ps()]) as run, \
             patch("ai_toolbox_cockpit.runtime.server_process.subprocess.Popen", return_value=process) as launch, \
             patch("ai_toolbox_cockpit.runtime.server_process.signal.signal"), \
             patch("builtins.input", return_value="r"):
            self.assertEqual(run_foreground_server(["podman", "run", "--name", "llama-cockpit-server", "image"],
                                                   "podman", "llama-cockpit-server"), 0)
        self.assertEqual(run.call_args_list[1].args[0], ["podman", "rm", "-f", "llama-cockpit-server"])
        self.assertTrue(launch.call_args.args[0][3].startswith("llama-cockpit-server-"))

    def test_host_network_alongside_is_rejected_without_stopping_first(self):
        with patch("ai_toolbox_cockpit.runtime.server_process.subprocess.run", return_value=self.ps("ds4-cockpit-server\n")) as run, \
             patch("ai_toolbox_cockpit.runtime.server_process.subprocess.Popen") as launch, \
             patch("builtins.input", return_value="a"):
            self.assertEqual(run_foreground_server(["podman", "run", "--name", "ds4-cockpit-server", "--network=host", "image"],
                                                   "podman", "ds4-cockpit-server"), 0)
        launch.assert_not_called()
        self.assertEqual(run.call_count, 1)

    def test_isolated_api_alongside_uses_new_host_port_and_original_container_port(self):
        process = Mock()
        process.wait.return_value = 0
        with patch("ai_toolbox_cockpit.runtime.server_process.subprocess.run", return_value=self.ps("halogen-cockpit-server\n")) as run, \
             patch("ai_toolbox_cockpit.runtime.server_process._host_port_available", side_effect=[False, True]), \
             patch("ai_toolbox_cockpit.runtime.server_process.IsolatedAPIRelay") as relay, \
             patch("ai_toolbox_cockpit.runtime.server_process.subprocess.Popen", return_value=process) as launch, \
             patch("ai_toolbox_cockpit.runtime.server_process.signal.signal"), \
             patch("builtins.input", side_effect=["a", "9001"]):
            self.assertEqual(run_foreground_server(["podman", "run", "--name", "halogen-cockpit-server", "image"],
                                                   "podman", "halogen-cockpit-server", isolated_api=("127.0.0.1", 9000)), 0)
        new_name = launch.call_args.args[0][3]
        relay.assert_called_once_with("podman", new_name, "127.0.0.1", 9001, container_port=9000)
        self.assertEqual(run.call_args_list[-1].args[0], ["podman", "rm", "-f", new_name])

    def test_isolated_api_lifecycle_cleans_up_only_launched_container(self):
        for outcome, expected in ((0, 0), (2, 2), (KeyboardInterrupt(), 130), (OSError("failed"), 127)):
            with self.subTest(outcome=outcome):
                process = Mock()
                if isinstance(outcome, BaseException):
                    process.wait.side_effect = [outcome, 0]
                else:
                    process.wait.return_value = outcome
                with patch("ai_toolbox_cockpit.runtime.server_process.subprocess.run", return_value=self.ps()) as run, \
                     patch("ai_toolbox_cockpit.runtime.server_process.IsolatedAPIRelay") as relay, \
                     patch("ai_toolbox_cockpit.runtime.server_process.subprocess.Popen", return_value=process) as launch, \
                     patch("ai_toolbox_cockpit.runtime.server_process.signal.signal"), \
                     patch("ai_toolbox_cockpit.runtime.server_process.pause_after_failure"):
                    code = run_foreground_server(["podman", "run", "--name", "server", "--network=none", "image"],
                                                 "podman", "server", isolated_api=("127.0.0.1", 9000))
                self.assertEqual(code, expected)
                name = launch.call_args.args[0][3]
                relay.assert_called_once_with("podman", name, "127.0.0.1", 9000)
                self.assertEqual(run.call_args_list[-1].args[0], ["podman", "rm", "-f", name])

    def test_occupied_relay_port_fails_without_launching_container(self):
        with patch("ai_toolbox_cockpit.runtime.server_process.subprocess.run", return_value=self.ps()) as run, \
             patch("ai_toolbox_cockpit.runtime.server_process.IsolatedAPIRelay") as relay, \
             patch("ai_toolbox_cockpit.runtime.server_process.subprocess.Popen") as launch, \
             patch("ai_toolbox_cockpit.runtime.server_process.signal.signal"), \
             patch("ai_toolbox_cockpit.runtime.server_process.pause_after_failure") as failure:
            relay.return_value.__enter__.side_effect = OSError("Address already in use")
            result = run_foreground_server(["docker", "run", "--name", "server", "--network=none", "image"],
                                           "docker", "server", isolated_api=("::1", 9000))
        self.assertEqual(result, 127)
        launch.assert_not_called()
        self.assertEqual(run.call_count, 1)
        self.assertIn("Address already in use", failure.call_args.args[0])

    def test_redaction_covers_repeated_and_equals_form_secrets(self) -> None:
        command = ["server", "--api-key", "first", "--api-key=second", "HF_TOKEN=secret"]
        self.assertEqual(redact_command(command),
                         ["server", "--api-key", "<redacted>", "--api-key=<redacted>", "HF_TOKEN=<redacted>"])

    def test_display_command_can_redact_secrets_without_changing_execution(self) -> None:
        process = Mock()
        process.wait.return_value = 0
        command = ["podman", "run", "--name", "server-name", "image", "--api-key", "secret"]
        display = redact_command(command)
        with patch("ai_toolbox_cockpit.runtime.server_process.subprocess.run", return_value=self.ps()) as run, \
             patch("ai_toolbox_cockpit.runtime.server_process.subprocess.Popen", return_value=process) as popen, \
             patch("ai_toolbox_cockpit.runtime.server_process.signal.signal"), \
             patch("builtins.print") as output:
            self.assertEqual(run_foreground_server(command, "podman", "server-name", display_command=display), 0)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(popen.call_args.args[0][-1], "secret")
        rendered = " ".join(str(call) for call in output.call_args_list)
        self.assertIn("<redacted>", rendered)
        self.assertNotIn("secret", rendered)

    def test_failed_server_waits_for_acknowledgement(self) -> None:
        process = Mock()
        process.wait.return_value = 2
        with patch("ai_toolbox_cockpit.runtime.server_process.subprocess.run", return_value=self.ps()), \
             patch("ai_toolbox_cockpit.runtime.server_process.subprocess.Popen", return_value=process), \
             patch("ai_toolbox_cockpit.runtime.server_process.signal.signal"), \
             patch("builtins.input", return_value="") as acknowledge:
            self.assertEqual(run_foreground_server(["podman", "run", "--name", "server"], "podman", "server"), 2)
        acknowledge.assert_called_once_with("\nPress Enter to return to AI Toolbox Cockpit...")

    def test_ctrl_c_exit_returns_without_failure_pause(self) -> None:
        process = Mock()
        process.wait.return_value = 130
        with patch("ai_toolbox_cockpit.runtime.server_process.subprocess.run", return_value=self.ps()), \
             patch("ai_toolbox_cockpit.runtime.server_process.subprocess.Popen", return_value=process), \
             patch("ai_toolbox_cockpit.runtime.server_process.signal.signal"), \
             patch("builtins.input") as acknowledge:
            self.assertEqual(run_foreground_server(["podman", "run", "--name", "server"], "podman", "server"), 130)
        acknowledge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
