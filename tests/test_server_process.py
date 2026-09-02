import unittest
from unittest.mock import Mock, patch

from ai_toolbox_cockpit.runtime.server_process import redact_command, run_foreground_server


class ServerProcessTests(unittest.TestCase):
    def test_redaction_covers_repeated_and_equals_form_secrets(self) -> None:
        command = [
            "server", "--api-key", "first", "--flag", "value",
            "--api-key=second", "--api-key", "third", "HF_TOKEN=secret",
        ]
        self.assertEqual(
            redact_command(command),
            [
                "server", "--api-key", "<redacted>", "--flag", "value",
                "--api-key=<redacted>", "--api-key", "<redacted>",
                "HF_TOKEN=<redacted>",
            ],
        )
        self.assertEqual(command[2], "first")

    def test_display_command_can_redact_secrets_without_changing_execution(self) -> None:
        process = Mock()
        process.wait.return_value = 0
        command = ["podman", "run", "image", "--api-key", "secret"]
        display = ["podman", "run", "image", "--api-key", "<redacted>"]
        with (
            patch("ai_toolbox_cockpit.runtime.server_process.subprocess.run") as run,
            patch("ai_toolbox_cockpit.runtime.server_process.subprocess.Popen", return_value=process) as popen,
            patch("ai_toolbox_cockpit.runtime.server_process.signal.signal"),
            patch("builtins.print") as output,
        ):
            self.assertEqual(
                run_foreground_server(command, "podman", "server-name", display_command=display),
                0,
            )
        run.assert_called_once_with(["podman", "rm", "-f", "server-name"], capture_output=True)
        popen.assert_called_once_with(command)
        rendered = " ".join(str(call) for call in output.call_args_list)
        self.assertIn("<redacted>", rendered)
        self.assertNotIn("secret", rendered)

    def test_failed_server_waits_for_acknowledgement(self) -> None:
        process = Mock()
        process.wait.return_value = 2
        with (
            patch("ai_toolbox_cockpit.runtime.server_process.subprocess.run"),
            patch(
                "ai_toolbox_cockpit.runtime.server_process.subprocess.Popen",
                return_value=process,
            ),
            patch("ai_toolbox_cockpit.runtime.server_process.signal.signal"),
            patch(
                "builtins.input",
                return_value="",
            ) as acknowledge,
        ):
            self.assertEqual(
                run_foreground_server(["podman", "run"], "podman", "server"),
                2,
            )

        acknowledge.assert_called_once_with(
            "\nPress Enter to return to AI Toolbox Cockpit..."
        )

    def test_ctrl_c_exit_returns_without_failure_pause(self) -> None:
        process = Mock()
        process.wait.return_value = 130
        with (
            patch("ai_toolbox_cockpit.runtime.server_process.subprocess.run"),
            patch(
                "ai_toolbox_cockpit.runtime.server_process.subprocess.Popen",
                return_value=process,
            ),
            patch("ai_toolbox_cockpit.runtime.server_process.signal.signal"),
            patch("builtins.input") as acknowledge,
        ):
            self.assertEqual(
                run_foreground_server(["podman", "run"], "podman", "server"),
                130,
            )

        acknowledge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
