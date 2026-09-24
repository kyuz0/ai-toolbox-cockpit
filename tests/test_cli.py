import io
import json
from contextlib import redirect_stderr, redirect_stdout
from unittest import TestCase
from unittest.mock import patch

from ai_toolbox_cockpit.catalog import CatalogError
from ai_toolbox_cockpit.cli.contracts import CommandEvent, OutputFormat
from ai_toolbox_cockpit.cli.parser import parse_invocation
from ai_toolbox_cockpit.cli.renderers import render_events
from ai_toolbox_cockpit.main import cli_main


class StaticCliTests(TestCase):
    def invoke(self, *args: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = cli_main(list(args))
        return result, stdout.getvalue(), stderr.getvalue()

    def test_info_text_is_human_readable(self) -> None:
        result, stdout, stderr = self.invoke("info")

        self.assertEqual(result, 0)
        self.assertEqual(stderr, "")
        self.assertIn("AI Toolbox Cockpit", stdout)
        self.assertIn("Platforms:", stdout)
        self.assertIn("Backends:", stdout)
        self.assertIn("Toolboxes:", stdout)

    def test_absent_boolean_flag_uses_false_default(self) -> None:
        parsed = parse_invocation(("info",))

        self.assertIs(parsed.invocation.arguments["full"], False)

    def test_info_does_not_start_the_textual_application(self) -> None:
        with patch("ai_toolbox_cockpit.main._load_app_class") as app_class:
            result, _, _ = self.invoke("info")

        self.assertEqual(result, 0)
        app_class.assert_not_called()

    def test_info_does_not_start_subprocesses(self) -> None:
        with patch("subprocess.Popen") as process, patch("subprocess.run") as run:
            result, _, _ = self.invoke("info")

        self.assertEqual(result, 0)
        process.assert_not_called()
        run.assert_not_called()

    def test_info_json_returns_catalog_summary_without_starting_the_tui(self) -> None:
        result, stdout, stderr = self.invoke("info", "--output", "json")

        payload = json.loads(stdout)
        self.assertEqual(result, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["command"], ["info"])
        self.assertTrue(payload["ok"])
        self.assertIn("application", payload["data"])
        self.assertIn("catalog", payload["data"])
        self.assertIn("platforms", payload["data"]["catalog"])
        self.assertIn("backends", payload["data"]["catalog"])
        self.assertTrue(payload["data"]["catalog"]["backends"]["llama_cpp"]["models"])
        self.assertTrue(payload["data"]["catalog"]["backends"]["comfyui"]["bundles"])

    def test_fullinfo_alias_returns_full_catalog_as_json(self) -> None:
        result, stdout, stderr = self.invoke("-fullinfo", "--output", "json")

        payload = json.loads(stdout)
        data = payload["data"]
        llama_cpp = data["catalog"]["backends"]["llama_cpp"]
        self.assertEqual(result, 0)
        self.assertEqual(stderr, "")
        self.assertTrue(data["full"])
        self.assertIn("config", llama_cpp)
        self.assertIn("runtime_profiles", data["catalog"])
        self.assertIn("backend_config", data["catalog"]["toolboxes"][0])

    def test_output_is_accepted_before_the_command(self) -> None:
        result, stdout, stderr = self.invoke("--output", "json", "info")

        self.assertEqual(result, 0)
        self.assertEqual(stderr, "")
        self.assertTrue(json.loads(stdout)["ok"])

    def test_legacy_info_alias_is_accepted_after_global_output(self) -> None:
        result, stdout, stderr = self.invoke("--output", "json", "-info")

        self.assertEqual(result, 0)
        self.assertEqual(stderr, "")
        self.assertTrue(json.loads(stdout)["ok"])

    def test_removed_json_alias_is_reported_as_an_argument_error(self) -> None:
        result, stdout, stderr = self.invoke("info", "--json")

        self.assertEqual(result, 2)
        self.assertEqual(stdout, "")
        self.assertIn("unrecognized arguments: --json", stderr)

    def test_invalid_arguments_use_the_json_error_envelope_when_requested(self) -> None:
        result, stdout, stderr = self.invoke("info", "--output", "json", "--unknown")

        payload = json.loads(stderr)
        self.assertEqual(result, 2)
        self.assertEqual(stdout, "")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "invalid_arguments")

    def test_last_output_option_selects_parse_error_format(self) -> None:
        result, stdout, stderr = self.invoke("info", "--output", "json", "--output", "text", "--unknown")

        self.assertEqual(result, 2)
        self.assertEqual(stdout, "")
        self.assertIn("unrecognized arguments: --unknown", stderr)

    def test_undocumented_long_info_alias_is_rejected(self) -> None:
        result, stdout, stderr = self.invoke("info", "--info")

        self.assertEqual(result, 2)
        self.assertEqual(stdout, "")
        self.assertIn("unrecognized arguments: --info", stderr)

    def test_classic_help_aliases_show_usage(self) -> None:
        for argument in ("-h", "-help"):
            with self.subTest(argument=argument):
                with self.assertRaises(SystemExit) as exited:
                    self.invoke(argument)

                self.assertEqual(exited.exception.code, 0)

    def test_server_commands_are_reserved_without_starting_a_container(self) -> None:
        result, stdout, stderr = self.invoke(
            "server",
            "start",
            "llama-cpp",
            "--model",
            "model.gguf",
            "--context-size",
            "4096",
            "--output",
            "json",
        )

        self.assertEqual(result, 3)
        self.assertEqual(stdout, "")
        payload = json.loads(stderr)
        self.assertEqual(payload["error"]["code"], "not_implemented")
        self.assertEqual(payload["command"], ["server", "start", "llama-cpp"])

    def test_server_stop_accepts_each_target_selector(self) -> None:
        for selector in (("--run", "run-1"), ("--session", "session-1"), ("--all",)):
            with self.subTest(selector=selector):
                result, stdout, stderr = self.invoke("server", "stop", *selector, "--output", "json")

                self.assertEqual(result, 3)
                self.assertEqual(stdout, "")
                self.assertEqual(json.loads(stderr)["error"]["code"], "not_implemented")

    def test_catalog_error_uses_the_json_error_envelope(self) -> None:
        with patch(
            "ai_toolbox_cockpit.cli.commands.info.load_toolbox_catalog",
            side_effect=CatalogError("invalid catalogue"),
        ):
            result, stdout, stderr = self.invoke("info", "--output", "json")

        payload = json.loads(stderr)
        self.assertEqual(result, 3)
        self.assertEqual(stdout, "")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "catalog_error")
        self.assertEqual(payload["error"]["message"], "invalid catalogue")

    def test_unexpected_command_error_uses_the_json_error_envelope(self) -> None:
        with patch("ai_toolbox_cockpit.cli.dispatch", side_effect=RuntimeError):
            result, stdout, stderr = self.invoke("info", "--output", "json")

        payload = json.loads(stderr)
        self.assertEqual(result, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(payload["error"]["code"], "internal_error")

    def test_future_json_events_are_json_lines(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        render_events(
            [
                CommandEvent(
                    kind="started",
                    sequence=1,
                    data={"message": "Starting"},
                    run_id="run-1",
                    session_id="session-1",
                ),
                CommandEvent(
                    kind="output",
                    sequence=2,
                    data={"message": "Ready"},
                    run_id="run-1",
                    session_id="session-1",
                ),
            ],
            ("server", "start", "llama-cpp"),
            OutputFormat.JSON,
            stdout,
            stderr,
        )

        lines = stdout.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["event"], "started")
        self.assertEqual(json.loads(lines[1])["event"], "output")
        self.assertEqual(stderr.getvalue(), "")