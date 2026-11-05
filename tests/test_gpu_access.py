import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ai_toolbox_cockpit.runtime.gpu_access import gpu_access_issue

MODULE = "ai_toolbox_cockpit.runtime.gpu_access"
COMMAND = ["podman", "run", "--device", "/dev/kfd", "--device=/dev/dri", "--group-add", "keep-groups", "image"]


class GpuAccessTests(unittest.TestCase):
    def setUp(self):
        patches = {
            "info": patch(MODULE + ".subprocess.run", return_value=Mock(returncode=0, stdout=json.dumps({"host": {"security": {"rootless": True}}}))),
            "glob": patch(MODULE + ".Path.glob", return_value=[Path("/dev/dri/renderD128")]),
            "stat": patch(MODULE + ".Path.stat", return_value=SimpleNamespace(st_gid=109)),
            "access": patch(MODULE + ".os.access", return_value=False),
            "groups": patch(MODULE + ".os.getgroups", return_value=[1000]),
            "egid": patch(MODULE + ".os.getegid", return_value=1000),
            "account": patch(MODULE + ".pwd.getpwuid", return_value=SimpleNamespace(pw_name="hpuser", pw_gid=1000)),
            "configured": patch(MODULE + ".os.getgrouplist", return_value=[1000]),
            "group": patch(MODULE + ".grp.getgrgid", return_value=SimpleNamespace(gr_name="render")),
        }
        self.mocks = {}
        for name, item in patches.items():
            self.mocks[name] = item.start()
            self.addCleanup(item.stop)
        clean_env = patch.dict(MODULE + ".os.environ", {}, clear=True)
        clean_env.start()
        self.addCleanup(clean_env.stop)

    def test_missing_group_names_devices_and_gives_exact_fix(self):
        issue = gpu_access_issue(COMMAND, "podman")
        self.assertIn("/dev/kfd (owning group: render)", issue)
        self.assertIn("/dev/dri/renderD128", issue)
        self.assertIn("sudo usermod -aG render hpuser", issue)
        self.assertIn("Fully log out and back in", issue)
        self.assertEqual(self.mocks["info"].call_count, 1)

    def test_nonstandard_device_group_does_not_suggest_privileged_membership(self):
        self.mocks["stat"].return_value = SimpleNamespace(st_gid=0)
        self.mocks["group"].return_value = SimpleNamespace(gr_name="root")
        issue = gpu_access_issue(COMMAND, "podman")
        self.assertIn("nonstandard group", issue)
        self.assertNotIn("usermod", issue)

    def test_stale_session_needs_login_not_another_usermod(self):
        self.mocks["configured"].return_value = [1000, 109]
        issue = gpu_access_issue(COMMAND, "podman")
        self.assertIn("Already configured but absent from this session: render", issue)
        self.assertNotIn("usermod", issue)
        self.assertIn("fresh login outside the old tmux server", issue)

    def test_acl_access_does_not_require_group_membership(self):
        self.mocks["access"].return_value = True
        self.assertIsNone(gpu_access_issue(COMMAND, "podman"))
        self.mocks["configured"].assert_not_called()

    def test_denied_despite_group_membership_reports_acl_check(self):
        self.mocks["groups"].return_value = [1000, 109]
        issue = gpu_access_issue(COMMAND, "podman")
        self.assertIn("getfacl", issue)
        self.assertNotIn("usermod", issue)

    def test_missing_kfd_is_driver_advice_not_group_advice(self):
        self.mocks["stat"].side_effect = FileNotFoundError()
        issue = gpu_access_issue(COMMAND, "podman")
        self.assertIn("/dev/kfd is missing", issue)
        self.assertNotIn("usermod", issue)

    def test_missing_render_devices_is_reported(self):
        self.mocks["glob"].return_value = []
        self.assertIn("no render devices", gpu_access_issue(COMMAND, "podman"))

    def test_non_rocm_and_docker_commands_are_not_host_checked(self):
        self.assertIsNone(gpu_access_issue(["podman", "run", "image"], "podman"))
        self.assertIsNone(gpu_access_issue(COMMAND, "docker"))
        self.mocks["info"].assert_not_called()

    def test_remote_and_rootful_podman_are_not_host_checked(self):
        for host in ({"security": {"rootless": False}}, {"security": {"rootless": True}, "serviceIsRemote": True}):
            with self.subTest(host=host):
                self.mocks["info"].return_value.stdout = json.dumps({"host": host})
                self.assertIsNone(gpu_access_issue(COMMAND, "podman"))
        self.mocks["stat"].assert_not_called()

    def test_remote_environment_is_not_host_checked(self):
        for variable in ("CONTAINER_HOST", "CONTAINER_CONNECTION"):
            with patch.dict(MODULE + ".os.environ", {variable: "remote"}):
                self.assertIsNone(gpu_access_issue(COMMAND, "podman"))
        self.mocks["info"].assert_not_called()

    def test_failed_info_does_not_claim_permission_failure(self):
        for result in (Mock(returncode=1, stdout=""), Mock(returncode=0, stdout="invalid")):
            self.mocks["info"].return_value = result
            self.assertIsNone(gpu_access_issue(COMMAND, "podman"))
        self.mocks["stat"].assert_not_called()

    def test_explicit_render_node_is_checked_without_unselected_devices(self):
        command = ["podman", "run", "--device=/dev/kfd", "--device", "/dev/dri/renderD129:/dev/dri/renderD129", "image"]
        issue = gpu_access_issue(command, "podman")
        self.assertIn("/dev/dri/renderD129", issue)
        self.assertNotIn("renderD128", issue)
        self.mocks["glob"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
