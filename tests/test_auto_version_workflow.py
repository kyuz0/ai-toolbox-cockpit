import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "auto-version.yml"


class AutoVersionWorkflowTests(unittest.TestCase):
    def test_main_push_bumps_pyproject_commits_tags_and_pushes(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        for required in (
            "branches: [main]",
            "contents: write",
            'date -u +"%-Y.%-m.%-d.%-H%M"',
            'sed -i "s/^version = .*/version =',
            "git add pyproject.toml",
            'git tag "v${{ steps.version.outputs.version }}"',
            "git push origin main --tags",
        ):
            self.assertIn(required, workflow)

    def test_workflow_version_replacement_keeps_pyproject_valid(self) -> None:
        source = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        updated, replacements = re.subn(
            r'^version = .*$',
            'version = "2099.12.31.2359"',
            source,
            count=1,
            flags=re.MULTILINE,
        )
        self.assertEqual(replacements, 1)
        self.assertEqual(tomllib.loads(updated)["project"]["version"], "2099.12.31.2359")


if __name__ == "__main__":
    unittest.main()
