import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP = REPO_ROOT / "setup_training.sh"


class SetupTrainingSafetyTest(unittest.TestCase):
    def _run(self, **overrides):
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env.update(
                {
                    "REPO_DIR": str(REPO_ROOT),
                    "BASE": tmp,
                    "LACWM_DATA": str(Path(tmp) / "data"),
                    "FETCH": "skip",
                    "DROID_ENABLE": "0",
                    "EGODEX_ENABLE": "0",
                    "AGIBOT_ENABLE": "0",
                    "ABC_ENABLE": "0",
                    "ALLOW_DATA_DOWNLOAD": "0",
                }
            )
            env.update({key: str(value) for key, value in overrides.items()})
            return subprocess.run(
                ["bash", str(SETUP), "datasets"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

    def test_default_datasets_command_downloads_nothing(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("no datasets enabled", result.stdout)

    def test_download_requires_explicit_ack(self):
        result = self._run(FETCH="download", DROID_ENABLE="1", DROID_LIMIT="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ALLOW_DATA_DOWNLOAD=1", result.stdout)

    def test_agibot_archive_download_is_blocked_even_with_ack(self):
        result = self._run(
            FETCH="download",
            ALLOW_DATA_DOWNLOAD="1",
            AGIBOT_ENABLE="1",
            AGIBOT_LIMIT="1",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AgiBot automatic download is blocked", result.stdout)

    def test_abc_requires_explicit_file_plan_before_download(self):
        result = self._run(
            FETCH="download",
            ALLOW_DATA_DOWNLOAD="1",
            ABC_ENABLE="1",
            ABC_LIMIT="1",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ABC_DOWNLOAD_PLAN is required", result.stdout)

    def test_egodex_requires_explicit_parts_before_download(self):
        result = self._run(
            FETCH="download",
            ALLOW_DATA_DOWNLOAD="1",
            EGODEX_ENABLE="1",
            EGODEX_LIMIT="1",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EGODEX_PARTS must explicitly list", result.stdout)

    def test_unbounded_limit_is_rejected_before_download(self):
        result = self._run(DROID_LIMIT="all")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("positive finite integer", result.stdout)


if __name__ == "__main__":
    unittest.main()
