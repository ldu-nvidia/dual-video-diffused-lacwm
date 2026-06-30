import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tools.validate_training_data import DatasetReport, _check_cap, main, parse_cap


class ValidateTrainingDataTest(unittest.TestCase):
    def test_parse_cap(self):
        self.assertIsNone(parse_cap("all"))
        self.assertEqual(parse_cap("10000"), 10_000)

    def test_cap_cannot_reach_expected_count(self):
        report = DatasetReport("x", "/tmp/x", cap=5, expected=6)
        _check_cap(report, available=10)
        self.assertFalse(report.passed)
        self.assertEqual(report.selected, 5)

    def _run_abc_files_only(self, root: Path, expected: int = 1) -> int:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            return main(
                [
                    "--data-root", str(root),
                    "--datasets", "abc",
                    "--files-only",
                    "--abc-cap", "1",
                    "--abc-expected", str(expected),
                ]
            )

    def test_files_only_accepts_complete_abc_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = root / "abc_pp" / "task" / "episode_000001"
            episode.mkdir(parents=True)
            for name in ("states.npz", "top.mp4", "left_wrist.mp4", "right_wrist.mp4"):
                (episode / name).write_bytes(b"placeholder")
            manifest = root / "abc_pp" / "manifest.txt"
            manifest.write_text(f"{episode}\n", encoding="utf-8")
            self.assertEqual(self._run_abc_files_only(root), 0)

    def test_files_only_rejects_missing_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = root / "abc_pp" / "task" / "episode_000001"
            episode.mkdir(parents=True)
            for name in ("states.npz", "top.mp4", "left_wrist.mp4"):
                (episode / name).write_bytes(b"placeholder")
            manifest = root / "abc_pp" / "manifest.txt"
            manifest.write_text(f"{episode}\n", encoding="utf-8")
            self.assertEqual(self._run_abc_files_only(root), 1)

    def test_empty_manifest_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "abc_pp" / "manifest.txt"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("", encoding="utf-8")
            self.assertEqual(self._run_abc_files_only(root), 1)

    def test_json_report_binds_invocation_and_selected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = root / "abc_pp" / "task" / "episode_000001"
            episode.mkdir(parents=True)
            for name in ("states.npz", "top.mp4", "left_wrist.mp4", "right_wrist.mp4"):
                (episode / name).write_bytes(b"placeholder")
            (root / "abc_pp" / "manifest.txt").write_text(f"{episode}\n")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(
                    [
                        "--data-root", str(root),
                        "--datasets", "abc",
                        "--files-only",
                        "--abc-cap", "1",
                        "--abc-expected", "1",
                        "--json",
                    ]
                )
            self.assertEqual(status, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["invocation"]["min_timesteps"], 66)
            fingerprint = payload["reports"][0]["fingerprint"]
            self.assertEqual(fingerprint["file_count"], 4)
            self.assertEqual(len(fingerprint["digest"]), 64)


if __name__ == "__main__":
    unittest.main()
