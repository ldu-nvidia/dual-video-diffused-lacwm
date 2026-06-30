import tempfile
import unittest
from pathlib import Path

from tools.build_training_manifests import ManifestError, build_abc, build_egodex


class BuildTrainingManifestsTest(unittest.TestCase):
    def test_egodex_is_capped_without_pipeline_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(3):
                (root / f"{index}.hdf5").write_bytes(b"h5")
                (root / f"{index}.mp4").write_bytes(b"mp4")
            output = root / "manifest.csv"
            self.assertEqual(build_egodex(root, output, 2), 2)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)

    def test_failed_build_does_not_replace_existing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "0.hdf5").write_bytes(b"h5")
            output = root / "manifest.csv"
            output.write_text("keep-me\n", encoding="utf-8")
            with self.assertRaises(ManifestError):
                build_egodex(root, output, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep-me\n")

    def test_abc_uses_only_success_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete = root / "abc_pp" / "task" / "episode_good"
            partial = root / "abc_pp" / "task" / "episode_partial"
            complete.mkdir(parents=True)
            partial.mkdir(parents=True)
            for name in ("states.npz", "top.mp4", "left_wrist.mp4", "right_wrist.mp4"):
                (complete / name).write_bytes(b"ok")
            (partial / "states.npz").write_bytes(b"partial")
            success = root / "abc_pp" / "manifest.success.txt"
            success.write_text(f"{complete}\n", encoding="utf-8")
            output = root / "abc_pp" / "manifest.txt"
            self.assertEqual(build_abc(success, output, 1), 1)
            self.assertEqual(output.read_text(encoding="utf-8").strip(), str(complete))

    def test_abc_refuses_to_overwrite_success_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.txt"
            path.write_text("/nonexistent\n", encoding="utf-8")
            with self.assertRaises(ManifestError):
                build_abc(path, path, 1)


if __name__ == "__main__":
    unittest.main()
