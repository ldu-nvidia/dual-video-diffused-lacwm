import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.validate_training_data import (
    DatasetReport,
    _check_agibot_camera_json,
    _check_agibot_timestamps,
    _check_cap,
    build_parser,
    main,
    parse_cap,
)


class ValidateTrainingDataTest(unittest.TestCase):
    def test_parse_cap(self):
        self.assertIsNone(parse_cap("all"))
        self.assertEqual(parse_cap("10000"), 10_000)

    def test_cap_cannot_reach_expected_count(self):
        report = DatasetReport("x", "/tmp/x", cap=5, expected=6)
        _check_cap(report, available=10)
        self.assertFalse(report.passed)
        self.assertEqual(report.selected, 5)

    def test_agibot_profile_defaults_to_production(self):
        args = build_parser().parse_args(["--datasets", "agibot"])
        self.assertEqual(args.agibot_profile, "production")

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

    def _make_agibot_files_only(self, root: Path) -> Path:
        agibot = root / "agibot"
        task, episode = "1", "2"
        (agibot / "manifest.csv").parent.mkdir(parents=True, exist_ok=True)
        (agibot / "manifest.csv").write_text(
            f"task_id,episode_id,dataset\n{task},{episode},scr\n",
            encoding="utf-8",
        )
        proprio = agibot / "proprio_stats" / task / episode / "proprio_stats.h5"
        videos = agibot / "observations" / task / episode / "videos"
        cameras = (
            agibot
            / "parameters"
            / task
            / episode
            / "parameters"
            / "camera"
        )
        proprio.parent.mkdir(parents=True, exist_ok=True)
        videos.mkdir(parents=True, exist_ok=True)
        cameras.mkdir(parents=True, exist_ok=True)
        proprio.write_bytes(b"placeholder")
        for name in ("head_color.mp4", "hand_left_color.mp4", "hand_right_color.mp4"):
            (videos / name).write_bytes(b"placeholder")
        for name in (
            "head_extrinsic_params_aligned.json",
            "hand_left_extrinsic_params_aligned.json",
            "hand_right_extrinsic_params_aligned.json",
        ):
            (cameras / name).write_text("[]", encoding="utf-8")
        return agibot

    def _write_agibot_production_lineage(self, agibot: Path) -> None:
        manifest = agibot / "manifest.csv"
        success = agibot / "manifest.success.csv"
        success.write_bytes(manifest.read_bytes())
        payload_paths = sorted(
            [
                agibot / "proprio_stats" / "1" / "2" / "proprio_stats.h5",
                *(
                    agibot / "observations" / "1" / "2" / "videos" / name
                    for name in (
                        "head_color.mp4",
                        "hand_left_color.mp4",
                        "hand_right_color.mp4",
                    )
                ),
                *(
                    agibot
                    / "parameters"
                    / "1"
                    / "2"
                    / "parameters"
                    / "camera"
                    / name
                    for name in (
                        "head_extrinsic_params_aligned.json",
                        "hand_left_extrinsic_params_aligned.json",
                        "hand_right_extrinsic_params_aligned.json",
                    )
                ),
            ],
            key=str,
        )
        payload_manifest = agibot / "payloads.sha256"
        payload_manifest.write_text(
            "".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                f"{path.relative_to(agibot).as_posix()}\n"
                for path in payload_paths
            ),
            encoding="utf-8",
        )
        plan_hash = "a" * 64
        (agibot / ".agibot_archive_plan.sha256").write_text(plan_hash + "\n")
        report = {
            "profile": "production",
            "preparer_sha256": hashlib.sha256(
                (Path(__file__).resolve().parents[2] / "tools" / "prepare_agibot.py").read_bytes()
            ).hexdigest(),
            "passed": True,
            "archive_verified": True,
            "official_inventory_verified": True,
            "source": {
                "repo_id": "agibot-world/AgiBotWorld-Alpha",
                "revision": "128665c9e0244c45d1cbe5c13f5a4706afd24f27",
            },
            "root": str(agibot.resolve()),
            "synthetic_camera_motion": False,
            "synthetic_base_pose": False,
            "archive_plan_sha256": plan_hash,
            "manifest": str(manifest.resolve()),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "selected_count": 1,
            "archive_results": [
                {
                    "section": section,
                    "archive": f"{section}/fixture.tar",
                    "sha256": str(index) * 64,
                    **(
                        {"covered_planned_payload_count": 7}
                        if index == 1
                        else {}
                    ),
                }
                for index, section in enumerate(
                    ("observations", "parameters", "proprio_stats"), 1
                )
            ],
            "official_inventory": [
                {
                    "section": section,
                    "path": f"{section}/fixture.tar",
                    "sha256": str(index) * 64,
                    "size": 1,
                }
                for index, section in enumerate(
                    ("observations", "parameters", "proprio_stats"), 1
                )
            ],
            "success_manifest": str(success.resolve()),
            "success_manifest_sha256": hashlib.sha256(success.read_bytes()).hexdigest(),
            "payload_manifest": str(payload_manifest.resolve()),
            "payload_manifest_sha256": hashlib.sha256(
                payload_manifest.read_bytes()
            ).hexdigest(),
            "payload_count": 7,
        }
        (agibot / "preparation_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

    def _run_agibot_files_only(
        self,
        root: Path,
        profile: str | None = None,
        upstream_error: str | None = None,
        extra_args: list[str] | None = None,
    ) -> tuple[int, str]:
        argv = [
            "--data-root", str(root),
            "--datasets", "agibot",
            "--files-only",
            "--agibot-cap", "1",
            "--agibot-expected", "1",
        ]
        if profile is not None:
            argv.extend(("--agibot-profile", profile))
        if extra_args:
            argv.extend(extra_args)
        output = io.StringIO()
        with mock.patch(
            "tools.validate_training_data._verify_agibot_official_inventory",
            return_value=upstream_error,
        ), contextlib.redirect_stdout(output):
            status = main(argv)
        return status, output.getvalue()

    def test_production_agibot_rejects_qualification_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_agibot_files_only(root)
            (root / "qualification_provenance.json").write_text(
                json.dumps(
                    {
                        "mode": "QUALIFICATION_ONLY_STATIC_EXTRINSIC_REPETITION",
                        "synthesized_identity_base_pose_lengths": {"1/2": 66},
                    }
                ),
                encoding="utf-8",
            )
            status, output = self._run_agibot_files_only(root)
            self.assertEqual(status, 1)
            self.assertIn("production profile forbids", output)

    def test_production_agibot_requires_positive_archive_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_agibot_files_only(root)
            status, output = self._run_agibot_files_only(root)
            self.assertEqual(status, 1)
            self.assertIn("archive-bound preparation_report", output)

    def test_production_agibot_explicitly_rejects_multiple_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agibot = self._make_agibot_files_only(root)
            viscam = root / "agibot_combined"
            task, episode = "3", "4"
            proprio = viscam / "proprio_stats" / task / episode / "proprio_stats.h5"
            videos = viscam / "observations" / task / episode / "videos"
            cameras = viscam / "parameters" / task / episode / "parameters" / "camera"
            proprio.parent.mkdir(parents=True)
            videos.mkdir(parents=True)
            cameras.mkdir(parents=True)
            proprio.write_bytes(b"placeholder")
            for name in ("head_color.mp4", "hand_left_color.mp4", "hand_right_color.mp4"):
                (videos / name).write_bytes(b"placeholder")
            for name in (
                "head_extrinsic_params_aligned.json",
                "hand_left_extrinsic_params_aligned.json",
                "hand_right_extrinsic_params_aligned.json",
            ):
                (cameras / name).write_text("[]", encoding="utf-8")
            (agibot / "manifest.csv").write_text(
                "task_id,episode_id,dataset\n1,2,scr\n3,4,viscam\n",
                encoding="utf-8",
            )
            status, output = self._run_agibot_files_only(
                root,
                extra_args=[
                    "--agibot-cap", "2",
                    "--agibot-expected", "2",
                    "--agibot-viscam-root", str(viscam),
                ],
            )
            self.assertEqual(status, 1)
            self.assertIn("one canonical prepared root", output)

    def test_production_agibot_accepts_matching_archive_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agibot = self._make_agibot_files_only(root)
            self._write_agibot_production_lineage(agibot)
            status, output = self._run_agibot_files_only(root)
            self.assertEqual(status, 0, output)

    def test_production_agibot_rejects_inventory_not_verified_upstream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agibot = self._make_agibot_files_only(root)
            self._write_agibot_production_lineage(agibot)
            status, output = self._run_agibot_files_only(
                root,
                upstream_error="claimed archive is absent at pinned revision",
            )
            self.assertEqual(status, 1)
            self.assertIn("absent at pinned revision", output)

    def test_production_agibot_rejects_manifest_changed_after_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agibot = self._make_agibot_files_only(root)
            self._write_agibot_production_lineage(agibot)
            manifest = agibot / "manifest.csv"
            manifest.write_text(
                "task_id,episode_id,dataset\n1,3,scr\n", encoding="utf-8"
            )
            status, output = self._run_agibot_files_only(root)
            self.assertEqual(status, 1)
            self.assertIn("manifest hash", output)

    def test_production_agibot_rejects_same_size_payload_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agibot = self._make_agibot_files_only(root)
            self._write_agibot_production_lineage(agibot)
            video = agibot / "observations" / "1" / "2" / "videos" / "head_color.mp4"
            original = video.read_bytes()
            video.write_bytes(b"x" * len(original))
            status, output = self._run_agibot_files_only(root)
            self.assertEqual(status, 1)
            self.assertIn("no longer match verified archives", output)

    def test_production_agibot_rejects_do_not_train_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agibot = self._make_agibot_files_only(root)
            (agibot / "QUALIFICATION_ONLY_DO_NOT_TRAIN.md").write_text(
                "Qualification-only synthesized camera/base data.\n",
                encoding="utf-8",
            )
            status, output = self._run_agibot_files_only(root)
            self.assertEqual(status, 1)
            self.assertIn("QUALIFICATION_ONLY_DO_NOT_TRAIN.md", output)

    def test_qualification_agibot_permits_marker_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_agibot_files_only(root)
            (root / "qualification_provenance.json").write_text(
                json.dumps(
                    {
                        "mode": "QUALIFICATION_ONLY_STATIC_EXTRINSIC_REPETITION",
                        "warning": "AgiBot base pose was synthesized by repeating identity",
                    }
                ),
                encoding="utf-8",
            )
            status, output = self._run_agibot_files_only(root, "qualification")
            self.assertEqual(status, 0)
            self.assertIn("qualification provenance accepted explicitly", output)
            self.assertIn("pipeline-only", output)

    def test_production_agibot_rejects_generic_synthesis_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_agibot_files_only(root)
            (root / "source_provenance.json").write_text(
                json.dumps(
                    {
                        "dataset": "agibot",
                        "mode": "QUALIFICATION_ONLY_STATIC_EXTRINSIC_REPETITION",
                    }
                ),
                encoding="utf-8",
            )
            status, _ = self._run_agibot_files_only(root)
            self.assertEqual(status, 1)

    def test_agibot_timestamps_must_be_finite_and_strictly_increasing(self):
        path = Path("episode.h5")
        self.assertEqual(
            _check_agibot_timestamps([1, 2, 3], path),
            [],
        )
        duplicate = _check_agibot_timestamps([1, 2, 2], path)
        self.assertTrue(any("strictly increasing" in item.message for item in duplicate))
        nonfinite = _check_agibot_timestamps([1.0, float("nan"), 3.0], path)
        self.assertTrue(any("NaN/Inf" in item.message for item in nonfinite))

    def test_agibot_static_genuine_camera_extrinsics_are_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera.json"
            record = {
                "extrinsic": {
                    "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "translation_vector": [0.1, -0.2, 0.3],
                }
            }
            path.write_text(json.dumps([record, record]), encoding="utf-8")
            self.assertEqual(_check_agibot_camera_json(path, 2), [])

    def test_agibot_camera_extrinsics_require_finite_se3_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera.json"
            records = [
                {
                    "extrinsic": {
                        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
                        "translation_vector": [0, 0, 0],
                    }
                },
                {
                    "extrinsic": {
                        "rotation_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                        "translation_vector": [0, 0, 0],
                    }
                },
                {
                    "extrinsic": {
                        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "translation_vector": [0, float("nan"), 0],
                    }
                },
            ]
            path.write_text(json.dumps(records), encoding="utf-8")
            findings = _check_agibot_camera_json(path, 3)
            messages = [item.message for item in findings]
            self.assertTrue(any("determinant must be positive" in item for item in messages))
            self.assertTrue(any("not orthonormal" in item for item in messages))
            self.assertTrue(any("translation" in item and "NaN/Inf" in item for item in messages))

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

    def test_files_only_accepts_explicit_external_abc_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            overlay = base / "overlay"
            source = base / "canonical" / "abc_pp"
            episode = source / "task" / "episode_000001"
            episode.mkdir(parents=True)
            for name in ("states.npz", "top.mp4", "left_wrist.mp4", "right_wrist.mp4"):
                (episode / name).write_bytes(b"placeholder")
            manifest = overlay / "abc_pp" / "manifest.txt"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(f"{episode}\n", encoding="utf-8")

            self.assertEqual(self._run_abc_files_only(overlay), 1)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(
                    [
                        "--data-root", str(overlay),
                        "--datasets", "abc",
                        "--files-only",
                        "--abc-cap", "1",
                        "--abc-expected", "1",
                        "--abc-allowed-root", str(source),
                        "--json",
                    ]
                )
            self.assertEqual(status, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(
                payload["invocation"]["allowed_external_roots"]["abc"],
                str(source.resolve()),
            )

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
