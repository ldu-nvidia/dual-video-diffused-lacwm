import hashlib
import contextlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import av
    import h5py
    import numpy as np
except ImportError:  # pragma: no cover - the B200 runtime supplies both
    av = None
    h5py = None
    np = None

from tools.prepare_agibot import (
    ALIGNED_EXTRINSICS,
    H5_FIELDS,
    Archive,
    Episode,
    PreparationError,
    _payload_hash_manifest,
    _validate_video,
    discover_episodes,
    extract_plan,
    parse_archive_plan,
    parse_episode_plan,
    prepare,
    required_episode_payloads,
    sha256_file,
    main,
)
import tools.prepare_agibot as prepare_agibot_module
from tools.validate_training_data import (
    AgibotEntry,
    _agibot_paths,
    _deep_agibot,
    main as validate_training_data_main,
)


IDENTITY = {
    "extrinsic": {
        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "translation_vector": [0.0, 0.0, 0.0],
    }
}


@unittest.skipUnless(
    h5py is not None and np is not None and av is not None,
    "h5py/numpy/PyAV are required",
)
class PrepareAgibotTest(unittest.TestCase):
    @staticmethod
    def _video(path: Path, timesteps: int) -> None:
        container = av.open(str(path), mode="w")
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(timesteps):
            pixels = np.full((16, 16, 3), index, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

    def _episode(self, root: Path, task: str, episode: str, timesteps: int = 4) -> None:
        video_dir = root / "observations" / task / episode / "videos"
        camera_dir = root / "parameters" / task / episode / "parameters" / "camera"
        proprio = root / "proprio_stats" / task / episode / "proprio_stats.h5"
        video_dir.mkdir(parents=True)
        camera_dir.mkdir(parents=True)
        proprio.parent.mkdir(parents=True)
        for name in ("head_color.mp4", "hand_left_color.mp4", "hand_right_color.mp4"):
            self._video(video_dir / name, timesteps)
        records = [IDENTITY for _ in range(timesteps)]
        for name in ALIGNED_EXTRINSICS:
            (camera_dir / name).write_text(json.dumps(records), encoding="utf-8")
        with h5py.File(proprio, "w") as handle:
            for key, tail in H5_FIELDS.items():
                if key == "timestamp":
                    values = np.arange(timesteps, dtype=np.int64)
                elif key == "state/robot/orientation":
                    values = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (timesteps, 1))
                elif key in {"state/end/orientation", "action/end/orientation"}:
                    values = np.zeros((timesteps, *tail), dtype=np.float32)
                    values[..., 3] = 1.0
                else:
                    values = np.zeros((timesteps, *tail), dtype=np.float32)
                handle.create_dataset(key, data=values)

    def _run(
        self,
        root: Path,
        *,
        execute: bool,
        limit: int = 1,
        validate_all: bool = False,
        episode_plan=None,
    ):
        payload_manifest = None
        payload_manifest_sha256 = None
        payload_count = 0
        if execute:
            candidates = list(episode_plan) if episode_plan is not None else discover_episodes(root)
            hashes = {
                path: sha256_file(path)
                for path in required_episode_payloads(root, candidates)
                if path.is_file()
            }
            payload_manifest = root / "payloads.sha256"
            payload_manifest.write_text(
                _payload_hash_manifest(root, hashes), encoding="utf-8"
            )
            payload_manifest_sha256 = sha256_file(payload_manifest)
            payload_count = len(hashes)
        return prepare(
            root=root,
            limit=limit,
            dataset_id="scr",
            min_timesteps=2,
            manifest=root / "manifest.csv",
            success_manifest=root / "manifest.success.csv",
            report_path=root / "preparation_report.json",
            execute=execute,
            validate_all=validate_all,
            episode_plan=episode_plan,
            archive_verified=execute,
            archive_plan_sha256="a" * 64 if execute else None,
            official_inventory=(
                [{"path": "fixture.tar", "sha256": "b" * 64, "size": 1}]
                if execute
                else None
            ),
            payload_manifest_path=payload_manifest,
            payload_manifest_sha256=payload_manifest_sha256,
            payload_count=payload_count,
        )

    def test_valid_episode_publishes_atomic_manifests_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agibot"
            self._episode(root, "10", "20")
            report = self._run(root, execute=True)
            self.assertTrue(report["passed"])
            self.assertFalse(report["synthetic_camera_motion"])
            self.assertFalse(report["synthetic_base_pose"])
            expected = "task_id,episode_id,dataset\n10,20,scr\n"
            self.assertEqual((root / "manifest.csv").read_text(encoding="utf-8"), expected)
            self.assertEqual((root / "manifest.success.csv").read_text(encoding="utf-8"), expected)
            self.assertTrue(json.loads((root / "preparation_report.json").read_text())["passed"])

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agibot"
            self._episode(root, "10", "20")
            report = self._run(root, execute=False)
            self.assertFalse(report["executed"])
            self.assertFalse((root / "manifest.csv").exists())
            self.assertFalse((root / "manifest.success.csv").exists())
            self.assertFalse((root / "preparation_report.json").exists())

    def test_qualification_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "agibot"
            self._episode(root, "10", "20")
            (parent / "QUALIFICATION_ONLY_DO_NOT_TRAIN.md").write_text("qualification")
            with self.assertRaisesRegex(PreparationError, "qualification-only"):
                self._run(root, execute=True)
            self.assertFalse((root / "manifest.csv").exists())

    def test_renamed_agibot_synthesis_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "agibot"
            self._episode(root, "10", "20")
            (parent / "source_provenance.json").write_text(
                json.dumps(
                    {
                        "dataset": "agibot",
                        "mode": "QUALIFICATION_ONLY_STATIC_EXTRINSIC_REPETITION",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PreparationError, "qualification-only"):
                self._run(root, execute=True)

    def test_incomplete_episode_is_excluded_not_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agibot"
            self._episode(root, "10", "bad")
            self._episode(root, "10", "good")
            (root / "observations" / "10" / "bad" / "videos" / "head_color.mp4").unlink()
            report = self._run(root, execute=True, validate_all=True)
            self.assertEqual(report["accepted_count"], 1)
            self.assertEqual(report["rejected_count"], 1)
            manifest = (root / "manifest.csv").read_text(encoding="utf-8")
            self.assertIn("10,good,scr", manifest)
            self.assertNotIn("10,bad,scr", manifest)

    def test_empty_base_pose_stream_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agibot"
            self._episode(root, "10", "20")
            proprio = root / "proprio_stats" / "10" / "20" / "proprio_stats.h5"
            with h5py.File(proprio, "r+") as handle:
                del handle["state/robot/position"]
                handle.create_dataset("state/robot/position", data=np.zeros((0,), dtype=np.float32))
            with self.assertRaisesRegex(PreparationError, "only 0 production-valid"):
                self._run(root, execute=True)

    def test_non_so3_camera_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agibot"
            self._episode(root, "10", "20")
            path = root / "parameters" / "10" / "20" / "parameters" / "camera" / ALIGNED_EXTRINSICS[0]
            records = json.loads(path.read_text())
            records[0]["extrinsic"]["rotation_matrix"][0][0] = 2.0
            path.write_text(json.dumps(records))
            with self.assertRaisesRegex(PreparationError, "only 0 production-valid"):
                self._run(root, execute=True)

    def test_corrupt_video_is_not_certified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agibot"
            self._episode(root, "10", "20")
            video = root / "observations" / "10" / "20" / "videos" / "head_color.mp4"
            video.write_bytes(b"not-a-video")
            with self.assertRaisesRegex(PreparationError, "only 0 production-valid"):
                self._run(root, execute=True)

    def test_middle_decode_error_is_not_hidden_by_valid_first_and_tail_frames(self):
        class Frame:
            width = 16
            height = 16

        class CodecContext:
            options = {}

        class Stream:
            width = 16
            height = 16
            frames = 4
            duration = 4
            codec_context = CodecContext()

        class Streams:
            video = [Stream()]

        class Container:
            streams = Streams()
            calls = 0

            def decode(self, _stream):
                self.calls += 1
                if self.calls == 1:
                    yield Frame()
                    raise RuntimeError("corrupt decoded frame in middle GOP")
                yield Frame()  # A tail-only probe would incorrectly pass.

            def close(self):
                pass

        with mock.patch("av.open", return_value=Container()):
            errors = _validate_video(Path("middle-corrupt.mp4"), 4)
        self.assertTrue(any("middle GOP" in message for message in errors), errors)

    def test_strict_full_decode_rejects_real_middle_gop_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "middle-corrupt.mp4"
            container = av.open(str(video), mode="w")
            stream = container.add_stream("mpeg4", rate=30)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = "yuv420p"
            stream.gop_size = 10
            y, x = np.indices((64, 64))
            for index in range(120):
                pixels = np.stack(
                    (
                        (x * 4 + index * 3) % 256,
                        (y * 4 + index * 7) % 256,
                        ((x + y) * 2 + index * 11) % 256,
                    ),
                    axis=-1,
                ).astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()

            container = av.open(str(video), mode="r")
            keyframes = [
                (packet.pos, packet.size)
                for packet in container.demux(video=0)
                if packet.is_keyframe
                and packet.pos is not None
                and packet.pos >= 0
                and packet.size >= 16
            ]
            container.close()
            self.assertGreaterEqual(len(keyframes), 3)
            position, size = keyframes[len(keyframes) // 2]
            payload = bytearray(video.read_bytes())
            start = position + size // 8
            end = position + 7 * size // 8
            payload[start:end] = b"\0" * (end - start)
            video.write_bytes(payload)

            errors = _validate_video(video, 120)
            self.assertTrue(errors, "middle-GOP corruption was incorrectly certified")
            self.assertTrue(
                any("corrupt" in message or "decode" in message for message in errors),
                errors,
            )

    def test_decoded_count_is_authoritative_when_container_metadata_is_absent(self):
        class Frame:
            width = 16
            height = 16
            is_corrupt = False

        class CodecContext:
            options = {}

        class Stream:
            width = 16
            height = 16
            frames = 0
            duration = 0
            codec_context = CodecContext()

        class Streams:
            video = [Stream()]

        class Container:
            streams = Streams()

            def decode(self, _stream):
                yield from (Frame() for _ in range(4))

            def close(self):
                pass

        with mock.patch("av.open", return_value=Container()):
            self.assertEqual(_validate_video(Path("metadata-free.mp4"), 4), [])

    def test_invalid_end_quaternion_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agibot"
            self._episode(root, "10", "20")
            proprio = root / "proprio_stats" / "10" / "20" / "proprio_stats.h5"
            with h5py.File(proprio, "r+") as handle:
                handle["state/end/orientation"][0, 0] = np.zeros(4)
            with self.assertRaisesRegex(PreparationError, "only 0 production-valid"):
                self._run(root, execute=True)

    def test_static_base_with_nonzero_velocity_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agibot"
            self._episode(root, "10", "20")
            proprio = root / "proprio_stats" / "10" / "20" / "proprio_stats.h5"
            with h5py.File(proprio, "r+") as handle:
                velocity = np.zeros((4, 2), dtype=np.float32)
                velocity[1, 0] = 0.5
                handle.create_dataset("action/robot/velocity", data=velocity)
            with self.assertRaisesRegex(PreparationError, "only 0 production-valid"):
                self._run(root, execute=True)

    def test_deep_validator_rejects_moving_zero_quaternion_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agibot"
            self._episode(root, "10", "20")
            proprio = root / "proprio_stats" / "10" / "20" / "proprio_stats.h5"
            with h5py.File(proprio, "r+") as handle:
                handle["state/robot/orientation"][...] = np.zeros((4, 4))
                moving = np.zeros((4, 3), dtype=np.float32)
                moving[:, 0] = np.arange(4)
                handle["state/robot/position"][...] = moving
            paths = _agibot_paths(AgibotEntry("10", "20", "scr"), {"scr": root})
            findings = _deep_agibot(paths, h5py, av, 2, "production")
            self.assertTrue(
                any("stationary fixed base" in item.message for item in findings),
                findings,
            )

    def test_episode_plan_controls_selection_from_coarse_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agibot"
            self._episode(root, "10", "first")
            self._episode(root, "10", "wanted")
            report = self._run(
                root,
                execute=True,
                episode_plan=[Episode("10", "wanted")],
            )
            self.assertTrue(report["episode_plan_supplied"])
            manifest = (root / "manifest.csv").read_text(encoding="utf-8")
            self.assertIn("10,wanted,scr", manifest)
            self.assertNotIn("10,first,scr", manifest)

    def test_cli_publishes_verified_archives_transactionally(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            raw = base / "raw"
            final = base / "published" / "agibot"
            self._episode(source, "10", "20")

            archive_specs = (
                ("observations", "observations/10/episodes.tar", source / "observations" / "10"),
                ("parameters", "parameters/all.tar", source / "parameters"),
                ("proprio_stats", "proprio_stats/all.tar", source / "proprio_stats"),
            )
            plan_lines = []
            for section, relative, tree in archive_specs:
                archive = raw / relative
                archive.parent.mkdir(parents=True, exist_ok=True)
                with tarfile.open(archive, "w") as bundle:
                    for path in sorted(tree.rglob("*")):
                        bundle.add(path, arcname=str(path.relative_to(tree)), recursive=False)
                plan_lines.append(
                    f"{section} {relative} {hashlib.sha256(archive.read_bytes()).hexdigest()}"
                )
            archive_plan = base / "archives.plan"
            archive_plan.write_text("\n".join(plan_lines) + "\n", encoding="utf-8")
            episode_plan = base / "episodes.csv"
            episode_plan.write_text(
                "task_id,episode_id,dataset\n10,20,scr\n", encoding="utf-8"
            )

            stdout = io.StringIO()
            stderr = io.StringIO()
            inventory = [
                {"section": section, "path": relative, "sha256": line.split()[-1], "size": 1}
                for (section, relative, _tree), line in zip(archive_specs, plan_lines)
            ]
            with mock.patch(
                "tools.prepare_agibot.verify_official_archive_plan",
                return_value=inventory,
            ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "--root", str(final),
                        "--limit", "1",
                        "--min-timesteps", "2",
                        "--archive-root", str(raw),
                        "--archive-plan", str(archive_plan),
                        "--episode-plan", str(episode_plan),
                        "--execute",
                    ]
                )
            self.assertEqual(status, 0, stderr.getvalue())
            self.assertTrue((final / "manifest.csv").is_file())
            report = json.loads((final / "preparation_report.json").read_text())
            self.assertEqual(report["profile"], "production")
            self.assertTrue(report["archive_verified"])
            self.assertEqual(report["root"], str(final.resolve()))
            self.assertEqual(
                report["manifest_sha256"],
                hashlib.sha256((final / "manifest.csv").read_bytes()).hexdigest(),
            )
            with mock.patch(
                "tools.validate_training_data._verify_agibot_official_inventory",
                return_value=None,
            ), contextlib.redirect_stdout(io.StringIO()):
                validation_status = validate_training_data_main(
                    [
                        "--data-root", str(final.parent),
                        "--datasets", "agibot",
                        "--agibot-cap", "1",
                        "--agibot-expected", "1",
                        "--min-timesteps", "2",
                        "--workers", "1",
                        "--agibot-profile", "production",
                    ]
                )
            self.assertEqual(validation_status, 0)
            self.assertFalse(any(final.parent.glob(".agibot.preparing-*")))
            original_manifest = (final / "manifest.csv").read_bytes()
            with mock.patch(
                "tools.prepare_agibot.verify_official_archive_plan",
                return_value=inventory,
            ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                repeated = main(
                    [
                        "--root", str(final),
                        "--limit", "1",
                        "--min-timesteps", "2",
                        "--archive-root", str(raw),
                        "--archive-plan", str(archive_plan),
                        "--episode-plan", str(episode_plan),
                        "--execute",
                    ]
                )
            self.assertEqual(repeated, 1)
            self.assertEqual((final / "manifest.csv").read_bytes(), original_manifest)


class AgibotArchivePlanTest(unittest.TestCase):
    def _tar(self, path: Path, name: str = "1/2/file.bin", *, symlink: bool = False) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "w") as bundle:
            info = tarfile.TarInfo(name)
            if symlink:
                info.type = tarfile.SYMTYPE
                info.linkname = "/tmp/escape"
                bundle.addfile(info)
            else:
                payload = b"payload"
                info.size = len(payload)
                bundle.addfile(info, io.BytesIO(payload))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_plan_requires_all_sections_and_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / "plan.txt"
            plan.write_text("observations observations/a.tar " + "0" * 64 + "\n")
            with self.assertRaisesRegex(PreparationError, "omits required sections"):
                parse_archive_plan(plan)

    def test_episode_plan_is_exact_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = Path(tmp) / "episodes.csv"
            plan.write_text(
                "task_id,episode_id,dataset\n1,2,scr\n3,4,scr\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_episode_plan(plan), [Episode("1", "2"), Episode("3", "4")])
            with self.assertRaisesRegex(PreparationError, "does not match"):
                parse_episode_plan(plan, expected_dataset_id="alpha")
            plan.write_text(
                "task_id,episode_id\n1,2\n1,2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PreparationError, "duplicate"):
                parse_episode_plan(plan)
            plan.write_text(
                "task_id,episode_id\n../escape,2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PreparationError, "unsafe task/episode"):
                parse_episode_plan(plan)

    def test_checked_archives_extract_under_declared_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            output = Path(tmp) / "out"
            archives = []
            for section in ("observations", "parameters", "proprio_stats"):
                relative = (
                    "observations/327/part.tar"
                    if section == "observations"
                    else f"{section}/part.tar"
                )
                digest = self._tar(raw / relative)
                archives.append(Archive(section, relative, digest))
            result = extract_plan(raw, output, archives, execute=True)
            self.assertEqual(len(result), 3)
            self.assertEqual(
                (output / "observations" / "327" / "1" / "2" / "file.bin").read_bytes(),
                b"payload",
            )
            for section in ("parameters", "proprio_stats"):
                self.assertEqual(
                    (output / section / "1" / "2" / "file.bin").read_bytes(),
                    b"payload",
                )
            resumed = extract_plan(raw, output, archives, execute=True)
            self.assertTrue(all(item["reused"] == 1 for item in resumed))
            self.assertTrue(all(item["written"] == 0 for item in resumed))
            observation_payload = output / "observations" / "327" / "1" / "2" / "file.bin"
            observation_payload.write_bytes(b"corrupt")  # same length as b"payload"
            repaired = extract_plan(raw, output, archives, execute=True)
            observation_result = next(item for item in repaired if item["section"] == "observations")
            self.assertEqual(observation_result["written"], 1)
            self.assertEqual(observation_payload.read_bytes(), b"payload")

    def test_symlink_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            relative = "observations/bad.tar"
            digest = self._tar(raw / relative, symlink=True)
            with self.assertRaisesRegex(PreparationError, "non-regular"):
                extract_plan(raw, Path(tmp) / "out", [Archive("observations", relative, digest)], execute=True)

    def test_symlinked_section_cannot_redirect_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            output = root / "out"
            outside = root / "outside"
            output.mkdir()
            outside.mkdir()
            (output / "observations").symlink_to(outside, target_is_directory=True)
            relative = "observations/327/data.tar"
            digest = self._tar(raw / relative)
            with self.assertRaisesRegex(PreparationError, "escapes output root|symlinked"):
                extract_plan(
                    raw,
                    output,
                    [Archive("observations", relative, digest)],
                    execute=True,
                )
            self.assertEqual(list(outside.rglob("*")), [])

    def test_archive_path_replacement_cannot_change_verified_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            raw = base / "raw"
            output = base / "out"
            relative = "observations/327/data.tar"
            archive_path = raw / relative
            digest = self._tar(archive_path)
            replacement = base / "replacement.tar"
            self._tar(replacement, name="1/2/evil.bin")
            original = prepare_agibot_module._extract_archive
            replaced = False

            def replace_after_hash(path, handle, *args, **kwargs):
                nonlocal replaced
                if not replaced:
                    os.replace(replacement, path)
                    replaced = True
                return original(path, handle, *args, **kwargs)

            with mock.patch(
                "tools.prepare_agibot._extract_archive",
                side_effect=replace_after_hash,
            ):
                extract_plan(
                    raw,
                    output,
                    [Archive("observations", relative, digest)],
                    execute=True,
                )
            self.assertEqual(
                (output / "observations" / "327" / "1" / "2" / "file.bin").read_bytes(),
                b"payload",
            )
            self.assertFalse((output / "observations" / "327" / "1" / "2" / "evil.bin").exists())

    def test_all_archive_hashes_pass_before_any_output_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            output = Path(tmp) / "out"
            good_path = "observations/327/good.tar"
            good_digest = self._tar(raw / good_path)
            bad_path = "parameters/bad.tar"
            self._tar(raw / bad_path)
            archives = [
                Archive("observations", good_path, good_digest),
                Archive("parameters", bad_path, "0" * 64),
            ]
            with self.assertRaisesRegex(PreparationError, "SHA-256 mismatch"):
                extract_plan(raw, output, archives, execute=True)
            self.assertFalse(output.exists())

    def test_cli_rejects_output_collision_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "agibot"
            episode_plan = base / "episodes.csv"
            episode_plan.write_text(
                "task_id,episode_id,dataset\n1,2,scr\n", encoding="utf-8"
            )
            archive_plan = base / "archives.plan"
            archive_plan.write_text(
                "\n".join(
                    f"{section} {section}/fixture.tar {'0' * 64}"
                    for section in ("observations", "parameters", "proprio_stats")
                )
                + "\n",
                encoding="utf-8",
            )
            base_args = [
                "--root", str(root),
                "--limit", "1",
                "--archive-root", str(base / "raw"),
                "--archive-plan", str(archive_plan),
                "--episode-plan", str(episode_plan),
                "--execute",
            ]
            cases = (
                (["--manifest", str(root / "payloads.sha256")], "collide with reserved"),
                (["--report", str(root / "custom-report.json")], "canonical path"),
                (["--manifest", str(base / "outside.csv")], "must be inside"),
            )
            for extra, expected in cases:
                with self.subTest(extra=extra):
                    stderr = io.StringIO()
                    with contextlib.redirect_stderr(stderr):
                        status = main([*base_args, *extra])
                    self.assertEqual(status, 1)
                    self.assertIn(expected, stderr.getvalue())
                    self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
