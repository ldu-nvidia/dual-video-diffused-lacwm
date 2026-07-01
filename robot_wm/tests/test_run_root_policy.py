import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import run_root_policy, training_preflight


class RunRootPolicyTest(unittest.TestCase):
    def test_defaults_remain_allowed(self):
        roots = run_root_policy.configured_allowed_run_roots({})

        self.assertIn(Path("/mnt/data1").resolve(strict=False), roots)
        self.assertIn(Path("/mnt/data2").resolve(strict=False), roots)
        selected = run_root_policy.canonical_allowed_run_root(
            "/mnt/data1/lacwm/runs", roots
        )
        self.assertEqual(selected, Path("/mnt/data1/lacwm/runs"))

    def test_configured_lustre_root_extends_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            lustre = (Path(temporary) / "lustre" / "lacwm").resolve(strict=False)
            roots = run_root_policy.configured_allowed_run_roots(
                {run_root_policy.ENVIRONMENT_VARIABLE: str(lustre)}
            )

        self.assertIn(lustre, roots)
        self.assertIn(Path("/mnt/data1").resolve(strict=False), roots)
        self.assertEqual(
            run_root_policy.canonical_allowed_run_root(lustre / "runs", roots),
            lustre / "runs",
        )

    def test_multiple_configured_roots_are_colon_separated(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = (Path(temporary) / "first").resolve(strict=False)
            second = (Path(temporary) / "second").resolve(strict=False)
            roots = run_root_policy.configured_allowed_run_roots(
                {
                    run_root_policy.ENVIRONMENT_VARIABLE: os.pathsep.join(
                        (str(first), str(second))
                    )
                }
            )

        self.assertIn(first, roots)
        self.assertIn(second, roots)

    def test_invalid_configured_roots_fail_closed(self):
        invalid_values = (
            "",
            "relative/path",
            "/",
            "/safe::/other",
            ":/safe",
            "/safe:",
            "/safe/../other",
        )
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(
                run_root_policy.RunRootPolicyError
            ):
                run_root_policy.configured_allowed_run_roots(
                    {run_root_policy.ENVIRONMENT_VARIABLE: value}
                )

    def test_symlinked_configured_root_is_not_canonical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            link = root / "link"
            real.mkdir()
            link.symlink_to(real, target_is_directory=True)

            with self.assertRaisesRegex(
                run_root_policy.RunRootPolicyError, "must be canonical"
            ):
                run_root_policy.configured_allowed_run_roots(
                    {run_root_policy.ENVIRONMENT_VARIABLE: str(link)}
                )

    def test_selected_run_root_is_canonicalized_and_boundary_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            allowed = Path(temporary).resolve()
            selected = run_root_policy.canonical_allowed_run_root(
                allowed / "nested" / ".." / "runs", (allowed,)
            )
            self.assertEqual(selected, allowed / "runs")
            with self.assertRaisesRegex(
                run_root_policy.RunRootPolicyError, "outside allowed roots"
            ):
                run_root_policy.canonical_allowed_run_root(
                    allowed.parent / f"{allowed.name}-sibling", (allowed,)
                )

    def test_preflight_uses_explicit_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            allowed = Path(temporary).resolve()
            results = training_preflight.Results()
            with mock.patch.dict(
                os.environ,
                {run_root_policy.ENVIRONMENT_VARIABLE: str(allowed)},
                clear=False,
            ):
                training_preflight.check_output_root(
                    results, allowed / "runs", "assets"
                )

        checks = {item.name: item for item in results.checks}
        self.assertTrue(checks["run root policy"].ok)
        self.assertIn(str(allowed), checks["run root policy"].detail)


if __name__ == "__main__":
    unittest.main()
