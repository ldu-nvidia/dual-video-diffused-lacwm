"""Static safety contract for the immutable V-JEPA cache-build launcher."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "tools" / "slurm" / "build_vjepa2_immutable_cache.sbatch"
SUBMIT = (
    ROOT / "tools" / "slurm" / "submit_build_vjepa2_immutable_cache.sh"
)


class VJEPA2CacheBuildLauncherTest(unittest.TestCase):
    def test_scientific_contract_is_literal_and_no_overwrite_is_used(self):
        source = SBATCH.read_text(encoding="utf-8")
        self.assertIn("readonly TRAIN_CLIPS=512", source)
        self.assertIn("readonly VAL_CLIPS=64", source)
        self.assertIn("readonly TEST_CLIPS=128", source)
        self.assertIn("readonly CLIPS_PER_EPISODE=1", source)
        self.assertIn("readonly BUILD_SEED=20260729", source)
        self.assertIn("readonly PCA_MAX_CLIPS=256", source)
        self.assertIn("readonly PCA_MAX_TOKENS=250000", source)
        self.assertNotIn('"--overwrite"', source)
        self.assertNotIn(" --overwrite", source)

    def test_allocation_is_full_node_b200_nonrequeueable_and_bounded(self):
        source = SBATCH.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gpus-per-node=8", source)
        self.assertIn("#SBATCH --time=04:00:00", source)
        self.assertIn("#SBATCH --exclusive", source)
        self.assertIn("#SBATCH --no-requeue", source)
        self.assertIn("all-B200 node", source)

    def test_submit_is_dry_run_first_and_fails_closed_on_active_jobs(self):
        source = SUBMIT.read_text(encoding="utf-8")
        dry_run = source.index("if ((EXECUTE == 0))")
        submit = source.index('JOB_ID="$("${COMMAND[@]}")"')
        self.assertLess(dry_run, submit)
        self.assertIn("check_active_user_jobs", source)
        self.assertIn("--allow-active-job-id", source)
        self.assertIn("--resume-existing", source)
        self.assertIn("repository must be clean", source)


if __name__ == "__main__":
    unittest.main()
