from contextlib import contextmanager, nullcontext
import unittest
from unittest import mock

import torch
import torch.nn as nn

from robot_wm.utils.trainer import Metrics, Trainer


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.aux_losses = {}
        self.forward_count = 0

    def forward(self, value):
        self.forward_count += 1
        loss = (self.weight * value).mean()
        self.aux_losses = {"twice": 2 * loss.detach()}
        return loss


class _DDPStub(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.no_sync_entries = 0

    def forward(self, **batch):
        return self.module(**batch)

    @contextmanager
    def no_sync(self):
        self.no_sync_entries += 1
        yield


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=1.0)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class _FakeScaler:
    def __init__(self):
        self.unscale_count = 0
        self.step_count = 0
        self.update_count = 0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        self.unscale_count += 1

    def step(self, optimizer):
        self.step_count += 1
        optimizer.step()

    def update(self):
        self.update_count += 1


def _make_trainer(*, accumulation_steps, values, use_amp=False):
    trainer = Trainer.__new__(Trainer)
    core = _ToyModel()
    trainer.model = _DDPStub(core)
    trainer.optimizer = _CountingSGD(trainer.model.parameters())
    trainer.lr_scheduler = mock.Mock()
    trainer.metrics = Metrics(world_size=1, batch_size=2)
    trainer.batch_size = 2
    trainer.gradient_accumulation_steps = accumulation_steps
    trainer._data_loader_iter = iter(
        {"value": torch.tensor([float(value)])} for value in values
    )
    trainer._to_device = lambda batch: batch
    trainer._require_finite_losses = mock.Mock()
    trainer._clip_grad_norm = mock.Mock(return_value=torch.tensor(1.25))
    trainer.local_rank = 0
    trainer.world_size = 1
    trainer.use_amp = use_amp
    trainer.dtype = torch.bfloat16
    trainer.scaler = _FakeScaler() if use_amp else None
    return trainer, core


class TrainerGradientAccumulationTest(unittest.TestCase):
    def test_operational_metrics_use_cpu_for_gloo_and_reduce_max(self):
        trainer = Trainer.__new__(Trainer)
        trainer.local_rank = 0
        trainer.world_size = 2
        trainer.batch_size = 4
        trainer.gradient_accumulation_steps = 2

        def reduce_max(values, op):
            self.assertEqual(values.device.type, "cpu")
            self.assertEqual(op, torch.distributed.ReduceOp.MAX)
            values.copy_(torch.tensor([3.0, 4.0, 5.0, 6.0]))

        with mock.patch(
            "robot_wm.utils.trainer.dist.is_initialized", return_value=True
        ), mock.patch(
            "robot_wm.utils.trainer.torch.distributed.get_backend",
            return_value="gloo",
        ), mock.patch(
            "robot_wm.utils.trainer.torch.distributed.all_reduce",
            side_effect=reduce_max,
        ), mock.patch(
            "robot_wm.utils.trainer.torch.cuda.is_available", return_value=False
        ):
            metrics = trainer._collect_operational_metrics(torch.tensor(1.0), 2.0)

        self.assertEqual(metrics["system/gradient_norm_max"], 3.0)
        self.assertEqual(metrics["system/optimizer_step_seconds_max"], 4.0)
        self.assertEqual(metrics["system/gpu_memory_allocated_gib_max"], 5.0)
        self.assertEqual(metrics["system/gpu_memory_reserved_gib_max"], 6.0)
        self.assertEqual(metrics["system/effective_global_batch_size"], 16)

    def test_accumulation_config_defaults_and_requires_positive_integer(self):
        self.assertEqual(Trainer._parse_gradient_accumulation_steps({}), 1)
        self.assertEqual(
            Trainer._parse_gradient_accumulation_steps(
                {"gradient_accumulation_steps": 4}
            ),
            4,
        )
        for value in (0, -1, 1.5, True, "2", None):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "positive integer"
            ):
                Trainer._parse_gradient_accumulation_steps(
                    {"gradient_accumulation_steps": value}
                )

    def test_non_amp_accumulates_mean_gradient_and_losses(self):
        trainer, core = _make_trainer(
            accumulation_steps=3, values=(1, 3, 5)
        )

        with mock.patch(
            "robot_wm.utils.trainer.dist.is_initialized", return_value=False
        ):
            losses = trainer._step()

        self.assertAlmostEqual(core.weight.item(), -2.0)
        self.assertAlmostEqual(losses["loss"], 3.0)
        self.assertAlmostEqual(losses["twice"], 6.0)
        self.assertEqual(trainer.model.no_sync_entries, 2)
        self.assertEqual(trainer.optimizer.step_count, 1)
        trainer.lr_scheduler.step.assert_called_once_with()
        trainer._clip_grad_norm.assert_called_once_with()
        self.assertEqual(trainer.metrics.total_observations, 6)
        self.assertEqual(trainer.metrics.samples_since_log, 6)
        self.assertEqual(
            trainer._last_operational_metrics["system/effective_global_batch_size"],
            6,
        )
        self.assertAlmostEqual(
            trainer._last_operational_metrics["system/gradient_norm_max"], 1.25
        )

    def test_amp_unscales_clips_and_steps_once_per_window(self):
        trainer, core = _make_trainer(
            accumulation_steps=2, values=(1, 3), use_amp=True
        )

        with mock.patch(
            "robot_wm.utils.trainer.dist.is_initialized", return_value=False
        ), mock.patch(
            "robot_wm.utils.trainer.torch.autocast", return_value=nullcontext()
        ):
            losses = trainer._step()

        self.assertAlmostEqual(core.weight.item(), -1.0)
        self.assertAlmostEqual(losses["loss"], 2.0)
        self.assertEqual(trainer.model.no_sync_entries, 1)
        self.assertEqual(trainer.scaler.unscale_count, 1)
        self.assertEqual(trainer.scaler.step_count, 1)
        self.assertEqual(trainer.scaler.update_count, 1)
        self.assertEqual(trainer.optimizer.step_count, 1)
        trainer._clip_grad_norm.assert_called_once_with()
        trainer.lr_scheduler.step.assert_called_once_with()
        self.assertEqual(trainer.metrics.total_observations, 4)

    def test_accumulation_one_preserves_single_batch_behavior(self):
        trainer, core = _make_trainer(accumulation_steps=1, values=(7,))

        with mock.patch(
            "robot_wm.utils.trainer.dist.is_initialized", return_value=False
        ):
            losses = trainer._step()

        self.assertAlmostEqual(core.weight.item(), -6.0)
        self.assertAlmostEqual(losses["loss"], 7.0)
        self.assertEqual(trainer.model.no_sync_entries, 0)
        self.assertEqual(trainer.optimizer.step_count, 1)
        self.assertEqual(trainer.metrics.total_observations, 2)

    def test_reschedule_checkpoint_waits_for_complete_accumulation_window(self):
        trainer, core = _make_trainer(
            accumulation_steps=3, values=(1, 3, 5)
        )
        trainer._start_iter = 0
        trainer._curr_iter = 0
        trainer.max_iter = 2
        trainer.log_every = 1_000
        trainer.val_every = 1_000
        trainer.viz_every = 1_000
        trainer.save_every = 1_000
        trainer.viz_data_loaders = None
        trainer._validate = mock.Mock(return_value={})
        trainer._log = mock.Mock()
        forward_counts_at_save = []
        trainer._save_snapshot = mock.Mock(
            side_effect=lambda: forward_counts_at_save.append(core.forward_count)
        )
        trainer._write_checkpoint_ack = mock.Mock()
        trainer._write_completion_marker = mock.Mock()
        trainer._install_checkpoint_signal_handlers = mock.Mock()
        trainer._restore_checkpoint_signal_handlers = mock.Mock()
        trainer._checkpoint_stop_requested_across_ranks = mock.Mock(
            side_effect=[False, True]
        )

        with mock.patch(
            "robot_wm.utils.trainer.dist.is_initialized", return_value=False
        ):
            outcome = trainer.train()

        self.assertEqual(outcome, "rescheduled")
        self.assertEqual(core.forward_count, 3)
        self.assertEqual(forward_counts_at_save, [3])
        trainer._write_checkpoint_ack.assert_called_once_with(
            checkpoint_written=True, next_iter=1
        )
        trainer._write_completion_marker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
