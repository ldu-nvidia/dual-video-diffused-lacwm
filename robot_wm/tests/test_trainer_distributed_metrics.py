from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn as nn

from robot_wm.utils.trainer import Metrics, Trainer


class _ConditionalMetricModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.aux_losses = {}

    def forward(self, value):
        loss = (self.weight * value).mean()
        self.aux_losses = {"morph_local": 2 * loss.detach()}
        return loss


class _DDPStub(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, **batch):
        return self.module(**batch)

    @contextmanager
    def no_sync(self):
        yield


def _dynamic_metric_collectives(test_case):
    def gather(gathered, local_keys):
        test_case.assertEqual(local_keys, ("loss", "morph_local"))
        gathered[:] = [local_keys, ("loss", "morph_remote")]

    def reduce(packed, op):
        test_case.assertEqual(op, torch.distributed.ReduceOp.SUM)
        test_case.assertEqual(tuple(packed.shape), (3, 2))
        # Deterministic order is loss, morph_local, morph_remote. Locally the
        # first two metrics each have two contributions; the remote morphology
        # is absent and therefore has a zero count rather than a zero sample.
        torch.testing.assert_close(
            packed,
            torch.tensor(
                [[4.0, 2.0], [8.0, 2.0], [0.0, 0.0]],
                dtype=torch.float64,
            ),
        )
        packed.add_(
            torch.tensor(
                [[8.0, 2.0], [0.0, 0.0], [30.0, 3.0]],
                dtype=torch.float64,
            )
        )

    return gather, reduce


class TrainerDistributedMetricsTest(unittest.TestCase):
    def test_step_reduces_dynamic_keys_by_contributing_count(self):
        trainer = Trainer.__new__(Trainer)
        trainer.model = _DDPStub(_ConditionalMetricModel())
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
        trainer.lr_scheduler = mock.Mock()
        trainer.metrics = Metrics(world_size=2, batch_size=1)
        trainer.batch_size = 1
        trainer.gradient_accumulation_steps = 2
        trainer._data_loader_iter = iter(
            [
                {"value": torch.tensor([1.0])},
                {"value": torch.tensor([3.0])},
            ]
        )
        trainer._to_device = lambda batch: batch
        trainer._require_finite_losses = mock.Mock()
        trainer._clip_grad_norm = mock.Mock(return_value=torch.tensor(1.0))
        trainer._should_collect_operational_metrics = mock.Mock(return_value=False)
        trainer.local_rank = 0
        trainer.world_size = 2
        trainer.use_amp = False
        trainer.dtype = torch.float32
        trainer.scaler = None
        gather, reduce = _dynamic_metric_collectives(self)

        with mock.patch(
            "robot_wm.utils.trainer.dist.is_initialized", return_value=True
        ), mock.patch(
            "robot_wm.utils.trainer.torch.distributed.all_gather_object",
            side_effect=gather,
        ), mock.patch(
            "robot_wm.utils.trainer.torch.distributed.all_reduce",
            side_effect=reduce,
        ):
            losses = trainer._step()

        self.assertEqual(list(losses), ["loss", "morph_local", "morph_remote"])
        self.assertAlmostEqual(losses["loss"], 3.0)
        self.assertAlmostEqual(losses["morph_local"], 4.0)
        self.assertAlmostEqual(losses["morph_remote"], 10.0)

    def test_validate_reduces_dynamic_keys_by_contributing_count(self):
        trainer = Trainer.__new__(Trainer)
        trainer.model = _DDPStub(_ConditionalMetricModel())
        trainer._val_data_loader_iters = [
            iter(
                [
                    {"value": torch.tensor([1.0])},
                    {"value": torch.tensor([3.0])},
                ]
            )
        ]
        trainer.val_data_loaders = [
            SimpleNamespace(dataset=SimpleNamespace(name="toy"))
        ]
        trainer.n_val_samples = 2
        trainer._to_device = lambda batch: batch
        trainer._curr_iter = 7
        trainer.world_size = 2
        trainer.use_amp = False
        trainer.dtype = torch.float32
        gather, reduce = _dynamic_metric_collectives(self)

        with mock.patch(
            "robot_wm.utils.trainer.dist.is_initialized", return_value=True
        ), mock.patch(
            "robot_wm.utils.trainer.torch.distributed.all_gather_object",
            side_effect=gather,
        ), mock.patch(
            "robot_wm.utils.trainer.torch.distributed.all_reduce",
            side_effect=reduce,
        ), mock.patch(
            "robot_wm.utils.trainer.torch.autocast", return_value=nullcontext()
        ):
            losses = trainer._validate()

        self.assertEqual(
            list(losses),
            [
                "toy_0/loss",
                "toy_0/morph_local",
                "toy_0/morph_remote",
            ],
        )
        self.assertAlmostEqual(losses["toy_0/loss"], 3.0)
        self.assertAlmostEqual(losses["toy_0/morph_local"], 4.0)
        self.assertAlmostEqual(losses["toy_0/morph_remote"], 10.0)

    def test_best_validation_metrics_include_per_dataset_and_aggregate(self):
        metrics = Metrics(world_size=2, batch_size=1)
        metrics.best_val_loss_dataset_a = 3.0
        metrics.best_val_loss_dataset_b = 5.0
        metrics.best_val_loss_avg = 4.25

        self.assertEqual(metrics.refresh_best_val_loss(), 4.25)
        logged = metrics.get_val_metrics(10, {"avg/loss": 4.5})

        self.assertEqual(logged["val_loss/best/dataset_a"], 3.0)
        self.assertEqual(logged["val_loss/best/dataset_b"], 5.0)
        self.assertEqual(logged["val_loss/best/avg"], 4.25)
        self.assertEqual(logged["val_loss/best_val_loss"], 4.25)

    def test_train_refreshes_best_aggregate_before_saving_and_logging(self):
        trainer = Trainer.__new__(Trainer)
        trainer.model = mock.Mock()
        trainer._start_iter = 0
        trainer._curr_iter = 0
        trainer.max_iter = 1
        trainer.log_every = 1
        trainer.val_every = 1
        trainer.viz_every = 1
        trainer.save_every = 1
        trainer.viz_data_loaders = None
        trainer.optimizer = mock.Mock(param_groups=[{"lr": 1e-4}])
        trainer.metrics = Metrics(world_size=2, batch_size=1, warmup=0)
        trainer.save_best = True
        trainer._step = mock.Mock(return_value={"loss": 1.0})
        trainer._validate = mock.Mock(
            return_value={
                "dataset_a/loss": 3.0,
                "dataset_b/loss": 5.0,
                "avg/loss": 4.0,
            }
        )
        logged = []
        trainer._log = lambda values: logged.append(values)
        best_aggregates_at_save = []

        def save_snapshot(*, is_best=False, dataset_name=None):
            if is_best:
                best_aggregates_at_save.append(
                    (dataset_name, trainer.metrics.best_val_loss)
                )

        trainer._save_snapshot = mock.Mock(side_effect=save_snapshot)
        trainer._write_completion_marker = mock.Mock()
        trainer._install_checkpoint_signal_handlers = mock.Mock()
        trainer._restore_checkpoint_signal_handlers = mock.Mock()
        trainer._checkpoint_stop_requested_across_ranks = mock.Mock(
            return_value=False
        )

        outcome = trainer.train()

        self.assertEqual(outcome, "completed")
        self.assertEqual(
            best_aggregates_at_save,
            [("dataset_a", 4.0), ("dataset_b", 4.0), ("avg", 4.0)],
        )
        validation_log = logged[-1]
        self.assertEqual(validation_log["val_loss/best/dataset_a"], 3.0)
        self.assertEqual(validation_log["val_loss/best/dataset_b"], 5.0)
        self.assertEqual(validation_log["val_loss/best_val_loss"], 4.0)

    def test_save_best_false_updates_metrics_without_best_snapshots(self):
        trainer = Trainer.__new__(Trainer)
        trainer.model = mock.Mock()
        trainer._start_iter = 0
        trainer._curr_iter = 0
        trainer.max_iter = 1
        trainer.log_every = 1
        trainer.val_every = 1
        trainer.viz_every = 1
        trainer.save_every = 1
        trainer.viz_data_loaders = None
        trainer.optimizer = mock.Mock(param_groups=[{"lr": 1e-4}])
        trainer.metrics = Metrics(world_size=2, batch_size=1, warmup=0)
        trainer.save_best = False
        trainer._step = mock.Mock(return_value={"loss": 1.0})
        trainer._validate = mock.Mock(return_value={"dataset/loss": 3.0})
        trainer._log = mock.Mock()
        trainer._save_snapshot = mock.Mock()
        trainer._write_completion_marker = mock.Mock()
        trainer._install_checkpoint_signal_handlers = mock.Mock()
        trainer._restore_checkpoint_signal_handlers = mock.Mock()
        trainer._checkpoint_stop_requested_across_ranks = mock.Mock(
            return_value=False
        )

        outcome = trainer.train()

        self.assertEqual(outcome, "completed")
        self.assertEqual(trainer.metrics.best_val_loss_dataset, 3.0)
        self.assertEqual(trainer.metrics.best_val_loss, 3.0)
        # Only the ordinary final resume checkpoint is written.
        self.assertEqual(trainer._save_snapshot.call_args_list, [mock.call()])

    def test_throughput_warmup_is_configurable_and_zero_logs_first_step(self):
        self.assertEqual(Trainer._parse_throughput_warmup_steps({}), 100)
        self.assertEqual(
            Trainer._parse_throughput_warmup_steps(
                {"throughput_warmup_steps": 0}
            ),
            0,
        )
        for value in (-1, 1.5, True, "0", None):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "nonnegative integer"
            ):
                Trainer._parse_throughput_warmup_steps(
                    {"throughput_warmup_steps": value}
                )

        metrics = Metrics(world_size=2, batch_size=2, warmup=0)
        metrics.update(step_time=2.0)
        logged = metrics.get_train_metrics(0, {"loss": 1.0})
        self.assertEqual(logged["samples_per_second"], 2.0)

    def test_save_best_config_requires_boolean(self):
        self.assertTrue(Trainer._parse_save_best({}))
        self.assertTrue(
            Trainer._parse_save_best({"validation": {"save_best": True}})
        )
        self.assertFalse(
            Trainer._parse_save_best({"validation": {"save_best": False}})
        )
        for value in (0, 1, "false", None):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "must be a boolean"
            ):
                Trainer._parse_save_best(
                    {"validation": {"save_best": value}}
                )


if __name__ == "__main__":
    unittest.main()
