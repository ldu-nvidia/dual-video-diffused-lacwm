import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from robot_wm.utils.trainer import Trainer


PROJECT_DIR = (
    Path(__file__).resolve().parents[2] / "projects" / "latent_action_models"
)
sys.path.insert(0, str(PROJECT_DIR))
try:
    from projects.latent_action_models import train as train_entrypoint
finally:
    sys.path.remove(str(PROJECT_DIR))


class _Iterator:
    def __init__(self, *, error=None):
        self.error = error
        self.shutdown_calls = 0

    def _shutdown_workers(self):
        self.shutdown_calls += 1
        if self.error is not None:
            raise self.error


class _Loader:
    def __init__(self, iterator):
        self._iterator = iterator
        self.state_dict = mock.Mock(
            side_effect=AssertionError("shutdown must not capture loader state")
        )
        self.load_state_dict = mock.Mock(
            side_effect=AssertionError("shutdown must not restore loader state")
        )
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        raise AssertionError("shutdown must not create a loader iterator")


def _trainer_with_loaders():
    train_iterator = _Iterator()
    validation_iterator = _Iterator()
    validation_loader_iterator = _Iterator()
    visualization_iterator = _Iterator()

    trainer = Trainer.__new__(Trainer)
    trainer.data_loader = _Loader(train_iterator)
    trainer.val_data_loaders = [_Loader(validation_loader_iterator)]
    trainer.viz_data_loaders = [_Loader(visualization_iterator)]
    trainer._data_loader_iter = train_iterator
    trainer._val_data_loader_iters = [validation_iterator]
    trainer._viz_data_loader_iters = [visualization_iterator]
    trainer._data_loaders_shutdown = False
    return trainer, (
        train_iterator,
        validation_iterator,
        validation_loader_iterator,
        visualization_iterator,
    )


class TrainerDataLoaderShutdownTest(unittest.TestCase):
    def test_real_spawn_workers_are_joined(self):
        loader = StatefulDataLoader(
            torch.utils.data.TensorDataset(torch.arange(8)),
            batch_size=2,
            num_workers=1,
            persistent_workers=True,
            pin_memory=False,
            multiprocessing_context="spawn",
        )
        trainer = Trainer.__new__(Trainer)
        trainer.data_loader = loader
        trainer.val_data_loaders = []
        trainer.viz_data_loaders = []
        trainer._data_loader_iter = iter(loader)
        trainer._val_data_loader_iters = []
        trainer._viz_data_loader_iters = []
        trainer._data_loaders_shutdown = False
        self.addCleanup(trainer.shutdown_data_loaders)

        next(trainer._data_loader_iter)
        workers = list(trainer._data_loader_iter._workers)
        self.assertTrue(all(worker.is_alive() for worker in workers))

        trainer.shutdown_data_loaders()
        trainer.shutdown_data_loaders()

        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertTrue(all(worker.exitcode == 0 for worker in workers))

    def test_shutdown_closes_all_unique_iterators_once_without_touching_state(self):
        trainer, iterators = _trainer_with_loaders()
        loaders = [
            trainer.data_loader,
            *trainer.val_data_loaders,
            *trainer.viz_data_loaders,
        ]

        trainer.shutdown_data_loaders()
        trainer.shutdown_data_loaders()

        self.assertTrue(trainer._data_loaders_shutdown)
        self.assertIsNone(trainer._data_loader_iter)
        self.assertIsNone(trainer._val_data_loader_iters)
        self.assertIsNone(trainer._viz_data_loader_iters)
        for iterator in iterators:
            self.assertEqual(iterator.shutdown_calls, 1)
        for loader in loaders:
            self.assertIsNone(loader._iterator)
            self.assertEqual(loader.iter_calls, 0)
            loader.state_dict.assert_not_called()
            loader.load_state_dict.assert_not_called()

    def test_shutdown_attempts_every_iterator_and_remains_idempotent_on_error(self):
        trainer, iterators = _trainer_with_loaders()
        iterators[0].error = ValueError("train worker failed")

        with self.assertRaisesRegex(ValueError, "train worker failed"):
            trainer.shutdown_data_loaders()

        self.assertEqual([iterator.shutdown_calls for iterator in iterators], [1] * 4)
        self.assertIsNone(trainer._data_loader_iter)
        self.assertIsNone(trainer.data_loader._iterator)

        # A cleanup failure must not make a second teardown call repeat private
        # worker-shutdown operations against partially closed queues.
        trainer.shutdown_data_loaders()
        self.assertEqual([iterator.shutdown_calls for iterator in iterators], [1] * 4)

    def test_shutdown_before_start_does_not_construct_iterators(self):
        loader = _Loader(None)
        trainer = Trainer.__new__(Trainer)
        trainer.data_loader = loader
        trainer.val_data_loaders = []
        trainer.viz_data_loaders = []
        trainer._data_loader_iter = None
        trainer._val_data_loader_iters = None
        trainer._viz_data_loader_iters = None
        trainer._data_loaders_shutdown = False

        trainer.shutdown_data_loaders()

        self.assertEqual(loader.iter_calls, 0)
        loader.state_dict.assert_not_called()
        loader.load_state_dict.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "cannot be restarted"):
            trainer.start_data_loaders()


class TrainEntrypointTeardownTest(unittest.TestCase):
    def test_loader_failure_still_finalizes_wandb_and_distributed_in_order(self):
        events = []

        def fail_loader_shutdown():
            events.append("loaders")
            raise RuntimeError("worker shutdown failed")

        trainer = SimpleNamespace(
            shutdown_data_loaders=fail_loader_shutdown,
            finalize_wandb=lambda: events.append("wandb"),
        )

        with mock.patch.object(
            train_entrypoint.dist,
            "is_initialized",
            side_effect=lambda: events.append("is_initialized") or True,
        ), mock.patch.object(
            train_entrypoint.dist,
            "destroy_process_group",
            side_effect=lambda: events.append("process_group"),
        ):
            with self.assertRaisesRegex(RuntimeError, "worker shutdown failed"):
                train_entrypoint._teardown(trainer)

        self.assertEqual(
            events,
            ["loaders", "wandb", "is_initialized", "process_group"],
        )

    def test_all_cleanup_phases_are_attempted_and_first_failure_is_preserved(self):
        events = []

        def fail(name, error):
            def cleanup():
                events.append(name)
                raise error

            return cleanup

        trainer = SimpleNamespace(
            shutdown_data_loaders=fail("loaders", ValueError("loader failed")),
            finalize_wandb=fail("wandb", KeyError("wandb failed")),
        )

        with mock.patch.object(
            train_entrypoint.dist,
            "is_initialized",
            side_effect=lambda: events.append("is_initialized") or True,
        ), mock.patch.object(
            train_entrypoint.dist,
            "destroy_process_group",
            side_effect=fail("process_group", OSError("process group failed")),
        ), mock.patch.object(train_entrypoint.logger, "error"):
            with self.assertRaisesRegex(ValueError, "loader failed"):
                train_entrypoint._teardown(trainer)

        self.assertEqual(
            events,
            ["loaders", "wandb", "is_initialized", "process_group"],
        )

    def test_wandb_failure_still_destroys_distributed_after_loader_shutdown(self):
        events = []

        def fail_wandb():
            events.append("wandb")
            raise RuntimeError("wandb finish failed")

        trainer = SimpleNamespace(
            shutdown_data_loaders=lambda: events.append("loaders"),
            finalize_wandb=fail_wandb,
        )

        with mock.patch.object(
            train_entrypoint.dist,
            "is_initialized",
            side_effect=lambda: events.append("is_initialized") or True,
        ), mock.patch.object(
            train_entrypoint.dist,
            "destroy_process_group",
            side_effect=lambda: events.append("process_group"),
        ):
            with self.assertRaisesRegex(RuntimeError, "wandb finish failed"):
                train_entrypoint._teardown(trainer)

        self.assertEqual(
            events,
            ["loaders", "wandb", "is_initialized", "process_group"],
        )

    def test_partial_setup_failure_tears_down_constructed_trainer(self):
        events = []

        def fail_start():
            events.append("start_loaders")
            raise RuntimeError("validation iterator failed")

        trainer = SimpleNamespace(
            resumed=False,
            initialize_wandb=lambda _cfg: events.append("initialize_wandb"),
            start_data_loaders=fail_start,
            shutdown_data_loaders=lambda: events.append("shutdown_loaders"),
            finalize_wandb=lambda: events.append("finalize_wandb"),
        )
        cfg = SimpleNamespace(seed=17, trainer=object())

        with mock.patch.object(
            train_entrypoint.dist,
            "init_process_group",
            side_effect=lambda: events.append("init_process_group"),
        ), mock.patch.object(
            train_entrypoint.dist, "get_global_rank", return_value=0
        ), mock.patch.object(
            train_entrypoint.dist, "is_initialized", return_value=True
        ), mock.patch.object(
            train_entrypoint.dist,
            "destroy_process_group",
            side_effect=lambda: events.append("destroy_process_group"),
        ), mock.patch.object(
            train_entrypoint, "_seed_all"
        ), mock.patch.object(
            train_entrypoint.hydra.utils, "instantiate", return_value=trainer
        ):
            with self.assertRaisesRegex(RuntimeError, "validation iterator failed"):
                train_entrypoint._setup(cfg)

        self.assertEqual(
            events,
            [
                "init_process_group",
                "initialize_wandb",
                "start_loaders",
                "shutdown_loaders",
                "finalize_wandb",
                "destroy_process_group",
            ],
        )

    def test_primary_training_failure_is_not_masked_by_teardown_failure(self):
        trainer = mock.Mock()
        trainer.train.side_effect = RuntimeError("training failed")

        with mock.patch.object(
            train_entrypoint.torch.multiprocessing, "set_start_method"
        ), mock.patch.object(
            train_entrypoint, "_setup", return_value=trainer
        ), mock.patch.object(
            train_entrypoint,
            "_teardown",
            side_effect=RuntimeError("teardown failed"),
        ), mock.patch.object(train_entrypoint.logger, "error"):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                train_entrypoint.main.__wrapped__(SimpleNamespace(debug=False))


if __name__ == "__main__":
    unittest.main()
