import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.io_struct import CheckpointReqInput
from sglang.srt.managers.scheduler_components.checkpoint_lifecycle import (
    SchedulerCheckpointLifecycle,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _lifecycle(**overrides) -> SchedulerCheckpointLifecycle:
    kwargs = dict(
        enable_memory_saver=True,
        offload_tags=set(),
        is_engine_paused=MagicMock(return_value=False),
        is_fully_idle=MagicMock(return_value=False),
        abort_all=MagicMock(),
        pause_generation=MagicMock(),
        continue_generation=MagicMock(),
        release_memory_occupation=MagicMock(),
        resume_memory_occupation=MagicMock(),
        send_output=MagicMock(),
    )
    kwargs.update(overrides)
    return SchedulerCheckpointLifecycle(**kwargs)


def _call(lifecycle, action, mode="wait"):
    return lifecycle.handle(CheckpointReqInput(action=action, mode=mode))


class TestSchedulerCheckpointLifecycle(unittest.TestCase):
    def test_releases_only_after_drain_and_repeats_are_no_ops(self):
        """Releasing the KV cache under in-flight requests asserts in the
        scheduler, so prepare must wait for the drain. Orchestrators redeliver
        prepare and resume, and a cycle may run again in place."""
        lifecycle = _lifecycle()
        release = lifecycle._release_memory_occupation
        resume = lifecycle._resume_memory_occupation

        for _ in range(2):
            lifecycle._is_fully_idle.return_value = False
            self.assertIsNone(_call(lifecycle, "prepare").refusal)
            self.assertIsNone(_call(lifecycle, "prepare").refusal)
            lifecycle.check_pending()
            self.assertEqual(_call(lifecycle, "state").state, "preparing")
            self.assertTrue(lifecycle.rejects_new_requests)
            release.assert_not_called()

            lifecycle._is_fully_idle.return_value = True
            lifecycle.check_pending()
            self.assertEqual(_call(lifecycle, "prepare").state, "prepared")
            release.assert_called_once()

            self.assertEqual(_call(lifecycle, "resume").state, "running")
            self.assertIsNone(_call(lifecycle, "resume").refusal)
            resume.assert_called_once()
            self.assertFalse(lifecycle.rejects_new_requests)
            release.reset_mock()
            resume.reset_mock()

    def test_refusal_changes_nothing_and_failure_latches(self):
        """A refused prepare must leave the server serving. A failed release
        must stop serving for good: resuming would bring back a half-released
        server."""
        lifecycle = _lifecycle(offload_tags={"kv_cache"})
        self.assertIn("already released", _call(lifecycle, "prepare").refusal)
        self.assertEqual(lifecycle.state, "running")
        lifecycle._abort_all.assert_not_called()

        lifecycle = _lifecycle(
            is_fully_idle=MagicMock(return_value=True),
            release_memory_occupation=MagicMock(side_effect=RuntimeError("oom")),
        )
        out = _call(lifecycle, "prepare", mode="abort")
        self.assertEqual((out.state, out.error), ("failed", "oom"))
        self.assertIsNotNone(_call(lifecycle, "resume").refusal)
        self.assertIsNotNone(_call(lifecycle, "prepare").refusal)
        self.assertTrue(lifecycle.rejects_new_requests)


if __name__ == "__main__":
    unittest.main()
