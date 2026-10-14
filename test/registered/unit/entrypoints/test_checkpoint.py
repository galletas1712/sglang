import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints import checkpoint, http_server
from sglang.srt.managers.tokenizer_manager import ServerStatus
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeTokenizerManager:
    def __init__(self):
        self.calls = []

    async def pause_generation(self, obj):
        await asyncio.sleep(0)
        self.calls.append(("pause", obj.mode))

    async def release_memory_occupation(self, obj, request=None):
        self.calls.append(("release", obj.tags))

    async def resume_memory_occupation(self, obj, request=None):
        self.calls.append(("resume", obj.tags))

    async def continue_generation(self, obj):
        self.calls.append(("continue",))


def _exec_config(*, enable_memory_saver: bool):
    return SimpleNamespace(
        features=SimpleNamespace(enable_memory_saver=enable_memory_saver)
    )


class TestCheckpointLifecycle(CustomTestCase):
    def setUp(self):
        patcher = patch.object(
            checkpoint, "get_exec", return_value=_exec_config(enable_memory_saver=True)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_repeated_and_concurrent_calls_run_each_step_once(self):
        """Orchestrators retry prepare and resume, so a repeat must not pause,
        release or resume the KV cache a second time; SGLang rejects a second
        resume of the same memory tag."""

        async def scenario():
            lifecycle = checkpoint.CheckpointLifecycle()
            tokenizer_manager = _FakeTokenizerManager()
            await asyncio.gather(
                lifecycle.prepare(tokenizer_manager),
                lifecycle.prepare(tokenizer_manager),
            )
            await lifecycle.prepare(tokenizer_manager)
            self.assertTrue(lifecycle.prepared)
            self.assertEqual(
                tokenizer_manager.calls,
                [("pause", "retract"), ("release", ["kv_cache"])],
            )

            for _ in range(2):
                await lifecycle.resume(tokenizer_manager)
            self.assertFalse(lifecycle.prepared)
            self.assertEqual(
                tokenizer_manager.calls[2:],
                [("resume", ["kv_cache"]), ("continue",)],
            )

        asyncio.run(scenario())

    def test_prepare_requires_memory_saver(self):
        tokenizer_manager = _FakeTokenizerManager()
        with patch.object(
            checkpoint,
            "get_exec",
            return_value=_exec_config(enable_memory_saver=False),
        ):
            with self.assertRaises(ValueError):
                asyncio.run(checkpoint.CheckpointLifecycle().prepare(tokenizer_manager))
        self.assertEqual(tokenizer_manager.calls, [])


class TestHealthWhilePrepared(CustomTestCase):
    def test_health_answers_without_generating(self):
        """A generating probe waits behind the pause, times out and marks the
        server unhealthy, so a liveness probe would restart a prepared server."""
        prior_state = http_server.get_global_state()
        # No generate_request: a probe that tries to generate raises.
        http_server.set_global_state(
            SimpleNamespace(
                tokenizer_manager=SimpleNamespace(
                    gracefully_exit=False, server_status=ServerStatus.Up
                )
            )
        )
        request = SimpleNamespace(url=SimpleNamespace(path="/health"))
        try:
            with patch.object(checkpoint.checkpoint_lifecycle, "prepared", True):
                response = asyncio.run(http_server.health_generate(request))
        finally:
            http_server._global_state = prior_state
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
