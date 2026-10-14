import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation import checkpoint
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import CheckpointTransportReqInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeQueue:
    def __init__(self):
        self.kv_manager = SimpleNamespace(supports_role_switch=True)
        self.calls = []

    def teardown_transport(self):
        self.calls.append("teardown")
        self.kv_manager = None

    def rebuild_transport(self):
        self.calls.append("rebuild")
        self.kv_manager = SimpleNamespace(supports_role_switch=True)


class TestCheckpointTransport(CustomTestCase):
    def setUp(self):
        for name, value in (
            ("_mooncake_engine_has_other_users", lambda: False),
            ("destroy_mooncake_transfer_engine", lambda: None),
            ("maybe_init_shared_mooncake_transfer_engine", lambda gpu_id: None),
            ("get_device", lambda: SimpleNamespace(gpu_id=0)),
            ("_open_rdma_device_count", lambda: 0),
        ):
            patcher = patch.object(checkpoint, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_repeated_requests_tear_down_and_rebuild_once(self):
        """Orchestrators retry prepare and resume; a second teardown or rebuild
        must not run on an already torn-down or rebuilt transport."""
        queue = _FakeQueue()
        scheduler = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.PREFILL,
            disagg_prefill_bootstrap_queue=queue,
            is_fully_idle=lambda: True,
        )
        for rebuild in (False, False, True, True):
            reply = checkpoint.handle_checkpoint_transport(
                scheduler, CheckpointTransportReqInput(rebuild=rebuild)
            )
            self.assertTrue(reply.success, reply.message)
        self.assertEqual(queue.calls, ["teardown", "rebuild"])

    def test_teardown_refuses_in_flight_requests(self):
        queue = _FakeQueue()
        scheduler = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.DECODE,
            disagg_decode_prealloc_queue=queue,
            is_fully_idle=lambda: False,
        )
        reply = checkpoint.handle_checkpoint_transport(
            scheduler, CheckpointTransportReqInput(rebuild=False)
        )
        self.assertFalse(reply.success)
        self.assertEqual(queue.calls, [])

    def test_restore_forgets_the_old_pod_address(self):
        """A restored process inherits the old pod's SGLANG_HOST_IP; the
        rebuilt transport must advertise the current address instead."""
        env = {"SGLANG_HOST_IP": "192.0.2.1", "HOST_IP": "127.0.0.1"}
        with patch.dict(os.environ, env):
            checkpoint._forget_stale_host_ip()
            self.assertNotIn("SGLANG_HOST_IP", os.environ)
            self.assertEqual(os.environ["HOST_IP"], "127.0.0.1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
