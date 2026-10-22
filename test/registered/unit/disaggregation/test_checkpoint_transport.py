import os
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation import checkpoint
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _interfaces(eth0_address: str):
    return {
        "lo": [SimpleNamespace(family=socket.AF_INET, address="127.0.0.1")],
        "eth0": [SimpleNamespace(family=socket.AF_INET, address=eth0_address)],
    }


class TestCheckpointTransport(unittest.TestCase):
    @patch.object(checkpoint, "get_context")
    @patch.object(checkpoint, "get_device")
    @patch.object(checkpoint, "maybe_init_shared_mooncake_transfer_engine")
    @patch.object(checkpoint.psutil, "net_if_addrs")
    def test_transport_starts_on_the_restored_pod_address(
        self, net_if_addrs, init_engine, _, get_context
    ):
        """The restored process inherits HOST_IP=$(POD_IP) of the source pod.
        The transport must start on the new pod's address on the same
        interface, or peers connect to the old pod. Queues that a later PD
        role switch rebuilds must start the real transport too."""
        seen = []
        init_engine.side_effect = lambda **_: seen.append(os.environ["HOST_IP"])
        queue = MagicMock()
        queue.start_kv_transport.side_effect = lambda: seen.append(
            os.environ["HOST_IP"]
        )
        transport = checkpoint.PdTransportCheckpoint(queue=queue, releasable=True)
        with patch.dict(os.environ, {"HOST_IP": "10.0.0.5"}):
            net_if_addrs.return_value = _interfaces("10.0.0.5")
            transport.checkpoint_prepare()
            net_if_addrs.return_value = _interfaces("10.0.9.7")
            transport.checkpoint_restore()
        self.assertEqual(seen, ["10.0.9.7", "10.0.9.7"])
        self.assertFalse(transport.releasable)
        override = get_context.return_value.override.call_args
        self.assertFalse(override.kwargs["checkpoint_prepare_on_startup"])


if __name__ == "__main__":
    unittest.main()
