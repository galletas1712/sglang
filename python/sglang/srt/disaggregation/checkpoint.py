"""The PD KV transfer transport across a process checkpoint.

Transfer engines keep network state for the life of the process (UCX route
caches, RDMA registrations), so with --checkpoint-prepare-on-startup the queues
start on the fake transport, which serves only warmup, and the first checkpoint
restore creates the real one. Once created it cannot be released.

A restored process keeps the captured environment, so SGLANG_HOST_IP or
HOST_IP still names the source pod's address. Prepare records which interface
carries it, and restore takes that interface's address in the new pod.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import psutil

from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    maybe_init_shared_mooncake_transfer_engine,
)
from sglang.srt.runtime_context import get_context, get_device

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodePreallocQueue
    from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue

logger = logging.getLogger(__name__)

_HOST_IP_ENVS = ("SGLANG_HOST_IP", "HOST_IP")


class PdTransportCheckpoint:
    def __init__(
        self,
        *,
        queue: Union[PrefillBootstrapQueue, DecodePreallocQueue],
        releasable: bool,
    ) -> None:
        self._queue = queue
        # False once the real transport exists.
        self.releasable = releasable
        # Env name -> (interface, address family) that carried its address.
        self._host_interfaces: Dict[str, Tuple[str, int]] = {}

    def checkpoint_prepare(self) -> None:
        self._host_interfaces = {
            name: interface
            for name in _HOST_IP_ENVS
            if (address := os.environ.get(name))
            and (interface := _interface_of(address)) is not None
        }

    def checkpoint_restore(self) -> None:
        for name, (interface, family) in self._host_interfaces.items():
            address = _address_of(interface, family)
            logger.info("Restored %s=%s from interface %s", name, address, interface)
            os.environ[name] = address
        maybe_init_shared_mooncake_transfer_engine(gpu_id=get_device().gpu_id)
        self._queue.start_kv_transport()
        self.releasable = False
        # Queues rebuilt later, by a PD role switch, start the real transport.
        get_context().override(
            "checkpoint.restore", checkpoint_prepare_on_startup=False
        )


def _interface_of(address: str) -> Optional[Tuple[str, int]]:
    # Names such as DNS names are kept as they are.
    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.address.split("%")[0] == address:
                return interface, addr.family
    return None


def _address_of(interface: str, family: int) -> str:
    for addr in psutil.net_if_addrs().get(interface, []):
        address = addr.address.split("%")[0]
        if addr.family == family and not ipaddress.ip_address(address).is_link_local:
            return address
    raise RuntimeError(
        f"Interface {interface} has no {socket.AddressFamily(family).name} "
        "address after the restore"
    )
