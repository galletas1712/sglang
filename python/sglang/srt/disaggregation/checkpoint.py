"""Tear down and rebuild the PD transfer transport around a process checkpoint.

Transfer-engine threads, sockets and memory registrations do not survive a
process checkpoint, and the restored process runs at a new address. Prepare
tears the transport down after the KV cache is released. Resume rebuilds it
after the KV cache is back, so the KV manager, the shared transfer engine and
the bootstrap registration all carry the current address. Surviving peers
reach the restored replica through its new address and drop the old one when
its heartbeats fail.

A restored process cannot register GPU memory through nvidia-peermem, so
Mooncake must run with WITH_NVIDIA_PEERMEM=0 (DMA-BUF registration) from
process start; Mooncake reads it once per process.
"""

from __future__ import annotations

import gc
import logging
import os
import socket
from typing import TYPE_CHECKING, Union

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    destroy_mooncake_transfer_engine,
    maybe_init_shared_mooncake_transfer_engine,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    CheckpointTransportReqInput,
    CheckpointTransportReqOutput,
)
from sglang.srt.runtime_context import get_device, get_disagg, get_exec, get_memory

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodePreallocQueue
    from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def handle_checkpoint_transport(
    scheduler: Scheduler, recv_req: CheckpointTransportReqInput
) -> CheckpointTransportReqOutput:
    queue = _transport_queue(scheduler)
    try:
        if recv_req.rebuild:
            _rebuild(queue)
        else:
            _teardown(scheduler, queue)
    except Exception as e:
        logger.exception("Checkpoint transport %s failed", _action(recv_req))
        return CheckpointTransportReqOutput(success=False, message=str(e))
    return CheckpointTransportReqOutput(success=True)


def _action(recv_req: CheckpointTransportReqInput) -> str:
    return "rebuild" if recv_req.rebuild else "teardown"


def _transport_queue(
    scheduler: Scheduler,
) -> Union[PrefillBootstrapQueue, DecodePreallocQueue]:
    if scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
        return scheduler.disagg_prefill_bootstrap_queue
    if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
        return scheduler.disagg_decode_prealloc_queue
    raise ValueError("instance is not running in PD disaggregation mode")


def _teardown(
    scheduler: Scheduler, queue: Union[PrefillBootstrapQueue, DecodePreallocQueue]
) -> None:
    if queue.kv_manager is None:
        return
    if (
        not queue.kv_manager.supports_role_switch
        or envs.SGLANG_DISAGG_STAGING_BUFFER.get()
    ):
        raise ValueError(
            f"transfer backend {get_disagg().disaggregation_transfer_backend!r} "
            "cannot be torn down for a checkpoint"
        )
    if _mooncake_engine_has_other_users():
        raise ValueError(
            "the mooncake transfer engine is shared with HiCache or elastic EP, "
            "which cannot rebuild it after restore"
        )
    if not scheduler.is_fully_idle():
        raise RuntimeError("requests are still in flight")
    queue.teardown_transport()
    destroy_mooncake_transfer_engine()
    # The native engine releases its devices only once the manager's reference
    # cycles are collected.
    gc.collect()
    # A process checkpoint cannot capture open RDMA devices.
    if _open_rdma_device_count() > 0:
        raise RuntimeError("RDMA devices are still open after transport teardown")


def _rebuild(queue: Union[PrefillBootstrapQueue, DecodePreallocQueue]) -> None:
    if queue.kv_manager is not None:
        return
    _forget_stale_host_ip()
    maybe_init_shared_mooncake_transfer_engine(gpu_id=get_device().gpu_id)
    queue.rebuild_transport()


def _mooncake_engine_has_other_users() -> bool:
    return (
        get_memory().enable_hierarchical_cache
        and get_memory().hicache_storage_backend == "mooncake"
        and envs.SGLANG_HICACHE_MOONCAKE_REUSE_TE.get()
    ) or get_exec().moe.elastic_ep_backend is not None


def _open_rdma_device_count() -> int:
    count = 0
    for fd in os.listdir("/proc/self/fd"):
        try:
            count += os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/infiniband/")
        except OSError:
            pass
    return count


def _forget_stale_host_ip() -> None:
    # A restored process inherits the captured environment, so an address set
    # there may belong to the old pod; detect the current one instead.
    for name in ("SGLANG_HOST_IP", "HOST_IP"):
        address = os.environ.get(name)
        if address and not _is_local_address(address):
            logger.info("Ignoring %s=%s, which is not a local address", name, address)
            del os.environ[name]


def _is_local_address(address: str) -> bool:
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    with socket.socket(family, socket.SOCK_DGRAM) as sock:
        try:
            sock.bind((address, 0))
        except OSError:
            return False
    return True
