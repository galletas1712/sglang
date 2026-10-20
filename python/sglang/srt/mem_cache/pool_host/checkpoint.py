"""HiCache host pools across a process checkpoint.

The default pools are pinned shared anonymous mappings, often hundreds of GB,
that a checkpoint would carry in full although preparing flushes the cache
they hold. Prepare unpins them and frees their pages, keeping the mappings and
addresses; resume pins the same ranges again, which faults the pages back in
as zeros.
"""

import ctypes
import mmap
from typing import List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host.common import (
    _CUDA_HOST_REGISTERED_RANGES_ATTR,
    HostTensorAllocator,
    _cuda_host_unregister,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

_MADV_REMOVE = 9
_LIBC = ctypes.CDLL(None, use_errno=True)


class HostPoolCheckpoint:
    def __init__(
        self,
        *,
        tree_cache: object,
        storage_backend: Optional[str],
    ) -> None:
        self._host_pool_group = (
            tree_cache.host_pool_group
            if isinstance(tree_cache, UnifiedRadixCache)
            else None
        )
        self._storage_backend = storage_backend
        # (buffer, registered (ptr, size) ranges) of the released buffers.
        self._released: List[Tuple[torch.Tensor, List[Tuple[int, int]]]] = []

    def checkpoint_blocker(self) -> Optional[str]:
        if self._storage_backend is not None:
            return (
                f"The HiCache storage backend {self._storage_backend!r} cannot be "
                "released for a checkpoint"
            )
        if self._host_pool_group is None or envs.SGLANG_HUGEPAGE_SIZE.get():
            return "These HiCache host pools cannot be released for a checkpoint"
        for entry in self._host_pool_group.entries:
            pool = entry.host_pool
            # Other allocators own their backing (shm files, registered
            # transfer buffers).
            if type(pool.allocator) is not HostTensorAllocator or not pool.pin_memory:
                return (
                    f"HiCache host pool {entry.name} uses an allocator that cannot "
                    "be released for a checkpoint"
                )
        return None

    def checkpoint_prepare(self) -> None:
        # Copies to and from the pools run on side streams.
        torch.cuda.synchronize()
        for buffer in self._buffers():
            ranges = list(getattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR))
            _cuda_host_unregister(buffer)
            nbytes = buffer.numel() * buffer.element_size()
            length = -(-nbytes // mmap.PAGESIZE) * mmap.PAGESIZE
            if _LIBC.madvise(
                ctypes.c_void_p(buffer.data_ptr()),
                ctypes.c_size_t(length),
                _MADV_REMOVE,
            ):
                raise OSError(ctypes.get_errno(), "madvise(MADV_REMOVE) failed")
            self._released.append((buffer, ranges))

    def checkpoint_restore(self) -> None:
        cudart = torch.cuda.cudart()
        for buffer, ranges in self._released:
            for ptr, size in ranges:
                rc = int(cudart.cudaHostRegister(ptr, size, 0))
                if rc != 0:
                    raise RuntimeError(
                        f"cudaHostRegister failed (rc={rc}) re-pinning a HiCache "
                        "host pool"
                    )
            setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, ranges)
        self._released = []

    def _buffers(self) -> List[torch.Tensor]:
        buffers = []
        for entry in self._host_pool_group.entries:
            kv_buffer = entry.host_pool.kv_buffer
            buffers += (
                kv_buffer if isinstance(kv_buffer, (list, tuple)) else [kv_buffer]
            )
        return buffers
