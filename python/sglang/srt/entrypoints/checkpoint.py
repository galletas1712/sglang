"""Routes for orchestrators that checkpoint and restore the server process.

Prepare pauses generation and releases the KV cache; weights stay on the GPU.
The orchestrator polls status until it returns 200, captures the process,
restores it elsewhere and calls resume.
"""

import asyncio
from http import HTTPStatus

from fastapi import APIRouter
from fastapi.responses import ORJSONResponse, Response

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    CheckpointTransportReqInput,
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils.auth import AuthLevel, auth_level

router = APIRouter()


class CheckpointLifecycle:
    """Serialize prepare and resume; repeated calls in the target state are no-ops."""

    def __init__(self):
        self.prepared = False
        self._lock = asyncio.Lock()

    async def prepare(self, tokenizer_manager) -> None:
        async with self._lock:
            if self.prepared:
                return
            if not get_exec().features.enable_memory_saver:
                raise ValueError("Checkpoint prepare requires --enable-memory-saver")
            # Retract also clears the overlap scheduler's last batch,
            # which release_memory_occupation requires to be empty.
            await tokenizer_manager.pause_generation(
                PauseGenerationReqInput(mode="retract")
            )
            await tokenizer_manager.release_memory_occupation(
                ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            )
            if _is_disaggregated(tokenizer_manager):
                await tokenizer_manager.checkpoint_transport(
                    CheckpointTransportReqInput(rebuild=False)
                )
            self.prepared = True

    async def resume(self, tokenizer_manager) -> None:
        async with self._lock:
            if not self.prepared:
                return
            await tokenizer_manager.resume_memory_occupation(
                ResumeMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            )
            # The transport registers the KV cache, so it is rebuilt only after
            # the cache is resident again.
            if _is_disaggregated(tokenizer_manager):
                await tokenizer_manager.checkpoint_transport(
                    CheckpointTransportReqInput(rebuild=True)
                )
            await tokenizer_manager.continue_generation(ContinueGenerationReqInput())
            self.prepared = False


checkpoint_lifecycle = CheckpointLifecycle()


def _is_disaggregated(tokenizer_manager) -> bool:
    return tokenizer_manager.disaggregation_mode != DisaggregationMode.NULL


def _tokenizer_manager():
    from sglang.srt.entrypoints.http_server import _global_state

    return _global_state.tokenizer_manager


@router.post("/checkpoint/prepare")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def checkpoint_prepare():
    """Pause generation and release the KV cache; returns once prepared."""
    try:
        await checkpoint_lifecycle.prepare(_tokenizer_manager())
    except ValueError as e:
        return ORJSONResponse({"error": str(e)}, status_code=HTTPStatus.BAD_REQUEST)
    return Response(status_code=HTTPStatus.OK)


@router.post("/checkpoint/resume")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def checkpoint_resume():
    """Restore the KV cache and continue generation; returns once resumed."""
    await checkpoint_lifecycle.resume(_tokenizer_manager())
    return Response(status_code=HTTPStatus.OK)


@router.get("/checkpoint/status")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def checkpoint_status():
    # Orchestrators poll this route to learn when the process is safe to capture.
    prepared = checkpoint_lifecycle.prepared
    return ORJSONResponse(
        {"prepared": prepared},
        status_code=HTTPStatus.OK if prepared else HTTPStatus.SERVICE_UNAVAILABLE,
    )
