"""Process checkpoint lifecycle of one scheduler.

An external orchestrator (for example ai-dynamo/snapshot) captures the whole
process once it is prepared, restores it elsewhere or keeps it, and resumes
it. The scheduler goes running -> preparing -> prepared -> resuming -> running
any number of times, or ends in failed. Each step is one scheduler operation,
so repeated and concurrent calls from several tokenizer workers or frontends
are no-ops.

Prepare drains with the requested mode, then releases the KV cache and the
resources' external state. Weights stay on the GPU. Resume brings them back
and serves again.
"""

import logging
from enum import Enum
from http import HTTPStatus
from typing import Callable, List, Optional, Protocol, Set

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.managers.io_struct import (
    AbortReq,
    CheckpointReqInput,
    CheckpointReqOutput,
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)

logger = logging.getLogger(__name__)

# How prepare drains: wait for in-flight requests, abort them, or retract them
# into the waiting queue so they continue after an in-place resume.
DRAIN_MODES = ("wait", "abort", "retract")


class CheckpointState(str, Enum):
    RUNNING = "running"
    # Draining, then releasing what a checkpoint must not contain.
    PREPARING = "preparing"
    # Safe to capture.
    PREPARED = "prepared"
    RESUMING = "resuming"
    FAILED = "failed"


class CheckpointResource(Protocol):
    """State outside the KV cache that a checkpoint must not contain, such as
    network endpoints and NIC registrations."""

    def checkpoint_blocker(self) -> Optional[str]:
        """Why this state cannot be released now, or None."""

    def checkpoint_prepare(self) -> None: ...

    def checkpoint_restore(self) -> None: ...


class SchedulerCheckpointLifecycle:
    def __init__(
        self,
        *,
        enable_memory_saver: bool,
        offload_tags: Set[str],
        is_engine_paused: Callable[[], bool],
        is_fully_idle: Callable[..., bool],
        abort_all: Callable[[], None],
        pause_generation: Callable[[PauseGenerationReqInput], None],
        continue_generation: Callable[[ContinueGenerationReqInput], None],
        release_memory_occupation: Callable[[ReleaseMemoryOccupationReqInput], None],
        resume_memory_occupation: Callable[[ResumeMemoryOccupationReqInput], None],
        send_output: Callable[[AbortReq, object], None],
        resources: List[CheckpointResource],
    ) -> None:
        self._enable_memory_saver = enable_memory_saver
        # Live view of the memory regions released through the admin API.
        self._offload_tags = offload_tags
        self._is_engine_paused = is_engine_paused
        self._is_fully_idle = is_fully_idle
        self._abort_all = abort_all
        self._pause_generation = pause_generation
        self._continue_generation = continue_generation
        self._release_memory_occupation = release_memory_occupation
        self._resume_memory_occupation = resume_memory_occupation
        self._send_output = send_output
        self._resources = resources
        self.state = CheckpointState.RUNNING
        self._error: Optional[str] = None
        self._mode: Optional[str] = None

    @property
    def rejects_new_requests(self) -> bool:
        return self.state != CheckpointState.RUNNING

    def handle(self, recv_req: CheckpointReqInput) -> CheckpointReqOutput:
        refusal = None
        if recv_req.action == "prepare":
            refusal = self._prepare(recv_req.mode)
        elif recv_req.action == "resume":
            refusal = self._resume()
        return CheckpointReqOutput(
            state=self.state.value, error=self._error, refusal=refusal
        )

    def check_pending(self) -> None:
        """Finish preparing once the drain is complete."""
        if self.state != CheckpointState.PREPARING or not self._is_fully_idle(
            ignore_waiting=self._mode == "retract"
        ):
            return
        try:
            self._release_memory_occupation(
                ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            )
            for resource in reversed(self._resources):
                resource.checkpoint_prepare()
        except Exception as e:
            self._fail(e)
        else:
            self.state = CheckpointState.PREPARED
            logger.info("Prepared for a process checkpoint")

    def reject(self, recv_req) -> None:
        """Answer a new request that arrives while the scheduler does not serve."""
        self._send_output(
            AbortReq(
                rid=recv_req.rid,
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": f"The server is {self.state.value} for a process "
                    "checkpoint; retry later.",
                },
            ),
            recv_req,
        )

    def _prepare(self, mode: str) -> Optional[str]:
        """Start draining, or return why the scheduler cannot prepare now
        with nothing changed."""
        if self.state in (CheckpointState.PREPARING, CheckpointState.PREPARED):
            return None
        if self.state != CheckpointState.RUNNING:
            return f"Cannot prepare while the checkpoint state is {self.state.value}"
        if mode not in DRAIN_MODES:
            return f"Unknown drain mode {mode!r}; expected one of {DRAIN_MODES}"
        if not self._enable_memory_saver:
            return "Preparing for a checkpoint requires --enable-memory-saver"
        if self._offload_tags:
            return f"Memory is already released: {sorted(self._offload_tags)}"
        if self._is_engine_paused():
            return "Preparing for a checkpoint requires unpaused generation"
        for resource in self._resources:
            if (blocker := resource.checkpoint_blocker()) is not None:
                return blocker

        logger.info("Preparing for a process checkpoint (drain mode %s)", mode)
        if mode == "abort":
            self._abort_all()
        elif mode == "retract":
            # Before PREPARING: retracted requests re-enter the waiting queue.
            self._pause_generation(PauseGenerationReqInput(mode="retract"))
        self._mode = mode
        self.state = CheckpointState.PREPARING
        self.check_pending()
        return None

    def _resume(self) -> Optional[str]:
        if self.state in (CheckpointState.RUNNING, CheckpointState.RESUMING):
            return None
        if self.state != CheckpointState.PREPARED:
            return f"Cannot resume while the checkpoint state is {self.state.value}"
        logger.info("Resuming from a process checkpoint")
        self.state = CheckpointState.RESUMING
        try:
            self._resume_memory_occupation(
                ResumeMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            )
            # After the KV cache, which transports register.
            for resource in self._resources:
                resource.checkpoint_restore()
            self.state = CheckpointState.RUNNING
            if self._is_engine_paused():
                self._continue_generation(ContinueGenerationReqInput())
        except Exception as e:
            self._fail(e)
        return None

    def _fail(self, error: Exception) -> None:
        logger.exception("Process checkpoint lifecycle failed")
        self.state = CheckpointState.FAILED
        self._error = str(error)
