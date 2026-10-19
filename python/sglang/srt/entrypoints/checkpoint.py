"""Routes for orchestrators that checkpoint and restore the server process.

An orchestrator polls /checkpoint/status until it returns 200, captures the
process, and calls /checkpoint/resume after a restore or in place. Status and
resume need no API key, so orchestrators without credentials can call them.
"""

from http import HTTPStatus

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from sglang.srt.managers.io_struct import CheckpointReqInput, CheckpointReqOutput
from sglang.srt.utils.auth import AuthLevel, auth_level


def _tokenizer_manager():
    from sglang.srt.entrypoints.http_server import _global_state

    return _global_state.tokenizer_manager


def _response(out: CheckpointReqOutput, status_code: int) -> ORJSONResponse:
    error = out.refusal or out.error
    content = (
        {"state": out.state} if error is None else {"state": out.state, "error": error}
    )
    return ORJSONResponse(content, status_code=status_code)


async def _call(action: str, mode: str = "wait") -> ORJSONResponse:
    out = await _tokenizer_manager().checkpoint(
        CheckpointReqInput(action=action, mode=mode)
    )
    if out.refusal is not None:
        return _response(out, HTTPStatus.CONFLICT)
    if out.state == "failed":
        return _response(out, HTTPStatus.INTERNAL_SERVER_ERROR)
    return _response(out, HTTPStatus.OK)


@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def checkpoint_prepare(mode: str = "wait"):
    """Drain with `mode` (wait, abort or retract) and release the KV cache.
    Returns once started; 409 with nothing changed if it cannot prepare now."""
    return await _call("prepare", mode=mode)


@auth_level(AuthLevel.PUBLIC)
async def checkpoint_resume():
    """Bring the KV cache back and serve, after a restore or in place.
    Orchestrators may deliver this more than once, so a resume in progress or
    done returns 200."""
    return await _call("resume")


@auth_level(AuthLevel.PUBLIC)
async def checkpoint_status():
    # 200 once safe to capture, 503 until then and 500 if that cannot happen.
    out = await _tokenizer_manager().checkpoint(CheckpointReqInput())
    if out.state == "prepared":
        return _response(out, HTTPStatus.OK)
    if out.state == "failed":
        return _response(out, HTTPStatus.INTERNAL_SERVER_ERROR)
    return _response(out, HTTPStatus.SERVICE_UNAVAILABLE)


def add_checkpoint_routes(app: FastAPI) -> None:
    # On the app itself: the auth middleware reads the auth level from the
    # matched endpoint, which an included APIRouter does not expose.
    app.post("/checkpoint/prepare")(checkpoint_prepare)
    app.post("/checkpoint/resume")(checkpoint_resume)
    app.get("/checkpoint/status")(checkpoint_status)
