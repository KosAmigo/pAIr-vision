"""POST /v1/sessions/{key}/start | mode | end (계약 §3-3)."""

from __future__ import annotations

import time
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from pair_vision.api.core import ApiError, Core, get_core, require_auth
from pair_vision.session.frames import is_valid_key
from pair_vision.session.registry import DrainingError, OverCapacityError, UnknownProfileError

router = APIRouter(prefix="/v1/sessions", dependencies=[Depends(require_auth)])


class StartRequest(BaseModel):
    mode: Literal["check", "exam"]
    startedAtMs: int | None = None
    rulesProfile: str = "default"
    recalibrate: bool = False


class ModeRequest(BaseModel):
    mode: Literal["exam"]
    startedAtMs: int | None = None


def _check_key(key: str) -> None:
    if not is_valid_key(key):
        raise ApiError(400, "BAD_REQUEST", "Invalid session key format.")


@router.post("/{key}/start")
def start(key: str, body: StartRequest, core: Core = Depends(get_core)) -> dict:
    _check_key(key)
    started = body.startedAtMs if body.startedAtMs is not None else int(time.time() * 1000)
    try:
        result = core.registry.start(key, body.mode, started, body.rulesProfile, body.recalibrate)
    except UnknownProfileError:
        raise ApiError(400, "BAD_REQUEST", "Unknown rulesProfile.") from None
    except DrainingError:
        raise ApiError(503, "DRAINING", "Server is draining.") from None
    except OverCapacityError:
        raise ApiError(503, "OVER_CAPACITY", "No free session slot.", {"Retry-After": "5"}) from None
    core.metrics.sessions.set(len(core.registry))
    return {
        "accepted": True,
        "resumed": result.resumed,
        "sessions": len(core.registry),
        "maxSessions": core.registry.max_sessions,
    }


@router.post("/{key}/mode")
def set_mode(key: str, body: ModeRequest, core: Core = Depends(get_core)) -> dict:
    _check_key(key)
    session = core.registry.get(key)
    if session is None:
        raise ApiError(404, "UNKNOWN_SESSION", "Unknown session; call start first.")
    if session.mode != "exam":
        session.switch_to_exam(body.startedAtMs)
    return {"accepted": True, "mode": session.mode}


@router.post("/{key}/end")
def end(key: str, core: Core = Depends(get_core)) -> dict:
    _check_key(key)
    session = core.registry.remove(key)
    flushed = 0
    if session is not None:
        events = session.end()
        core.pipeline.emit(key, events)
        core.pipeline.drop_key(key)
        core.detector.forget(key)
        flushed = len(events)
    core.metrics.sessions.set(len(core.registry))
    return {"flushedEvents": flushed}
