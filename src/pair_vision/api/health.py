"""/healthz (프로세스) · /readyz (모델·여유) · /metrics (Prometheus) · POST /v1/admin/drain."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response

from pair_vision.api.core import Core, get_core, require_auth

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@router.get("/readyz", dependencies=[Depends(require_auth)])
def readyz(core: Core = Depends(get_core)) -> JSONResponse:
    # drain 중에도 200 을 준다: 503 이면 백엔드가 인스턴스를 죽은 것으로 보고 감독 중단 처리하기 때문
    ready = core.models_loaded
    body = {
        "ready": ready,
        "modelsLoaded": core.models_loaded,
        "gpu": core.detector.gpu,
        "sessions": len(core.registry),
        "maxSessions": core.registry.max_sessions,
        "draining": core.registry.draining,
        "version": core.settings.version,
        "rulesVersion": core.rules.version,
    }
    return JSONResponse(body, status_code=200 if ready else 503)


@router.get("/metrics", dependencies=[Depends(require_auth)])
def metrics(core: Core = Depends(get_core)) -> Response:
    core.metrics.sessions.set(len(core.registry))
    for stage, (calls, seconds) in core.detector.stats().items():
        core.metrics.stage_calls.labels(stage).set(calls)
        core.metrics.stage_seconds.labels(stage).set(seconds)
    return Response(core.metrics.render(), media_type="text/plain; version=0.0.4")


@router.post("/v1/admin/drain", status_code=202, dependencies=[Depends(require_auth)])
def drain(core: Core = Depends(get_core)) -> dict:
    core.registry.draining = True
    return {"draining": True, "sessions": len(core.registry)}
