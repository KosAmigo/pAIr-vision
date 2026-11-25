"""앱 전체가 공유하는 객체 묶음과 공통 오류·인증 의존성."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Request
from fastapi.responses import JSONResponse

from pair_vision.auth.bearer import BearerVerifier
from pair_vision.config import Settings
from pair_vision.evidence.callback import EventSink
from pair_vision.metrics import Metrics
from pair_vision.models.base import Detector
from pair_vision.pipeline.worker import Hub, Pipeline
from pair_vision.rules.config import RulesConfig
from pair_vision.session.registry import SessionRegistry


@dataclass
class Core:
    settings: Settings
    rules: RulesConfig
    detector: Detector
    registry: SessionRegistry
    hub: Hub
    pipeline: Pipeline
    sink: EventSink
    metrics: Metrics
    verifier: BearerVerifier
    models_loaded: bool = False


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None):
        super().__init__(code)
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}

    def response(self) -> JSONResponse:
        return JSONResponse(
            {"code": self.code, "message": self.message}, status_code=self.status, headers=self.headers
        )


def get_core(request: Request) -> Core:
    return request.app.state.core


def require_auth(request: Request, core: Core = Depends(get_core)) -> None:
    if not core.verifier.verify(request.headers.get("authorization")):
        raise ApiError(401, "UNAUTHORIZED", "Invalid or missing bearer token.", {"WWW-Authenticate": "Bearer"})
