"""FastAPI 앱 조립. 테스트에서는 detector·sink 를 주입한다."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from pair_vision.api import health, sessions, stream
from pair_vision.api.core import ApiError, Core
from pair_vision.auth.bearer import BearerVerifier
from pair_vision.config import Settings
from pair_vision.evidence.callback import CallbackSender, EventSink, NullSender
from pair_vision.metrics import Metrics
from pair_vision.models.base import Detector
from pair_vision.models.fake import create_detector
from pair_vision.pipeline.worker import Hub, Pipeline
from pair_vision.rules.config import RulesConfig, load_rules
from pair_vision.session.registry import SessionRegistry

log = logging.getLogger("pair_vision")


def build_core(
    settings: Settings,
    *,
    detector: Detector | None = None,
    sink: EventSink | None = None,
    rules: RulesConfig | None = None,
) -> Core:
    rules = rules or load_rules(settings.rules_path)
    detector = detector or create_detector(
        settings.detector,
        settings.fake_scenario,
        models_dir=settings.models_dir,
        model_rules=rules.models,
        lock_path=settings.models_lock_path,
        verify=settings.verify_models,
    )
    metrics = Metrics()
    if sink is None:
        if settings.callback_url:
            client = httpx.AsyncClient(timeout=10.0)
            sink = CallbackSender(
                settings.callback_url, settings.callback_key_id, settings.callback_secret, client,
                metrics=metrics,
            )
        else:
            log.warning("VISION_CALLBACK_URL 이 없어 이벤트 콜백을 보내지 않습니다.")
            sink = NullSender(metrics)
    registry = SessionRegistry(rules, settings.max_sessions)
    hub = Hub()
    pipeline = Pipeline(registry, detector, hub, sink, metrics, rules.version, settings.max_frame_bytes)
    return Core(
        settings=settings, rules=rules, detector=detector, registry=registry, hub=hub,
        pipeline=pipeline, sink=sink, metrics=metrics, verifier=BearerVerifier(settings.api_keys),
    )


def create_app(
    settings: Settings | None = None,
    *,
    detector: Detector | None = None,
    sink: EventSink | None = None,
    rules: RulesConfig | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env(os.environ)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        core = build_core(settings, detector=detector, sink=sink, rules=rules)
        await asyncio.to_thread(core.detector.load)
        core.models_loaded = True
        app.state.core = core
        tasks = [
            asyncio.create_task(core.pipeline.run()),
            asyncio.create_task(core.pipeline.state_loop(settings.state_interval_s)),
        ]
        if isinstance(core.sink, CallbackSender):
            tasks.append(asyncio.create_task(core.sink.run()))
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            if isinstance(core.sink, CallbackSender):
                await core.sink.close()

    app = FastAPI(title="pAIr vision", version=settings.version, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return exc.response()

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"code": "BAD_REQUEST", "message": "Invalid request body."}, status_code=400)

    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(stream.router)
    return app
