"""Prometheus 지표."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class Metrics:
    """앱마다 독립 레지스트리를 쓴다(테스트에서 중복 등록 방지)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.frames_received = Counter("pair_vision_frames_received_total", "받은 프레임", ["channel"], registry=r)
        self.frames_dropped = Counter("pair_vision_frames_dropped_total", "버린 프레임", ["reason"], registry=r)
        self.frames_processed = Counter("pair_vision_frames_processed_total", "분석한 프레임", ["channel"], registry=r)
        self.frame_latency = Histogram(
            "pair_vision_frame_latency_seconds", "수신~분석 완료 지연", registry=r,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
        )
        self.sessions = Gauge("pair_vision_sessions", "활성 세션 수", registry=r)
        self.events = Counter("pair_vision_events_total", "만든 이벤트", ["type", "final"], registry=r)
        self.callback_failures = Counter("pair_vision_callback_failures_total", "콜백 실패", ["reason"], registry=r)
        self.callback_dropped = Counter("pair_vision_callback_dropped_total", "버린 콜백", ["reason"], registry=r)
        self.callback_queue = Gauge("pair_vision_callback_queue_size", "콜백 대기 건수", registry=r)
        self.gpu_utilization = Gauge("pair_vision_gpu_utilization_ratio", "GPU 사용률(0~1, 가짜 모델은 0)", registry=r)
        self.detector_errors = Counter("pair_vision_detector_errors_total", "모델 실행 오류", registry=r)

    def render(self) -> bytes:
        return generate_latest(self.registry)
