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
            # 합격 기준(p95 <= 3초)을 버킷 경계로 정확히 판정하려고 3 을 넣었다
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 10),
        )
        # 같은 칸(응시자·채널)에서 분석된 프레임 사이 recvMs 간격. 끊김 허용(휴대폰 3초·웹캠 2.5초)을 넘는 비율을 본다.
        self.processed_gap = Histogram(
            "pair_vision_processed_gap_seconds", "같은 칸에서 분석된 프레임 사이 간격(recvMs 기준)", ["channel"], registry=r,
            buckets=(0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 10, 30),
        )
        self.batch_size = Histogram(
            "pair_vision_batch_size", "묶음 추론 단계별 한 번에 돌린 프레임 수", ["stage"], registry=r,
            buckets=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
        )
        self.inflight = Gauge("pair_vision_inflight_frames", "모델에 넘겨 처리 중인 프레임 수(연속 처리)", registry=r)
        self.sessions = Gauge("pair_vision_sessions", "활성 세션 수", registry=r)
        self.events = Counter("pair_vision_events_total", "만든 이벤트", ["type", "final"], registry=r)
        self.shadow_events = Counter("pair_vision_shadow_events_total", "그림자 모드 이벤트(콜백 안 보냄)", ["type"], registry=r)
        self.callback_failures = Counter("pair_vision_callback_failures_total", "콜백 실패", ["reason"], registry=r)
        self.callback_dropped = Counter("pair_vision_callback_dropped_total", "버린 콜백", ["reason"], registry=r)
        self.callback_queue = Gauge("pair_vision_callback_queue_size", "콜백 대기 건수", registry=r)
        self.gpu_utilization = Gauge("pair_vision_gpu_utilization_ratio", "GPU 사용률(0~1, 가짜 모델은 0)", registry=r)
        self.stage_seconds = Gauge("pair_vision_stage_seconds_total", "단계별 누적 처리 시간(초, 모든 워커 합)", ["stage"], registry=r)
        self.stage_calls = Gauge("pair_vision_stage_calls_total", "단계별 누적 호출 수(모든 워커 합)", ["stage"], registry=r)
        self.detector_errors = Counter("pair_vision_detector_errors_total", "모델 실행 오류", registry=r)

    def render(self) -> bytes:
        return generate_latest(self.registry)
