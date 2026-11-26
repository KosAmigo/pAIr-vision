"""가짜 Detector: 시나리오대로 손·키보드 위치를 내놓는 재생기.

우선순위
1. 프레임 JPEG 안에 `FAKE:<STATE>` 표식이 있으면 그 상태 (가짜 클라이언트·테스트용)
2. 환경 변수 `VISION_FAKE_SCENARIO` 의 시나리오(세션 첫 프레임부터의 경과 시간 기준)

시나리오는 이름(`normal` `hands_out` `one_hand_out` `flicker`)이거나
`NORMAL:3,BOTH_OUT:10,NORMAL:1` 같은 직접 표기(마지막 구간이 계속 유지됨).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pair_vision.models.base import (
    PHONE,
    Box,
    Detection,
    Detector,
    Frame,
    HandDet,
    ObjectDet,
    Shoulders,
    WebcamFeatures,
)

KEYBOARD = Box(0.20, 0.50, 0.60, 0.90)
MOUSE = Box(0.65, 0.60, 0.80, 0.80)
LEFT_ON_KEYBOARD = HandDet("left", 0.35, 0.70)
RIGHT_ON_MOUSE = HandDet("right", 0.72, 0.70)
LEFT_FAR = HandDet("left", 0.05, 0.10)
RIGHT_FAR = HandDet("right", 0.95, 0.10)

# 상태 이름 -> (손 목록, 책상 보임). 키보드·마우스는 항상 보인다.
STATES: dict[str, tuple[tuple[HandDet, ...], bool]] = {
    "NORMAL": ((LEFT_ON_KEYBOARD, RIGHT_ON_MOUSE), True),
    "BOTH_OUT": ((LEFT_FAR, RIGHT_FAR), True),  # 두 손이 보이지만 영역 밖
    "NO_HANDS": ((), True),  # 손이 안 보임(설정상 "밖")
    "ONE_OUT": ((LEFT_ON_KEYBOARD, RIGHT_FAR), True),  # 한 손만 영역 밖
    "ONE_HAND": ((LEFT_ON_KEYBOARD,), True),  # 한 손만 보임
    "NO_DESK": ((LEFT_ON_KEYBOARD, RIGHT_ON_MOUSE), False),
    "TABLET": ((LEFT_ON_KEYBOARD, RIGHT_ON_MOUSE), True),  # 정상 + 태블릿 보임
    "NOTE": ((LEFT_ON_KEYBOARD, RIGHT_ON_MOUSE), True),  # 정상 + 노트 보임
}

# 상태 이름 -> 그 상태에서 보이는 추가 물체(가짜 시나리오·표식 시험용)
FAKE_OBJECTS: dict[str, ObjectDet] = {
    "TABLET": ObjectDet("tablet", 0.8, Box(0.1, 0.1, 0.4, 0.3)),
    "NOTE": ObjectDet("note", 0.7, Box(0.1, 0.1, 0.4, 0.3)),
}

# 웹캠 가짜 상태(프레임 표식 `FAKE:<이름>`). 표식이 없으면 FRONT(보정 자세와 같은 정면).
_SH = Shoulders(0.5, 0.7, 0.4)
WEBCAM_STATES: dict[str, WebcamFeatures] = {
    "FRONT": WebcamFeatures(1, 0.0, 0.0, 0.0, 0.0, 0.0, _SH),
    "NO_FACE": WebcamFeatures(0, shoulders=_SH),
    "MULTI_FACE": WebcamFeatures(2, 0.0, 0.0, 0.0, 0.0, 0.0, _SH),
    "HEAD_TURN": WebcamFeatures(1, 40.0, 0.0, 0.0, 0.0, 0.0, _SH),
    "HEAD_DOWN": WebcamFeatures(1, 0.0, 35.0, 0.0, 0.0, 0.0, _SH),
    "GAZE_AWAY": WebcamFeatures(1, 0.0, 0.0, 0.7, 0.0, 0.0, _SH),
    "TORSO_SHIFT": WebcamFeatures(1, 0.0, 0.0, 0.0, 0.0, 0.0, Shoulders(0.75, 0.7, 0.4)),
    "TORSO_TURN": WebcamFeatures(1, 0.0, 0.0, 0.0, 0.0, 0.0, Shoulders(0.5, 0.7, 0.2)),
}

_MARKER = re.compile(rb"FAKE:([A-Z_]+)")

NAMED_SCENARIOS: dict[str, list[tuple[str, float]]] = {
    "normal": [("NORMAL", 0)],
    "hands_out": [("NORMAL", 3), ("BOTH_OUT", 10), ("NORMAL", 1)],
    "one_hand_out": [("NORMAL", 3), ("ONE_OUT", 12), ("NORMAL", 1)],
    # 끊김 허용(3초) 시험용: 밖 4초 + 정상 2초가 반복되어도 하나의 이상 상태로 이어 센다
    "flicker": [("NORMAL", 2), ("BOTH_OUT", 4), ("NORMAL", 2), ("BOTH_OUT", 4), ("NORMAL", 1)],
}


def parse_scenario(spec: str) -> list[tuple[str, float]]:
    spec = spec.strip()
    if spec in NAMED_SCENARIOS:
        return NAMED_SCENARIOS[spec]
    segments: list[tuple[str, float]] = []
    for part in spec.split(","):
        state, _, seconds = part.strip().partition(":")
        if state not in STATES or not seconds:
            raise ValueError(f"알 수 없는 가짜 시나리오: {spec!r}")
        segments.append((state, float(seconds)))
    return segments


def state_at(segments: list[tuple[str, float]], elapsed_s: float) -> str:
    acc = 0.0
    for state, seconds in segments:
        acc += seconds
        if elapsed_s < acc:
            return state
    return segments[-1][0]


class FakeDetector(Detector):
    name = "fake"
    gpu = False

    def __init__(self, scenario: str = "normal") -> None:
        self._segments = parse_scenario(scenario)

    def _state_for(self, frame: Frame) -> str:
        match = _MARKER.search(frame.jpeg)
        if match:
            state = match.group(1).decode("ascii")
            if state in STATES:
                return state
        return state_at(self._segments, max(0, frame.recv_ms - frame.origin_ms) / 1000)

    def run(self, frames: Sequence[Frame]) -> list[Detection]:
        result: list[Detection] = []
        for frame in frames:
            if frame.channel != PHONE:
                match = _MARKER.search(frame.jpeg)
                name = match.group(1).decode("ascii") if match else "FRONT"
                result.append(
                    Detection(
                        frame.key, frame.channel, frame.seq, frame.recv_ms,
                        webcam=WEBCAM_STATES.get(name, WEBCAM_STATES["FRONT"]),
                    )
                )
                continue
            state = self._state_for(frame)
            hands, desk = STATES[state]
            result.append(
                Detection(
                    key=frame.key,
                    channel=frame.channel,
                    seq=frame.seq,
                    recv_ms=frame.recv_ms,
                    keyboard=KEYBOARD,
                    mouse=MOUSE,
                    hands=hands,
                    desk_visible=desk,
                    objects=(FAKE_OBJECTS[state],) if state in FAKE_OBJECTS else (),
                )
            )
        return result


def create_detector(
    kind: str,
    scenario: str,
    *,
    models_dir: str = "models",
    model_rules=None,
    lock_path: str | None = None,
    verify: bool = True,
    workers: int = 1,
    worker_threads: int = 0,
) -> Detector:
    if workers > 1:  # 모델을 프로세스 여러 개에 올려 병렬 처리(실측: 한 프로세스는 CPU 에서 초당 약 5장에서 막힘)
        from pair_vision.models.pool import PooledDetector

        spec = {"kind": kind, "scenario": scenario, "models_dir": models_dir, "model_rules": model_rules,
                "lock_path": lock_path, "verify": verify}
        return PooledDetector(spec, workers, worker_threads)
    if kind == "fake":
        return FakeDetector(scenario)
    if kind == "real":
        from pair_vision.models.real import RealDetector  # torch 등은 load() 에서야 import 된다
        from pair_vision.rules.config import ModelRules

        return RealDetector(
            models_dir, model_rules or ModelRules(), lock_path=lock_path, verify=verify
        )
    raise ValueError(f"지원하지 않는 detector: {kind!r} (fake | real)")
