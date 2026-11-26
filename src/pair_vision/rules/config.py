"""규칙 설정 YAML 로더 (계약 §9). 값은 설정 파일에서만 온다."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from pair_vision.rules.webcam import WEBCAM_EVENT_TYPES, WebcamThresholds


@dataclass(frozen=True)
class HandRules:
    hands_out_ms: int = 7000
    hand_out_one_ms: int = 10000
    gap_tolerance_ms: int = 3000
    cooldown_ms: int = 10000
    unseen_hand_counts_as_out: bool = True
    area_margin: float = 0.05


@dataclass(frozen=True)
class ObjectRule:
    """물체(태블릿·노트·스마트워치 등) 이벤트 규칙: 연속으로 보이는 시간이 min_ms 이상이면 이벤트."""

    min_ms: int
    gap_ms: int  # 안 보여도 이 시간까지는 같은 상태로 이어 센다(OWLv2 는 주기 검출이라 주기의 2배 이상이어야 한다)
    cooldown_ms: int  # 이벤트가 끝난 시점부터 재발 최소 간격
    shadow: bool = False  # true 면 콜백을 보내지 않고 지표·로그에만 남긴다(그림자 모드)


# 설계 문서에 값이 없어 보수적으로 둔 시작값(실측 전 임시). 근거는 README "가정한 값".
DEFAULT_OBJECT_RULES: dict[str, ObjectRule] = {
    "tablet": ObjectRule(min_ms=10000, gap_ms=10000, cooldown_ms=30000),
    "note": ObjectRule(min_ms=10000, gap_ms=3000, cooldown_ms=30000),
    "smartwatch": ObjectRule(min_ms=15000, gap_ms=10000, cooldown_ms=60000),
    "many_hands": ObjectRule(min_ms=5000, gap_ms=3000, cooldown_ms=30000, shadow=True),
}


# 웹캠 규칙 시작값 = 프론트 pcRules.ts 의 PC_RULES(지속·쿨다운). 끊김 허용(gap)은 프론트의 0.4초가 아니라
# 서버가 초당 1장 분석하므로 프레임 2~3장 간격(2.5초)으로 둔 가정값.
DEFAULT_WEBCAM_RULES: dict[str, ObjectRule] = {
    "no_face": ObjectRule(min_ms=2000, gap_ms=2500, cooldown_ms=8000),
    "multi_face": ObjectRule(min_ms=1000, gap_ms=2500, cooldown_ms=8000),
    "head_turn": ObjectRule(min_ms=1500, gap_ms=2500, cooldown_ms=6000),
    "head_down": ObjectRule(min_ms=5000, gap_ms=2500, cooldown_ms=10000),
    "gaze_away": ObjectRule(min_ms=5000, gap_ms=2500, cooldown_ms=6000),
    "torso_shift": ObjectRule(min_ms=2000, gap_ms=2500, cooldown_ms=8000),
    "torso_turn": ObjectRule(min_ms=2000, gap_ms=2500, cooldown_ms=8000),
}


# 기본으로 꺼 두는 웹캠 규칙. 시선(GAZE_AWAY)은 계약 해상도(400px·q55)에서 값이 절반 아래로 줄어 믿기 어렵다고
# 측정돼(README) 2026-11-26 사용자 결정으로 끈다. 켜려면 규칙 파일에서 `enabled: true`.
DISABLED_WEBCAM_RULES = frozenset({"gaze_away"})


@dataclass(frozen=True)
class WebcamRules:
    thresholds: WebcamThresholds = field(default_factory=WebcamThresholds)
    rules: Mapping[str, ObjectRule] = field(
        default_factory=lambda: {k: v for k, v in DEFAULT_WEBCAM_RULES.items() if k not in DISABLED_WEBCAM_RULES}
    )
    calibration_min_samples: int = 3  # 프론트는 5. 초당 1장 x 3초 보정이면 4장쯤 모인다


@dataclass(frozen=True)
class CheckRules:
    pass_hold_ms: int = 3000
    min_interval_ms: int = 500


@dataclass(frozen=True)
class Profile:
    name: str
    calibration_ms: int = 3000
    hands: HandRules = field(default_factory=HandRules)
    check: CheckRules = field(default_factory=CheckRules)
    objects: Mapping[str, ObjectRule] = field(default_factory=lambda: dict(DEFAULT_OBJECT_RULES))
    webcam: WebcamRules = field(default_factory=WebcamRules)


@dataclass(frozen=True)
class ModelRules:
    """실제 모델 설정(기준값·주기). 값은 규칙 파일의 `models:` 에서 오고 rulesVersion 으로 추적된다."""

    keyboard_threshold: float = 0.3
    mouse_threshold: float = 0.4
    note_threshold: float = 0.3  # COCO book
    tablet_threshold: float = 0.5
    smartwatch_threshold: float = 0.5
    phone_threshold: float = 0.5
    owl_interval_ms: int = 5000
    hand_threshold: float = 0.5
    max_hands: int = 4
    hand_merge_distance: float = 0.06  # 이보다 가까운 두 손은 하나로(프레임 비율, 점수 높은 쪽 유지)
    pose_every: int = 3  # 웹캠: 얼굴은 매 프레임, 포즈(어깨)는 N 프레임마다(사이에는 직전 값 재사용, CPU 절약. 프론트는 4)


@dataclass(frozen=True)
class RulesConfig:
    version: str
    profiles: Mapping[str, Profile]
    models: ModelRules = field(default_factory=ModelRules)

    def profile(self, name: str) -> Profile | None:
        return self.profiles.get(name)


def _positive(section: str, key: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"규칙 설정 {section}.{key} 는 0 이상의 정수여야 합니다: {value!r}")
    return value


def _parse_profile(name: str, raw: Mapping) -> Profile:
    hands_raw = raw.get("hands", {}) or {}
    check_raw = raw.get("check", {}) or {}
    defaults_h, defaults_c = HandRules(), CheckRules()
    hands = HandRules(
        hands_out_ms=_positive("hands", "hands_out_ms", hands_raw.get("hands_out_ms", defaults_h.hands_out_ms)),
        hand_out_one_ms=_positive("hands", "hand_out_one_ms", hands_raw.get("hand_out_one_ms", defaults_h.hand_out_one_ms)),
        gap_tolerance_ms=_positive("hands", "gap_tolerance_ms", hands_raw.get("gap_tolerance_ms", defaults_h.gap_tolerance_ms)),
        cooldown_ms=_positive("hands", "cooldown_ms", hands_raw.get("cooldown_ms", defaults_h.cooldown_ms)),
        unseen_hand_counts_as_out=bool(
            hands_raw.get("unseen_hand_counts_as_out", defaults_h.unseen_hand_counts_as_out)
        ),
        area_margin=float(hands_raw.get("area_margin", defaults_h.area_margin)),
    )
    check = CheckRules(
        pass_hold_ms=_positive("check", "pass_hold_ms", check_raw.get("pass_hold_ms", defaults_c.pass_hold_ms)),
        min_interval_ms=_positive("check", "min_interval_ms", check_raw.get("min_interval_ms", defaults_c.min_interval_ms)),
    )
    return Profile(
        name=name,
        calibration_ms=_positive("profile", "calibration_ms", raw.get("calibration_ms", 3000)),
        hands=hands,
        check=check,
        objects=_parse_objects(raw.get("objects", {}) or {}),
        webcam=_parse_webcam(raw.get("webcam", {}) or {}),
    )


def _parse_objects(raw: Mapping) -> dict[str, ObjectRule]:
    unknown = set(raw) - set(DEFAULT_OBJECT_RULES)
    if unknown:
        raise ValueError(f"규칙 설정 objects 에 알 수 없는 항목: {sorted(unknown)}")
    rules: dict[str, ObjectRule] = {}
    for label, default in DEFAULT_OBJECT_RULES.items():
        item = raw.get(label, {}) or {}
        section = f"objects.{label}"
        rules[label] = ObjectRule(
            min_ms=_positive(section, "min_ms", item.get("min_ms", default.min_ms)),
            gap_ms=_positive(section, "gap_ms", item.get("gap_ms", default.gap_ms)),
            cooldown_ms=_positive(section, "cooldown_ms", item.get("cooldown_ms", default.cooldown_ms)),
            shadow=bool(item.get("shadow", default.shadow)),
        )
    return rules


def _parse_webcam(raw: Mapping) -> WebcamRules:
    d = WebcamThresholds()
    th_raw = raw.get("thresholds", {}) or {}
    unknown = set(th_raw) - set(vars(d))
    if unknown:
        raise ValueError(f"규칙 설정 webcam.thresholds 에 알 수 없는 항목: {sorted(unknown)}")
    values = {}
    for key, default in vars(d).items():
        value = th_raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"규칙 설정 webcam.thresholds.{key} 는 0 보다 커야 합니다: {value!r}")
        values[key] = float(value)
    rules_raw = raw.get("rules", {}) or {}
    unknown = set(rules_raw) - set(WEBCAM_EVENT_TYPES)
    if unknown:
        raise ValueError(f"규칙 설정 webcam.rules 에 알 수 없는 항목: {sorted(unknown)}")
    rules = {}
    for label, default_rule in DEFAULT_WEBCAM_RULES.items():
        item = rules_raw.get(label, {}) or {}
        section = f"webcam.rules.{label}"
        if not item.get("enabled", label not in DISABLED_WEBCAM_RULES):
            continue  # 꺼진 규칙은 엔진에 넣지 않아 이벤트가 나지 않는다
        rules[label] = ObjectRule(
            min_ms=_positive(section, "min_ms", item.get("min_ms", default_rule.min_ms)),
            gap_ms=_positive(section, "gap_ms", item.get("gap_ms", default_rule.gap_ms)),
            cooldown_ms=_positive(section, "cooldown_ms", item.get("cooldown_ms", default_rule.cooldown_ms)),
        )
    return WebcamRules(
        thresholds=WebcamThresholds(**values),
        rules=rules,
        calibration_min_samples=_positive("webcam", "calibration_min_samples", raw.get("calibration_min_samples", 3)),
    )


def _score(section: str, key: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < value <= 1.0:
        raise ValueError(f"규칙 설정 {section}.{key} 는 0 초과 1 이하여야 합니다: {value!r}")
    return float(value)


def _parse_models(raw: Mapping) -> ModelRules:
    d = ModelRules()
    max_hands = raw.get("max_hands", d.max_hands)
    if not isinstance(max_hands, int) or isinstance(max_hands, bool) or max_hands < 2:
        raise ValueError(f"규칙 설정 models.max_hands 는 2 이상의 정수여야 합니다: {max_hands!r}")
    merge = raw.get("hand_merge_distance", d.hand_merge_distance)
    if isinstance(merge, bool) or not isinstance(merge, (int, float)) or merge < 0:
        raise ValueError(f"규칙 설정 models.hand_merge_distance 는 0 이상이어야 합니다: {merge!r}")
    scores = {
        key: _score("models", key, raw.get(key, getattr(d, key)))
        for key in (
            "keyboard_threshold", "mouse_threshold", "note_threshold", "tablet_threshold",
            "smartwatch_threshold", "phone_threshold", "hand_threshold",
        )
    }
    pose_every = raw.get("pose_every", d.pose_every)
    if not isinstance(pose_every, int) or isinstance(pose_every, bool) or pose_every < 1:
        raise ValueError(f"규칙 설정 models.pose_every 는 1 이상의 정수여야 합니다: {pose_every!r}")
    return ModelRules(
        **scores,
        pose_every=pose_every,
        owl_interval_ms=_positive("models", "owl_interval_ms", raw.get("owl_interval_ms", d.owl_interval_ms)),
        max_hands=max_hands,
        hand_merge_distance=float(merge),
    )


def parse_rules(data: Mapping) -> RulesConfig:
    version = str(data.get("version", "")).strip()
    if not version:
        raise ValueError("규칙 설정에 version 이 필요합니다.")
    profiles_raw = data.get("profiles") or {}
    if "default" not in profiles_raw:
        raise ValueError("규칙 설정에 default 프로필이 필요합니다.")
    profiles = {name: _parse_profile(name, raw or {}) for name, raw in profiles_raw.items()}
    models = _parse_models(data.get("models") or {})
    for profile in profiles.values():
        for label in ("tablet", "smartwatch"):  # OWLv2 주기 검출: 끊김 허용이 주기의 2배보다 짧으면 이벤트가 안 난다
            if profile.objects[label].gap_ms < 2 * models.owl_interval_ms:
                raise ValueError(
                    f"규칙 설정 objects.{label}.gap_ms 는 models.owl_interval_ms 의 2배 이상이어야 합니다 "
                    f"(프로필 {profile.name})."
                )
    return RulesConfig(version=version, profiles=profiles, models=models)


def load_rules(path: str | Path) -> RulesConfig:
    with open(path, encoding="utf-8") as fh:
        return parse_rules(yaml.safe_load(fh) or {})
