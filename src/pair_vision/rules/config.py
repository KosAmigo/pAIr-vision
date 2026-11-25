"""규칙 설정 YAML 로더 (계약 §9). 값은 설정 파일에서만 온다."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class HandRules:
    hands_out_ms: int = 7000
    hand_out_one_ms: int = 10000
    gap_tolerance_ms: int = 3000
    cooldown_ms: int = 10000
    unseen_hand_counts_as_out: bool = True
    area_margin: float = 0.05


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


@dataclass(frozen=True)
class RulesConfig:
    version: str
    profiles: Mapping[str, Profile]

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
    )


def parse_rules(data: Mapping) -> RulesConfig:
    version = str(data.get("version", "")).strip()
    if not version:
        raise ValueError("규칙 설정에 version 이 필요합니다.")
    profiles_raw = data.get("profiles") or {}
    if "default" not in profiles_raw:
        raise ValueError("규칙 설정에 default 프로필이 필요합니다.")
    profiles = {name: _parse_profile(name, raw or {}) for name, raw in profiles_raw.items()}
    return RulesConfig(version=version, profiles=profiles)


def load_rules(path: str | Path) -> RulesConfig:
    with open(path, encoding="utf-8") as fh:
        return parse_rules(yaml.safe_load(fh) or {})
