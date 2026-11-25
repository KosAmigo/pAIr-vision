"""입장 점검(check 모드) 판정: 영상·사진 없이 항목·힌트·holdMs·passed 만 만든다 (계약 §3-1·§3-2)."""

from __future__ import annotations

from pair_vision.models.base import Box, Detection
from pair_vision.rules.config import CheckRules

EDGE = 0.01
ITEM_ORDER = ("keyboard", "mouse", "leftHand", "rightHand", "desk")


def _inside_any(hand_x: float, hand_y: float, boxes: list[Box]) -> bool:
    return any(b.contains(hand_x, hand_y) for b in boxes)


def _edge_hints(prefix: str, box: Box | None) -> list[str]:
    if box is None:
        return []
    hints = []
    if box.x0 <= EDGE:
        hints.append(f"{prefix}_LEFT")
    if box.x1 >= 1 - EDGE:
        hints.append(f"{prefix}_RIGHT")
    if box.y0 <= EDGE:
        hints.append(f"{prefix}_TOP")
    if box.y1 >= 1 - EDGE:
        hints.append(f"{prefix}_BOTTOM")
    return hints


class CheckEvaluator:
    def __init__(self, rules: CheckRules, area_margin: float) -> None:
        self._rules = rules
        self._margin = area_margin
        self.reset()

    def reset(self) -> None:
        self._hold_start_ms: int | None = None
        self._last_sent_ms: int | None = None

    def evaluate(self, det: Detection) -> dict | None:
        """휴대폰 프레임 하나를 판정. 너무 자주(0.5초 안) 부르면 None(보내지 않음)."""
        items, hints = self._items_and_hints(det)
        all_ok = all(items.values())
        if all_ok:
            if self._hold_start_ms is None:
                self._hold_start_ms = det.recv_ms
        else:
            self._hold_start_ms = None
        hold = det.recv_ms - self._hold_start_ms if self._hold_start_ms is not None else 0
        if self._last_sent_ms is not None and det.recv_ms - self._last_sent_ms < self._rules.min_interval_ms:
            return None
        self._last_sent_ms = det.recv_ms
        return {
            "items": items,
            "hints": hints,
            "holdMs": hold,
            "passed": hold >= self._rules.pass_hold_ms,
        }

    def _items_and_hints(self, det: Detection) -> tuple[dict[str, bool], list[str]]:
        boxes = [b.expanded(self._margin) for b in (det.keyboard, det.mouse) if b is not None]
        left = any(h.side == "left" and _inside_any(h.x, h.y, boxes) for h in det.hands)
        right = any(h.side == "right" and _inside_any(h.x, h.y, boxes) for h in det.hands)
        items = {
            "keyboard": det.keyboard is not None,
            "mouse": det.mouse is not None,
            "leftHand": left,
            "rightHand": right,
            "desk": det.desk_visible,
        }
        hints: list[str] = []
        if not items["keyboard"]:
            hints.append("KEYBOARD_MISSING")
        if not items["mouse"]:
            hints.append("MOUSE_MISSING")
        if not left:
            hints.append("LEFT_HAND_MISSING")
        if not right:
            hints.append("RIGHT_HAND_MISSING")
        if not items["desk"]:
            hints.append("DESK_PARTIAL")
        hints += _edge_hints("KEYBOARD", det.keyboard)
        return items, hints
