"""Authorization: Bearer 검증. 키 두 개(VISION_API_KEY, VISION_API_KEY_NEXT)가 동시에 유효하다."""

from __future__ import annotations

import hmac


class BearerVerifier:
    def __init__(self, keys: tuple[str, ...]) -> None:
        self._keys = tuple(k.encode() for k in keys if k)

    def verify(self, header: str | None) -> bool:
        if not header:
            return False
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return False
        candidate = token.strip().encode()
        ok = False
        for key in self._keys:  # 전부 비교해 시간차로 어느 키인지 새지 않게 한다
            ok |= hmac.compare_digest(candidate, key)
        return ok
