"""POST /v1/scan: 방 스캔(사진 최대 24장 -> 사람·휴대폰·모니터·책·태블릿 검출과 판정).

실시간 프레임 처리와 같은 Detector 를 쓰고(Detector.scan), 사진 안 물체 수를 세어 PASS/WARN 을 낸다.
사진은 메모리에서만 쓰고 디스크·로그에 남기지 않는다.
"""

from __future__ import annotations

import re
import time

from fastapi import APIRouter, Depends, File, UploadFile

from pair_vision.api.core import ApiError, Core, get_core, require_auth
from pair_vision.models.base import ObjectDet, ScanDecodeError

router = APIRouter(prefix="/v1", dependencies=[Depends(require_auth)])

MAX_IMAGES = 24
MAX_IMAGE_BYTES = 400 * 1024
JPEG_MAGIC = b"\xff\xd8\xff"
# 가로 h00~h11, 세로 v0~v4, 수동 모드 m_front m_left m_back m_right m_floor m_ceiling
NAME_RE = re.compile(r"^(h(0[0-9]|1[01])|v[0-4]|m_(front|left|back|right|floor|ceiling))$")
# 응답 objects[].label (COCO 이름 그대로, 태블릿만 OWLv2 이름)
API_LABEL = {"person": "person", "phone": "cell phone", "tv": "tv", "laptop": "laptop", "book": "book", "tablet": "tablet"}
# reasons 순서
REASON_ORDER = ("MULTI_PERSON", "PHONE", "MULTI_MONITOR", "BOOK", "TABLET")


def _count(objects: list[ObjectDet], label: str) -> int:
    return sum(1 for o in objects if o.label == label)


def build_item(name: str, objects: list[ObjectDet]) -> dict:
    return {
        "name": name,
        "persons": _count(objects, "person"),
        "phones": _count(objects, "phone"),
        "monitors": _count(objects, "tv"),
        "laptops": _count(objects, "laptop"),
        "books": _count(objects, "book"),
        "tablets": _count(objects, "tablet"),
        "objects": [
            {
                "label": API_LABEL[o.label],
                "score": round(o.score, 2),
                "box": [round(min(1.0, max(0.0, v)), 3) for v in (o.box.x0, o.box.y0, o.box.x1, o.box.y1)],
            }
            for o in objects
        ],
    }


def build_result(items: list[dict]) -> dict:
    """사진별 항목으로 요약·판정을 만든다. 요약은 사진 한 장 기준 최댓값이다(한 바퀴 돌면 같은 물체가 여러 장에 찍히므로 합계는 부풀려진다)."""

    def peak(field: str) -> int:
        return max((i[field] for i in items), default=0)

    summary = {
        "maxPersons": peak("persons"),
        "phones": peak("phones"),
        "monitors": peak("monitors"),
        "laptops": peak("laptops"),
        "books": peak("books"),
        "tablets": peak("tablets"),
    }
    names: dict[str, list[str]] = {
        "MULTI_PERSON": [i["name"] for i in items if i["persons"] >= 2],
        "PHONE": [i["name"] for i in items if i["phones"] >= 1],
        # 모니터 여러 대는 사진 한 장 안에서 tv+laptop 이 2개 이상일 때만(같은 모니터가 여러 장에 찍히므로 summary 로 판단하지 않는다)
        "MULTI_MONITOR": [i["name"] for i in items if i["monitors"] + i["laptops"] >= 2],
        "BOOK": [i["name"] for i in items if i["books"] >= 1],
        "TABLET": [i["name"] for i in items if i["tablets"] >= 1],
    }
    reasons = [{"code": code, "names": names[code]} for code in REASON_ORDER if names[code]]
    return {"summary": summary, "verdict": {"status": "WARN" if reasons else "PASS", "reasons": reasons}}


def _bad(message: str) -> ApiError:
    return ApiError(400, "BAD_SCAN", message)


@router.post("/scan")
async def scan(images: list[UploadFile] = File(default=[]), core: Core = Depends(get_core)) -> dict:
    started = time.perf_counter()
    if not 1 <= len(images) <= MAX_IMAGES:
        raise _bad(f"Send 1 to {MAX_IMAGES} images.")
    names: list[str] = []
    blobs: list[bytes] = []
    for upload in images:
        name = (upload.filename or "").rsplit("/", 1)[-1].removesuffix(".jpeg").removesuffix(".jpg")  # h03 / h03.jpg 모두
        if not NAME_RE.match(name) or name in names:
            raise _bad("Unknown or duplicate image name.")
        blob = await upload.read(MAX_IMAGE_BYTES + 1)
        if len(blob) > MAX_IMAGE_BYTES:
            raise _bad("Image too large (max 400KB).")
        if not blob.startswith(JPEG_MAGIC):
            raise _bad("Images must be JPEG.")
        names.append(name)
        blobs.append(blob)
    try:
        found = await core.detector.scan_async(blobs)
    except ScanDecodeError:
        raise _bad("Cannot decode image.") from None
    items = [build_item(n, objs) for n, objs in zip(names, found, strict=True)]
    return {"items": items, **build_result(items), "ms": round((time.perf_counter() - started) * 1000)}
