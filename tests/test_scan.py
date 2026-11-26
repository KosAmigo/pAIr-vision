"""방 스캔 API(POST /v1/scan): 가짜 모델의 `SCAN:` 표식으로 검출을 정해 계약을 시험한다."""

import pytest
from fastapi.testclient import TestClient

from pair_vision.api.app import create_app
from pair_vision.api.scan import build_result
from pair_vision.models.fake import FakeDetector


def jpeg(*labels: str, pad: int = 100) -> bytes:
    return b"\xff\xd8\xff\xe0SCAN:" + ",".join(labels).encode() + b"\x00" * pad


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, detector=FakeDetector())) as c:
        yield c


def post(client, auth, files):
    return client.post("/v1/scan", headers=auth, files=[("images", (n, b, "image/jpeg")) for n, b in files])


def test_requires_auth(client):
    assert client.post("/v1/scan", files=[("images", ("h00", jpeg(), "image/jpeg"))]).status_code == 401


def test_pass_when_nothing_suspicious(client, auth):
    res = post(client, auth, [("h00", jpeg("person", "tv")), ("v0", jpeg("person"))])
    assert res.status_code == 200
    body = res.json()
    assert body["verdict"] == {"status": "PASS", "reasons": []}
    assert body["summary"] == {"maxPersons": 1, "phones": 0, "monitors": 1, "laptops": 0, "books": 0, "tablets": 0}
    item = body["items"][0]
    assert item["name"] == "h00" and item["persons"] == 1 and item["monitors"] == 1
    assert {o["label"] for o in item["objects"]} == {"person", "tv"}
    assert all(0 <= v <= 1 for o in item["objects"] for v in o["box"]) and isinstance(body["ms"], int)


def test_warn_reasons_and_names(client, auth):
    files = [
        ("h03", jpeg("person", "person")),
        ("h04", jpeg("phone", "book")),
        ("m_left", jpeg("tablet")),
    ]
    body = post(client, auth, files).json()
    reasons = {r["code"]: r["names"] for r in body["verdict"]["reasons"]}
    assert body["verdict"]["status"] == "WARN"
    assert reasons == {"MULTI_PERSON": ["h03"], "PHONE": ["h04"], "BOOK": ["h04"], "TABLET": ["m_left"]}
    assert [r["code"] for r in body["verdict"]["reasons"]] == ["MULTI_PERSON", "PHONE", "BOOK", "TABLET"]
    assert next(o for o in body["items"][1]["objects"] if o["label"] == "cell phone")


def test_multi_monitor_only_within_one_photo():
    # 같은 모니터가 두 장에 찍혀도(summary 합계 2) 한 장 안에 1대뿐이면 경고하지 않는다
    items = [
        {"name": "h00", "persons": 1, "phones": 0, "monitors": 1, "laptops": 0, "books": 0, "tablets": 0},
        {"name": "h01", "persons": 1, "phones": 0, "monitors": 1, "laptops": 0, "books": 0, "tablets": 0},
    ]
    assert build_result(items)["verdict"]["status"] == "PASS"
    items[1] = {**items[1], "laptops": 1}  # 한 장 안에 tv + laptop
    assert build_result(items)["verdict"]["reasons"] == [{"code": "MULTI_MONITOR", "names": ["h01"]}]


def test_multi_monitor_via_api(client, auth):
    body = post(client, auth, [("h00", jpeg("tv", "laptop")), ("h01", jpeg("tv"))]).json()
    assert body["verdict"]["reasons"] == [{"code": "MULTI_MONITOR", "names": ["h00"]}]


@pytest.mark.parametrize(
    "files",
    [
        [],
        [("h12", b"\xff\xd8\xff\xe0")],  # 이름 범위 밖
        [("hello", b"\xff\xd8\xff\xe0")],
        [("h00", b"\xff\xd8\xff\xe0"), ("h00", b"\xff\xd8\xff\xe0")],  # 중복
        [("h00", b"GIF89a....")],  # JPEG 아님
        [("h00", b"\xff\xd8\xff" + b"\x00" * (400 * 1024))],  # 400KB 초과
    ],
)
def test_bad_scan(client, auth, files):
    res = post(client, auth, files) if files else client.post("/v1/scan", headers=auth)
    assert res.status_code == 400 and res.json()["code"] == "BAD_SCAN"


def test_too_many_images(client, auth):
    names = [f"h{i:02d}" for i in range(12)] + [f"v{i}" for i in range(5)] + ["m_front", "m_left", "m_back", "m_right", "m_floor", "m_ceiling"]
    assert len(names) == 23
    assert post(client, auth, [(n, jpeg()) for n in names]).status_code == 200
    assert post(client, auth, [(n, jpeg()) for n in names + ["h00"]]).status_code == 400  # 24장째가 중복
    # 25장 이상은 장수 오류
    many = [(f"h{i % 12:02d}", jpeg()) for i in range(25)]
    assert post(client, auth, many).json()["code"] == "BAD_SCAN"


def test_filename_with_extension_is_accepted(client, auth):
    assert post(client, auth, [("h01.jpg", jpeg("person"))]).json()["items"][0]["name"] == "h01"
