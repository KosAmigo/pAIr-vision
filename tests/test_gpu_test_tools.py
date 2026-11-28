"""GPU 합격 시험 도구(tools/gpu_test/load_test.py)의 판정 계산: 히스토그램 분위수·버킷 경계 판정·채널별 끊김 초과."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "gpu_test"))

import load_test as lt  # noqa: E402

LAT_LE = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 10)
GAP_LE = (0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 10, 30)


def hist(name, values, les, labels=""):
    out = {}
    for le in les:
        out[(name + "_bucket", "{" + labels + (", " if labels else "") + f'le="{float(le)}"' + "}")] = float(
            sum(v <= le for v in values))
    out[(name + "_bucket", "{" + labels + (", " if labels else "") + 'le="+Inf"}')] = float(len(values))
    out[(name + "_count", "{" + labels + "}" if labels else "")] = float(len(values))
    out[(name + "_sum", "{" + labels + "}" if labels else "")] = float(sum(values))
    return out


def metrics(lat, phone_gaps, web_gaps, received, dropped):
    m = {("pair_vision_frames_received_total", '{channel="phone"}'): received,
         ("pair_vision_frames_processed_total", '{channel="phone"}'): received - dropped,
         ("pair_vision_frames_dropped_total", '{reason="replaced"}'): dropped,
         ("pair_vision_events_total", '{type="HANDS_OUT",final="false"}'): 1.0}
    m |= hist("pair_vision_frame_latency_seconds", lat, LAT_LE)
    m |= hist("pair_vision_processed_gap_seconds", phone_gaps, GAP_LE, 'channel="phone"')
    m |= hist("pair_vision_processed_gap_seconds", web_gaps, GAP_LE, 'channel="webcam"')
    return m


def run_judge(after, sink=3, require_events=True):
    args = SimpleNamespace(students=20, fps=1.0, warmup=60, allow_cpu=False, require_events=require_events)
    sampler = SimpleNamespace(gpu=[50.0], mem=[1000.0], load=[1.0])
    ready = {"gpu": True, "modelsLoaded": True}
    return lt.judge(args, ready, {}, after, after, 1200.0, sampler, {}, 0, sink)


def test_parse_and_quantile_interpolates_inside_bucket():
    m = lt.parse_metrics('a_bucket{le="1.0"} 50\na_bucket{le="2.0"} 100\na_bucket{le="+Inf"} 100\n# c\n')
    bs = lt.buckets(m, {}, "a")
    assert bs == [(1.0, 50.0), (2.0, 100.0), (float("inf"), 100.0)]
    assert abs(lt.quantile(bs, 0.95) - 1.9) < 1e-9


def test_judge_passes_when_all_three_criteria_hold():
    rep = run_judge(metrics([0.5] * 990 + [2.9] * 10, [1.0] * 1000, [1.0] * 1000, 2000.0, 10.0))
    assert rep["verdict"] == "합격" and rep["passed"] and rep["valid"]


def test_judge_fails_each_criterion():
    over_drop = run_judge(metrics([0.5] * 1000, [1.0] * 1000, [1.0] * 1000, 2000.0, 30.0))  # 1.5% 버림
    assert not over_drop["criteria"]["drop"]["ok"] and over_drop["verdict"] == "불합격"
    slow = run_judge(metrics([0.5] * 900 + [3.5] * 100, [1.0] * 1000, [1.0] * 1000, 2000.0, 0.0))  # 3초 안 90%
    assert not slow["criteria"]["latency_p95"]["ok"]
    gaps = run_judge(metrics([0.5] * 1000, [1.0] * 1000, [1.0] * 980 + [2.8] * 20, 2000.0, 0.0))  # 웹캠 2% > 2.5초
    assert not gaps["criteria"]["gap_over"]["ok"]
    assert gaps["criteria"]["gap_over"]["by_channel"]["phone"]["over_ratio"] == 0.0  # 2.8초는 휴대폰 한도(3초) 안


def test_missing_callbacks_make_the_verdict_invalid():
    rep = run_judge(metrics([0.5] * 1000, [1.0] * 1000, [1.0] * 1000, 2000.0, 0.0), sink=0)
    assert rep["passed"] and not rep["valid"] and rep["verdict"].startswith("판정 무효")


def test_main_run_without_events_is_a_criteria_failure_not_invalid():
    m = metrics([0.5] * 1000, [1.0] * 1000, [1.0] * 1000, 2000.0, 0.0)
    m[("pair_vision_events_total", '{type="HANDS_OUT",final="false"}')] = 0.0
    assert not run_judge(m, sink=0, require_events=True)["valid"]  # 짧은 확인: 콜백을 확인 못 했으니 무효
    assert run_judge(m, sink=0, require_events=False)["valid"]  # 본 시험: 이벤트가 없을 뿐 판정은 유효
