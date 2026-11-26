"""GPU 묶음 추론: CPU 에서 "여러 장을 한 번에 넣은 결과 = 한 장씩 넣은 결과"인지(허용 오차 안) 확인한다.

가중치를 받지 않도록 작은 D-FINE 을 무작위 초기화로 만든다(값은 의미 없고 배치 동치성만 본다). torch·transformers 가 없으면 건너뛴다.
OWLv2 는 토크나이저 파일이 필요해 이 시험에 넣지 못했다(RunPod/실제 모델 시험에서 확인).
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from pair_vision.models.options import RunOptions  # noqa: E402
from pair_vision.models.real import RealDetector  # noqa: E402
from pair_vision.rules.config import ModelRules  # noqa: E402


@pytest.fixture(scope="module")
def detector():
    from transformers import DFineConfig, DFineForObjectDetection, RTDetrImageProcessor

    torch.manual_seed(0)
    cfg = DFineConfig(
        num_labels=91, encoder_hidden_dim=32, encoder_in_channels=[64, 128, 256], d_model=32,
        decoder_in_channels=[32, 32, 32], num_queries=20, decoder_layers=2, encoder_layers=1,
        encoder_ffn_dim=64, decoder_ffn_dim=64, num_feature_levels=3, decoder_attention_heads=2,
        encoder_attention_heads=2, feat_strides=[8, 16, 32], hidden_expansion=0.5,
        decoder_n_points=[2, 2, 2], anchor_image_size=None,
    )
    det = RealDetector("unused", ModelRules(), verify=False, options=RunOptions(batch_gpu=True))
    det._torch, det._device = torch, "cpu"
    det._dfine = DFineForObjectDetection(cfg).eval()
    det._dfine_proc = RTDetrImageProcessor(size={"height": 640, "width": 640})
    return det


def images():
    rng = np.random.default_rng(1)
    sizes = [(400, 300), (640, 480), (320, 320)]  # 크기가 서로 달라도 한 묶음으로 들어간다
    return [Image.fromarray((rng.random((h, w, 3)) * 255).astype("uint8")) for w, h in sizes]


def test_batch_matches_single_frames(detector):
    ims = images()
    batch = detector._dfine_raw(ims, floor=0.0)
    singles = [detector._dfine_raw([im], floor=0.0)[0] for im in ims]
    assert [len(b) for b in batch] == [len(s) for s in singles] and all(batch)
    for b_img, s_img in zip(batch, singles, strict=True):
        for (bl, bs, bb), (sl, ss, sb) in zip(b_img, s_img, strict=True):
            assert bl == sl
            assert bs == pytest.approx(ss, abs=1e-4)
            for x, y in zip((bb.x0, bb.y0, bb.x1, bb.y1), (sb.x0, sb.y0, sb.x1, sb.y1), strict=True):
                assert x == pytest.approx(y, abs=1e-4)


def test_stage_stats_count_frames_and_reset(detector):
    detector.stage_stats(reset=True)
    detector._dfine_raw(images(), floor=0.5)
    n, ms = detector.stage_stats()["dfine"]
    assert n == 3 and ms > 0
    assert detector.stage_stats(reset=True) and detector.stage_stats() == {}
