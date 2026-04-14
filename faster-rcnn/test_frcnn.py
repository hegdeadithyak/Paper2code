"""
Tests for the Faster R-CNN scratch components: anchors, box encode/decode, NMS.
"""

import pytest
import torch

from frcnn_scratch import (
    generate_anchor_base,
    generate_anchors,
    encode_boxes,
    decode_boxes,
    box_iou,
    nms,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ---------- anchors ----------

def test_anchor_base_count():
    base = generate_anchor_base(scales=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0))
    assert base.shape == (9, 4)


def test_anchor_base_centered_at_origin():
    base = generate_anchor_base()
    centers = (base[:, :2] + base[:, 2:]) / 2
    assert torch.allclose(centers, torch.zeros_like(centers), atol=1e-5)


def test_anchor_base_aspect_ratios():
    """Ratio 0.5 → width 2x height (wider); ratio 2.0 → height 2x width (taller)."""
    base = generate_anchor_base(scales=(100,), aspect_ratios=(0.5, 1.0, 2.0))
    for i, r in enumerate([0.5, 1.0, 2.0]):
        w = base[i, 2] - base[i, 0]
        h = base[i, 3] - base[i, 1]
        # h/w should equal r
        assert abs((h / w).item() - r) < 1e-4


def test_anchor_tiling_shape():
    base = generate_anchor_base(scales=(32,), aspect_ratios=(1.0,))
    H, W = 4, 5
    anchors = generate_anchors((H, W), stride=16, anchor_base=base)
    assert anchors.shape == (H * W * 1, 4)


def test_anchor_tiling_centers_on_grid():
    base = generate_anchor_base(scales=(32,), aspect_ratios=(1.0,))
    anchors = generate_anchors((2, 2), stride=16, anchor_base=base)
    centers = (anchors[:, :2] + anchors[:, 2:]) / 2
    # Should be at stride/2 + k*stride for k=0,1,...
    expected = torch.tensor([[8., 8.], [24., 8.], [8., 24.], [24., 24.]])
    assert torch.allclose(centers, expected, atol=1e-4)


# ---------- box encode/decode roundtrip ----------

def test_encode_decode_roundtrip():
    anchors = torch.tensor([[10, 10, 30, 30], [50, 50, 100, 100]], dtype=torch.float32)
    gt = torch.tensor([[12, 14, 32, 34], [55, 55, 95, 110]], dtype=torch.float32)
    deltas = encode_boxes(anchors, gt)
    recovered = decode_boxes(anchors, deltas)
    assert torch.allclose(recovered, gt, atol=1e-4)


def test_zero_deltas_return_anchors():
    anchors = torch.tensor([[0, 0, 10, 10], [20, 20, 40, 40]], dtype=torch.float32)
    deltas = torch.zeros(2, 4)
    assert torch.allclose(decode_boxes(anchors, deltas), anchors, atol=1e-4)


# ---------- IoU ----------

def test_iou_identical_boxes():
    boxes = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    assert abs(box_iou(boxes, boxes)[0, 0].item() - 1.0) < 1e-6


def test_iou_disjoint():
    a = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    b = torch.tensor([[100, 100, 110, 110]], dtype=torch.float32)
    assert box_iou(a, b)[0, 0].item() == 0.0


def test_iou_half_overlap():
    a = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    b = torch.tensor([[5, 0, 15, 10]], dtype=torch.float32)
    # intersection = 5*10=50, union = 100+100-50 = 150 → iou = 1/3
    iou = box_iou(a, b)[0, 0].item()
    assert abs(iou - 1 / 3) < 1e-5


# ---------- NMS ----------

def test_nms_keeps_highest():
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32)
    scores = torch.tensor([0.5, 0.9])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [1]


def test_nms_keeps_disjoint():
    boxes = torch.tensor([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8])
    keep = sorted(nms(boxes, scores, 0.5).tolist())
    assert keep == [0, 1]


def test_nms_matches_torchvision():
    from torchvision.ops import nms as tv_nms
    boxes = torch.rand(50, 4) * 100
    boxes[:, 2:] = boxes[:, :2] + torch.rand(50, 2) * 50 + 1
    scores = torch.rand(50)
    ours = sorted(nms(boxes, scores, 0.5).tolist())
    theirs = sorted(tv_nms(boxes, scores, 0.5).tolist())
    assert ours == theirs


def test_nms_empty():
    keep = nms(torch.empty(0, 4), torch.empty(0))
    assert keep.numel() == 0
