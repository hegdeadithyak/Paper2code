"""
Tests for SSD scratch components.
"""

import math
import pytest
import torch

from ssd_scratch import (
    default_box_sizes,
    generate_default_boxes_for_map,
    generate_default_boxes,
    cxcywh_to_xyxy,
    xyxy_to_cxcywh,
    hard_negative_mining,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ---------- scale schedule ----------

def test_scale_schedule_endpoints():
    scales, _ = default_box_sizes(num_maps=6, s_min=0.2, s_max=0.9)
    assert abs(scales[0] - 0.2) < 1e-6
    assert abs(scales[-1] - 0.9) < 1e-6


def test_scale_schedule_monotonic():
    scales, _ = default_box_sizes(num_maps=6)
    assert all(scales[i] < scales[i + 1] for i in range(5))


def test_s_prime_geometric_mean():
    """s'_k should be sqrt(s_k * s_{k+1})."""
    scales, s_primes = default_box_sizes(num_maps=6)
    for k in range(5):
        expected = math.sqrt(scales[k] * scales[k + 1])
        assert abs(s_primes[k] - expected) < 1e-5


# ---------- default boxes per map ----------

def test_default_boxes_per_cell_count():
    # 3 aspect ratios + 1 extra square → 4 boxes per cell
    boxes = generate_default_boxes_for_map((3, 3), s_k=0.2, s_prime=0.25,
                                           aspect_ratios=(1.0, 2.0, 0.5))
    assert boxes.shape == (3 * 3 * 4, 4)


def test_default_boxes_centered_on_grid():
    boxes = generate_default_boxes_for_map((2, 2), s_k=0.2, s_prime=0.25,
                                           aspect_ratios=(1.0,))
    # 2 ar-boxes + 1 extra = 2 boxes per cell
    # first two cells: row 0, cols 0 and 1 → centers (0.25, 0.25) and (0.75, 0.25)
    assert abs(boxes[0, 0].item() - 0.25) < 1e-5
    assert abs(boxes[0, 1].item() - 0.25) < 1e-5
    assert abs(boxes[2, 0].item() - 0.75) < 1e-5


def test_full_pyramid_box_count():
    """SSD300 paper: 38×38×4 + 19×19×6 + 10×10×6 + 5×5×6 + 3×3×4 + 1×1×4 = 8732.
    Our default config uses 3 aspect ratios + 1 extra → 4 per cell, uniform.
    So: 38² + 19² + 10² + 5² + 3² + 1² = 1940 cells × 4 = 7760."""
    sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    boxes = generate_default_boxes(sizes, aspect_ratios=(1.0, 2.0, 0.5))
    total_cells = sum(h * w for h, w in sizes)
    assert boxes.shape == (total_cells * 4, 4)


# ---------- format conversion ----------

def test_cxcywh_xyxy_roundtrip():
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.1, 0.3, 0.05, 0.1]])
    assert torch.allclose(xyxy_to_cxcywh(cxcywh_to_xyxy(boxes)), boxes, atol=1e-6)


def test_cxcywh_to_xyxy_correctness():
    box = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
    out = cxcywh_to_xyxy(box)
    assert torch.allclose(out, torch.tensor([[0.4, 0.3, 0.6, 0.7]]), atol=1e-6)


# ---------- hard negative mining ----------

def test_hnm_ratio_respected():
    losses = torch.randn(100).abs()
    pos = torch.zeros(100, dtype=torch.bool)
    pos[:5] = True
    keep = hard_negative_mining(losses, pos, neg_pos_ratio=3)
    # should keep 5 positives + 15 negatives = 20
    assert keep.sum().item() == 20


def test_hnm_picks_hardest_negatives():
    """The kept negatives should be the ones with highest loss."""
    losses = torch.arange(10, dtype=torch.float32)
    pos = torch.zeros(10, dtype=torch.bool)
    pos[0] = True   # 1 positive
    keep = hard_negative_mining(losses, pos, neg_pos_ratio=3)
    # should keep idx 0 (pos) + top-3 negatives by loss → idx 7, 8, 9
    kept_idx = set(keep.nonzero().flatten().tolist())
    assert kept_idx == {0, 7, 8, 9}


def test_hnm_cap_at_available_negatives():
    losses = torch.randn(10).abs()
    pos = torch.zeros(10, dtype=torch.bool)
    pos[:5] = True
    # 5 positives × 3 = 15 desired, but only 5 negatives available
    keep = hard_negative_mining(losses, pos, neg_pos_ratio=3)
    assert keep.sum().item() == 10   # all kept
