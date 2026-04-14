"""
SSD (Single Shot MultiBox Detector) — key components from scratch.
Liu et al., 2016. SSD-ResNet is just SSD with ResNet swapped in for VGG.

SSD's one big idea: skip the region-proposal stage. Instead, predict class +
box offsets directly at every location on *multiple* feature maps (different
scales). Because a cat at scale 1 looks like a kitten at scale 2, running
detection at several pyramid levels naturally handles objects of different
sizes. The network is "single-shot" — one forward pass, no second stage.

The novel algorithmic contributions:
    1. Default boxes     — like anchors but laid out across a feature pyramid
    2. Multi-scale heads — one classifier + regressor per pyramid level
    3. Hard negative mining — SSD's answer to massive class imbalance
    4. Matching strategy  — which default box "owns" which ground truth

This file implements 1, 2's default-box schedule, and 3. Matching uses the
same IoU logic as Faster R-CNN (see frcnn_scratch.box_iou).
"""

import math
import torch


# ---------- default boxes ----------

def default_box_sizes(num_maps, s_min=0.2, s_max=0.9):
    """
    SSD paper's schedule: linearly interpolate scale s_k from s_min at the
    first (largest) feature map to s_max at the last (smallest).
    Returns s_k, s'_k for k=1..num_maps.
    s_k is the base scale; s'_k = sqrt(s_k * s_{k+1}) is an extra square box.
    """
    scales = []
    for k in range(1, num_maps + 1):
        s_k = s_min + (s_max - s_min) * (k - 1) / (num_maps - 1)
        scales.append(s_k)
    # s'_k using next scale; for the last map, extrapolate
    s_primes = []
    for k in range(num_maps):
        next_s = scales[k + 1] if k + 1 < num_maps else s_max + (s_max - scales[-1])
        s_primes.append(math.sqrt(scales[k] * next_s))
    return scales, s_primes


def generate_default_boxes_for_map(feature_hw, s_k, s_prime, aspect_ratios=(1.0, 2.0, 0.5, 3.0, 1.0/3)):
    """
    For a single feature map of shape (H, W), generate default boxes at each
    cell. Returns [H*W*num_boxes_per_cell, 4] in (cx, cy, w, h) normalized coords.

    Per the SSD paper: at each cell we produce
        - one square box of scale s_k for each aspect ratio
        - one extra square box of scale s'_k (ratio=1)
    """
    H, W = feature_hw
    boxes = []

    # cell centers in [0, 1] coordinates
    for i in range(H):
        for j in range(W):
            cx = (j + 0.5) / W
            cy = (i + 0.5) / H
            # aspect-ratio boxes at scale s_k
            for ar in aspect_ratios:
                w = s_k * math.sqrt(ar)
                h = s_k / math.sqrt(ar)
                boxes.append([cx, cy, w, h])
            # extra square box at scale s'_k
            boxes.append([cx, cy, s_prime, s_prime])

    return torch.tensor(boxes, dtype=torch.float32)


def generate_default_boxes(feature_map_sizes, s_min=0.2, s_max=0.9, aspect_ratios=None):
    """
    Generate default boxes across all feature map levels.
    Returns [total_boxes, 4] in (cx, cy, w, h) normalized coords.
    """
    scales, s_primes = default_box_sizes(len(feature_map_sizes), s_min, s_max)
    all_boxes = []
    default_ars = aspect_ratios or (1.0, 2.0, 0.5)
    for hw, s_k, s_p in zip(feature_map_sizes, scales, s_primes):
        all_boxes.append(generate_default_boxes_for_map(hw, s_k, s_p, default_ars))
    return torch.cat(all_boxes, dim=0)


def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


# ---------- hard negative mining ----------

def hard_negative_mining(conf_loss_per_box, positive_mask, neg_pos_ratio=3):
    """
    SSD has way more negative (background) default boxes than positive ones.
    Training on all of them → everything collapses to "predict background."
    Solution: for each sample, keep all positives and the top-K hardest
    (highest-loss) negatives, where K = neg_pos_ratio * num_positives.

    Args:
        conf_loss_per_box : [N] per-box classification loss
        positive_mask     : [N] bool, True where the box matched a GT
        neg_pos_ratio     : keep this many negatives per positive (paper: 3)
    Returns:
        [N] bool mask — True for boxes to include in the loss.
    """
    num_pos = int(positive_mask.sum().item())
    if num_pos == 0:
        # nothing to match against — fall back to top few negatives
        num_neg = min(3, conf_loss_per_box.size(0))
    else:
        num_neg = min(num_pos * neg_pos_ratio, (~positive_mask).sum().item())

    # among the negatives, pick the top num_neg by loss
    neg_losses = conf_loss_per_box.clone()
    neg_losses[positive_mask] = -float("inf")   # positives can't win in neg ranking
    _, neg_idx = neg_losses.topk(num_neg)
    neg_mask = torch.zeros_like(positive_mask)
    neg_mask[neg_idx] = True

    return positive_mask | neg_mask


if __name__ == "__main__":
    # SSD300 paper's feature map schedule: 38, 19, 10, 5, 3, 1
    sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    boxes = generate_default_boxes(sizes)
    print(f"{boxes.size(0)} default boxes across {len(sizes)} feature maps")

    # hard negative mining demo
    losses = torch.randn(20).abs()
    pos = torch.zeros(20, dtype=torch.bool)
    pos[:3] = True
    keep = hard_negative_mining(losses, pos, neg_pos_ratio=3)
    print(f"positives: 3, total kept: {keep.sum().item()} (expected 3 + 3*3 = 12)")
