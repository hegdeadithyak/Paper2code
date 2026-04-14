"""
Faster R-CNN — key components from scratch.

Ren et al., 2015 — the paper that made two-stage detection end-to-end trainable.

A full Faster R-CNN is a few thousand lines of code (backbone + RPN + ROI
pooling + detection head + losses + training loop + multi-dataset handling).
This file focuses on the three novel algorithms the paper contributed, which
are the interesting bits to build from first principles:

    1. Anchor generation        — a grid of reference boxes at each feature cell
    2. Box decoding             — from "tx,ty,tw,th" deltas back to (x1,y1,x2,y2)
    3. Non-maximum suppression  — dedupe overlapping predictions

These are the atoms that both RPN (region proposal network) and the final
detection head are built out of. torchvision's fasterrcnn_resnet50_fpn wires
these together with a ResNet+FPN backbone; see frcnn_library.py for that path.
"""

import math
import torch


# ---------- anchors ----------

def generate_anchor_base(scales=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
    """
    Generate a set of reference anchor boxes centered at (0, 0).
    Returns [A, 4] in (x1, y1, x2, y2) format where A = len(scales)*len(ratios).
    """
    anchors = []
    for s in scales:
        for r in aspect_ratios:
            w = s * math.sqrt(1.0 / r)
            h = s * math.sqrt(r)
            anchors.append([-w / 2, -h / 2, w / 2, h / 2])
    return torch.tensor(anchors, dtype=torch.float32)


def generate_anchors(feature_hw, stride, anchor_base):
    """
    Tile anchor_base over a feature map of shape (H, W).
    Args:
        feature_hw : (H, W) spatial size of the feature map
        stride     : pixel stride between adjacent feature cells
        anchor_base: [A, 4] reference anchors around origin
    Returns:
        [H*W*A, 4] anchors in image coords, (x1,y1,x2,y2).
    """
    H, W = feature_hw
    A = anchor_base.size(0)

    # Centers of each feature cell, in image coordinates.
    shift_x = (torch.arange(W, dtype=torch.float32) + 0.5) * stride
    shift_y = (torch.arange(H, dtype=torch.float32) + 0.5) * stride
    sy, sx = torch.meshgrid(shift_y, shift_x, indexing="ij")
    shifts = torch.stack([sx, sy, sx, sy], dim=-1).view(-1, 1, 4)   # [H*W, 1, 4]

    anchors = anchor_base.view(1, A, 4) + shifts                    # [H*W, A, 4]
    return anchors.view(-1, 4)


# ---------- box ops ----------

def decode_boxes(anchors, deltas):
    """
    Convert (tx, ty, tw, th) regression targets back to (x1, y1, x2, y2).
    This is the inverse of what the network regressed:
        tx = (gx - ax) / aw,   ty = (gy - ay) / ah
        tw = log(gw / aw),     th = log(gh / ah)
    Args:
        anchors: [N, 4] (x1,y1,x2,y2)
        deltas : [N, 4] (tx,ty,tw,th)
    """
    ax = (anchors[:, 0] + anchors[:, 2]) / 2
    ay = (anchors[:, 1] + anchors[:, 3]) / 2
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]

    gx = deltas[:, 0] * aw + ax
    gy = deltas[:, 1] * ah + ay
    gw = torch.exp(deltas[:, 2].clamp(max=math.log(1000.0))) * aw
    gh = torch.exp(deltas[:, 3].clamp(max=math.log(1000.0))) * ah

    return torch.stack([gx - gw / 2, gy - gh / 2, gx + gw / 2, gy + gh / 2], dim=1)


def encode_boxes(anchors, gt_boxes):
    """Inverse of decode_boxes — what the network should regress to."""
    ax = (anchors[:, 0] + anchors[:, 2]) / 2
    ay = (anchors[:, 1] + anchors[:, 3]) / 2
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]

    gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]

    return torch.stack([
        (gx - ax) / aw,
        (gy - ay) / ah,
        torch.log(gw / aw),
        torch.log(gh / ah),
    ], dim=1)


def box_iou(a, b):
    """Pairwise IoU. a: [N,4], b: [M,4] -> [N, M]."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


# ---------- NMS ----------

def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-maximum suppression. Greedy: pick the highest-scoring box, suppress
    everything that overlaps it by more than iou_threshold, repeat.
    Returns indices into boxes that survive.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i:i+1], boxes[order[1:]])[0]
        mask = ious <= iou_threshold
        order = order[1:][mask]
    return torch.tensor(keep, dtype=torch.long)


if __name__ == "__main__":
    base = generate_anchor_base(scales=(32, 64), aspect_ratios=(0.5, 1.0, 2.0))
    anchors = generate_anchors((8, 8), stride=16, anchor_base=base)
    print(f"{base.size(0)} base anchors × 8×8 cells = {anchors.size(0)} total")

    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.95])
    print(f"NMS kept: {nms(boxes, scores, 0.5).tolist()}  (expect [2, 0] — box 1 suppressed by 0)")
