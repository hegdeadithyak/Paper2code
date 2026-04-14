"""
Faster R-CNN via torchvision.

torchvision.models.detection.fasterrcnn_resnet50_fpn wires together:
  - ResNet-50 + FPN backbone (multi-scale feature pyramid)
  - RPN with anchors/NMS (what frcnn_scratch.py builds from atoms)
  - RoI Align + two-stage head
  - Multi-task loss (box cls + box reg + mask if enabled)

Running this without pretrained weights first (they download ~170MB). Pass
pretrained=True to grab the COCO-pretrained weights.
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.ops import nms as torchvision_nms


def make_faster_rcnn(pretrained=False, num_classes=91):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    return fasterrcnn_resnet50_fpn_v2(weights=weights, num_classes=num_classes if not pretrained else None)


def nms_library(boxes, scores, iou_threshold=0.5):
    """torchvision's NMS — the reference we test our scratch NMS against."""
    return torchvision_nms(boxes, scores, iou_threshold)


if __name__ == "__main__":
    # Smoke test the whole two-stage pipeline on a tiny random input.
    model = make_faster_rcnn(pretrained=False, num_classes=10)
    model.eval()
    with torch.no_grad():
        images = [torch.rand(3, 256, 256)]
        out = model(images)
    print(f"detections on random noise: {len(out[0]['boxes'])} boxes")
