"""
SSD via torchvision.

torchvision.models.detection.ssd300_vgg16 wires:
  - VGG16 backbone (truncated at conv5)
  - Extra conv layers producing a pyramid at 38x38, 19x19, 10x10, 5x5, 3x3, 1x1
  - Per-scale classification + box regression heads
  - Default-box generation + hard negative mining during training

The paper calls this SSD-VGG. "SSD-ResNet" in this repo folder is just SSD
with a ResNet backbone — torchvision doesn't ship that combo directly; you'd
build it by swapping the backbone inside the detection-model factory.
"""

import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights


def make_ssd(pretrained=False, num_classes=91):
    weights = SSD300_VGG16_Weights.DEFAULT if pretrained else None
    model = ssd300_vgg16(weights=weights, num_classes=num_classes if not pretrained else None)
    return model


if __name__ == "__main__":
    model = make_ssd(pretrained=False, num_classes=10)
    model.eval()
    with torch.no_grad():
        out = model([torch.rand(3, 300, 300)])
    print(f"detections on random noise: {len(out[0]['boxes'])} boxes")
