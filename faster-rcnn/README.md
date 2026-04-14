# Faster R-CNN

> Ren et al., *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*, NeurIPS 2015.

The paper that killed Selective Search and made two-stage detection end-to-end trainable. Before this, object detection pipelines were held together with shell scripts. After this, the RPN became the template every detector copied for five years.

<p align="center">
  <img src="./faster_rcnn.svg" alt="Faster R-CNN architecture" width="620" />
</p>

<sub><i>Image: Wikimedia Commons, CC BY-SA 4.0.</i></sub>

## Honest disclaimer

A full Faster R-CNN is **several thousand lines of code** and takes days to train. Reimplementing all of it from scratch in a learning repo would produce a subtly-broken toy you can't trust. Instead, this folder does two things:

- **`frcnn_scratch.py`** — the three *novel algorithmic* contributions of the paper, implemented from first principles with zero abstractions: **anchor generation**, **box encode/decode**, **non-maximum suppression**. These are the atoms everything else is built from.
- **`frcnn_library.py`** — the full production pipeline via `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`.

If you understand the three algorithms below, you understand everything the paper contributed. The rest is ResNet backbone + ROI align (which is a 2017 paper, not this one) + standard multi-task training.

## The three atoms

### 1. Anchors — a structured guess

Naively, "where might an object be?" has infinite answers (any x, y, w, h). The RPN trick: predefine a discrete grid of reference boxes — at every spatial location, place `A` boxes of fixed (scale, aspect ratio) combinations. Typical config: 3 scales × 3 ratios = **9 anchors per cell**. On a 38×50 feature map, that's 17,100 anchors covering the image. Now the network only has to output refinements relative to these, not raw coordinates.

### 2. Box encode / decode — relative targets

The network regresses four numbers per anchor: `(tx, ty, tw, th)`. These are **deltas** in a smart log-scale parameterization:

```
tx = (gx - ax) / aw        ty = (gy - ay) / ah
tw = log(gw / aw)          th = log(gh / ah)
```

Where `a` is the anchor, `g` is the ground truth. At inference, the network outputs `(tx,ty,tw,th)` and you invert these equations to get `(x1,y1,x2,y2)`. Regressing deltas instead of absolute coords means the network's output distribution is centered on 0 regardless of image size — much easier to learn.

### 3. Non-maximum suppression — dedupe

One object will fire multiple nearby anchors with high scores. NMS greedy-picks the highest-scoring box and suppresses any box overlapping it by more than `iou_threshold`, then repeats. Simple algorithm, ~10 lines, but it's what makes a detection output look like "one box per object" instead of "a cloud of boxes near every object."

## Files

| File | What |
|---|---|
| `frcnn_scratch.py` | Anchor generation, box encode/decode, IoU, NMS. Pure tensor ops, ~150 lines. |
| `frcnn_library.py` | `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2` — full pipeline. Smoke test runs on random input without pretrained weights. |
| `test_frcnn.py` | 14 tests: anchor shapes + centering + aspect ratios, encode/decode roundtrip, IoU edge cases, NMS matches `torchvision.ops.nms` on random inputs. |

## Run it

```bash
python3 frcnn_scratch.py
python3 -m pytest test_frcnn.py -v -p no:anyio
# full pipeline smoke test (no pretrained weights, so no download):
python3 frcnn_library.py
```

## What to notice

- Our scratch NMS produces the **exact same indices** as `torchvision.ops.nms` on 50 random boxes. The algorithm really is that simple.
- The encode/decode roundtrip is exact to `1e-4`. If you've ever been confused by `deltas * aw + ax` code in a detector, this test makes it obvious: it's just the inverse of the encoding formula.
- The 9 anchors at 3 scales × 3 ratios: you can see in `test_anchor_base_aspect_ratios` that ratio 2.0 produces a tall box, 0.5 a wide one, 1.0 square. Not arbitrary — those cover most real object aspect ratios.

## References

- Ren, He, Girshick, Sun — [Faster R-CNN](https://arxiv.org/abs/1506.01497)
- Girshick — [Fast R-CNN](https://arxiv.org/abs/1504.08083) (the predecessor this paper fixes)
- He et al. — [Mask R-CNN](https://arxiv.org/abs/1703.06870) (the successor with RoI Align + a mask head)
