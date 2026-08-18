"""One evaluator for every method.

Two numbers, because they answer different questions:

  mAP              is the box right?
  centre-coverage  did we point at the object at all?

They diverge sharply here — fire scores 0.775 centre-coverage against AP 0.05,
i.e. Claude finds three fires in four and boxes almost none of them tightly.
For a platform that flies to a coordinate, the second number is the
operationally relevant one, and no other benchmark reports it.
"""
from collections import Counter

import torch
from torchmetrics.detection import MeanAveragePrecision

from src.constants import CLS, NAMES, W, H

# 3 prompts x 100 cached queries = up to 300 detections/frame for SAM3.
# torchmetrics defaults to [1, 10, 100] and SILENTLY DROPS the lowest-scoring
# excess, biasing AP downward. Every method must use the same cap or the
# comparison is invalid.
MAX_DETS = [1, 10, 300]


def _norm(dets):
    """Detection objects or dicts -> [(label, x1, y1, x2, y2, score)], clamped."""
    out = []
    for d in dets:
        if isinstance(d, dict):
            label, bx, sc = d['label'], d['box'], d.get('confidence', 1.0)
        else:
            label, bx, sc = d.label, d.box, getattr(d, 'confidence', 1.0)
        x1, y1, x2, y2 = bx
        x1, x2 = max(0., min(float(x1), W)), max(0., min(float(x2), W))
        y1, y2 = max(0., min(float(y1), H)), max(0., min(float(y2), H))
        if x2 - x1 < 1 or y2 - y1 < 1:          # drop degenerate boxes
            continue
        out.append((label, x1, y1, x2, y2, float(sc)))
    return out


def score(results, meta, load_gt, backend='faster_coco_eval'):
    """results: {key: [Detection|dict]}   meta: {key: (stem, split)}"""
    metric = MeanAveragePrecision(
        box_format='xyxy', iou_type='bbox', class_metrics=True,
        backend=backend, max_detection_thresholds=MAX_DETS,
    )
    metric.warn_on_many_detections = False
    hit, tot = Counter(), Counter()

    for key, dets in results.items():
        stem, split = meta[key]
        gt = load_gt(stem, split)
        pred = _norm(dets)

        for gl, gx1, gy1, gx2, gy2 in gt:
            cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
            tot[gl] += 1
            hit[gl] += any(pl == gl and x1 <= cx <= x2 and y1 <= cy <= y2
                           for pl, x1, y1, x2, y2, _ in pred)

        metric.update(
            [dict(
                boxes=torch.tensor([p[1:5] for p in pred],
                                   dtype=torch.float32).reshape(-1, 4),
                scores=torch.tensor([p[5] for p in pred], dtype=torch.float32),
                labels=torch.tensor([CLS[p[0]] for p in pred], dtype=torch.int64),
            )],
            [dict(
                boxes=torch.tensor([g[1:5] for g in gt],
                                   dtype=torch.float32).reshape(-1, 4),
                labels=torch.tensor([CLS[g[0]] for g in gt], dtype=torch.int64),
            )],
        )

    m = metric.compute()
    per = {}
    for cid, ap in zip(m['classes'].tolist(), m['map_per_class'].tolist()):
        c = NAMES[int(cid)]
        per[c] = dict(ap=float(ap), centre_hit=hit[c], centre_total=tot[c],
                      centre_cov=hit[c] / tot[c] if tot[c] else float('nan'))

    # overriding max_detection_thresholds renames the recall keys:
    # torchmetrics emits mar_1 / mar_10 / mar_300, and there is no mar_100.
    return dict(frames=len(results), map_50=float(m['map_50']),
                map=float(m['map']), mar=float(m[f'mar_{MAX_DETS[-1]}']),
                per_class=per)


def report(r, label=''):
    print(f"{label}  frames={r['frames']}")
    print(f"  mAP@50     {r['map_50']:.4f}")
    print(f"  mAP@50:95  {r['map']:.4f}")
    print(f"  mAR@{MAX_DETS[-1]}    {r['mar']:.4f}")
    for c, v in r['per_class'].items():
        # torchmetrics returns -1 as a sentinel for "no GT of this class"
        ap = 'n/a' if v['ap'] < 0 else f"{v['ap']:.4f}"
        cov = 'n/a' if not v['centre_total'] else f"{v['centre_cov']:.3f}"
        print(f"    {c:<7} AP {ap:>7}   centre-cov "
              f"{v['centre_hit']}/{v['centre_total']} = {cov}")