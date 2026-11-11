"""Checks for the detection and segmentation exercise.

Plausible wrong implementations these checks catch:

- box_iou that forgets to clamp the intersection width and height at zero.
  Two boxes that are apart in both x and y get a product of two negative
  numbers, which is a positive "intersection". The diagonal-boxes check
  catches it.
- nms that compares a box with every higher-scored box, including the ones
  already suppressed, instead of only the kept ones. The worked NMS example
  catches it: box 2 overlaps only a suppressed box.
- match_detections that lets two detections claim one ground-truth box. The
  duplicate in the hand-built example catches it.
"""

import numpy as np

from mlcheck import almost, check, close, equal, fail, finite_difference, has, shape


def random_boxes(rng, n, spread=20.0, size=8.0):
    xy = rng.uniform(0, spread, size=(n, 2))
    wh = rng.uniform(1.0, size, size=(n, 2))
    return np.concatenate([xy, xy + wh], axis=1)


def loop_iou(a, b):
    out = np.zeros((len(a), len(b)))
    for i, p in enumerate(a):
        for j, q in enumerate(b):
            w = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
            h = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
            inter = w * h
            union = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
            out[i, j] = inter / union
    return out


NMS_BOXES = np.array(
    [
        [0.0, 0.0, 10.0, 10.0],
        [3.0, 0.0, 13.0, 10.0],
        [6.0, 0.0, 16.0, 10.0],
        [30.0, 0.0, 40.0, 10.0],
    ]
)
NMS_SCORES = np.array([0.9, 0.8, 0.7, 0.6])

GT_BOXES = np.array(
    [[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0], [40.0, 0.0, 50.0, 10.0]]
)
DET_BOXES = np.array(
    [
        [1.0, 0.0, 11.0, 10.0],
        [21.0, 1.0, 31.0, 11.0],
        [0.0, 1.0, 10.0, 11.0],
        [60.0, 0.0, 70.0, 10.0],
        [42.0, 0.0, 52.0, 10.0],
        [22.0, 0.0, 32.0, 10.0],
    ]
)
DET_SCORES = np.array([0.95, 0.90, 0.80, 0.70, 0.60, 0.50])


@check("box_iou reproduces the hand examples: 1/7 for a partial overlap, 1 for identical, 0.25 for containment")
def test_iou_hand(sub):
    iou = has(sub, "box_iou")
    a = np.array([[0.0, 0.0, 4.0, 4.0]])
    b = np.array([[2.0, 2.0, 6.0, 6.0], [0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 3.0, 3.0]])
    got = iou(a, b)
    shape(got, (1, 3), label="box_iou output")
    close(got, np.array([[1.0 / 7.0, 1.0, 0.25]]), atol=1e-12, label="IoU of [0,0,4,4] with three boxes")


@check("box_iou gives exactly 0 for boxes that are apart, including boxes apart in both x and y")
def test_iou_disjoint(sub):
    iou = has(sub, "box_iou")
    a = np.array([[0.0, 0.0, 2.0, 2.0]])
    b = np.array(
        [
            [5.0, 0.0, 7.0, 2.0],  # apart in x only
            [0.0, 5.0, 2.0, 7.0],  # apart in y only
            [5.0, 5.0, 7.0, 7.0],  # apart in both: two negative widths multiply to a positive
            [2.0, 0.0, 4.0, 2.0],  # touching edges
        ]
    )
    got = iou(a, b)
    close(got, np.zeros((1, 4)), atol=1e-12, label="IoU of boxes with no overlap (expected all 0)")


@check("box_iou on random boxes matches a double loop, and is symmetric when both inputs are the same set")
def test_iou_random(sub):
    iou = has(sub, "box_iou")
    rng = np.random.default_rng(0)
    a = random_boxes(rng, 7)
    b = random_boxes(rng, 5)
    got = iou(a, b)
    shape(got, (7, 5), label="box_iou output")
    close(got, loop_iou(a, b), atol=1e-12, label="vectorised vs loop")
    same = iou(a, a)
    close(same, same.T, atol=1e-12, label="IoU of a set with itself")
    close(np.diag(same), np.ones(7), atol=1e-12, label="diagonal")


@check("nms keeps boxes 0, 2, and 3 in the worked example, because box 2 only overlaps a suppressed box")
def test_nms_worked(sub):
    nms = has(sub, "nms")
    got = np.asarray(nms(NMS_BOXES, NMS_SCORES, 0.5))
    equal(got.tolist(), [0, 2, 3], label="kept indices at threshold 0.5")
    got = np.asarray(nms(NMS_BOXES[::-1].copy(), NMS_SCORES[::-1].copy(), 0.5))
    equal(sorted(got.tolist()), [0, 1, 3], label="kept indices after reversing the input order")


def reference_nms(boxes, scores, thr):
    remaining = list(range(len(boxes)))
    keep = []
    while remaining:
        best = max(remaining, key=lambda i: (scores[i], -i))
        keep.append(best)
        ious = loop_iou(boxes[[best]], boxes[remaining])[0]
        remaining = [i for i, v in zip(remaining, ious) if i != best and v <= thr]
    return keep


@check("nms matches a brute-force reference on random boxes at three thresholds")
def test_nms_random(sub):
    nms = has(sub, "nms")
    rng = np.random.default_rng(1)
    for thr in (0.1, 0.3, 0.5):
        boxes = random_boxes(rng, 60)
        scores = rng.uniform(size=60)
        got = np.asarray(nms(boxes, scores, thr)).tolist()
        want = reference_nms(boxes, scores, thr)
        equal(got, want, label=f"kept indices at threshold {thr}")
        kept = boxes[got]
        pair = loop_iou(kept, kept) - np.eye(len(kept))
        if pair.max() > thr + 1e-12:
            fail(f"two kept boxes overlap with IoU {pair.max():.3f}, above the threshold {thr}")


@check("match_detections labels the hand-built example [TP, TP, FP, FP, TP, FP], and the duplicate is a false positive")
def test_match(sub):
    match = has(sub, "match_detections")
    img_d = np.zeros(6, dtype=int)
    img_g = np.zeros(3, dtype=int)
    order, is_tp = match(DET_BOXES, DET_SCORES, img_d, GT_BOXES, img_g, 0.5)
    equal(np.asarray(order).tolist(), [0, 1, 2, 3, 4, 5], label="order")
    equal(np.asarray(is_tp).astype(int).tolist(), [1, 1, 0, 0, 1, 0], label="true-positive flags")
    # Shuffle the detections: the labels must follow the scores, not the input order.
    perm = np.array([4, 2, 5, 0, 3, 1])
    order, is_tp = match(DET_BOXES[perm], DET_SCORES[perm], img_d, GT_BOXES, img_g, 0.5)
    equal(np.asarray(is_tp).astype(int).tolist(), [1, 1, 0, 0, 1, 0], label="flags after shuffling the input")
    equal(perm[np.asarray(order)].tolist(), [0, 1, 2, 3, 4, 5], label="order mapped back to the original indices")
    # A ground-truth box in another image cannot be matched.
    order, is_tp = match(DET_BOXES[:1], DET_SCORES[:1], np.array([1]), GT_BOXES, img_g, 0.5)
    equal(np.asarray(is_tp).astype(int).tolist(), [0], label="a detection in image 1 against ground truth in image 0")


@check("precision_recall and average_precision give 0.8667 on the hand-built example, 1 for a perfect run, and 0 for no hits")
def test_ap(sub):
    pr = has(sub, "precision_recall")
    ap = has(sub, "average_precision")
    is_tp = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    p, r = pr(is_tp, 3)
    close(p, np.array([1, 1, 2 / 3, 0.5, 0.6, 0.5]), atol=1e-12, label="precision")
    close(r, np.array([1 / 3, 2 / 3, 2 / 3, 2 / 3, 1.0, 1.0]), atol=1e-12, label="recall")
    almost(ap(p, r), 1 / 3 + 1 / 3 + 0.6 / 3, 1e-12, label="all-point AP of the hand example")
    p, r = pr(np.array([1, 1, 1], dtype=bool), 3)
    almost(ap(p, r), 1.0, 1e-12, label="AP of a perfect detector")
    p, r = pr(np.array([0, 0, 0], dtype=bool), 3)
    almost(ap(p, r), 0.0, 1e-12, label="AP when every detection is wrong")
    p, r = pr(np.array([1, 0], dtype=bool), 4)
    almost(ap(p, r), 0.25, 1e-12, label="AP when only one of four boxes is found")


@check("mask_iou and dice_score agree with the hand example, with Dice = 2 IoU / (1 + IoU), and score empty masks 1")
def test_masks(sub):
    miou = has(sub, "mask_iou")
    dice = has(sub, "dice_score")
    gt = np.zeros((4, 4), dtype=bool)
    gt[1:3, 1:3] = True
    pred = np.zeros((4, 4), dtype=bool)
    pred[1:3, 1:4] = True
    almost(miou(pred, gt), 4 / 6, 1e-12, label="pixel IoU")
    almost(dice(pred, gt), 0.8, 1e-12, label="Dice")
    rng = np.random.default_rng(2)
    for _ in range(5):
        a = rng.uniform(size=(8, 8)) < 0.4
        b = rng.uniform(size=(8, 8)) < 0.4
        j = miou(a, b)
        almost(dice(a, b), 2 * j / (1 + j), 1e-12, label="Dice against 2 IoU / (1 + IoU)")
    empty = np.zeros((3, 3), dtype=bool)
    almost(miou(empty, empty), 1.0, 1e-12, label="IoU of two empty masks")
    almost(dice(empty, empty), 1.0, 1e-12, label="Dice of two empty masks")


@check("soft_dice_loss gives 0.15 on the worked example, falls to near 0 on a perfect mask, and its gradient matches finite differences")
def test_soft_dice(sub):
    loss_fn = has(sub, "soft_dice_loss")
    p = np.array([0.9, 0.8, 0.2, 0.1])
    g = np.array([1.0, 1.0, 0.0, 0.0])
    loss, grad = loss_fn(p, g, eps=0.0)
    almost(loss, 0.15, 1e-12, label="loss at eps = 0")
    close(grad, np.array([-0.2875, -0.2875, 0.2125, 0.2125]), atol=1e-12, label="gradient at eps = 0")
    loss, _ = loss_fn(g.copy(), g, eps=1.0)
    almost(loss, 0.0, 1e-12, label="loss for a perfect prediction")
    rng = np.random.default_rng(3)
    for eps in (0.0, 1.0):
        p = rng.uniform(0.05, 0.95, size=(3, 5))
        g = (rng.uniform(size=(3, 5)) < 0.4).astype(float)
        _, grad = loss_fn(p, g, eps=eps)
        shape(grad, (3, 5), label="gradient")
        want = finite_difference(lambda q: loss_fn(q, g, eps=eps)[0], p.copy())
        close(grad, want, atol=1e-7, rtol=1e-5, label=f"gradient against finite differences, eps = {eps}")
    loss, grad = loss_fn(np.zeros(4), np.zeros(4), eps=1.0)
    almost(loss, 0.0, 1e-12, label="loss when prediction and target are both empty")


@check("NMS lifts the average precision of a detector that fires several times on each object")
def test_pipeline(sub):
    nms = has(sub, "nms")
    match = has(sub, "match_detections")
    pr = has(sub, "precision_recall")
    ap = has(sub, "average_precision")
    rng = np.random.default_rng(4)
    gt, gt_img, boxes, scores, img = [], [], [], [], []
    for im in range(4):
        objs = np.array([[10.0 + 40 * k, 10.0, 30.0 + 40 * k, 30.0] for k in range(3)])
        for o in objs:
            gt.append(o)
            gt_img.append(im)
            for _ in range(4):  # four jittered detections per object
                boxes.append(o + rng.normal(scale=1.0, size=4))
                scores.append(rng.uniform(0.4, 1.0))
                img.append(im)
        for _ in range(3):  # background clutter
            xy = rng.uniform(0, 100, size=2)
            boxes.append(np.concatenate([xy, xy + 20]))
            scores.append(rng.uniform(0.0, 0.6))
            img.append(im)
    gt = np.array(gt)
    gt_img = np.array(gt_img)
    boxes = np.array(boxes)
    scores = np.array(scores)
    img = np.array(img)

    def run(b, s, i):
        order, tp = match(b, s, i, gt, gt_img, 0.5)
        p, r = pr(np.asarray(tp), len(gt))
        return ap(p, r)

    before = run(boxes, scores, img)
    keep_idx = []
    for im in range(4):
        idx = np.where(img == im)[0]
        kept = np.asarray(nms(boxes[idx], scores[idx], 0.5))
        keep_idx.extend(idx[kept].tolist())
    keep_idx = np.array(keep_idx)
    after = run(boxes[keep_idx], scores[keep_idx], img[keep_idx])
    if not (0.0 <= before <= 1.0 and 0.0 <= after <= 1.0):
        fail(f"AP must lie in [0, 1], got {before:.3f} before NMS and {after:.3f} after")
    if after < before + 0.1:
        fail(f"expected NMS to raise AP by at least 0.1, got {before:.3f} before and {after:.3f} after")
    if after < 0.8:
        fail(f"expected AP of at least 0.8 after NMS, got {after:.3f}")
