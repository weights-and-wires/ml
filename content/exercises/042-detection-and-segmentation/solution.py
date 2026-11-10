"""Reference solution: boxes, NMS, average precision, and Dice."""

from __future__ import annotations

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the pairwise intersection over union of two sets of boxes.

    Args:
        a: Array of shape (N, 4) holding boxes as (x1, y1, x2, y2), x1 < x2, y1 < y2.
        b: Array of shape (M, 4) in the same format.

    Returns:
        Array of shape (N, M) whose entry [i, j] is the IoU of a[i] and b[j].
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
    """Greedy non-maximum suppression.

    Args:
        boxes: Array of shape (N, 4) as (x1, y1, x2, y2).
        scores: Array of shape (N,).
        thr: A box is suppressed when its IoU with a kept box is greater than thr.

    Returns:
        Integer array of kept indices, in descending score order. Ties go to
        the lower index.
    """
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    ious = box_iou(boxes, boxes)
    keep: list[int] = []
    for i in order:
        if all(ious[i, k] <= thr for k in keep):
            keep.append(int(i))
    return np.array(keep, dtype=int)


def match_detections(boxes, scores, image_ids, gt_boxes, gt_image_ids, thr):
    """Label each detection as a true or false positive.

    Detections are visited from the highest score to the lowest. Each one takes
    the not-yet-matched ground-truth box of the same image with the highest IoU.
    It is a true positive when that IoU is at least thr, and that ground-truth
    box is then used up. Otherwise it is a false positive.

    Args:
        boxes: (N, 4) detections. scores: (N,). image_ids: (N,) integers.
        gt_boxes: (G, 4). gt_image_ids: (G,) integers.
        thr: IoU threshold.

    Returns:
        (order, is_tp): order is the indices of the detections sorted by
        descending score (ties by lower index), and is_tp is a boolean array
        in that order.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float)
    image_ids = np.asarray(image_ids)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    gt_image_ids = np.asarray(gt_image_ids)
    order = np.argsort(-scores, kind="stable")
    used = np.zeros(len(gt_boxes), dtype=bool)
    is_tp = np.zeros(len(order), dtype=bool)
    ious = box_iou(boxes, gt_boxes) if len(gt_boxes) else np.zeros((len(boxes), 0))
    for rank, i in enumerate(order):
        free = (gt_image_ids == image_ids[i]) & ~used
        if not free.any():
            continue
        candidates = np.where(free)[0]
        j = candidates[np.argmax(ious[i, candidates])]
        if ious[i, j] >= thr:
            used[j] = True
            is_tp[rank] = True
    return order, is_tp


def precision_recall(is_tp: np.ndarray, n_gt: int):
    """Return precision and recall after each detection.

    Args:
        is_tp: Boolean array in descending score order.
        n_gt: Number of ground-truth boxes.

    Returns:
        (precision, recall), each of shape (len(is_tp),).
    """
    is_tp = np.asarray(is_tp, dtype=float)
    tp = np.cumsum(is_tp)
    count = np.arange(1, len(is_tp) + 1)
    return tp / count, tp / n_gt


def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """All-point average precision (PASCAL VOC 2010 and later).

    Args:
        precision: Precision after each detection, in descending score order.
        recall: Recall after each detection.

    Returns:
        The area under the precision envelope, a float in [0, 1].
    """
    r = np.concatenate([[0.0], np.asarray(recall, dtype=float), [1.0]])
    p = np.concatenate([[0.0], np.asarray(precision, dtype=float), [0.0]])
    p = np.maximum.accumulate(p[::-1])[::-1]
    steps = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[steps + 1] - r[steps]) * p[steps + 1]))


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pixel IoU of two binary masks. Two empty masks score 1.0."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient of two binary masks. Two empty masks score 1.0."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / total)


def soft_dice_loss(p: np.ndarray, g: np.ndarray, eps: float = 1.0):
    """Soft Dice loss and its gradient with respect to p.

    Args:
        p: Predicted probabilities, any shape, values in [0, 1].
        g: Targets of the same shape.
        eps: Smoothing constant added to the numerator and the denominator.

    Returns:
        (loss, grad): loss = 1 - (2 sum(p g) + eps) / (sum(p) + sum(g) + eps),
        and grad has the shape of p.
    """
    p = np.asarray(p, dtype=float)
    g = np.asarray(g, dtype=float)
    num = 2.0 * np.sum(p * g) + eps
    den = np.sum(p) + np.sum(g) + eps
    loss = 1.0 - num / den
    grad = -(2.0 * g * den - num) / den**2
    return float(loss), grad


if __name__ == "__main__":
    a = np.array([[0.0, 0.0, 4.0, 4.0]])
    b = np.array([[2.0, 2.0, 6.0, 6.0]])
    print(box_iou(a, b))
    print(soft_dice_loss(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1.0, 1.0, 0.0, 0.0]), eps=0.0))
