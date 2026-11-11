"""Starter: boxes, NMS, average precision, and Dice."""

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
    raise NotImplementedError("write box_iou")


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
    raise NotImplementedError("write nms")


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
    raise NotImplementedError("write match_detections")


def precision_recall(is_tp: np.ndarray, n_gt: int):
    """Return precision and recall after each detection.

    Args:
        is_tp: Boolean array in descending score order.
        n_gt: Number of ground-truth boxes.

    Returns:
        (precision, recall), each of shape (len(is_tp),).
    """
    raise NotImplementedError("write precision_recall")


def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """All-point average precision (PASCAL VOC 2010 and later).

    Args:
        precision: Precision after each detection, in descending score order.
        recall: Recall after each detection.

    Returns:
        The area under the precision envelope, a float in [0, 1].
    """
    raise NotImplementedError("write average_precision")


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pixel IoU of two binary masks. Two empty masks score 1.0."""
    raise NotImplementedError("write mask_iou")


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient of two binary masks. Two empty masks score 1.0."""
    raise NotImplementedError("write dice_score")


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
    raise NotImplementedError("write soft_dice_loss")


if __name__ == "__main__":
    a = np.array([[0.0, 0.0, 4.0, 4.0]])
    b = np.array([[2.0, 2.0, 6.0, 6.0]])
    print(box_iou(a, b))
    print(soft_dice_loss(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1.0, 1.0, 0.0, 0.0]), eps=0.0))
