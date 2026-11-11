#!/usr/bin/env python3
"""Numbers and SVG blocks for the detection-and-segmentation figures.

Figure 1: the IoU of box A = (0, 0, 4, 4) with box B = (2, 2, 6, 6), and the
IoU of A with B slid along the diagonal by 0 to 4 units.
Figure 2: the four boxes of the NMS worked example, with every pairwise IoU
that matters.
Figure 3: precision and recall for the six detections of the average-precision
worked example, the all-point envelope, and the 101-point value for comparison.

Run it to print every value the lesson quotes. Run it with an output
directory to also write the three figure blocks as fig1.svg, fig2.svg, and
fig3.svg:

    python3 scripts/figures/042-detection-and-segmentation.py [OUT_DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def all_point_ap(precision, recall):
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[0.0], precision, [0.0]])
    env = np.maximum.accumulate(p[::-1])[::-1]
    steps = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[steps + 1] - r[steps]) * env[steps + 1])), env[1:-1]


def ap_101(precision, recall):
    total = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recall >= t - 1e-12
        total += precision[mask].max() if mask.any() else 0.0
    return total / 101


A = (0.0, 0.0, 4.0, 4.0)
B = (2.0, 2.0, 6.0, 6.0)
print("IoU of A and B:", iou(A, B), "= 1/7 =", 1 / 7)
shifts = [0, 1, 2, 3, 4]
shift_iou = [iou(A, (s, s, 4.0 + s, 4.0 + s)) for s in shifts]
print("IoU after sliding B along the diagonal by", shifts, ":", [round(v, 4) for v in shift_iou])
print("Disjoint diagonal boxes without a clamp: raw widths -3 and -3 give product", (-3) * (-3))

nms_boxes = [(0, 0, 10, 10), (3, 0, 13, 10), (6, 0, 16, 10), (30, 0, 40, 10)]
nms_scores = [0.9, 0.8, 0.7, 0.6]
pairs = [(0, 1), (1, 2), (0, 2)]
pair_iou = {p: iou(nms_boxes[p[0]], nms_boxes[p[1]]) for p in pairs}
print("NMS pair IoUs:", {p: round(v, 4) for p, v in pair_iou.items()})
keep = []
for i in np.argsort(-np.array(nms_scores)):
    if all(iou(nms_boxes[i], nms_boxes[k]) <= 0.5 for k in keep):
        keep.append(int(i))
print("NMS keeps (threshold 0.5):", keep)
wrong = []
for i in np.argsort(-np.array(nms_scores)):
    if all(iou(nms_boxes[i], nms_boxes[k]) <= 0.5 for k in np.argsort(-np.array(nms_scores))[: list(np.argsort(-np.array(nms_scores))).index(i)]):
        wrong.append(int(i))
print("Wrong NMS (compares with suppressed boxes too) keeps:", wrong)

is_tp = np.array([1, 1, 0, 0, 1, 0], dtype=float)
tp = np.cumsum(is_tp)
precision = tp / np.arange(1, 7)
recall = tp / 3
ap, envelope = all_point_ap(precision, recall)
print("precision:", np.round(precision, 4).tolist())
print("recall:   ", np.round(recall, 4).tolist())
print("envelope: ", np.round(envelope, 4).tolist())
print("all-point AP:", round(ap, 4), "(exact 1/3 + 1/3 + 0.6/3 =", round(2 / 3 + 0.2, 4), ")")
print("101-point AP:", round(ap_101(precision, recall), 4))

p = np.array([0.9, 0.8, 0.2, 0.1])
g = np.array([1.0, 1.0, 0.0, 0.0])
num, den = 2 * np.sum(p * g), p.sum() + g.sum()
print("soft Dice:", num / den, "loss:", 1 - num / den, "gradient:", (-(2 * g * den - num) / den**2).tolist())
gt_mask = np.zeros((4, 4), bool)
gt_mask[1:3, 1:3] = True
pred_mask = np.zeros((4, 4), bool)
pred_mask[1:3, 1:4] = True
inter = (gt_mask & pred_mask).sum()
print("mask IoU:", inter / (gt_mask | pred_mask).sum(), "Dice:", 2 * inter / (gt_mask.sum() + pred_mask.sum()))


def fig1():
    s = 20
    ox, oy = 30, 40
    r = lambda b: (ox + b[0] * s, oy + b[1] * s, (b[2] - b[0]) * s, (b[3] - b[1]) * s)
    ax, ay, aw, ah = r(A)
    bx, by, bw, bh = r(B)
    ix, iy = ox + 2 * s, oy + 2 * s
    out = [
        '<svg viewBox="0 0 600 250" role="img" aria-label="Left, two overlapping boxes with an intersection of 4 and a union of 28. Right, IoU falls from 1 to 0 as one box slides away along the diagonal.">',
        f'<rect x="{ax}" y="{ay}" width="{aw}" height="{ah}" class="fill-soft-a stroke-a"/>',
        f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" class="fill-soft-b stroke-b"/>',
        f'<rect x="{ix}" y="{iy}" width="{2*s}" height="{2*s}" class="fill-c stroke-c"/>',
        f'<text x="{ax+6}" y="{ay+18}" class="fig-bold">A</text>',
        f'<text x="{bx+bw-18}" y="{by+bh-8}" class="fig-bold">B</text>',
        f'<text x="{ix+8}" y="{iy+25}" class="fig-small fill-paper">4</text>',
        '<text x="30" y="212" class="fig-small">intersection 4, union 16 + 16 - 4 = 28</text>',
        f'<text x="30" y="232" class="fig-small">IoU = 4 / 28 = {1/7:.3f}</text>',
    ]
    x0, y0, w, h = 330, 30, 240, 150
    out.append(f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" class="stroke-muted"/>')
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" class="stroke-muted"/>')
    bw_ = 36
    for k, (sh, v) in enumerate(zip(shifts, shift_iou)):
        bx_ = x0 + 12 + k * 46
        bh_ = v * h
        out.append(f'<rect x="{bx_}" y="{y0+h-bh_:.1f}" width="{bw_}" height="{bh_:.1f}" class="fill-b"/>')
        out.append(f'<text x="{bx_+bw_/2}" y="{y0+h-bh_-6:.1f}" text-anchor="middle" class="fig-small">{v:.2f}</text>')
        out.append(f'<text x="{bx_+bw_/2}" y="{y0+h+18}" text-anchor="middle" class="fig-small">{sh}</text>')
    out.append(f'<text x="{x0+w/2}" y="{y0+h+40}" text-anchor="middle" class="fig-small">slide B along the diagonal by this many units</text>')
    out.append(f'<text x="{x0}" y="{y0-10}" class="fig-small">IoU of A and the slid box</text>')
    out.append("</svg>")
    out.append("Figure 1. Two boxes overlap by 4 of 28 units, an IoU of 0.14. Sliding the second box by a single unit already drops the IoU from 1 to 0.39, so IoU punishes small shifts hard.")
    return "```figure\n" + "\n".join(out) + "\n```\n"


def fig2():
    s = 12
    ox = 60
    rows = [40, 90, 140, 190]
    out = [
        '<svg viewBox="0 0 600 260" role="img" aria-label="Four boxes drawn as bars along x. Box 0 is kept, box 1 is suppressed, box 2 is kept, and box 3 is kept.">',
        '<text x="20" y="22" class="fig-bold">Boxes share one y range, so each is drawn as a bar along x</text>',
    ]
    status = {0: ("keep", "fill-soft-c stroke-c"), 1: ("suppressed by 0", "fill-soft-b stroke-b"), 2: ("keep", "fill-soft-c stroke-c"), 3: ("keep", "fill-soft-c stroke-c")}
    for i, (b, sc) in enumerate(zip(nms_boxes, nms_scores)):
        y = rows[i]
        out.append(f'<rect x="{ox+b[0]*s}" y="{y}" width="{(b[2]-b[0])*s}" height="28" rx="4" class="{status[i][1]}"/>')
        out.append(f'<text x="{ox+b[0]*s+8}" y="{y+19}" class="fig-mono">box {i}  score {sc}</text>')
        out.append(f'<text x="{ox+b[2]*s+10 if i<3 else ox+b[0]*s-10}" y="{y+19}" {"" if i<3 else "text-anchor=\"end\" "}class="fig-small">{status[i][0]}</text>')
    out.append(f'<text x="20" y="245" class="fig-small">IoU(0,1) = {pair_iou[(0,1)]:.3f}   IoU(1,2) = {pair_iou[(1,2)]:.3f}   IoU(0,2) = {pair_iou[(0,2)]:.3f}   threshold 0.5</text>')
    out.append("</svg>")
    out.append("Figure 2. Box 2 overlaps suppressed box 1 (IoU 0.538) but not kept box 0 (IoU 0.25), so it survives. A version that also compares against suppressed boxes would wrongly drop it.")
    return "```figure\n" + "\n".join(out) + "\n```\n"


def fig3():
    x0, y0, w, h = 70, 30, 460, 180
    X = lambda r: x0 + r * w
    Y = lambda p: y0 + h - p * h
    out = [
        '<svg viewBox="0 0 600 260" role="img" aria-label="Precision against recall for six detections, with the envelope whose area is the average precision of 0.867.">',
        f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" class="stroke-muted"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" class="stroke-muted"/>',
    ]
    # envelope as filled steps: recall segments (0,1/3] at 1, (1/3,2/3] at 1, (2/3,1] at 0.6
    seg = [(0, 1 / 3, 1.0), (1 / 3, 2 / 3, 1.0), (2 / 3, 1.0, 0.6)]
    path = f"M {X(0):.1f} {Y(0):.1f}"
    for a, b, v in seg:
        path += f" L {X(a):.1f} {Y(v):.1f} L {X(b):.1f} {Y(v):.1f}"
    path += f" L {X(1):.1f} {Y(0):.1f} Z"
    out.append(f'<path d="{path}" class="fill-soft-c stroke-c"/>')
    for k in range(6):
        out.append(f'<circle cx="{X(recall[k]):.1f}" cy="{Y(precision[k]):.1f}" r="5" class="fill-a"/>')
        out.append(f'<text x="{X(recall[k])+8:.1f}" y="{Y(precision[k])+(-8 if k in (0,1,4) else 16):.1f}" class="fig-small">{k+1}</text>')
    for t in (0, 0.5, 1):
        out.append(f'<text x="{X(t):.1f}" y="{y0+h+18}" text-anchor="middle" class="fig-small">{t:g}</text>')
        out.append(f'<text x="{x0-10}" y="{Y(t)+4:.1f}" text-anchor="end" class="fig-small">{t:g}</text>')
    out.append(f'<text x="{x0+w/2}" y="{y0+h+40}" text-anchor="middle" class="fig-small">recall</text>')
    out.append(f'<text x="{x0}" y="{y0-10}" class="fig-small">precision. Green area = AP = {ap:.3f}</text>')
    out.append("</svg>")
    out.append("Figure 3. Each dot is the precision and recall after one more detection. Detection 3 (a duplicate) and detection 4 (clutter) pull precision down, but the envelope only keeps the best precision at each recall or higher, so the area is 0.867.")
    return "```figure\n" + "\n".join(out) + "\n```\n"


if len(sys.argv) > 1:
    d = Path(sys.argv[1])
    d.mkdir(parents=True, exist_ok=True)
    for name, fn in (("fig1", fig1), ("fig2", fig2), ("fig3", fig3)):
        (d / f"{name}.svg").write_text(fn())
    print("wrote figure blocks to", d)
