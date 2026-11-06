"""Reference solution for the safe steps exercise."""

import math


def warmup_cosine_lr(step: float, warmup: float, total: float, peak: float, floor: float) -> float:
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def global_norm(grads: list) -> float:
    return math.sqrt(sum(float((g**2).sum()) for g in grads))


def clip_global_norm(grads: list, max_norm: float) -> float:
    total = global_norm(grads)
    if total > max_norm:
        scale = max_norm / total
        for g in grads:
            g *= scale
    return total


def lr_trace(warmup: float, total: float, peak: float, floor: float, n_points: int) -> list:
    """Given: the schedule at `n_points` evenly spaced steps from 0 to `total`."""
    return [
        warmup_cosine_lr(total * i / (n_points - 1), warmup, total, peak, floor)
        for i in range(n_points)
    ]


if __name__ == "__main__":
    trace = lr_trace(warmup=10, total=50, peak=2.0, floor=0.1, n_points=26)
    print("the schedule, one bar every two steps:")
    for i, value in enumerate(trace):
        print(f"step {2 * i:2d}  {value:5.3f}  {'#' * round(value * 20)}")

    grads = [[3.0, 4.0], [0.0]]
    import numpy as np

    arrays = [np.array(g) for g in grads]
    print("global norm:", global_norm(arrays))
    before = clip_global_norm(arrays, max_norm=2.5)
    print("norm before the clip:", before)
    print("after the clip:", [a.tolist() for a in arrays])
