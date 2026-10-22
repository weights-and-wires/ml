"""Safe steps: a warmup-then-cosine learning rate, the global norm, and a clip.

You write three functions: `warmup_cosine_lr`, `global_norm`, and
`clip_global_norm`. The function `lr_trace` is already written. It calls your
schedule at evenly spaced steps so you can see its shape.
"""

import math


def warmup_cosine_lr(step: float, warmup: float, total: float, peak: float, floor: float) -> float:
    """The learning rate at one step: a straight ramp, then half a cosine.

    Below `warmup`, the rate is peak * step / warmup. From `warmup` to
    `total` it glides from `peak` down to `floor` along a cosine, and it
    equals `floor` at step == total. The two pieces agree at step == warmup.

    The function maps any number to a rate. `train_step` will pass the count
    of the update about to be applied, starting at 1.

    Args:
        step: The step number. May be a float.
        warmup: The number of warmup steps.
        total: The number of steps in the schedule, total > warmup.
        peak: The rate at the end of warmup.
        floor: The rate at step == total.

    Returns:
        The learning rate, as a float.
    """
    raise NotImplementedError("warmup_cosine_lr")


def global_norm(grads: list) -> float:
    """The length of all the gradients taken together.

    Treat every entry of every array as one long vector, and return the
    square root of the sum of the squares of all entries. Do not change the
    arrays.

    Args:
        grads: A list of numpy arrays or torch tensors of any shapes.

    Returns:
        The global norm, as a plain float.
    """
    raise NotImplementedError("global_norm")


def clip_global_norm(grads: list, max_norm: float) -> float:
    """Shorten the gradients in place so their global norm is at most `max_norm`.

    When the global norm is over `max_norm`, multiply every array, in place,
    by the same number max_norm / norm. When it is not, change nothing.

    Args:
        grads: A list of numpy arrays or torch tensors of any shapes.
        max_norm: The limit on the global norm.

    Returns:
        The global norm from before any clipping, as a float.
    """
    raise NotImplementedError("clip_global_norm")


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
