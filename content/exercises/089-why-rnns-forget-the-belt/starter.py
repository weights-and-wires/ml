"""How far does a gradient survive?

You build three small functions. The first two describe a plain recurrent net,
whose gradient is multiplied by the same factor at every step. The third
describes a belt, where the factor at each step is a forget gate that can
differ from step to step.
"""

import numpy as np


def gradient_after_steps(factor: float, steps: int) -> np.ndarray:
    """Return the share of the gradient left after each of the first `steps` steps.

    Args:
        factor: The multiplier applied at every step, between 0 and 1.
        steps: How many steps to look back, at least 1.

    Returns:
        An array of shape (steps,). Entry t - 1 holds `factor ** t`, so the
        array starts with `factor` and ends with `factor ** steps`.
    """
    raise NotImplementedError("gradient_after_steps")


def steps_until_below(factor: float, threshold: float) -> int:
    """Return the smallest number of steps t with `factor ** t < threshold`.

    Args:
        factor: The multiplier applied at every step, strictly between 0 and 1.
        threshold: The share of the gradient you call "gone", strictly between
            0 and 1.

    Returns:
        The smallest whole number t such that `factor ** t` is below
        `threshold`, as a Python int.
    """
    raise NotImplementedError("steps_until_below")


def forget_gate_gradient_product(forget_gates: np.ndarray) -> np.ndarray:
    """Return the running product of forget gates along the time axis.

    Args:
        forget_gates: Forget-gate values, shape (T, hidden_size), every entry
            in (0, 1). Row 0 is the gate nearest the loss.

    Returns:
        An array of shape (T, hidden_size). Row t holds
        `forget_gates[0] * forget_gates[1] * ... * forget_gates[t]`, entry by
        entry: the share of the gradient that survives t + 1 steps along the belt.
    """
    raise NotImplementedError("forget_gate_gradient_product")


if __name__ == "__main__":
    plain = gradient_after_steps(0.45, 200)
    belt = forget_gate_gradient_product(np.full((200, 3), 0.995))
    print(f"plain net after 4 steps:   {plain[:4].round(4)}")
    print(f"plain net after 200 steps: {plain[-1]:.3g}")
    print(f"belt after 200 steps:      {belt[-1].round(4)}")
    for factor in (0.45, 0.9, 0.99, 0.995):
        reach = steps_until_below(factor, 0.01)
        print(f"factor {factor}: below 1% after {reach} steps")
