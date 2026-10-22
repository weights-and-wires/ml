"""Reference solution for the exercise on how far a gradient survives."""

import numpy as np


def gradient_after_steps(factor: float, steps: int) -> np.ndarray:
    exponents = np.arange(1, steps + 1)
    return float(factor) ** exponents


def steps_until_below(factor: float, threshold: float) -> int:
    if not 0.0 < factor < 1.0:
        raise ValueError("factor must be strictly between 0 and 1")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    return int(np.floor(np.log(threshold) / np.log(factor))) + 1


def forget_gate_gradient_product(forget_gates: np.ndarray) -> np.ndarray:
    return np.cumprod(forget_gates, axis=0)


if __name__ == "__main__":
    plain = gradient_after_steps(0.45, 200)
    belt = forget_gate_gradient_product(np.full((200, 3), 0.995))
    print(f"plain net after 4 steps:   {plain[:4].round(4)}")
    print(f"plain net after 200 steps: {plain[-1]:.3g}")
    print(f"belt after 200 steps:      {belt[-1].round(4)}")
    for factor in (0.45, 0.9, 0.99, 0.995):
        reach = steps_until_below(factor, 0.01)
        print(f"factor {factor}: below 1% after {reach} steps")
