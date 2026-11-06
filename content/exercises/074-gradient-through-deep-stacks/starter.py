"""How much gradient survives a deep stack, with and without a shortcut."""

import numpy as np


def orthogonal_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    """A random orthogonal matrix: it rotates or flips a vector and keeps its length.

    Draw a Gaussian matrix, take the Q factor of its QR decomposition, and
    multiply each column of Q by the sign of the matching diagonal entry of R.

    Args:
        d: Size of the square matrix.
        rng: Source of randomness. Draw from it exactly once, with
            rng.normal(size=(d, d)).

    Returns:
        A (d, d) array Q with Q.T @ Q equal to the identity.
    """
    raise NotImplementedError("orthogonal_matrix")


def scalar_gradient_factor(J: float, depth: int, residual: bool) -> float:
    """The toy model: every layer multiplies the gradient by the same number.

    A plain layer multiplies the gradient by J. A residual layer multiplies it
    by 1 + J. Return the factor after depth layers.

    Args:
        J: The slope of one layer's branch.
        depth: Number of layers.
        residual: Whether each layer has a shortcut.

    Returns:
        J ** depth for a plain stack, (1 + J) ** depth for a residual stack.
    """
    raise NotImplementedError("scalar_gradient_factor")


def gradient_norm_through_stack(depth: int, residual: bool) -> float:
    """Length of the gradient that reaches the input of a stack of layers.

    Build `depth` layers of width 16, each 0.1 times orthogonal_matrix(16, rng),
    drawing every matrix from one rng = np.random.default_rng(0), in order.
    Start from g = np.ones(16) / np.sqrt(16). Walk the layers in reverse order.
    A plain layer sends g to g @ W. A residual layer sends g to g + g @ W.

    Args:
        depth: Number of layers in the stack.
        residual: Whether each layer has a shortcut.

    Returns:
        The Euclidean norm of g after the last step, as a float.
    """
    raise NotImplementedError("gradient_norm_through_stack")


if __name__ == "__main__":
    print("depth   residual length   plain length   scalar (1+J)^L   scalar J^L")
    for depth in (1, 2, 10, 20, 50):
        r = gradient_norm_through_stack(depth, residual=True)
        p = gradient_norm_through_stack(depth, residual=False)
        sr = scalar_gradient_factor(0.1, depth, residual=True)
        sp = scalar_gradient_factor(0.1, depth, residual=False)
        print(f"{depth:5d}   {r:15.4f}   {p:12.3e}   {sr:14.4g}   {sp:10.3e}")

    rng = np.random.default_rng(1)
    q = orthogonal_matrix(16, rng)
    v = rng.normal(size=16)
    print(f"\nlength of v: {np.linalg.norm(v):.6f}, length of v @ Q: {np.linalg.norm(v @ q):.6f}")
