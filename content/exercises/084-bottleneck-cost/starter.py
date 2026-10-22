"""Counting the cost of a bottleneck block, in multiply-adds."""


def conv_flops(c_in: int, c_out: int, k: int, h: int, w: int) -> int:
    """Multiply-adds of one convolution that keeps the picture at h by w.

    Assume stride 1, a k by k kernel, and no bias. Each of the h * w * c_out
    output numbers needs c_in * k * k products.

    Args:
        c_in: Input channels.
        c_out: Output channels.
        k: Kernel height and width.
        h: Output height.
        w: Output width.

    Returns:
        The multiply-add count, as an int.
    """
    raise NotImplementedError("conv_flops")


def bottleneck_flops(c_in: int, c_mid: int, c_out: int, h: int, w: int) -> int:
    """Multiply-adds of a bottleneck block's three convolutions.

    The block is a 1x1 convolution from c_in to c_mid channels, a 3x3
    convolution from c_mid to c_mid channels, and a 1x1 convolution from c_mid
    to c_out channels, all at h by w. The shortcut is not counted.

    Args:
        c_in: Input channels.
        c_mid: Channels inside the block.
        c_out: Output channels.
        h: Height.
        w: Width.

    Returns:
        The total multiply-add count, as an int.
    """
    raise NotImplementedError("bottleneck_flops")


def bottleneck_savings(c: int, h: int, w: int) -> float:
    """How many times cheaper a bottleneck is than two full-width 3x3 convolutions.

    The bottleneck goes c -> c // 4 -> c. The alternative is two 3x3
    convolutions from c channels to c channels.

    Args:
        c: Outer channel count, divisible by 4.
        h: Height.
        w: Width.

    Returns:
        The cost of the two 3x3 convolutions divided by the bottleneck's cost,
        as a float.
    """
    raise NotImplementedError("bottleneck_savings")


if __name__ == "__main__":
    print("2x2 picture, 4 channels in and out:")
    print(f"  1x1 convolution: {conv_flops(4, 4, 1, 2, 2)}")
    print(f"  3x3 convolution: {conv_flops(4, 4, 3, 2, 2)}")

    c, h, w = 256, 14, 14
    bottleneck = bottleneck_flops(c, c // 4, c, h, w)
    basic = 2 * conv_flops(c, c, 3, h, w)
    print(f"\n256 -> 64 -> 256 at 14x14:")
    print(f"  bottleneck:           {bottleneck:,}")
    print(f"  two 3x3 at 256:       {basic:,}")
    print(f"  savings:              {bottleneck_savings(c, h, w):.2f}x")

    print("\nmiddle width sweep at 256 channels, 14x14:")
    for mid in (16, 32, 64, 128, 256):
        print(f"  c_mid = {mid:3d}: {bottleneck_flops(c, mid, c, h, w):>12,}")
