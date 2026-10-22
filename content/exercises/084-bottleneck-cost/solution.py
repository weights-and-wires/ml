"""Reference solution: counting the cost of a bottleneck block."""


def conv_flops(c_in: int, c_out: int, k: int, h: int, w: int) -> int:
    return int(h * w * c_in * c_out * k * k)


def bottleneck_flops(c_in: int, c_mid: int, c_out: int, h: int, w: int) -> int:
    shrink = conv_flops(c_in, c_mid, 1, h, w)
    work = conv_flops(c_mid, c_mid, 3, h, w)
    grow = conv_flops(c_mid, c_out, 1, h, w)
    return int(shrink + work + grow)


def bottleneck_savings(c: int, h: int, w: int) -> float:
    two_full_width = 2 * conv_flops(c, c, 3, h, w)
    return float(two_full_width / bottleneck_flops(c, c // 4, c, h, w))


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
