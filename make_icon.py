"""
Generate a simple application icon (icon.ico + icon.png) for the GGUF Benchmark
Tool using only the standard library (no Pillow dependency).

The icon is a dark rounded square with a stylized green speed/bar-chart motif.
Writes 256x256 PNG and a multi-size ICO.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    chunk = tag + data
    return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)


def make_png(size: int = 256) -> bytes:
    """Build an RGBA PNG for the icon at the given size."""
    w = h = size
    # background colors
    bg = (30, 30, 30, 255)          # #1e1e1e
    bar = (80, 220, 130, 255)        # green
    bar_dim = (55, 160, 95, 255)     # darker green
    transparent = (0, 0, 0, 0)

    radius = size // 8  # rounded corner radius

    def in_rounded(x: int, y: int) -> bool:
        # rounded-rectangle test
        if radius <= 0:
            return True
        # corners
        cx = min(max(x, radius), w - 1 - radius)
        cy = min(max(y, radius), h - 1 - radius)
        dx = x - cx
        dy = y - cy
        return dx * dx + dy * dy <= radius * radius

    # three ascending bars (bar-chart / speed motif)
    margin = size * 0.18
    gap = size * 0.06
    bar_w = (size - 2 * margin - 2 * gap) / 3.0
    heights = [0.35, 0.58, 0.82]  # fractions of inner height
    base_y = size - margin

    bars = []
    for i, frac in enumerate(heights):
        x0 = margin + i * (bar_w + gap)
        x1 = x0 + bar_w
        top = base_y - frac * (size - 2 * margin)
        bars.append((x0, x1, top, base_y))

    rows = bytearray()
    for y in range(h):
        rows.append(0)  # filter type 0 (none) per scanline
        for x in range(w):
            if not in_rounded(x, y):
                r, g, b, a = transparent
            else:
                color = bg
                for idx, (x0, x1, top, bottom) in enumerate(bars):
                    if x0 <= x <= x1 and top <= y <= bottom:
                        color = bar if idx == 2 else bar_dim
                        break
                r, g, b, a = color
            rows.extend((r, g, b, a))

    raw = bytes(rows)
    compressed = zlib.compress(raw, 9)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)  # 8-bit RGBA
    png = sig
    png += _png_chunk(b"IHDR", ihdr)
    png += _png_chunk(b"IDAT", compressed)
    png += _png_chunk(b"IEND", b"")
    return png


def make_ico(sizes=(16, 32, 48, 64, 128, 256)) -> bytes:
    """Bundle several PNGs into a single .ico (PNG-compressed entries)."""
    images = [(s, make_png(s)) for s in sizes]
    count = len(images)

    header = struct.pack("<HHH", 0, 1, count)  # reserved, type=1 (icon), count
    directory = b""
    offset = 6 + count * 16
    image_data = b""
    for size, png in images:
        dim = 0 if size >= 256 else size  # 0 means 256 in ICO
        directory += struct.pack(
            "<BBBBHHII",
            dim, dim, 0, 0, 1, 32, len(png), offset,
        )
        image_data += png
        offset += len(png)
    return header + directory + image_data


def main():
    here = Path(__file__).resolve().parent
    (here / "icon.png").write_bytes(make_png(256))
    (here / "icon.ico").write_bytes(make_ico())
    print("Wrote icon.png and icon.ico")


if __name__ == "__main__":
    main()
