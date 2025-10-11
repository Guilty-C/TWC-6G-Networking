"""Minimal PNG plotting helpers without third-party dependencies."""
from __future__ import annotations

import math
import os
import struct
import zlib
from typing import List, Sequence, Tuple


Color = Tuple[int, int, int]


class Canvas:
    def __init__(self, width: int, height: int, background: Color = (255, 255, 255)) -> None:
        self.width = int(width)
        self.height = int(height)
        self.pixels: List[List[Color]] = [
            [background for _ in range(self.width)] for _ in range(self.height)
        ]

    def set_pixel(self, x: int, y: int, color: Color) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y][x] = color

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def draw_rect(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        for x in range(x0, x1 + 1):
            self.set_pixel(x, y0, color)
            self.set_pixel(x, y1, color)
        for y in range(y0, y1 + 1):
            self.set_pixel(x0, y, color)
            self.set_pixel(x1, y, color)

    def fill_rect(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.set_pixel(x, y, color)


_FONT: dict[str, Tuple[str, ...]] = {
    " ": ("     ",) * 7,
    "0": (" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "),
    "1": ("  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "),
    "2": (" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"),
    "3": (" ### ", "#   #", "    #", " ### ", "    #", "#   #", " ### "),
    "4": ("   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "),
    "5": ("#####", "#    ", "#    ", "#### ", "    #", "#   #", " ### "),
    "6": (" ### ", "#   #", "#    ", "#### ", "#   #", "#   #", " ### "),
    "7": ("#####", "    #", "   # ", "  #  ", "  #  ", "  #  ", "  #  "),
    "8": (" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "),
    "9": (" ### ", "#   #", "#   #", " ####", "    #", "#   #", " ### "),
    "A": (" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"),
    "B": ("#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "),
    "C": (" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "),
    "D": ("#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "),
    "E": ("#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"),
    "F": ("#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "),
    "G": (" ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ### "),
    "H": ("#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"),
    "I": (" ### ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "),
    "J": ("  ###", "   #", "   #", "   #", "   #", "#  #", " ## "),
    "K": ("#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"),
    "L": ("#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"),
    "M": ("#   #", "## ##", "# # #", "# # #", "#   #", "#   #", "#   #"),
    "N": ("#   #", "##  #", "# # #", "#  ##", "#   #", "#   #", "#   #"),
    "O": (" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "),
    "P": ("#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "),
    "Q": (" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"),
    "R": ("#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"),
    "S": (" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "),
    "T": ("#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "),
    "U": ("#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "),
    "V": ("#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "),
    "W": ("#   #", "#   #", "#   #", "# # #", "# # #", "## ##", "#   #"),
    "X": ("#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"),
    "Y": ("#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "),
    "Z": ("#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"),
    "-": ("     ", "     ", "     ", "#####", "     ", "     ", "     "),
    "/": ("    #", "   # ", "  #  ", " #   ", "#    ", "     ", "     "),
    ".": ("     ", "     ", "     ", "     ", "     ", " ##  ", " ##  "),
}


def draw_text(canvas: Canvas, x: int, y: int, text: str, color: Color = (0, 0, 0)) -> None:
    cursor_x = x
    text = text.upper()
    for ch in text:
        glyph = _FONT.get(ch, _FONT[" "])
        for row_idx, row in enumerate(glyph):
            for col_idx, val in enumerate(row):
                if val != " ":
                    canvas.set_pixel(cursor_x + col_idx, y + row_idx, color)
        cursor_x += len(glyph[0]) + 1


def ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    idx = (len(data) - 1) * min(max(q, 0.0), 1.0)
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper:
        return data[lower]
    weight = idx - lower
    return data[lower] * (1.0 - weight) + data[upper] * weight


def _save_png(path: str, canvas: Canvas) -> None:
    height = canvas.height
    width = canvas.width
    raw = bytearray()
    for row in canvas.pixels:
        raw.append(0)
        for r, g, b in row:
            raw.extend([int(r), int(g), int(b)])
    compressed = zlib.compress(bytes(raw), 9)

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack("!I", len(data))
        crc = zlib.crc32(chunk_type)
        crc = zlib.crc32(data, crc)
        crc_bytes = struct.pack("!I", crc & 0xFFFFFFFF)
        return length + chunk_type + data + crc_bytes

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", compressed))
        f.write(chunk(b"IEND", b""))


def save_line_plot(
    x_values: Sequence[float],
    y_series: Sequence[Sequence[float]],
    labels: Sequence[str],
    path: str,
    title: str,
    x_label: str,
    y_label: str,
    palette: Sequence[Color] | None = None,
) -> None:
    ensure_dir(path)
    palette = tuple(palette or [(52, 120, 246), (216, 66, 70), (60, 179, 113), (148, 0, 211)])
    canvas = Canvas(800, 480)
    left, right = 80, 760
    bottom, top = 420, 60

    xs = [float(x) for x in x_values]
    ys_flat = [float(v) for series in y_series for v in series]
    if not xs or not ys_flat:
        _save_png(path, canvas)
        return
    x_min = min(xs)
    x_max = max(xs)
    if abs(x_max - x_min) < 1e-9:
        x_max = x_min + 1.0
    y_low = min(ys_flat)
    y_high = max(ys_flat)
    q_low = _quantile(ys_flat, 0.05)
    q_high = _quantile(ys_flat, 0.95)
    span = max(q_high - q_low, 1e-6)
    y_min = min(y_low, q_low - 0.1 * span)
    y_max = max(y_high, q_high + 0.1 * span)
    if abs(y_max - y_min) < 1e-6:
        y_max = y_min + 1.0

    def project_x(val: float) -> int:
        ratio = (val - x_min) / (x_max - x_min)
        return int(left + ratio * (right - left))

    def project_y(val: float) -> int:
        ratio = (val - y_min) / (y_max - y_min)
        return int(bottom - ratio * (bottom - top))

    canvas.draw_line(left, bottom, right, bottom, (0, 0, 0))
    canvas.draw_line(left, bottom, left, top, (0, 0, 0))

    for i in range(5):
        gy = top + i * (bottom - top) // 4
        canvas.draw_line(left, gy, right, gy, (220, 220, 220))
        value = y_min + (y_max - y_min) * (4 - i) / 4
        draw_text(canvas, 10, gy - 3, f"{value:.2f}")
    for idx, xv in enumerate(xs):
        gx = project_x(xv)
        canvas.draw_line(gx, bottom, gx, bottom - 5, (0, 0, 0))
        draw_text(canvas, gx - 10, bottom + 10, f"{xv:.2f}")

    for idx, series in enumerate(y_series):
        color = palette[idx % len(palette)]
        for i in range(len(series) - 1):
            x0 = project_x(xs[i])
            y0 = project_y(series[i])
            x1 = project_x(xs[i + 1])
            y1 = project_y(series[i + 1])
            canvas.draw_line(x0, y0, x1, y1, color)
            canvas.fill_rect(x0 - 2, y0 - 2, x0 + 2, y0 + 2, color)
        if series:
            x_last = project_x(xs[min(len(xs) - 1, len(series) - 1)])
            y_last = project_y(series[-1])
            canvas.fill_rect(x_last - 2, y_last - 2, x_last + 2, y_last + 2, color)

    legend_x = right - 150
    legend_y = top + 10
    for idx, label in enumerate(labels):
        color = palette[idx % len(palette)]
        canvas.fill_rect(legend_x, legend_y + idx * 20, legend_x + 15, legend_y + idx * 20 + 10, color)
        draw_text(canvas, legend_x + 20, legend_y + idx * 20, label)

    draw_text(canvas, left, top - 35, title)
    draw_text(canvas, left + (right - left) // 2 - 40, bottom + 40, x_label)
    draw_text(canvas, 10, top - 10, y_label)

    _save_png(path, canvas)
