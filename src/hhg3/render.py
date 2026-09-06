"""Inline terminal rendering, so a screen recording shows the evidence itself.

Images are drawn with the upper-half-block character: each cell carries two
vertical pixels - the foreground colour paints the top half, the background the
bottom - so a text grid holds twice the vertical resolution it otherwise would.

Everything degrades to plain text when the stream is not a colour-capable TTY.
The panel, not the picture, is what carries the facts; the picture is there so a
viewer can see at a glance that the match is the same person.
"""

from __future__ import annotations

import os
import shutil
import sys
import unicodedata
from pathlib import Path

RESET = "\x1b[0m"
HALF = "▀"  # upper half block
_BOX = {"tl": "┌", "tr": "┐", "bl": "└", "br": "┘",
        "h": "─", "v": "│", "l": "├", "r": "┤"}


def enable_vt() -> None:
    """Turn on ANSI escape processing on a legacy Windows console."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        for handle_id in (-11, -12):  # stdout, stderr
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:  # pragma: no cover - best effort, never fatal
        pass


def supports_color(stream=None) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


def paint(code: str, text: str, enabled: bool = True) -> str:
    return "\x1b[%sm%s%s" % (code, text, RESET) if enabled else text


def display_width(text: str) -> int:
    """Columns `text` occupies. CJK and most emoji are double-width, and post
    titles routinely carry both - counting characters would skew the box edges."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - display_width(text))


def clean(value: object) -> str:
    """Flatten a value to one printable line."""
    text = " ".join(str(value).split())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cc")


# --- pictures ----------------------------------------------------------
def preview_cols(count: int = 2, gap: int = 4, cap: int = 72) -> int:
    """How wide each preview may be, given the terminal and how many share the row.

    Bounded by height as well as width: a face crop is roughly square, so at two
    pixel rows per cell a wide block is also a tall one, and a picture taller than
    the window is worse than a smaller one that fits.
    """
    term = shutil.get_terminal_size((100, 30))
    by_width = (term.columns - gap * (count - 1) - 2) // max(count, 1)
    by_height = int((term.lines - 6) * 2 / 1.15)  # rows -> cols for a square-ish crop
    return max(20, min(by_width, by_height, cap))


def _crop(img, bbox, pad_frac: float):
    """Face box grown by `pad_frac` of its longest side, clamped to the image.

    Mirrors pipeline._crop deliberately: importing the pipeline here would drag
    the face models into a module that only draws pictures.
    """
    x, y, w, h = (int(v) for v in bbox)
    grow = int(pad_frac * max(w, h))
    x0, y0 = max(x - grow, 0), max(y - grow, 0)
    x1 = min(x + w + grow, img.shape[1])
    y1 = min(y + h + grow, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return img
    return img[y0:y1, x0:x1]


def image_block(path: str | Path, cols: int = 34, bbox=None, pad_frac: float = 0.35) -> list[str]:
    """Render an image `cols` characters wide, cropped to `bbox` when given.

    Candidate images are whole posts - a thumbnail, a group shot - so the face
    can be a small part of the frame. Cropping to the box the matcher actually
    scored puts that face at full size, which is the point of showing it.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - cv2 is a core dep
        return []

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return []
    if bbox:
        img = _crop(img, bbox, pad_frac)

    height, width = img.shape[:2]
    # A cell is about twice as tall as it is wide and holds two pixel rows, so
    # halving here keeps the picture at its true aspect ratio.
    rows = max(1, round(cols * height / (2 * width)))
    # A tightly cropped face is often smaller than the block we are drawing it
    # into, and INTER_AREA only looks right shrinking.
    interp = cv2.INTER_AREA if cols < width else cv2.INTER_CUBIC
    small = cv2.resize(img, (cols, rows * 2), interpolation=interp)

    lines = []
    for r in range(rows):
        top, bottom = small[2 * r], small[2 * r + 1]
        cells = [
            "\x1b[38;2;%d;%d;%d;48;2;%d;%d;%dm%s"
            % (top[c][2], top[c][1], top[c][0], bottom[c][2], bottom[c][1], bottom[c][0], HALF)
            for c in range(cols)
        ]
        lines.append("".join(cells) + RESET)
    return lines


def captioned(path: str | Path, caption: str, cols: int, color: bool, bbox=None) -> tuple[list[str], int]:
    """An image block with a heading above it, padded to a fixed width."""
    block = image_block(path, cols, bbox)
    if not block:
        block = [pad("(image unavailable)", cols)]
    return [paint("1", pad(caption[:cols], cols), color)] + block, cols


def side_by_side(blocks: list[tuple[list[str], int]], gap: int = 4) -> list[str]:
    """Set fixed-width blocks next to each other, padding the shorter ones."""
    blocks = [b for b in blocks if b[0]]
    if not blocks:
        return []
    height = max(len(lines) for lines, _ in blocks)
    out = []
    for i in range(height):
        out.append((" " * gap).join(
            lines[i] if i < len(lines) else " " * width for lines, width in blocks
        ))
    return out


# --- panel -------------------------------------------------------------
def _chunks(value: str, width: int) -> list[str]:
    """Hard-chunk, not word-wrap: hashes and URLs have no spaces to break on."""
    if width < 1:
        return [value]
    out, line, used = [], "", 0
    for ch in value:
        size = display_width(ch)
        if used + size > width:
            out.append(line)
            line, used = "", 0
        line += ch
        used += size
    out.append(line)
    return out


def panel(sections: list[tuple[str, list[tuple[str, str]]]], color: bool = True) -> list[str]:
    """Box titled label/value sections, wrapping any value that overruns."""
    term = shutil.get_terminal_size((100, 30)).columns
    label_w = max((len(k) for _, rows in sections for k, _ in rows), default=0)
    natural = max(
        [display_width(t) for t, _ in sections]
        + [label_w + 2 + display_width(v) for _, rows in sections for _, v in rows]
    )
    inner = max(min(natural, term - 5), 24)
    value_w = inner - label_w - 2

    bar = _BOX["h"] * (inner + 2)
    out = [_BOX["tl"] + bar + _BOX["tr"]]
    for idx, (title, rows) in enumerate(sections):
        if idx:
            out.append(_BOX["l"] + bar + _BOX["r"])
        out.append("%s %s %s" % (_BOX["v"], paint("1", pad(title, inner), color), _BOX["v"]))
        out.append(_BOX["l"] + bar + _BOX["r"])
        for label, value in rows:
            for j, chunk in enumerate(_chunks(str(value), value_w)):
                key = paint("2", (label if j == 0 else "").ljust(label_w), color)
                out.append("%s %s  %s %s" % (_BOX["v"], key, pad(chunk, value_w), _BOX["v"]))
    out.append(_BOX["bl"] + bar + _BOX["br"])
    return out
