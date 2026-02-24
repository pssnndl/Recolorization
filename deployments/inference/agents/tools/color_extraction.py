"""Extract palettes from images using colorthief and pylette."""

import io
import base64
from typing import Optional

import numpy as np
from PIL import Image
from colorthief import ColorThief

# from pylette import extract_colors


def _pad_palette(colors: list[list[int]], target: int = 6) -> list[list[int]]:
    """Pad palette to target length by repeating the last color."""
    while len(colors) < target:
        colors.append(list(colors[-1]))
    return colors[:target]


def extract_colorthief(image_b64: str, color_count: int = 6) -> Optional[list[list[int]]]:
    """Extract dominant colors from a base64 image using ColorThief."""

    try:
        image_bytes = base64.b64decode(image_b64)
        ct = ColorThief(io.BytesIO(image_bytes))
        palette = ct.get_palette(color_count=color_count, quality=5)
        result = [list(c) for c in palette]
        return _pad_palette(result)
    except Exception:
        return None


# def extract_pylette(image_b64: str, color_count: int = 6) -> Optional[list[list[int]]]:
#     """Extract palette from a base64 image using pylette."""
#     try:
#         image_bytes = base64.b64decode(image_b64)
#         img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         colors = extract_colors(
#             image=np.array(img),
#             palette_size=color_count,
#             sort_mode="frequency",
#         )
#         result = [[int(c.rgb[0]), int(c.rgb[1]), int(c.rgb[2])] for c in colors]
#         return _pad_palette(result)
#     except Exception:
#         return None
