"""Palette manipulation utilities — variations, parsing, formatting."""

import re
import random
import colorsys
from typing import Optional


# --- Variation generation ---

def _clamp(v: int) -> int:
    return max(0, min(255, v))


def generate_variation(
    palette: list[list[int]],
    variation_type: str = "subtle",
) -> list[list[int]]:
    """Generate a variation of the given 6-color palette."""
    result = []
    for color in palette:
        r, g, b = color

        if variation_type == "subtle":
            d = random.randint(-25, 25)
            r, g, b = r + d, g + d, b + d

        elif variation_type == "bold":
            r += random.randint(-60, 60)
            g += random.randint(-60, 60)
            b += random.randint(-60, 60)

        elif variation_type == "warmer":
            r = min(255, r + 30)
            b = max(0, b - 20)

        elif variation_type == "cooler":
            r = max(0, r - 20)
            b = min(255, b + 30)

        elif variation_type == "complementary":
            r, g, b = 255 - r, 255 - g, 255 - b
            d = random.randint(-15, 15)
            r, g, b = r + d, g + d, b + d

        elif variation_type == "saturated":
            h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
            s = min(1.0, s + 0.2)
            r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
            r, g, b = int(r2 * 255), int(g2 * 255), int(b2 * 255)

        elif variation_type == "desaturated":
            h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
            s = max(0.0, s - 0.2)
            r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
            r, g, b = int(r2 * 255), int(g2 * 255), int(b2 * 255)

        else:
            d = random.randint(-30, 30)
            r, g, b = r + d, g + d, b + d

        result.append([_clamp(r), _clamp(g), _clamp(b)])
    return result


# --- Keyword mapping for natural language adjustments ---

ADJUSTMENT_KEYWORDS: dict[str, str] = {
    "warmer": "warmer", "warm": "warmer", "hot": "warmer", "fiery": "warmer",
    "cooler": "cooler", "cool": "cooler", "cold": "cooler", "icy": "cooler",
    "brighter": "bold", "bolder": "bold", "vibrant": "bold", "vivid": "bold",
    "subtle": "subtle", "softer": "subtle", "muted": "subtle", "pastel": "subtle",
    "opposite": "complementary", "complement": "complementary", "invert": "complementary",
    "saturated": "saturated", "richer": "saturated",
    "desaturated": "desaturated", "duller": "desaturated", "greyer": "desaturated",
}


def detect_variation_type(text: str) -> str:
    """Detect variation type from natural language text."""
    text_lower = text.lower()
    for keyword, vtype in ADJUSTMENT_KEYWORDS.items():
        if keyword in text_lower:
            return vtype
    return "subtle"


# --- Color parsing ---

def parse_hex_colors(text: str) -> list[list[int]]:
    """Extract hex color codes from text and convert to RGB lists."""
    matches = re.findall(r'#([0-9a-fA-F]{6})', text)
    return [[int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)] for h in matches]


def parse_rgb_colors(text: str) -> list[list[int]]:
    """Extract RGB tuples/lists from text."""
    matches = re.findall(
        r'\[?\(?\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)?\]?',
        text,
    )
    return [[int(r), int(g), int(b)] for r, g, b in matches]


def parse_colors_from_text(text: str) -> list[list[int]]:
    """Parse all color values (hex and RGB) from text."""
    colors = parse_hex_colors(text)
    colors.extend(parse_rgb_colors(text))
    return colors


# --- Formatting ---

def palette_to_hex(palette: list[list[int]]) -> str:
    """Format palette as space-separated hex codes."""
    return " ".join(f"#{r:02x}{g:02x}{b:02x}" for r, g, b in palette)


def palette_display(palette: Optional[list[list[int]]]) -> str:
    """Human-readable palette display."""
    if not palette:
        return "None"
    return palette_to_hex(palette)
