"""Colormind API client for palette generation."""

import random
import requests
from typing import Optional

COLORMIND_URL = "http://colormind.io/api/"
TIMEOUT = 5


def _add_sixth_color(colors: list[list[int]]) -> list[list[int]]:
    """Colormind returns 5 colors; derive a 6th as shifted complement of the average."""
    avg = [sum(c[i] for c in colors) // len(colors) for i in range(3)]
    complement = [max(0, min(255, 255 - c + random.randint(-20, 20))) for c in avg]
    return colors[:5] + [complement]


def fetch_palette(model: str = "default") -> Optional[list[list[int]]]:
    """Fetch a random 6-color palette from Colormind."""
    try:
        resp = requests.post(COLORMIND_URL, json={"model": model}, timeout=TIMEOUT)
        if resp.status_code == 200:
            colors = resp.json()["result"]
            return _add_sixth_color(colors)
    except Exception:
        pass
    return None


def fetch_palette_with_seed(
    seed_colors: list[Optional[list[int]]],
    model: str = "default",
) -> Optional[list[list[int]]]:
    """
    Generate a palette with some colors locked (seed) and others generated.
    seed_colors: up to 5 entries, None for slots to be generated.
    """
    input_palette = []
    for i in range(5):
        if i < len(seed_colors) and seed_colors[i] is not None:
            input_palette.append(seed_colors[i])
        else:
            input_palette.append("N")

    try:
        resp = requests.post(
            COLORMIND_URL,
            json={"model": model, "input": input_palette},
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            colors = resp.json()["result"]
            return _add_sixth_color(colors)
    except Exception:
        pass
    return None
