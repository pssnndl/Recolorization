import json
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from .color_extraction import extract_colorthief, extract_pylette
from .colormind import fetch_palette, fetch_palette_with_seed
from .palette_utils import parse_colors_from_text, generate_variation


def _palette_from_llm(
    description: str,
    existing_palette: list[list[int]] | None = None,
) -> list[list[int]] | None:
    """Ask Llama3 to suggest 6 RGB colors matching a text description."""
    llm = ChatOllama(model="llama3.1:8b", temperature=0.8, num_predict=256)

    context = ""
    if existing_palette:
        context = f"\nCurrent palette for reference: {existing_palette}\n"

    prompt = (
        f'Generate exactly 6 RGB colors that match this description: "{description}"\n'
        f"{context}\n"
        "Respond with ONLY a JSON array of 6 arrays, each with 3 integers 0-255.\n"
        "Example: [[255,100,50],[200,180,60],[30,120,200],[255,200,150],[80,80,80],[240,240,230]]\n\n"
        "Your response (JSON only, no explanation):"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    match = re.search(r'\[\s*\[.*?\]\s*\]', text, re.DOTALL)
    if match:
        try:
            colors = json.loads(match.group())
            if len(colors) >= 6:
                return [
                    [max(0, min(255, int(c))) for c in rgb]
                    for rgb in colors[:6]
                ]
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def extract_colors_from_image(image_b64) -> str:
        """Extract dominant colors from the user's uploaded image."""
        if not image_b64:
            return "Error: No image uploaded yet."
        results = []
        ct = extract_colorthief(image_b64)
        if ct:
            results.append({"colors": ct, "source": "colorthief"})
        py = extract_pylette(image_b64)
        if py:
            results.append({"colors": py, "source": "pylette"})
        return json.dumps(results) if results else "Could not extract colors."


def generate_palette_from_description(description: str, current_palette) -> str:
        """Generate a 6-color palette from a natural language description like 'warm sunset' or 'ocean blues'."""
        colors = _palette_from_llm(description, current_palette)
        if colors:
            return json.dumps(colors)
        return "Could not generate palette from description."


def get_random_palette() -> str:
        """Get a random harmonious 6-color palette from Colormind."""
        colors = fetch_palette()
        if colors:
            return json.dumps(colors)
        return "Could not fetch palette."


def parse_user_colors(text: str) -> str:
        """Parse specific hex codes (#RRGGBB) or RGB values from text and build a 6-color palette."""
        colors = parse_colors_from_text(text)
        if not colors:
            return "No color values found in text."
        if len(colors) >= 6:
            return json.dumps(colors[:6])
        filled = fetch_palette_with_seed(colors)
        if filled:
            return json.dumps(filled)
        padded = colors[:]
        while len(padded) < 6:
            padded.append(padded[-1])
        return json.dumps(padded[:6])


def create_palette_variation(variation_type: str, current_palette) -> str:
    """Create a variation of the current palette. variation_type: subtle, bold, warmer, cooler, complementary, saturated, desaturated."""
    if not current_palette:
        return "Error: No current palette to vary."
    colors = generate_variation(current_palette, variation_type)
    return json.dumps(colors)

