import json
import logging
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from .color_extraction import extract_colorthief, extract_pylette
from .colormind import fetch_palette, fetch_palette_with_seed
from .palette_utils import parse_colors_from_text, generate_variation

logger = logging.getLogger("palette_agent.tools")


def _palette_from_llm(
    description: str,
    existing_palette: list[list[int]] | None = None,
) -> list[list[int]] | None:
    """Ask Llama3 to suggest 6 RGB colors matching a text description."""
    logger.info("_palette_from_llm called | description: '%s'", description[:80])
    llm = ChatOllama(model="llama3.1:8b", temperature=0.8, num_predict=256)

    context = ""
    if existing_palette:
        context = f"\nCurrent palette for reference: {existing_palette}\n"
        logger.debug("Using existing palette as context: %s", existing_palette)

    prompt = (
        f'Generate exactly 6 RGB colors that match this description: "{description}"\n'
        f"{context}\n"
        "Respond with ONLY a JSON array of 6 arrays, each with 3 integers 0-255.\n"
        "Example: [[255,100,50],[200,180,60],[30,120,200],[255,200,150],[80,80,80],[240,240,230]]\n\n"
        "Your response (JSON only, no explanation):"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()
    logger.debug("LLM raw response: %s", text[:200])

    match = re.search(r'\[\s*\[.*?\]\s*\]', text, re.DOTALL)
    if match:
        try:
            colors = json.loads(match.group())
            if len(colors) >= 6:
                result = [
                    [max(0, min(255, int(c))) for c in rgb]
                    for rgb in colors[:6]
                ]
                logger.info("LLM palette generated successfully: %s", result)
                return result
            logger.warning("LLM returned %d colors, expected >= 6", len(colors))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Failed to parse LLM palette JSON: %s", exc)
    else:
        logger.warning("No JSON array found in LLM response")
    return None


def extract_colors_from_image(image_b64) -> str:
        """Extract dominant colors from the user's uploaded image."""
        logger.info("extract_colors_from_image called | image present: %s", bool(image_b64))
        if not image_b64:
            logger.warning("No image data provided")
            return "Error: No image uploaded yet."
        results = []
        ct = extract_colorthief(image_b64)
        if ct:
            results.append({"colors": ct, "source": "colorthief"})
            logger.info("colorthief extraction succeeded: %d colors", len(ct))
        else:
            logger.warning("colorthief extraction failed")
        py = extract_pylette(image_b64)
        if py:
            results.append({"colors": py, "source": "pylette"})
            logger.info("pylette extraction succeeded: %d colors", len(py))
        else:
            logger.warning("pylette extraction failed")
        logger.info("extract_colors_from_image returning %d result(s)", len(results))
        return json.dumps(results) if results else "Could not extract colors."


def generate_palette_from_description(description: str, current_palette) -> str:
        """Generate a 6-color palette from a natural language description like 'warm sunset' or 'ocean blues'."""
        logger.info("generate_palette_from_description called | description: '%s'", description[:80])
        colors = _palette_from_llm(description, current_palette)
        if colors:
            logger.info("Palette generated from description successfully")
            return json.dumps(colors)
        logger.warning("Failed to generate palette from description: '%s'", description[:80])
        return "Could not generate palette from description."


def get_random_palette() -> str:
        """Get a random harmonious 6-color palette from Colormind."""
        logger.info("get_random_palette called")
        colors = fetch_palette()
        if colors:
            logger.info("Random palette fetched: %s", colors)
            return json.dumps(colors)
        logger.warning("Colormind API returned no palette")
        return "Could not fetch palette."


def parse_user_colors(text: str) -> str:
        """Parse specific hex codes (#RRGGBB) or RGB values from text and build a 6-color palette."""
        logger.info("parse_user_colors called | text: '%s'", text[:100])
        colors = parse_colors_from_text(text)
        if not colors:
            logger.warning("No color values found in text")
            return "No color values found in text."
        logger.info("Parsed %d color(s) from text: %s", len(colors), colors)
        if len(colors) >= 6:
            logger.info("User provided >= 6 colors, using first 6")
            return json.dumps(colors[:6])
        logger.debug("User provided %d colors, filling remaining with colormind", len(colors))
        filled = fetch_palette_with_seed(colors)
        if filled:
            logger.info("Colormind seed-fill succeeded: %s", filled)
            return json.dumps(filled)
        padded = colors[:]
        while len(padded) < 6:
            padded.append(padded[-1])
        logger.info("Padded palette to 6 colors (colormind unavailable): %s", padded[:6])
        return json.dumps(padded[:6])


def create_palette_variation(variation_type: str, current_palette) -> str:
    """Create a variation of the current palette. variation_type: subtle, bold, warmer, cooler, complementary, saturated, desaturated."""
    logger.info("create_palette_variation called | type: '%s', has_palette: %s", variation_type, bool(current_palette))
    if not current_palette:
        logger.warning("No current palette to create variation from")
        return "Error: No current palette to vary."
    colors = generate_variation(current_palette, variation_type)
    logger.info("Variation '%s' generated: %s", variation_type, colors)
    return json.dumps(colors)

