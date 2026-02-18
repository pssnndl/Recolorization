"""Palette Agent — sources palettes via multiple strategies based on user intent."""

import json
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage

from ..state import PaletteCandidate
from ..tools.colormind import fetch_palette, fetch_palette_with_seed
from ..tools.color_extraction import extract_colorthief, extract_pylette
from ..tools.palette_utils import (
    generate_variation,
    detect_variation_type,
    parse_colors_from_text,
    palette_to_hex,
)


def _palette_from_llm(
    description: str,
    existing_palette: list[list[int]] | None = None,
) -> list[list[int]] | None:
    """Ask Llama3 to suggest 6 RGB colors matching a text description."""
    llm = ChatOllama(model="llama3", temperature=0.8, num_predict=256)

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

    # Extract JSON array from response
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


def palette_agent(state: dict) -> dict:
    # user_intent is set by Send() in multi-intent path; fall back to user_intents list
    intent = state.get("user_intent")
    if not intent:
        palette_intents = {"set_palette", "describe_palette", "extract_palette", "variation", "adjust_palette"}
        intents = state.get("user_intents", [])
        intent = next((i for i in intents if i in palette_intents), "describe_palette")
    last_msg = state["messages"][-1].content if state["messages"] else ""
    image_b64 = state.get("image_b64")
    current_palette = state.get("palette")
    candidates: list[PaletteCandidate] = []

    # --- Branch 1: Extract from image ---
    if intent == "extract_palette":
        target_image = image_b64
        if target_image:
            ct_palette = extract_colorthief(target_image)
            if ct_palette:
                candidates.append({
                    "colors": ct_palette,
                    "source": "colorthief",
                    "description": "Dominant colors from your image (ColorThief)",
                })
            py_palette = extract_pylette(target_image)
            if py_palette:
                candidates.append({
                    "colors": py_palette,
                    "source": "pylette",
                    "description": "Color palette from your image (Pylette)",
                })
        else:
            return {
                "messages": [AIMessage(content=(
                    "I need an image to extract colors from. "
                    "Please upload one first."
                ))],
            }

    # --- Branch 2: LLM-described palette ---
    elif intent == "describe_palette":
        llm_palette = _palette_from_llm(last_msg, current_palette)
        if llm_palette:
            candidates.append({
                "colors": llm_palette,
                "source": "llm_suggested",
                "description": f"AI-suggested palette for: {last_msg[:80]}",
            })
        # Also try Colormind for variety
        cm_palette = fetch_palette()
        if cm_palette:
            candidates.append({
                "colors": cm_palette,
                "source": "colormind_api",
                "description": "Random harmonious palette from Colormind",
            })

    # --- Branch 3: User providing specific colors ---
    elif intent == "set_palette":
        parsed_colors = parse_colors_from_text(last_msg)

        if parsed_colors:
            if len(parsed_colors) >= 6:
                candidates.append({
                    "colors": parsed_colors[:6],
                    "source": "user_manual",
                    "description": "Your hand-picked palette",
                })
            else:
                # Fewer than 6 — auto-complete via Colormind
                filled = fetch_palette_with_seed(parsed_colors)
                if filled:
                    candidates.append({
                        "colors": filled,
                        "source": "colormind_api",
                        "description": (
                            f"Your {len(parsed_colors)} color(s) + "
                            "auto-completed to 6"
                        ),
                    })
                # Also offer the raw colors padded
                if len(parsed_colors) >= 2:
                    padded = parsed_colors[:]
                    while len(padded) < 6:
                        padded.append(padded[-1])
                    candidates.append({
                        "colors": padded[:6],
                        "source": "user_manual",
                        "description": f"Your {len(parsed_colors)} color(s), padded to 6",
                    })
        else:
            # Could not parse — fall back to LLM interpretation
            llm_palette = _palette_from_llm(last_msg, current_palette)
            if llm_palette:
                candidates.append({
                    "colors": llm_palette,
                    "source": "llm_suggested",
                    "description": "Interpreted palette from your description",
                })

    # --- Branch 4: Variation of current palette ---
    elif intent in ("variation", "adjust_palette"):
        if current_palette:
            vtype = detect_variation_type(last_msg)
            # Generate the detected type + a few alternatives
            for t in [vtype, "subtle", "bold", "warmer", "cooler"]:
                v = generate_variation(current_palette, t)
                candidates.append({
                    "colors": v,
                    "source": "variation",
                    "description": f"{t.capitalize()} variation",
                })
            # Deduplicate by removing duplicate variation types
            seen = set()
            unique = []
            for c in candidates:
                if c["description"] not in seen:
                    seen.add(c["description"])
                    unique.append(c)
            candidates = unique[:4]  # Max 4 candidates
        else:
            cm_palette = fetch_palette()
            if cm_palette:
                candidates.append({
                    "colors": cm_palette,
                    "source": "colormind_api",
                    "description": "Suggested palette (no existing palette to vary)",
                })

    # --- Fallback ---
    if not candidates:
        fallback = fetch_palette()
        if fallback:
            candidates.append({
                "colors": fallback,
                "source": "colormind_api",
                "description": "Auto-generated harmonious palette",
            })

    if candidates:
        selected = candidates[0]

        lines = ["Here are some palette options:\n"]
        for i, c in enumerate(candidates):
            hex_display = palette_to_hex(c["colors"])
            marker = " (selected)" if i == 0 else ""
            lines.append(
                f"{i+1}. **{c['description']}**{marker}\n   {hex_display}"
            )

        lines.append(
            "\nI've selected option 1. Say 'use 2' to pick another, "
            "'vary' for variations, or describe what you'd like to change."
        )

        return {
            "palette": selected["colors"],
            "palette_candidates": candidates,
            "palette_source": selected["source"],
            "error": None,
            "messages": [AIMessage(content="\n".join(lines))],
        }
    else:
        return {
            "error": "Could not generate any palettes",
            "messages": [AIMessage(content=(
                "I wasn't able to generate a palette. Could you try "
                "describing the colors you want in a different way?"
            ))],
        }
