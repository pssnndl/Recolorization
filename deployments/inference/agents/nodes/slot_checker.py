"""Slot Checker — validates that image + palette are ready for recolorization.

Replaces the old join_slots + respond pattern. If slots are incomplete,
produces an informative message and routes back to chat_agent for more input.
If both slots are filled, routes to recolor_agent.
"""

import logging

from langchain_core.messages import AIMessage

logger = logging.getLogger("slot_checker")


def slot_checker(state: dict) -> dict:
    """
    Check whether image and 6-color palette are both available.

    Returns:
        next_node: "recolor_agent" if ready, "chat_agent" if not.
        messages:  informative message when slots are incomplete.
    """
    has_image = state.get("image_b64") is not None
    has_palette = (
        state.get("palette") is not None
        and len(state.get("palette", [])) == 6
    )

    logger.info(
        "slot_checker | has_image=%s, has_palette=%s",
        has_image, has_palette,
    )

    if has_image and has_palette:
        logger.info("Both slots filled — routing to recolor_agent")
        return {"next_node": "recolor_agent"}

    # Build a helpful message about what is missing
    missing = []
    if not has_image:
        missing.append("an image (drag and drop or click upload)")
    if not has_palette:
        missing.append(
            "a 6-color palette (describe a theme, provide hex codes, "
            "or ask me to extract colors from your image)"
        )

    msg = (
        "Almost there! To recolor, I still need:\n"
        + "\n".join(f"- {m}" for m in missing)
    )

    logger.info("Slots incomplete — routing back to chat_agent")
    return {
        "next_node": "chat_agent",
        "messages": [AIMessage(content=msg)],
    }
