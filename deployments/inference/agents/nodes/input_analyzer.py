"""Input Analyzer — deterministic slot validation that sets routing.

Messages flow from chat_agent through state automatically.
This node only decides which worker agents to dispatch (next_nodes).

Possible next_nodes values:
- ["chat_agent"]                      no actionable intent → loop back
- ["slot_checker"]                    recolor requested, both slots already filled
- ["image_agent"]                     need image processing
- ["palette_agent"]                   need palette generation
- ["image_agent", "palette_agent"]    parallel dispatch
"""

import logging

logger = logging.getLogger("input_analyzer")

PALETTE_INTENTS = {
    "set_palette",
    "describe_palette",
    "extract_palette",
    "variation",
    "adjust_palette",
}


def input_analyzer(state: dict) -> dict:
    """
    Deterministic execution planner.
    Converts user_intents + slot completeness into
    an execution plan (next_nodes).
    """

    intents = state.get("user_intents", [])

    has_image = state.get("image_b64") is not None
    has_palette = (
        state.get("palette") is not None
        and len(state.get("palette", [])) == 6
    )

    logger.info(
        "input_analyzer | intents=%s, has_image=%s, has_palette=%s",
        intents, has_image, has_palette,
    )

    execution_plan = []

    # --- Image intent ---
    if "upload_image" in intents:
        execution_plan.append("image_agent")

    # --- Palette intents ---
    if any(i in intents for i in PALETTE_INTENTS):
        execution_plan.append("palette_agent")

    # --- Recolor intent ---
    # recolor_agent is only reachable via slot_checker, never dispatched directly.
    if "recolor" in intents:
        if has_image and has_palette and len(execution_plan) == 0:
            # Both slots filled, no other agents needed — shortcut to slot_checker
            logger.info("Recolor shortcut: both slots filled → slot_checker")
            return {"next_nodes": ["slot_checker"]}
        # Otherwise ensure prerequisite agents run
        if not has_image and "image_agent" not in execution_plan:
            execution_plan.append("image_agent")
        if not has_palette and "palette_agent" not in execution_plan:
            execution_plan.append("palette_agent")

    # --- Nothing actionable → loop back to chat_agent ---
    if not execution_plan:
        logger.info("No actionable intent → routing back to chat_agent")
        return {"next_nodes": ["chat_agent"]}

    # Remove duplicates while preserving order
    execution_plan = list(dict.fromkeys(execution_plan))

    logger.info("Execution plan: %s", execution_plan)
    return {"next_nodes": execution_plan}
