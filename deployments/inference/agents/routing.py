"""Conditional edge routing functions for the LangGraph state machine."""

import logging

from langgraph.types import Send

logger = logging.getLogger("routing")

# Maximum times chat_agent may run before the graph force-terminates.
MAX_CHAT_ITERATIONS = 20



def _iteration_limit_reached(state: dict) -> bool:
    """Check whether chat_iterations has exceeded the cap."""
    return state.get("chat_iterations", 0) >= MAX_CHAT_ITERATIONS


def route_after_analyzer(state: dict) -> str | list:
    """
    Routes after input_analyzer.
    - "chat_agent"   : no actionable intent → loop back (interrupt waits for input)
    - "__end__"      : iteration limit reached → terminate graph
    - "slot_checker" : recolor shortcut (both slots already filled)
    - single agent   : "image_agent" or "palette_agent"
    - list[Send]     : parallel fan-out for multiple agents
    """
    next_nodes = state.get("next_nodes", ["chat_agent"])

    if len(next_nodes) == 1:
        target = next_nodes[0]

        # If looping back, check iteration limit first
        if target == "chat_agent" and _iteration_limit_reached(state):
            logger.warning(
                "Iteration limit (%d) reached at route_after_analyzer — ending graph",
                MAX_CHAT_ITERATIONS,
            )
            return "__end__"

        logger.info("route_after_analyzer → %s", target)
        return target

    # Multi-agent: fan out via Send()
    sends = []
    for node in next_nodes:
        sends.append(Send(node, {**state}))

    logger.info(
        "route_after_analyzer → parallel fan-out: %s",
        [s.node for s in sends],
    )
    return sends


def route_after_slot_check(state: dict) -> str:
    """
    Routes after slot_checker based on slot completeness.
    - "recolor_agent" : both image and palette ready
    - "chat_agent"    : incomplete → loop back for more input
    - "__end__"       : incomplete but iteration limit reached
    """
    target = state.get("next_node", "chat_agent")

    if target == "chat_agent" and _iteration_limit_reached(state):
        logger.warning(
            "Iteration limit (%d) reached at route_after_slot_check — ending graph",
            MAX_CHAT_ITERATIONS,
        )
        return "__end__"

    logger.info("route_after_slot_check → %s", target)
    return target
