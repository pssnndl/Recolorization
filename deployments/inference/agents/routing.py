"""Conditional edge routing functions for the LangGraph state machine."""

from langgraph.types import Send

# Maps intent labels to agent node names (for Send() context)
_INTENT_TO_AGENT: dict[str, str] = {
    "upload_image":     "image_agent",
    "set_palette":      "palette_agent",
    "describe_palette": "palette_agent",
    "extract_palette":  "palette_agent",
    "variation":        "palette_agent",
    "adjust_palette":   "palette_agent",
}


def route_after_analyzer(state: dict) -> str | list:
    """
    Routes after input_analyzer.
    - Single agent: returns the node name string.
    - Multiple agents: returns Send() objects for parallel fan-out.
    """
    next_nodes = state.get("next_nodes", ["respond"])

    if len(next_nodes) == 1:
        return next_nodes[0]

    # Multi-agent: fan out via Send(), passing user_intent per agent
    intents = state.get("user_intents", [])
    sends = []
    for node in next_nodes:
        agent_intent = next(
            (i for i in intents if _INTENT_TO_AGENT.get(i) == node), None
        )
        sends.append(Send(node, {**state, "user_intent": agent_intent}))
    return sends


def route_after_join(state: dict) -> str:
    """Routes after join_slots based on slot completeness."""
    return state.get("next_node", "respond")
