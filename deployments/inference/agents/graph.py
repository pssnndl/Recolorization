"""LangGraph state machine — assembles all nodes and edges."""

from langgraph.graph import START, StateGraph, END

from .state import RecolorState
from .nodes.chat_agent import chat_agent
from .nodes.input_analyzer import input_analyzer
from .nodes.image_agent import image_agent
from .nodes.palette_agent import palette_agent
from .nodes.recolor_agent import recolor_agent
from .nodes.respond import respond
from .routing import (
    route_after_analyzer,
    route_after_join,
)

def join_slots(state: dict):
    has_image = state.get("image_b64") is not None
    has_palette = (
        state.get("palette") is not None
        and len(state.get("palette", [])) == 6
    )

    if has_image and has_palette:
        return {"next_node": "recolor_agent"}

    return {"next_node": "respond"}



def build_graph():
    graph = StateGraph(RecolorState)

    # --- Add nodes ---
    graph.add_node("chat_agent", chat_agent)
    graph.add_node("input_analyzer", input_analyzer)
    graph.add_node("image_agent", image_agent)
    graph.add_node("palette_agent", palette_agent)
    graph.add_node("join_slots", join_slots)
    graph.add_node("recolor_agent", recolor_agent)
    graph.add_node("respond", respond)

    # --- Entry point ---
    graph.add_edge(START, "chat_agent")

    # chat_agent NEVER routes anymore
    graph.add_edge("chat_agent", "input_analyzer")

    # --- Edges from input_analyzer ---
    # No path_map — route_after_analyzer may return Send() objects for multi-agent dispatch
    graph.add_conditional_edges("input_analyzer", route_after_analyzer)

    graph.add_edge("image_agent", "join_slots")
    graph.add_edge("palette_agent", "join_slots")

    # --- Join decides recolor ---
    graph.add_conditional_edges(
        "join_slots",
        route_after_join,
        {
            "recolor_agent": "recolor_agent",
            "respond": "respond",
        },
    )

    # --- Final ---
    graph.add_edge("recolor_agent", "respond")
    graph.add_edge("respond", END)



    return graph.compile()


# Singleton compiled graph
app_graph = build_graph()
