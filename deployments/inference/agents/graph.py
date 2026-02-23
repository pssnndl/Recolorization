"""LangGraph state machine — assembles all nodes and edges.

Topology (looping graph with interrupt at chat_agent):

              ┌──────────────────────────────────────┐
              │                                      │
              ▼                                      │
  START → chat_agent ─(interrupt: wait for input)    │
              │                                      │
              ▼                                      │
        input_analyzer                               │
              │                                      │
    ┌─────────┼──────────────┐                       │
    │         │              │                       │
(no intent) (shortcut)  (has intent)                 │
    │         │         ┌────┴────┐                  │
    │         │         ▼         ▼                  │
    │         │   image_agent  palette_agent          │
    │         │         │         │                  │
    │         │         └────┬────┘                  │
    │         │              ▼                       │
    │         └──────► slot_checker                  │
    │                   │        │                   │
    │              (ready)  (incomplete)              │
    │ (under limit)     │        │                   │
    │    ───────────────│   (under limit) ───────────┘
    │                   │        │
    │ (limit hit)       │   (limit hit) ──► END
    │    ──► END        ▼
    │             recolor_agent
    │                   │
    └───────────────    ▼
                       END

Both loop-back points (no-intent and incomplete-slots) check
chat_iterations against MAX_CHAT_ITERATIONS before looping.
If the limit is hit, the graph routes to END instead.
"""

import os
from dotenv import load_dotenv

# Load .env before any langchain/langsmith imports so tracing env vars are set
load_dotenv(".env")

from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import RecolorState
from nodes.chat_agent import chat_agent
from nodes.input_analyzer import input_analyzer
from nodes.image_agent import image_agent
from nodes.palette_agent import palette_agent
from nodes.slot_checker import slot_checker
from nodes.recolor_agent import recolor_agent
from routing import (
    route_after_analyzer,
    route_after_slot_check,
)


def build_graph():
    graph = StateGraph(RecolorState)

    # --- Nodes ---
    graph.add_node("chat_agent", chat_agent)
    graph.add_node("input_analyzer", input_analyzer)
    graph.add_node("image_agent", image_agent)
    graph.add_node("palette_agent", palette_agent)
    graph.add_node("slot_checker", slot_checker)
    graph.add_node("recolor_agent", recolor_agent)

    # --- Entry point ---
    graph.add_edge(START, "chat_agent")

    # chat_agent always flows to input_analyzer
    graph.add_edge("chat_agent", "input_analyzer")

    # --- After analysis: fan-out, slot_checker shortcut, loop back, or end ---
    # route_after_analyzer may return:
    #   - "chat_agent"    (no actionable intent → loop back, interrupt waits for input)
    #   - "__end__"       (iteration limit reached → terminate graph)
    #   - "slot_checker"  (recolor requested, both slots already filled)
    #   - single agent name ("image_agent" / "palette_agent")
    #   - list of Send() objects for parallel dispatch
    #   Note: Send() objects bypass the path map; it only applies to string returns.
    graph.add_conditional_edges(
        "input_analyzer",
        route_after_analyzer,
        {
            "chat_agent": "chat_agent",
            "__end__": END,
            "slot_checker": "slot_checker",
            "image_agent": "image_agent",
            "palette_agent": "palette_agent",
        },
    )

    # --- Worker agents converge at slot_checker ---
    graph.add_edge("image_agent", "slot_checker")
    graph.add_edge("palette_agent", "slot_checker")

    # --- Slot check decides: recolor, loop back, or end (iteration limit) ---
    graph.add_conditional_edges(
        "slot_checker",
        route_after_slot_check,
        {
            "recolor_agent": "recolor_agent",
            "chat_agent": "chat_agent",
            "__end__": END,
        },
    )

    # --- Recolor terminates the graph ---
    graph.add_edge("recolor_agent", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Singleton compiled graph
app_graph = build_graph()
