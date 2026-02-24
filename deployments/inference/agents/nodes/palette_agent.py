"""Palette Agent — uses LLM tool calling to create palettes."""

import json
import logging

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langsmith import traceable
from pydantic import BaseModel, Field, field_validator
import sys

sys.path.insert(0, "../")
from state import PaletteCandidate
from tools.colormind import fetch_palette
from tools.palette_utils import (
    palette_to_hex,
)
from tools.palette_formation import generate_palette_from_description, get_random_palette, parse_user_colors, create_palette_variation

logger = logging.getLogger("palette_agent")


# ── Strict output schemas ──────────────────────────────────────────────

class RGBColor(BaseModel):
    """A single RGB color value."""
    r: int = Field(..., ge=0, le=255)
    g: int = Field(..., ge=0, le=255)
    b: int = Field(..., ge=0, le=255)

    @classmethod
    def from_list(cls, rgb: list[int]) -> "RGBColor":
        return cls(r=rgb[0], g=rgb[1], b=rgb[2])

    def to_list(self) -> list[int]:
        return [self.r, self.g, self.b]


class PaletteCandidateSchema(BaseModel):
    """Validated palette candidate."""
    colors: list[RGBColor] = Field(..., min_length=6, max_length=6)
    source: str
    description: str

    @classmethod
    def from_dict(cls, d: dict) -> "PaletteCandidateSchema":
        return cls(
            colors=[RGBColor.from_list(c) for c in d["colors"]],
            source=d["source"],
            description=d["description"],
        )

    def to_state_dict(self) -> PaletteCandidate:
        return {
            "colors": [c.to_list() for c in self.colors],
            "source": self.source,
            "description": self.description,
        }


class PaletteAgentOutput(BaseModel):
    """Strict schema for palette agent return value."""
    palette: list[RGBColor] = Field(..., min_length=6, max_length=6)
    palette_candidates: list[PaletteCandidateSchema] = Field(..., min_length=1)
    palette_source: str
    error: None = None

    @field_validator("palette_candidates")
    @classmethod
    def all_candidates_have_six_colors(cls, v: list[PaletteCandidateSchema]) -> list[PaletteCandidateSchema]:
        for i, c in enumerate(v):
            if len(c.colors) != 6:
                raise ValueError(f"Candidate {i} has {len(c.colors)} colors, expected 6")
        return v



PALETTE_SYSTEM = """You are a palette creation assistant. Your ONLY job is to call tools — NEVER respond with plain text.

RULES:
1. You MUST call exactly one tool for every user message.
2. Do NOT write explanations. Just call the tool.

TOOL SELECTION — pick based on the user message:

| User says…                              | Tool to call                          |
|----------------------------------------|---------------------------------------|
| Describes a mood, theme, or colors     | generate_palette_from_description     |
| (e.g. "warm", "ocean", "rainbow",     |                                       |
|  "sunset", "forest", "neon", etc.)     |                                       |
| Gives hex codes like #FF6B6B           | parse_user_colors                     |
| Gives RGB values like (255, 100, 50)   | parse_user_colors                     |
| Says "random" or "surprise me"         | get_random_palette                    |
| Wants to adjust/tweak current palette  | create_palette_variation              |

When in doubt, use generate_palette_from_description — it handles any text description."""

TOOLS = [
        generate_palette_from_description,
        get_random_palette,
        parse_user_colors,
        create_palette_variation,
    ]

def call_model(state: MessagesState):
    """Invoke the LLM with the current messages and bound tools."""
    model = ChatOllama(model="llama3.1:8b", temperature=0).bind_tools(TOOLS)
    logger.debug("Invoking LLM with %d message(s)", len(state["messages"]))
    response = model.invoke(state["messages"])
    if response.tool_calls:
        logger.info(
            "LLM requested tool calls: %s",
            [tc["name"] for tc in response.tool_calls],
        )
    else:
        logger.debug("LLM returned text response (no tool calls)")
    return {"messages": [response]}


def build_palette_agent():
    """Build and compile the LangGraph agent.

    Returns the compiled StateGraph ready for .invoke().
    """
    graph = StateGraph(MessagesState)

    # Nodes
    graph.add_node("call_model", call_model)
    graph.add_node("tools", ToolNode(TOOLS))

    # Edges
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", tools_condition)
    graph.add_edge("tools", "call_model")

    agent = graph.compile()
    return agent

@traceable(run_type="chain", name="palette_interface")
def palette_agent(state: dict) -> dict:
    """Outer-graph node: invokes the palette sub-graph and parses results."""
    last_msg = state["messages"][-1].content if state["messages"] else ""
    logger.info("palette_agent invoked | user_message: %s", last_msg[:120])

    # Build and invoke the palette sub-graph
    agent = build_palette_agent()
    logger.debug("Sub-graph compiled, invoking with system + user message")
    result = agent.invoke({
        "messages": [
            SystemMessage(content=PALETTE_SYSTEM),
            HumanMessage(content=last_msg),
        ]
    })
    logger.debug(
        "Sub-graph returned %d message(s): types=%s",
        len(result["messages"]),
        [type(m).__name__ for m in result["messages"]],
    )

    # Parse ToolMessage results to collect palette candidates
    candidates: list[PaletteCandidate] = []
    for msg in result["messages"]:
        if not isinstance(msg, ToolMessage):
            continue
        tool_name = getattr(msg, "name", "unknown")
        logger.debug("Processing ToolMessage from '%s': %s", tool_name, msg.content[:200])
        try:
            parsed = json.loads(msg.content)
            if isinstance(parsed, list) and all(isinstance(p, dict) for p in parsed):
                # extract_colors_from_image returns [{colors, source}, ...]
                for p in parsed:
                    candidates.append({
                        "colors": p["colors"],
                        "source": p["source"],
                        "description": f"Extracted ({p['source']})",
                    })
                logger.info("Parsed %d extraction candidate(s) from '%s'", len(parsed), tool_name)
            elif isinstance(parsed, list) and len(parsed) == 6:
                candidates.append({
                    "colors": parsed,
                    "source": tool_name,
                    "description": f"Generated by {tool_name}",
                })
                logger.info("Parsed 6-color palette from '%s'", tool_name)
            else:
                logger.warning(
                    "ToolMessage from '%s' had unexpected structure: type=%s len=%s",
                    tool_name, type(parsed).__name__,
                    len(parsed) if isinstance(parsed, list) else "N/A",
                )
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.error("Failed to parse ToolMessage from '%s': %s", tool_name, exc)

    logger.info("Collected %d palette candidate(s) total", len(candidates))

    # Build response for the outer graph
    if candidates:
        # Validate every candidate through the strict schema
        validated: list[PaletteCandidate] = []
        for i, c in enumerate(candidates):
            try:
                schema = PaletteCandidateSchema.from_dict(c)
                validated.append(schema.to_state_dict())
            except Exception as exc:
                logger.warning("Dropping candidate %d (%s): validation failed — %s", i, c.get("source"), exc)

        if not validated:
            logger.error("All %d candidate(s) failed schema validation", len(candidates))
            # fall through to the fallback path below
        else:
            selected = validated[0]
            lines = ["Here are some palette options:\n"]
            for i, c in enumerate(validated):
                hex_display = palette_to_hex(c["colors"])
                marker = " (selected)" if i == 0 else ""
                lines.append(
                    f"{i+1}. **{c['description']}**{marker}\n   {hex_display}"
                )
            lines.append(
                "\nI've selected option 1. Say 'use 2' to pick another, "
                "'vary' for variations, or describe what you'd like to change."
            )

            # Final output validation
            try:
                output = PaletteAgentOutput(
                    palette=[RGBColor.from_list(c) for c in selected["colors"]],
                    palette_candidates=[PaletteCandidateSchema.from_dict(c) for c in validated],
                    palette_source=selected["source"],
                )
                logger.info(
                    "Returning %d validated candidate(s) | selected source: %s",
                    len(validated), selected["source"],
                )
            except Exception as exc:
                logger.error("Final output validation failed: %s", exc)
                # fall through to fallback
            else:
                return {
                    "palette": selected["colors"],
                    "palette_candidates": validated,
                    "palette_source": selected["source"],
                    "error": None,
                    "messages": [AIMessage(content="\n".join(lines))],
                }

    # Fallback — no tool produced valid results
    logger.warning("No valid candidates from tools, attempting colormind fallback")
    fallback = fetch_palette()
    if fallback:
        try:
            fb_candidate = PaletteCandidateSchema.from_dict({
                "colors": fallback,
                "source": "colormind_api",
                "description": "Auto-generated palette",
            })
            logger.info("Colormind fallback succeeded: %s", palette_to_hex(fallback))
        except Exception as exc:
            logger.error("Colormind fallback palette failed validation: %s", exc)
            fb_candidate = None

        if fb_candidate:
            return {
                "palette": fallback,
                "palette_candidates": [fb_candidate.to_state_dict()],
                "palette_source": "colormind_api",
                "error": None,
                "messages": [AIMessage(content=(
                    "Here's an auto-generated palette:\n"
                    + palette_to_hex(fallback)
                ))],
            }

    logger.error("All palette generation paths failed")
    return {
        "error": "Could not generate any palettes",
        "messages": [AIMessage(content=(
            "I wasn't able to generate a palette. Could you try "
            "describing the colors you want in a different way?"
        ))],
    }
