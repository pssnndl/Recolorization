"""Shared test helpers for the agent graph."""

import base64
import io
import os
import uuid

from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from graph import app_graph


def make_test_image(width=64, height=64, color=(255, 0, 0)) -> str:
    """Create a small test image and return its base64."""
    test_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "frontend", "public", "recolor_icon.png",
    )
    if os.path.exists(test_path):
        with open(test_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def new_thread() -> dict:
    """Create a fresh thread config for the checkpointer."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def get_state(config: dict) -> dict:
    """Read current graph state from the checkpointer."""
    return app_graph.get_state(config).values


def is_interrupted(config: dict) -> bool:
    """Check if the graph is paused at an interrupt."""
    snapshot = app_graph.get_state(config)
    return bool(snapshot.next)


def last_ai(state: dict) -> str:
    """Get the last AI message content."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""


def first_message(
    message: str,
    config: dict,
    image_b64: str = None,
    image_filename: str = None,
) -> dict:
    """Start a new graph execution with the first user message."""
    initial = {"messages": [HumanMessage(content=message)]}
    if image_b64:
        initial["image_b64"] = image_b64
        initial["image_filename"] = image_filename
    app_graph.invoke(initial, config)
    return get_state(config)


def next_message(
    message: str,
    config: dict,
    image_b64: str = None,
    image_filename: str = None,
) -> dict:
    """Resume the graph from an interrupt with a new user message."""
    resume_data = {"message": message}
    if image_b64:
        resume_data["image_b64"] = image_b64
        resume_data["image_filename"] = image_filename
    app_graph.invoke(Command(resume=resume_data), config)
    return get_state(config)
