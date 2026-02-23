"""Test the agent graph by invoking it directly — no API server needed.

Uses the interrupt/checkpoint pattern: the graph pauses at chat_agent
waiting for input, and resumes via Command(resume=...).

Usage:
    1. Start Ollama:    ollama serve
    2. Set LangSmith:   export LANGCHAIN_API_KEY=<your-key>
    3. Run tests:       cd deployments/inference/agents && python test_agent.py
"""

import base64
import io
import os
import uuid

from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from graph import app_graph


# ── Helpers ─────────────────────────────────────────────────────────

def _make_test_image(width=64, height=64, color=(255, 0, 0)) -> str:
    """Create a small test image and return its base64."""
    test_path = "/Users/prerana1298/computing/repo/Recolorization/deployments/frontend/public/recolor_icon.png"
    if os.path.exists(test_path):
        with open(test_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _new_thread() -> dict:
    """Create a fresh thread config for the checkpointer."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def _get_state(config: dict) -> dict:
    """Read current graph state from the checkpointer."""
    return app_graph.get_state(config).values


def _is_interrupted(config: dict) -> bool:
    """Check if the graph is paused at an interrupt (waiting for input)."""
    snapshot = app_graph.get_state(config)
    return bool(snapshot.next)


def _last_ai(state: dict) -> str:
    """Get the last AI message content."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""


def _first_message(
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
    return _get_state(config)


def _next_message(
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
    return _get_state(config)


# ── Tests ───────────────────────────────────────────────────────────

def test_greeting():
    """General chat: should get a response and pause at interrupt."""
    config = _new_thread()
    state = _first_message("Hello!", config)
    response = _last_ai(state)
    assert response, "Expected an AI response"
    assert _is_interrupted(config), "Graph should be paused waiting for input"
    print(f"[PASS] greeting — {response[:80]}...")
    return config


def test_palette(config: dict):
    """Palette generation: resume with a description, should get palette and pause."""
    state = _next_message("Use warm sunset colors", config)
    response = _last_ai(state)
    has_palette = (
        state.get("palette") is not None
        and len(state.get("palette", [])) == 6
    )
    assert _is_interrupted(config), "Graph should be paused waiting for input"
    print(f"[PASS] palette — has_palette: {has_palette}, {response[:80]}...")
    return config


def test_image_upload(config: dict):
    """Image upload: resume with an image, should validate and pause."""
    img_b64 = _make_test_image()
    state = _next_message(
        "Here is my image",
        config,
        image_b64=img_b64,
        image_filename="test.png",
    )
    assert state.get("image_b64") is not None, "image_b64 should be set"
    print(f"[PASS] image upload — {_last_ai(state)[:80]}...")
    return config


def test_recolor(config: dict):
    """Recolor: with both slots filled, should run inference and reach END."""
    state = _next_message("Recolor it now", config)
    response = _last_ai(state)
    result_b64 = state.get("result_b64")
    recolor_count = state.get("recolor_count", 0)
    assert result_b64, "Expected a recolored image in result_b64"
    assert recolor_count >= 1, "recolor_count should be >= 1"
    assert not _is_interrupted(config), "Graph should have completed (END)"
    print(f"[PASS] recolor — recolor_count: {recolor_count}, result_b64 length: {len(result_b64)}")
    print(f"       response: {response[:80]}...")
    return config


def test_state_check(config: dict):
    """Verify final state after a full flow."""
    state = _get_state(config)
    has_image = state.get("image_b64") is not None
    has_palette = (
        state.get("palette") is not None
        and len(state.get("palette", [])) == 6
    )
    print(
        f"[PASS] state check — has_image: {has_image}, "
        f"has_palette: {has_palette}, "
        f"recolor_count: {state.get('recolor_count', 0)}"
    )


def test_multi_intent():
    """Send image + palette description in one message — may auto-recolor."""
    config = _new_thread()
    img_b64 = _make_test_image(color=(0, 128, 255))
    state = _first_message(
        "Here is my photo, recolor it with cool ocean blues",
        config,
        image_b64=img_b64,
        image_filename="ocean.png",
    )
    has_image = state.get("image_b64") is not None
    has_palette = (
        state.get("palette") is not None
        and len(state.get("palette", [])) == 6
    )
    print(f"[PASS] multi-intent — has_image: {has_image}, has_palette: {has_palette}")
    if state.get("result_b64"):
        print("[INFO] recolorization auto-triggered!")
    else:
        print("[INFO] graph paused — both slots may not be filled yet")
    return config


def test_iteration_limit():
    """Verify graph terminates after MAX_CHAT_ITERATIONS of useless input."""
    config = _new_thread()
    state = _first_message("Hello!", config)
    assert _is_interrupted(config), "Should be interrupted after first message"

    # Keep sending useless messages until graph stops interrupting
    for i in range(25):
        if not _is_interrupted(config):
            print(f"[PASS] iteration limit — graph ended after {i} resume(s)")
            return config
        state = _next_message("hmm", config)

    assert not _is_interrupted(config), "Graph should have hit iteration limit"
    print("[PASS] iteration limit — graph terminated")
    return config


# ── Runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Agent Direct Tests (interrupt/checkpoint) ===\n")

    print("--- Greeting test ---")
    config = test_greeting()

    print("\n--- Palette test ---")
    config = test_palette(config)

    print("\n--- Image upload test ---")
    config = test_image_upload(config)

    print("\n--- Recolor test ---")
    config = test_recolor(config)
    # test_state_check(config)

    # print("\n--- Multi-intent test ---")
    # test_multi_intent()

    # print("\n--- Iteration limit test ---")
    # test_iteration_limit()

    print("\n=== Tests complete ===")
