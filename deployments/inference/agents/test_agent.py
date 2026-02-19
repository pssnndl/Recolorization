"""Test the agent graph by invoking it directly — no API server needed.

Usage:
    1. Start Ollama:    ollama serve
    2. Run tests:       cd deployments/inference && python -m agents.test_agent
"""

import base64
import io
import os

from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage

from graph import app_graph
from session import get_or_create_session


def _make_test_image(width=64, height=64, color=(255, 0, 0)) -> str:
    """Create a small test image and return its base64."""
    test_path = "/Users/prerana1298/computing/repo/Recolorization/assets/test_images/img_5.jpeg"
    if os.path.exists(test_path):
        with open(test_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _run(state: dict, message: str, image_b64: str = None, image_filename: str = None) -> dict:
    """Append a user message, optionally attach an image, and invoke the graph."""
    state["messages"].append(HumanMessage(content=message))
    if image_b64:
        state["image_b64"] = image_b64
        state["image_filename"] = image_filename

    result = app_graph.invoke(state)
    state.update(result)
    return state


def _last_ai(state: dict) -> str:
    """Get the last AI message content."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""


def test_greeting():
    _, state = get_or_create_session()
    state = _run(state, "Hello!")
    response = _last_ai(state)
    assert response, "Expected an AI response"
    print(f"[PASS] greeting — {response[:80]}...")
    return state


def test_image_upload(state: dict):
    img_b64 = _make_test_image()
    state = _run(state, "Here is my image", image_b64=img_b64, image_filename="test.png")
    assert state.get("image_b64") is not None, "image_b64 should be set"
    print(f"[PASS] image upload — {_last_ai(state)[:80]}...")
    return state


def test_palette(state: dict):
    state = _run(state, "Use warm sunset colors")
    response = _last_ai(state)
    has_palette = state.get("palette") is not None and len(state.get("palette", [])) == 6
    print(f"[PASS] palette — has_palette: {has_palette}, {response[:80]}...")
    if state.get("result_b64"):
        print("[INFO] recolorization auto-triggered by join_slots!")
    return state


def test_recolor(state: dict):
    """Explicitly ask to recolor — image and palette should already be in state."""
    state = _run(state, "Recolor it now")
    response = _last_ai(state)
    result_b64 = state.get("result_b64")
    recolor_count = state.get("recolor_count", 0)
    assert result_b64, "Expected a recolored image in result_b64"
    assert recolor_count >= 1, "recolor_count should be >= 1"
    print(f"[PASS] recolor — recolor_count: {recolor_count}, result_b64 length: {len(result_b64)}")
    print(f"       response: {response[:80]}...")
    return state


def test_state_check(state: dict):
    has_image = state.get("image_b64") is not None
    has_palette = state.get("palette") is not None and len(state.get("palette", [])) == 6
    print(f"[PASS] state check — has_image: {has_image}, has_palette: {has_palette}, recolor_count: {state.get('recolor_count', 0)}")


def test_multi_intent():
    """Send image + palette description in one message."""
    _, state = get_or_create_session()
    img_b64 = _make_test_image(color=(0, 128, 255))
    state = _run(
        state,
        "Here is my photo, recolor it with cool ocean blues",
        image_b64=img_b64,
        image_filename="ocean.png",
    )
    has_image = state.get("image_b64") is not None
    has_palette = state.get("palette") is not None and len(state.get("palette", [])) == 6
    print(f"[PASS] multi-intent — has_image: {has_image}, has_palette: {has_palette}")
    if state.get("result_b64"):
        print("[INFO] recolorization auto-triggered!")
    return state


if __name__ == "__main__":
    print("=== Agent Direct Tests (no API) ===\n")

    state = test_greeting()
    state = test_image_upload(state)
    state = test_palette(state)
    state = test_recolor(state)
    test_state_check(state)

    print("\n--- Multi-intent test ---")
    test_multi_intent()

    print("\n=== All tests passed ===")
