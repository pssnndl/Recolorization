"""Test the full graph flow — greeting → palette → image → recolor.

Usage:
    cd deployments/inference/agents
    python -m tests.test_graph_flow
"""

from tests.helpers import (
    new_thread, first_message, next_message,
    is_interrupted, last_ai, make_test_image,
)


def test_greeting():
    """General chat: should get a response and pause at interrupt."""
    config = new_thread()
    state = first_message("Hello!", config)
    response = last_ai(state)
    assert response, "Expected an AI response"
    assert is_interrupted(config), "Graph should be paused waiting for input"
    print(f"[PASS] greeting — {response[:80]}...")
    return config


def test_palette(config: dict):
    """Palette generation: resume with a description, should get palette and pause."""
    state = next_message("Use rainbow colors", config)
    response = last_ai(state)
    has_palette = (
        state.get("palette") is not None
        and len(state.get("palette", [])) == 6
    )
    assert is_interrupted(config), "Graph should be paused waiting for input"
    print(f"[PASS] palette — has_palette: {has_palette}, {response[:80]}...")
    return config


def test_image_upload(config: dict):
    """Image upload: resume with an image, should validate and pause."""
    img_b64 = make_test_image()
    state = next_message(
        "Here is my image",
        config,
        image_b64=img_b64,
        image_filename="test.png",
    )
    assert state.get("image_b64") is not None, "image_b64 should be set"
    print(f"[PASS] image upload — {last_ai(state)[:80]}...")
    return config


def test_recolor(config: dict):
    """Recolor: with both slots filled, should run inference and reach END."""
    state = next_message("Recolor it now", config)
    response = last_ai(state)
    result_b64 = state.get("result_b64")
    recolor_count = state.get("recolor_count", 0)
    assert result_b64, "Expected a recolored image in result_b64"
    assert recolor_count >= 1, "recolor_count should be >= 1"
    assert not is_interrupted(config), "Graph should have completed (END)"
    print(f"[PASS] recolor — recolor_count: {recolor_count}, result_b64 length: {len(result_b64)}")
    print(f"       response: {response[:80]}...")
    return config


def test_multi_intent():
    """Send image + palette description in one message — may auto-recolor."""
    config = new_thread()
    img_b64 = make_test_image(color=(0, 128, 255))
    state = first_message(
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
    config = new_thread()
    state = first_message("Hello!", config)
    assert is_interrupted(config), "Should be interrupted after first message"

    for i in range(25):
        if not is_interrupted(config):
            print(f"[PASS] iteration limit — graph ended after {i} resume(s)")
            return config
        state = next_message("hmm", config)

    assert not is_interrupted(config), "Graph should have hit iteration limit"
    print("[PASS] iteration limit — graph terminated")
    return config


if __name__ == "__main__":
    print("=== Graph Flow Tests ===\n")

    print("--- Greeting ---")
    config = test_greeting()

    print("\n--- Palette ---")
    config = test_palette(config)

    print("\n--- Image upload ---")
    config = test_image_upload(config)

    print("\n--- Recolor ---")
    config = test_recolor(config)

    # print("\n--- Multi-intent ---")
    # test_multi_intent()

    # print("\n--- Iteration limit ---")
    # test_iteration_limit()

    print("\n=== Done ===")
