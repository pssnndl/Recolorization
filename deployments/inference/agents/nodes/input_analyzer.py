"""Input Analyzer — deterministic slot validation that sets routing.

Messages flow from chat_agent through state automatically.
This node only decides which agents to dispatch (next_nodes).
"""



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

    execution_plan = []

    # --- Image Intent ---
    if "upload_image" in intents:
        execution_plan.append("image_agent")

    # --- Palette Intents ---
    if any(i in intents for i in [
        "set_palette",
        "describe_palette",
        "extract_palette",
        "variation",
        "adjust_palette"
    ]):
        execution_plan.append("palette_agent")

    # --- Recolor Intent ---
    if "recolor" in intents:
        if has_image and has_palette:
            execution_plan.append("recolor_agent")
        else:
            if not has_image:
                execution_plan.append("image_agent")
            if not has_palette:
                execution_plan.append("palette_agent")

    # --- Nothing actionable ---
    if not execution_plan:
        execution_plan = ["respond"]

    # Remove duplicates while preserving order
    execution_plan = list(dict.fromkeys(execution_plan))

    return {"next_nodes": execution_plan}