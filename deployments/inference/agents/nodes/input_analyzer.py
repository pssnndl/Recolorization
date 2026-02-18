"""Input Analyzer — deterministic slot validation that overrides LLM routing."""

from langchain_core.messages import AIMessage


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
    messages = []

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

        # both slots ready → recolor directly
        if has_image and has_palette:
            execution_plan.append("recolor_agent")

        else:
            # missing slots → schedule required agents
            if not has_image:
                execution_plan.append("image_agent")
                messages.append(
                    AIMessage(
                        content="I'd love to recolor that — please upload an image first!"
                    )
                )

            if not has_palette:
                execution_plan.append("palette_agent")
                messages.append(
                    AIMessage(
                        content=(
                            "Almost ready! I just need a 6-color palette. "
                            "Describe a mood or I can suggest one."
                        )
                    )
                )

    # --- Nothing actionable ---
    if not execution_plan:
        execution_plan = ["respond"]

    # Remove duplicates while preserving order
    execution_plan = list(dict.fromkeys(execution_plan))

    return {
        "next_nodes": execution_plan,
        "messages": messages if messages else None,
    }