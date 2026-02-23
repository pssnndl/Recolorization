"""Chat Agent — orchestrator that classifies intent and generates conversational responses.

Uses interrupt() to pause the graph and wait for user input on loop
iterations.  On first entry the user message is already in state; on
subsequent entries (looped back from input_analyzer or slot_checker)
the graph pauses here and the API layer resumes with
Command(resume=<new_user_message>).
"""

import logging

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from langgraph.types import interrupt
import sys
sys.path.insert(0, "../")
from tools.palette_utils import palette_display

logger = logging.getLogger("chat_agent")

SYSTEM_PROMPT = """You are a creative recolorization assistant. You help users recolor images by guiding them through:
1. Uploading a source image
2. Building or choosing a 6-color palette
3. Running the recolorization model
4. Iterating on the result with palette adjustments

Keep responses concise and friendly. When the user provides an image or palette info, acknowledge it and guide them to the next step.

Current state:
- Image uploaded: {has_image}
- Palette ready: {has_palette}
- Palette colors: {palette_str}
- Times recolored: {recolor_count}

Respond naturally. Do NOT output JSON or structured data."""

INTENT_PROMPT = """Given this user message and conversation context, classify ALL applicable intents.

Respond with ONE OR MORE of these labels separated by commas (no spaces):

upload_image
set_palette
describe_palette
extract_palette
recolor
adjust_palette
variation
general_chat

Examples:
"here's an image recolor it warm" → upload_image,describe_palette,recolor
"make it more blue" → adjust_palette,recolor
"new image use same palette" → upload_image,recolor

User message: {user_message}
Has image: {has_image}
Has palette: {has_palette}

Intents:"""

VALID_INTENTS = {
    "upload_image", "set_palette", "describe_palette", "extract_palette",
    "recolor", "adjust_palette", "variation", "general_chat",
}

MAX_CONTEXT_MESSAGES = 20


@traceable(run_type="llm", name="chat_interface")
def chat_agent(state: dict) -> dict:
    iteration = state.get("chat_iterations", 0) + 1
    logger.info(
        "chat_agent invoked | iteration=%d", iteration,
    )

    # ── Pause and wait for new user input ───────────────────────────
    # On the very first call the user message is already in state.
    # On subsequent loops (from input_analyzer or slot_checker) the
    # graph pauses here; the API layer resumes with
    # Command(resume={"message": "...", ...}).
    if iteration > 1:
        logger.info("Iteration > 1 — interrupting to wait for user input")
        user_input = interrupt(
            {"type": "waiting_for_input", "iteration": iteration}
        )
        # user_input is the dict passed via Command(resume=...)
        new_message = user_input.get("message", "")
        logger.info("Resumed with user message: %s", new_message[:120])

        # Inject the new message into state so downstream nodes see it
        new_human_msg = HumanMessage(content=new_message)

        # If the API layer also passes image data, surface it
        if user_input.get("image_b64"):
            state["image_b64"] = user_input["image_b64"]
            state["image_filename"] = user_input.get("image_filename")
    else:
        new_human_msg = None

    llm = ChatOllama(model="llama3.1:8b", temperature=0.7, num_predict=512)
    intent_llm = ChatOllama(model="llama3.1:8b", temperature=0.0, num_predict=64)

    has_image = state.get("image_b64") is not None
    has_palette = (
        state.get("palette") is not None
        and len(state.get("palette", [])) == 6
    )

    # Use the new message if we resumed, otherwise the latest in state
    last_message = new_human_msg if new_human_msg else state["messages"][-1]

    # ── Intent classification ───────────────────────────────────────
    intent_response = intent_llm.invoke([
        HumanMessage(content=INTENT_PROMPT.format(
            user_message=last_message.content,
            has_image=has_image,
            has_palette=has_palette,
        ))
    ])

    raw = intent_response.content.lower()
    intents = [
        i.strip().replace(" ", "_")
        for i in raw.split(",")
    ]

    # Normalize + validate
    intents = [i for i in intents if i in VALID_INTENTS]
    if not intents:
        intents = ["general_chat"]

    logger.info("Classified intents: %s", intents)

    # ── Conversational response ─────────────────────────────────────
    system = SystemMessage(content=SYSTEM_PROMPT.format(
        has_image=has_image,
        has_palette=has_palette,
        palette_str=palette_display(state.get("palette")),
        recolor_count=state.get("recolor_count", 0),
    ))

    # Truncate context to avoid overflowing Ollama context window
    context_messages = state["messages"][-MAX_CONTEXT_MESSAGES:]
    if new_human_msg:
        context_messages = context_messages + [new_human_msg]
    response = llm.invoke([system] + context_messages)

    logger.info("Response generated | intents=%s", intents)

    result = {
        "messages": ([new_human_msg] if new_human_msg else []) + [response],
        "user_intents": intents,
        "chat_iterations": iteration,
        "error": None,
    }

    # Pass through image data if it came from a resume
    if new_human_msg and state.get("image_b64"):
        result["image_b64"] = state["image_b64"]
        result["image_filename"] = state.get("image_filename")

    return result
