"""Chat Agent — orchestrator that classifies intent and generates conversational responses."""

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import sys
sys.path.insert(0, "../")
from tools.palette_utils import palette_display

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


def chat_agent(state: dict) -> dict:
    llm = ChatOllama(model="llama3.1:8b", temperature=0.7, num_predict=512)
    intent_llm = ChatOllama(model="llama3.1:8b", temperature=0.0, num_predict=64)

    has_image = state.get("image_b64") is not None
    has_palette = (
        state.get("palette") is not None
        and len(state.get("palette", [])) == 6
    )

    # Get latest user message
    last_message = state["messages"][-1]

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

    # Step 3: Generate conversational response
    system = SystemMessage(content=SYSTEM_PROMPT.format(
        has_image=has_image,
        has_palette=has_palette,
        palette_str=palette_display(state.get("palette")),
        recolor_count=state.get("recolor_count", 0),
    ))

    # Truncate context to avoid overflowing Ollama context window
    context_messages = state["messages"][-MAX_CONTEXT_MESSAGES:]
    response = llm.invoke([system] + context_messages)

    return {
        "messages": [response],
        "user_intents": intents,
        "error": None,
    }
