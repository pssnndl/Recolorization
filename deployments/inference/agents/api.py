"""FastAPI router for the agent chat system — REST + WebSocket endpoints."""

import json
import asyncio
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from .graph import app_graph
from .session import get_or_create_session, get_session

agent_router = APIRouter(prefix="/agent", tags=["agent"])


# --- Helpers ---

async def _run_graph(
    state: dict,
    user_message: str,
    image_b64: Optional[str] = None,
    image_filename: Optional[str] = None,
) -> dict:
    """Run the LangGraph graph with a new user message."""
    state["messages"].append(HumanMessage(content=user_message))

    if image_b64:
        state["image_b64"] = image_b64
        state["image_filename"] = image_filename

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, app_graph.invoke, state)

    # Update session with result
    state.update(result)
    return state


def _extract_response(state: dict) -> str:
    """Get the latest AI message content."""
    ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
    return ai_messages[-1].content if ai_messages else ""


def _state_payload(state: dict) -> dict:
    """Build the state payload to send to the client."""
    return {
        "has_image": state.get("image_b64") is not None,
        "has_palette": (
            state.get("palette") is not None
            and len(state.get("palette", [])) == 6
        ),
        "palette": state.get("palette"),
        "palette_candidates": state.get("palette_candidates"),
        "result_base64": state.get("result_b64"),
        "recolor_count": state.get("recolor_count", 0),
        "error": state.get("error"),
    }


# --- REST endpoints ---

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    image_base64: Optional[str] = None
    image_filename: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    has_image: bool
    has_palette: bool
    palette: Optional[list[list[int]]] = None
    palette_candidates: Optional[list[dict]] = None
    result_base64: Optional[str] = None
    recolor_count: int
    error: Optional[str] = None


@agent_router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """REST endpoint for chat interaction (polling-based fallback)."""
    session_id, state = get_or_create_session(req.session_id)

    state = await _run_graph(
        state,
        user_message=req.message,
        image_b64=req.image_base64,
        image_filename=req.image_filename,
    )

    return ChatResponse(
        session_id=session_id,
        response=_extract_response(state),
        has_image=state.get("image_b64") is not None,
        has_palette=(
            state.get("palette") is not None
            and len(state.get("palette", [])) == 6
        ),
        palette=state.get("palette"),
        palette_candidates=state.get("palette_candidates"),
        result_base64=state.get("result_b64"),
        recolor_count=state.get("recolor_count", 0),
        error=state.get("error"),
    )


@agent_router.post("/chat/{session_id}/select-palette/{index}")
async def select_palette(session_id: str, index: int):
    """Select a specific palette from candidates."""
    state = get_session(session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    candidates = state.get("palette_candidates", [])
    if index < 0 or index >= len(candidates):
        raise HTTPException(
            400,
            f"Invalid palette index. Available: 0-{len(candidates)-1}",
        )

    selected = candidates[index]
    state["palette"] = selected["colors"]
    state["palette_source"] = selected["source"]

    return {
        "palette": selected["colors"],
        "source": selected["source"],
        "description": selected["description"],
    }


# --- WebSocket endpoint ---

@agent_router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat.

    Client sends:
      { "type": "text", "content": "make it warmer" }
      { "type": "image", "content": "<base64>", "filename": "photo.jpg" }
      { "type": "select_palette", "index": 2 }

    Server sends:
      { "type": "status", "content": "Thinking..." }
      { "type": "message", "content": "...", "state": {...} }
      { "type": "result", "content": "...", "state": {...} }
      { "type": "error", "content": "..." }
    """
    await websocket.accept()
    _, state = get_or_create_session(session_id)

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            try:
                if msg["type"] == "text":
                    await websocket.send_json({
                        "type": "status",
                        "content": "Thinking...",
                    })
                    state = await _run_graph(state, user_message=msg["content"])

                elif msg["type"] == "image":
                    await websocket.send_json({
                        "type": "status",
                        "content": "Processing image...",
                    })
                    state = await _run_graph(
                        state,
                        user_message="I've uploaded an image.",
                        image_b64=msg["content"],
                        image_filename=msg.get("filename"),
                    )

                elif msg["type"] == "select_palette":
                    idx = msg["index"]
                    candidates = state.get("palette_candidates", [])
                    if 0 <= idx < len(candidates):
                        state["palette"] = candidates[idx]["colors"]
                        state["palette_source"] = candidates[idx]["source"]

                # Build response
                response_type = (
                    "result" if state.get("result_b64") else "message"
                )
                await websocket.send_json({
                    "type": response_type,
                    "content": _extract_response(state),
                    "state": _state_payload(state),
                })

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "content": str(e),
                })

    except WebSocketDisconnect:
        pass
