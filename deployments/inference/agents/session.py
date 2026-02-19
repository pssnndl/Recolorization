"""In-memory session management for agent conversations."""

import time
import uuid
from typing import Optional

from state import RecolorState

# Session store
_sessions: dict[str, RecolorState] = {}
_timestamps: dict[str, float] = {}

SESSION_TTL_SECONDS = 3600  # 1 hour


def get_or_create_session(
    session_id: Optional[str] = None,
) -> tuple[str, RecolorState]:
    """Get an existing session or create a new one."""
    if session_id and session_id in _sessions:
        _timestamps[session_id] = time.time()
        return session_id, _sessions[session_id]

    sid = session_id or str(uuid.uuid4())
    state: RecolorState = {
        "messages": [],
        "image_b64": None,
        "image_filename": None,
        "image_size": None,
        "palette": None,
        "palette_candidates": [],
        "palette_source": None,
        "next_nodes": ["respond"],
        "result_b64": None,
        "recolor_count": 0,
        "error": None,
        "session_id": sid,
        "user_intents": [],
    }
    _sessions[sid] = state
    _timestamps[sid] = time.time()
    return sid, state


def get_session(session_id: str) -> Optional[RecolorState]:
    """Get a session by ID, or None if not found."""
    if session_id in _sessions:
        _timestamps[session_id] = time.time()
        return _sessions[session_id]
    return None


def cleanup_old_sessions() -> int:
    """Remove sessions older than TTL. Returns number of sessions removed."""
    cutoff = time.time() - SESSION_TTL_SECONDS
    expired = [sid for sid, ts in _timestamps.items() if ts < cutoff]
    for sid in expired:
        _sessions.pop(sid, None)
        _timestamps.pop(sid, None)
    return len(expired)
