from typing import TypedDict, Optional
from langgraph.graph import MessagesState


class PaletteCandidate(TypedDict):
    colors: list[list[int]]  # Exactly 6 RGB triplets [[R,G,B], ...]
    source: str  # "colormind_api", "colorthief", "pylette", "llm_suggested", "user_manual"
    description: str  # Human-readable label


class RecolorState(MessagesState):
    """
    Shared state for the recolorization agent graph.
    Extends MessagesState to inherit automatic `messages` list management.
    """
    # Slot status
    image_b64: Optional[str]
    image_filename: Optional[str]
    image_size: Optional[tuple[int, int]]

    # Palette state
    palette: Optional[list[list[int]]]
    palette_candidates: list[PaletteCandidate]
    palette_source: Optional[str]

    # Routing
    next_nodes: list[str]
    next_node: Optional[str]  # used by join_slots / recolor_agent for single-step routing


    # Recolor results
    result_b64: Optional[str]
    recolor_count: int

    # Error tracking
    error: Optional[str]

    # Session metadata
    session_id: str
    user_intents: list[str]

