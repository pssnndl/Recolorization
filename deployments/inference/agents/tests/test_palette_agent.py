"""Test the palette sub-graph directly — bypasses chat_agent and routing.

Usage:
    cd deployments/inference/agents
    python -m tests.test_palette_agent
"""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import HumanMessage, AIMessage
from nodes.palette_agent import palette_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
)

TEST_CASES = [
    "Use rainbow colors",
    "warm sunset palette",
    "ocean blues and teals",
    "neon cyberpunk colors",
    "soft pastel pinks and lavenders",
    "dark moody forest greens",
    "give me random colors",
    "#FF6B6B #4ECDC4 #45B7D1 #96CEB4 #FFEAA7 #DDA0DD",
]


def test_palette_agent_direct():
    """Run all palette test cases against the palette sub-graph."""
    passed = 0
    failed = 0

    for msg in TEST_CASES:
        print(f"\n  Testing: \"{msg}\"")
        state = {
            "messages": [HumanMessage(content=msg)],
            "palette": None,
            "palette_candidates": [],
            "palette_source": None,
            "image_b64": None,
            "error": None,
        }

        result = palette_agent(state)
        palette = result.get("palette")
        source = result.get("palette_source")
        error = result.get("error")
        candidates = result.get("palette_candidates", [])
        response = ""
        for m in result.get("messages", []):
            if isinstance(m, AIMessage):
                response = m.content
                break

        if palette and len(palette) == 6:
            print(f"    [PASS] palette={palette}, source={source}, candidates={len(candidates)}")
            passed += 1
        else:
            print(f"    [FAIL] error={error}, palette={palette}")
            if response:
                print(f"    response: {response[:120]}")
            failed += 1

    print(f"\n  Results: {passed}/{passed + failed} passed")
    return passed, failed


if __name__ == "__main__":
    print("=== Palette Agent Direct Test ===\n")
    passed, failed = test_palette_agent_direct()
    sys.exit(1 if failed else 0)
