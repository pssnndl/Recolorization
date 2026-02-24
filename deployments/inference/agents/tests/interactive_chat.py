"""Interactive chat REPL — talk to the graph in real time.

Usage:
    cd deployments/inference/agents
    python -m tests.interactive_chat
    python -m tests.interactive_chat --image path/to/image.png

Commands during chat:
    /image <path>   — attach an image with next message
    /state          — print current graph state
    /quit           — exit
"""

import base64
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
)

from tests.helpers import (
    new_thread, first_message, next_message,
    is_interrupted, get_state, last_ai,
)


def main():
    config = new_thread()
    pending_image = None
    pending_filename = None
    first = True

    # Load image from CLI if provided
    if "--image" in sys.argv:
        idx = sys.argv.index("--image")
        if idx + 1 < len(sys.argv):
            img_path = sys.argv[idx + 1]
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    pending_image = base64.b64encode(f.read()).decode()
                pending_filename = os.path.basename(img_path)
                print(f"[image loaded: {pending_filename}]")
            else:
                print(f"[warning] file not found: {img_path}")

    thread_id = config["configurable"]["thread_id"][:8]
    print(f"Thread: {thread_id}...")
    print("Commands: /image <path>, /state, /quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exiting]")
            break

        if not user_input:
            continue

        # ── Commands ────────────────────────────────────────────
        if user_input == "/quit":
            print("[exiting]")
            break

        if user_input == "/state":
            state = get_state(config)
            has_image = state.get("image_b64") is not None
            palette = state.get("palette")
            has_palette = palette is not None and len(palette or []) == 6
            print(f"  has_image:       {has_image}")
            print(f"  has_palette:     {has_palette}")
            if has_palette:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
                from tools.palette_utils import palette_to_hex
                print(f"  palette:         {palette_to_hex(palette)}")
            print(f"  palette_source:  {state.get('palette_source')}")
            print(f"  candidates:      {len(state.get('palette_candidates', []))}")
            print(f"  recolor_count:   {state.get('recolor_count', 0)}")
            print(f"  chat_iterations: {state.get('chat_iterations', 0)}")
            print(f"  has_result:      {state.get('result_b64') is not None}")
            print(f"  error:           {state.get('error')}")
            print(f"  interrupted:     {is_interrupted(config)}")
            print()
            continue

        if user_input.startswith("/image "):
            img_path = user_input[7:].strip()
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    pending_image = base64.b64encode(f.read()).decode()
                pending_filename = os.path.basename(img_path)
                print(f"  [loaded: {pending_filename} — will attach to next message]\n")
            else:
                print(f"  [error: file not found: {img_path}]\n")
            continue

        # ── Send message ────────────────────────────────────────
        try:
            if first:
                state = first_message(
                    user_input, config,
                    image_b64=pending_image,
                    image_filename=pending_filename,
                )
                first = False
            elif is_interrupted(config):
                state = next_message(
                    user_input, config,
                    image_b64=pending_image,
                    image_filename=pending_filename,
                )
            else:
                # Graph completed — start fresh thread
                print("  [graph completed — new thread]\n")
                config = new_thread()
                first = True
                state = first_message(
                    user_input, config,
                    image_b64=pending_image,
                    image_filename=pending_filename,
                )
                first = False

            pending_image = None
            pending_filename = None

            # ── Show response ───────────────────────────────────
            response = last_ai(state)
            print(f"\nBot: {response or '[no response]'}\n")

            # ── Status bar ──────────────────────────────────────
            has_image = state.get("image_b64") is not None
            palette = state.get("palette")
            has_palette = palette is not None and len(palette or []) == 6
            has_result = state.get("result_b64") is not None
            interrupted = is_interrupted(config)

            parts = []
            parts.append(f"img:{'yes' if has_image else 'no'}")
            if has_palette:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
                from tools.palette_utils import palette_to_hex
                parts.append(f"palette:{palette_to_hex(palette)}")
            else:
                parts.append("palette:no")
            if has_result:
                parts.append(f"recolored:{state.get('recolor_count', 0)}")
            parts.append("waiting" if interrupted else "DONE")
            print(f"  [{' | '.join(parts)}]\n")

            # Save result if recolorization completed
            if has_result:
                result_path = f"result_{thread_id}.png"
                with open(result_path, "wb") as f:
                    f.write(base64.b64decode(state["result_b64"]))
                print(f"  [result saved: {result_path}]\n")

        except Exception as e:
            print(f"\n  [error: {e}]\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
