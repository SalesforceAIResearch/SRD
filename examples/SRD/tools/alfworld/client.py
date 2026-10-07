"""Training-side client for the ``alfworld_step`` tool that forwards tool calls to the ALFWorld sidecar."""

import os

from miles.rollout.generate_hub.multi_turn import current_trajectory_metadata, current_trajectory_session_id
from miles.utils.http_utils import post

ALFWORLD_SIDECAR_URL = os.environ.get("SDPO_REACT_ALFWORLD_SIDECAR_URL") or "http://127.0.0.1:8423"
MAX_RESULT_CHARS = 4000

_TOOL_NAMES = {"alfworld_step"}


def _clip(text: str) -> str:
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "\n...[truncated]..."


async def call_alfworld_tool(args: dict) -> str:
    session_id = current_trajectory_session_id()
    metadata = current_trajectory_metadata()
    game_file = metadata.get("alfworld_game_file")
    split = metadata.get("alfworld_split", "train")
    try:
        payload = await post(
            f"{ALFWORLD_SIDECAR_URL}/session/{session_id}/step",
            {"action": args.get("action", ""), "game_file": game_file, "split": split},
            max_retries=3,
            action="post",
        )
    except Exception as e:
        return f"error:\nalfworld sidecar unreachable: {e}"
    if payload.get("done"):
        metadata["episode_won"] = bool(payload.get("won", False))
    return _clip(payload.get("observation", "(no output)"))


async def execute_tool(name: str, params) -> str:
    """Single-tool ``--generate-execute-tool-function-path`` target for an
    alfworld-ONLY run. Multi-domain runs should point at
    ``examples.SRD.tools.registry.execute_tool`` instead."""
    if name not in _TOOL_NAMES:
        return f"error:\nunknown tool '{name}' (this executor only handles alfworld_step)"
    args = params if isinstance(params, dict) else {}
    return await call_alfworld_tool(args)