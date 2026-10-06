"""Training-side client for the search/open/find tool set that forwards tool calls to the search sidecar."""

import os

from miles.rollout.generate_hub.multi_turn import current_trajectory_session_id
from miles.utils.http_utils import post

SEARCH_SIDECAR_URL = os.environ.get("SDPO_REACT_SEARCH_SIDECAR_URL", "http://127.0.0.1:8421")
MAX_RESULT_CHARS = 4000

_TOOL_NAMES = {"search", "open", "find"}


def _clip(text: str) -> str:
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "\n...[truncated]..."


async def call_search_tool(name: str, args: dict) -> str:
    session_id = current_trajectory_session_id()
    try:
        payload = await post(
            f"{SEARCH_SIDECAR_URL}/session/{session_id}/call",
            {"tool": name, "args": args},
            max_retries=3,
            action="post",
        )
    except Exception as e:
        return f"error:\nsearch sidecar unreachable: {e}"
    return _clip(payload.get("observation", "(no output)"))


async def execute_tool(name: str, params) -> str:
    """Single-tool ``--generate-execute-tool-function-path`` target for a
    search-ONLY run. Multi-tool runs should point at
    ``examples.SRD.tools.registry.execute_tool`` instead."""
    if name not in _TOOL_NAMES:
        return f"error:\nunknown tool '{name}' (this executor only handles search|open|find)"
    args = params if isinstance(params, dict) else {}
    return await call_search_tool(name, args)
