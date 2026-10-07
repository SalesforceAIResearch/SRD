"""Training-side client for the ``webshop_step`` tool that forwards tool calls to the WebShop sidecar."""

import os

from miles.rollout.generate_hub.multi_turn import current_trajectory_metadata, current_trajectory_session_id
from miles.utils.http_utils import post

WEBSHOP_SIDECAR_URL = os.environ.get("SDPO_REACT_WEBSHOP_SIDECAR_URL") or "http://127.0.0.1:8422"
MAX_RESULT_CHARS = 4000

_TOOL_NAMES = {"webshop_step"}


def _clip(text: str) -> str:
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "\n...[truncated]..."


async def call_webshop_tool(args: dict) -> str:
    session_id = current_trajectory_session_id()
    metadata = current_trajectory_metadata()
    task_id = metadata.get("webshop_task_id")
    # Must be a real int, not a numeric string: the sidecar's env.reset()
    # only pins a specific goal `if isinstance(session, int)`; a string
    # silently falls through to a random goal instead (confirmed live --
    # see docker/server.py's StepRequest.task_id comment for the full story).
    if task_id is not None:
        task_id = int(task_id)
    try:
        payload = await post(
            f"{WEBSHOP_SIDECAR_URL}/session/{session_id}/step",
            {"action": args.get("action", ""), "task_id": task_id},
            max_retries=3,
            action="post",
        )
    except Exception as e:
        return f"error:\nwebshop sidecar unreachable: {e}"
    if payload.get("done"):
        metadata["episode_won"] = bool(payload.get("won", False))
        metadata["webshop_task_score"] = payload.get("task_score", 0.0)
    return _clip(payload.get("observation", "(no output)"))


async def execute_tool(name: str, params) -> str:
    """Single-tool ``--generate-execute-tool-function-path`` target for a
    webshop-ONLY run. Multi-domain runs should point at
    ``examples.SRD.tools.registry.execute_tool`` instead."""
    if name not in _TOOL_NAMES:
        return f"error:\nunknown tool '{name}' (this executor only handles webshop_step)"
    args = params if isinstance(params, dict) else {}
    return await call_webshop_tool(args)
