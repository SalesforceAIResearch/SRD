"""HTTP client for the ``code_interpreter`` tool that talks to the sandbox sidecar."""

import json
import os

from miles.utils.http_utils import post

SANDBOX_URL = os.environ.get("SDPO_REACT_SANDBOX_URL", "http://127.0.0.1:8420")
DEFAULT_CODE_TIMEOUT_SECONDS = 10.0
MAX_RESULT_CHARS = 4000


def _clip(text: str) -> str:
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "\n...[truncated]..."


def _format_result(payload: dict) -> str:
    if payload.get("timed_out"):
        return f"[timeout] {payload.get('error', 'execution timed out')}"
    stdout = (payload.get("stdout") or "").strip()
    error = payload.get("error")
    if error:
        return f"stdout:\n{stdout}\nerror:\n{error}" if stdout else f"error:\n{error}"
    return stdout if stdout else "(no output)"


def _looks_fenced(code: str) -> bool:
    """The model wrapped the code in a markdown fence (```py / ```python / ```)."""
    return isinstance(code, str) and code.lstrip().startswith("```")


async def run_code(code: str, stdin: str | None = None, timeout: float = DEFAULT_CODE_TIMEOUT_SECONDS) -> str:
    """POST raw Python source to the sandbox's ``/execute`` endpoint and return
    the formatted observation string. Shared by both code_interpreter
    (tools/code) and cli_exec (tools/cli, which wraps a shell command into
    Python source first)."""
    if not isinstance(code, str) or not code.strip():
        return "error:\nno code provided"
    try:
        # Run the code EXACTLY as the model emitted it -- we do NOT strip a
        # markdown fence: the model must LEARN not to emit one, so a fenced call
        # must genuinely fail (SyntaxError). We only make that failure legible.
        request = {"code": code, "timeout": timeout}
        if stdin is not None:
            request["stdin"] = stdin
        payload = await post(f"{SANDBOX_URL}/execute", request, max_retries=3, action="post")
    except Exception as e:
        # A dead/unreachable sandbox should surface as a tool-observation error
        # (so the rollout keeps going and dynamic filters like check_no_aborted
        # can catch it at the group level) rather than crashing the rollout task.
        return f"error:\ncode_interpreter sandbox unreachable: {e}"

    result = _clip(_format_result(payload))
    # If the model fenced its code AND it errored, append a concrete corrective
    # hint so it can learn from its OWN mistake (no silent fix on our side): the
    # ```py fence is part of the code string and is invalid Python.
    if _looks_fenced(code) and "error" in result.lower():
        result += (
            "\n\n[hint] Your code was wrapped in a markdown code fence (```). The "
            "fence is NOT valid Python and was executed literally. Pass the raw "
            "source only -- remove the ```python / ``` markers -- and retry."
        )
    return result


async def execute_tool(name: str, params) -> str:
    """Single-tool ``--generate-execute-tool-function-path`` target for a
    code_interpreter-ONLY run. Multi-tool runs should point at
    ``examples.SRD.tools.registry.execute_tool`` instead, which
    dispatches across every registered tool including this one."""
    # An untrained/weak model can emit tool-call arguments that are valid JSON
    # but not a dict -- e.g. a JSON-encoded string (double-encoded arguments)
    # or a bare code string instead of {"code": ...}. tool_call_utils.py's
    # execute_tool_calls does `json.loads(call.arguments)` with no shape
    # check, so `params` here is whatever that produced. Normalize defensively
    # rather than letting `.get()` crash the whole rollout task (observed
    # failure: "AttributeError: 'str' object has no attribute 'get'").
    if isinstance(params, str):
        try:
            decoded = json.loads(params)
            params = decoded if isinstance(decoded, dict) else {"code": params}
        except json.JSONDecodeError:
            params = {"code": params}
    elif not isinstance(params, dict):
        params = {}

    if name != "code_interpreter":
        return f"error:\nunknown tool '{name}' (this executor only handles code_interpreter)"
    code = params.get("code", "")
    if not isinstance(code, str) or not code.strip():
        return "error:\nno code provided"
    return await run_code(code, stdin=params.get("stdin"))
