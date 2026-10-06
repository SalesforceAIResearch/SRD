"""Executor for the ``cli_exec`` tool."""

import json

from examples.SRD.tools.code.client import run_code

DEFAULT_CLI_TIMEOUT_SECONDS = 10.0


async def run_command(command: str, timeout: float = DEFAULT_CLI_TIMEOUT_SECONDS) -> str:
    if not isinstance(command, str) or not command.strip():
        return "error:\nno command provided"
    # Marshal the command through the code sandbox as a subprocess call. json.dumps
    # safely escapes the command string into a Python literal.
    wrapper = (
        "import subprocess\n"
        f"p = subprocess.run({json.dumps(command)}, shell=True, capture_output=True, text=True, timeout={timeout})\n"
        "out = (p.stdout or '')\n"
        "err = (p.stderr or '')\n"
        "print(out, end='')\n"
        "import sys\n"
        "if err: sys.stderr.write(err)\n"
    )
    return await run_code(wrapper, timeout=timeout)


async def execute_tool(name: str, params) -> str:
    """Single-tool ``--generate-execute-tool-function-path`` target for a
    cli_exec-ONLY run. Multi-tool runs should point at
    ``examples.SRD.tools.registry.execute_tool`` instead."""
    if isinstance(params, str):
        try:
            decoded = json.loads(params)
            params = decoded if isinstance(decoded, dict) else {"command": params}
        except json.JSONDecodeError:
            params = {"command": params}
    elif not isinstance(params, dict):
        params = {}

    if name != "cli_exec":
        return f"error:\nunknown tool '{name}' (this executor only handles cli_exec)"
    return await run_command(params.get("command", ""))
