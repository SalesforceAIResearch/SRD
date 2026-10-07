"""Tool registry: name -> {spec, handler, sets}, selected by $SDPO_REACT_TOOLSET."""

import json
import os

from examples.SRD.tools.alfworld.client import call_alfworld_tool
from examples.SRD.tools.alfworld.spec import ALFWORLD_STEP_SPEC
from examples.SRD.tools.cli.client import run_command
from examples.SRD.tools.cli.spec import CLI_EXEC_SPEC
from examples.SRD.tools.code.client import run_code
from examples.SRD.tools.code.spec import CODE_INTERPRETER_SPEC
from examples.SRD.tools.search.client import call_search_tool
from examples.SRD.tools.search.spec import FIND_SPEC, OPEN_SPEC, SEARCH_SPEC
from examples.SRD.tools.webshop.client import call_webshop_tool
from examples.SRD.tools.webshop.spec import WEBSHOP_STEP_SPEC

# --------------------------------------------------------------------------- #
# Backends. Each is `async handler(params: dict) -> str`. Thin adapters over
# each tool subpackage's own client -- see tools/code/client.py,
# tools/cli/client.py, tools/search/client.py for the actual HTTP calls.
# --------------------------------------------------------------------------- #


async def _handle_code_interpreter(params: dict) -> str:
    return await run_code(params.get("code", ""), stdin=params.get("stdin"))


async def _handle_cli_exec(params: dict) -> str:
    return await run_command(params.get("command", ""))


async def _handle_search(params: dict) -> str:
    return await call_search_tool("search", params)


async def _handle_open(params: dict) -> str:
    return await call_search_tool("open", params)


async def _handle_find(params: dict) -> str:
    return await call_search_tool("find", params)


async def _handle_webshop_step(params: dict) -> str:
    return await call_webshop_tool(params)


async def _handle_alfworld_step(params: dict) -> str:
    return await call_alfworld_tool(params)


# --------------------------------------------------------------------------- #
# Registry: name -> (spec, handler, set-membership). Add a tool HERE only --
# the spec/handler themselves live in that tool's own subpackage.
# --------------------------------------------------------------------------- #
_REGISTRY = {
    "code_interpreter": {"spec": CODE_INTERPRETER_SPEC, "handler": _handle_code_interpreter, "sets": {"math", "code", "all"}},
    "cli_exec": {"spec": CLI_EXEC_SPEC, "handler": _handle_cli_exec, "sets": {"code", "cli", "all"}},
    "search": {"spec": SEARCH_SPEC, "handler": _handle_search, "sets": {"search", "deepsearch", "all"}},
    "open": {"spec": OPEN_SPEC, "handler": _handle_open, "sets": {"search", "deepsearch", "all"}},
    "find": {"spec": FIND_SPEC, "handler": _handle_find, "sets": {"search", "deepsearch", "all"}},
    "webshop_step": {"spec": WEBSHOP_STEP_SPEC, "handler": _handle_webshop_step, "sets": {"webshop", "agentic", "all"}},
    "alfworld_step": {"spec": ALFWORLD_STEP_SPEC, "handler": _handle_alfworld_step, "sets": {"alfworld", "agentic", "all"}},
}


def _active_set_name() -> str:
    return os.environ.get("SDPO_REACT_TOOLSET", "math").strip().lower()


def active_tool_specs() -> list[dict]:
    """The specs for the tool set named by $SDPO_REACT_TOOLSET (default "math").
    Pointed at by --generate-tool-specs-path (called with no args by
    load_function, so this is a zero-arg callable returning the list)."""
    s = _active_set_name()
    return [t["spec"] for t in _REGISTRY.values() if s in t["sets"]]


# Back-compat alias: the base modules import `tool_specs` as a plain list.
# Evaluate the default ("math") set at import time so existing imports keep a
# list, while new launchers can call active_tool_specs() for env-driven sets.
tool_specs = [t["spec"] for t in _REGISTRY.values() if "math" in t["sets"]]

# Plain module-level lists for --generate-tool-specs-path (which load_function
# dereferences to the object directly -- a LIST, not a function, so the rollout
# parser gets the specs without calling anything). all_tool_specs = every tool
# (code_interpreter + cli_exec + search/open/find + webshop_step/alfworld_step)
# for a multi-domain run where the rollout must parse tool calls from ALL
# domains' rows.
all_tool_specs = [t["spec"] for t in _REGISTRY.values() if "all" in t["sets"]]

# agentic_tool_specs = just webshop_step + alfworld_step, for the agentic run
# script's "agentic" (webshop+alfworld combined) domain -- deliberately NOT
# all_tool_specs, so a mixed webshop/alfworld rollout can't accidentally
# parse a stray code_interpreter/search call no row in that domain ever
# declares (harmless if it happened, but confusing/wasteful).
agentic_tool_specs = [t["spec"] for t in _REGISTRY.values() if "agentic" in t["sets"]]


from examples.SRD.prompt.system import MINIMAL_SYSTEM_PROMPT


async def execute_tool(name: str, params) -> str:
    """The --generate-execute-tool-function-path target. Normalizes `params`
    (an untrained policy can emit non-dict/double-encoded arguments) then
    dispatches to the registered backend. Unknown/inactive tools return an
    error observation rather than raising, so a bad call never crashes the
    rollout task."""
    if isinstance(params, str):
        try:
            decoded = json.loads(params)
            params = decoded if isinstance(decoded, dict) else {"code": params}
        except json.JSONDecodeError:
            params = {"code": params}
    elif not isinstance(params, dict):
        params = {}

    entry = _REGISTRY.get(name)
    if entry is None:
        return f"error:\nunknown tool '{name}'"
    return await entry["handler"](params)
