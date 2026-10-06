"""OpenAI function-call spec for the ``cli_exec`` tool."""

CLI_EXEC_SPEC = {
    "type": "function",
    "function": {
        "name": "cli_exec",
        "description": (
            "Run a shell command in the sandbox and return its combined stdout/stderr. "
            "Use for file/OS-level tasks (ls, cat, grep, running a script). Runs in the "
            "same isolated container as code_interpreter; no network access."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run, e.g. \"ls -la && cat foo.py\"."}
            },
            "required": ["command"],
        },
    },
}
