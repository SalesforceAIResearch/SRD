"""OpenAI function-call spec for the ``code_interpreter`` tool."""

CODE_INTERPRETER_SPEC = {
    "type": "function",
    "function": {
        "name": "code_interpreter",
        "description": (
            "Execute Python code in an isolated sandbox and return its stdout. "
            "Use this for calculations, symbolic math (sympy), or verifying a "
            "numeric answer before giving your final answer. The sandbox has "
            "sympy, numpy, and scipy preinstalled; it has no network access."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Raw Python source code to execute -- NOT wrapped in a markdown "
                        "code fence. Do NOT include ```python or ``` markers; pass only "
                        "the executable code itself. Print anything you want to see."
                    ),
                },
                "stdin": {
                    "type": "string",
                    "description": (
                        "Optional text piped to the program's real stdin (so input() / "
                        "sys.stdin.read() work as normal). Use this to test a program that "
                        "reads from stdin -- do NOT hardcode `sys.stdin = io.StringIO(...)` "
                        "in the code itself: for a stdin-reading program that will be "
                        "GRADED, the LAST code_interpreter call's `code` is run again "
                        "verbatim against the REAL hidden test input, and a hardcoded "
                        "stdin override in the source would silently replace that real "
                        "input and make the submission fail regardless of its logic."
                    ),
                },
            },
            "required": ["code"],
        },
    },
}
