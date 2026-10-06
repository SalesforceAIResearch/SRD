"""OpenAI function-call spec for the ``alfworld_step`` tool."""

ALFWORLD_STEP_SPEC = {
    "type": "function",
    "function": {
        "name": "alfworld_step",
        "description": (
            "Take one action in the ALFWorld household task environment. The observation "
            "returned after each call includes the current room/object state and the list "
            "of currently admissible actions -- choose your next action from that list "
            "(free text, e.g. 'go to countertop 1', 'take apple 1 from countertop 1', "
            "'heat mug 1 with microwave 1'). The episode ends when the task is completed "
            "or the turn budget runs out."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to take, exactly as it appears in the admissible-actions list.",
                },
            },
            "required": ["action"],
        },
    },
}

alfworld_specs = [ALFWORLD_STEP_SPEC]
