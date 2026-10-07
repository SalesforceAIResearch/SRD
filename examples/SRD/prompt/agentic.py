"""Prompt-text suffixes for the agentic domains (alfworld / webshop builders).

Appended to MINIMAL_SYSTEM_PROMPT there; kept as pure text here."""
ALFWORLD_SYSTEM_PROMPT_SUFFIX = (
    "\n\nThis is a household task: call alfworld_step with a single free-text "
    "action from the admissible-actions list shown in each observation."
)

WEBSHOP_SYSTEM_PROMPT_SUFFIX = (
    "\n\nThis is an online-shopping task: call webshop_step with a single action, "
    "either search[query] or click[button text]."
)
