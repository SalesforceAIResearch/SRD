"""Prompt text extracted from examples/SRD/tools/registry.py.

Imported back by that module; edit the wording here."""

# --------------------------------------------------------------------------- #
# Shared MINIMAL system prompt (one source of truth for all domains).
#
# Design (per the proposal's "let the model choose + no over-prompting" intent):
# every row exposes ALL tools (all_tool_specs) and the prompt says NOTHING about
# which tool to use, how many hops to take, or any task-specific workflow -- the
# tool schemas are injected separately by the native <tools> block, and the
# CAPABILITY (multi-hop search, code verification, tool selection) must come from
# TRAINING, not from a hand-written recipe in the prompt. The only non-question
# content is the final-answer FORMAT contract, which grading depends on.
# --------------------------------------------------------------------------- #
MINIMAL_SYSTEM_PROMPT = (
    "You solve the user's problem step by step. You have tools available "
    "(declared below); call any that help, as many times as you need (multiple "
    "tool calls in one turn run concurrently). Do NOT answer from memory alone: "
    "before you commit to a final answer, you MUST use a tool to VERIFY it -- "
    "run code to check a computation, or search to confirm a fact. Only after a "
    "tool has confirmed your reasoning, give your final answer inside <answer> "
    "and </answer> tags, e.g. <answer>42</answer>. Put ONLY the final answer "
    "inside the tags."
)
