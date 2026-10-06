"""SDPO group-RM wrapper for multi-turn tool-calling rollouts (SDPO_ReAct).

This module only adds the tool-call bookkeeping SDPO_ReAct needs on top 
(a message-dict trace dump for post-hoc inspection).
"""

import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any

from examples.SRD.reward import (
    _grade_one_alfworld,
    _grade_one_code,
    _grade_one_search,
    _grade_one_webshop,
    _is_correct,
    _llm_judge_correct,
    _sample_domain,
)
from examples.SRD.sdpo import _BLIND_PREDICT_SYSTEM
from examples.SRD.sdpo import _PITFALL_PREDICT_SYSTEM
from examples.SRD.sdpo import EVAL_SKILL_CORRECT_TEMPLATE
from examples.SRD.sdpo import EVAL_SKILL_INSTRUCTION
from examples.SRD.sdpo import EVAL_SKILL_PITFALL_TEMPLATE
from examples.SRD.sdpo import _blind_predict_user_prompt
from examples.SRD.sdpo import _gen_prompt_suffix
from examples.SRD.sdpo import _generate_skill_text
from examples.SRD.sdpo import _pitfall_predict_user_prompt
from examples.SRD.sdpo import _tokenizer
from examples.SRD.sdpo import sdpo_eval_reward as _sdpo_eval_reward
from examples.SRD.sdpo import sdpo_group_reward as _sdpo_group_reward
from miles.rollout.base_types import GenerateFnOutput
from miles.rollout.generate_hub.multi_turn import generate as _multi_turn_generate
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# Observation prefixes the tool executor (tools/code/client.py) emits on any
# failure. Feeds only the tool_error_count diagnostic; never affects training.
_TOOL_ERROR_PREFIXES = ("error:", "[timeout]")


def _extract_tool_trace(sample: Sample) -> list[dict[str, str]]:
    """(tool_call, observation) pairs, read verbatim from the tool_trace that
    multi_turn.generate records on sample.metadata."""
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    native = md.get("tool_trace")
    if not native:
        return []
    return [
        {"tool_call": t.get("tool_call", ""), "observation": t.get("observation", "")}
        for t in native
        if isinstance(t, dict)
    ]


def _count_tool_errors(tool_trace: list[dict[str, str]]) -> int:
    return sum(1 for t in tool_trace if t["observation"].strip().lower().startswith(_TOOL_ERROR_PREFIXES))


def _prompt_to_messages(prompt: str) -> list[dict[str, Any]]:
    """Split the chat-templated prompt string back into {role, content}
    messages. Falls back to a single user message if there are no ChatML markers."""
    if not isinstance(prompt, str) or "<|im_start|>" not in prompt:
        return [{"role": "user", "content": prompt or ""}]
    import re
    msgs: list[dict[str, Any]] = []
    for m in re.finditer(r"<\|im_start\|>(\w+)\s*\n(.*?)(?:<\|im_end\|>|$)", prompt, re.DOTALL):
        role, content = m.group(1), m.group(2).strip()
        # Drop the trailing empty assistant generation prompt (no closing
        # <|im_end|>) -- the live messages carry the real assistant content.
        if role == "assistant" and re.sub(r"</?think>", "", content).strip() == "":
            continue
        msgs.append({"role": role, "content": content})
    return msgs or [{"role": "user", "content": prompt}]


def _reconstruct_messages(sample: Sample) -> list[dict[str, Any]]:
    """OpenAI-standard message-dict trace: the prompt's system+user turns then
    the live conversation multi_turn.generate recorded on metadata["messages"].
    Falls back to the raw response as one assistant turn when no live record
    exists (e.g. truncated early)."""
    prompt = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    head = _prompt_to_messages(prompt)

    # multi_turn.generate stores the running conversation directly (already
    # OpenAI-standard: assistant.tool_calls + tool.tool_call_id).
    live = md.get("messages")
    if live:
        return [*head, *live]

    return [*head, {"role": "assistant", "content": sample.response or ""}]


def _dump_agentic_traces(args: Namespace, group: list[Sample]) -> None:
    """Append one message-dict trace per sample to
    --dump-details/agentic_traces/{rollout_id}.jsonl for post-hoc inspection.
    Called once per group; append mode (never awaits mid-write) so concurrent
    groups for the same rollout_id can't interleave. Non-fatal: a dump bug
    never breaks rollout."""
    dump_dir = getattr(args, "dump_details", None)
    if dump_dir is None:
        return
    try:
        rollout_id = None
        for sample in group:
            if isinstance(sample.metadata, dict) and sample.metadata.get("rollout_id") is not None:
                rollout_id = sample.metadata["rollout_id"]
                break
        records = [_sample_to_agentic_trace_record(sample) for sample in group]
        _append_agentic_trace_records(dump_dir, rollout_id, records)
    except Exception as e:  # dumping must never break rollout
        logger.warning(f"SDPO_ReAct agentic trace dump failed (non-fatal): {e!r}")


def _sample_to_agentic_trace_record(sample: Sample) -> dict[str, Any]:
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    messages = _reconstruct_messages(sample)
    return {
        "messages": messages,
        "label": sample.label,
        "tool_call_count": md.get("tool_call_count"),
        "tool_error_count": md.get("tool_error_count"),
        "sdpo_correct": md.get("sdpo_correct"),
        "status": sample.status.value if sample.status is not None else None,
        # For the agentic dashboard's per-domain / per-task-type breakdowns.
        "domain": md.get("domain"),
        "episode_won": md.get("episode_won"),
        "task_type": md.get("task_type"),
        "alfworld_game_file": md.get("alfworld_game_file"),
        "webshop_task_id": md.get("webshop_task_id"),
        "reward": md.get("reward"),
    }


def _append_agentic_trace_records(dump_dir: str, rollout_id: Any, records: list[dict[str, Any]]) -> None:
    path = Path(dump_dir) / "agentic_traces" / f"{rollout_id if rollout_id is not None else 'unknown'}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _dump_agentic_trace_one(args: Namespace, sample: Sample) -> None:
    """Single-sample counterpart to ``_dump_agentic_traces`` for reward paths
    that never see the full group (the plain-GRPO baseline: --custom-rm-path
    without --group-rm). Appends one record to the same dump file."""
    dump_dir = getattr(args, "dump_details", None)
    if dump_dir is None:
        return
    try:
        md = sample.metadata if isinstance(sample.metadata, dict) else {}
        rollout_id = md.get("rollout_id")
        _append_agentic_trace_records(dump_dir, rollout_id, [_sample_to_agentic_trace_record(sample)])
    except Exception as e:  # dumping must never break rollout
        logger.warning(f"SDPO_ReAct agentic trace dump failed (non-fatal): {e!r}")


async def sdpo_react_group_reward(args: Namespace, group: list[Sample], **kwargs: Any) -> list[float]:
    for sample in group:
        if not isinstance(sample.metadata, dict):
            continue
        # Stamp tool_trace (for sdpo.py's env_feedback skill path) and
        # tool_error_count (diagnostic) onto metadata.
        tool_trace = _extract_tool_trace(sample)
        sample.metadata["tool_trace"] = tool_trace
        sample.metadata["tool_error_count"] = _count_tool_errors(tool_trace)

    rewards = await _sdpo_group_reward(args, group, **kwargs)
    # Dump AFTER _sdpo_group_reward so traces include the sdpo_correct result.
    _dump_agentic_traces(args, group)
    return rewards


async def sdpo_react_eval_reward(args: Namespace, sample: Sample, **kwargs: Any) -> float:
    """Per-sample eval RM (--eval-custom-rm-path). Grading is identical to
    SDPO's (pass@1 never touches the prefix/distillation machinery), so we
    delegate directly -- same pattern as examples/EPO/epo.py::epo_eval_reward."""
    return await _sdpo_eval_reward(args, sample, **kwargs)


async def sdpo_react_plain_grpo_reward(args: Namespace, sample: Sample, **kwargs: Any) -> float:
    """Single-sample reward for the plain-GRPO baseline arm (--custom-rm-path,
    no --group-rm). Domain-routes to the SAME per-domain graders _grade_group
    uses (code -> test cases, search -> EM, webshop/alfworld -> episode_won,
    else math/judge), so the ablation isolates only the SDPO/skill machinery,
    not a grading-rule difference."""
    if not isinstance(sample.metadata, dict):
        return 1.0 if _is_correct(sample, args) else 0.0
    sample.metadata["tool_trace"] = _extract_tool_trace(sample)
    domain = _sample_domain(sample)
    if domain == "code":
        ok = await _grade_one_code(sample, args)
    elif domain == "search":
        ok = await _grade_one_search(sample, args)
    elif domain == "webshop":
        ok = _grade_one_webshop(sample, args)
    elif domain == "alfworld":
        ok = _grade_one_alfworld(sample, args)
    elif sample.metadata.get("amo_use_judge") and getattr(args, "sdpo_judge", False) and (sample.response or "").strip():
        ok = await _llm_judge_correct(args, sample)
    else:
        ok = _is_correct(sample, args)
    # Stamp sdpo_correct so the trace dump reflects the grading result.
    sample.metadata["sdpo_correct"] = 1.0 if ok else 0.0
    _dump_agentic_trace_one(args, sample)
    return 1.0 if ok else 0.0


# --------------------------------------------------------------------------- #
# EVAL-time skill augmentation for agentic (tool-calling) domains. Same skill
# self-predict + prompt-splice as sdpo.py::sdpo_eval_generate, but dispatches to
# multi_turn.generate so the second pass keeps the full tool-calling loop
# (sdpo.py's version hardcodes single-turn generate).
# --------------------------------------------------------------------------- #


async def sdpo_react_eval_generate_with_skill(input: Any) -> Any:
    """--custom-generate-function-path that evaluates the model WITH its own
    self-predicted skill spliced into the prompt, on the normal multi-turn
    tool-calling loop. No-op (falls through to multi_turn.generate) during
    training or when --sdpo-eval-skill-mode is 'off'."""
    args = input.args
    sample = input.sample
    mode = getattr(args, "sdpo_eval_skill_mode", "off")

    if not input.evaluation or mode == "off" or not isinstance(sample.prompt, str):
        return await _multi_turn_generate(input)

    try:
        sections = []
        if mode in ("correct", "all"):
            skill = await _generate_skill_text(
                args, _BLIND_PREDICT_SYSTEM, _blind_predict_user_prompt(sample.prompt), "self"
            )
            if skill.strip():
                sections.append(EVAL_SKILL_CORRECT_TEMPLATE.format(skill=skill.strip()))
        if mode in ("pitfall", "all"):
            skill = await _generate_skill_text(
                args, _PITFALL_PREDICT_SYSTEM, _pitfall_predict_user_prompt(sample.prompt), "self"
            )
            if skill.strip():
                sections.append(EVAL_SKILL_PITFALL_TEMPLATE.format(skill=skill.strip()))

        if sections:
            tok = _tokenizer(args)
            gen_suffix = _gen_prompt_suffix(tok, getattr(args, "apply_chat_template_kwargs", None))
            section = "".join(sections) + EVAL_SKILL_INSTRUCTION
            if gen_suffix and gen_suffix in sample.prompt:
                idx = sample.prompt.rfind(gen_suffix)
                sample.prompt = sample.prompt[:idx] + section + sample.prompt[idx:]
            else:
                sample.prompt = sample.prompt + section
    except Exception as e:
        logger.warning(f"eval skill augmentation failed ({e!r}); evaluating on the unaugmented prompt.")

    output = await _multi_turn_generate(input)
    return output if isinstance(output, GenerateFnOutput) else GenerateFnOutput(samples=output)
