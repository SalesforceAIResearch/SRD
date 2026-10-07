"""SDPO (prefix-conditioned self-distillation) reward function for Miles.

Rollout is GRPO. After a group is generated, each trace is graded, given a random
correct peer's solution as a teacher "prefix", and scored by the teacher over
``prompt + prefix + response``; a per-token divergence between teacher (with
prefix) and student (without prefix) is written to ``sample.opd_reverse_kl`` and
subtracted from the GRPO advantage by the training side (opd.py).

Wiring (used as a *group* reward model)::

    --group-rm
    --custom-rm-path examples.SRD.sdpo.sdpo_group_reward
    --rm-url http://<TEACHER_IP>:<TEACHER_PORT>/generate
    --use-opd --opd-type sglang --opd-log-prob-top-k 128
    --opd-kl-coef 1.0
    --sdpo-divergence jsd            # reverse_kl | forward_kl | jsd
    --sdpo-logprob-mode topk         # topk | sampled

Only supports ``context_parallel_size == 1``. Correctness grading lives in
``reward.py`` next to this file and is imported below.
"""

import asyncio
import json
import logging
import math
import os
import random
import re
import time
from argparse import Namespace
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from miles.utils.http_utils import post  # miles' shared HTTP client: retries + shared pool
from miles.utils.types import Sample

# Correctness / grading subsystem (reward.py). Re-exported so external call sites
# (e.g. examples/EPO/epo.py) that import _grade_group from here keep working.
from examples.SRD.reward import (
    _extract_answer,
    _grade_group,
    _grade_one_alfworld,
    _grade_one_code,
    _grade_one_search,
    _grade_one_webshop,
    _is_correct,
    _judge_semaphore,
    _llm_judge_correct,
    _sample_domain,
)

logger = logging.getLogger(__name__)

# Per-phase wall-clock accumulators for SDPO scoring; sum across concurrent
# traces (compare ratios, not absolutes). Logged per group-reward batch.
_sdpo_timing = {"tokenize": 0.0, "student_maps": 0.0, "teacher_http": 0.0, "teacher_maps": 0.0, "divergence": 0.0}
_sdpo_calls = 0

from examples.SRD.prompt.skill import (
    SOLUTION_TEMPLATE,
    PREFIX_INSTRUCTION,
    PITFALLS_TEMPLATE,
    _SKILL_SYSTEM_PROMPT,
    _SKILL_SYSTEM_PROMPT_INCORRECT,
    _PITFALL_SUMMARY_SYSTEM,
    _PITFALL_PREDICT_SYSTEM,
    FAILURES_TEMPLATE,
    _BLIND_PREDICT_SYSTEM,
    CORRECT_INFO_TEMPLATE,
    CORRECT_SKILLS_TEMPLATE,
    EVAL_SKILL_CORRECT_TEMPLATE,
    EVAL_SKILL_PITFALL_TEMPLATE,
    EVAL_SKILL_INSTRUCTION,
)

# Skill-KD 'self-success' teacher privileged hint: the student skill-gen prompt
# already contains the worked solution, so the only new signal the teacher has is
# CONFIRMATION that it is correct -- hint with just that, not a restated solution.
SKILL_SELF_SUCCESS_HINT = (
    "\n\n(This worked solution has been verified CORRECT. Distill the skill with "
    "full confidence.)\n\n"
)


def _strip_thinking_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from a response (thinking models like Qwen3)."""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return stripped.strip()


# --- multi-turn native-trace reframing for the teacher prefix --------------- #
# Replace ChatML turn boundaries (<|im_end|><|im_start|>role) in a native
# tool-calling peer trace with short NLP markers (assistant -> "Round N reasoning
# and tool call:", tool/user -> "Observation:"), keeping inner tags
# (<think>/<tool_call>/<tool_response>) intact. Splicing raw boundaries into the
# teacher's USER turn would teach it to emit control-token garbage. No-op for
# single-turn traces. Gated by --sdpo-reframe-multiturn-prefix.
_IM_SPLIT_RE = re.compile(r"<\|im_end\|>\s*<\|im_start\|>\s*(assistant|user|system)\b[ \t]*\n?")
_IM_ANY_RE = re.compile(r"<\|im_(?:start|end)\|>[ \t]*(?:assistant|user|system)?[ \t]*\n?")


def _reframe_multiturn_trace(text: str) -> str:
    """Replace ChatML turn boundaries with per-round NLP markers, keeping
    <think>/<tool_call>/<tool_response> content verbatim. No-op for single-turn."""
    if not text or "<|im_" not in text:
        return text

    # Split on each "<|im_end|><|im_start|>role" boundary, remembering the role
    # that opens each subsequent segment.
    segments: list[tuple[str, str]] = []  # (opening_role, segment_text)
    last_end = 0
    role_for_next = "assistant"  # trace begins inside the assistant's turn
    for m in _IM_SPLIT_RE.finditer(text):
        segments.append((role_for_next, text[last_end : m.start()]))
        role_for_next = m.group(1)
        last_end = m.end()
    segments.append((role_for_next, text[last_end:]))

    def _clean(seg: str) -> str:
        # Strip only stray <|im_*|> tokens; keep <think>/<tool_call>/<tool_response>.
        return _IM_ANY_RE.sub("", seg).strip()

    parts: list[str] = []
    round_no = 0
    for role, seg in segments:
        seg = _clean(seg)
        if not seg:
            continue
        if role == "assistant":
            round_no += 1
            parts.append(f"Round {round_no} reasoning and tool call:\n{seg}")
        else:  # user/tool turn = the observation carrying <tool_response>
            parts.append(f"Observation:\n{seg}")
    return "\n\n".join(parts) if parts else _clean(text)


def _tool_call_args_str(arguments) -> str:
    """Normalize a tool call's arguments to a JSON string."""
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, ensure_ascii=False)
    except Exception:
        return str(arguments)


def _render_tool_call(name: str, arguments, grammar: str = "qwen25") -> str:
    """Render ONE tool call in the target model's native grammar (must byte-match
    the rollout): qwen25 = JSON in <tool_call> tags; qwen3_coder = XML
    <function=NAME><parameter=P> tags."""
    if grammar == "qwen3_coder":
        # XML function/parameter form. arguments -> dict of parameters.
        args_str = _tool_call_args_str(arguments)
        try:
            params = json.loads(args_str) if isinstance(args_str, str) else (arguments or {})
        except Exception:
            params = {}
        if not isinstance(params, dict):
            params = {}
        lines = [f"<function={name}>"]
        for k, v in params.items():
            vs = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            lines.append(f"<parameter={k}>\n{vs}\n</parameter>")
        lines.append("</function>")
        return "<tool_call>\n" + "\n".join(lines) + "\n</tool_call>"
    # default qwen25: JSON-in-tags
    call_json = f'{{"name": "{name}", "arguments": {_tool_call_args_str(arguments)}}}'
    return f"<tool_call>\n{call_json}\n</tool_call>"


def _render_tool_response(obs: str, grammar: str = "qwen25") -> str:
    """Observation grammar: wrap the tool result in <tool_response>...</tool_response>."""
    return f"<tool_response>\n{obs}\n</tool_response>"


def _reframe_messages_to_prose(messages: list, remove_thinking: bool = False, grammar: str = "qwen25") -> str:
    """NLP-ize a peer trace from its structured message dict (metadata["messages"])
    into per-round "Round N reasoning and tool call:" / "Observation:" text. Only
    the ChatML turn boundaries are NLP-ized; the inner <tool_call>/<tool_response>
    blocks are rendered in the target model's native grammar (must byte-match the
    rollout, else tool-call rate collapses). Dict-native counterpart of
    _reframe_multiturn_trace."""
    parts: list[str] = []
    round_no = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content") or ""
        if role == "assistant":
            round_no += 1
            # multi_turn.generate splits raw output into reasoning_content +
            # content; re-wrap the reasoning in <think> so remove_thinking behaves
            # as it did when reasoning lived inline in `content`.
            reasoning = (m.get("reasoning_content") or "").strip()
            if reasoning:
                content = f"<think>\n{reasoning}\n</think>\n\n{content}"
            if remove_thinking:
                content = _strip_thinking_blocks(content)
            seg = content.strip()
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "tool")
                seg = (seg + "\n" + _render_tool_call(name, fn.get("arguments", ""), grammar)).strip()
            if seg:
                parts.append(f"Round {round_no} reasoning and tool call:\n{seg}")
        elif role == "tool":
            obs = content.strip()
            if obs:
                parts.append(f"Observation:\n{_render_tool_response(obs, grammar)}")
        # skip system/user turns: shared problem context, already in student prompt.
    return "\n\n".join(parts)


def _render_prefix(peer_response: str, remove_thinking: bool = False, pitfalls: str = "") -> str:
    content = _strip_thinking_blocks(peer_response) if remove_thinking else peer_response
    # Skip the solution section when there is no base solution (e.g. a pitfalls-only
    # prefix); an empty "Correct solution:" heading would mislead the teacher.
    section = SOLUTION_TEMPLATE.format(successful_previous_attempt=content) if content and content.strip() else ""
    if pitfalls and pitfalls.strip():
        section += PITFALLS_TEMPLATE.format(pitfalls=pitfalls.strip())
    return section + PREFIX_INSTRUCTION


def _gen_prompt_suffix(tok, chat_template_kwargs: dict | None = None) -> str:
    """The exact string the chat template appends after the user content when
    add_generation_prompt=True (e.g. '<|im_end|>\\n<|im_start|>assistant\\n').
    Derived from the tokenizer so it is template-agnostic; used to splice the
    correct-peer solution into the USER turn, before the assistant marker.

    chat_template_kwargs must match the run's, because the suffix differs by
    thinking mode (the student prompt was rendered with the run's kwargs, so the
    suffix must be derived the same way or the splice won't find it).
    """
    # Sentinel with NO surrounding spaces: some templates strip user-content
    # whitespace, so a space-padded sentinel wouldn't be found verbatim.
    sentinel = "SDPO_SENTINEL"
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": sentinel}], tokenize=False, add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )
    return rendered.split(sentinel, 1)[1] if sentinel in rendered else ""


# EOS markers a rollout response may end with; must be stripped before embedding
# the response in the teacher's USER turn, else a stray marker closes it early.
_RESPONSE_EOS_MARKERS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>")


def _strip_response_eos(text: str) -> str:
    """Remove trailing chat/EOS markers (and whitespace) from a rollout response."""
    out = (text or "").rstrip()
    changed = True
    while changed:
        changed = False
        for marker in _RESPONSE_EOS_MARKERS:
            if out.endswith(marker):
                out = out[: -len(marker)].rstrip()
                changed = True
    return out


def _choose_peer(args: Namespace, group: list[Sample], peers: list[int]) -> int:
    """Pick one correct peer (already self-excluded) as the teacher prefix. Uniform
    random by default; with --sdpo-prefer-tool-use-peer, prefer a peer that called a
    tool (tool_call_count > 0), falling back to uniform random over all peers."""
    if not getattr(args, "sdpo_prefer_tool_use_peer", False):
        return random.choice(peers)
    tool_using_peers = [
        j
        for j in peers
        if isinstance(group[j].metadata, dict) and (group[j].metadata.get("tool_call_count") or 0) > 0
    ]
    return random.choice(tool_using_peers) if tool_using_peers else random.choice(peers)


def _build_teacher_prompt_str(student_prompt: str, gen_suffix: str, peer_response: str, remove_thinking: bool = False, pitfalls: str = "", reframe_multiturn: bool = False, peer_messages: list | None = None, grammar: str = "qwen25", max_prefix_chars: int = 0) -> str:
    """Insert the correct-peer solution + instruction into the USER turn of the
    student's chat-templated prompt, before the assistant marker; return the full
    teacher prompt string. The peer response is EOS-stripped first (a stray marker
    would close the user turn early).

    reframe_multiturn (--sdpo-reframe-multiturn-prefix): for native multi-turn
    peer traces, re-template mid-trace control tokens into per-round prose. No-op
    for single-turn.

    max_prefix_chars (--sdpo-max-prefix-chars, 0=off): cap the prose to its LAST N
    chars (keeps the final answer; guards against oversized traces OOMing the
    teacher forward)."""
    # Prefer the dict-native prose when structured messages are available (can't
    # mis-split on a stray marker); else fall back to the raw-text reframe.
    if peer_messages:
        cleaned = _reframe_messages_to_prose(peer_messages, remove_thinking=remove_thinking, grammar=grammar)
    else:
        cleaned = _strip_response_eos(peer_response)
        if reframe_multiturn:
            cleaned = _reframe_multiturn_trace(cleaned)
    if max_prefix_chars and len(cleaned) > max_prefix_chars:
        cleaned = cleaned[-max_prefix_chars:]
    # remove_thinking already applied for the dict path; re-applying is a no-op.
    solution_section = _render_prefix(cleaned, remove_thinking=remove_thinking and not peer_messages, pitfalls=pitfalls)
    if gen_suffix and gen_suffix in student_prompt:
        idx = student_prompt.rfind(gen_suffix)
        return student_prompt[:idx] + solution_section + student_prompt[idx:]
    # Fallback (unknown template): append at the end.
    return student_prompt + solution_section


def _build_skill_self_success_teacher_prompt_str(student_prompt: str, gen_suffix: str) -> str:
    """Skill-KD 'self-success' teacher: the skill-gen prompt plus a privileged
    confirm-correct hint, inserted before the assistant marker (see
    SKILL_SELF_SUCCESS_HINT for why this is not the response-SDPO template)."""
    if gen_suffix and gen_suffix in student_prompt:
        idx = student_prompt.rfind(gen_suffix)
        return student_prompt[:idx] + SKILL_SELF_SUCCESS_HINT + student_prompt[idx:]
    return student_prompt + SKILL_SELF_SUCCESS_HINT


def _build_failure_teacher_prompt_str(student_prompt: str, gen_suffix: str, failure_info: str) -> str:
    """Splice the group's per-trace failure skills into the USER turn of a
    problem-only pitfall-prediction prompt as privileged info (pitfall-condense
    skill-KD teacher). Empty failure_info -> teacher == student (KD signal 0)."""
    if not (failure_info and failure_info.strip()):
        return student_prompt
    section = FAILURES_TEMPLATE.format(successful_previous_attempt=failure_info.strip())
    if gen_suffix and gen_suffix in student_prompt:
        idx = student_prompt.rfind(gen_suffix)
        return student_prompt[:idx] + section + student_prompt[idx:]
    return student_prompt + section


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #


def _divergence_mode(args: Namespace) -> str:
    mode = getattr(args, "sdpo_divergence", "jsd")
    if mode not in ("reverse_kl", "forward_kl", "jsd", "jeffrey", "jeffrey_jsd"):
        raise ValueError(
            f"Unknown --sdpo-divergence {mode!r}; use "
            "reverse_kl | forward_kl | jsd | jeffrey | jeffrey_jsd."
        )
    return mode


def _logprob_mode(args: Namespace) -> str:
    mode = getattr(args, "sdpo_logprob_mode", "topk")
    if mode not in ("topk", "sampled"):
        raise ValueError(f"Unknown --sdpo-logprob-mode {mode!r}; use topk | sampled.")
    return mode


def _prompt_len(sample: Sample) -> int:
    return len(sample.tokens) - sample.response_length


def _response_tokens(sample: Sample) -> list[int]:
    return sample.tokens[_prompt_len(sample) :]


# --------------------------------------------------------------------------- #
# Trace condensation / SkillOpt  (distill the correct peer trace into a SKILL)
# --------------------------------------------------------------------------- #
# With --sdpo-trace-condense, the correct-peer solution is distilled into a short
# transferable SKILL by an LLM, and that skill becomes the teacher prefix.





def _pitfall_summary_user_prompt(problem: str, pitfall_sets: list[str]) -> str:
    blocks = "\n\n".join(f"FAILED ATTEMPT {k + 1} PITFALLS:\n{p}" for k, p in enumerate(pitfall_sets))
    return (
        f"PROBLEM:\n{_clean_problem_for_skill(problem)}\n\n"
        f"{blocks}\n\n"
        "Synthesize the common recurring pitfalls into 1-3 [Error]/[Rule]/[Example] "
        "blocks (merge duplicates, drop one-off noise; no solution, no answer)."
    )





def _pitfall_predict_user_prompt(problem: str) -> str:
    return (
        f"PROBLEM:\n{_clean_problem_for_skill(problem)}\n\n"
        "Predict the pitfalls to avoid as 1-3 [Error]/[Rule]/[Example] tiny-skill "
        "blocks (no solution, no answer)."
    )


# --------------------------------------------------------------------------- #
# "blind-correct" skill-KD (--sdpo-skill-kd-mode blind-correct / both-blind):
# symmetric counterpart to pitfall-condense for CORRECT traces. Regenerates the
# student from the PROBLEM ONLY so the teacher's privileged info (this trace's
# actual correct solution) is a genuine, large information gap -- unlike
# self-success, whose student already contains the worked solution.
# --------------------------------------------------------------------------- #





def _blind_predict_user_prompt(problem: str) -> str:
    return (
        f"PROBLEM:\n{_clean_problem_for_skill(problem)}\n\n"
        "Predict the general knowledge/rules needed as 1-3 [Knowledge/Rule]/"
        "[Details/Examples] tiny-skill blocks (no solution, no answer)."
    )


def _build_blind_correct_teacher_prompt_str(
    student_prompt: str,
    gen_suffix: str,
    correct_info: str,
    template: str = CORRECT_INFO_TEMPLATE,
) -> str:
    """Splice the privileged correct-side info into the USER turn of a problem-only
    knowledge-prediction prompt (blind-correct skill-KD teacher). `template` selects
    the label (CORRECT_INFO_TEMPLATE for the trace's own solution, or
    CORRECT_SKILLS_TEMPLATE for the group's hindsight skills). Empty correct_info ->
    teacher == student (KD signal 0)."""
    if not (correct_info and correct_info.strip()):
        return student_prompt
    section = template.format(successful_previous_attempt=correct_info.strip())
    if gen_suffix and gen_suffix in student_prompt:
        idx = student_prompt.rfind(gen_suffix)
        return student_prompt[:idx] + section + student_prompt[idx:]
    return student_prompt + section


_CONDENSE_SEM: "asyncio.Semaphore | None" = None
_CONDENSE_SEM_LIMIT: int | None = None


def _condense_semaphore(args: Namespace) -> asyncio.Semaphore:
    global _CONDENSE_SEM, _CONDENSE_SEM_LIMIT
    limit = int(getattr(args, "sdpo_condense_max_concurrency", 32))
    if _CONDENSE_SEM is None or _CONDENSE_SEM_LIMIT != limit:
        _CONDENSE_SEM = asyncio.Semaphore(limit)
        _CONDENSE_SEM_LIMIT = limit
    return _CONDENSE_SEM


# Answer-format scaffolding datasets inject into the problem text; if left in the
# skill-gen PROBLEM the model appends "Answer: \boxed{...}", leaking the answer into
# the skill-KD target. Strip these instruction lines.
_ANSWER_FORMAT_PATTERNS = [
    re.compile(r"Solve the following math problem step by step\.\s*", re.IGNORECASE),
    re.compile(r"The last line of your response should be of the form[^\n]*\n?", re.IGNORECASE),
    re.compile(r"Remember to put your answer[^\n]*\n?", re.IGNORECASE),
    re.compile(r"[Pp]ut your (?:final )?answer (?:in|inside)[^\n]*\\boxed\{\}[^\n]*\n?"),
]


# A rollout sample.prompt is the FULL chat-templated string, not the raw question.
# Embedding it as the skill-gen "PROBLEM" would nest a chat template (a second
# system turn that confuses the skill/pitfall prompt); strip it down to the last
# user turn's content so the generator sees only the actual problem.
_CHAT_USER_BLOCK = re.compile(
    r"<\|im_start\|>\s*user\s*\n(.*?)<\|im_end\|>", re.DOTALL
)


def _strip_chat_template(text: str) -> str:
    """Recover the raw user-turn text from a chat-templated prompt (last user block's
    content), or return the input unchanged if it is already raw."""
    if not text or "<|im_start|>" not in text:
        return text
    matches = _CHAT_USER_BLOCK.findall(text)
    if matches:
        return matches[-1].strip()
    # Markers present but no closed user block (unusual template): return as-is.
    return text


def _clean_problem_for_skill(problem: str) -> str:
    """Remove chat-template scaffolding and answer-format instructions so the skill
    generator sees only the problem (no nested system turn, no answer-format leak)."""
    out = _strip_chat_template(problem or "")
    for pat in _ANSWER_FORMAT_PATTERNS:
        out = pat.sub("", out)
    return out.strip()


def _skill_user_prompt(problem: str, solution: str) -> str:
    return (
        f"PROBLEM:\n{_clean_problem_for_skill(problem)}\n\n"
        f"WORKED SOLUTION (reference, do not echo):\n{solution}\n\n"
        "Distill the transferable know-how into 1-3 [Knowledge/Rule]/[Details/Examples] "
        "tiny-skill blocks (each reusable on OTHER problems; do not state this "
        "problem's final answer)."
    )


def _failure_kind(args: Namespace, sample: Sample) -> str:
    """Classify why a trace failed (truncated | format | wrong) so the pitfall
    generator can tailor its diagnosis."""
    try:
        if sample.status == Sample.Status.TRUNCATED:
            return "truncated"
    except Exception:
        pass
    if _extract_answer(args, sample) is None:
        return "format"
    return "wrong"


_FAILURE_NOTE = {
    "truncated": (
        "NOTE: this attempt was CUT OFF by the response-length limit before it "
        "finished. The reasoning may have been on track; the pitfall is more likely "
        "about efficiency/length (e.g. being too verbose, not reaching the answer in "
        "time) than a conceptual error. Judge accordingly."
    ),
    "format": (
        "NOTE: this attempt produced NO parseable final answer (missing/emptly answer "
        "tag). The reasoning may be fine but the OUTPUT FORMAT is broken. The pitfall "
        "should stress following the required answer format."
    ),
    "wrong": (
        "NOTE: this attempt gave a complete but INCORRECT answer. The pitfall should "
        "target the conceptual/computational mistake that led to the wrong answer."
    ),
}


def _skill_user_prompt_incorrect(
    problem: str, attempt: str, ground_truth: str, failure_kind: str = "wrong", env_feedback: str = ""
) -> str:
    """User prompt for the incorrect-trace skill: give the ground-truth answer only
    so the model can localize the mistake, then emit pitfall warnings (no solution,
    no answer). failure_kind tailors the diagnosis; env_feedback (optional) grounds
    it in the trace's own tool-execution output."""
    gt = (ground_truth or "").strip()
    note = _FAILURE_NOTE.get(failure_kind, _FAILURE_NOTE["wrong"])
    env_section = f"TOOL EXECUTION FEEDBACK FROM THIS ATTEMPT:\n{env_feedback.strip()}\n\n" if env_feedback.strip() else ""
    return (
        f"PROBLEM:\n{_clean_problem_for_skill(problem)}\n\n"
        f"FAILED ATTEMPT (wrong, do not echo):\n{attempt}\n\n"
        f"GROUND-TRUTH ANSWER (for locating the mistake only, do NOT put it in the output):\n{gt}\n\n"
        f"{env_section}"
        f"{note}\n\n"
        "Identify the specific mistake(s) and write them as [Error]/[Rule]/[Example] "
        "blocks (1-3 blocks, using the literal headers; specific and concrete; no "
        "solution, no answer)."
    )


async def _condense_trace_to_skill(args: Namespace, problem: str, solution: str) -> str:
    """Distill one worked solution into a short skill via the OpenAI-compatible LLM;
    returns the original full solution on any failure."""
    base_url = getattr(args, "sdpo_condense_base_url", "https://api.openai.com/v1").rstrip("/")
    model = getattr(args, "sdpo_condense_model", "gpt-5.4-mini")
    api_key = os.environ.get(getattr(args, "sdpo_condense_api_key_env", "OPENAI_API_KEY"), "") or "EMPTY"
    max_tokens = int(getattr(args, "sdpo_condense_max_tokens", 2048))

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SKILL_SYSTEM_PROMPT},
            {"role": "user", "content": _skill_user_prompt(problem, solution)},
        ],
    }
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_completion_tokens"] = max_tokens
        payload["temperature"] = 0.0
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        out = await post(f"{base_url}/chat/completions", payload, max_retries=3, headers=headers)
        skill = (out["choices"][0]["message"].get("content") or "").strip()
        return skill if skill else solution  # fall back to full trace on empty
    except Exception as e:
        logger.warning(f"trace condense failed ({e!r}); falling back to full trace.")
        return solution


async def _external_llm_chat(args: Namespace, system: str, user: str) -> str | None:
    """Single (system, user) -> text completion via the OpenAI-compatible condenser
    endpoint; returns None on failure."""
    base_url = getattr(args, "sdpo_condense_base_url", "https://api.openai.com/v1").rstrip("/")
    model = getattr(args, "sdpo_condense_model", "gpt-5.4-mini")
    api_key = os.environ.get(getattr(args, "sdpo_condense_api_key_env", "OPENAI_API_KEY"), "") or "EMPTY"
    max_tokens = int(getattr(args, "sdpo_condense_max_tokens", 2048))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_completion_tokens"] = max_tokens
        payload["temperature"] = 0.0
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        out = await post(f"{base_url}/chat/completions", payload, max_retries=3, headers=headers)
        return (out["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        logger.warning(f"external LLM chat failed ({e!r}).")
        return None


async def _generate_skill_text(args: Namespace, system: str, user: str, backend: str) -> str:
    """Generate a skill/pitfall text from a (system, user) prompt using backend
    'self' (the current policy over the rollout engine) or 'external' (the
    OpenAI-compatible LLM). Returns "" on failure."""
    if backend == "external":
        return (await _external_llm_chat(args, system, user)) or ""
    # self / policy path: chat-template, generate on the rollout engine, decode.
    tok = _tokenizer(args)
    text = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        **_skill_gen_template_kwargs(),
    )
    prompt_ids = tok.encode(text, add_special_tokens=False)
    res = await _self_generate_skill(args, prompt_ids)
    return res[0] if res else ""


async def _condense_solutions(args: Namespace, pairs: list[tuple[str, str]]) -> list[str]:
    """Condense a batch of (problem, solution) pairs into skills concurrently (with a
    process-wide cap), deduplicating identical pairs."""
    sem = _condense_semaphore(args)
    # dedup
    uniq: dict[tuple[str, str], int] = {}
    for p in pairs:
        uniq.setdefault(p, len(uniq))
    keys = list(uniq.keys())

    async def _one(problem: str, solution: str) -> str:
        async with sem:
            return await _condense_trace_to_skill(args, problem, solution)

    skills = await asyncio.gather(*(_one(pr, sol) for pr, sol in keys))
    skill_of = {k: s for k, s in zip(keys, skills)}
    return [skill_of[p] for p in pairs]


# --------------------------------------------------------------------------- #
# Self-generated skill  (the current policy writes the skill during rollout)
# --------------------------------------------------------------------------- #


def _skill_gen_prompt_ids(
    args,
    tok,
    problem: str,
    solution: str,
    *,
    correct: bool = True,
    ground_truth: str = "",
    failure_kind: str = "wrong",
    env_feedback: str = "",
) -> list[int]:
    """Tokenized, chat-templated skill-generation prompt (the STUDENT context the
    skill is generated in). correct=False switches to the incorrect-trace framing
    (attempt is wrong, ground truth supplied, distill a pitfall-prevention skill);
    failure_kind and env_feedback tailor that diagnosis."""
    solution = _strip_response_eos(solution)
    if correct:
        system = _SKILL_SYSTEM_PROMPT
        user = _skill_user_prompt(problem, solution)
    else:
        system = _SKILL_SYSTEM_PROMPT_INCORRECT
        user = _skill_user_prompt_incorrect(
            problem, solution, ground_truth, failure_kind=failure_kind, env_feedback=env_feedback
        )
    text = tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
        **_skill_gen_template_kwargs(),
    )
    return tok.encode(text, add_special_tokens=False)


def _skill_gen_template_kwargs() -> dict:
    """Chat-template kwargs for SKILL GENERATION: force enable_thinking=False so a
    thinking model spends its whole budget on the skill, not a <think> chain that
    leaks into the KD prefix. Harmless for non-thinking models."""
    return {"enable_thinking": False}


def _strip_think_tokens(tok, tokens: list[int], logprobs: list[float]) -> tuple[list[int], list[float]]:
    """Drop everything up to and including the first </think> from a token sequence,
    keeping tokens and logprobs aligned. Located at the TOKEN level so the returned
    tokens are exactly the ones the policy generated (required for skill-KD logprob
    alignment). No-op if no </think> is present."""
    if not tokens:
        return tokens, logprobs
    full = tok.decode(tokens)
    if "</think>" not in full:
        return tokens, logprobs
    # "</think>" in decode(tokens[:j]) is monotonic in j, so binary-search the
    # smallest j whose prefix already closes the tag.
    lo, hi = 1, len(tokens)
    while lo < hi:
        mid = (lo + hi) // 2
        if "</think>" in tok.decode(tokens[:mid]):
            hi = mid
        else:
            lo = mid + 1
    return tokens[lo:], logprobs[lo:]


async def _self_generate_skill(args: Namespace, prompt_ids: list[int]):
    """Have the current policy generate a skill from the skill-gen prompt. Returns
    (skill_text, skill_token_ids, skill_logprobs) or None on failure."""
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "temperature": getattr(args, "rollout_temperature", 1.0),
            "max_new_tokens": int(getattr(args, "sdpo_skill_max_new_tokens", 512)),
            "skip_special_tokens": False,
        },
        "return_logprob": True,
    }
    try:
        out = await post(url, payload)
        meta = out.get("meta_info", {})
        otl = meta.get("output_token_logprobs")  # list of [logprob, token_id, ...]
        if not otl:
            return None
        skill_tokens = [int(x[1]) for x in otl]
        skill_logprobs = [float(x[0]) for x in otl]
        # Drop trailing EOS/stop tokens, keeping tokens+logprobs aligned.
        tok = _tokenizer(args)
        stop_ids = {tok.eos_token_id}
        for s in ("<|endoftext|>", "<|im_end|>", "<|eot_id|>"):
            try:
                sid = tok.convert_tokens_to_ids(s)
                if isinstance(sid, int) and sid >= 0:
                    stop_ids.add(sid)
            except Exception:
                pass
        while skill_tokens and skill_tokens[-1] in stop_ids:
            skill_tokens.pop()
            skill_logprobs.pop()
        # Strip a leading <think>...</think> chain at the token level so the KD skill
        # is the actual roadmap. No-op when the flag is off or there is no </think>.
        if getattr(args, "sdpo_remove_thinking_from_demonstration", False):
            skill_tokens, skill_logprobs = _strip_think_tokens(tok, skill_tokens, skill_logprobs)
        if not skill_tokens:
            return None
        return tok.decode(skill_tokens), skill_tokens, skill_logprobs
    except Exception as e:
        logger.warning(f"self-skill generation failed ({e!r}); falling back to full trace.")
        return None


def _tokenizer(args: Namespace):
    # Reuse rollout's process-wide GenerateState tokenizer so we tokenize exactly
    # like the rollout engine.
    from miles.rollout.sglang_rollout import GenerateState

    return GenerateState(args).tokenizer


# --------------------------------------------------------------------------- #
# teacher scoring
# --------------------------------------------------------------------------- #


def _teacher_url(args: Namespace) -> str:
    """Where to send teacher scoring requests: the rollout engine itself under
    --sdpo-self-teacher (default, true self-distillation), else the fixed external
    teacher at --rm-url."""
    if getattr(args, "sdpo_self_teacher", True):
        return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    return args.rm_url


async def _teacher_score(args: Namespace, input_ids: list[int], token_ids: list[int] | None):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if token_ids:
        payload["token_ids_logprob"] = token_ids
    # miles' shared HTTP client (retries + shared pool), same as rollout/OPD scoring.
    return await post(_teacher_url(args), payload)


def _trim_to_response(values: list[Any], response_length: int) -> list[Any]:
    """Drop SGLang's leading placeholder position, then keep the response span, so
    position i is the teacher's prediction for response token i (aligned to student)."""
    if values is None:
        raise ValueError("Teacher response is missing an expected meta_info logprob field.")
    trimmed = values[1:][-response_length:] if response_length > 0 else []
    if len(trimmed) != response_length:
        raise ValueError(
            f"Teacher/response alignment mismatch: got {len(trimmed)} positions, expected {response_length}."
        )
    return trimmed


def _entries_to_map(entries: Any) -> dict[int, float]:
    if not entries:
        return {}
    return {int(e[1]): float(e[0]) for e in entries if e is not None}


# --------------------------------------------------------------------------- #
# divergences
# --------------------------------------------------------------------------- #


def _distribution_divergence(p_s: Sequence[float], p_t: Sequence[float], mode: str) -> float:
    eps = 1e-12
    if mode == "reverse_kl":
        return sum(s * math.log((s + eps) / (t + eps)) for s, t in zip(p_s, p_t, strict=True))
    if mode == "forward_kl":
        return sum(t * math.log((t + eps) / (s + eps)) for s, t in zip(p_s, p_t, strict=True))
    if mode == "jeffrey":  # forward KL + reverse KL
        return sum(
            s * math.log((s + eps) / (t + eps)) + t * math.log((t + eps) / (s + eps))
            for s, t in zip(p_s, p_t, strict=True)
        )
    if mode == "jeffrey_jsd":  # forward KL + JSD (reverse-KL half swapped for JSD)
        total = 0.0
        for s, t in zip(p_s, p_t, strict=True):
            m = 0.5 * (s + t)
            fkl = t * math.log((t + eps) / (s + eps))
            jsd = 0.5 * s * math.log((s + eps) / (m + eps)) + 0.5 * t * math.log((t + eps) / (m + eps))
            total += fkl + jsd
        return total
    total = 0.0  # jsd
    for s, t in zip(p_s, p_t, strict=True):
        m = 0.5 * (s + t)
        total += 0.5 * s * math.log((s + eps) / (m + eps)) + 0.5 * t * math.log((t + eps) / (m + eps))
    return total


def _probs_with_tail(logps: Sequence[float]) -> list[float]:
    """Turn full-vocab-normalised log-probs over a token subset into a distribution by
    appending one aggregated tail bucket for the remaining vocabulary mass."""
    probs = [math.exp(lp) for lp in logps]
    tail = max(0.0, 1.0 - math.fsum(probs))
    return probs + [tail]


def _sampled_divergence(student_logp: float, teacher_logp: float, mode: str) -> float:
    """Per-token divergence when only the sampled token's log-prob is available:
    treats the token as a 2-point (sampled vs. rest) Bernoulli."""
    if mode == "reverse_kl":
        return student_logp - teacher_logp
    if mode == "forward_kl":
        return teacher_logp - student_logp
    p_s = min(max(math.exp(student_logp), 0.0), 1.0)  # jsd over {sampled, rest}
    p_t = min(max(math.exp(teacher_logp), 0.0), 1.0)
    return _distribution_divergence([p_s, 1.0 - p_s], [p_t, 1.0 - p_t], "jsd")


# --------------------------------------------------------------------------- #
# per-sample KL computation
# --------------------------------------------------------------------------- #


def _topk_divergences_np(
    student_maps: list[dict[int, float]],
    teacher_maps: list[dict[int, float]],
    response_tokens: list[int],
    divergence_mode: str,
) -> tuple[list[float], list[float], list[float]]:
    """Numpy-vectorized per-token divergence (CPU only). Top-k logprobs -> probs plus
    an aggregated tail bucket, missing teacher id -> logprob -100. Rows are the
    student's top-k ids per position (ragged rows handled per-position). Returns
    (divergences, student_sampled_logps, teacher_sampled_logps)."""
    n = len(student_maps)
    eps = 1e-12
    NEG = -100.0

    widths = [len(m) for m in student_maps]
    k = max(widths, default=0)
    if n == 0 or k == 0:
        return [0.0] * n, [], []

    uniform = all(w == k for w in widths)
    student_sampled_logps: list[float] = []
    teacher_sampled_logps: list[float] = []

    if uniform:
        s_rows = [[student_maps[i][t] for t in student_maps[i]] for i in range(n)]
        t_rows = [[teacher_maps[i].get(t, NEG) for t in student_maps[i]] for i in range(n)]
        s = np.asarray(s_rows, dtype=np.float64)
        t = np.asarray(t_rows, dtype=np.float64)
        p_s = np.exp(s)
        p_t = np.exp(t)
        tail_s = np.clip(1.0 - p_s.sum(1), 0.0, None)[:, None]
        tail_t = np.clip(1.0 - p_t.sum(1), 0.0, None)[:, None]
        p_s = np.concatenate([p_s, tail_s], axis=1)
        p_t = np.concatenate([p_t, tail_t], axis=1)
        if divergence_mode == "reverse_kl":
            div = (p_s * np.log((p_s + eps) / (p_t + eps))).sum(1)
        elif divergence_mode == "forward_kl":
            div = (p_t * np.log((p_t + eps) / (p_s + eps))).sum(1)
        else:  # jsd
            m = 0.5 * (p_s + p_t)
            div = (0.5 * p_s * np.log((p_s + eps) / (m + eps)) + 0.5 * p_t * np.log((p_t + eps) / (m + eps))).sum(1)
        divergences = div.tolist()
    else:
        # Ragged case (some positions have < k ids): per-position fallback.
        divergences = []
        for i in range(n):
            sm = student_maps[i]
            if not sm:
                divergences.append(0.0)
                continue
            tm = teacher_maps[i]
            ids = list(sm.keys())
            p_s = np.exp(np.asarray([sm[t] for t in ids], dtype=np.float64))
            p_t = np.exp(np.asarray([tm.get(t, NEG) for t in ids], dtype=np.float64))
            p_s = np.append(p_s, max(0.0, 1.0 - p_s.sum()))
            p_t = np.append(p_t, max(0.0, 1.0 - p_t.sum()))
            if divergence_mode == "reverse_kl":
                divergences.append(float((p_s * np.log((p_s + eps) / (p_t + eps))).sum()))
            elif divergence_mode == "forward_kl":
                divergences.append(float((p_t * np.log((p_t + eps) / (p_s + eps))).sum()))
            else:
                m = 0.5 * (p_s + p_t)
                divergences.append(
                    float(
                        (0.5 * p_s * np.log((p_s + eps) / (m + eps)) + 0.5 * p_t * np.log((p_t + eps) / (m + eps))).sum()
                    )
                )

    # Sampled-token diagnostics (cheap scalar gather).
    for i in range(n):
        sm = student_maps[i]
        tok = response_tokens[i]
        if tok in sm:
            student_sampled_logps.append(sm[tok])
            teacher_sampled_logps.append(teacher_maps[i].get(tok, NEG))

    return divergences, student_sampled_logps, teacher_sampled_logps


async def _compute_kl_for_sample(
    args: Namespace,
    sample: Sample,
    prefix_sample: Sample,
    logprob_mode: str,
    divergence_mode: str,
) -> torch.Tensor:
    n = sample.response_length
    prompt_tokens = sample.tokens[: _prompt_len(sample)]
    response_tokens = _response_tokens(sample)

    # Insert the correct peer solution as a prefix between prompt and response
    # (response stays at the tail -> aligned). NOTE: this legacy sglang-teacher path
    # splices inside the assistant turn; the active megatron path (sdpo_group_reward)
    # instead puts the solution in the USER turn -- port _build_teacher_prompt_str
    # here if this path is revived.
    _t = time.perf_counter()
    prefix_text = _render_prefix(prefix_sample.response)
    prefix_tokens = _tokenizer(args).encode(prefix_text, add_special_tokens=False)
    teacher_input = prompt_tokens + prefix_tokens + response_tokens
    _sdpo_timing["tokenize"] += time.perf_counter() - _t

    if logprob_mode == "sampled":
        student_logps = sample.rollout_log_probs
        if student_logps is None or len(student_logps) != n:
            raise ValueError(
                f"sampled mode needs rollout_log_probs of length {n}, got "
                f"{None if student_logps is None else len(student_logps)}."
            )
        teacher = await _teacher_score(args, teacher_input, token_ids=None)
        teacher_entries = _trim_to_response(teacher["meta_info"]["input_token_logprobs"], n)
        divergences = []
        for i in range(n):
            # Alignment guarantee: teacher's token at this position IS response[i].
            if int(teacher_entries[i][1]) != response_tokens[i]:
                raise ValueError(
                    f"Token misalignment at position {i}: teacher={teacher_entries[i][1]}, "
                    f"student={response_tokens[i]}."
                )
            divergences.append(_sampled_divergence(student_logps[i], float(teacher_entries[i][0]), divergence_mode))
        return torch.tensor(divergences, dtype=torch.float32)

    # topk: per-position distribution over the student's top-k tokens + a tail bucket.
    raw = sample.metadata.get("opd_student_top_logprobs")
    if raw is None:
        raise ValueError("topk mode needs student top-k logprobs; set --opd-log-prob-top-k > 0 (e.g. 128).")
    _t = time.perf_counter()
    student_maps = [_entries_to_map(pos) for pos in (raw[-n:] if n > 0 else [])]
    if len(student_maps) != n:
        raise ValueError(f"Student top-k length mismatch: got {len(student_maps)}, expected {n}.")
    # Query the teacher for exactly the student's top-k token ids at each position.
    union_ids = sorted({tid for pos in student_maps for tid in pos})
    _sdpo_timing["student_maps"] += time.perf_counter() - _t

    _t = time.perf_counter()
    teacher = await _teacher_score(args, teacher_input, token_ids=union_ids)
    _sdpo_timing["teacher_http"] += time.perf_counter() - _t

    _t = time.perf_counter()
    teacher_maps = [
        _entries_to_map(pos) for pos in _trim_to_response(teacher["meta_info"]["input_token_ids_logprobs"], n)
    ]
    _sdpo_timing["teacher_maps"] += time.perf_counter() - _t

    # Numpy-vectorized per-token divergence, run in a worker thread so this
    # CPU-bound work does not block the event loop.
    _t = time.perf_counter()
    divergences, student_sampled_logps, teacher_sampled_logps = await asyncio.to_thread(
        _topk_divergences_np, student_maps, teacher_maps, response_tokens, divergence_mode
    )
    _sdpo_timing["divergence"] += time.perf_counter() - _t

    # Stash per-sample scalar diagnostics for rollout logging.
    if student_sampled_logps:
        s_mean = sum(student_sampled_logps) / len(student_sampled_logps)
        t_mean = sum(teacher_sampled_logps) / len(teacher_sampled_logps)
        if isinstance(sample.metadata, dict):
            sample.metadata["sdpo_student_logp_mean"] = s_mean
            sample.metadata["sdpo_teacher_logp_mean"] = t_mean
            sample.metadata["sdpo_logp_diff_mean"] = s_mean - t_mean

    return torch.tensor(divergences, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# entry point: group-level async reward model
# --------------------------------------------------------------------------- #


async def sdpo_group_reward(args: Namespace, group: list[Sample], **kwargs: Any) -> list[float]:
    """Group RM: returns the task reward per trace and, as a side effect, writes the
    per-token SDPO divergence into ``sample.opd_reverse_kl`` (or, on the megatron
    teacher path, the teacher prompt tokens for the training side to score)."""
    logprob_mode = _logprob_mode(args)
    divergence_mode = _divergence_mode(args)

    # Correctness selects which traces can serve as a correct-peer prefix. With
    # --sdpo-judge this is an LLM-as-judge grade; otherwise deterministic matching.
    correctness = await _grade_group(args, group)
    correct_indices = [i for i, ok in enumerate(correctness) if ok]

    # Log true task success + perplexity per trace on metadata (returned reward is 0
    # under pure distill, so success rate would otherwise be invisible).
    for ok, s in zip(correctness, group, strict=True):
        if not isinstance(s.metadata, dict):
            continue
        s.metadata["sdpo_correct"] = 1.0 if ok else 0.0
        # PPL of the sampled response = exp(mean negative student log-prob).
        logps = s.rollout_log_probs
        if logps:
            nll = -sum(logps) / len(logps)
            s.metadata["sdpo_ppl"] = math.exp(min(nll, 20.0))  # clamp to avoid overflow

    # Pure distillation (default): return 0 task reward so the GRPO advantage is 0 and
    # the target is exactly -opd_kl_coef * divergence. Else keep mixed GRPO + distill.
    if getattr(args, "sdpo_pure_distill", True):
        rewards = [0.0 for _ in group]
    else:
        rewards = [1.0 if ok else 0.0 for ok in correctness]

    # Need >= 1 correct trace for a valid peer prefix. A single correct trace can
    # serve as prefix for all incorrect traces; the correct trace itself gets an
    # empty prefix (self-excluded -> empty pool).
    enable_kl = len(correct_indices) >= 1

    if getattr(args, "sdpo_teacher_backend", "sglang") == "megatron":
        # Megatron teacher path: don't score here. Pick a correct peer and stash its
        # rendered+tokenized prefix on the sample; the training side (megatron actor)
        # forwards prompt+prefix+response with the current policy weights and computes
        # opd_reverse_kl (opd.py) -- a batched CUDA-graph'd forward, far faster than
        # SGLang eager full-seq-logprob scoring.
        tok = _tokenizer(args)
        gen_suffix = _gen_prompt_suffix(tok, getattr(args, "apply_chat_template_kwargs", None))
        remove_thinking = getattr(args, "sdpo_remove_thinking_from_demonstration", False)
        reframe_multiturn = getattr(args, "sdpo_reframe_multiturn_prefix", False)
        tool_grammar = getattr(args, "sdpo_tool_grammar", "qwen25")  # qwen25 JSON | qwen3_coder XML
        condense = getattr(args, "sdpo_trace_condense", False)
        self_skill = getattr(args, "sdpo_self_skill", False)
        skill_kd = self_skill and getattr(args, "sdpo_skill_kd", False)
        skill_kd_mode = getattr(args, "sdpo_skill_kd_mode", "self-success")
        skill_source = getattr(args, "sdpo_skill_source", "correct")
        # self-skill and trace-condense both produce a skill prefix; only one allowed.
        assert not (self_skill and condense), "use only one of --sdpo-self-skill / --sdpo-trace-condense"
        # pitfall-condense/both-blind distil FAILED traces -> skill-source must cover them.
        assert not (
            skill_kd and skill_kd_mode in ("pitfall-condense", "both-blind") and skill_source not in ("incorrect", "all")
        ), "--sdpo-skill-kd-mode pitfall-condense|both-blind requires --sdpo-skill-source incorrect|all"
        assert not (skill_kd and skill_kd_mode == "both" and skill_source != "all"), (
            "--sdpo-skill-kd-mode both trains correct (self-success) AND failed "
            "(pitfall-condense) traces, so it requires --sdpo-skill-source all"
        )
        assert not (
            skill_kd and skill_kd_mode == "blind-correct" and skill_source not in ("correct", "all")
        ), "--sdpo-skill-kd-mode blind-correct requires --sdpo-skill-source correct|all"
        assert not (skill_kd and skill_kd_mode == "both-blind" and skill_source != "all"), (
            "--sdpo-skill-kd-mode both-blind trains correct (blind-correct) AND failed "
            "(pitfall-condense) traces, so it requires --sdpo-skill-source all"
        )
        # --sdpo-blind-correct-info only reaches a teacher prompt via the blind-correct pass.
        assert getattr(args, "sdpo_blind_correct_info", "trace") == "trace" or (
            skill_kd and skill_kd_mode in ("blind-correct", "both-blind")
        ), (
            "--sdpo-blind-correct-info only applies to the blind-correct teacher, so it "
            "requires --sdpo-skill-kd with --sdpo-skill-kd-mode blind-correct|both-blind"
        )

        response_prefix = getattr(args, "sdpo_response_prefix", "trace")
        pitfall_backend = getattr(args, "sdpo_pitfall_summary_backend", "self")
        # Pitfall injection: when self-skill distils failed traces, the group's common
        # failure lessons are spliced into FAILED traces' teacher prefix.
        pitfall_active = self_skill and skill_source in ("incorrect", "all")
        # A FAILED trace's prefix is built from other failed traces' pitfalls, not a
        # correct peer, so an all-wrong group (0 correct) must still get prefixes --
        # override enable_kl's "at least 1 correct trace" gate.
        if pitfall_active:
            enable_kl = True

        # Pass 1: pick each trace's correct peer (self-excluded). prefix_text is the
        # peer's solution (full response, or its distilled skill under trace-condense).
        # peer_by_idx remembers the peer so --sdpo-response-prefix skill can later swap
        # in the peer's skill. Under pitfall injection, no-peer FAILED traces still
        # enter prefix_text_by_idx (empty base) so pitfalls can be spliced in.
        prefix_text_by_idx: dict[int, str] = {}
        # Peer's structured message dict (metadata["messages"]) when present -- the
        # dict-native prefix source, preferred over the raw-text reframe.
        prefix_messages_by_idx: dict[int, list] = {}
        peer_by_idx: dict[int, int] = {}
        # The group's sole correct trace (no correct peer to borrow from). Nothing of
        # its own to diagnose, but the group's failed traces produced pitfalls -- give
        # it those in pass 2 rather than zero teacher signal.
        sole_correct_no_peer_idxs: set[int] = set()
        for i, sample in enumerate(group):
            if not isinstance(sample.metadata, dict):
                continue
            if sample.response_length == 0 or not enable_kl:
                sample.metadata["sdpo_teacher_prompt_tokens"] = []
                continue
            peers = [j for j in correct_indices if j != i]
            if not peers:
                # No correct peer. Under pitfall injection, both failed traces and the
                # sole correct trace get an empty base prefix (pitfalls appended in
                # pass 2); with injection off, a no-peer trace gets no prefix.
                self_ok = bool(correctness[i]) if i < len(correctness) else False
                if pitfall_active:
                    prefix_text_by_idx[i] = ""
                    if self_ok:
                        sole_correct_no_peer_idxs.add(i)
                else:
                    sample.metadata["sdpo_teacher_prompt_tokens"] = []
                continue
            peer_j = _choose_peer(args, group, peers)
            peer_by_idx[i] = peer_j
            prefix_text_by_idx[i] = group[peer_j].response
            peer_md = group[peer_j].metadata if isinstance(group[peer_j].metadata, dict) else {}
            peer_msgs = peer_md.get("messages")
            if peer_msgs:
                prefix_messages_by_idx[i] = peer_msgs

        # Optional: the current policy self-generates a skill from each trace's own
        # response, and (for skill-KD) a second SDPO runs on the skill tokens.
        # --sdpo-skill-source gates WHICH traces get a skill; it does not change the
        # response teacher prefix (still a correct peer, above).
        if self_skill:

            def _problem_of(j: int) -> str:
                md = group[j].metadata if isinstance(group[j].metadata, dict) else {}
                q = md.get("question")
                if q:
                    return str(q)
                p = group[j].prompt
                return p if isinstance(p, str) else str(p)

            def _has_env_feedback(i: int) -> bool:
                md = group[i].metadata if isinstance(group[i].metadata, dict) else {}
                return bool(md.get("tool_trace"))

            def _skill_eligible(i: int) -> bool:
                if group[i].response_length == 0:
                    return False
                self_ok = bool(correctness[i]) if i < len(correctness) else False
                if skill_source == "correct":
                    return self_ok
                if skill_source == "incorrect":
                    return not self_ok
                if skill_source == "env_feedback":
                    # Grounded pitfall generation from the trace's own tool-execution
                    # trace (metadata["tool_trace"], stashed by sdpo_react.py); only for
                    # a FAILED trace that actually called a tool.
                    return (not self_ok) and _has_env_feedback(i)
                return True  # "all"

            skill_idxs = [i for i in range(len(group)) if isinstance(group[i].metadata, dict) and _skill_eligible(i)]

            def _render_env_feedback(i: int) -> str:
                """Render trace i's tool_trace as plain text for the pitfall prompt,
                truncated to --sdpo-env-feedback-max-chars; "" when the trace has no
                tool_trace (a no-op for _skill_user_prompt_incorrect)."""
                md = group[i].metadata if isinstance(group[i].metadata, dict) else {}
                trace = md.get("tool_trace") or []
                if not trace:
                    return ""
                max_chars = int(getattr(args, "sdpo_env_feedback_max_chars", 2000))
                parts = [f"[call {j + 1}] {t['tool_call']}\n[result {j + 1}] {t['observation']}" for j, t in enumerate(trace)]
                text = "\n\n".join(parts)
                return text if len(text) <= max_chars else text[-max_chars:]

            async def _gen_one(i: int):
                # Distill the trace's own response into a skill: for a correct trace
                # "extract the transferable procedure", for an incorrect one "diagnose
                # the error -> pitfall warnings" (tailored by failure_kind; ground truth
                # passed to localize the mistake; env_feedback grounds it in tool calls).
                problem = _problem_of(i)
                self_ok = bool(correctness[i]) if i < len(correctness) else False
                fkind = "wrong" if self_ok else _failure_kind(args, group[i])
                # Prefer the dict-native prose (metadata["messages"]) over the raw
                # ChatML response so the generator reads clean per-round prose.
                _md_i = group[i].metadata if isinstance(group[i].metadata, dict) else {}
                _msgs_i = _md_i.get("messages")
                solution_i = _reframe_messages_to_prose(_msgs_i, grammar=tool_grammar) if _msgs_i else group[i].response
                gen_prompt_ids = _skill_gen_prompt_ids(
                    args,
                    tok,
                    problem,
                    solution_i,
                    correct=self_ok,
                    ground_truth=(group[i].label or "") if not self_ok else "",
                    failure_kind=fkind,
                    env_feedback=_render_env_feedback(i) if not self_ok else "",
                )
                res = await _self_generate_skill(args, gen_prompt_ids)
                return i, gen_prompt_ids, res

            gen_results = await asyncio.gather(*(_gen_one(i) for i in skill_idxs))
            for i, gen_prompt_ids, res in gen_results:
                if res is None:
                    continue
                skill_text, skill_tokens, skill_logprobs = res
                md = group[i].metadata
                md["sdpo_skill"] = skill_text
                # Preserve the ORIGINAL per-trace skill/pitfall under a stable key: the
                # pitfall-condense and blind-correct passes below overwrite sdpo_skill
                # with a problem-only prediction, but the group summary and those passes'
                # privileged info need the original per-trace text.
                self_ok_i = bool(correctness[i]) if i < len(correctness) else False
                if not self_ok_i:
                    md["sdpo_trace_pitfall"] = skill_text
                else:
                    md["sdpo_trace_skill"] = skill_text
                # rollout-side skill metrics: length + perplexity.
                md["sdpo_skill_len"] = float(len(skill_tokens))
                if skill_logprobs:
                    _nll = -sum(skill_logprobs) / len(skill_logprobs)
                    md["sdpo_skill_ppl"] = math.exp(min(_nll, 20.0))
                # Skill tokens/prompt/logprobs are stashed unconditionally (dump-only;
                # the skill-KD training path is gated on --sdpo-skill-kd at its call
                # site, not on these keys).
                md["sdpo_skill_tokens"] = skill_tokens
                md["sdpo_skill_prompt_tokens"] = gen_prompt_ids
                md["sdpo_skill_rollout_logprobs"] = skill_logprobs
                # Skill-KD teacher hint (KD-only): self-success -> skill-gen prompt +
                # this sample's own trace; problem-only -> skill-gen prompt, no hint;
                # pitfall-condense/both -> handled in dedicated passes below.
                self_ok_kd = bool(correctness[i]) if i < len(correctness) else False
                use_self_success = skill_kd_mode == "self-success" or (skill_kd_mode == "both" and self_ok_kd)
                if skill_kd and (use_self_success or skill_kd_mode == "problem-only"):
                    if use_self_success:
                        gen_prompt_str = tok.decode(gen_prompt_ids)
                        skill_teacher_str = _build_skill_self_success_teacher_prompt_str(gen_prompt_str, gen_suffix)
                        md["sdpo_skill_teacher_prompt_tokens"] = tok.encode(
                            skill_teacher_str, add_special_tokens=False
                        )
                    else:  # problem-only: no hint -> teacher context == student context
                        md["sdpo_skill_teacher_prompt_tokens"] = list(gen_prompt_ids)

            # pitfall-condense skill-KD: a separate skill OPD on failed traces. Student
            # predicts pitfalls from the PROBLEM ONLY; teacher = same prompt + the
            # group's per-trace failure skills as privileged info; KD target = the
            # student's own problem-only prediction. Regenerated under the problem-only
            # context so the KD'd tokens match it.
            if skill_kd and skill_kd_mode in ("pitfall-condense", "both", "both-blind"):
                failed_idxs = [
                    i for i in skill_idxs
                    if not (bool(correctness[i]) if i < len(correctness) else False)
                    and isinstance(group[i].metadata, dict)
                    and (group[i].metadata.get("sdpo_trace_pitfall") or "").strip()
                ]
                # Privileged failure info = all failed traces' per-trace pitfalls.
                failure_info = "\n\n".join(
                    (group[j].metadata.get("sdpo_trace_pitfall") or "").strip() for j in failed_idxs
                )

                async def _gen_predict(i: int):
                    # _skill_gen_template_kwargs() (enable_thinking=False) to match every
                    # other skill-generation call site; without it this student would
                    # think freely while its siblings are forced no-think.
                    stu_text = tok.apply_chat_template(
                        [
                            {"role": "system", "content": _PITFALL_PREDICT_SYSTEM},
                            {"role": "user", "content": _pitfall_predict_user_prompt(_problem_of(i))},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        **_skill_gen_template_kwargs(),
                    )
                    stu_ids = tok.encode(stu_text, add_special_tokens=False)
                    res2 = await _self_generate_skill(args, stu_ids)
                    return i, stu_ids, res2

                predict_results = await asyncio.gather(*(_gen_predict(i) for i in failed_idxs))
                for i, stu_ids, res2 in predict_results:
                    if res2 is None:
                        continue
                    p_text, p_tokens, p_logprobs = res2
                    md = group[i].metadata
                    # Overwrite the skill-KD payload with the problem-only student and
                    # failure-informed teacher.
                    md["sdpo_skill"] = p_text
                    md["sdpo_skill_len"] = float(len(p_tokens))
                    if p_logprobs:
                        _nll = -sum(p_logprobs) / len(p_logprobs)
                        md["sdpo_skill_ppl"] = math.exp(min(_nll, 20.0))
                    md["sdpo_skill_tokens"] = p_tokens
                    md["sdpo_skill_prompt_tokens"] = stu_ids
                    md["sdpo_skill_rollout_logprobs"] = p_logprobs
                    stu_prompt_str = tok.decode(stu_ids)
                    teacher_str = _build_failure_teacher_prompt_str(stu_prompt_str, gen_suffix, failure_info)
                    md["sdpo_skill_teacher_prompt_tokens"] = tok.encode(teacher_str, add_special_tokens=False)

            # blind-correct skill-KD: symmetric counterpart to pitfall-condense for
            # CORRECT traces. Student predicts general knowledge from the PROBLEM ONLY;
            # teacher = same prompt + privileged correct-side info selected by
            # --sdpo-blind-correct-info ('trace': this trace's own solution;
            # 'group-skills': every correct trace's hindsight skill); KD target = the
            # student's own problem-only knowledge prediction.
            if skill_kd and skill_kd_mode in ("blind-correct", "both-blind"):
                correct_idxs = [
                    i for i in skill_idxs
                    if (bool(correctness[i]) if i < len(correctness) else False)
                ]
                # Group-shared privileged info, built once (mirrors failure_info),
                # capped by --sdpo-max-prefix-chars so it doesn't outgrow the trace it
                # replaces (the teacher prompt is a real forward pass).
                blind_info_mode = str(getattr(args, "sdpo_blind_correct_info", "trace"))
                group_correct_skills = ""
                if blind_info_mode == "group-skills":
                    _skills = [
                        s for s in (
                            (group[j].metadata.get("sdpo_trace_skill") or "").strip()
                            if isinstance(group[j].metadata, dict) else ""
                            for j in correct_idxs
                        ) if s
                    ]
                    group_correct_skills = "\n\n".join(_skills)
                    _cap = int(getattr(args, "sdpo_max_prefix_chars", 0) or 0)
                    if _cap and len(group_correct_skills) > _cap:
                        group_correct_skills = group_correct_skills[:_cap]

                async def _gen_blind(i: int):
                    stu_text = tok.apply_chat_template(
                        [
                            {"role": "system", "content": _BLIND_PREDICT_SYSTEM},
                            {"role": "user", "content": _blind_predict_user_prompt(_problem_of(i))},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        **_skill_gen_template_kwargs(),
                    )
                    stu_ids = tok.encode(stu_text, add_special_tokens=False)
                    res3 = await _self_generate_skill(args, stu_ids)
                    return i, stu_ids, res3

                blind_results = await asyncio.gather(*(_gen_blind(i) for i in correct_idxs))
                for i, stu_ids, res3 in blind_results:
                    if res3 is None:
                        continue
                    b_text, b_tokens, b_logprobs = res3
                    md = group[i].metadata
                    # Overwrite the skill-KD payload with the problem-only student and
                    # correct-solution-informed teacher.
                    md["sdpo_skill"] = b_text
                    md["sdpo_skill_len"] = float(len(b_tokens))
                    if b_logprobs:
                        _nll = -sum(b_logprobs) / len(b_logprobs)
                        md["sdpo_skill_ppl"] = math.exp(min(_nll, 20.0))
                    md["sdpo_skill_tokens"] = b_tokens
                    md["sdpo_skill_prompt_tokens"] = stu_ids
                    md["sdpo_skill_rollout_logprobs"] = b_logprobs
                    stu_prompt_str = tok.decode(stu_ids)
                    # Privileged info = this trace's own correct response (prefer the
                    # dict-native prose over raw ChatML, can't mis-split on a marker).
                    own_msgs = group[i].metadata.get("messages") if isinstance(group[i].metadata, dict) else None
                    correct_info = (
                        _reframe_messages_to_prose(own_msgs, grammar=tool_grammar)
                        if own_msgs
                        else _strip_response_eos(group[i].response)
                    )
                    info_template = CORRECT_INFO_TEMPLATE
                    if group_correct_skills:
                        # --sdpo-blind-correct-info group-skills (falls back to the
                        # per-trace solution when no correct trace produced a skill).
                        correct_info = group_correct_skills
                        info_template = CORRECT_SKILLS_TEMPLATE
                    teacher_str = _build_blind_correct_teacher_prompt_str(
                        stu_prompt_str, gen_suffix, correct_info, template=info_template
                    )
                    md["sdpo_skill_teacher_prompt_tokens"] = tok.encode(teacher_str, add_special_tokens=False)

        # Optional: distill each chosen peer trace into a transferable SKILL and use
        # that as the prefix instead of the full trace (SkillOpt / trace_condense).
        if condense and prefix_text_by_idx:
            idxs = list(prefix_text_by_idx.keys())

            def _problem_of(j: int) -> str:
                md = group[j].metadata if isinstance(group[j].metadata, dict) else {}
                q = md.get("question")
                if q:
                    return str(q)
                p = group[j].prompt
                return p if isinstance(p, str) else str(p)

            # Prefer the peer's dict-native prose over the raw ChatML response so the
            # condenser LLM reads clean per-round prose (see pass 1 / self-skill path).
            def _peer_solution(j: int) -> str:
                peer_msgs = prefix_messages_by_idx.get(j)
                if peer_msgs:
                    return _reframe_messages_to_prose(peer_msgs, grammar=tool_grammar)
                cleaned = _strip_response_eos(prefix_text_by_idx[j])
                return _reframe_multiturn_trace(cleaned) if reframe_multiturn else cleaned

            pairs = [(_problem_of(i), _peer_solution(i)) for i in idxs]
            skills = await _condense_solutions(args, pairs)
            for i, skill in zip(idxs, skills):
                full_trace = prefix_text_by_idx[i]
                prefix_text_by_idx[i] = skill
                # Record the distilled skill for the dump; condensed=False means the LLM
                # failed and we fell back to the full trace.
                if isinstance(group[i].metadata, dict):
                    group[i].metadata["sdpo_skill"] = skill
                    group[i].metadata["sdpo_skill_condensed"] = skill != full_trace

        # --sdpo-response-prefix skill: swap the response teacher prefix from the peer's
        # full trace to that peer's self-generated skill (fall back to the trace if the
        # peer has none). Skip FAILED traces under skill-source=incorrect (the peer never
        # gets a skill there, and these traces get wiped to pitfalls-only below).
        if response_prefix == "skill" and self_skill:
            for i in list(prefix_text_by_idx.keys()):
                if pitfall_active and skill_source == "incorrect" and not (
                    bool(correctness[i]) if i < len(correctness) else False
                ):
                    continue
                peer_j = peer_by_idx.get(i)
                peer_md = group[peer_j].metadata if (peer_j is not None and isinstance(group[peer_j].metadata, dict)) else {}
                peer_skill = peer_md.get("sdpo_skill")
                # 1.0 if the prefix used the peer's skill, 0.0 if it fell back to the
                # full trace. Aggregated as the response-prefix-is-skill fraction.
                if isinstance(group[i].metadata, dict):
                    group[i].metadata["sdpo_response_prefix_is_skill"] = 1.0 if peer_skill else 0.0
                if peer_skill:
                    prefix_text_by_idx[i] = peer_skill
                    # _build_teacher_prompt_str prefers peer_messages over peer_response
                    # whenever peer_messages is truthy; prefix_messages_by_idx[i] still
                    # holds the peer's ORIGINAL full trace, so clear it or the skill swap
                    # is silently overridden back to the full raw trace.
                    prefix_messages_by_idx.pop(i, None)

        # Group-aggregated pitfalls (self-skill over incorrect|all): stage 1 (above)
        # produced each failed trace's own pitfall warnings; stage 2 (here) synthesizes
        # them into one short shared list (per --sdpo-pitfall-summary-backend), spliced
        # ONLY into failed traces' teacher prefix (correct traces keep a clean peer prefix).
        group_pitfalls = ""
        if pitfall_active:
            def _problem_text(j: int) -> str:
                md_j = group[j].metadata if isinstance(group[j].metadata, dict) else {}
                q = md_j.get("question")
                if q:
                    return str(q)
                p = group[j].prompt
                return p if isinstance(p, str) else str(p)

            per_trace_pitfalls = []
            first_failed = None
            for i in range(len(group)):
                self_ok = bool(correctness[i]) if i < len(correctness) else False
                if self_ok:
                    continue
                md = group[i].metadata if isinstance(group[i].metadata, dict) else {}
                sk = md.get("sdpo_trace_pitfall")
                if sk and sk.strip():
                    per_trace_pitfalls.append(sk.strip())
                    if first_failed is None:
                        first_failed = i
            if len(per_trace_pitfalls) == 1:
                # One failed trace -> nothing to synthesize; use it directly.
                group_pitfalls = per_trace_pitfalls[0]
            elif per_trace_pitfalls:
                problem = _problem_text(first_failed)
                summary = await _generate_skill_text(
                    args,
                    _PITFALL_SUMMARY_SYSTEM,
                    _pitfall_summary_user_prompt(problem, per_trace_pitfalls),
                    pitfall_backend,
                )
                group_pitfalls = summary.strip() if summary and summary.strip() else "\n\n".join(per_trace_pitfalls)

        # Under skill-source=incorrect, FAILED traces get ONLY the group pitfall
        # summary as their prefix (there is no correct-peer skill to keep), so drop the
        # raw-trace prefix pass 1 picked. Under skill-source=all they keep BOTH the
        # peer's skill/trace AND the summary (_render_prefix appends pitfalls, doesn't
        # replace).
        if pitfall_active and skill_source == "incorrect":
            for i in list(prefix_text_by_idx.keys()):
                self_ok_i = bool(correctness[i]) if i < len(correctness) else False
                if not self_ok_i:
                    prefix_text_by_idx[i] = ""
                    prefix_messages_by_idx.pop(i, None)

        # Pass 2: build the teacher prompt (peer solution/skill spliced into the USER
        # turn) and tokenize. Shared pitfalls go into failed traces' prefix plus the
        # group's sole correct no-peer trace (else it would get zero teacher signal);
        # every other correct trace keeps a clean peer prefix.
        for i, sample in enumerate(group):
            if i not in prefix_text_by_idx:
                continue
            self_ok = bool(correctness[i]) if i < len(correctness) else False
            pitfalls_for_i = (
                group_pitfalls if (pitfall_active and (not self_ok or i in sole_correct_no_peer_idxs)) else ""
            )
            student_prompt = sample.prompt if isinstance(sample.prompt, str) else ""
            teacher_prompt_str = _build_teacher_prompt_str(
                student_prompt, gen_suffix, prefix_text_by_idx[i], remove_thinking=remove_thinking,
                pitfalls=pitfalls_for_i, reframe_multiturn=reframe_multiturn,
                peer_messages=prefix_messages_by_idx.get(i), grammar=tool_grammar,
                max_prefix_chars=getattr(args, "sdpo_max_prefix_chars", 0),
            )
            teacher_prompt_ids = tok.encode(teacher_prompt_str, add_special_tokens=False)
            # Training side builds teacher seq = teacher_prompt_ids + response_ids
            # (response at the tail so response-span outputs stay aligned).
            sample.metadata["sdpo_teacher_prompt_tokens"] = teacher_prompt_ids
            if isinstance(sample.metadata, dict):
                sample.metadata["sdpo_group_pitfalls"] = pitfalls_for_i
        return rewards

    # SGLang teacher path (original): score each trace against the rollout engine
    # over HTTP and write per-token opd_reverse_kl here. Concurrent to spread load.
    async def _score(i: int, sample: Sample) -> torch.Tensor:
        n = sample.response_length
        if n == 0 or not enable_kl:
            return torch.zeros((n,), dtype=torch.float32)
        peers = [j for j in correct_indices if j != i]
        prefix_sample = group[_choose_peer(args, group, peers)]
        return await _compute_kl_for_sample(args, sample, prefix_sample, logprob_mode, divergence_mode)

    global _sdpo_calls
    _wall = time.perf_counter()
    kls = await asyncio.gather(*(_score(i, s) for i, s in enumerate(group)))
    _wall = time.perf_counter() - _wall
    for sample, kl in zip(group, kls, strict=True):
        sample.opd_reverse_kl = kl

    # Log per-phase timing every 8 groups (sums are across concurrent traces, so
    # compare ratios, not absolute vs wall).
    _sdpo_calls += 1
    if _sdpo_calls % 8 == 0:
        t = _sdpo_timing
        logger.info(
            "SDPO timing (cumulative over %d groups): wall_last_group=%.1fs | "
            "teacher_http=%.1fs student_maps=%.1fs teacher_maps=%.1fs tokenize=%.1fs divergence=%.1fs",
            _sdpo_calls,
            _wall,
            t["teacher_http"],
            t["student_maps"],
            t["teacher_maps"],
            t["tokenize"],
            t["divergence"],
        )

    return rewards


async def plain_grpo_reward(args: Namespace, sample: Sample, **kwargs: Any) -> float:
    """Single-sample reward for a plain-GRPO baseline (--custom-rm-path, no
    --group-rm) -- the "no SDPO" ablation arm. Reuses _is_correct directly so it
    grades with the same criterion (--sdpo-grader dapo, strict_box_verify=True) every
    other arm's group reward uses, keeping the baseline comparable."""
    return 1.0 if _is_correct(sample, args) else 0.0


# --------------------------------------------------------------------------- #
# EVAL-time skill augmentation (--sdpo-eval-skill-mode, wired via
# --custom-generate-function-path examples.SRD.sdpo.sdpo_eval_generate): before
# the real eval rollout, self-predict a blind skill from the problem alone (the
# same prompts training uses) and splice it into the eval prompt, so eval measures
# the model answering WITH its own self-predicted skill in context. The mode is set
# manually to match whichever skill type(s) a given run trained (all | correct |
# pitfall).
# --------------------------------------------------------------------------- #



async def sdpo_eval_generate(input: Any) -> Any:
    """--custom-generate-function-path for eval-time skill augmentation. No-op during
    training or when the mode is 'off'; during eval, self-predicts the configured skill
    type(s) from the problem alone, splices them into the user turn, and runs the eval
    rollout on the skill-augmented prompt."""
    from miles.rollout.base_types import GenerateFnOutput
    from miles.rollout.sglang_rollout import generate

    args = input.args
    sample = input.sample
    mode = getattr(args, "sdpo_eval_skill_mode", "off")

    if not input.evaluation or mode == "off" or not isinstance(sample.prompt, str):
        sample = await generate(args, sample, input.sampling_params, evaluation=input.evaluation)
        return GenerateFnOutput(samples=sample)

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

    sample = await generate(args, sample, input.sampling_params, evaluation=input.evaluation)
    return GenerateFnOutput(samples=sample)


def _eval_judge_on(args: Namespace) -> bool:
    """Whether the LLM judge is enabled on the EVAL path: --sdpo-judge (both training
    and eval) or --sdpo-eval-judge (eval only, keeping the training grader)."""
    return bool(getattr(args, "sdpo_judge", False) or getattr(args, "sdpo_eval_judge", False))


async def sdpo_eval_reward(args: Namespace, sample: Sample, **kwargs: Any) -> float:
    """Per-sample eval RM for SDPO (--eval-custom-rm-path). Eval measures pass@1 and
    never uses the distillation signal, so it just returns the task reward, graded the
    same way as the group RM."""
    if _sample_domain(sample) == "code":
        # Code eval: run the program against its test cases (bounded by the shared cap).
        async with _judge_semaphore(args):
            ok = await _grade_one_code(sample, args)
    elif _sample_domain(sample) == "search":
        # Search/QA eval: EM with the same optional LLM-judge fallback as training.
        async with _judge_semaphore(args):
            ok = await _grade_one_search(sample, args)
    elif _sample_domain(sample) == "webshop":
        # Webshop eval: same sidecar-stamped episode_won flag as training (no judge call).
        ok = _grade_one_webshop(sample, args)
    elif _sample_domain(sample) == "alfworld":
        ok = _grade_one_alfworld(sample, args)
    elif _eval_judge_on(args) and (sample.response or "").strip():
        # Honor the shared concurrency cap (eval fans out one coroutine per sample).
        async with _judge_semaphore(args):
            ok = await _llm_judge_correct(args, sample)
    elif getattr(args, "sdpo_grader", "mcq") == "dapo":
        # Math eval: same grader as training (_is_correct's dapo path extracts the
        # <answer> tag, \boxed{}-wraps it, and runs DAPO's scorer).
        ok = _is_correct(sample, args)
    else:
        ok = _is_correct(sample, args)
    return 1.0 if ok else 0.0
