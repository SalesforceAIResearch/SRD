"""
Simple multi-turn generation with tool calling.
"""

import argparse
import contextvars
import uuid
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    update_sample_from_response,
)
from miles.rollout.generate_utils.tool_call_utils import (
    create_tool_call_parser,
    execute_tool_calls,
    update_sample_with_tool_responses,
)
from miles.utils.http_utils import post
from miles.utils.misc import load_function

# Per-trajectory identifier for STATEFUL tools (e.g. examples/SRD's
# search/open/find, where open/find refer to "the page most recently shown" --
# see that module's tools/search/client.py for the consumer side). Unrelated
# to Sample.session_id (only populated under --sglang-router-policy
# consistent_hashing, for a different purpose -- routing, not tool state).
#
# Set once per `generate()` call, BEFORE the turn loop's `execute_tool_calls`
# (which runs each tool call in its own Task via asyncio.gather). A
# contextvars.ContextVar's value is snapshotted into a Task at CREATION time
# and never written back to the parent -- so the set() below must happen in
# THIS coroutine's own frame, not inside a gather-spawned child, or every
# subsequent turn's tool calls would see an unset var and get a fresh id each
# time, defeating the whole point of a per-trajectory identifier. Stateless
# tools (code_interpreter, cli_exec) never read this and are unaffected.
_TRAJECTORY_SESSION_ID: "contextvars.ContextVar[str | None]" = contextvars.ContextVar(
    "multi_turn_trajectory_session_id", default=None
)


def current_trajectory_session_id() -> str:
    """The current trajectory's session id, for tools that need to correlate
    calls within one multi-turn rollout (see the ContextVar's docstring
    above). Raises if called outside a `generate()` invocation -- there is no
    sensible fallback id to hand out."""
    session_id = _TRAJECTORY_SESSION_ID.get()
    if session_id is None:
        raise RuntimeError(
            "current_trajectory_session_id() called outside multi_turn.generate()'s turn loop"
        )
    return session_id


# Per-trajectory METADATA channel for stateful tools that need a pre-assigned
# episode/task id (e.g. examples/SRD's webshop_step/alfworld_step,
# which must reset into the SAME webshop task / alfworld game file on every
# train or eval pass for a given row, so success rate is comparable across
# arms and checkpoints -- see those tools' client.py for the consumer side).
# This is the SAME Sample.metadata dict object (not a copy), set once per
# trajectory -- so a tool client reading e.g. metadata["alfworld_game_file"]
# on the way IN, and a tool client writing metadata["episode_won"] on the way
# OUT (after a terminal step), both take effect on the real sample.metadata
# with zero extra plumbing back to the caller. Never rendered into the
# prompt: compute_prompt_ids_from_sample only ever reads sample.prompt.
#
# Same "set in THIS coroutine's own frame, before any execute_tool_calls()
# spawns child Tasks" requirement as _TRAJECTORY_SESSION_ID above (a
# contextvars.ContextVar snapshots at Task-creation time).
_TRAJECTORY_METADATA: "contextvars.ContextVar[dict | None]" = contextvars.ContextVar(
    "multi_turn_trajectory_metadata", default=None
)


def current_trajectory_metadata() -> dict:
    """The current trajectory's Sample.metadata dict (see the ContextVar's
    docstring above). Raises if called outside a `generate()` invocation."""
    metadata = _TRAJECTORY_METADATA.get()
    if metadata is None:
        raise RuntimeError(
            "current_trajectory_metadata() called outside multi_turn.generate()'s turn loop"
        )
    return metadata


def _split_reasoning_content(text: str) -> tuple[str, str]:
    """Split raw generated text into ``(reasoning_content, content)``.

    Qwen3/Qwen3.5's chat template bakes the OPENING ``<think>\\n`` into the
    generation prefix (part of the prompt, not generated text), so a thinking
    turn's raw output only carries the CLOSING ``</think>``, e.g.
    ``"...reasoning...\\n</think>\\n\\n<answer>...</answer>"``. Without this
    split, that whole blob lands in ``content`` verbatim -- corrupting every
    downstream consumer that expects `content` to be the model's VISIBLE
    output (agentic_traces dumps, SDPO teacher-prefix reframing, answer-tag
    parsing). Strips a stray leading ``<think>`` too, for robustness.
    """
    if "</think>" in text:
        reasoning, _, content = text.partition("</think>")
        return reasoning.removeprefix("<think>").strip(), content.strip()
    return "", text


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    # ----------------------- Setup -------------------------

    args = input.args
    sample = deepcopy(input.sample)
    tokenizer = input.state.tokenizer
    assert not args.partial_rollout, "Partial rollout is not supported"

    # Snapshot NOW, before any `await` below -- input.state is a shared
    # singleton (GenerateState) across every concurrently-running generate()
    # coroutine for this rollout step. Reading state.rollout_id AFTER the
    # multi-turn tool-call loop (as this used to do) races the next training
    # step: a slow trajectory (many tool round trips) can still be running
    # when generate_rollout_async advances state.rollout_id for the NEXT
    # step, so this trajectory's read sees the wrong step's id -- confirmed
    # live via a real ablation run's agentic_traces/ dump, where an
    # unknown.jsonl (2002 rows, bigger than any single per-step file) held
    # alfworld/webshop traces that should have landed in 0..9.jsonl. Capturing
    # it here, before the first `await post(...)`, is race-free: this
    # coroutine's own frame is untouched by any other coroutine.
    rollout_id = getattr(input.state, "rollout_id", None)

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    execute_tool_function = load_function(args.generate_execute_tool_function_path)

    tool_specs = load_function(args.generate_tool_specs_path)
    tool_call_parser = create_tool_call_parser(tool_specs, args.generate_tool_call_parser)

    # Chat-template kwargs (e.g. {"enable_thinking": False}) -- forwarded to the
    # per-turn tool-response tokenization so EVERY assistant turn's generation
    # prompt matches the initial one (else no-thinking is only enforced turn 1;
    # see tokenize_tool_responses). Already a dict (arg is type=json.loads).
    chat_template_kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}

    multi_samples = []

    # Fresh per-trajectory id for stateful tools (see _TRAJECTORY_SESSION_ID's
    # docstring above) -- set HERE, in this coroutine's own frame, before any
    # execute_tool_calls() call below spawns child Tasks that need to inherit
    # it.
    _TRAJECTORY_SESSION_ID.set(str(uuid.uuid4()))
    _TRAJECTORY_METADATA.set(sample.metadata if isinstance(sample.metadata, dict) else {})

    # ----------------------- Initial prompts -------------------------

    prompt_tokens_ids = compute_prompt_ids_from_sample(input.state, sample, tools=tool_specs)

    sample.tokens = prompt_tokens_ids.copy()

    # Per-sample override so a single job can run more turns at eval than at
    # train (e.g. --eval-config's metadata_overrides: {generate_max_turns: 20}
    # while --generate-max-turns 5 governs training) without a second global
    # arg. Same pattern as Sample.generate_function_path's per-sample override
    # of --custom-generate-function-path.
    if isinstance(sample.metadata, dict):
        max_turns = sample.metadata.get("generate_max_turns", args.generate_max_turns)
    else:
        max_turns = args.generate_max_turns

    rounds_used = 0
    # Turn-by-turn record of (tool call, tool observation) pairs, accumulated
    # live as each round executes. Two consumers, both generic (nothing here is
    # SDPO-specific):
    #   - tool_call_count: how many tool calls the whole trajectory made.
    #   - tool_trace: [{"tool_call": "<name>({args})", "observation": <result>}]
    #     -- the schema examples/SRD/sdpo.py::_render_env_feedback expects for
    #     --sdpo-skill-source env_feedback (grounding a failed trace's pitfall
    #     skill in what its tools ACTUALLY returned). Recorded here (rather than
    #     re-derived by regex over the decoded text) because this loop is the
    #     one place the real call/observation structure is known for certain.
    tool_trace: list[dict[str, str]] = []
    # Live message-dict record of the conversation, built as it happens: each
    # assistant turn's text, then the tool observation message(s) it triggered.
    # This is the correct-by-construction trace for post-hoc dumps (SDPO_ReAct's
    # agentic_traces) -- no fragile re-parsing of the decoded ChatML token stream.
    messages: list[dict[str, str]] = []
    for _turn in range(max_turns):
        # ----------------------- Call inference endpoint -------------------------

        payload, halt_status = compute_request_payload(args, sample.tokens, input.sampling_params)
        if payload is None:
            sample.status = halt_status
            if args.generate_multi_samples and multi_samples:
                multi_samples[-1].status = halt_status
            break

        if args.generate_multi_samples:
            sample = deepcopy(input.sample)

        output = await post(url, payload)
        await update_sample_from_response(args, sample, payload=payload, output=output, update_loss_mask=True)
        rounds_used += 1

        if args.generate_multi_samples:
            multi_samples.append(deepcopy(sample))

        if output["meta_info"]["finish_reason"]["type"] in ("abort", "length"):
            # Truncated/aborted assistant turn -- no parseable tool call; record
            # the raw text as the final assistant message and stop.
            reasoning, content = _split_reasoning_content(output.get("text", ""))
            messages.append({"role": "assistant", "reasoning_content": reasoning, "content": content})
            break

        # ----------------------- Execute tools -------------------------

        # normal_text = the assistant's reasoning with the <tool_call> blocks
        # stripped out; tool_calls = the parsed structured calls.
        normal_text, tool_calls = tool_call_parser.parse_non_stream(output["text"])
        if len(tool_calls) == 0:
            # No tool call -> this is the final assistant answer turn.
            reasoning, content = _split_reasoning_content(output.get("text", ""))
            messages.append({"role": "assistant", "reasoning_content": reasoning, "content": content})
            break

        tool_messages = await execute_tool_calls(tool_calls, execute_tool_function)
        # OpenAI-standard assistant message: reasoning in `content`, the calls in a
        # structured `tool_calls` field (id/type/function{name,arguments}). The id
        # links to each tool result's tool_call_id below. This matches the widely
        # used trajectory schema (e.g. i-DeepSearch observation-masking logs), so
        # the dump can be re-sent through any chat template / OpenAI client without
        # re-parsing the raw <tool_call> XML out of content.
        assistant_tool_calls = []
        for call, msg in zip(tool_calls, tool_messages, strict=False):
            tool_trace.append(
                {
                    "tool_call": f"{getattr(call, 'name', 'tool')}({getattr(call, 'parameters', '') or ''})",
                    "observation": msg.get("content", ""),
                }
            )
            assistant_tool_calls.append({
                "id": msg.get("tool_call_id", ""),
                "type": "function",
                "function": {
                    "name": getattr(call, "name", "") or "",
                    "arguments": getattr(call, "parameters", "") or "",
                },
            })
        reasoning, content = _split_reasoning_content(normal_text or "")
        messages.append({
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": content,
            "tool_calls": assistant_tool_calls,
        })
        # Each tool observation is its own message, linked by tool_call_id.
        for msg in tool_messages:
            messages.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", ""),
                "name": msg.get("name", ""),
                "content": msg.get("content", ""),
            })
        update_sample_with_tool_responses(
            sample, tool_messages, tokenizer=tokenizer, chat_template_kwargs=chat_template_kwargs
        )

    # Generic multi-turn diagnostic: how many inference rounds this trajectory
    # took. Feeds --log-multi-turn's existing multi_turn_metric/round_number_*
    # panel (miles/backends/training_utils/log_utils.py::log_multi_turn_data),
    # which already reads rollout_data["round_number"] but had no producer for
    # this generate function -- any --custom-generate-function-path pointed at
    # miles.rollout.generate_hub.multi_turn.generate now populates it for free.
    if args.generate_multi_samples:
        for i, s in enumerate(multi_samples):
            if isinstance(s.metadata, dict):
                s.metadata["round_number"] = i + 1
    elif isinstance(sample.metadata, dict):
        sample.metadata["round_number"] = rounds_used
        # Generic tool bookkeeping for any downstream RM / metric (e.g.
        # examples/SRD env_feedback skill, the agentic/* wandb panel). Only
        # populated on the single-sample path -- generate_multi_samples splits
        # one trajectory across many samples, so a single accumulated trace
        # can't be attributed to one of them unambiguously.
        sample.metadata["tool_call_count"] = len(tool_trace)
        sample.metadata["tool_trace"] = tool_trace
        # Live conversation record (assistant + tool turns, in order) for
        # correct-by-construction post-hoc dumps (SDPO_ReAct agentic_traces).
        sample.metadata["messages"] = messages
        # Current train step (snapshotted at the TOP of this function, before
        # this trajectory's first await -- see that comment for why reading
        # input.state.rollout_id here instead would race concurrent
        # trajectories), so downstream per-step dumps (e.g. SDPO_ReAct
        # agentic_traces/{rollout_id}.jsonl) split by training step instead
        # of one huge file. None during eval (state.rollout_id stays unset)
        # -> dump falls back to its own naming. Mirrors the legacy
        # generate_with_tools.generate.
        if rollout_id is not None:
            sample.metadata["rollout_id"] = rollout_id

    return GenerateFnOutput(samples=multi_samples if args.generate_multi_samples else sample)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-max-turns", type=int, default=16)
    parser.add_argument("--generate-tool-specs-path", type=str)
    parser.add_argument("--generate-tool-call-parser", type=str)
    parser.add_argument("--generate-execute-tool-function-path", type=str)
    parser.add_argument("--generate-multi-samples", action="store_true")


generate.add_arguments = _add_arguments
