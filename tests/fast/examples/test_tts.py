"""CPU-only tests for examples/TTS (test-time scaffolding).

No GPU, no Docker, no network: engines are mocked. Covers:
  1. engine config parsing + role dedup + backend selection (build_engines);
  2. OpenAI/Anthropic request-shaping + response-parsing via a stub post_json;
  3. optimizer-output parsing contract (DIAGNOSIS / ===CORRECT=== / ===PITFALL===);
  4. the full SDPO-style group loop with mock engines: N rollouts -> grade ->
     distill correct-knowledge + pitfalls -> re-solve -> optimize prompt pair.
"""

import asyncio

import pytest
from tests.ci.ci_register import register_cpu_ci

from examples.TTS.engines import (
    AnthropicChatBackend,
    ChatResult,
    EngineClient,
    EngineConfig,
    OpenAIChatBackend,
    build_engines,
)
from examples.TTS.grading import grade, make_grader_args
from examples.TTS.optimizer import PromptOptimizer
from examples.TTS.prompts import (
    SkillPrompts,
    combine_skill,
    parse_optimizer_output,
    skill_augmented_user,
)
from examples.TTS.scaffold import Scaffold

register_cpu_ci(est_time=20, suite="stage-b-cpu", labels=[])


# --------------------------------------------------------------------------- #
# 1. engine config + factory
# --------------------------------------------------------------------------- #
def test_engine_config_parsing_and_extra_body():
    c = EngineConfig.from_dict(
        {"backend": "openai", "model": "gpt-5.6-luna", "base_url": "https://x/v1",
         "api_key_env": "OPENAI_API_KEY", "reasoning_effort": "high"}
    )
    assert c.extra_body == {"reasoning_effort": "high"}
    assert c.api_key_env == "OPENAI_API_KEY"


def test_build_engines_dedup_and_backends():
    client = EngineClient()
    roles = {
        "solver": {"backend": "openai", "model": "qwen", "base_url": "http://127.0.0.1:30000/v1"},
        "skill_writer": {"backend": "openai", "model": "qwen", "base_url": "http://127.0.0.1:30000/v1"},
        "optimizer": {"backend": "anthropic", "model": "claude-opus-4-8",
                      "base_url": "https://api.anthropic.com/v1", "api_key_env": "ANTHROPIC_API_KEY"},
    }
    eng = build_engines(roles, client)
    assert eng["solver"] is eng["skill_writer"]
    assert eng["optimizer"] is not eng["solver"]
    assert isinstance(eng["solver"], OpenAIChatBackend)
    assert isinstance(eng["optimizer"], AnthropicChatBackend)


def test_reasoning_model_detection():
    client = EngineClient()
    for model, expect in [("gpt-5.6-luna", True), ("o3-mini", True), ("qwen2.5-7b", False)]:
        cfg = EngineConfig(backend="openai", model=model, base_url="x")
        assert OpenAIChatBackend(cfg, client)._is_reasoning_model() is expect


# --------------------------------------------------------------------------- #
# 2. request shaping / response parsing without sockets
# --------------------------------------------------------------------------- #
class _StubClient:
    def __init__(self, response):
        self._response = response
        self.calls = []

    async def post_json(self, url, payload, headers, max_retries):
        self.calls.append((url, payload, headers))
        return self._response


def test_openai_backend_shapes_request_and_parses():
    stub = _StubClient({"choices": [{"message": {"content": " hi "}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 3}})
    cfg = EngineConfig(backend="openai", model="qwen", base_url="http://x/v1", api_key_env="")
    res = asyncio.run(OpenAIChatBackend(cfg, stub).chat([{"role": "user", "content": "q"}],
                                                        temperature=0.5, max_tokens=64))
    url, payload, headers = stub.calls[-1]
    assert url.endswith("/chat/completions")
    assert payload["max_tokens"] == 64 and payload["temperature"] == 0.5
    assert "Authorization" not in headers
    assert res.text == "hi" and res.prompt_tokens == 3


def test_openai_reasoning_model_omits_temperature():
    stub = _StubClient({"choices": [{"message": {"content": "x"}}]})
    cfg = EngineConfig(backend="openai", model="gpt-5.6-luna", base_url="http://x/v1")
    asyncio.run(OpenAIChatBackend(cfg, stub).chat([{"role": "user", "content": "q"}], max_tokens=100))
    _, payload, _ = stub.calls[-1]
    assert payload["max_completion_tokens"] == 100
    assert "temperature" not in payload and "max_tokens" not in payload


def test_anthropic_backend_hoists_system_and_parses_blocks():
    stub = _StubClient({"content": [{"type": "text", "text": "ans"}], "stop_reason": "end_turn",
                        "usage": {"output_tokens": 5}})
    cfg = EngineConfig(backend="anthropic", model="claude-opus-4-8", base_url="http://x/v1", api_key_env="")
    res = asyncio.run(AnthropicChatBackend(cfg, stub).chat(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}], max_tokens=32))
    url, payload, headers = stub.calls[-1]
    assert url.endswith("/messages")
    assert payload["system"] == "sys"
    assert payload["messages"] == [{"role": "user", "content": "u"}]
    assert headers["anthropic-version"]
    assert res.text == "ans" and res.completion_tokens == 5


# --------------------------------------------------------------------------- #
# 3. optimizer-output parsing (prompt PAIR) + skill combine/splice
# --------------------------------------------------------------------------- #
def test_parse_optimizer_output_pair():
    seed = SkillPrompts(correct="OLD_CORRECT_PROMPT_LONG_ENOUGH", pitfall="OLD_PITFALL_PROMPT_LONG_ENOUGH")
    out = (
        "DIAGNOSIS: skills too vague and pitfalls not actionable.\n"
        "===CORRECT===\n"
        "A new correct-knowledge prompt that is clearly long enough to be accepted.\n"
        "===PITFALL===\n"
        "A new pitfall prompt that is also clearly long enough to be accepted here."
    )
    diag, new = parse_optimizer_output(out, fallback=seed)
    assert diag.startswith("skills too vague")
    assert new.correct.startswith("A new correct-knowledge")
    assert new.pitfall.startswith("A new pitfall prompt")


def test_parse_optimizer_output_missing_section_keeps_fallback():
    seed = SkillPrompts(correct="OLD_CORRECT_PROMPT_LONG_ENOUGH", pitfall="OLD_PITFALL_PROMPT_LONG_ENOUGH")
    out = "DIAGNOSIS: only fixed correct.\n===CORRECT===\nA sufficiently long new correct prompt body here."
    diag, new = parse_optimizer_output(out, fallback=seed)
    assert new.correct.startswith("A sufficiently long")
    assert new.pitfall == "OLD_PITFALL_PROMPT_LONG_ENOUGH"  # missing section -> fallback


def test_combine_and_splice():
    skill = combine_skill("[Knowledge/Rule]\nadd", "[Error]\noff-by-one")
    assert "KNOWLEDGE" in skill and "PITFALLS" in skill
    u = skill_augmented_user("What is 2+2?", skill)
    assert "Now solve this problem" in u and "What is 2+2?" in u
    assert skill_augmented_user("bare", "") == "bare"


# --------------------------------------------------------------------------- #
# 4. full SDPO-style group loop with mock engines
# --------------------------------------------------------------------------- #
class _ScriptedSolver:
    """Solver that returns a fixed correct answer for some problems and a wrong
    one for others, so groups have both correct and failed traces."""

    def __init__(self, answer_by_problem):
        self._map = answer_by_problem
        self.name = "solver"

    async def chat(self, messages, **kw):
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        # crude: find which problem this is by substring
        for prob, ans in self._map.items():
            if prob in user:
                return ChatResult(text=f"work... <answer>{ans}</answer>")
        return ChatResult(text="<answer>0</answer>")


class _MockWriter:
    def __init__(self, text):
        self._text = text
        self.name = "writer"

    async def complete(self, system, user, **kw):
        return self._text


def test_full_group_loop_with_mocks():
    # problem "2+2?" -> solver says 4 (correct); "3+3?" -> solver says 5 (wrong)
    solver = _ScriptedSolver({"2+2?": "4", "3+3?": "5"})
    correct_writer = _MockWriter("[Knowledge/Rule]\nsum\n[Details/Examples]\n1+1=2")
    pitfall_writer = _MockWriter("[Error]\nmiscount\n[Rule]\ncount carefully\n[Example]\n2+2=4")
    optimizer_engine = _MockWriter(
        "DIAGNOSIS: ok.\n===CORRECT===\nRevised correct prompt that is long enough to be accepted here.\n"
        "===PITFALL===\nRevised pitfall prompt that is long enough to be accepted here too."
    )

    scaffold = Scaffold(correct_writer, pitfall_writer, solver,
                        "solver system <answer></answer>", n_rollouts=4)
    grade_fn = (lambda resp, label, domain: grade(resp, label, make_grader_args("dapo")))
    items = [{"problem": "2+2?", "label": "4", "domain": "math"},
             {"problem": "3+3?", "label": "6", "domain": "math"}]

    bases = asyncio.run(scaffold.run_baseline_batch(items, grade_fn, concurrency=4))
    assert bases[0].baseline_acc == 1.0  # all 4 rollouts of 2+2 correct
    assert bases[1].baseline_acc == 0.0  # all rollouts of 3+3 wrong

    results = asyncio.run(scaffold.run_with_skill_batch(bases, SkillPrompts.seed(), grade_fn, concurrency=4))
    # correct group distilled a [Knowledge] skill; failed group distilled pitfalls
    assert "KNOWLEDGE" in results[0].skill  # 2+2 had correct traces
    assert "PITFALLS" in results[1].skill   # 3+3 had failed traces

    step = asyncio.run(PromptOptimizer(optimizer_engine).step(SkillPrompts.seed(), results))
    assert step.changed is True
    assert "Revised correct" in step.new_prompts.correct
    assert "Revised pitfall" in step.new_prompts.pitfall
    assert abs(step.mean_baseline_acc - 0.5) < 1e-9  # (1.0 + 0.0) / 2
