"""Correctness / grading subsystem for SDPO, split out of ``sdpo.py`` for size.

Answers ONE question -- "is this trace correct?" -- across every task domain
SDPO trains on. ``_grade_group`` is the entry point ``sdpo.py``'s ``sdpo_group_reward`` and
``sdpo_eval_reward`` call into; everything else here is a helper for it. This
module has NO dependency on the teacher-prefix / skill-generation machinery
in ``sdpo.py`` (and vice versa is one-directional: ``sdpo.py`` imports FROM
here, never the other way), so it stays independently testable and does not
create an import cycle.
"""

import asyncio
import json
import logging
import os
import re
from argparse import Namespace

from miles.rollout.rm_hub.math_dapo_utils import compute_score as _dapo_compute_score
from miles.rollout.rm_hub.math_utils import extract_answer as extract_boxed_answer
from miles.rollout.rm_hub.math_utils import grade_answer_verl
from miles.utils.http_utils import post  # miles' shared HTTP client: retries + shared pool
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# deterministic grading (MCQ / DAPO-boxed / open-ended math)
# --------------------------------------------------------------------------- #


def _extract_tagged_answer(text: str, tag: str = "answer") -> str | None:
    """Extract the content of the LAST <tag>...</tag> block, or None if absent.

    Open-ended SDPO asks the model to wrap its final answer in <answer>...</answer>
    (see examples/SRD/build_sci_dataset.py). We take the last occurrence so a
    model that reasons and revises still yields its final answer. Returns None
    (not "") when the tag is missing so callers can treat "no answer" as wrong.
    """
    if not text:
        return None
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    # Tolerate an unclosed final tag ("<answer> foo" with no </answer>).
    m = re.search(rf"<{tag}>(.*)$", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_answer(args: Namespace, sample: Sample) -> str | None:
    """The model's final answer: prefer the <answer> tag, fall back to \\boxed{}."""
    tag = getattr(args, "sdpo_answer_tag", "answer")
    tagged = _extract_tagged_answer(sample.response, tag)
    if tagged is not None:
        return tagged
    return extract_boxed_answer(sample.response)


def _is_correct(sample: Sample, args: Namespace | None = None) -> bool:
    """Grade a trace against its label with exact/heuristic matching (no LLM).

    The SciKnowEval dataset (see build_sci_dataset.py) is multiple choice: the
    label is the answer LETTER and the model outputs the letter inside
    <answer>...</answer>. We do a case-insensitive letter match on the extracted
    answer. Non-letter labels (if any) fall back to math-style grading.
    """
    # DAPO math dataset: integer answers in \boxed{}. Use DAPO's own grader with
    # strict_box_verify=True (the default minerva path expects an "Answer: X" line;
    # the strict-box path extracts the \boxed{} answer, which is what the models emit).
    if args is not None and getattr(args, "sdpo_grader", "mcq") == "dapo":
        label = (sample.label or "").strip()
        if not label:
            return False
        # is_correct_strict_box (see math_dapo_utils.py) scans the response's
        # last 100 chars for the LAST \boxed{...} occurrence -- correct for
        # examples that teach the model to write \boxed{} directly (e.g. raw
        # DAPO data), but wrong for examples like SDPO_ReAct whose prompt
        # contracts <answer>...</answer> as the ONLY final-answer marker and
        # never mentions \boxed{} at all. For those, a stray \boxed{} left
        # over from intermediate reasoning prose (or one omitted entirely
        # from inside a `<answer>N</answer>` with no \boxed{} wrapper) makes
        # grading depend on incidental formatting instead of the tag the
        # model was actually told to use -- observed as "sometimes \boxed{}
        # counts, sometimes the <answer> tag counts" inconsistency. When the
        # tag is present, grade ONLY its content (wrapped in \boxed{} so it
        # still flows through DAPO's own normalization/matching logic) --
        # this is a no-op for prompts that never use the tag (extracted is
        # None -> falls through to the raw-response path unchanged).
        tag = getattr(args, "sdpo_answer_tag", "answer")
        extracted = _extract_tagged_answer(sample.response, tag)
        graded_text = f"\\boxed{{{extracted}}}" if extracted is not None else (sample.response or "")
        try:
            if bool(_dapo_compute_score(graded_text, label, strict_box_verify=True)["acc"]):
                return True
        except Exception:
            pass
        # DAPO's strict_box_verify is an EXACT string match on the boxed content
        # (see math_dapo_utils.is_correct_strict_box) -- fine for AIME's plain
        # integers, but on Minerva Math's LaTeX answers it misses anything with a
        # cosmetically different but mathematically identical form ('\frac{d
        # x}{d t}=k x-a' vs '\frac{dx}{dt} = kx - a', '1+\sqrt{3} i' vs '1 +
        # \sqrt{3}i'). Confirmed on a live eval dump: strict-box-only scored
        # Minerva at 15.6% while grade_answer_verl's mathd/sympy-normalized
        # fallback (same grader run-qwen3-8B-sdpo.sh's mcq path already uses)
        # brings the SAME rollouts to 38.6%, and the combined AIME+Minerva pass@1
        # from 28.2%->45.0% -- matching the ~45% expected for this checkpoint.
        return bool(grade_answer_verl(graded_text, label))

    tag = getattr(args, "sdpo_answer_tag", "answer") if args is not None else "answer"
    extracted = _extract_tagged_answer(sample.response, tag)
    if extracted is None:
        extracted = extract_boxed_answer(sample.response)
    label = (sample.label or "").strip()
    if not label:
        return False

    # Legacy single-letter labels (old MCQ dataset): case-insensitive letter match.
    if len(label) == 1 and label.isalpha():
        pred = (extracted or "").strip()
        if len(pred) > 1:  # e.g. "B." or "(B)" -> keep first alpha char
            pred = next((c for c in pred if c.isalpha()), pred)
        return pred.upper() == label.upper()

    # Open-ended: exact (normalized) match or math-style grading.
    pred = (extracted or "").strip()
    if pred and pred.lower() == label.lower():
        return True
    return bool(grade_answer_verl(extracted or sample.response, label))


# --------------------------------------------------------------------------- #
# LLM-as-judge grading  (defeats the MCQ letter-guessing reward hack)
# --------------------------------------------------------------------------- #

from examples.SRD.prompt.judge import (
    _JUDGE_SYSTEM,
    _SEARCH_JUDGE_SYSTEM,
)


def _build_judge_prompt(args: Namespace, sample: Sample) -> tuple[str, str]:
    """Return (system, user) messages for the judge. Includes BOTH the full response
    and the extracted answer, per the requirement to give the judge both."""
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    question = meta.get("question") or (sample.prompt if isinstance(sample.prompt, str) else "")
    reference = (sample.label or "").strip()
    extracted = _extract_answer(args, sample)
    full = sample.response or ""
    # Cap the full response so the judge prompt stays bounded on very long traces.
    if len(full) > 8000:
        full = full[:4000] + "\n...[truncated]...\n" + full[-3000:]
    user = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"MODEL FULL RESPONSE:\n{full}\n\n"
        f"MODEL EXTRACTED ANSWER:\n{extracted if extracted is not None else '(none — no <answer> tag found)'}\n\n"
        "Is the model's answer correct? Reply CORRECT or INCORRECT."
    )
    return _JUDGE_SYSTEM, user


# GLOBAL judge concurrency limiter. Each rollout spawns one generate_and_rm_group
# task PER GROUP (rollout_batch_size groups) and they all run concurrently, so a
# per-call semaphore would only bound the ~n_samples_per_prompt traces within one
# group — the real cap must be process-wide. This single semaphore is shared by
# every group's judge calls, so --sdpo-judge-max-concurrency bounds TOTAL in-flight
# judge requests to the gateway (else 32 groups x 8 traces = 256 at once).
_JUDGE_SEM: "asyncio.Semaphore | None" = None
_JUDGE_SEM_LIMIT: int | None = None


def _judge_semaphore(args: Namespace) -> asyncio.Semaphore:
    global _JUDGE_SEM, _JUDGE_SEM_LIMIT
    limit = int(getattr(args, "sdpo_judge_max_concurrency", 32))
    # Recreate if the limit changed (or first use). Safe: single event loop.
    if _JUDGE_SEM is None or _JUDGE_SEM_LIMIT != limit:
        _JUDGE_SEM = asyncio.Semaphore(limit)
        _JUDGE_SEM_LIMIT = limit
    return _JUDGE_SEM


# --------------------------------------------------------------------------- #
# judge transport: OpenAI-compatible HTTP (default) — see _judge_call_openai;
# AWS Bedrock Converse is the opt-in alternative — see _judge_call_bedrock
# --------------------------------------------------------------------------- #

# Bedrock rejects maxTokens below 16 for some models (gpt-5.6-luna among them).
_BEDROCK_MIN_MAX_TOKENS = 16
# Many Bedrock models (all openai.* included) are not invocable on-demand by their
# raw modelId and require a cross-region inference profile; the profile id is just
# the modelId with a region-scope prefix. Try the raw id, then these.
_BEDROCK_PROFILE_PREFIXES = ("us.", "global.")

# Gateway-style judge names -> Bedrock modelIds. Every existing run script passes
# the gateway model name (--sdpo-judge-model gpt-5.6-luna), which is NOT a Bedrock id
# (Bedrock namespaces by provider: openai.gpt-5.6-luna). Without this map, flipping
# the default backend to bedrock would send those runs to a nonexistent model and
# silently degrade every judged sample to the deterministic fallback.
_BEDROCK_MODEL_ALIASES = {
    "gpt-5.6-luna": "openai.gpt-5.6-luna",
    "gpt-5.4-mini": "openai.gpt-5.4-mini",
}

# One boto3 client per (region) per process. botocore clients are thread-safe for
# issuing calls, and creating one costs a session + endpoint resolution, so we must
# NOT build one per judge request (thousands per rollout).
_BEDROCK_CLIENTS: dict[str, object] = {}
# modelId -> the id that actually worked, so the on-demand/inference-profile probe
# runs once per process instead of on every call.
_BEDROCK_RESOLVED_MODEL: dict[str, str] = {}


# A reasoning judge that hits maxTokens returns reasoning only and no verdict; the
# retry gets this much budget instead. 4x the default 2048 is far above the 7-327
# output tokens a real judge prompt actually uses, so the retry is rare and cheap.
_BEDROCK_RETRY_MAX_TOKENS = 8192


def _bedrock_reply_text(resp: dict) -> str | None:
    """The assistant's visible text from a Converse response, or None if there is none.

    MUST NOT be content[0]["text"]. gpt-5.6-luna is a reasoning model, and on a
    real judge prompt it returns TWO content blocks with the reasoning FIRST:
    [{'reasoningContent': ...}, {'text': 'CORRECT'}]. Indexing block 0 raises
    KeyError('text') -- observed as 491 consecutive
    "LLM judge failed (KeyError('text'))" lines in a live run, i.e. EVERY judged
    sample silently degrading to deterministic grading. Short prompts happen to
    answer with a single text block, which is why a smoke test misses this.

    Returns None when the model spent the whole budget reasoning and never emitted
    a verdict (stopReason == 'max_tokens'). The reasoning is NOT a usable
    substitute: luna returns it as reasoningContent.redactedContent, encrypted
    bytes. The caller retries with a bigger budget.
    """
    blocks = resp.get("output", {}).get("message", {}).get("content", []) or []
    text = "\n".join(b["text"] for b in blocks if isinstance(b, dict) and "text" in b).strip()
    if text:
        return text
    # Plaintext reasoning (some models/settings) does carry the verdict -- use it.
    reasoning = [
        ((b.get("reasoningContent") or {}).get("reasoningText") or {}).get("text", "")
        for b in blocks
        if isinstance(b, dict) and "reasoningContent" in b
    ]
    joined = "\n".join(r for r in reasoning if r).strip()
    return joined or None


def _bedrock_client(region: str):
    client = _BEDROCK_CLIENTS.get(region)
    if client is None:
        import boto3  # imported lazily: only the bedrock backend needs it

        client = boto3.client("bedrock-runtime", region_name=region)
        _BEDROCK_CLIENTS[region] = client
    return client


def _bedrock_converse_sync(region: str, model: str, system: str, user: str, max_tokens: int) -> str:
    """Blocking Bedrock Converse call returning the reply text.

    Runs in a worker thread (see _judge_call_bedrock) because boto3 is sync and
    would otherwise stall the rollout event loop for the whole judge latency.
    """
    from botocore.exceptions import ClientError

    rt = _bedrock_client(region)
    kwargs = {
        "messages": [{"role": "user", "content": [{"text": user}]}],
        "system": [{"text": system}],
        "inferenceConfig": {"maxTokens": max(max_tokens, _BEDROCK_MIN_MAX_TOKENS)},
    }

    candidates = [_BEDROCK_RESOLVED_MODEL.get(model, model)]
    for prefix in _BEDROCK_PROFILE_PREFIXES:
        cand = prefix + model
        if not model.startswith(_BEDROCK_PROFILE_PREFIXES) and cand not in candidates:
            candidates.append(cand)

    last_err: Exception | None = None
    for candidate in candidates:
        try:
            resp = rt.converse(modelId=candidate, **kwargs)
        except ClientError as e:
            last_err = e
            msg = e.response.get("Error", {}).get("Message", "")
            # Only the "needs an inference profile" error is worth retrying under a
            # different id; anything else (throttling, auth, bad request) is real.
            if "on-demand throughput" in msg or "inference profile" in msg:
                continue
            raise
        _BEDROCK_RESOLVED_MODEL[model] = candidate
        text = _bedrock_reply_text(resp)
        if text is not None:
            return text
        # Budget went entirely to (encrypted) reasoning, so there is no verdict to
        # parse. Retry once with more room rather than failing into the deterministic
        # fallback. Rare: 2 of ~1280 calls in a live eval.
        logger.warning(
            "Bedrock judge returned reasoning but no verdict (stopReason=%s, maxTokens=%d); "
            "retrying with maxTokens=%d.",
            resp.get("stopReason"),
            kwargs["inferenceConfig"]["maxTokens"],
            _BEDROCK_RETRY_MAX_TOKENS,
        )
        retry_kwargs = {**kwargs, "inferenceConfig": {"maxTokens": _BEDROCK_RETRY_MAX_TOKENS}}
        text = _bedrock_reply_text(rt.converse(modelId=candidate, **retry_kwargs))
        if text is not None:
            return text
        raise ValueError(
            f"Bedrock Converse returned no verdict even at maxTokens={_BEDROCK_RETRY_MAX_TOKENS}"
        )
    raise last_err  # type: ignore[misc]


async def _judge_call_bedrock(args: Namespace, system: str, user: str) -> str:
    model = getattr(args, "sdpo_judge_model", "us.openai.gpt-5.6-luna")
    model = _BEDROCK_MODEL_ALIASES.get(model, model)
    region = getattr(args, "sdpo_judge_region", "us-west-2")
    max_tokens = int(getattr(args, "sdpo_judge_max_tokens", 2048))
    return await asyncio.to_thread(_bedrock_converse_sync, region, model, system, user, max_tokens)


async def _judge_call_openai(args: Namespace, system: str, user: str) -> str:
    """Legacy OpenAI-compatible HTTP judge (custom gateway or api.openai.com)."""
    base_url = getattr(args, "sdpo_judge_base_url", "https://api.openai.com/v1").rstrip("/")
    model = getattr(args, "sdpo_judge_model", "gpt-5.4-mini")
    api_key = os.environ.get(getattr(args, "sdpo_judge_api_key_env", "OPENAI_API_KEY"), "") or "EMPTY"
    max_tokens = int(getattr(args, "sdpo_judge_max_tokens", 2048))

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": max_tokens,
    }
    # gpt-5*/o-series are reasoning models: they reject an explicit temperature.
    if not model.startswith(("gpt-5", "o1", "o3", "o4")):
        payload["temperature"] = 0.0
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    # Low retry count: the judge is best-effort, fall back fast on failure.
    out = await post(f"{base_url}/chat/completions", payload, max_retries=3, headers=headers)
    return out["choices"][0]["message"].get("content") or ""


def _parse_judge_verdict(text: str) -> bool:
    """Parse the judge's reply into a bool. Looks for the last CORRECT/INCORRECT."""
    if not text:
        return False
    up = text.upper()
    # INCORRECT contains CORRECT, so search for whole-word tokens and take the last.
    hits = re.findall(r"\b(INCORRECT|CORRECT)\b", up)
    if not hits:
        return False
    return hits[-1] == "CORRECT"


async def _llm_judge_correct(
    args: Namespace,
    sample: Sample,
    *,
    system: str | None = None,
    user: str | None = None,
    fallback: bool | None = None,
) -> bool:
    """Grade one trace via the LLM judge.

    Transport is picked by --sdpo-judge-backend: 'openai' (default) uses the
    OpenAI-compatible HTTP gateway (--sdpo-judge-base-url/-api-key-env),
    'bedrock' goes straight to AWS Bedrock Converse via boto3 on the instance
    IAM role.

    system/user default to the science-exam prompt (_build_judge_prompt); pass
    both to use a different prompt (e.g. _build_search_judge_prompt) against
    the SAME backend/model/concurrency config.

    On any judge failure (bad response, timeout, etc.), returns `fallback` if
    given, else deterministic _is_correct(sample, args) -- so a flaky gateway
    never stalls or crashes training. The _is_correct default is math/mcq-
    shaped: callers grading a different domain (e.g. search's EM-miss fallback
    path) MUST pass an explicit `fallback` (typically False, i.e. "keep
    whatever cheaper deterministic check already ran"), since _is_correct
    would otherwise silently mis-grade non-math samples on judge failure.
    """
    if system is None or user is None:
        system, user = _build_judge_prompt(args, sample)
    backend = getattr(args, "sdpo_judge_backend", "openai")

    try:
        if backend == "bedrock":
            content = await _judge_call_bedrock(args, system, user)
        else:
            content = await _judge_call_openai(args, system, user)
        return _parse_judge_verdict(content)
    except Exception as e:
        if fallback is not None:
            logger.warning(f"LLM judge failed ({e!r}); using caller-supplied fallback={fallback}.")
            return fallback
        logger.warning(f"LLM judge failed ({e!r}); falling back to deterministic grading.")
        return _is_correct(sample, args)


# --------------------------------------------------------------------------- #
# domain-aware routing (math / code / search)
# --------------------------------------------------------------------------- #


def _sample_domain(sample: Sample) -> str:
    """The task domain of a sample, from metadata['domain'] (set by the
    per-domain data prep). Defaults to 'math' so existing single-domain math
    runs are unchanged. Used to route grading to the right judge (code samples
    -> test-case execution, math -> dapo/boxed matching)."""
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    return (md.get("domain") or "math").strip().lower()


def _sample_uses_judge(sample: Sample) -> bool:
    """Whether this sample requires LLM-judge grading (e.g. AMO-Bench
    description-type problems that can't be matched deterministically)."""
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    return bool(md.get("amo_use_judge"))


def _code_candidate(sample: Sample, args: Namespace | None = None) -> str:
    """The program to grade for a code sample. TOOL-MANDATORY mode
    (--sdpo-code-require-tool, default on for code): the graded candidate is the
    LAST code the model actually RAN through code_interpreter (from
    metadata['tool_trace']), NOT a text ```python fence. This makes the tool
    100% necessary -- a trace that never runs its solution has no candidate and
    is graded wrong, so the ONLY path to a correct grade is executing the
    solution via the tool. Falls back to the response fence when tool-mandatory
    is off (then code_judge extracts the fence itself)."""
    require_tool = getattr(args, "sdpo_code_require_tool", True) if args is not None else True
    if not require_tool:
        return sample.response or ""
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    trace = md.get("tool_trace") or []
    # tool_trace entries are {"tool_call": "code_interpreter(<params>)", ...}
    # where <params> is the tool call's raw parameters -- for code_interpreter,
    # a JSON string like {"code": "..."} (see multi_turn.generate's recording).
    # Take the LAST call's `code` argument as the submitted solution.
    last_code = ""
    for t in trace:
        call = t.get("tool_call", "") if isinstance(t, dict) else ""
        m = re.search(r"\((.*)\)\s*$", call, re.DOTALL)
        params = m.group(1) if m else call
        code = ""
        try:
            obj = json.loads(params)
            if isinstance(obj, dict):
                code = str(obj.get("code", ""))
            elif isinstance(obj, str):
                code = obj
        except Exception:
            code = params  # not JSON -> treat the raw params as code
        if code.strip():
            last_code = code
    return last_code


async def _grade_one_code(sample: Sample, args: Namespace | None = None) -> bool:
    """Grade a code sample by running its candidate program against its test
    cases in the sandbox (examples/SRD/tools/code/judge.py). Correct =
    ALL tests pass (all-or-nothing, matching math). The candidate is the code
    the model RAN via the tool (see _code_candidate) so the tool is mandatory.
    Imported lazily so examples/SRD has no hard dependency on SDPO_ReAct."""
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    tests = md.get("test_cases") or []
    candidate = _code_candidate(sample, args)
    if not candidate.strip() or not tests:
        return False
    try:
        from examples.SRD.tools.code.judge import grade_code

        # candidate is already raw code (not a response) -> grade_code's
        # _extract_code_block is a no-op on plain code, so pass it directly.
        return (await grade_code(candidate, tests)) >= 1.0
    except Exception as e:  # a judge failure must not crash the whole group
        logger.warning(f"code judge failed ({e!r}); grading trace as incorrect.")
        return False


def _em_check_search(sample: Sample, args: Namespace | None = None) -> bool:
    """EM-grade a search/QA sample against golden_answers (multi-hop QA).
    Extracts the <answer> tag and exact-matches (normalized) against any golden
    answer, reusing examples/search-r1/qa_em_format.py::em_check. golden_answers
    live on metadata['golden_answers']. Lazy import (no hard dep)."""
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    golden = md.get("golden_answers") or ([sample.label] if sample.label else [])
    pred = _extract_tagged_answer(sample.response, getattr(args, "sdpo_answer_tag", "answer") if args else "answer")
    if pred is None or not golden:
        return False
    try:
        from examples.search_r1.qa_em_format import em_check  # package path may vary
    except Exception:
        try:
            import importlib.util

            _p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "search-r1", "qa_em_format.py")
            spec = importlib.util.spec_from_file_location("qa_em_format", _p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            em_check = mod.em_check
        except Exception as e:
            logger.warning(f"search EM grader import failed ({e!r}); grading incorrect.")
            return False
    return bool(em_check(pred, golden))


def _build_search_judge_prompt(args: Namespace, sample: Sample) -> tuple[str, str]:
    """Same shape as _build_judge_prompt, but for search/QA: the reference is
    the FULL golden_answers list (multi-hop QA often has multiple acceptable
    phrasings, e.g. "USA" vs "United States"), not just sample.label (EM's
    single golden pick). Question comes from metadata['question'] when the
    builder recorded it (build_search_data.py); falls back to the rendered
    prompt string (works, but includes the <tools> block / system prompt --
    acceptable degradation, this path only runs when EM already said "wrong",
    a second opinion with extra context in the prompt is still informative)."""
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    question = md.get("question") or (sample.prompt if isinstance(sample.prompt, str) else "")
    golden = md.get("golden_answers") or ([sample.label] if sample.label else [])
    reference = "; ".join(str(g) for g in golden)
    extracted = _extract_answer(args, sample)
    full = sample.response or ""
    if len(full) > 8000:
        full = full[:4000] + "\n...[truncated]...\n" + full[-3000:]
    user = (
        f"QUESTION:\n{question}\n\n"
        f"ACCEPTABLE REFERENCE ANSWER(S):\n{reference}\n\n"
        f"MODEL FULL RESPONSE:\n{full}\n\n"
        f"MODEL EXTRACTED ANSWER:\n{extracted if extracted is not None else '(none — no <answer> tag found)'}\n\n"
        "Is the model's answer correct? Reply CORRECT or INCORRECT."
    )
    return _SEARCH_JUDGE_SYSTEM, user




async def _grade_one_search(sample: Sample, args: Namespace | None = None) -> bool:
    """Grade a search/QA sample: EM first (cheap, exact), then -- only when EM
    says "wrong" and --sdpo-search-judge-fallback is set -- a second opinion
    from the LLM judge before finalizing "incorrect". EM's normalized string
    equality has no tolerance for a correct answer phrased differently than the
    single golden string it happens to compare against (multi-hop QA often has
    several acceptable phrasings; EM only catches an exact one), so a plain EM
    miss is not strong evidence of an actually-wrong trace -- this reduces
    those false negatives without touching an EM HIT (EM correct always short-
    circuits, no judge call, no double up-weighting of the same trace)."""
    if _em_check_search(sample, args):
        return True
    if not getattr(args, "sdpo_search_judge_fallback", False):
        return False
    # No extracted <answer> at all (e.g. a multi-turn trace truncated mid-tool-
    # use by --generate-max-turns/--rollout-max-response-len before it ever
    # wrote one -- observed on ~half of a random EM-wrong sample) is
    # unambiguously wrong; skip the judge call entirely rather than spend a
    # gateway round-trip confirming the obvious.
    tag = getattr(args, "sdpo_answer_tag", "answer") if args else "answer"
    if _extract_tagged_answer(sample.response, tag) is None:
        return False
    try:
        system, user = _build_search_judge_prompt(args, sample)
        # fallback=False: on a judge/gateway failure, keep EM's "incorrect"
        # rather than falling through to _llm_judge_correct's own default
        # fallback (_is_correct, a math/mcq grader -- meaningless for search).
        return await _llm_judge_correct(args, sample, system=system, user=user, fallback=False)
    except Exception as e:
        logger.warning(f"search LLM-judge fallback failed ({e!r}); keeping EM's 'incorrect' verdict.")
        return False


def _grade_one_webshop(sample: Sample, args: Namespace | None = None) -> bool:
    """Grade a webshop episode: correct iff the episode reached a terminal
    step with won=True. UNLIKE code/search, there is no text to parse or
    judge -- the WebShop sidecar itself is the ground truth, stamped onto
    metadata['episode_won'] by tools/webshop/client.py the moment a terminal
    step() response comes back (see that module's docstring). A trajectory
    truncated by --generate-max-turns before any terminal step simply never
    gets this key -- .get(..., False) correctly grades it a loss, same as
    _grade_one_search treats a missing <answer> tag as an unambiguous miss.
    Plain sync (no judge/gateway call, no semaphore needed)."""
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    return bool(md.get("episode_won", False))


def _grade_one_alfworld(sample: Sample, args: Namespace | None = None) -> bool:
    """Grade an alfworld episode: correct iff the episode reached a terminal
    step with won=True. Same ground-truth-from-the-sidecar shape as
    _grade_one_webshop -- see that function's docstring."""
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    return bool(md.get("episode_won", False))


async def _grade_group(args: Namespace, group: list[Sample]) -> list[bool]:
    """Correctness for every trace in a group. Domain-aware: code samples
    (metadata['domain']=='code') run their program against test cases; search
    samples (=='search') EM-match golden answers; webshop/alfworld samples
    read their sidecar-stamped reward/episode_won signal; the rest use the LLM
    judge (if --sdpo-judge) or deterministic matching. A mixed
    math+code+search group grades each sample by its OWN domain, so one
    multi-domain run trains all domains with consistent correctness."""
    domains = {_sample_domain(s) for s in group}
    special = domains & {"code", "search", "webshop", "alfworld"}
    # Fast path: pure math/mcq group + no judge -> the original sync path.
    if not special and not getattr(args, "sdpo_judge", False):
        return [_is_correct(s, args) for s in group]

    if special:
        sem = _judge_semaphore(args)  # reuse the bounded-concurrency cap

        async def _one_mixed(s: Sample) -> bool:
            dom = _sample_domain(s)
            if dom == "code":
                async with sem:
                    return await _grade_one_code(s, args)
            if dom == "search":
                # EM is cheap/sync and short-circuits on a hit; only take the
                # semaphore (shared with code/math judge calls) when an EM MISS
                # will actually fall through to a judge gateway call.
                if _em_check_search(s, args) or not getattr(args, "sdpo_search_judge_fallback", False):
                    return await _grade_one_search(s, args)
                async with sem:
                    return await _grade_one_search(s, args)
            if dom == "webshop":
                return _grade_one_webshop(s, args)
            if dom == "alfworld":
                return _grade_one_alfworld(s, args)
            if getattr(args, "sdpo_judge", False) and (s.response or "").strip():
                async with sem:
                    return await _llm_judge_correct(args, s)
            return _is_correct(s, args)

        return list(await asyncio.gather(*(_one_mixed(s) for s in group)))

    if not getattr(args, "sdpo_judge", False):
        return [_is_correct(s, args) for s in group]

    # Process-wide cap (shared across all concurrently-running groups), so the
    # judge calls are concurrent — both within a group (gather) and across groups
    # (each group is its own asyncio task) — without overloading the gateway.
    sem = _judge_semaphore(args)

    async def _one(s: Sample) -> bool:
        # Skip the judge for empty responses (cheap, obviously wrong).
        if not (s.response or "").strip():
            return False
        async with sem:
            return await _llm_judge_correct(args, s)

    return list(await asyncio.gather(*(_one(s) for s in group)))
