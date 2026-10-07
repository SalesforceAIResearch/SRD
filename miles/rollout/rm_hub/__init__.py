import asyncio
import random

import aiohttp

from miles.utils.misc import load_function
from miles.utils.types import Sample

from .deepscaler import get_deepscaler_rule_based_reward, get_gemma_math_reward
from .f1 import f1_score
from .gpqa import compute_gpqa_reward
from .math_dapo_utils import compute_score as compute_score_dapo
from .math_utils import extract_answer as extract_boxed_answer
from .math_utils import grade_answer_verl


async def remote_rm(args, sample: Sample):
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def async_rm(args, sample: Sample, evaluation: bool = False, **kwargs):
    # --eval-custom-rm-path is documented (see arguments.py) as an eval-only grader
    # that runs "instead of" the training reward -- --group-rm examples rely on
    # this since their group reward function can't run in eval (no group step).
    # Non-group-rm examples reach this same fork, but previously --custom-rm-path
    # always won regardless of `evaluation`, so an eval-only grader (e.g. one that
    # does real math-equivalence checking instead of the training-time strict
    # \boxed{} string match) never actually ran during eval.
    if evaluation and getattr(args, "eval_custom_rm_path", None) is not None:
        rm_function = load_function(args.eval_custom_rm_path)
        return await rm_function(args, sample, **kwargs)
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
    response = sample.response
    label = sample.label
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]

    # This function is intended for remote or time-consuming reward model evaluation.
    # Implement the actual logic as needed.
    if rm_type == "remote_rm":
        return await remote_rm(args, sample)
    elif rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)
    elif rm_type == "gemma_math":
        return get_gemma_math_reward(response, label)
    elif rm_type == "dapo":
        return compute_score_dapo(response, label)
    elif rm_type == "math":
        return 1 if grade_answer_verl(response, label) else 0
    elif rm_type == "f1":
        return f1_score(response, label)[0]
    elif rm_type == "gpqa":
        return compute_gpqa_reward(response, label, metadata=metadata)
    elif rm_type == "ifbench":
        from .ifbench import compute_ifbench_reward

        return compute_ifbench_reward(response, label, metadata=metadata)
    elif rm_type == "random":
        return random.randint(0, 1)
    elif rm_type:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
    else:
        raise NotImplementedError("Rule-based RM type is not specified.")


async def batched_async_rm(
    args,
    samples: list[Sample],
    inplace_set_reward_field: bool = False,
    **kwargs,
) -> list[int | float] | None:
    if inplace_set_reward_field:
        rewards = await batched_async_rm(args, samples, **kwargs)
        for sample, reward in zip(samples, rewards, strict=True):
            assert (
                sample.reward is None
            ), f"Overriding sample.reward from {sample.reward} to {reward}, is this intended?"
            sample.reward = reward
        return None

    # --eval-custom-rm-path's documented signature is per-sample (`eval_rm(args, sample)`),
    # not batched like --custom-rm-path, so route through async_rm per-sample instead of
    # the batch-mode custom_rm_path function below.
    if kwargs.get("evaluation") and getattr(args, "eval_custom_rm_path", None) is not None:
        tasks = [async_rm(args, sample, **kwargs) for sample in samples]
        return await asyncio.gather(*tasks)

    if args.custom_rm_path is not None:
        # Ensure the custom reward function is implemented in batch mode
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
