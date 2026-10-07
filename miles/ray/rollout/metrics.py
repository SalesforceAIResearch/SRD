import json
import logging
import os
from typing import Any

import numpy as np

from miles.utils import tracking_utils
from miles.utils.iter_utils import group_by
from miles.utils.metric_utils import (
    compute_pass_rate,
    compute_rollout_step,
    compute_statistics,
    dict_add_prefix,
    has_repetition,
)
from miles.utils.misc import load_function
from miles.utils.types import Sample


logger = logging.getLogger(__name__)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    if (x := args.custom_eval_rollout_log_function_path) is not None:
        custom_log_func = load_function(x)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}
    all_rewards: list[float] = []  # pooled across all eval datasets for eval/all
    per_dataset_acc: list[float] = []  # per-dataset acc for the equal-weight eval/mean
    for key in data.keys():
        # A sample can reach eval aggregation with reward=None (e.g. an agentic
        # episode that timed out / never recorded a model call, so no RM ever
        # ran on it -- see miles.rollout.generate_hub.agentic_tool_call.generate's
        # ABORTED path). Treat that the same as a failed episode (0.0) rather
        # than crashing the whole eval on `sum(rewards)` -- one slow tau2/
        # webshop/alfworld episode under real infra load must not take down
        # every OTHER dataset's score in the same eval pass.
        rewards = [r if r is not None else 0.0 for r in data[key]["rewards"]]
        all_rewards.extend(rewards)
        dataset_acc = sum(rewards) / len(rewards)
        per_dataset_acc.append(dataset_acc)
        log_dict[f"eval/{key}"] = dataset_acc
        # val-aux: per-domain accuracy (avg@N) lives with the auxiliary metrics.
        log_dict[f"val-aux/{key}_acc"] = dataset_acc
        webshop_core = {}  # webshop_score, promoted to val-core (see below)
        if (samples := data[key].get("samples")) is not None:
            # response_len, truncated, repetition, etc. -> val-aux
            log_dict |= dict_add_prefix(_compute_metrics_from_samples(args, samples), f"val-aux/{key}/")
            # alfworld_ood's per-task-type (Pick/Look/Clean/Heat/Cool/Pick2)
            # breakdown -> val-aux (additive, keyed off metadata stamped at
            # data-build/rollout time).
            log_dict |= dict_add_prefix(_compute_per_task_type_eval_metrics(args, samples), f"val-aux/{key}/")
            # webshop's continuous partial-credit webshop_score is a more
            # informative headline than its binary pass@1, so it is webshop's
            # CORE val metric (-> val-core/<key>/webshop_score); the pass@1 block
            # below then keeps webshop's pass@1 in val-aux instead. {} for
            # non-webshop datasets, whose samples carry no webshop_task_score.
            webshop_core = _compute_webshop_score_eval_metrics(samples)
            log_dict |= dict_add_prefix(webshop_core, f"val-core/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"val-aux/{key}_truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            pr = compute_pass_rate(flat_rewards=rewards, group_size=args.n_samples_per_eval_prompt)
            # per-domain pass@1 -> val-core (the headline); pass@2/4/... -> val-aux.
            # EXCEPTION: webshop's headline is its continuous webshop_score
            # (promoted to val-core above), so webshop's pass@1 goes to val-aux too.
            if "pass@1" in pr and not webshop_core:
                log_dict[f"val-core/{key}_pass@1"] = pr["pass@1"]
                aux_pr = {k: v for k, v in pr.items() if k != "pass@1"}
            else:
                aux_pr = pr
            log_dict |= dict_add_prefix(aux_pr, f"val-aux/{key}_")
            # keep the legacy eval/ keys too for backward compatibility
            log_dict |= dict_add_prefix(pr, f"eval/{key}-")

    # Two overall val scores across all domains:
    #   eval/all  = sample-pooled mean (domains with more samples weigh more)
    #   eval/mean = equal-weight mean of per-domain accuracies (each domain 1/N)
    if len(data) > 1 and all_rewards:
        log_dict["eval/all"] = sum(all_rewards) / len(all_rewards)
        log_dict["eval/mean"] = sum(per_dataset_acc) / len(per_dataset_acc)
        log_dict["val-aux/all_acc"] = log_dict["eval/all"]
        log_dict["val-aux/mean_acc"] = log_dict["eval/mean"]
        if args.log_passrate:
            pr_all = compute_pass_rate(flat_rewards=all_rewards, group_size=args.n_samples_per_eval_prompt)
            # merged pass@1 -> val-core (the single headline val score you want)
            if "pass@1" in pr_all:
                log_dict["val-core/all_pass@1"] = pr_all["pass@1"]
            log_dict |= dict_add_prefix({k: v for k, v in pr_all.items() if k != "pass@1"}, "val-aux/all_")
            log_dict |= dict_add_prefix(pr_all, "eval/all-")

    logger.info(f"eval {rollout_id}: {log_dict}")

    # Eval-only sidecar dump of the SAME log_dict that goes to wandb, for the
    # batch eval harnesses (examples/SRD/ablation/eval-*.sh) that read a
    # benchmark number out of a one-shot eval job instead of a wandb run.
    metrics_file = os.environ.get("MILES_EVAL_METRICS_FILE")
    if metrics_file:
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(log_dict, f, indent=2)
        logger.info(f"eval metrics written to {metrics_file}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    tracking_utils.log(args, log_dict, step_key="eval/step")

    return log_dict


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if (x := args.custom_rollout_log_function_path) is not None:
        custom_log_func = load_function(x)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    sample_metrics = _compute_metrics_from_samples(args, samples)
    log_dict = {**(rollout_extra_metrics or {})}
    # skill/* and domain/* already carry their own panel prefix -> keep them
    # top-level (NOT under rollout/) so they show as their own wandb panels
    # (skill/, domain/code/, domain/math/, ...); everything else goes under rollout/.
    _own_panel = ("skill/", "domain/")
    log_dict |= dict_add_prefix(
        {k: v for k, v in sample_metrics.items() if not k.startswith(_own_panel)}, "rollout/"
    )
    for k, v in sample_metrics.items():
        if k.startswith(_own_panel):
            log_dict[k] = v
    log_dict |= dict_add_prefix(_compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    # Dedicated "response_len/" panel: all length-related metrics in one place.
    for k, v in sample_metrics.items():
        if k.startswith("response_len/") or k == "truncated_ratio":
            log_dict[f"response_len/{k[len('response_len/'):] if k.startswith('response_len/') else k}"] = v
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking_utils.log(args, log_dict, step_key="rollout/step")


def _compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_prefix_cache_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()

    oldest_versions = [s.oldest_weight_version for s in samples if s.oldest_weight_version is not None]
    if oldest_versions:
        log_dict |= dict_add_prefix(compute_statistics(oldest_versions), "weight_version/")
        mixed = sum(1 for s in samples if len(set(s.weight_versions)) > 1)
        log_dict["weight_version/mixed_version_ratio"] = mixed / len(samples)

    # SDPO diagnostics (examples/SRD/sdpo.py): per-sample mean log-prob of the
    # sampled token under student vs teacher, and their diff (sampled-token
    # reverse KL). No-op for non-SDPO runs where these keys are absent.
    for mkey, out_key in (
        ("sdpo_student_logp_mean", "sdpo/student_logp"),
        ("sdpo_teacher_logp_mean", "sdpo/teacher_logp"),
        ("sdpo_logp_diff_mean", "sdpo/logp_diff"),
        # True task success rate (survives the zeroed reward under pure distill)
        # and response perplexity of the sampled rollout.
        ("sdpo_correct", "sdpo/success_rate"),
        ("sdpo_ppl", "sdpo/ppl"),
    ):
        vals = [s.metadata[mkey] for s in samples if isinstance(s.metadata, dict) and mkey in s.metadata]
        if vals:
            log_dict[out_key] = float(np.mean(vals))

    # SDPO self-skill (rollout-side): dedicated skill/ panel — skill length
    # min/max/mean and skill perplexity. Present only when --sdpo-self-skill is on
    # and at least one skill was generated this rollout.
    skill_lens = [s.metadata["sdpo_skill_len"] for s in samples if isinstance(s.metadata, dict) and "sdpo_skill_len" in s.metadata]
    if skill_lens:
        log_dict["skill/length_mean"] = float(np.mean(skill_lens))
        log_dict["skill/length_min"] = float(np.min(skill_lens))
        log_dict["skill/length_max"] = float(np.max(skill_lens))
        log_dict["skill/count"] = float(len(skill_lens))
    skill_ppls = [s.metadata["sdpo_skill_ppl"] for s in samples if isinstance(s.metadata, dict) and "sdpo_skill_ppl" in s.metadata]
    if skill_ppls:
        log_dict["skill/ppl"] = float(np.mean(skill_ppls))
    # --sdpo-response-prefix skill: fraction of response teacher prefixes that
    # actually used the peer's skill (vs falling back to the peer's full trace).
    rp_skill = [
        s.metadata["sdpo_response_prefix_is_skill"]
        for s in samples
        if isinstance(s.metadata, dict) and "sdpo_response_prefix_is_skill" in s.metadata
    ]
    if rp_skill:
        log_dict["skill/response_prefix_is_skill_frac"] = float(np.mean(rp_skill))

    log_dict |= _compute_agentic_tool_metrics(args, samples)
    log_dict |= _compute_per_domain_metrics(args, samples)
    log_dict |= _compute_reward_breakdown_metrics(args, samples)

    tito_vals = [s.metadata.get("tito_session_mismatch") for s in samples]
    tito_vals = [v for v in tito_vals if v is not None]
    if tito_vals:
        log_dict["tito_session_mismatch_rate"] = np.mean([len(v) > 0 for v in tito_vals]).item()
        for mtype in ("special_token_count", "special_token_type", "non_assistant_text", "assistant_text"):
            log_dict[f"tito_session_mismatch_rate/{mtype}"] = np.mean(
                [any(m.get("type") == mtype for m in v) for v in tito_vals]
            ).item()
        if args.ci_test:
            for strict_type in ("special_token_count", "special_token_type", "non_assistant_text"):
                rate = log_dict.get(f"tito_session_mismatch_rate/{strict_type}", 0)
                assert rate == 0, (
                    f"tito_session_mismatch_rate/{strict_type}={rate:.4f} must be 0 — "
                    "this indicates a bug in the TITO algorithm or chat template. "
                    "Please check your tito model and chat template."
                )
            # assistant_text mismatch is non-critical: assistant tokens are inherited
            # from the pretokenized prefix and may differ from canonical tokenization.

    return log_dict


def _compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    counts = {reward: len(items) for reward, items in group_by(interesting_rewards).items()}
    log_dict = {f"zero_std/count_{reward}": count for reward, count in counts.items()}

    # Percentages over total groups, so "too hard" (all-0) and "too easy"
    # (all-1) rates are comparable across runs without needing to know the
    # rollout batch size.
    total_groups = len(all_sample_groups)
    if total_groups > 0:
        log_dict["zero_std/all_zero_percentage"] = counts.get("0.0", 0) / total_groups
        log_dict["zero_std/all_one_percentage"] = counts.get("1.0", 0) / total_groups

    return log_dict


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if args.sglang_speculative_algorithm is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}


def _compute_per_domain_metrics(args, all_samples: list[Sample]):
    """Per-task-type (domain) breakdown of the key training metrics, for MIXED
    multi-task rollouts (e.g. shuffled math + code + search). Groups samples by
    metadata['domain'] and emits, under a per-domain panel `domain/<name>/...`:
      - success_rate (sdpo_correct), count, frac_of_batch
      - response_len_mean, truncated_ratio
      - tool_call_count_mean, zero_tool_call_frac, round_number_mean
    So math vs code vs search progress is visible separately (an aggregate hides
    which domain is improving/collapsing). No-op (empty) when all samples share
    ONE domain -- single-domain runs already have the global metrics above, so
    this only adds panels when there's genuinely a mix to break out."""
    def _dom(s):
        md = s.metadata if isinstance(s.metadata, dict) else {}
        return (md.get("domain") or "math").strip().lower()

    by_domain: dict[str, list[Sample]] = {}
    for s in all_samples:
        by_domain.setdefault(_dom(s), []).append(s)
    if len(by_domain) <= 1:
        return {}

    def _meanmd(subset, key):
        vals = [s.metadata[key] for s in subset if isinstance(s.metadata, dict) and key in s.metadata]
        return float(np.mean(vals)) if vals else None

    out = {}
    n_total = len(all_samples)
    for dom, subset in sorted(by_domain.items()):
        p = f"domain/{dom}/"
        out[p + "count"] = float(len(subset))
        out[p + "frac_of_batch"] = len(subset) / n_total if n_total else 0.0
        out[p + "response_len_mean"] = float(np.mean([s.effective_response_length for s in subset]))
        out[p + "truncated_ratio"] = float(np.mean([int(s.status == Sample.Status.TRUNCATED) for s in subset]))
        for mkey, okey in (("sdpo_correct", "success_rate"), ("tool_call_count", "tool_call_count_mean"),
                           ("round_number", "round_number_mean"), ("tool_error_count", "tool_error_count_mean")):
            v = _meanmd(subset, mkey)
            if v is not None:
                out[p + okey] = v
        tcc = [s.metadata["tool_call_count"] for s in subset if isinstance(s.metadata, dict) and "tool_call_count" in s.metadata]
        if tcc:
            out[p + "zero_tool_call_frac"] = float(np.mean([int(t == 0) for t in tcc]))
    return out


def _compute_per_task_type_eval_metrics(args, samples: list[Sample]):
    """Per-ALFRED-task-type (Pick/Look/Clean/Heat/Cool/Pick2) breakdown of an
    eval dataset's win rate, keyed off metadata['task_type'] (stamped by
    examples/SRD/data/build_alfworld_data.py's _task_type_from_path).
    No-op for non-alfworld eval datasets (e.g. webshop), whose samples never
    carry this key. The combined val-aux/<key>_acc / eval/<key> score already
    computed by log_eval_rollout_data covers the aggregate across all 6 types
    -- this only adds the per-type split, so neither replaces the other."""
    by_type: dict[str, list[Sample]] = {}
    for s in samples:
        md = s.metadata if isinstance(s.metadata, dict) else {}
        task_type = md.get("task_type")
        if task_type:
            by_type.setdefault(task_type, []).append(s)
    if not by_type:
        return {}

    out = {}
    for task_type, subset in sorted(by_type.items()):
        rewards = [s.get_reward_value(args) if s.reward is not None else 0.0 for s in subset]
        out[f"task_type/{task_type}"] = float(np.mean(rewards))
    return out


def _compute_webshop_score_eval_metrics(samples: list[Sample]):
    """Webshop's continuous partial-credit task_score, additive alongside the
    binary win-rate acc/pass@k that log_eval_rollout_data already computes
    from sample.reward. Stamped onto metadata['webshop_task_score'] by
    examples/SRD/tools/webshop/client.py on the terminal step. No-op
    for non-webshop eval datasets, whose samples never carry this key."""
    scores = [
        s.metadata["webshop_task_score"]
        for s in samples
        if isinstance(s.metadata, dict) and "webshop_task_score" in s.metadata
    ]
    if not scores:
        return {}
    return {"webshop_score": float(np.mean(scores))}


def _compute_reward_breakdown_metrics(args, all_samples: list[Sample]):
    """Generic per-component reward breakdown, for domains whose grader
    computes a reward from several independent checks (e.g. tau2's DB/
    ENV_ASSERTION/ACTION/COMMUNICATE/NL_ASSERTION checks, multiplied
    together into the single scalar sample.reward -- a 0.0 doesn't say
    WHICH check failed). No-op (empty dict) unless a custom generate/reward
    function stashed one or more `tau2_reward_<COMPONENT>` keys on
    sample.metadata (see examples/SRD/tools/tau2/agent_function.py).

    Emits, under the reward_breakdown/ panel: reward_breakdown/<component>
    (mean across samples that have that key -- not every sample necessarily
    has every component, e.g. NL_ASSERTION only appears on tasks that carry
    NL assertions).
    """
    keys = set()
    for s in all_samples:
        if isinstance(s.metadata, dict):
            keys.update(k for k in s.metadata if k.startswith("tau2_reward_"))
    out = {}
    for key in sorted(keys):
        vals = [s.metadata[key] for s in all_samples if isinstance(s.metadata, dict) and key in s.metadata]
        if vals:
            component = key[len("tau2_reward_") :]
            out[f"reward_breakdown/{component}"] = float(np.mean(vals))
    return out


def _compute_agentic_tool_metrics(args, all_samples: list[Sample]):
    """Fine-grained tool-calling diagnostics for agentic/multi-turn rollout.

    No-op (empty dict) unless a custom generate/reward function populated one
    of these on sample.metadata -- this is a generic panel, not tied to any
    one example. round_number is populated for free by
    ``miles.rollout.generate_hub.multi_turn.generate`` (any run using it gets
    this panel automatically); tool_call_count/tool_error_count are populated
    by whichever --custom-rm-path/--custom-generate-function-path chooses to
    stash them (e.g. examples/SRD/sdpo_react.py).

    Keys, all under the agentic/ panel:
      - round_number_{mean,max,min}: turns used per trajectory.
      - hit_max_turns_frac: fraction that used the FULL turn budget (proxy for
        "ran out of turns before answering" -- distinct from round_number's
        mean, which a few long outliers can hide).
      - tool_call_count_{mean,max}: tool calls per trajectory.
      - zero_tool_call_frac: fraction that never called a tool at all.
      - tool_error_rate: errors / total tool calls, when the generate/reward
        function also tags samples with tool_error_count (calls whose
        observation text signals failure, e.g. a sandbox exception).
    """
    round_numbers = [
        s.metadata["round_number"] for s in all_samples if isinstance(s.metadata, dict) and "round_number" in s.metadata
    ]
    tool_call_counts = [
        s.metadata["tool_call_count"]
        for s in all_samples
        if isinstance(s.metadata, dict) and "tool_call_count" in s.metadata
    ]
    tool_error_counts = [
        s.metadata["tool_error_count"]
        for s in all_samples
        if isinstance(s.metadata, dict) and "tool_error_count" in s.metadata
    ]

    if not round_numbers and not tool_call_counts:
        return {}

    metrics = {}
    if round_numbers:
        metrics["agentic/round_number_mean"] = float(np.mean(round_numbers))
        metrics["agentic/round_number_max"] = float(np.max(round_numbers))
        metrics["agentic/round_number_min"] = float(np.min(round_numbers))
        max_turns = getattr(args, "generate_max_turns", None)
        if max_turns:
            metrics["agentic/hit_max_turns_frac"] = float(np.mean([r >= max_turns for r in round_numbers]))

    if tool_call_counts:
        metrics["agentic/tool_call_count_mean"] = float(np.mean(tool_call_counts))
        metrics["agentic/tool_call_count_max"] = float(np.max(tool_call_counts))
        metrics["agentic/zero_tool_call_frac"] = float(np.mean([c == 0 for c in tool_call_counts]))

    if tool_error_counts and tool_call_counts and len(tool_error_counts) == len(tool_call_counts):
        total_calls = sum(tool_call_counts)
        if total_calls > 0:
            metrics["agentic/tool_error_rate"] = float(sum(tool_error_counts) / total_calls)

    return metrics
