from argparse import Namespace
from collections.abc import Callable
from typing import Protocol

import torch

from miles.backends.training_utils.cp_utils import (
    all_gather_with_cp,
    get_local_response_loss_masks,
    get_sum_of_sample_mean,
)
from miles.backends.training_utils.loss_hub.corrections import vanilla_tis_function
from miles.backends.training_utils.loss_hub.logit_processors import (
    get_log_probs_and_entropy,
    get_responses,
    get_values,
)
from miles.backends.training_utils.loss_hub.math_utils import (
    _gather_true_on_policy_full_logits,
    compute_approx_kl,
    compute_ess_ratio_contribution,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.misc import load_function
from miles.utils.types import RolloutBatch


def _sdpo_alpha_from_args(args: Namespace) -> float:
    """Map --sdpo-divergence to the original SDPO alpha for the ALPHA-MIXTURE modes
    (0=forward KL[default], 1=reverse KL, 0.5=JSD). The additive modes (jeffrey,
    jeffrey_jsd) are NOT alpha-mixtures and are handled separately in the KD loss;
    this returns 0.0 for them (unused)."""
    mode = getattr(args, "sdpo_divergence", "forward_kl")
    return {"forward_kl": 0.0, "reverse_kl": 1.0, "jsd": 0.5}.get(mode, 0.0)


def _sdpo_add_tail_logprobs(log_probs: torch.Tensor) -> torch.Tensor:
    """Append one aggregated tail-bucket log-prob = log(1 - sum(exp(top-k logp))),
    computed stably via logsumexp/expm1 (matches original SDPO add_tail)."""
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)  # ensure sum(top-k probs) < 1
    tail_log = torch.log(-torch.expm1(log_s))  # log(1 - exp(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def _sdpo_kd_loss_per_token(
    args: Namespace,
    logits: torch.Tensor,
    batch: RolloutBatch,
) -> torch.Tensor | None:
    """SDPO distribution-level KD loss, per response token, GRAD-ENABLED.

    Aligned with the original lasgroup/SDPO ``compute_self_distillation_loss``:
    - student log-probs at the TEACHER's top-k ids (teacher defines the support),
      from the grad-enabled current forward; teacher log-probs are the detached
      target.
    - tail bucket via logsumexp (_sdpo_add_tail_logprobs) so top-k + tail is a
      proper distribution.
    - alpha-parameterised divergence in LOG space with F.kl_div(log_target=True):
      alpha=0 -> forward KL KL(teacher‖student) [default, mass-covering KD],
      alpha=1 -> reverse KL KL(student‖teacher),
      0<alpha<1 -> generalized JSD via the log-mixture + torch.lerp.
    - optional importance-sampling ratio correction exp(student_logp-old_logp)
      clamped to <= is_clip (for off-policy/async), applied per token.

    Returns a 1D tensor of per-token loss concatenated across samples (same layout
    as get_log_probs_and_entropy["log_probs"]), or None if no teacher target.
    """
    import torch.nn.functional as F

    t_lp_list = batch.get("sdpo_teacher_topk_logprobs")
    t_ids_list = batch.get("sdpo_teacher_topk_ids")
    if t_lp_list is None or t_ids_list is None:
        return None

    parallel_state = get_parallel_state()
    tp_group = parallel_state.tp.group
    device = logits.device

    # Divergence mode. Alpha-mixture modes (forward_kl/reverse_kl/jsd) use `alpha`
    # below (0.0 = forward KL, 1.0 = reverse KL, 0.5 = JSD). Additive modes are:
    #   jeffrey      = forward KL + reverse KL           (symmetric, RKL + FKL)
    #   jeffrey_jsd  = forward KL + JSD                  (Jeffrey with its RKL half
    #                  swapped for the milder, bounded JSD — less mode-seeking)
    div_mode = getattr(args, "sdpo_divergence", "forward_kl")
    alpha = _sdpo_alpha_from_args(args)
    add_tail = getattr(args, "sdpo_distillation_add_tail", True)
    is_clip = getattr(args, "sdpo_is_clip", 2.0)
    if is_clip is not None and is_clip <= 0:
        is_clip = None

    # Old (rollout) sampled log-probs, per sample, for the IS ratio correction.
    old_lp_list = batch.get("rollout_log_probs")

    # Cap the KD computation to the first `kd_max_tokens` response tokens per
    # sample (0/None = whole response). This bounds the expensive full-vocab
    # log_softmax (grad) memory, and focuses distillation on the early tokens.
    kd_max_tokens = getattr(args, "sdpo_kd_max_tokens", 0) or 0

    per_token = []
    for idx, (logits_chunk, _tokens_chunk) in enumerate(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens", None),
        )
    ):
        n = logits_chunk.size(0)
        t_ids = t_ids_list[idx]
        if n == 0 or t_ids is None or t_ids.numel() == 0:
            per_token.append(torch.zeros((n,), device=device))
            continue
        # Restrict to the first kd_max_tokens response positions (tokens beyond
        # the cap contribute 0 to the KD loss, keeping per-sample length == n).
        m = n if kd_max_tokens <= 0 else min(n, kd_max_tokens)
        t_ids = t_ids[:m].to(device)  # [m, k]
        teacher_lp = t_lp_list[idx][:m].to(device).float()  # [m, k] detached target

        # Student log-softmax (grad) at the teacher's top-k ids (first m positions).
        full_logits = _gather_true_on_policy_full_logits(
            logits_chunk[:m].contiguous(), tp_group, vocab_size=getattr(args, "vocab_size", None)
        )
        student_logsm = torch.log_softmax(full_logits.float(), dim=-1)  # [m, V] grad
        student_lp = torch.gather(student_logsm, dim=-1, index=t_ids)  # [n, k] grad

        if add_tail:
            student_lp = _sdpo_add_tail_logprobs(student_lp)  # [n, k+1]
            teacher_lp = _sdpo_add_tail_logprobs(teacher_lp)
        else:
            student_lp = student_lp - torch.logsumexp(student_lp, dim=-1, keepdim=True)
            teacher_lp = teacher_lp - torch.logsumexp(teacher_lp, dim=-1, keepdim=True)

        # Per-token divergence terms in LOG space. F.kl_div(input=log-prob, target,
        # log_target=True) = sum target*(logtarget - input), summed over the vocab
        # axis to get a [m] per-token value.
        def _forward_kl() -> torch.Tensor:  # KL(teacher ‖ student), mass-covering
            return F.kl_div(student_lp, teacher_lp, reduction="none", log_target=True).sum(-1)

        def _reverse_kl() -> torch.Tensor:  # KL(student ‖ teacher), mode-seeking
            return F.kl_div(teacher_lp, student_lp, reduction="none", log_target=True).sum(-1)

        def _jsd(a: float = 0.5) -> torch.Tensor:  # generalized JSD via log-mixture
            at = torch.tensor(a, dtype=student_lp.dtype, device=student_lp.device)
            mixture_lp = torch.logsumexp(
                torch.stack([student_lp + torch.log1p(-at), teacher_lp + torch.log(at)]), dim=0
            )
            kl_teacher = F.kl_div(mixture_lp, teacher_lp, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_lp, student_lp, reduction="none", log_target=True)
            return torch.lerp(kl_student, kl_teacher, a).sum(-1)

        if div_mode == "jeffrey":
            # Jeffrey = forward KL + reverse KL (symmetric; RKL with an added FKL).
            d = _forward_kl() + _reverse_kl()
        elif div_mode == "jeffrey_jsd":
            # Jeffrey with the reverse-KL half replaced by JSD (milder, bounded).
            d = _forward_kl() + _jsd()
        elif alpha == 0.0:
            d = _forward_kl()
        elif alpha == 1.0:
            d = _reverse_kl()
        else:
            d = _jsd(alpha)

        # Importance-sampling ratio correction (off-policy/async), on sampled token.
        if is_clip is not None and old_lp_list is not None and idx < len(old_lp_list):
            old_lp = old_lp_list[idx]
            if torch.is_tensor(old_lp) and old_lp.numel() == n:
                tok_m = _tokens_chunk[:m].to(device).unsqueeze(-1)
                s_sampled = torch.gather(student_logsm, dim=-1, index=tok_m).squeeze(-1)
                neg_approx_kl = torch.clamp((s_sampled - old_lp[:m].to(device)).detach(), min=-20.0, max=20.0)
                ratio = torch.exp(neg_approx_kl).clamp(max=is_clip)
                d = d * ratio
        # Pad back to full response length n so it aligns with active_tokens /
        # log_probs (tokens beyond the KD cap contribute 0).
        if d.shape[0] != n:
            d = torch.cat([d, d.new_zeros(n - d.shape[0])], dim=0)
        per_token.append(d)

    return torch.cat(per_token, dim=0)


class LossFunction(Protocol):
    """Common signature of the per-loss-type functions dispatched by `get_loss_function`."""

    def __call__(
        self,
        args: Namespace,
        batch: RolloutBatch,
        logits: torch.Tensor,
        sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute a scalar loss with gradient + a dict of detached scalar metrics.

        Args:
            args: Configuration. Each loss type reads its own subset of flags
                (PPO eps, KL coefficients, value clip, TIS settings, OPSM, etc.).
            batch: Mini-batch (`miles.utils.types.RolloutBatch`). Loss-type-
                dependent keys; common ones are `unconcat_tokens`, `response_lengths`,
                `total_lengths`, `loss_masks`, optional `max_seq_lens` (required for
                qkv_format="bshd"). Per-implementation docstrings list extras.
            logits: Float32. Last dim is vocab_size (policy/sft) or 1 (value).
                Outer shape is `[1, T, ...]` for qkv_format="thd" (T = sum of
                total_lengths) or `[B, max_seq_len, ...]` for "bshd".
            sum_of_sample_mean: CP-aware reducer; takes a flat per-token tensor
                and returns a scalar, sample-mean-weighted by `loss_masks`.

        Returns:
            `(loss, metrics)`:
              * `loss`: scalar tensor with grad, un-rescaled (the dispatcher
                applies Megatron scaling on top).
              * `metrics`: dict of detached 0-d scalars; surfaced under `train/`
                in the training log / wandb. Keys per loss type are documented
                on each implementation.
        """
        ...


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    parallel_state = get_parallel_state()
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    calculate_entropy = args.entropy_coef != 0 or args.observe_training_entropy

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=calculate_entropy,
        entropy_requires_grad=args.entropy_coef != 0,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    train_log_probs_list = log_probs
    old_log_probs_list = old_log_probs

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    local_loss_mask_list = get_local_response_loss_masks(
        total_lengths,
        response_lengths,
        batch["loss_masks"],
        args.qkv_format,
        max_seq_lens,
    )
    local_loss_masks = torch.cat(local_loss_mask_list, dim=0).to(device=ppo_kl.device)
    active_tokens = local_loss_masks.bool()
    ppo_kl = torch.where(
        active_tokens,
        torch.nan_to_num(ppo_kl, nan=0.0, posinf=0.0, neginf=0.0),
        ppo_kl.new_zeros(()),
    )
    advantages = torch.where(
        active_tokens,
        torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0),
        advantages.new_zeros(()),
    )

    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

    if getattr(args, "dump_details", None) is not None:
        from miles.backends.training_utils.debug_dump import maybe_dump_policy_loss_debug

        maybe_dump_policy_loss_debug(
            args=args,
            batch=batch,
            train_log_probs=train_log_probs_list,
            old_log_probs=old_log_probs_list,
            rollout_log_probs=batch.get("rollout_log_probs"),
            advantages=batch["advantages"],
            local_loss_masks=local_loss_mask_list,
            ppo_kl=ppo_kl,
            pg_loss=pg_loss,
        )

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": batch["log_probs"],
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
            "parallel_state": parallel_state,
            "max_seq_lens": max_seq_lens,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # [decouple IS and rejection] Rebuild sum_of_sample_mean with modified_response_masks for denominator correction
        # modified_response_masks will be sliced with cp in get_sum_of_sample_mean
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    # Determine pg_loss reducer: use custom if specified, otherwise default
    if args.custom_pg_loss_reducer_function_path is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        # Determine which loss_masks to use for pg_loss reducer
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    # ESS (Effective Sample Size) ratio from per-token IS weights
    # w = π_new/π_old = exp(-ppo_kl).  A value of 1.0 is on-policy; near 0
    # means the per-token weights are highly concentrated.
    ess_ratio_sum = compute_ess_ratio_contribution(
        ppo_kl=ppo_kl,
        loss_masks=batch["loss_masks"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        qkv_format=args.qkv_format,
        max_seq_lens=max_seq_lens,
        calculate_per_token_loss=args.calculate_per_token_loss,
    )

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # Per-response-token skill mask (aligned with the concatenated per-token layout),
    # built once for both the skill entropy metric and the skill KD split.
    _is_skill = batch.get("sdpo_is_skill")
    skill_tok_mask = None
    if _is_skill is not None and any(bool(x) for x in _is_skill):
        _rl = batch["response_lengths"]
        skill_tok_mask = torch.cat(
            [pg_loss.new_full((int(_rl[j]),), 1.0 if bool(_is_skill[j]) else 0.0) for j in range(len(_rl))]
        ).bool()

    # Per-response-token "was this skill distilled from a CORRECT trace" mask
    # (self-success/blind-correct) vs a FAILED one (pitfall-condense) -- see
    # actor.py::_append_sdpo_skill_samples's sdpo_skill_is_correct. Only
    # meaningful where skill_tok_mask is True; lets skill/kl split below tell
    # apart which half of skill-KD is contributing (motivating question behind
    # --sdpo-skill-kd-mode blind-correct/both-blind).
    _skill_correct = batch.get("sdpo_skill_is_correct")
    skill_correct_tok_mask = None
    if skill_tok_mask is not None and _skill_correct is not None and len(_skill_correct) == len(_rl):
        skill_correct_tok_mask = torch.cat(
            [pg_loss.new_full((int(_rl[j]),), 1.0 if bool(_skill_correct[j]) else 0.0) for j in range(len(_rl))]
        ).bool()

    entropy_loss = pg_loss.new_zeros(())
    skill_entropy = pg_loss.new_zeros(())
    # Per-key denominator override for train/entropy_loss. train/entropy_loss must
    # keep its ORIGINAL meaning: the mean entropy over the TRAIN-ROLLOUT (response)
    # tokens only. But skill-KD appends skill samples, and aggregate_train_losses
    # divides every metric by the batch-wide token count (response + skill). Report
    # the response-token count as entropy_loss's own denominator so the metric is a
    # pure response-token mean regardless of how many skill samples were appended.
    # (calculate_per_token_loss => sum_of_token returns a raw response-token SUM, so
    # dividing by the summed response-token count is exact under the mb/DP reduce.)
    # Whether this RUN uses skill-KD (a run-level flag, NOT per-microbatch). The
    # reported_loss key set must be identical across every microbatch, because
    # aggregate_train_losses sums the value tensors positionally — so the
    # __denom__entropy_loss key must be emitted for ALL microbatches of a skill-KD
    # run, even ones that happen to contain no skill samples (after interleaving,
    # some microbatches are pure-response). Keying it off the per-microbatch
    # skill_tok_mask made the key count vary (13 vs 14) and crashed the reduce.
    skill_kd_run = bool(getattr(args, "sdpo_skill_kd", False)) and args.calculate_per_token_loss
    entropy_denom = None
    loss = pg_loss
    if calculate_entropy:
        entropy = log_probs_and_entropy["entropy"]
        entropy = torch.cat(entropy, dim=0)
        if skill_tok_mask is not None:
            # entropy over response tokens only (skill tokens excluded), plus a
            # separate skill-token entropy metric.
            resp_ent = torch.where(skill_tok_mask, entropy.new_zeros(()), entropy)
            entropy_loss = sum_of_sample_mean(resp_ent)
            skill_entropy = sum_of_sample_mean(torch.where(skill_tok_mask, entropy, entropy.new_zeros(()))).detach()
            resp_tok_mask = (~skill_tok_mask).to(entropy.dtype)
        else:
            entropy_loss = sum_of_sample_mean(entropy)
            # No skill samples in this microbatch: every response token counts.
            resp_tok_mask = entropy.new_ones(entropy.shape)
        # Response-only denominator for the reported entropy metric, via the SAME
        # reducer as the numerator so cp/mb reduction matches. Emitted for every
        # microbatch of a skill-KD run (stable key set); None otherwise so non-skill
        # runs keep the original global-token normalization exactly.
        if skill_kd_run:
            entropy_denom = sum_of_sample_mean(resp_tok_mask)
        if args.entropy_coef != 0:
            loss = pg_loss - args.entropy_coef * entropy_loss
        else:
            entropy_loss = entropy_loss.detach()

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl = torch.where(
            active_tokens,
            torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0),
            kl.new_zeros(()),
        )
        kl_loss = sum_of_sample_mean(kl)

        if args.kl_loss_coef != 0:
            loss = loss + args.kl_loss_coef * kl_loss

    # SDPO distribution-level knowledge-distillation loss (grad through student):
    # loss += kd_coef * mean_token D(student_dist ‖ teacher_dist). This is a real
    # supervised distillation objective (strong, directional gradient), unlike the
    # advantage-hook path that fed the divergence into REINFORCE.
    sdpo_kd_loss = pg_loss.new_zeros(())
    sdpo_skill_kd_loss = pg_loss.new_zeros(())
    sdpo_skill_kd_loss_correct = pg_loss.new_zeros(())
    sdpo_skill_kd_loss_pitfall = pg_loss.new_zeros(())
    sdpo_kd_clip_cov_frac = pg_loss.new_zeros(())
    # Skill-KD is orthogonal to how the RESPONSE teacher signal is consumed
    # (additive KD loss here vs. --sdpo-rlsd's advantage reweighting in
    # loss_hub/rlsd.py): it always runs on the appended skill-tagged samples'
    # own top-k target when --sdpo-skill-kd is set, even under --sdpo-rlsd
    # (which never populates sdpo_teacher_topk_* for the response span --
    # actor.py seeds empty response entries so this divergence call still
    # produces a real skill-span target, response contributing 0).
    run_kd = getattr(args, "sdpo_kd_loss", False) or (
        getattr(args, "sdpo_skill_kd", False) and skill_tok_mask is not None
    )
    if run_kd:
        kd = _sdpo_kd_loss_per_token(args, logits, batch)
        if kd is not None:
            kd = torch.where(
                active_tokens,
                torch.nan_to_num(kd, nan=0.0, posinf=0.0, neginf=0.0),
                kd.new_zeros(()),
            )
            # KD Clip-Cov (arXiv:2505.22617, adapted to distillation): the entropy
            # collapse is driven by a few tokens with the LARGEST per-token KD
            # divergence (the '<answer>'/letter positions the answer-in-prefix
            # teacher over-weights). Detach the gradient on the top-`frac` of ACTIVE
            # tokens (batch-wide) so those positions no longer push the policy toward
            # a spike; the loss value is unchanged (kept for logging), only grad is cut.
            cc_frac = float(getattr(args, "sdpo_kd_clip_cov_frac", 0.0) or 0.0)
            if cc_frac > 0.0:
                with torch.no_grad():
                    active_kd = kd[active_tokens]
                    clip_mask = None
                    if active_kd.numel() > 0:
                        k = max(1, int(active_kd.numel() * cc_frac))
                        # threshold = k-th largest active KD divergence value
                        thresh = torch.topk(active_kd, k, largest=True).values.min()
                        clip_mask = active_tokens & (kd >= thresh)
                        sdpo_kd_clip_cov_frac = clip_mask.float().sum() / active_tokens.float().sum().clamp(min=1.0)
                if clip_mask is not None:
                    # Freeze gradient on clipped tokens: keep value, drop grad.
                    kd = torch.where(clip_mask, kd.detach(), kd)

            # Skill-KD (option A): skill sequences are appended to the batch tagged
            # sdpo_is_skill. Split the per-token KD into response vs skill and weight
            # them by their own coefficients (skill uses --sdpo-skill-kd-coef). Reuse
            # the per-token skill mask built above (for the entropy split).
            if skill_tok_mask is not None:
                resp_kd = torch.where(skill_tok_mask, kd.new_zeros(()), kd)
                skill_kd = torch.where(skill_tok_mask, kd, kd.new_zeros(()))
                sdpo_kd_loss = sum_of_sample_mean(resp_kd)
                sdpo_skill_kd_loss = sum_of_sample_mean(skill_kd)
                loss = loss + getattr(args, "sdpo_kd_coef", 1.0) * sdpo_kd_loss
                loss = loss + getattr(args, "sdpo_skill_kd_coef", 1.0) * sdpo_skill_kd_loss
                # Further split skill_kd by provenance (correct trace -> self-success/
                # blind-correct vs failed trace -> pitfall-condense), diagnostic only
                # (not separate loss terms -- both already summed into
                # sdpo_skill_kd_loss above with the SAME coefficient; this just tells
                # you which half is contributing). See skill_correct_tok_mask's setup.
                if skill_correct_tok_mask is not None:
                    correct_kd = torch.where(skill_correct_tok_mask, skill_kd, skill_kd.new_zeros(()))
                    pitfall_kd = torch.where(
                        skill_tok_mask & ~skill_correct_tok_mask, skill_kd, skill_kd.new_zeros(())
                    )
                    sdpo_skill_kd_loss_correct = sum_of_sample_mean(correct_kd)
                    sdpo_skill_kd_loss_pitfall = sum_of_sample_mean(pitfall_kd)
            else:
                sdpo_kd_loss = sum_of_sample_mean(kd)
                loss = loss + getattr(args, "sdpo_kd_coef", 1.0) * sdpo_kd_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_scored_log_probs = old_log_probs
    train_rollout_logprob_abs_diff = None
    train_rollout_kl = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        abs_diff = (train_scored_log_probs - rollout_log_probs).abs()
        abs_diff = torch.where(
            active_tokens,
            torch.nan_to_num(abs_diff, nan=0.0, posinf=0.0, neginf=0.0),
            abs_diff.new_zeros(()),
        )
        train_rollout_logprob_abs_diff = sum_of_sample_mean(abs_diff)

        # KL(rollout || train) at sampled tokens via Schulman k3 with per-token clamp [-10, 10]
        rollout_train_kl = compute_approx_kl(rollout_log_probs, train_scored_log_probs, kl_loss_type="low_var_kl")
        rollout_train_kl = torch.where(
            active_tokens,
            torch.nan_to_num(rollout_train_kl, nan=0.0, posinf=0.0, neginf=0.0),
            rollout_train_kl.new_zeros(()),
        )
        train_rollout_kl = sum_of_sample_mean(rollout_train_kl)

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
        "ess_ratio": ess_ratio_sum.squeeze(),
    }

    # Per-key denominator for entropy_loss (see the entropy block): a "__denom__<key>"
    # entry tells aggregate_train_losses to divide <key> by this summed count instead
    # of the batch-wide token count, so train/entropy_loss stays a response-only mean.
    if entropy_denom is not None:
        reported_loss["__denom__entropy_loss"] = entropy_denom.clone().detach()

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()
    if train_rollout_kl is not None:
        reported_loss["train_rollout_kl"] = train_rollout_kl.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if getattr(args, "sdpo_kd_loss", False):
        reported_loss["sdpo_kd_loss"] = sdpo_kd_loss.clone().detach()
        if float(getattr(args, "sdpo_kd_clip_cov_frac", 0.0) or 0.0) > 0.0:
            reported_loss["sdpo_kd_clip_cov_frac"] = (
                sdpo_kd_clip_cov_frac.clone().detach()
                if isinstance(sdpo_kd_clip_cov_frac, torch.Tensor)
                else sdpo_kd_clip_cov_frac
            )
    # Skill-KD's own report is gated on the run-level --sdpo-skill-kd flag
    # ALONE, not on run_kd -- run_kd also requires skill_tok_mask is not None,
    # which is computed PER MICROBATCH (None whenever that microbatch happens
    # to contain zero skill-tagged samples, e.g. under dynamic microbatching).
    # Gating reported_loss's key set on run_kd made these 4 keys appear only
    # on microbatches that happened to have a skill sample, so
    # aggregate_train_losses' positional sum across microbatches saw
    # mismatched tensor lengths and crashed ("size of tensor a (13) must
    # match size of tensor b (18)") -- observed live, deterministically, on
    # every arm-5/5.1 attempt. sdpo_skill_kd_loss/_correct/_pitfall are
    # zero-initialized above regardless of skill_tok_mask, so gating on the
    # plain flag (a true run-level condition) is safe: microbatches without
    # skill samples just report 0 for these keys, exactly like
    # entropy_denom's "stable key set" fix above.
    if getattr(args, "sdpo_skill_kd", False):
        reported_loss["sdpo_skill_kd_loss"] = sdpo_skill_kd_loss.clone().detach()
        # dedicated skill/ panel: skill KD (= skill "kl") and skill-token entropy.
        reported_loss["skill/kl"] = sdpo_skill_kd_loss.clone().detach()
        reported_loss["skill/entropy"] = (
            skill_entropy.clone().detach() if isinstance(skill_entropy, torch.Tensor) else skill_entropy
        )
        # skill/kl split by provenance: correct-trace (self-success/blind-correct)
        # vs failed-trace (pitfall-condense) contribution to skill/kl above.
        # Emitted unconditionally whenever skill-KD runs (a run-level condition,
        # same as the block above) so the reported_loss key set stays IDENTICAL
        # across every microbatch of a run -- microbatches with no correct-trace
        # (or no failed-trace) skill samples this step just report 0 for that
        # half, mirroring entropy_denom's own "stable key set" reasoning above.
        reported_loss["skill/kl_correct"] = sdpo_skill_kd_loss_correct.clone().detach()
        reported_loss["skill/kl_pitfall"] = sdpo_skill_kd_loss_pitfall.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    # Add OPD metrics if available
    if batch.get("opd_reverse_kl") is not None:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def get_loss_function(args: Namespace) -> LossFunction:
    match args.loss_type:
        case "policy_loss":
            return policy_loss_function
        case "value_loss":
            return value_loss_function
        case "sft_loss":
            return sft_loss_function
        case "custom_loss":
            return load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")
