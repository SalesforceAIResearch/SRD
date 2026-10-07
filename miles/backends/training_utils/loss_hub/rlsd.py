from argparse import Namespace

import torch

from miles.utils.types import RolloutBatch


def _rlsd_lambda(args: Namespace, rollout_id: int | None) -> float:
    """Linearly decay lambda from --sdpo-rlsd-lambda-init to 0 over
    --sdpo-rlsd-lambda-warmup-steps rollouts (matches the RLSD paper's
    lambda: 0.5 -> 0 over the first 50 steps), so training settles into plain
    GRPO rather than an abrupt on/off transition. rollout_id=None (e.g. unit
    tests that don't thread a step counter through) keeps lambda at its init
    value."""
    lam0 = float(getattr(args, "sdpo_rlsd_lambda_init", 0.5))
    warmup = int(getattr(args, "sdpo_rlsd_lambda_warmup_steps", 50) or 0)
    if rollout_id is None or warmup <= 0:
        return lam0
    frac = min(1.0, rollout_id / warmup)
    return lam0 * (1.0 - frac)


def apply_rlsd_credit_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
    rollout_id: int | None = None,
) -> None:
    """RLSD (arXiv:2604.03128, "Self-Distilled RLVR"): multiplicatively
    reweight the GRPO advantage by the self-teacher's per-token evidence
    ratio, instead of an additive KD loss (--sdpo-kd-loss) or KL penalty
    (--use-opd). Direction is anchored EXCLUSIVELY to sign(A) (the
    environment reward); the teacher only modulates magnitude within a
    trajectory, so a WRONG trace can never be pulled toward tokens the
    teacher favors -- the privileged-information-leakage failure mode the
    paper identifies in plain distribution-matching (OPSD/SDPO's KD loss).

        delta_t  = sg(log P_T(y_t) - log P_S(y_t))
        w_t      = exp(sign(A) * delta_t) = (P_T(y_t) / P_S(y_t)) ** sign(A)
        credit_t = (1-lambda) + lambda * clip(w_t, 1-eps_w, 1+eps_w)
        A_hat_t  = A * credit_t

    P_T/P_S are both computed from the SAME (current) weights under
    forward_only (no grad) -- teacher via SDPO's Megatron self-teacher
    forward over prompt+prefix+response (rollout_data["teacher_log_probs"],
    requires --sdpo-teacher-backend megatron + --sdpo-logprob-mode sampled),
    student via the plain prompt+response forward already in
    `student_log_probs` -- so delta_t needs no explicit stop-gradient, both
    inputs are already detached.

    Off-policy/async correction for train-vs-rollout log-prob drift (student
    forward here vs the policy that actually SAMPLED the rollout) is NOT done
    here -- that is exactly what --use-tis / vanilla_tis_function already
    computes (tis = exp(train_log_probs - rollout_log_probs), multiplied into
    pg_loss downstream in policy_loss_function). RLSD's own delta_t/credit_t
    is a separate, orthogonal factor on the advantage; adding a second IS
    term here would double-apply the same train-vs-rollout drift correction
    once --use-tis is also enabled. Use --use-tis for that axis instead.
    """
    if student_log_probs is None:
        return

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError(
            "--sdpo-rlsd requires rollout_data['teacher_log_probs'] "
            "(set --sdpo-teacher-backend megatron --sdpo-logprob-mode sampled)."
        )

    if not (len(advantages) == len(student_log_probs) == len(teacher_log_probs)):
        raise ValueError(
            f"RLSD length mismatch: advantages={len(advantages)}, "
            f"student_log_probs={len(student_log_probs)}, teacher_log_probs={len(teacher_log_probs)}."
        )

    device = student_log_probs[0].device
    teacher_log_probs = [t.to(device=device) for t in teacher_log_probs]

    eps_w = float(getattr(args, "sdpo_rlsd_clip_eps", 0.2))
    lam = _rlsd_lambda(args, rollout_id)

    credits = []
    for i, adv in enumerate(advantages):
        s_lp = student_log_probs[i]
        t_lp = teacher_log_probs[i]
        if adv.shape != s_lp.shape or adv.shape != t_lp.shape:
            raise ValueError(
                f"RLSD shape mismatch at sample {i}: advantages={tuple(adv.shape)}, "
                f"student_log_probs={tuple(s_lp.shape)}, teacher_log_probs={tuple(t_lp.shape)}. "
                "RLSD expects per-token advantages; broadcast scalar advantages must be expanded before this call."
            )
        delta = (t_lp - s_lp).detach()
        sign_a = torch.sign(adv)
        w = torch.exp(sign_a * delta)
        w_clipped = torch.clamp(w, min=1.0 - eps_w, max=1.0 + eps_w)
        credit = (1.0 - lam) + lam * w_clipped

        advantages[i] = adv * credit
        credits.append(credit)

    rollout_data["sdpo_rlsd_credit"] = credits
    rollout_data["sdpo_rlsd_lambda"] = [lam] * len(credits)
