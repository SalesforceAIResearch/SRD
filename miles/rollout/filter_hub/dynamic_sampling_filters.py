import torch

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

__all__ = ["check_reward_nonzero_std", "check_no_aborted", "check_sdpo_group_has_prefix"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    # A None reward means the sample was never scored (e.g. Sample.Status.ABORTED --
    # confirmed live: a tau2 sidecar crash mid-run left every in-flight sample without
    # a reward, and torch.tensor() on a list containing None raises TypeError, which
    # previously took down the whole training job instead of just dropping the group).
    if any(r is None for r in rewards):
        return DynamicFilterOutput(keep=False, reason="reward_none_aborted")
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-8
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def check_sdpo_group_has_prefix(args, samples: list[Sample], **kwargs):
    """SDPO dynamic-sampling filter: keep a group only if EVERY trace in it can be
    given a correct-peer prefix.

    SDPO builds each trace's teacher prefix from a *different* correct peer in the
    same group (``peers = [j for j in correct_indices if j != i]``; see
    ``examples/SRD/sdpo.py``). A trace gets no prefix — so its teacher == student
    and its KD/KL loss is exactly 0 — whenever that pool is empty. For EVERY trace
    to have a prefix, the group needs at least 2 correct traces (with exactly 1
    correct, that lone correct trace self-excludes to an empty pool). Groups below
    the threshold contribute dead (zero-gradient) samples under ``--sdpo-pure-distill``,
    so we drop and re-sample them (DAPO-style) instead of wasting the batch slot.

    Requires the SDPO group RM (``sdpo_group_reward``), which stamps
    ``sample.metadata["sdpo_correct"]`` (1.0/0.0) before this filter runs. The
    threshold is ``args.sdpo_dynamic_filter_min_correct`` (default 0 = keep every
    group, i.e. inert; set 2 for full prefix coverage).
    """
    min_correct = int(getattr(args, "sdpo_dynamic_filter_min_correct", 0))

    # Threshold 0 disables the filter entirely (keep every group). Return before
    # touching metadata so the no-op path never depends on the SDPO group RM.
    if min_correct <= 0:
        return DynamicFilterOutput(keep=True)

    flat = list(_flatten_samples(samples))
    graded = [s for s in flat if isinstance(s.metadata, dict) and "sdpo_correct" in s.metadata]
    if not graded:
        raise ValueError(
            "check_sdpo_group_has_prefix requires sample.metadata['sdpo_correct'], which is set "
            "by the SDPO group RM (examples.SRD.sdpo.sdpo_group_reward). Set --custom-rm-path to "
            "it (and --group-rm), or use a different --dynamic-sampling-filter-path."
        )

    num_correct = sum(1 for s in graded if float(s.metadata.get("sdpo_correct", 0.0)) > 0.5)
    keep = num_correct >= min_correct
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"sdpo_lt_{min_correct}_correct_{num_correct}",
    )


def _flatten_samples(samples):
    """Flatten samples that may contain nested lists (from --generate-multi-samples)."""
    for s in samples:
        if isinstance(s, list):
            yield from s
        else:
            yield s


def check_no_aborted(args, samples: list[Sample], **kwargs):
    """Reject entire group if any sample was aborted (e.g. env timeout, Docker crash)."""
    if any(s.status == Sample.Status.ABORTED for s in _flatten_samples(samples)):
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    return DynamicFilterOutput(keep=True)
