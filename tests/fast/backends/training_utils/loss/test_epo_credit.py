"""Unit tests for EPO's credit-weighted GRPO advantage fusion.

EPO decouples SDPO's teacher-direction from its density weight: the GRPO
advantage (reward - baseline) provides the DIRECTION, and the per-token PMI
credit_t (computed by MegatronTrainRayActor._compute_epo_credit) provides the
DENSITY, fused multiplicatively in compute_advantages_and_returns. These tests
cover the fusion math and its guard rails without needing GPUs/Megatron.
"""

from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.loss import compute_advantages_and_returns
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

# This module intentionally has no explicit CI registration call: modules under
# tests/fast are implicitly assigned to the stage-a-cpu suite by the CI collector
# (an explicit default-form call would be rejected by the AC-9 meta-test).


@pytest.fixture(autouse=True)
def _trivial_parallel_state():
    def trivial() -> GroupInfo:
        return GroupInfo(rank=0, size=1, group=None)

    state = ParallelState(
        intra_dp=trivial(),
        intra_dp_cp=trivial(),
        cp=trivial(),
        tp=trivial(),
        pp=trivial(),
        ep=trivial(),
        etp=trivial(),
        is_pp_last_stage=True,
    )
    set_parallel_state(state)
    yield state


def _args(epo_credit_loss: bool = True, **overrides) -> Namespace:
    d = dict(
        advantage_estimator="grpo",
        use_rollout_logprobs=False,
        kl_coef=0.0,
        kl_loss_type="k1",
        use_opd=False,
        normalize_advantages=False,
        epo_credit_loss=epo_credit_loss,
    )
    d.update(overrides)
    return Namespace(**d)


def _rollout_data(epo_credit=None) -> dict:
    d = {
        "log_probs": [torch.tensor([-1.0, -2.0]), torch.tensor([-0.5, -0.5, -0.5])],
        "ref_log_probs": None,
        "rewards": [1.0, 0.0],
        "values": None,
        "response_lengths": [2, 3],
        "loss_masks": [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0, 1.0])],
        "total_lengths": [10, 12],
    }
    if epo_credit is not None:
        d["epo_credit"] = epo_credit
    return d


def test_fuses_credit_into_grpo_advantage_elementwise():
    args = _args(epo_credit_loss=True)
    rollout_data = _rollout_data(
        epo_credit=[torch.tensor([2.0, 0.5]), torch.tensor([1.0, 1.0, 1.0])],
    )

    compute_advantages_and_returns(args, rollout_data)

    # sample 0: reward=1.0 broadcast, credit=[2.0, 0.5] -> advantages=[2.0, 0.5]
    assert torch.allclose(rollout_data["advantages"][0], torch.tensor([2.0, 0.5]))
    # sample 1: reward=0.0 broadcast, credit=[1,1,1] -> advantages stay 0
    assert torch.allclose(rollout_data["advantages"][1], torch.tensor([0.0, 0.0, 0.0]))


def test_noop_when_epo_credit_loss_disabled():
    args = _args(epo_credit_loss=False)
    rollout_data = _rollout_data(
        epo_credit=[torch.tensor([2.0, 0.5]), torch.tensor([1.0, 1.0, 1.0])],
    )

    compute_advantages_and_returns(args, rollout_data)

    # Plain GRPO: advantages == broadcast reward, credit is ignored entirely.
    assert torch.allclose(rollout_data["advantages"][0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(rollout_data["advantages"][1], torch.tensor([0.0, 0.0, 0.0]))


def test_noop_when_epo_credit_key_missing():
    # epo_credit_loss=True but the training-side credit computation never ran
    # (e.g. no privileged-context peer was available anywhere in the rollout):
    # falls back to plain GRPO rather than crashing.
    args = _args(epo_credit_loss=True)
    rollout_data = _rollout_data(epo_credit=None)

    compute_advantages_and_returns(args, rollout_data)

    assert torch.allclose(rollout_data["advantages"][0], torch.tensor([1.0, 1.0]))


def test_raises_on_length_mismatch():
    args = _args(epo_credit_loss=True)
    rollout_data = _rollout_data(
        # 2 credit tensors but the batch mismatch is engineered via response_lengths/log_probs
        # trimmed to 1 sample below.
        epo_credit=[torch.tensor([2.0, 0.5]), torch.tensor([1.0])],
    )
    rollout_data["log_probs"] = [torch.tensor([-1.0, -2.0])]
    rollout_data["rewards"] = [1.0]
    rollout_data["response_lengths"] = [2]
    rollout_data["loss_masks"] = [torch.tensor([1.0, 1.0])]
    rollout_data["total_lengths"] = [10]

    with pytest.raises(AssertionError, match="EPO credit length mismatch"):
        compute_advantages_and_returns(args, rollout_data)
