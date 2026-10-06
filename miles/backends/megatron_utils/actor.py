import logging
import math
import random
import socket
from pathlib import Path
from argparse import Namespace
from contextlib import nullcontext
from typing import TYPE_CHECKING

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from torch_memory_saver import torch_memory_saver

from miles.ray.train_actor import TrainRayActor
from miles.utils import train_dump_utils
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group, init_process_group
from miles.utils.hf_config import load_hf_config
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.processing_utils import load_tokenizer
from miles.utils.ray_utils import Box
from miles.utils.reloadable_process_group import (
    destroy_process_groups,
    monkey_patch_dist_checkpointing_merge,
    monkey_patch_torch_dist,
    reload_process_groups,
)
from miles.utils.replay_base import all_replay_managers, routing_replay_manager
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils import init_tracking
from miles.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from ..training_utils.data import DataIterator, get_data_iterator, get_rollout_data, sync_actor_critic_data
from ..training_utils.log_utils import log_cpu_memory, log_perf_data, log_rollout_data
from ..training_utils.loss import (
    compute_advantages_and_returns,
    get_log_probs_and_entropy,
    get_topk_logprobs,
    get_values,
)
from ..training_utils.parallel import get_parallel_state
from ..training_utils.replay_data import fill_replay_data, register_replay_list_sequential
from .checkpoint import load_checkpoint
from .initialize import init, is_megatron_main_rank
from .lora_utils import is_lora_enabled
from .model import forward_only, initialize_model_and_optimizer, save, train
from .parallel import verify_megatron_parallel_state
from .replay_utils import register_replay_list_moe
from .update_weight.common import named_params_and_buffers
from .update_weight.update_weight_from_distributed.broadcast import UpdateWeightFromDistributed
from .update_weight.update_weight_from_distributed.p2p import UpdateWeightP2P
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

if TYPE_CHECKING:
    from miles.ray.rollout.rollout_manager import EnginesAndLock

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _sdpo_topk_distribution_divergence(
    s_lp: torch.Tensor,  # [R, k] student top-k log-probs
    s_ids: torch.Tensor,  # [R, k] student top-k token ids
    t_lp: torch.Tensor,  # [R, kt] teacher top-k log-probs
    t_ids: torch.Tensor,  # [R, kt] teacher top-k token ids
    mode: str,
) -> torch.Tensor:
    """Per-token distribution-level divergence over the student's top-k tokens
    plus one aggregated tail bucket, matching the original SGLang SDPO recipe.

    student prob p_s = exp(student top-k log-probs), tail_s = 1 - sum(p_s).
    teacher prob p_t at those SAME ids: looked up from the teacher's own top-k;
    ids the teacher didn't rank get ~0 (folded into tail_t = 1 - sum(found)).
    All CPU float64 tensors (small, [R, k]); returns [R] float32.
    """
    eps = 1e-12
    s_lp = s_lp.double()
    p_s = torch.exp(s_lp)  # [R, k]
    R, k = p_s.shape
    if R == 0:
        return torch.zeros((0,), dtype=torch.float32)

    # Build teacher prob at the student ids via a per-row lookup: for each student
    # id, find it in the teacher's top-k ids; missing -> prob 0.
    t_lp = t_lp.double()
    p_t = torch.zeros((R, k), dtype=torch.float64)
    for r in range(R):
        tmap = {int(tid): float(tlp) for tid, tlp in zip(t_ids[r].tolist(), t_lp[r].tolist())}
        for c in range(k):
            sid = int(s_ids[r, c])
            if sid in tmap:
                p_t[r, c] = math.exp(tmap[sid])

    tail_s = (1.0 - p_s.sum(dim=1)).clamp_min(0.0).unsqueeze(1)
    tail_t = (1.0 - p_t.sum(dim=1)).clamp_min(0.0).unsqueeze(1)
    P = torch.cat([p_s, tail_s], dim=1)  # [R, k+1]
    Q = torch.cat([p_t, tail_t], dim=1)

    if mode == "reverse_kl":
        div = (P * torch.log((P + eps) / (Q + eps))).sum(dim=1)
    elif mode == "forward_kl":
        div = (Q * torch.log((Q + eps) / (P + eps))).sum(dim=1)
    elif mode == "jeffrey":
        div = (P * torch.log((P + eps) / (Q + eps))).sum(dim=1) + (Q * torch.log((Q + eps) / (P + eps))).sum(dim=1)
    else:  # jsd
        M = 0.5 * (P + Q)
        div = 0.5 * (P * torch.log((P + eps) / (M + eps))).sum(dim=1) + 0.5 * (
            Q * torch.log((Q + eps) / (M + eps))
        ).sum(dim=1)
    return div.to(torch.float32)


class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        monkey_patch_torch_dist()
        monkey_patch_dist_checkpointing_merge()

        super().init(args, role, with_ref, with_opd_teacher=with_opd_teacher)

        for m in all_replay_managers:
            m.register_replay_list_func = register_replay_list_sequential
        routing_replay_manager.register_replay_list_func = register_replay_list_moe

        init(args)

        if args.dumper_enable:
            from sglang.srt.debug_utils.dumper import dumper

            dumper.apply_source_patches()

        self._is_main_rank = is_megatron_main_rank()

        if self._is_main_rank:
            init_tracking(args, primary=False)

        unsupported = {"train_actor", "train_log_probs"} & set(args.profile_target)
        if unsupported and args.use_pytorch_profiler:
            raise NotImplementedError(
                f"--profile-target {' '.join(sorted(unsupported))} is not supported for Megatron backend"
            )
        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = load_hf_config(args.hf_checkpoint)
                self.tokenizer = load_tokenizer(
                    self.args.hf_checkpoint, chat_template_path=self.args.chat_template_path, trust_remote_code=True
                )
            dist.barrier(group=get_gloo_group())

        self.train_parallel_config = {
            "dp_size": get_parallel_state().intra_dp.size,
        }
        dist.barrier(group=get_gloo_group())

        if args.offload_train:
            if (x := args.train_memory_margin_bytes) > 0:
                # --train-memory-margin-bytes can tune this
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x

        if self.args.debug_rollout_only:
            return 0

        if role == "critic":
            self.args.load = self.args.critic_load
            self.args.save = self.args.critic_save
            self.args.lr = self.args.critic_lr
            self.args.lr_warmup_iters = self.args.critic_lr_warmup_iters
        else:
            for m in all_replay_managers:
                m.enabled = getattr(self.args, f"use_{m.name}_replay", False)
                m.enable_check_replay_result = m.enabled and self.args.ci_test

        self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id = initialize_model_and_optimizer(
            args, role
        )

        parallel_state = get_parallel_state()
        if parallel_state.cp.size > 1:
            from miles_plugins.models.cp_utils import detect_and_setup_hybrid_cp

            for model_chunk in self.model:
                detect_and_setup_hybrid_cp(
                    model_chunk, parallel_state.cp.group, parallel_state.cp.rank, parallel_state.cp.size
                )

        verify_megatron_parallel_state(self.model)

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return

        start_rollout_id = loaded_rollout_id + 1

        self.weights_backuper = TensorBackuper.create(
            source_getter=lambda: named_params_and_buffers(
                self.args,
                self.model,
                convert_to_global_name=args.megatron_to_hf_mode == "raw",
                translate_gpu_to_cpu=not self.args.enable_weights_backuper,
            ),
            single_tag=None if args.enable_weights_backuper else "actor",
        )
        self._active_model_tag: str | None = "actor"
        if self._enable_weight_backup:
            self.weights_backuper.backup("actor")

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        # Load teacher model for Megatron-based on-policy distillation
        if with_opd_teacher:
            self.load_other_checkpoint("teacher", args.opd_teacher_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        # SDPO/EPO/RLSD EMA teacher: initialize a "sdpo_teacher" weight snapshot =
        # current (initial) actor weights. Each train step it is EMA-blended toward
        # the student (see _update_sdpo_ema_teacher), and the SDPO/EPO/RLSD teacher
        # forward runs against this slow copy instead of the live policy.
        # _enable_weight_backup is True whenever sdpo_ema_teacher is set together
        # with KD-loss, EPO's credit_t, or RLSD (see the property), so this works
        # even with KL off (no ref model).
        if getattr(self.args, "sdpo_ema_teacher", False) and (
            getattr(self.args, "sdpo_kd_loss", False)
            or getattr(self.args, "epo_credit_loss", False)
            or getattr(self.args, "sdpo_rlsd", False)
        ):
            assert self._enable_weight_backup, (
                "--sdpo-ema-teacher needs the weight backuper active (it should be, via the "
                "_enable_weight_backup property)."
            )
            assert self.args.enable_weights_backuper, (
                "--sdpo-ema-teacher needs the multi-tag weight backuper; do not pass "
                "--disable-weights-backuper (the single-tag backuper cannot hold a "
                "separate EMA-teacher snapshot)."
            )
            self.weights_backuper.backup("sdpo_teacher")

        if self.args.vocab_size is None:
            self.args.vocab_size = self.tokenizer.vocab_size

        if self.args.colocate:
            update_weight_cls = UpdateWeightFromTensor
        else:
            if self.args.update_weight_transfer_mode == "broadcast":
                update_weight_cls = UpdateWeightFromDistributed
            elif self.args.update_weight_transfer_mode == "disk-delta":
                # Lazy import: keeps the delta deps (numpy/zstandard/xxhash) off the other paths.
                from .update_weight.update_weight_from_distributed.delta import UpdateWeightFromDiskDelta

                update_weight_cls = UpdateWeightFromDiskDelta
            else:
                update_weight_cls = UpdateWeightP2P
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
            is_lora=is_lora_enabled(args),
        )

        # empty cache after initialization
        clear_memory()

        self._switch_model("actor")
        if self.args.offload_train:
            self.sleep()

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if (x := self.args.rollout_data_postprocess_path) is not None:
            from miles.utils.misc import load_function

            self.rollout_data_postprocess = load_function(x)

        self.prof.on_init_end()

        return start_rollout_id

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        destroy_process_groups()

        tag = "default" if is_lora_enabled(self.args) else None
        torch_memory_saver.pause(tag=tag)

        print_memory("after offload model")

        if self._is_main_rank and hasattr(self, "_last_rollout_id"):
            log_cpu_memory(self._last_rollout_id, self.args, "after_offload_train")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        tag = "default" if is_lora_enabled(self.args) else None
        torch_memory_saver.resume(tag=tag)

        clear_memory()
        reload_process_groups()
        print_memory("after wake_up model")

    @property
    def _enable_weight_backup(self) -> bool:
        """Weight backup is only needed for CPU-side model switching or colocated tensor weight sync."""
        return (
            self.with_ref
            or self.with_opd_teacher
            or self.args.keep_old_actor
            or self.args.colocate
            # SDPO/EPO/RLSD EMA teacher keeps a separate weight snapshot that must
            # be swapped in for the teacher forward, so it needs the backuper even
            # without a ref. EPO's credit_t and RLSD's evidence-ratio forward both
            # reuse this same snapshot.
            or (
                getattr(self.args, "sdpo_ema_teacher", False)
                and (
                    getattr(self.args, "sdpo_kd_loss", False)
                    or getattr(self.args, "epo_credit_loss", False)
                    or getattr(self.args, "sdpo_rlsd", False)
                )
            )
        )

    def _switch_model(self, target_tag: str) -> None:
        if not self._enable_weight_backup:
            return
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def _set_replay_stage(self, stage: str) -> None:
        for m in all_replay_managers:
            m.stage = stage

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        self._last_rollout_id = rollout_id
        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = get_rollout_data(self.args, rollout_data_ref)
            if self.args.debug_rollout_only:
                log_rollout_data(rollout_id, self.args, rollout_data)
                return

        if self.role == "critic":
            return self.train_critic(rollout_id, rollout_data)
        else:
            return self.train_actor(rollout_id, rollout_data)

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        rollout_data.update(
            forward_only(
                get_values,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
            )
        )

        if rollout_id >= self.args.num_critic_only_steps:
            sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)

        compute_advantages_and_returns(self.args, rollout_data, rollout_id=rollout_id)

        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
        )

    def _use_rollout_replay(self, m) -> bool:
        return getattr(self.args, f"use_rollout_{m.name}_replay", False)

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        # Remember the pre-skill-append sample count; skill-KD appends extra samples
        # later (in _compute_sdpo_teacher_log_probs) and we must rebuild the iterator.
        _orig_num_samples = len(rollout_data["tokens"])

        for m in all_replay_managers:
            if self._use_rollout_replay(m):
                fill_replay_data(
                    args=self.args,
                    models=self.model,
                    data_iterator=data_iterator,
                    num_microbatches=num_microbatches,
                    rollout_data=rollout_data,
                    data_key=m.data_key,
                    replay_list=m.replays,
                    register_replay_list_func=m.register_replay_list_func,
                    if_sp_region=m.if_sp_region,
                    indices_are_token_positions=m.replay_indices_are_token_positions,
                )

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    self._set_replay_stage("fallthrough")
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )
                # Forward teacher model to get teacher_log_probs for Megatron-based OPD
                if "teacher" in self.weights_backuper.backup_tags:
                    self._set_replay_stage("fallthrough")
                    self._switch_model("teacher")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="teacher_",
                        )
                    )
                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                    for m in all_replay_managers:
                        if m.enabled:
                            if self._use_rollout_replay(m):
                                m.stage = "replay_forward"
                            else:
                                m.stage = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="",
                        )
                    )
                    for m in all_replay_managers:
                        if self._use_rollout_replay(m):
                            m.clear_all_forward()

                # SDPO megatron self-teacher: forward the policy over
                # prompt+prefix+response to get teacher log-probs on the response
                # span (aligned with the student's, since the response stays at
                # the tail). With --sdpo-ema-teacher, switch to the slow EMA weight
                # snapshot for this forward (matches lasgroup/SDPO), then switch
                # back to the live actor; otherwise use the live actor weights.
                if "sdpo_teacher_prompt_tokens" in rollout_data:
                    use_ema = getattr(self.args, "sdpo_ema_teacher", False) and (
                        "sdpo_teacher" in self.weights_backuper.backup_tags
                    )
                    if use_ema:
                        # DIAG: verify the EMA teacher weights actually differ from the
                        # live student before switching (else EMA is a no-op = live teacher).
                        if self._is_main_rank:
                            try:
                                import torch as _t

                                live = next(iter(self.weights_backuper.get("actor").values()))
                                ema = next(iter(self.weights_backuper.get("sdpo_teacher").values()))
                                diff = (live.float() - ema.float()).abs().mean().item()
                                logger.info(
                                    f"[SDPO-EMA-DIAG] rollout {getattr(self, '_last_rollout_id', '?')}: "
                                    f"mean|actor-ema_teacher|={diff:.3e} (0 => EMA is a no-op / same as live)"
                                )
                            except Exception as e:
                                logger.warning(f"[SDPO-EMA-DIAG] failed: {e!r}")
                        self._set_replay_stage("fallthrough")
                        self._switch_model("sdpo_teacher")
                    if getattr(self.args, "epo_credit_loss", False):
                        self._compute_epo_credit(rollout_data)
                    else:
                        self._compute_sdpo_teacher_log_probs(rollout_data)
                    if use_ema:
                        self._switch_model("actor")

                if self.args.use_critic:
                    sync_actor_critic_data(
                        self.args,
                        rollout_data,
                        self._actor_critic_groups,
                    )
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data, rollout_id=rollout_id)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args)

            log_rollout_data(rollout_id, self.args, rollout_data)

            # Skill-KD (option A) appended skill samples to rollout_data inside
            # _compute_sdpo_teacher_log_probs, AFTER the data_iterator above was
            # built. Rebuild the iterator so the training forward actually covers
            # the appended skill samples (otherwise skill KD is silently 0).
            if rollout_data.get("sdpo_is_skill") and len(rollout_data["tokens"]) != _orig_num_samples:
                data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

            # Train
            self._set_replay_stage("replay_backward")
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        for m in all_replay_managers:
            if m.enabled:
                m.clear_all()

        # update the cpu actor weight to the latest model
        if self._enable_weight_backup:
            self.weights_backuper.backup("actor")
        else:
            torch.cuda.synchronize()

        # SDPO EMA teacher update: blend the teacher snapshot toward the freshly
        # trained student (teacher = (1-rate)*teacher + rate*student). The live
        # model still holds the student weights here, so this reads the source
        # directly. Done after the actor backup so ordering is unambiguous.
        if (
            getattr(self.args, "sdpo_ema_teacher", False)
            and "sdpo_teacher" in self.weights_backuper.backup_tags
        ):
            # DIAG: measure how much the EMA teacher moves this step (should be small
            # but non-zero; 0 => update is a no-op).
            _ema_before = None
            if self._is_main_rank:
                try:
                    _ema_before = next(iter(self.weights_backuper.get("sdpo_teacher").values())).clone()
                except Exception:
                    _ema_before = None
            self.weights_backuper.ema_update_from_source(
                "sdpo_teacher", float(getattr(self.args, "sdpo_ema_teacher_rate", 0.05))
            )
            if self._is_main_rank and _ema_before is not None:
                try:
                    _ema_after = next(iter(self.weights_backuper.get("sdpo_teacher").values()))
                    moved = (_ema_after.float() - _ema_before.float()).abs().mean().item()
                    logger.info(
                        f"[SDPO-EMA-DIAG] rollout {rollout_id}: EMA teacher moved "
                        f"mean|Δ|={moved:.3e} this step (0 => update no-op)"
                    )
                except Exception as e:
                    logger.warning(f"[SDPO-EMA-DIAG] update check failed: {e!r}")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        log_perf_data(rollout_id, self.args, extra_metrics=self.weight_updater.pop_metrics())

    def _compute_sdpo_teacher_log_probs(self, rollout_data: RolloutBatch) -> None:
        """SDPO self-teacher via the current policy (Megatron forward).

        For each sample, build the teacher sequence prompt+prefix+response by
        inserting the correct-peer prefix tokens between the prompt and the
        response. The response stays at the tail, so the response-span log-probs
        this forward returns are token-aligned with the student's (which were
        computed over prompt+response). Samples with an empty prefix get teacher
        log-probs equal to the student's (zero KL). Results are written to
        rollout_data["teacher_log_probs"], consumed by opd.py (additive KL-in-
        advantage) or, when --sdpo-rlsd is set, by loss_hub/rlsd.py (multiplicative
        advantage reweighting) -- exactly like the Megatron-OPD path.

        This replaces SGLang HTTP full-sequence-logprob scoring (which forces
        eager prefill) with a single batched, CUDA-graph'd forward — the same
        approach as veRL's RefWorker, ~50x faster.
        """
        response_lengths = rollout_data["response_lengths"]
        student_log_probs = rollout_data.get("log_probs")

        # Build the prefix-augmented teacher rollout_data (prompt+prefix+response).
        teacher_rollout_data, has_prefix = self._build_sdpo_teacher_rollout_data(rollout_data)
        teacher_iter, teacher_nmb = get_data_iterator(self.args, self.model, teacher_rollout_data)

        distribution_mode = getattr(self.args, "sdpo_logprob_mode", "topk") == "topk"

        if not distribution_mode:
            # Sampled-token reverse KL: teacher log-prob of the sampled tokens.
            # Also the path --sdpo-rlsd reads (rollout_data["teacher_log_probs"]) --
            # RLSD only needs the sampled token's P_T/P_S ratio, not a top-k
            # distribution, so it shares this cheaper single-forward branch with
            # the legacy sampled-mode OPD path rather than the KD-loss top-k branch.
            out = self.compute_log_prob(teacher_iter, teacher_nmb, store_prefix="")
            teacher_lp = out["log_probs"]
            if student_log_probs is not None:
                teacher_lp = [teacher_lp[i] if has_prefix[i] else student_log_probs[i] for i in range(len(teacher_lp))]
            rollout_data["teacher_log_probs"] = teacher_lp
            del teacher_rollout_data, teacher_iter, out
            torch.cuda.empty_cache()
            self._dump_sdpo_prompts(rollout_data, has_prefix)
            # Skill-KD (option A) runs independently of how the RESPONSE teacher
            # signal is consumed (RLSD's advantage reweighting here vs. the
            # sdpo_kd_loss top-k branch below) -- skill-KD has always been a
            # top-k distribution match on the skill's OWN tokens, orthogonal to
            # the response mechanism (per instruction: RLSD only replaces the
            # non-skill/response part; self-skill/pitfall/skill-KD stay as-is).
            # Seed inert empty top-k targets for every response sample (kd=0 for
            # them) so _append_sdpo_skill_samples can extend the list with the
            # skill's real top-k targets; losses.py's skill_tok_mask split then
            # isolates the skill span so only the skill-KD loss term fires.
            if getattr(self.args, "sdpo_skill_kd", False):
                topk = int(getattr(self.args, "opd_log_prob_top_k", 128) or 128)
                n_resp = len(rollout_data["tokens"])
                rollout_data["sdpo_teacher_topk_logprobs"] = [torch.zeros((0, topk)) for _ in range(n_resp)]
                rollout_data["sdpo_teacher_topk_ids"] = [
                    torch.zeros((0, topk), dtype=torch.long) for _ in range(n_resp)
                ]
                self._append_sdpo_skill_samples(rollout_data, topk)
            return

        topk = int(getattr(self.args, "opd_log_prob_top_k", 128) or 128)

        if getattr(self.args, "sdpo_kd_loss", False):
            # KD-loss mode (preferred): only compute the TEACHER top-k distribution
            # (with prefix) as a DETACHED target. The student distribution is
            # computed later, grad-enabled, inside policy_loss_function from the
            # training forward's logits, and the divergence is the loss (not an
            # advantage). So here we do a single teacher top-k forward and stash
            # per-token [k] logprobs+ids; policy_loss_function reads them.
            t_out = self.compute_topk_logprobs(teacher_iter, teacher_nmb, topk=topk)
            del teacher_rollout_data, teacher_iter
            torch.cuda.empty_cache()
            t_lp, t_ids = t_out["sdpo_topk_logprobs"], t_out["sdpo_topk_ids"]
            # Empty target for no-prefix samples -> KD loss 0 for them.
            for i in range(len(t_lp)):
                if not has_prefix[i]:
                    t_lp[i] = t_lp[i][:0]
                    t_ids[i] = t_ids[i][:0]
            rollout_data["sdpo_teacher_topk_logprobs"] = t_lp
            rollout_data["sdpo_teacher_topk_ids"] = t_ids
            del t_out
            torch.cuda.empty_cache()
            self._dump_sdpo_prompts(rollout_data, has_prefix)
            # Skill-KD (option A): append the self-generated skill sequences to the
            # training batch as extra samples, tagged sdpo_is_skill=True, with their
            # own teacher top-k. The single training forward then produces skill
            # logits too, and policy_loss_function applies the skill KD (own coef).
            if getattr(self.args, "sdpo_skill_kd", False):
                self._append_sdpo_skill_samples(rollout_data, topk)
            return

        # Legacy advantage-hook distribution path: divergence over the student's
        # top-k tokens, written to opd_reverse_kl and subtracted from advantages
        # by opd.py. Kept for backward compat / ablation. (Weak gradient signal —
        # prefer --sdpo-kd-loss.)
        divergence_mode = getattr(self.args, "sdpo_divergence", "jsd")
        student_iter, student_nmb = get_data_iterator(self.args, self.model, dict(rollout_data))
        s_out = self.compute_topk_logprobs(student_iter, student_nmb, topk=topk)
        t_out = self.compute_topk_logprobs(teacher_iter, teacher_nmb, topk=topk)
        del teacher_rollout_data, teacher_iter, student_iter
        torch.cuda.empty_cache()

        s_lp, s_ids = s_out["sdpo_topk_logprobs"], s_out["sdpo_topk_ids"]
        t_lp, t_ids = t_out["sdpo_topk_logprobs"], t_out["sdpo_topk_ids"]
        reverse_kls = []
        for i in range(len(s_lp)):
            n = int(response_lengths[i])
            if n == 0 or not has_prefix[i]:
                reverse_kls.append(torch.zeros((n,), dtype=torch.float32))
                continue
            reverse_kls.append(
                _sdpo_topk_distribution_divergence(s_lp[i], s_ids[i], t_lp[i], t_ids[i], divergence_mode)
            )
        rollout_data["opd_reverse_kl"] = reverse_kls
        del s_out, t_out
        torch.cuda.empty_cache()

    def _compute_epo_credit_abs_logp_diff(
        self, rollout_data: RolloutBatch, teacher_rollout_data: RolloutBatch, has_prefix: list[bool]
    ) -> list[torch.Tensor]:
        """credit_t = |logp(y_t|x,f,y_<t>) - logp(y_t|x,y_<t>)| on the SAMPLED
        token only (--epo-credit-mode abs_logp_diff, default): one compute_log_prob
        forward per side. Cheap; the literal pointwise-log-likelihood-ratio
        reading of the PMI definition."""
        response_lengths = rollout_data["response_lengths"]

        teacher_iter, teacher_nmb = get_data_iterator(self.args, self.model, teacher_rollout_data)
        plus_out = self.compute_log_prob(teacher_iter, teacher_nmb, store_prefix="")
        del teacher_iter
        torch.cuda.empty_cache()

        plain_iter, plain_nmb = get_data_iterator(self.args, self.model, dict(rollout_data))
        minus_out = self.compute_log_prob(plain_iter, plain_nmb, store_prefix="")
        del plain_iter
        torch.cuda.empty_cache()

        logp_plus = plus_out["log_probs"]
        logp_minus = minus_out["log_probs"]
        clip = float(getattr(self.args, "epo_credit_clip", 5.0) or 0.0)

        credits: list[torch.Tensor] = []
        for i in range(len(logp_minus)):
            n = int(response_lengths[i])
            if n == 0 or not has_prefix[i]:
                credits.append(torch.ones((n,), dtype=torch.float32, device=logp_minus[i].device))
                continue
            c = (logp_plus[i].float() - logp_minus[i].float()).abs()
            if clip > 0:
                c = c.clamp(max=clip)
            credits.append(c)
        del plus_out, minus_out
        return credits

    def _compute_epo_credit_topk_divergence(
        self, rollout_data: RolloutBatch, teacher_rollout_data: RolloutBatch, has_prefix: list[bool]
    ) -> list[torch.Tensor]:
        """credit_t = D(p(.|x,f,y_<t)) ‖ p(.|x,y_<t))) over the top-k next-token
        DISTRIBUTION (--epo-credit-mode topk_divergence): reuses SDPO's
        compute_topk_logprobs + _sdpo_topk_distribution_divergence (same helper
        as the legacy sdpo_divergence advantage-hook path, actor.py:61-110).
        Captures how much the WHOLE next-token belief shifts, not just the
        sampled token, at the cost of a second top-k forward (vs plain
        compute_log_prob) and the O(R*k) per-sample teacher-id lookup inside
        _sdpo_topk_distribution_divergence."""
        response_lengths = rollout_data["response_lengths"]
        topk = int(getattr(self.args, "opd_log_prob_top_k", 128) or 128)
        divergence_mode = getattr(self.args, "epo_credit_divergence", "jsd")

        teacher_iter, teacher_nmb = get_data_iterator(self.args, self.model, teacher_rollout_data)
        t_out = self.compute_topk_logprobs(teacher_iter, teacher_nmb, topk=topk)
        del teacher_iter
        torch.cuda.empty_cache()

        plain_iter, plain_nmb = get_data_iterator(self.args, self.model, dict(rollout_data))
        s_out = self.compute_topk_logprobs(plain_iter, plain_nmb, topk=topk)
        del plain_iter
        torch.cuda.empty_cache()

        t_lp, t_ids = t_out["sdpo_topk_logprobs"], t_out["sdpo_topk_ids"]
        s_lp, s_ids = s_out["sdpo_topk_logprobs"], s_out["sdpo_topk_ids"]
        clip = float(getattr(self.args, "epo_credit_clip", 5.0) or 0.0)

        credits: list[torch.Tensor] = []
        for i in range(len(s_lp)):
            n = int(response_lengths[i])
            if n == 0 or not has_prefix[i]:
                credits.append(torch.ones((n,), dtype=torch.float32))
                continue
            c = _sdpo_topk_distribution_divergence(s_lp[i], s_ids[i], t_lp[i], t_ids[i], divergence_mode)
            if clip > 0:
                c = c.clamp(max=clip)
            credits.append(c)
        del t_out, s_out
        return credits

    def _compute_epo_credit_category_metrics(
        self, rollout_data: RolloutBatch, credits: list[torch.Tensor], has_prefix: list[bool]
    ) -> dict[str, float]:
        """Diagnostic (plan): split credit_t by a cheap regex-based token
        category (epistemic/strategy/compute/format/other, see
        miles.utils.token_category) and report the per-category mean credit —
        "does epistemic-token credit collapse first?" (diagnosis experiment 1.1)
        becomes a free-riding wandb panel instead of a separate offline script.
        Local to THIS rollout batch only (not tracked across checkpoints/steps
        beyond whatever wandb keeps), gated by --epo-credit-token-diagnostics
        since classify_response_tokens costs one convert_ids_to_tokens call per
        sample (cheap, but non-zero — skip it when not actively diagnosing).
        """
        from miles.utils.token_category import CATEGORIES, classify_response_tokens

        tokens_list = rollout_data["tokens"]
        response_lengths = rollout_data["response_lengths"]
        sums = {cat: 0.0 for cat in CATEGORIES}
        counts = {cat: 0 for cat in CATEGORIES}

        for i, (credit, hp) in enumerate(zip(credits, has_prefix, strict=True)):
            if not hp or credit.numel() == 0:
                continue
            n = int(response_lengths[i])
            t = tokens_list[i]
            tok_ids = t.tolist() if torch.is_tensor(t) else list(t)
            resp_ids = tok_ids[len(tok_ids) - n :]
            categories = classify_response_tokens(self.tokenizer, resp_ids)
            credit_cpu = credit.detach().float().cpu()
            for cat, val in zip(categories, credit_cpu.tolist(), strict=True):
                sums[cat] += val
                counts[cat] += 1

        return {
            f"epo_credit_{cat}_mean": (sums[cat] / counts[cat] if counts[cat] > 0 else 0.0) for cat in CATEGORIES
        }

    def _compute_epo_credit(self, rollout_data: RolloutBatch) -> None:
        """EPO: per-token PMI credit_t, both forwards under the SAME teacher
        weights (the EMA snapshot if --sdpo-ema-teacher, else the live self-
        teacher weights already active when this is called — see the use_ema
        switch in train_actor). `f` is the correct-peer prefix (full trace, or
        with --epo-credit-skill the peer's self-generated skill) stashed by
        epo_group_reward on sdpo_teacher_prompt_tokens (reuses SDPO's Megatron
        teacher plumbing: _build_sdpo_teacher_rollout_data keeps the response at
        the tail so per-position outputs stay aligned with the plain student
        sequence). --epo-credit-mode selects the credit formula:
        'abs_logp_diff' (default) or 'topk_divergence' (see
        _compute_epo_credit_{abs_logp_diff,topk_divergence} above).

        Writes rollout_data["epo_credit"] (one [R] tensor per sample, R =
        response_length; 1 — a NEUTRAL weight, not 0 — for samples with no
        privileged-context peer, e.g. the sole correct trace in a group, so
        those traces fall back to plain GRPO instead of losing their reward
        signal entirely), fused into the GRPO advantages by
        compute_advantages_and_returns (loss.py) when --epo-credit-loss is set.
        Also writes a broadcast "epo_credit_cv" scalar (coefficient of variation
        over active tokens) so it surfaces at rollout/epo_credit_cv via
        log_rollout_data's generic numeric-list path, with no extra logging
        plumbing. Reuses SDPO's _dump_sdpo_prompts so --dump-details produces
        the SAME sdpo_prompts/ (student vs. teacher/privileged-context text)
        and skill/ (when --epo-credit-skill is on) dumps SDPO already writes.
        """
        teacher_rollout_data, has_prefix = self._build_sdpo_teacher_rollout_data(rollout_data)

        mode = getattr(self.args, "epo_credit_mode", "abs_logp_diff")
        if mode == "topk_divergence":
            credits = self._compute_epo_credit_topk_divergence(rollout_data, teacher_rollout_data, has_prefix)
        else:
            credits = self._compute_epo_credit_abs_logp_diff(rollout_data, teacher_rollout_data, has_prefix)
        del teacher_rollout_data
        torch.cuda.empty_cache()

        normalize = getattr(self.args, "epo_credit_normalize", True)
        active = [c for c, hp in zip(credits, has_prefix, strict=True) if hp and c.numel() > 0]
        cv = 0.0
        if active:
            flat = torch.cat([c.to(active[0].device) for c in active])
            mean = flat.mean().clamp(min=1e-8)
            cv = (flat.std() / mean).item()
            if normalize:
                credits = [c / mean.to(c.device) if hp else c for c, hp in zip(credits, has_prefix, strict=True)]

        rollout_data["epo_credit"] = [c.to(dtype=torch.float32) for c in credits]
        rollout_data["epo_credit_cv"] = [cv] * len(credits)

        if getattr(self.args, "epo_credit_token_diagnostics", False):
            # NOTE: must run on EVERY DP rank (each on its own local shard, not
            # redundantly), NOT main-rank-only. log_rollout_data's
            # gather_log_data does a cross-DP-rank dist.gather_object keyed by
            # whatever keys are in log_dict — every rank's rollout_data needs
            # the SAME key set or the post-gather `d[key]` lookup KeyErrors on
            # ranks that skipped this block. Best-effort: a diagnostic panel,
            # not a training-critical value, so a failure here must never
            # break training (mirrors _dump_sdpo_prompts's own try/except
            # policy for the same reason) -- but ALL ranks must at least agree
            # to fall back to the same (empty) key set on failure.
            try:
                cat_metrics = self._compute_epo_credit_category_metrics(rollout_data, credits, has_prefix)
            except Exception as e:
                logger.warning(f"[EPO] credit token-category diagnostics failed (non-fatal): {e!r}")
                from miles.utils.token_category import CATEGORIES

                cat_metrics = {f"epo_credit_{cat}_mean": 0.0 for cat in CATEGORIES}
            for key, val in cat_metrics.items():
                rollout_data[key] = [val] * len(credits)

        self._dump_sdpo_prompts(rollout_data, has_prefix)
        torch.cuda.empty_cache()

    def _dump_sdpo_prompts(self, rollout_data: RolloutBatch, has_prefix: list[bool]) -> None:
        """Dump the SDPO student (prompt+response) and teacher (prompt+prefix+
        response) full sequences — decoded text + token ids — for post-hoc
        inspection. Only on the main rank, only when --dump-details is set."""
        dump_dir = getattr(self.args, "dump_details", None)
        if dump_dir is None or not self._is_main_rank:
            return
        try:
            tok = self.tokenizer
            tokens_list = rollout_data["tokens"]
            resp_lens = rollout_data["response_lengths"]
            teacher_prompt_list = rollout_data["sdpo_teacher_prompt_tokens"]
            # Skill dump: triggered whenever a skill was GENERATED (sdpo_skill text is
            # present), independent of skill-KD. self-skill/trace-condense always set
            # the text; the token/logprob fields below are the extra skill-KD payload
            # (only present with --sdpo-skill-kd) and are logged when available.
            skill_text_list = rollout_data.get("sdpo_skill")
            skill_tokens_list = rollout_data.get("sdpo_skill_tokens")
            skill_prompt_list = rollout_data.get("sdpo_skill_prompt_tokens")
            skill_teacher_list = rollout_data.get("sdpo_skill_teacher_prompt_tokens")
            # Failure-pitfall text fields (see examples/SRD/sdpo.py): the raw per-trace
            # pitfall (before pitfall-condense overwrites sdpo_skill) and the group's
            # shared common-pitfall summary spliced into failed traces' teacher prefix.
            trace_pitfall_list = rollout_data.get("sdpo_trace_pitfall")
            group_pitfalls_list = rollout_data.get("sdpo_group_pitfalls")
            # Correct-side mirror of trace_pitfall: the per-trace HINDSIGHT skill, kept
            # even when the blind-correct pass overwrites sdpo_skill with a problem-only
            # knowledge prediction. This is what --sdpo-blind-correct-info group-skills
            # concatenates into the teacher prompt.
            trace_skill_list = rollout_data.get("sdpo_trace_skill")
            # Per-trace correctness (stashed by the SDPO group RM). Lets the skill dump
            # label whether each skill is a solution-skill (from a correct trace) or a
            # pitfall (from an incorrect trace) — the two flavors --sdpo-skill-source
            # all/incorrect produce. Absent -> unknown (None).
            #
            # rollout_data["sdpo_correct"] is GLOBAL/unpartitioned on purpose
            # (log_passrate needs the full rollout_batch_size x n_samples_per_prompt
            # shape), unlike tokens/sdpo_skill/sdpo_trace_pitfall/sdpo_group_pitfalls,
            # which ARE partitioned to this rank's local slice -- so it can't be
            # indexed with the LOCAL loop index i below (that silently read a
            # DIFFERENT sample's correctness, observed as skill_kind='pitfall' next
            # to a "CORRECT worked solution" skill-gen prompt). Use
            # "sdpo_correct_local" instead: process_rollout_data (miles/utils/
            # data.py) stashes a copy already reordered by this rank's partition,
            # aligned with tokens/response_lengths/etc, specifically for this.
            correct_list = rollout_data.get("sdpo_correct_local")
            records = []
            skill_records = []
            for i in range(len(tokens_list)):
                t = tokens_list[i]
                tl = t.tolist() if torch.is_tensor(t) else list(t)
                n = int(resp_lens[i])
                split = len(tl) - n  # prompt|response boundary
                prompt_ids, resp_ids = tl[:split], tl[split:]
                teacher_prompt_ids = (
                    list(teacher_prompt_list[i]) if (i < len(teacher_prompt_list) and has_prefix[i]) else []
                )
                # Teacher sequence = teacher_prompt (system + user+solution + assistant
                # marker) followed by the student's response.
                teacher_ids = (teacher_prompt_ids + resp_ids) if teacher_prompt_ids else tl
                rec = {
                    "has_prefix": bool(has_prefix[i]),
                    "student_text": tok.decode(tl),
                    "teacher_text": tok.decode(teacher_ids),
                    "prompt_text": tok.decode(prompt_ids),
                    "teacher_prompt_text": tok.decode(teacher_prompt_ids) if teacher_prompt_ids else "",
                    "response_text": tok.decode(resp_ids),
                    "response_length": n,
                }
                records.append(rec)
                # Skill records go to a SEPARATE skill/ folder (not mixed into the
                # response prompts — too hard to find otherwise). Dump whenever a skill
                # was generated (sdpo_skill text present), regardless of skill-KD. The
                # skill-KD token payload (sk_ids/sk_prompt/sk_teacher) is optional: when
                # present, prefer the decoded tokens (exact) and add the student/teacher
                # sequences; when absent, fall back to the raw skill text.
                sk_text = (
                    skill_text_list[i] if (skill_text_list is not None and i < len(skill_text_list)) else ""
                )
                sk_ids = (
                    list(skill_tokens_list[i])
                    if (skill_tokens_list is not None and i < len(skill_tokens_list) and skill_tokens_list[i])
                    else []
                )
                if sk_text or sk_ids:
                    sk_prompt = (
                        list(skill_prompt_list[i]) if (skill_prompt_list and i < len(skill_prompt_list) and skill_prompt_list[i]) else []
                    )
                    sk_teacher = (
                        list(skill_teacher_list[i])
                        if (skill_teacher_list and i < len(skill_teacher_list) and skill_teacher_list[i])
                        else []
                    )
                    # skill_length: exact token count from the KD payload when present,
                    # else the tokenized length of the skill text (so it is never null
                    # just because skill-KD is off).
                    sk_len = len(sk_ids) if sk_ids else (len(tok.encode(sk_text)) if sk_text else 0)
                    # Label the skill's provenance: a correct trace yields a
                    # solution-skill, an incorrect one a pitfall (see _gen_one in
                    # examples/SRD/sdpo.py). None when correctness is unavailable.
                    # correct_list ("sdpo_correct_local") is already reordered to
                    # this rank's local partition (see comment above), so it lines
                    # up with tokens_list/skill_text_list at the same local index i.
                    self_correct = (
                        bool(float(correct_list[i]) > 0.5)
                        if (correct_list is not None and i < len(correct_list) and correct_list[i] is not None)
                        else None
                    )
                    skill_kind = None if self_correct is None else ("solution" if self_correct else "pitfall")
                    trace_pitfall = (
                        trace_pitfall_list[i] if (trace_pitfall_list is not None and i < len(trace_pitfall_list)) else ""
                    )
                    group_pitfalls = (
                        group_pitfalls_list[i] if (group_pitfalls_list is not None and i < len(group_pitfalls_list)) else ""
                    )
                    trace_skill = (
                        trace_skill_list[i] if (trace_skill_list is not None and i < len(trace_skill_list)) else ""
                    )
                    skill_records.append(
                        {
                            "index": i,
                            "self_correct": self_correct,
                            "skill_kind": skill_kind,
                            "skill_length": sk_len,
                            "skill_text": tok.decode(sk_ids) if sk_ids else sk_text,
                            # Raw per-trace pitfall (failed traces): preserved even when
                            # pitfall-condense overwrites skill_text with a problem-only
                            # prediction. The group's shared common-pitfall summary that
                            # was spliced into this trace's teacher prefix (failed only).
                            "trace_pitfall_text": trace_pitfall,
                            "group_pitfalls_text": group_pitfalls,
                            # Correct traces: the hindsight skill, preserved the same way
                            # (blind-correct overwrites skill_text with the problem-only
                            # knowledge prediction).
                            "trace_skill_text": trace_skill,
                            "problem_text": tok.decode(prompt_ids),
                            "skill_student_prompt_text": tok.decode(sk_prompt) if sk_prompt else "",
                            "skill_teacher_prompt_text": tok.decode(sk_teacher) if sk_teacher else "",
                            "skill_student_text": tok.decode(sk_prompt + sk_ids) if (sk_prompt and sk_ids) else "",
                            "skill_teacher_text": tok.decode(sk_teacher + sk_ids) if (sk_teacher and sk_ids) else "",
                        }
                    )
            rollout_id = getattr(self, "_last_rollout_id", 0)
            import json

            path = Path(dump_dir) / "sdpo_prompts" / f"{rollout_id}_rank{dist.get_rank()}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                for r in records:
                    f.write(json.dumps({"rollout_id": rollout_id, **r}, ensure_ascii=False) + "\n")
            logger.info(f"Dumped {len(records)} SDPO teacher/student prompts to {path}")
            # Separate skill/ folder.
            if skill_records:
                spath = Path(dump_dir) / "skill" / f"{rollout_id}_rank{dist.get_rank()}.jsonl"
                spath.parent.mkdir(parents=True, exist_ok=True)
                with open(spath, "w") as f:
                    for r in skill_records:
                        f.write(json.dumps({"rollout_id": rollout_id, **r}, ensure_ascii=False) + "\n")
                logger.info(f"Dumped {len(skill_records)} SDPO skills to {spath}")
        except Exception as e:  # dumping must never break training
            logger.warning(f"SDPO prompt dump failed (non-fatal): {e!r}")

    def _build_sdpo_teacher_rollout_data(self, rollout_data: RolloutBatch):
        """Build a minimal rollout_data whose token sequences are
        teacher_prompt + response, where teacher_prompt is the FULL chat-templated
        prompt with the correct-peer solution spliced into the USER turn (built on
        the rollout side, see examples/SRD/sdpo.py). The response is kept at the
        tail so response-span outputs stay token-aligned with the student's. This
        matches lasgroup/SDPO (teacher_input_ids = cat([teacher_prompt, response])).
        Returns (teacher_rollout_data, has_prefix flags)."""
        teacher_prompt_list = rollout_data["sdpo_teacher_prompt_tokens"]
        tokens_list = rollout_data["tokens"]
        response_lengths = rollout_data["response_lengths"]
        teacher_tokens, teacher_total_lengths, teacher_loss_masks, has_prefix = [], [], [], []
        for i, tokens in enumerate(tokens_list):
            resp_len = int(response_lengths[i])
            teacher_prompt = teacher_prompt_list[i] if i < len(teacher_prompt_list) else []
            tok_list = tokens.tolist() if torch.is_tensor(tokens) else list(tokens)
            if not teacher_prompt or resp_len == 0:
                has_prefix.append(False)
                new_tokens = tok_list
            else:
                has_prefix.append(True)
                # response = last resp_len tokens of the student sequence.
                resp_ids = tok_list[len(tok_list) - resp_len :]
                new_tokens = list(teacher_prompt) + resp_ids
            teacher_tokens.append(new_tokens)
            teacher_total_lengths.append(len(new_tokens))
            teacher_loss_masks.append([1] * resp_len)

        device = torch.cuda.current_device()
        teacher_rollout_data: RolloutBatch = {
            "tokens": [torch.tensor(t, dtype=torch.long, device=device) for t in teacher_tokens],
            "response_lengths": list(response_lengths),
            "total_lengths": teacher_total_lengths,
            "loss_masks": [torch.tensor(m, dtype=torch.int, device=device) for m in teacher_loss_masks],
        }
        if self.args.qkv_format == "bshd":
            max_seq_len = max(teacher_total_lengths)
            pad_size = get_parallel_state().tp.size * self.args.data_pad_size_multiplier
            max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size
            teacher_rollout_data["max_seq_lens"] = [max_seq_len] * len(teacher_tokens)
        return teacher_rollout_data, has_prefix

    def _build_sdpo_skill_rollout_data(self, rollout_data: RolloutBatch):
        """Build STUDENT and TEACHER sequences for the self-generated skill KD.

        For each sample that carries a skill (sdpo_skill_tokens non-empty):
          student seq = skill_prompt + skill_tokens          (context skill was generated in)
          teacher seq = skill_teacher_prompt + skill_tokens  (skill_prompt + own-trace hint)
        The skill tokens are kept at the tail so the skill-span outputs align.
        Returns (student_rd, teacher_rd, skill_idx) where skill_idx maps each built
        sequence back to its original sample index, or (None, None, []) if none."""
        skill_tokens_list = rollout_data.get("sdpo_skill_tokens")
        skill_prompt_list = rollout_data.get("sdpo_skill_prompt_tokens")
        skill_teacher_list = rollout_data.get("sdpo_skill_teacher_prompt_tokens")
        if not skill_tokens_list:
            return None, None, []

        device = torch.cuda.current_device()

        # A skill sample is KD-eligible only when its tokens, STUDENT prompt, AND
        # TEACHER prompt are all present. Compute this index set ONCE and drive both
        # _pack and skill_idx from it, so student_rd / teacher_rd / skill_idx stay
        # index-aligned. (Previously _pack filtered on tokens+prompt only while
        # skill_idx also required the teacher prompt; a sample with an empty teacher
        # prompt then landed in _pack but not skill_idx, shifting every later entry
        # and pairing skill tokens with the wrong response_length — which blew up
        # log_rollout_data's split-by-response_lengths.)
        skill_idx = [
            i
            for i in range(len(skill_tokens_list))
            if skill_tokens_list[i]
            and i < len(skill_prompt_list)
            and skill_prompt_list[i]
            and i < len(skill_teacher_list)
            and skill_teacher_list[i]
        ]
        if not skill_idx:
            return None, None, []

        def _pack(prompt_ids_list):
            toks, totals, masks, resp_lens = [], [], [], []
            for i in skill_idx:
                skill_ids = list(skill_tokens_list[i])
                prompt_ids = list(prompt_ids_list[i])
                seq = prompt_ids + skill_ids
                toks.append(seq)
                totals.append(len(seq))
                masks.append([1] * len(skill_ids))
                resp_lens.append(len(skill_ids))
            return toks, totals, masks, resp_lens

        def _make(prompt_ids_list):
            toks, totals, masks, resp_lens = _pack(prompt_ids_list)
            rd: RolloutBatch = {
                "tokens": [torch.tensor(t, dtype=torch.long, device=device) for t in toks],
                "response_lengths": resp_lens,
                "total_lengths": totals,
                "loss_masks": [torch.tensor(m, dtype=torch.int, device=device) for m in masks],
            }
            if self.args.qkv_format == "bshd":
                pad_size = get_parallel_state().tp.size * self.args.data_pad_size_multiplier
                msl = (max(totals) + pad_size - 1) // pad_size * pad_size
                rd["max_seq_lens"] = [msl] * len(toks)
            return rd

        return _make(skill_prompt_list), _make(skill_teacher_list), skill_idx

    def _num_local_gbs(self) -> int:
        """Local (per-DP-rank) global batch size — the divisor get_data_iterator
        requires every rollout_data sample count to be a multiple of."""
        return self.args.global_batch_size // get_parallel_state().intra_dp.size

    @staticmethod
    def _pad_min_rollout_data_to_multiple(rd: RolloutBatch, multiple: int, device) -> int:
        """Pad a MINIMAL teacher/student rollout_data (tokens/total_lengths/
        response_lengths/loss_masks[/max_seq_lens]) with 1-token dummy samples so
        its sample count is a multiple of `multiple` (get_data_iterator requires
        num_local_samples % num_local_gbs == 0). Returns the number of real
        (non-pad) samples so callers can drop the padded outputs."""
        n_real = len(rd["tokens"])
        if multiple <= 0:
            return n_real
        pad = (-n_real) % multiple
        for _ in range(pad):
            rd["tokens"].append(torch.tensor([0], dtype=torch.long, device=device))
            rd["total_lengths"].append(1)
            rd["response_lengths"].append(1)
            rd["loss_masks"].append(torch.zeros((1,), dtype=torch.int, device=device))
        if "max_seq_lens" in rd and rd["max_seq_lens"]:
            rd["max_seq_lens"] = [rd["max_seq_lens"][0]] * len(rd["tokens"])
        return n_real

    def _pad_rollout_to_dp_agreed_count(self, rollout_data, is_skill, gbs, device, sk_lp, sk_ids) -> int:
        """Pad rollout_data with inert 1-token dummies up to a count that is (a) a
        multiple of gbs and (b) IDENTICAL across all DP ranks. Returns n_pad.

        CRITICAL (DP collective): each DP rank appends a DIFFERENT number of skill
        samples (its local correct/incorrect trace count varies), so a purely-local
        pad to a gbs multiple would leave ranks with DIFFERENT sample counts ->
        DIFFERENT num_steps_per_rollout (= num_local_samples // gbs). get_data_iterator
        then all-reduces a num_microbatches tensor whose LENGTH is num_steps, so
        mismatched step counts crash/hang the collective. Agree on a common padded
        count = the batch-wide MAX (rounded up to a gbs multiple) so every rank runs
        the same number of training steps. EVERY rank (including ones with no skill
        samples) must call this so the MAX all-reduce doesn't desync. Extra dummies
        are inert: is_skill=False, empty loss mask -> zero loss, zero grad.
        """
        local_count = len(rollout_data["tokens"])
        target_count = ((local_count + gbs - 1) // gbs) * gbs if gbs > 0 else local_count
        try:
            t = torch.tensor([target_count], device=device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=get_parallel_state().intra_dp.group)
            target_count = int(t.item())
        except Exception as e:
            logger.warning(f"[SDPO skill-KD] DP count all-reduce failed ({e!r}); using local pad.")
        n_pad = target_count - local_count
        if n_pad <= 0:
            return 0

        # Empty teacher top-k slices matching the real entries' tensor type. Prefer an
        # existing skill teacher tensor; else derive an empty one from the response
        # teacher target already in rollout_data.
        def _empty_like_topk(key):
            if sk_lp is not None and key == "sdpo_teacher_topk_logprobs" and len(sk_lp):
                return sk_lp[0][:0]
            if sk_ids is not None and key == "sdpo_teacher_topk_ids" and len(sk_ids):
                return sk_ids[0][:0]
            lst = rollout_data.get(key)
            if lst:
                return lst[0][:0]
            return torch.zeros((0,), device=device)

        explicit = {
            "tokens", "total_lengths", "response_lengths", "loss_masks",
            "sdpo_teacher_topk_logprobs", "sdpo_teacher_topk_ids", "sdpo_is_skill", "max_seq_lens",
        }
        list_keys = [
            k for k, v in rollout_data.items()
            if isinstance(v, list) and len(v) == local_count and k not in explicit
        ]

        def _placeholder(template):
            if torch.is_tensor(template):
                return torch.zeros((1,), dtype=template.dtype, device=template.device)
            if isinstance(template, (list, tuple)):
                return type(template)([0.0])
            return 0.0

        for _ in range(n_pad):
            rollout_data["tokens"].append(torch.tensor([0], dtype=torch.long, device=device))
            rollout_data["total_lengths"].append(1)
            rollout_data["response_lengths"].append(1)
            rollout_data["loss_masks"].append(torch.zeros((1,), dtype=torch.int, device=device))
            if "sdpo_teacher_topk_logprobs" in rollout_data:
                rollout_data["sdpo_teacher_topk_logprobs"].append(_empty_like_topk("sdpo_teacher_topk_logprobs"))
            if "sdpo_teacher_topk_ids" in rollout_data:
                rollout_data["sdpo_teacher_topk_ids"].append(_empty_like_topk("sdpo_teacher_topk_ids"))
            is_skill.append(False)
            for k in list_keys:
                rollout_data[k].append(_placeholder(rollout_data[k][0]))
        return n_pad

    def _append_sdpo_skill_samples(self, rollout_data: RolloutBatch, topk: int) -> None:
        """Compute the skill teacher top-k and APPEND the skill sequences to
        rollout_data as extra training samples (tagged sdpo_is_skill). The response
        samples already present are tagged sdpo_is_skill=False. Skill samples carry
        their own sdpo_teacher_topk_* (skill teacher target) so the shared KD path
        produces per-token divergence on the skill span; policy_loss_function scales
        them by --sdpo-skill-kd-coef via the sdpo_is_skill mask."""
        n_resp = len(rollout_data["tokens"])
        # Mark existing (response) samples as non-skill up front.
        is_skill = [False] * n_resp
        # Per-sample "was this skill distilled from a CORRECT trace" tag (self-success/
        # blind-correct) vs a FAILED one (pitfall-condense) -- lets losses.py split the
        # aggregate skill/kl into skill/kl_correct vs skill/kl_pitfall so a run can see
        # which half of the skill-KD signal is actually doing the work (motivating
        # question behind --sdpo-skill-kd-mode blind-correct/both-blind). False for
        # response (non-skill) samples; meaningless there, never read for them.
        skill_is_correct = [False] * n_resp
        device = torch.cuda.current_device()
        gbs = self._num_local_gbs()

        student_rd, teacher_rd, skill_idx = self._build_sdpo_skill_rollout_data(rollout_data)
        if not skill_idx:
            # This rank has no skill samples. Other DP ranks may still append some and
            # pad the batch UP, changing num_steps_per_rollout. We must agree on the
            # SAME padded sample count on every rank (see the DP-collective note in the
            # pad block below) or get_data_iterator's num_microbatches all-reduce
            # (length == num_steps) desyncs and hangs. Participate in the MAX all-reduce
            # with our own count, then pad to the agreed target with inert dummies.
            rollout_data["sdpo_is_skill"] = is_skill
            rollout_data["sdpo_skill_is_correct"] = skill_is_correct
            self._pad_rollout_to_dp_agreed_count(rollout_data, is_skill, gbs, device, None, None)
            return

        # Teacher top-k over the skill (skill_teacher_prompt + skill_tokens). Pad the
        # sub-batch to a num_local_gbs multiple so get_data_iterator accepts it, then
        # keep only the real outputs.
        n_skill = self._pad_min_rollout_data_to_multiple(teacher_rd, gbs, device)
        teacher_iter, teacher_nmb = get_data_iterator(self.args, self.model, teacher_rd)
        sk_out = self.compute_topk_logprobs(teacher_iter, teacher_nmb, topk=topk)
        sk_lp = sk_out["sdpo_topk_logprobs"][:n_skill]
        sk_ids = sk_out["sdpo_topk_ids"][:n_skill]
        del teacher_iter, sk_out
        torch.cuda.empty_cache()

        # Append each skill sample's student sequence + its teacher target to the
        # main rollout_data lists so the training forward covers them.
        skill_toks = student_rd["tokens"]
        skill_total = student_rd["total_lengths"]
        skill_resp = student_rd["response_lengths"]
        skill_masks = student_rd["loss_masks"]
        skill_rollout_lp = rollout_data.get("sdpo_skill_rollout_logprobs")

        # Keys we set explicitly per skill sample below.
        explicit = {
            "tokens",
            "total_lengths",
            "response_lengths",
            "loss_masks",
            "sdpo_teacher_topk_logprobs",
            "sdpo_teacher_topk_ids",
            "sdpo_is_skill",
            "max_seq_lens",
        }
        # Any OTHER per-sample list of length n_resp (log_probs, ref_log_probs,
        # rewards, raw_reward, advantages, returns, rollout_log_probs, ...) needs an
        # inert placeholder per skill sample so lengths stay consistent through
        # compute_advantages_and_returns / get_batch. Skill samples are KD-only:
        # advantages/pg contribute nothing (pure distill), and per-response-token
        # tensors just need the right length; scalars get 0.
        def _placeholder(key, resp_len, template):
            if torch.is_tensor(template):
                return torch.zeros((resp_len,), dtype=template.dtype, device=template.device)
            if isinstance(template, (list, tuple)):
                return type(template)([0.0] * resp_len)
            return 0.0

        list_keys = [
            k
            for k, v in rollout_data.items()
            if isinstance(v, list) and len(v) == n_resp and k not in explicit
        ]

        correct_local = rollout_data.get("sdpo_correct_local")
        for j, orig_i in enumerate(skill_idx):
            rl = int(skill_resp[j])
            rollout_data["tokens"].append(skill_toks[j])
            rollout_data["total_lengths"].append(skill_total[j])
            rollout_data["response_lengths"].append(rl)
            rollout_data["loss_masks"].append(skill_masks[j])
            rollout_data["sdpo_teacher_topk_logprobs"].append(sk_lp[j])
            rollout_data["sdpo_teacher_topk_ids"].append(sk_ids[j])
            is_skill.append(True)
            skill_is_correct.append(
                bool(float(correct_local[orig_i]) > 0.5) if (correct_local is not None and orig_i < len(correct_local)) else False
            )
            for k in list_keys:
                if k == "rollout_log_probs":
                    # skill's own rollout logprobs (for the IS ratio); else zeros.
                    lp = skill_rollout_lp[orig_i] if (skill_rollout_lp and orig_i < len(skill_rollout_lp)) else None
                    if lp is not None and not torch.is_tensor(lp):
                        lp = torch.tensor(lp, dtype=torch.float32, device=device) if lp else None
                    if torch.is_tensor(lp):
                        lp = lp.to(device)
                    else:
                        lp = torch.zeros((rl,), device=device)
                    # CRITICAL: this per-token tensor MUST have length == rl (the
                    # skill sample's response_length), or log_rollout_data's
                    # concat-then-split-by-response_lengths crashes, and the KD IS
                    # ratio (guarded on numel==n) silently drops. The rollout
                    # logprobs can drift from len(skill_tokens) — e.g. </think>
                    # stripping trims skill_tokens but not the captured logprobs, or
                    # a max-new-tokens hit. Force-align to rl (pad with zeros / clip).
                    if lp.numel() != rl:
                        fixed = torch.zeros((rl,), dtype=lp.dtype, device=device)
                        c = min(rl, lp.numel())
                        fixed[:c] = lp[:c]
                        lp = fixed
                    rollout_data[k].append(lp)
                else:
                    rollout_data[k].append(_placeholder(k, rl, rollout_data[k][0]))

        # Pad the (response + skill) batch to a DP-agreed count with inert 1-token
        # dummies so get_data_iterator accepts it AND every DP rank runs the same
        # number of training steps (see _pad_rollout_to_dp_agreed_count).
        n_pad = self._pad_rollout_to_dp_agreed_count(rollout_data, is_skill, gbs, device, sk_lp, sk_ids)
        skill_is_correct.extend([False] * n_pad)
        rollout_data["sdpo_is_skill"] = is_skill
        rollout_data["sdpo_skill_is_correct"] = skill_is_correct

        # Interleave response + skill samples across the batch. Steps are carved by
        # get_data_iterator sequentially (step i = samples[i*gbs:(i+1)*gbs]); with
        # skill samples appended at the TAIL, the early steps would be pure-response
        # and the late steps pure-skill, so per-step logs alternate 0 / value on
        # entropy_loss vs sdpo_skill_kd_loss (a sawtooth on wandb). Spread each kind
        # evenly so every step mixes both and its metrics are all non-zero.
        #
        # This is a REORDER ONLY — it changes which samples land in which step, not
        # the sample count, global batch size, microbatch sizing, step count, LR
        # schedule, or EMA cadence (all derived downstream from the same lists).
        # Advantages are order-independent (per-sample; skill samples carry inert
        # placeholder rewards), and per-step membership doesn't affect the summed,
        # DP-all-reduced gradient. Keep the trailing dummies (is_skill=False, empty
        # loss mask) at the very end untouched.
        n_total = len(rollout_data["tokens"])
        n_real = n_total - n_pad
        resp_pos = [i for i in range(n_real) if not is_skill[i]]
        skill_pos = [i for i in range(n_real) if is_skill[i]]
        if resp_pos and skill_pos:
            # Round-robin merge (evenly spread the smaller set through the larger).
            merged: list[int] = []
            a, b = (resp_pos, skill_pos) if len(resp_pos) >= len(skill_pos) else (skill_pos, resp_pos)
            ratio = len(a) / len(b)
            bi = 0.0
            ai = 0
            next_b = 0
            while ai < len(a) or next_b < len(b):
                # emit from the larger set, injecting from the smaller on schedule
                if next_b < len(b) and (ai >= len(a) or ai >= (next_b + 1) * ratio):
                    merged.append(b[next_b])
                    next_b += 1
                else:
                    merged.append(a[ai])
                    ai += 1
            perm = merged + list(range(n_real, n_total))  # dummies stay at the tail
            assert sorted(perm) == list(range(n_total)), "interleave permutation must be a bijection"
            for k, v in list(rollout_data.items()):
                if isinstance(v, list) and len(v) == n_total:
                    rollout_data[k] = [v[p] for p in perm]

        # bshd needs a consistent max_seq_len across all (response+skill) samples.
        if self.args.qkv_format == "bshd" and "max_seq_lens" in rollout_data:
            msl = max(rollout_data["total_lengths"])
            pad_size = get_parallel_state().tp.size * self.args.data_pad_size_multiplier
            msl = (msl + pad_size - 1) // pad_size * pad_size
            rollout_data["max_seq_lens"] = [msl] * len(rollout_data["tokens"])

    def compute_topk_logprobs(self, data_iterator, num_microbatches, topk: int = 128):
        return forward_only(
            get_topk_logprobs,
            self.args,
            self.model,
            data_iterator,
            num_microbatches,
            store_prefix="",
            f_kwargs={"sdpo_topk": topk},
        )

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return

        # torch dist may trigger nccl communication during saving.
        if self.args.offload_train:
            reload_process_groups()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            from miles.backends.megatron_utils.model import save_hf_model

            save_hf_model(self.args, rollout_id, self.model)

        if self.args.offload_train:
            destroy_process_groups()

    @timer
    def update_weights(self, info: "EnginesAndLock") -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        rollout_engines = info.rollout_engines
        rollout_engine_lock = info.rollout_engine_lock
        has_new_engines = info.has_new_engines
        engine_gpu_counts = info.engine_gpu_counts
        engine_gpu_offsets = info.engine_gpu_offsets
        del info

        if self.args.offload_train:
            reload_process_groups()

        if has_new_engines:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_has_new_engines.remote())

        if self.args.debug_skip_weight_update:
            if dist.get_rank() == 0:
                logger.warning("Skipping actor-to-rollout weight update because " "--debug-skip-weight-update is set.")
            if self.args.offload_train:
                destroy_process_groups()
            return

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if self.args.ci_test and len(rollout_engines) > 0 and not is_lora_enabled(self.args):
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

        if self.args.offload_train:
            destroy_process_groups()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        # load_checkpoint reads self.args.ckpt_step to pick which iteration to load.
        # Temporarily override it for ref/teacher loads, then restore after the load below.
        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step

        if model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.opd_teacher_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        if model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag

    def connect_actor_critic(
        self,
        actor_handle: ActorHandle | None = None,
        master_address: str | None = None,
        master_port: int | None = None,
    ) -> None:
        if self.role == "actor":
            master_address = ray.util.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            actor_handle.connect_actor_critic.remote(master_address=master_address, master_port=master_port)

        group_name = "actor_critic"
        world_size = 2
        self._actor_critic_groups = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0 if self.role == "actor" else 1,
            group_name=group_name,
        )
