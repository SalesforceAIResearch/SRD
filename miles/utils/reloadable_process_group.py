import logging
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

from miles.utils.memory_utils import print_memory

logger = logging.getLogger(__name__)

old_new_group_dict = {}

_dist_checkpointing_merge_patched = False


def monkey_patch_dist_checkpointing_merge():
    """Widen Megatron's dist_checkpointing dict_utils.merge() optimizer
    param_state truncation to BOTH directions.

    Megatron's own merge() (dict_utils.py) already special-cases optimizer
    param_state lists: if the loaded checkpoint's list (`x2`, on-disk) is
    SHORTER than the current run's expected list (`x1`), the extra padding
    entries on `x1` are truncated -- this is the "dp_reshardable padding"
    case the upstream comment describes. But under TP-reshaped resume (e.g.
    loading a checkpoint saved at a different TP than the current run) the
    on-disk list can instead be LONGER than the current run's expected list
    (observed live: x1=93 (current, real) vs x2=196 (on-disk, padded) ->
    ValueError, not the case upstream's guard covers). Since both directions
    are the SAME padding-mismatch situation -- just observed from either
    side -- truncate whichever side is longer, not only x1.
    """
    global _dist_checkpointing_merge_patched
    if _dist_checkpointing_merge_patched:
        return
    _dist_checkpointing_merge_patched = True

    from megatron.core.dist_checkpointing import dict_utils, serialization, tensor_aware_state_dict

    original_merge = dict_utils.merge

    def merge(x1, x2, key=()):
        if (
            isinstance(x1, list)
            and isinstance(x2, list)
            and len(x1) != len(x2)
            and dict_utils._is_optimizer_param_state_key(key)
        ):
            if len(x2) > len(x1):
                logger.info(
                    f"dist_checkpointing merge: truncating on-disk optimizer param_state "
                    f"list at {key} from {len(x2)} to {len(x1)} entries (checkpoint was "
                    f"saved under a different parallel layout)."
                )
                del x2[len(x1) :]
        return original_merge(x1, x2, key=key)

    # `merge` is recursive via a BARE name lookup inside dict_utils.py itself, so
    # rebinding the module attribute is enough for that recursion. But
    # serialization.py / tensor_aware_state_dict.py did `from .dict_utils import
    # merge` -- a name binding at import time -- so their `merge` name is frozen to
    # the pre-patch function object and rebinding dict_utils.merge alone would be
    # invisible at their call sites (exactly where checkpoint load calls merge()).
    # Patch every module-level name too.
    dict_utils.merge = merge
    serialization.merge = merge
    tensor_aware_state_dict.merge = merge
    logger.info("Applying monkey patch to megatron.core.dist_checkpointing.dict_utils.merge")


def monkey_patch_torch_dist():
    pid = os.getpid()
    if pid in old_new_group_dict:
        assert dist.old_new_group == old_new_group_dict[pid]
        return

    logger.info("Applying monkey patch to torch.distributed")

    old_new_group = dist.new_group
    old_new_group_dict[pid] = old_new_group
    dist.old_new_group = old_new_group

    def new_group(*args, **kwargs):
        group = old_new_group(*args, **kwargs)
        # skip none nccl group.
        if len(args) >= 3 and args[2] == "gloo" or "backend" in kwargs and kwargs["backend"] == "gloo":
            return group

        # Get ranks from arguments
        if len(args) >= 1 and args[0] is not None:
            ranks = args[0]
        elif "ranks" in kwargs and kwargs["ranks"] is not None:
            ranks = kwargs["ranks"]
        else:
            # If no ranks specified, use all ranks in world
            ranks = list(range(dist.get_world_size()))

        if len(ranks) == 1:
            return group

        group = ReloadableProcessGroup(group, ranks)
        return group

    dist.new_group = new_group

    def get_new_function(func):
        def new_function(*args, **kwargs):
            args = tuple([arg.group if isinstance(arg, ReloadableProcessGroup) else arg for arg in args])
            kwargs = {k: (v.group if isinstance(v, ReloadableProcessGroup) else v) for k, v in kwargs.items()}
            with _wrap_low_level_call():
                return func(*args, **kwargs)

        return new_function

    dist.get_rank = get_new_function(dist.get_rank)
    dist.get_world_size = get_new_function(dist.get_world_size)
    dist.get_backend = get_new_function(dist.get_backend)
    dist.get_global_rank = get_new_function(dist.get_global_rank)
    dist.get_group_rank = get_new_function(dist.get_group_rank)
    dist.get_process_group_ranks = get_new_function(dist.get_process_group_ranks)

    dist.all_reduce = get_new_function(dist.all_reduce)
    dist.all_gather = get_new_function(dist.all_gather)
    dist.all_gather_into_tensor = get_new_function(dist.all_gather_into_tensor)
    dist.all_gather_object = get_new_function(dist.all_gather_object)
    dist.all_to_all = get_new_function(dist.all_to_all)
    dist.all_to_all_single = get_new_function(dist.all_to_all_single)
    dist.broadcast = get_new_function(dist.broadcast)
    dist.broadcast_object_list = get_new_function(dist.broadcast_object_list)
    dist.reduce = get_new_function(dist.reduce)
    dist.reduce_scatter = get_new_function(dist.reduce_scatter)
    dist.reduce_scatter_tensor = get_new_function(dist.reduce_scatter_tensor)
    dist.scatter = get_new_function(dist.scatter)
    dist.gather = get_new_function(dist.gather)
    dist.barrier = get_new_function(dist.barrier)
    dist.send = get_new_function(dist.send)
    dist.recv = get_new_function(dist.recv)
    dist._coalescing_manager = get_new_function(dist._coalescing_manager)

    # p2p
    old_isend = dist.isend
    old_irecv = dist.irecv

    dist.isend = get_new_function(dist.isend)
    dist.irecv = get_new_function(dist.irecv)

    def get_new_p2pop_function(func):
        def new_function(*args, **kwargs):
            def convert(arg):
                if isinstance(arg, ReloadableProcessGroup):
                    return arg.group
                elif arg == dist.isend:
                    arg = old_isend
                elif arg == dist.irecv:
                    arg = old_irecv
                return arg

            args = (convert(arg) for arg in args)
            kwargs = {k: convert(v) for k, v in kwargs.items()}
            return func(*args, **kwargs)

        return new_function

    dist.P2POp.__new__ = get_new_p2pop_function(dist.P2POp.__new__)
    dist.P2POp.__init__ = get_new_p2pop_function(dist.P2POp.__init__)


class ReloadableProcessGroup(torch.distributed.ProcessGroup):
    GROUPS = {}

    def __init__(self, group, ranks):
        super().__init__(
            rank=dist.get_rank(group),
            size=dist.get_world_size(group),
        )
        self.group = group
        self.group_info = {
            "ranks": ranks,
        }
        pid = os.getpid()
        if pid not in ReloadableProcessGroup.GROUPS:
            ReloadableProcessGroup.GROUPS[pid] = []
        ReloadableProcessGroup.GROUPS[pid].append(self)

    def __getattr__(self, name):
        return getattr(self.group, name)

    @staticmethod
    def destroy_process_groups():
        pid = os.getpid()
        for reloadable_group in ReloadableProcessGroup.GROUPS.get(pid, []):
            if reloadable_group.group is None:
                continue
            try:
                dist.destroy_process_group(reloadable_group.group)
            except ValueError as e:
                logger.warning(
                    f"Process group already invalid/destroyed; skipping cleanup. Exception: {e}",
                    exc_info=True,
                )

            del reloadable_group.group
            reloadable_group.group = None

    @staticmethod
    def reload_process_groups():
        pid = os.getpid()
        reloadable_groups = ReloadableProcessGroup.GROUPS.get(pid, [])
        logger.info(f"Reloading {len(reloadable_groups)} process groups in pid {pid}")
        old_new_group = old_new_group_dict.get(pid)
        for reloadable_group in reloadable_groups:
            if reloadable_group.group is not None:
                continue
            group = old_new_group(ranks=reloadable_group.group_info["ranks"], backend="nccl")
            reloadable_group.group = group

    def rank(self) -> int:
        return self.group.rank()

    def size(self) -> int:
        return self.group.size()

    def name(self) -> str:
        return self.group.name()

    def shutdown(self) -> None:
        if self.group is not None:
            self.group.shutdown()

    def abort(self) -> None:
        if self.group is not None:
            self.group.abort()

    def _fwd(self, method, *args, **kwargs):
        inner = self.group
        if inner is None:
            raise RuntimeError("ReloadableProcessGroup: inner PG is None, call reload() first.")
        with _wrap_low_level_call():
            return getattr(inner, method)(*args, **kwargs)

    def barrier(self, *a, **kw):
        return self._fwd("barrier", *a, **kw)

    def broadcast(self, *a, **kw):
        return self._fwd("broadcast", *a, **kw)

    def allreduce(self, *a, **kw):
        return self._fwd("allreduce", *a, **kw)

    def allreduce_coalesced(self, *a, **kw):
        return self._fwd("allreduce_coalesced", *a, **kw)

    def reduce(self, *a, **kw):
        return self._fwd("reduce", *a, **kw)

    def allgather(self, *a, **kw):
        return self._fwd("allgather", *a, **kw)

    def _allgather_base(self, *a, **kw):
        return self._fwd("_allgather_base", *a, **kw)

    def allgather_coalesced(self, *a, **kw):
        return self._fwd("allgather_coalesced", *a, **kw)

    def allgather_into_tensor_coalesced(self, *a, **kw):
        return self._fwd("allgather_into_tensor_coalesced", *a, **kw)

    def gather(self, *a, **kw):
        return self._fwd("gather", *a, **kw)

    def scatter(self, *a, **kw):
        return self._fwd("scatter", *a, **kw)

    def reduce_scatter(self, *a, **kw):
        return self._fwd("reduce_scatter", *a, **kw)

    def _reduce_scatter_base(self, *a, **kw):
        return self._fwd("_reduce_scatter_base", *a, **kw)

    def reduce_scatter_tensor_coalesced(self, *a, **kw):
        return self._fwd("reduce_scatter_tensor_coalesced", *a, **kw)

    def alltoall_base(self, *a, **kw):
        return self._fwd("alltoall_base", *a, **kw)

    def alltoall(self, *a, **kw):
        return self._fwd("alltoall", *a, **kw)

    def send(self, *a, **kw):
        return self._fwd("send", *a, **kw)

    def recv(self, *a, **kw):
        return self._fwd("recv", *a, **kw)

    def recv_anysource(self, *a, **kw):
        return self._fwd("recv_anysource", *a, **kw)

    def _start_coalescing(self, *a, **kw):
        return self._fwd("_start_coalescing", *a, **kw)

    def _end_coalescing(self, *a, **kw):
        return self._fwd("_end_coalescing", *a, **kw)

    def _get_backend_name(self):
        return self._fwd("_get_backend_name")

    def _get_backend(self, *a, **kw):
        return self._fwd("_get_backend", *a, **kw)

    def _set_default_backend(self, *a, **kw):
        return self._fwd("_set_default_backend", *a, **kw)

    @property
    def bound_device_id(self):
        return self.group.bound_device_id

    @bound_device_id.setter
    def bound_device_id(self, dev):
        self.group.bound_device_id = dev


def destroy_process_groups():
    """Destroy all reloadable process groups."""
    ReloadableProcessGroup.destroy_process_groups()


def reload_process_groups():
    """Reload all reloadable process groups."""
    ReloadableProcessGroup.reload_process_groups()


@contextmanager
def _wrap_low_level_call():
    try:
        yield
    except Exception as e:
        mem_info = print_memory("after torch distributed error")
        if hasattr(e, "add_note"):
            e.add_note(f"{mem_info=}")
        raise
