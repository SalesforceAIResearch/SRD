import json
import logging
from pathlib import Path

import torch

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _jsonl_safe(v):
    """Best-effort convert a sample field to something JSON-serializable."""
    if isinstance(v, torch.Tensor):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _jsonl_safe(x) for k, x in v.items() if not isinstance(x, torch.Tensor) or x.numel() <= 4096}
    if isinstance(v, (list, tuple)):
        return [_jsonl_safe(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _write_samples_jsonl(path: Path, rollout_id, samples: list[dict]) -> None:
    """Write a human-readable jsonl: one line per sample with the key fields
    (prompt, response, reward, label, response_length + a few sdpo metadata keys).
    Heavy per-token arrays (tokens, logprobs) are dropped to keep it readable."""
    keep = ("prompt", "response", "label", "reward", "response_length", "status", "index")
    with open(path, "w") as f:
        for s in samples:
            meta = s.get("metadata") if isinstance(s.get("metadata"), dict) else {}
            row = {"rollout_id": rollout_id}
            for k in keep:
                if k in s:
                    row[k] = _jsonl_safe(s[k])
            # surface useful SDPO metadata (correctness / ppl / domain), skip
            # token arrays. "domain" lets multi-domain dump consumers (e.g.
            # examples/agentic/'s dashboard) group rows by task without
            # re-deriving it from prompt/response text. "task_type"/
            # "tau2_domain" are the per-subdomain breakdown keys
            # examples/agentic/loaders/alfworld.py and loaders/tau2.py group
            # by -- previously missing from this whitelist, so their
            # per-task-type/per-subdomain charts always silently rendered
            # empty (every row's "task_type"/"tau2_domain" column was NaN,
            # `if "task_type" in group.columns` never caught it because the
            # DataFrame still had a "domain" column, just not this one).
            # "tau2_messages": tau2 samples never populate prompt/response (the
            # conversation is driven externally by tau2's own Orchestrator, see
            # tools/tau2/agent_function.py's module docstring) -- sample.prompt
            # is a fixed placeholder string, so without this key a tau2 row's
            # rollout_data dump has no trace of the real conversation at all,
            # including the user's actual first turn. tau2_messages (already
            # {role, content, tool_calls}-shaped) IS that real conversation.
            for mk in ("sdpo_correct", "sdpo_ppl", "domain", "task_type", "tau2_domain", "tau2_messages"):
                if mk in meta:
                    row[mk] = _jsonl_safe(meta[mk])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_debug_rollout_data(args, rollout_id: int):
    data = torch.load(
        args.load_debug_rollout_data.format(rollout_id=rollout_id),
        weights_only=False,
    )["samples"]
    data = [Sample.from_dict(sample) for sample in data]
    if (ratio := args.load_debug_rollout_data_subsample) is not None:
        original_num_rows = len(data)
        rough_subsample_num_rows = int(original_num_rows * ratio)
        data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
        logger.info(
            f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
        )
    return data


def save_debug_rollout_data(args, data, rollout_id, evaluation: bool):
    # TODO to be refactored (originally Buffer._set_data)
    if (path_template := args.save_debug_rollout_data) is not None:
        path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
        logger.info(f"Save debug rollout data to {path.with_suffix('.jsonl')}")
        path.parent.mkdir(parents=True, exist_ok=True)

        # TODO may improve the format
        if evaluation:
            dump_data = dict(
                samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
            )
        else:
            dump_data = dict(
                samples=[sample.to_dict() for sample in data],
            )

        # Only write the human-readable jsonl (prompt/response/reward per sample;
        # heavy per-token arrays dropped). The .pt (torch.save) dump is skipped —
        # it's only needed for load_debug_rollout_data replay, which this run
        # doesn't use, and it bloats the dump dir. Re-enable it below if you need
        # to replay rollouts via --load-debug-rollout-data.
        try:
            _write_samples_jsonl(path.with_suffix(".jsonl"), rollout_id, dump_data["samples"])
        except Exception as e:  # never break training on a logging convenience
            logger.warning(f"jsonl dump failed (non-fatal): {e!r}")
