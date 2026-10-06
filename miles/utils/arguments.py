import argparse
import json
import logging
import os
from typing import Any

import yaml
from sglang_router.launch_router import RouterArgs

from miles.backends.sglang_utils.arguments import add_sglang_arguments
from miles.backends.sglang_utils.arguments import validate_args as sglang_validate_args
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType
from miles.utils.environ import enable_experimental_rollout_refactor
from miles.utils.eval_config import EvalDatasetConfig, build_eval_dataset_configs, ensure_dataset_list
from miles.utils.hf_config import is_dsa, load_hf_config
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import load_function

logger = logging.getLogger(__name__)


def reset_arg(parser, name, **kwargs):
    """
    Reset the default value of a Megatron argument.
    :param parser: The argument parser.
    :param name: The name of the argument to reset.
    :param default: The new default value.
    """
    for action in parser._actions:
        if name in action.option_strings:
            if "default" in kwargs:
                action.default = kwargs["default"]
            break
    else:
        parser.add_argument(name, **kwargs)


def get_miles_extra_args_provider(add_custom_arguments=None):
    def add_miles_arguments(parser):
        # Ray
        def add_cluster_arguments(parser):
            parser.add_argument("--actor-num-nodes", type=int, default=1, help="Number of nodes for training actor")
            parser.add_argument(
                "--actor-num-gpus-per-node", type=int, default=8, help="Number of gpus per node for training actor"
            )
            parser.add_argument(
                "--critic-num-nodes", type=int, default=None, help="Number of nodes for training actor"
            )
            parser.add_argument(
                "--critic-num-gpus-per-node", type=int, default=None, help="Number of gpus per node for training actor"
            )

            parser.add_argument(
                "--rollout-num-gpus",
                type=int,
                default=None,
                help=(
                    "Number of GPUs for inference. Note that when using --colocate, "
                    "i.e. the training and the inference engines are on the same gpus, this param will be ignored and will be set as "
                    "actor_num_gpus_per_node * actor_num_nodes."
                ),
            )
            parser.add_argument(
                "--rollout-num-gpus-per-engine",
                type=int,
                default=1,
                help="Number of GPUs per inference engine, just like the tp_size in sglang.",
            )
            parser.add_argument(
                "--num-gpus-per-node",
                type=int,
                default=8,
                help=(
                    "Number of gpus per node for rollout."
                    "Notice: If you are going to use less than 8 gpus per node under colocate mode, you should set this number."
                ),
            )
            parser.add_argument(
                "--colocate",
                action="store_true",
                default=False,
                help=(
                    "Whether to colocate the inference engines and the actor. "
                    "Turning this on will also set --offload to true."
                ),
            )
            parser.add_argument(
                "--offload",
                action="store_true",
                default=False,
                help=("Equivalent to --offload-train + --offload-rollout. "),
            )
            parser.add_argument(
                "--offload-train",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the training actor to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )
            parser.add_argument(
                "--offload-rollout",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the rollout generator to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )

            parser.add_argument(
                "--offload-rollout-level",
                type=str,
                nargs="+",
                default=["kv_cache", "weight"],
                help=(
                    "Specifies what to offload during rollout when offload-rollout is set. "
                    "Possible values: 'kv_cache', 'weight'. Default: both 'kv_cache' and 'weight'. "
                    "Example: --offload-rollout-level kv_cache weight"
                ),
            )

            reset_arg(parser, "--distributed-backend", type=str, default="nccl")
            reset_arg(parser, "--distributed-timeout-minutes", type=int, default=10)

            return parser

        def add_train_arguments(parser):
            parser.add_argument(
                "--train-backend",
                type=str,
                choices=["megatron", "fsdp"],
                default="megatron",
                help="The backend for training.",
            )
            parser.add_argument(
                "--qkv-format",
                type=str,
                choices=["thd", "bshd"],
                default="thd",
                help="The qkv layout.",
            )
            parser.add_argument(
                "--linear-attention-backend",
                type=str,
                choices=["fla", "flashqla"],
                default="fla",
                help=(
                    "Backend for Qwen GDN linear-attention layers. "
                    "'fla' (flash-linear-attention) is portable and runs on any supported GPU. "
                    "'flashqla' (FlashQLA) requires NVIDIA SM90 (Hopper) or newer, CUDA 12.8+, and PyTorch 2.8+."
                ),
            )
            parser.add_argument(
                "--miles-dsa-topk-backend",
                type=str,
                choices=["torch", "flashinfer"],
                default="torch",
                help="Top-k backend for Miles DSA indexer.",
            )
            parser.add_argument(
                "--true-on-policy-mode",
                action="store_true",
                default=False,
                help="Whether to enable true-on-policy mode.",
            )
            parser.add_argument(
                "--recompute-logprobs-via-prefill",
                action="store_true",
                default=False,
                help=(
                    "Recompute rollout logprobs via SGLang prefill instead of decode kernels. "
                    "Only needed for models whose prefill and decode paths are not numerically identical."
                ),
            )
            parser.add_argument(
                "--train-env-vars",
                type=json.loads,
                default="{}",
                help="Extra environment variables for training process, e.g. PyTorch memory management ones.",
            )
            parser.add_argument(
                "--train-memory-margin-bytes",
                type=int,
                default=1024**3,
                help="Add margin for train memory allocation. By default we will reserve 1GB as margin.",
            )
            parser.add_argument(
                "--debug-skip-weight-update",
                action="store_true",
                default=False,
                help=(
                    "Debug-only: preserve the train/rollout offload-onload schedule, "
                    "but skip the actual actor-to-rollout weight update."
                ),
            )
            parser.add_argument(
                "--debug-disable-optimizer",
                action="store_true",
                default=False,
                help=(
                    "Debug-only: do not initialize the Megatron optimizer or LR scheduler. "
                    "Training still runs rollout, log-prob forward, and actor forward/backward, "
                    "but skips optimizer state allocation and optimizer updates."
                ),
            )
            parser.add_argument(
                "--disable-weights-backuper",
                action="store_false",
                dest="enable_weights_backuper",
                help=(
                    "Applies to `megatron` training backend only. "
                    "Disables the system that backups model weights (Actor, Ref, Old Actor) to CPU RAM. "
                    "Disabling saves significant host memory but prevents features that rely on weight-swapping, such as computing KL-divergence against a reference model. "
                    "Note: do not set `--ref-load` and `--keep-old-actor` if disable weights backuper."
                ),
            )
            parser.add_argument(
                "--megatron-to-hf-mode",
                choices=["raw", "bridge"],
                default="raw",
                help="The method to convert megatron weights to hugging face weights for SGLang.",
            )
            parser.add_argument(
                "--dsa-attention-backend",
                choices=["megatron", "tilelang"],
                default="tilelang",
                help=(
                    "DSA sparse-MLA kernel backend for GLM (glm_moe_dsa) under --megatron-to-hf-mode bridge. "
                    "'tilelang' (default) uses the fused TileLang kernels (SparseMLA + lighting_indexer, vendored from slime) for "
                    "rollout<->train numerical parity; 'megatron' uses the portable unfused megatron-core "
                    "kernels. 'tilelang' requires --qkv-format thd and the optional tilelang dep, and is "
                    "training/forward-only (no KV cache, cannot serve inference). Both support GLM-5.1 and "
                    "GLM-5.2, full or LoRA. No effect on non-DSA models or the 'raw' path."
                ),
            )
            parser.add_argument(
                "--extra-high-precision-layers-hf",
                type=str,
                nargs="*",
                default=(),
                help=("Extra substrings for HF weight names to skip quantization " "(e.g. .kv_b_proj.)."),
            )
            parser.add_argument(
                "--extra-high-precision-layers-megatron",
                type=str,
                nargs="*",
                default=(),
                help=(
                    "Extra substrings for Megatron weight names to skip quantization in Megatron-to-HF paths "
                    "(e.g. .linear_kv_up_proj.)."
                ),
            )
            parser.add_argument(
                "--custom-model-provider-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom model provider function. "
                    "If set, we will use this function instead of the default model provider. "
                    "The function should have the signature "
                    "`def custom_model_provider(pre_process: bool, post_process: bool, vp_stage: int | None = None) -> GPTModel`. "
                    "Example: 'my_module.my_model_provider'."
                ),
            )
            parser.add_argument(
                "--recompute-loss-function",
                action="store_true",
                help="Whether to enable recompute loss function to save memory during training.",
            )
            parser.add_argument(
                "--log-probs-chunk-size", type=int, default=-1, help="Chunk size to compute log probs to save memory"
            )
            parser.add_argument(
                "--allgather-cp",
                action="store_true",
                default=False,
            )
            reset_arg(
                parser,
                "--low-memory-resume",
                action="store_true",
                default=False,
                help=(
                    "Allocate optimizer states on CPU during checkpoint loading to prevent GPU OOM on memory spike. "
                ),
            )

            return parser

        # rollout
        def add_rollout_arguments(parser):
            parser.add_argument(
                "--hf-checkpoint",
                type=str,
                default=None,
                help=(
                    "The huggingface checkpoint of the trained model. "
                    "This is used to initialize sglang and also provide the tokenizer. "
                    "Note that, we will always update the parameters in sglang with that of megatron before training, "
                    "so you only need to provide a huggingface checkpoint that has the same architecture as the model you want to train. "
                    "It doesn't necessary need to contain the most up-to-date parameters."
                ),
            )
            parser.add_argument(
                "--model-name",
                type=str,
                default=None,
                help=(
                    "The name of the model, this is used to convert the megatron weights into huggingface format. "
                    "If not set, we will use `type(load_hf_config(args.hf_checkpoint)).__name__.lower()` as model_name. "
                    "Also, sometimes this will help alleviate the bug that transformers cannot find certain model."
                ),
            )
            parser.add_argument(
                "--rollout-function-path",
                type=str,
                default=(
                    "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn"
                    if enable_experimental_rollout_refactor()
                    else "miles.rollout.sglang_rollout.generate_rollout"
                ),
                help=(
                    "Path to the rollout generation function. "
                    "Use this to create your own custom rollout function and set this to its path. "
                    "The function is called as `fn(args, rollout_id, data_source, evaluation=evaluation)`, "
                    "so its signature should be "
                    "`def generate_rollout(args, rollout_id, data_source, evaluation=False) "
                    "-> RolloutFnTrainOutput | RolloutFnEvalOutput` "
                    "(see `miles.rollout.sglang_rollout.generate_rollout` for the default impl). "
                    "Within each output sample, set at least `tokens`, `response_length`, `reward`, "
                    "and `truncated`."
                ),
            )
            parser.add_argument(
                "--rollout-temperature",
                type=float,
                default=1.0,
                help="the temperature for the inference engine during rollout.",
            )
            parser.add_argument(
                "--rollout-top-p", type=float, default=1.0, help="the top-p for the inference engine during rollout."
            )
            parser.add_argument(
                "--rollout-top-k", type=int, default=-1, help="the top-k for the inference engine during rollout."
            )
            parser.add_argument(
                "--rollout-max-context-len",
                type=int,
                default=None,
                help=(
                    "The maximum context size for the inference engine during rollout."
                    "It should no exceed the `max_position_embeddinds` in Huggingface model's `config.json`"
                ),
            )
            parser.add_argument(
                "--rollout-max-prompt-len",
                type=int,
                default=None,
                help=(
                    "The maximum length of the prompt for the inference engine during rollout. "
                    "If set, we will filter out the long prompts during initialization of the global dataset. "
                    "This is not recommended if the dataset is large."
                ),
            )
            parser.add_argument(
                "--rollout-max-response-len",
                type=int,
                default=None,
                help=(
                    "The maximum length of the response for the inference engine during rollout. "
                    "It is basically `max_tokens` in sglang."
                ),
            )
            parser.add_argument(
                "--rollout-skip-special-tokens",
                action="store_true",
                default=False,
                help=(
                    "Whether to skip special tokens in the response during rollout. "
                    "This is useful when you want to use the response as a prompt for the next rollout."
                ),
            )
            parser.add_argument(
                "--rollout-stop",
                type=str,
                nargs="+",
                default=None,
                help=(
                    "The stop words for the inference engine during rollout. "
                    "It can be a list of strings or a single string. "
                    "It may be hard to pass special tokens in command line, in that case rollout_stop_token_ids can be used."
                ),
            )
            parser.add_argument(
                "--rollout-stop-token-ids",
                type=int,
                nargs="+",
                default=None,
                help=(
                    "The stop token ids for the inference engine during rollout. "
                    "It can be a list of integers or a single integer."
                ),
            )
            parser.add_argument(
                "--rollout-shuffle",
                action="store_true",
                default=False,
                help=("Whether to shuffle the prompts during rollout."),
            )
            parser.add_argument(
                "--rollout-seed",
                type=int,
                default=42,
                help=(
                    "The seed for the random number generator during rollout. "
                    "This is used to shuffle the prompts and also for the random sampling of the prompts."
                ),
            )

            # sampling
            parser.add_argument(
                "--over-sampling-batch-size",
                type=int,
                default=None,
                help=(
                    "This defines the granularity of the sampling batch in the rollout function. "
                    "When the number of available samples falls below the target, a sampling "
                    "operation of size over_sampling_batch_size will be triggered."
                    "Regardless of whether partial rollout is used or filters are applied, "
                    "the sampling granularity is always determined by this value. "
                    "If this value is None, rollout_batch_size will be used as the default over_sampling_batch_size."
                ),
            )
            parser.add_argument(
                "--dynamic-sampling-filter-path",
                type=str,
                default=None,
                help=(
                    "This is the filter function for dynamic sampling. "
                    "It should be able to judge whether the result of a prompt should be selected or not."
                    "We will do dynamic filter for sampling as in DAPO. e.g. not all correct or all wrong samples."
                    "You could use `miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` as an example."
                ),
            )

            # partial rollout
            parser.add_argument(
                "--partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Whether to use partial rollout. "
                    "If set, the unfinished samples during dynamic sampling will be recycled back to data buffer. "
                    "This is useful for long responses."
                ),
            )
            parser.add_argument(
                "--mask-offpolicy-in-partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Whether to mask previous generation in partial rollout. "
                    "If set, only on-policy generated tokens will be used in training"
                ),
            )
            parser.add_argument(
                "--max-weight-staleness",
                type=int,
                default=None,
                help=(
                    "Maximum allowed gap between a group's oldest weight version and the current "
                    "engine weight version. Groups exceeding this threshold are recycled back to "
                    "the data buffer instead of being sent to training. Only effective in fully "
                    "async mode. None (default) disables staleness filtering."
                ),
            )
            parser.add_argument(
                "--custom-generate-function-path",
                type=str,
                default=None,
                help=(
                    "Only substitue the `def generate(args, sample, sampling_params)` function within the example rollout function. "
                    "This should be useful if you need to implement some special rollout logic, e.g. multi-turn, function calling."
                ),
            )
            parser.add_argument(
                "--custom-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging rollout data. The signature of the functions is: "
                    "def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )
            parser.add_argument(
                "--custom-eval-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging eval rollout data. "
                    "def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )

            parser.add_argument(
                "--buffer-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the buffer filter function. "
                    "It should be able to select the samples in the buffer. "
                    "The function should take list[list[Sample]] and return list[list[Sample]]."
                ),
            )
            # update weight
            parser.add_argument(
                "--update-weight-buffer-size",
                type=int,
                default=512 * 1024**2,
                help=(
                    "buffer size for update weight, in bytes. "
                    "This is used for updating weights by chunk and should be useful for MoE models."
                ),
            )
            parser.add_argument(
                "--update-weights-interval",
                type=int,
                default=1,
                help="Interval for updating the weights",
            )
            parser.add_argument(
                "--pause-generation-mode",
                type=str,
                choices=["abort", "retract", "in_place"],
                default="retract",
                help=(
                    "How SGLang pauses in-flight requests during weight updates. "
                    "'abort' immediately terminates all requests (previous default). "
                    "'retract' moves running requests back to the waiting queue and "
                    "recomputes KV cache after update. "
                    "'in_place' freezes requests and resumes with existing KV cache."
                ),
            )
            parser.add_argument(
                "--keep-old-actor",
                action="store_true",
                help="Whether to keep the rollout model on training process",
            )

            parser.add_argument(
                "--rollout-data-postprocess-path",
                type=str,
                default=None,
                help=(
                    "The called after we have all the rollout data including log_probs. "
                    "It may be helpful for updating loss mask."
                ),
            )
            parser.add_argument(
                "--pin-rollout-manager-to-head",
                action="store_true",
                default=False,
                help=(
                    "Pin the RolloutManager (and its co-located router process) to the Ray head node. "
                    "Useful in K8s where the head pod has a stable Service address so that "
                    "external agent environments can reliably reach the router."
                ),
            )
            parser.add_argument(
                "--rollout-external",
                action="store_true",
                default=False,
                help="Use external SGLang instances instead of launching them inside the framework.",
            )
            parser.add_argument(
                "--rollout-external-engine-addrs",
                type=str,
                default=None,
                nargs="+",
                help="Address and ports of the external engines.",
            )
            parser.add_argument(
                "--update-weight-transfer-mode",
                choices=["broadcast", "p2p", "disk-delta"],
                default="broadcast",
                help=(
                    "The method to transfer weights to remote rollout engines during update weight. "
                    "'disk-delta' diffs each sync against a CPU snapshot of the previous one and publishes "
                    "only the changed bytes to --update-weight-disk-dir; each engine's /pull_weights applies "
                    "them into a host-local checkpoint that the engine reloads from."
                ),
            )
            parser.add_argument(
                "--update-weight-disk-dir",
                type=str,
                default=None,
                help=(
                    "Filesystem directory disk-delta weight sync publishes to: one delta directory "
                    "(changed tensors only) per sync, written by the trainer and read by every "
                    "rollout host. Required for --update-weight-transfer-mode=disk-delta."
                ),
            )
            parser.add_argument(
                "--update-weight-local-checkpoint-dir",
                type=str,
                default=None,
                help=(
                    "Rollout-host-local directory (e.g. NVMe) holding a full HF checkpoint kept in "
                    "sync by each engine's /pull_weights: every host seeds it from the engine's model "
                    "path and patches published deltas in place, and the engines reload from it. "
                    "Required for --update-weight-transfer-mode=disk-delta. The read-side counterpart "
                    "of --custom-update-weight-post-write-path is the engine's "
                    "--sglang-custom-pull-weights-pre-read-hook."
                ),
            )
            parser.add_argument(
                "--update-weight-delta-encoding",
                choices=["xor", "overwrite"],
                default="xor",
                help=(
                    "On-disk delta encoding for disk-delta weight sync. 'xor' (default): new ^ old — "
                    "smallest wire and fastest, but an involution that must be applied exactly once "
                    "against the correct base (applying it twice reverts). 'overwrite': changed positions "
                    "+ new absolute values — larger, but idempotent. Both are byte-level and dtype-blind; "
                    "the engine reads the choice from each version's index metadata."
                ),
            )
            parser.add_argument(
                "--update-weight-delta-checksum",
                choices=["xxh3-128", "blake3", "adler32"],
                default="xxh3-128",
                help=(
                    "Per-tensor integrity checksum for disk-delta apply. 'xxh3-128' (default): widest fast "
                    "non-cryptographic digest. 'blake3': cryptographic, for untrusted storage. 'adler32': "
                    "for interop. The engine reads the choice from each version's index metadata."
                ),
            )
            parser.add_argument(
                "--custom-update-weight-post-write-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom function called on each trainer rank after a disk-delta sync's "
                    "files are written, before the engines read them — to publish the writes on a "
                    "non-POSIX filesystem (no cross-host visibility without an explicit sync). "
                    "Signature: ``def hook(args, version_dir: str, rollout_engines) -> None``; the hook gates itself."
                ),
            )
            parser.add_argument(
                "--p2p-transfer-num-workers",
                type=int,
                default=4,
                help="Number of thread pool workers for P2P weight transfer.",
            )
            parser.add_argument(
                "--p2p-transfer-timeout",
                type=float,
                default=30.0,
                help="Timeout in seconds for each P2P transfer operation.",
            )
            return parser

        def add_fault_tolerance_arguments(parser):
            parser.add_argument(
                "--use-fault-tolerance",
                action="store_true",
                default=False,
                help="Whether to enable the fault tolerance function during rollout.",
            )
            parser.add_argument(
                "--rollout-health-check-interval",
                type=float,
                default=30.0,
                help="Interval in seconds between rollout engine /health_generate checks during generate/eval.",
            )
            parser.add_argument(
                "--rollout-health-check-timeout",
                type=float,
                default=30.0,
                help="Timeout in seconds to wait for a rollout engine /health_generate response before killing it.",
            )
            parser.add_argument(
                "--rollout-health-check-first-wait",
                type=float,
                default=0,
                help="Initial grace period (in seconds) before starting health checks. This allows time for model compilation and initialization. Increase this value significantly when using deepgemm.",
            )
            return parser

        # data
        def add_data_arguments(parser):
            # dataset
            # TODO: maybe add an num_epoch and calculate the num_rollout from buffer
            parser.add_argument(
                "--num-rollout",
                type=int,
                default=None,
                help="Number of rollout steps. If not set, we will calculate the number of rollout steps from the dataset size.",
            )
            parser.add_argument(
                "--num-epoch",
                type=int,
                default=None,
                help=(
                    "Number of epochs for the training. "
                    "This is used to calculate the number of rollout steps from the dataset size. "
                    "If set, we will calculate the number of rollout steps as `num_rollout = num_epoch * dataset_size // rollout_batch_size`."
                    "If both `--num-epoch` and `--num-rollout` are set, `--num-epoch` will be ignored."
                ),
            )

            parser.add_argument(
                "--disable-rollout-global-dataset",
                action="store_false",
                dest="rollout_global_dataset",
                help=(
                    "Disable the global dataset for rollout. By default, Miles loads `--prompt-data` into a global dataset and samples from it for rollout. "
                    "Setting this flag turns off this behavior, Use this flag only when providing a custom `--rollout-function-path` (and usually a custom `--data-source-path`) that handles data loading independently."
                ),
            )

            parser.add_argument(
                "--data-source-path",
                type=str,
                default="miles.rollout.data_source.RolloutDataSourceWithBuffer",
                help="The data source class for rollout data.",
            )
            parser.add_argument(
                "--prompt-data",
                type=str,
                default=None,
                help=(
                    "The path to the prompt data. "
                    "Currently we only support jsonl format, and each line should contains --input-key and --label-key, "
                    "which will be used as the prompt and the label respectively."
                    "If you want to use a custom template, you can set --apply-chat-template to true, in that case, "
                    "the input should be the same structure as an openai message, e.g. [{'role': 'user', 'content': 'blabla'}]. "
                ),
            )
            parser.add_argument("--apply-chat-template", action="store_true", default=False)
            # Temporarily be JSON-serialized str, will be a real dict after using Omegaconf
            parser.add_argument("--apply-chat-template-kwargs", type=json.loads, default="{}")
            parser.add_argument(
                "--chat-template-path",
                type=str,
                default=None,
                help="Path to an explicit custom Jinja chat template file (.jinja). "
                "Sets tokenizer.chat_template when loading via load_tokenizer, "
                "and also sets --sglang-chat-template so the sglang server uses the same template. "
                "For Miles-maintained fixed templates, leave this unset and pass "
                "--tito-model plus --tito-allowed-append-roles so Miles can auto-resolve "
                "the registered template. The literal value 'autofix' is kept only as a "
                "deprecated compatibility alias for that auto-resolve path. "
                "The path must be accessible on all Ray worker nodes "
                "(e.g. a path inside the miles repo, or a shared filesystem like NFS).",
            )
            parser.add_argument("--input-key", type=str, default="input", help="JSON dataset key")
            parser.add_argument("--label-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument(
                "--multimodal-keys",
                type=json.loads,
                default=None,
                help=(
                    'JSON string for multimodal data mapping media types to data keys. Example: \'{"image": "image_file"}\''
                ),
            )
            parser.add_argument("--metadata-key", type=str, default="metadata", help="JSON dataset key")
            parser.add_argument(
                "--tool-key",
                type=str,
                default="tools",
                help=(
                    "When need to add tools during apply_chat_template, you should provide the key for the tools in the prompt dataset."
                ),
            )
            parser.add_argument(
                "--tool-specs-resolver-path",
                type=str,
                default=None,
                help=(
                    "Zero-arg callable path (e.g. 'module.submodule.fn') returning the "
                    "CURRENT list of tool specs. When set, takes precedence over --tool-key: "
                    "every Dataset construction re-derives tools from this live resolver "
                    "instead of a value baked into the prompt data at build time, so a "
                    "renamed/removed tool in the resolver's source module never leaves a "
                    "stale <tools> block in previously-built jsonl files."
                ),
            )

            parser.add_argument(
                "--start-rollout-id",
                type=int,
                default=None,
                help=(
                    "The starting rollout step, if not set, will try to load the step from --load when doing continue training, "
                    "otherwise will be set to 0, meaning training from start."
                ),
            )

            # batch sizes
            parser.add_argument(
                "--rollout-batch-size",
                type=int,
                required=True,
                help=(
                    "The number of prompts in each rollout step. "
                    "The total data returned should be rollout_batch_size * n_samples_per_prompt. "
                ),
            )
            parser.add_argument(
                "--n-samples-per-prompt", type=int, default=1, help="Number of responses for each prompt in generation"
            )

            # gbs of the training, note that the gbs is of sample, not of prompts,
            # so if you hope to train 1 step for each rollout, the global_bach_size should be set as
            # `rollout_batch_size * n_samples_per_prompt`.
            reset_arg(parser, "--global-batch-size", type=int, default=None)
            parser.add_argument(
                "--num-steps-per-rollout",
                type=int,
                default=None,
                help=(
                    "Number of steps per rollout, e.g. It is equivalent to setting gbs as "
                    "`rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`."
                ),
            )
            # mbs for the training, will be ignored if `use_dynamic_batch_size` is set.
            reset_arg(parser, "--micro-batch-size", type=int, default=1)
            parser.add_argument(
                "--balance-data",
                action="store_true",
                default=False,
                help=(
                    "Repartition each rollout batch so each data-parallel rank gets a similar total token count via Karmarkar-Karp method. "
                    "It may be beneficial for training speed but changes per-rank sample grouping and adds a small CPU scheduling overhead."
                ),
            )

            parser.add_argument(
                "--use-dynamic-batch-size",
                action="store_true",
                default=False,
                help=(
                    "Because the sample length varies, to maximize the GPU utilization, "
                    "we will use the dynamic batch size to adjust the micro batch size according to the maximum number of tokens each gpu can run. "
                    "For example, if we have 3 samples, with the length of 100, 200, and 300, and the max_tokens_per_gpu is 300, when enabling "
                    "dynamic batch size, miles will make 2 micro batches, i.e. [100, 200], [300]."
                ),
            )
            parser.add_argument(
                "--max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for dynamic batch size. "
                    "Note that when enabling context parallel (CP), the max tokens per gpu should be around "
                    "`max_response_len // cp_size` instead of `max_response_len`."
                ),
            )
            parser.add_argument(
                "--log-probs-max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for calculating log probs. "
                    "This is used to calculate the log probs of the responses during rollout, "
                    "and should be set to a larger value than `max_tokens_per_gpu` if you want better performance. "
                ),
            )
            return parser

        def add_eval_arguments(parser):
            parser.add_argument(
                "--eval-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the eval generation function."
                    "If not set, we will use rollout_function_path as the default. "
                ),
            )

            # change the default value of eval_interval from Megatron to None
            reset_arg(parser, "--eval-interval", type=int, default=None)

            parser.add_argument(
                "--eval-prompt-data",
                type=str,
                default=None,
                nargs="+",
                help=(
                    "Path to the evaluation prompt data, "
                    "should first input the name of the eval dataset and then the path, e.g. "
                    "aime /path/to/aime.jsonl"
                ),
            )
            parser.add_argument(
                "--eval-config",
                type=str,
                default=None,
                help=(
                    "Path to an OmegaConf YAML/JSON file describing evaluation datasets. "
                    "When provided, this overrides --eval-prompt-data."
                ),
            )
            parser.add_argument(
                "--skip-eval-before-train",
                action="store_true",
                default=False,
                help="Whether to skip evaluation before training.",
            )

            # The following keys are used to override the rollout version during eval.
            parser.add_argument("--eval-input-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument("--eval-label-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument("--eval-tool-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument(
                "--n-samples-per-eval-prompt",
                type=int,
                default=1,
                help="number of responses for each prompt in generation",
            )
            parser.add_argument("--eval-temperature", type=float, default=None)
            parser.add_argument("--eval-top-p", type=float, default=None)
            parser.add_argument("--eval-top-k", type=int, default=None)
            parser.add_argument("--eval-max-response-len", type=int, default=None)
            parser.add_argument("--eval-max-prompt-len", type=int, default=None)
            parser.add_argument("--eval-min-new-tokens", type=int, default=None)
            parser.add_argument("--eval-max-context-len", type=int, default=None)

            return parser

        def add_algo_arguments(parser):
            parser.add_argument(
                "--ref-load",
                type=str,
                default=None,
                help=(
                    "The checkpoint for reference model. "
                    "When --load is not set, this will be used as the initial checkpoint for training. "
                ),
            )
            parser.add_argument(
                "--ref-ckpt-step", type=int, default=None, help="The checkpoint step for reference model. "
            )
            reset_arg(parser, "--load", type=str, default=None)
            reset_arg(parser, "--save", type=str, default=None)
            reset_arg(parser, "--save-interval", type=int, default=None)
            reset_arg(parser, "--async-save", action="store_true")
            reset_arg(
                parser,
                "--no-save-optim",
                action="store_true",
                default=False,
                help=(
                    "If set, do not save the optimizer state when saving checkpoints. "
                    "This reduces checkpoint size but disables training resumption from the saved checkpoint."
                ),
            )
            parser.add_argument(
                "--save-hf",
                type=str,
                default=None,
                help=(
                    "Path to save the model in HuggingFace format when using Megatron backend. "
                    "The model will be saved to `save_hf.format(rollout_id)`. "
                ),
            )
            reset_arg(parser, "--seed", type=int, default=1234)
            reset_arg(parser, "--clip-grad", type=float, default=1.0)
            reset_arg(parser, "--calculate-per-token-loss", action="store_true")
            reset_arg(parser, "--lr", type=float, default=1e-6)

            parser.add_argument("--num-critic-only-steps", type=int, default=0, help="Number of critic only steps")
            parser.add_argument("--critic-load", type=str, default=None, help="The checkpoint for critic model.")
            parser.add_argument("--critic-save", type=str, default=None, help="The checkpoint for critic model.")
            parser.add_argument("--critic-lr", type=float, default=None, help="The lr for critic model")
            parser.add_argument(
                "--critic-lr-warmup-iters",
                type=int,
                default=0,
                help="number of iterations to linearly warmup for critic model.",
            )

            parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clip range")
            parser.add_argument("--eps-clip-high", type=float, default=None, help="PPO clip upper range")
            parser.add_argument(
                "--eps-clip-c",
                type=float,
                default=None,
                help="lower bound of the value for Dual-clip PPO from https://arxiv.org/pdf/1912.09729",
            )
            parser.add_argument("--value-clip", type=float, default=0.2, help="the clip for value loss")
            parser.add_argument(
                "--kl-coef",
                type=float,
                default=0.00,
                help="KL penalty coefficient for reward shaping. This is applied to the reward signal before advantage calculation.",
            )
            parser.add_argument(
                "--loss-type",
                type=str,
                choices=["policy_loss", "sft_loss", "custom_loss"],
                default="policy_loss",
                help=(
                    "Choose loss type, currently support ppo policy_loss or sft_loss, "
                    "if custom_loss is set, we will use the function path from `--custom-loss-function-path`."
                ),
            )
            parser.add_argument(
                "--custom-loss-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom loss function, if the loss_type is `custom_loss`, "
                    "we will use this function to calculate the loss. "
                ),
            )
            parser.add_argument(
                "--kl-loss-type",
                type=str,
                choices=["k1", "k2", "k3", "low_var_kl"],
                default="k1",
                help="Choose KL loss type: kl, k2, k3, low_var_kl",
            )
            parser.add_argument(
                "--advantage-estimator",
                type=str,
                choices=[
                    "grpo",
                    "gspo",
                    "reinforce_plus_plus",
                    "reinforce_plus_plus_baseline",
                    "ppo",
                ],
                default="grpo",
                help=(
                    "Advantage estimator to use. Note: on-policy distillation (OPD) is now orthogonal "
                    "to the advantage estimator. Use --opd-kl-coef > 0 to enable OPD on top of any estimator."
                ),
            )
            parser.add_argument(
                "--disable-compute-advantages-and-returns",
                action="store_false",
                dest="compute_advantages_and_returns",
                help=(
                    "Whether to disable computing advantages and returns. "
                    "If set, we will not compute the advantages and returns, "
                    "This is useful for sft or custom loss function."
                ),
            )
            parser.add_argument(
                "--use-kl-loss", action="store_true", default=False, help="whether to use KL loss from GRPO"
            )
            parser.add_argument(
                "--kl-loss-coef",
                type=float,
                default=0.0,
                help="KL penalty coefficient for the loss function. This is added to the final PPO loss.",
            )
            parser.add_argument(
                "--use-unbiased-kl",
                action="store_true",
                default=False,
                help="Whether to enable unbiased KL estimation.",
            )
            parser.add_argument(
                "--ref-update-interval",
                type=int,
                default=None,
                help="Interval (in rollout steps) to update ref model from actor. If None, ref model is not updated.",
            )
            parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy loss coef")
            parser.add_argument("--gamma", type=float, default=1.0, help="PPO GAE gamma")
            parser.add_argument("--lambd", type=float, default=1.0, help="PPO GAE lambd")
            parser.add_argument("--normalize-advantages", action="store_true", default=False)
            parser.add_argument(
                "--disable-grpo-std-normalization",
                action="store_false",
                dest="grpo_std_normalization",
                help="from Dr.GRPO https://arxiv.org/pdf/2503.20783",
            )
            parser.add_argument(
                "--disable-rewards-normalization",
                action="store_false",
                dest="rewards_normalization",
                help="Disable rewards normalization",
            )
            parser.add_argument(
                "--use-rollout-entropy",
                action="store_true",
                default=False,
                help=(
                    "Whether to calculate the entropy when calculating the logprobs from actor and reference model. "
                    "This is useful for doing special loss mask."
                ),
            )
            parser.add_argument(
                "--observe-training-entropy",
                action="store_true",
                default=False,
                help=(
                    "Compute training entropy as a logged metric even when --entropy-coef is 0. "
                    "When the coefficient is 0, the observed entropy is detached and does not affect backward."
                ),
            )
            parser.add_argument(
                "--get-mismatch-metrics",
                action="store_true",
                default=False,
                help="Whether to calculate the mismatch metrics.",
            )
            parser.add_argument(
                "--reset-optimizer-states",
                action="store_true",
                default=False,
                help=(
                    "Whether to reset optimizer states after each rollout. "
                    "If enabled, the optimizer's history will be cleared at the end of each rollout, which can sometimes help with training stability or fulfill specific experiment requirements."
                ),
            )
            parser.add_argument(
                "--use-rollout-logprobs",
                action="store_true",
                default=False,
                help=(
                    "Whether to use the rollout logprobs when calculating the importance sampling ratios. "
                    "If not set, we will use the logprobs from the actor model."
                ),
            )
            # Off-Policy Correction using Importance Sampling: https://fengyao.notion.site/off-policy-rl
            parser.add_argument(
                "--use-tis",
                action="store_true",
                default=False,
                help="Enable TIS from https://fengyao.notion.site/off-policy-rl#279721e3f6c48092bbe2fcfe0e9c6b33.",
            )
            parser.add_argument(
                "--tis-clip",
                type=float,
                default=2.0,
                help="Clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--tis-clip-low",
                type=float,
                default=0,
                help="Lower bound clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--custom-tis-function-path",
                type=str,
                default=None,
                help="Path to the custom TIS/RS function (e.g., examples/train_infer_mismatch_helper/mis.py:compute_mis_weights_with_cp).",
            )
            parser.add_argument(
                "--custom-pg-loss-reducer-function-path",
                type=str,
                default=None,
                help="Path to a custom reducer function for pg_loss only. When set, pg_loss will use this custom reducer while other metrics (pg_clipfrac, ppo_kl, entropy_loss, etc.) still use the default sum_of_sample_mean. (e.g., examples/Dr.GRPO/custom_reducer.py:get_pg_loss_reducer).",
            )

            parser.add_argument(
                "--use-routing-replay",
                action="store_true",
                default=False,
                help="The routing replay technique from https://arxiv.org/abs/2507.18071",
            )
            parser.add_argument(
                "--use-rollout-routing-replay",
                action="store_true",
                default=False,
                help="The rollout routing replay technique from https://arxiv.org/abs/2510.11370 (R3): "
                "replay the rollout's MoE routing in training. MoE-only; the GLM-5 launchers pass it "
                "explicitly.",
            )
            parser.add_argument(
                "--use-indexer-replay",
                action="store_true",
                default=False,
                help="Replay indexer topk decisions for layers with indexers.",
            )
            parser.add_argument(
                "--use-rollout-indexer-replay",
                action="store_true",
                default=False,
                help="Replay indexer topk from rollout during training.",
            )
            parser.add_argument(
                "--use-opsm",
                action="store_true",
                default=False,
                help="Whether to enable Off-Policy Sequence Masking (OPSM).",
            )
            parser.add_argument(
                "--opsm-delta",
                type=float,
                default=1e-4,
                help="The threshold for Off-Policy Sequence Masking (OPSM).",
            )
            return parser

        def add_on_policy_distillation_arguments(parser):
            """Add on-policy distillation (OPD) related arguments.

            OPD is orthogonal to advantage estimators and can be applied on top of
            any estimator (GRPO, PPO, etc.) by adding a KL penalty to advantages.
            """
            parser.add_argument(
                "--use-opd",
                action="store_true",
                default=False,
                help="Enable on-policy distillation (OPD). Must specify --opd-type when enabled.",
            )
            parser.add_argument(
                "--opd-type",
                type=str,
                choices=["sglang", "megatron"],
                default=None,
                help=(
                    "Type of on-policy distillation. "
                    "'sglang': Teacher log-probs are obtained from external SGLang server during rollout. "
                    "'megatron': Teacher model is loaded via --opd-teacher-load and forwarded during training."
                ),
            )
            parser.add_argument(
                "--opd-kl-coef",
                type=float,
                default=1.0,
                help="On-policy distillation KL penalty coefficient. Default is 1.0.",
            )
            parser.add_argument(
                "--opd-log-prob-top-k",
                type=int,
                default=0,
                help=(
                    "Number of top-k tokens to use for the re-think OPD token-level reward. "
                    "Set to 0 to use sampled-token OPD."
                ),
            )
            parser.add_argument(
                "--opd-top-k-strategy",
                type=str,
                choices=["only-student", "only-teacher", "intersection", "union", "xor"],
                default="only-student",
                help="Token set strategy for top-k OPD.",
            )
            parser.add_argument(
                "--opd-reward-weight-mode",
                type=str,
                choices=["student_p", "teacher_p", "none"],
                default="student_p",
                help="Weighting scheme for top-k OPD token rewards.",
            )
            parser.add_argument(
                "--opd-teacher-load",
                type=str,
                default=None,
                help=(
                    "The checkpoint for OPD teacher model. Required when --opd-type=megatron. "
                    "The teacher model should have the same architecture as policy/ref model."
                ),
            )
            parser.add_argument(
                "--opd-teacher-ckpt-step", type=int, default=None, help="The checkpoint step for OPD teacher model."
            )
            parser.add_argument(
                "--sdpo-divergence",
                type=str,
                choices=["reverse_kl", "forward_kl", "jsd", "jeffrey", "jeffrey_jsd"],
                default="jsd",
                help=(
                    "Divergence used by the SDPO example (examples/SRD/sdpo.py) between the teacher "
                    "distribution (conditioned on a correct peer prefix) and the student distribution "
                    "(original rollout, no prefix). 'jeffrey' = forward KL + reverse KL (symmetric). "
                    "'jeffrey_jsd' = forward KL + JSD (Jeffrey with the reverse-KL half swapped for "
                    "the milder, bounded JSD — less mode-seeking / entropy collapse)."
                ),
            )
            parser.add_argument(
                "--sdpo-logprob-mode",
                type=str,
                choices=["topk", "sampled"],
                default="topk",
                help=(
                    "SDPO log-prob granularity. 'topk': compare distributions over the student top-k "
                    "token set plus one aggregated tail bucket for the remaining vocabulary mass "
                    "(needs --opd-log-prob-top-k > 0, e.g. 128). 'sampled': compare only the sampled "
                    "token's log-prob (scalar per token)."
                ),
            )
            parser.add_argument(
                "--sdpo-self-teacher",
                action=argparse.BooleanOptionalAction,
                default=True,
                help=(
                    "True SDPO self-distillation: score the teacher against the rollout engine "
                    "(the current policy, re-synced every rollout) instead of a fixed external "
                    "teacher. Use --no-sdpo-self-teacher to score against --rm-url instead."
                ),
            )
            parser.add_argument(
                "--sdpo-ema-teacher",
                action=argparse.BooleanOptionalAction,
                default=False,
                help=(
                    "Use an EMA (exponential-moving-average) copy of the policy as the SDPO "
                    "teacher instead of the live policy. Matches lasgroup/SDPO "
                    "(teacher_regularization='ema'). The teacher weights are a slow moving "
                    "average of the student (teacher = (1-rate)*teacher + rate*student, updated "
                    "each train step), so the teacher does NOT instantly follow the student into "
                    "the 'skip reasoning, emit the answer' degenerate mode the prefix induces — "
                    "breaking the self-reinforcing collapse that live-policy self-teaching causes. "
                    "Only affects --sdpo-teacher-backend megatron."
                ),
            )
            parser.add_argument(
                "--sdpo-ema-teacher-rate",
                type=float,
                default=0.05,
                help=(
                    "EMA update rate for the SDPO teacher (lasgroup/SDPO default 0.05): each train "
                    "step, teacher = (1-rate)*teacher + rate*student. Smaller = slower/more stable "
                    "teacher. Only used with --sdpo-ema-teacher."
                ),
            )
            parser.add_argument(
                "--sdpo-teacher-backend",
                type=str,
                choices=["sglang", "megatron"],
                default="sglang",
                help=(
                    "Where SDPO computes teacher log-probs over prompt+prefix+response. "
                    "'sglang' (default): score via the rollout engine over HTTP during reward "
                    "(full-sequence logprob forces eager prefill in sglang -> slow). "
                    "'megatron': the rollout reward only picks the correct-peer prefix; the "
                    "training actor then forwards prompt+prefix+response with the current policy "
                    "weights (a batched, CUDA-graph'd forward) to get teacher log-probs — far "
                    "faster (like veRL's RefWorker)."
                ),
            )
            parser.add_argument(
                "--sdpo-kd-loss",
                action="store_true",
                default=False,
                help=(
                    "Use a real distribution-level knowledge-distillation LOSS instead of feeding "
                    "the divergence into the advantage (REINFORCE). With --sdpo-teacher-backend "
                    "megatron, the teacher top-k distribution (with prefix) is a detached target, "
                    "and the student distribution (no prefix, grad-enabled) is pulled toward it via "
                    "D(student‖teacher) over the teacher's top-k ids + a tail bucket. Divergence set "
                    "by --sdpo-divergence. This is the strong, directional objective matching the "
                    "original SDPO full_logit_distillation; the advantage-hook path has weak gradient."
                ),
            )
            parser.add_argument(
                "--sdpo-kd-coef",
                type=float,
                default=1.0,
                help="Weight of the SDPO distribution KD loss (only used with --sdpo-kd-loss).",
            )
            parser.add_argument(
                "--sdpo-kd-max-tokens",
                type=int,
                default=0,
                help=(
                    "Cap the SDPO KD loss to the first N response tokens per sample (0 = whole "
                    "response). Bounds the full-vocab log-softmax memory and focuses distillation "
                    "on early tokens. Tokens beyond N contribute 0 to the KD loss."
                ),
            )
            parser.add_argument(
                "--sdpo-is-clip",
                type=float,
                default=2.0,
                help=(
                    "Importance-sampling ratio clip for the SDPO KD loss under off-policy/async "
                    "(matches original SDPO is_clip=2.0). Per token, the KD loss is scaled by "
                    "exp(student_logp - old_logp).clamp(max=is_clip). Set <=0 to disable."
                ),
            )
            parser.add_argument(
                "--sdpo-kd-clip-cov-frac",
                type=float,
                default=0.0,
                help=(
                    "KD Clip-Cov (adapted from arXiv:2505.22617 'Entropy Mechanism of RL'): "
                    "detach the gradient of the top-fraction of response tokens with the LARGEST "
                    "per-token KD divergence, batch-wide. Those few high-divergence tokens (e.g. "
                    "the '<answer>'/letter positions the answer-in-prefix teacher over-weights) "
                    "drive the entropy collapse; freezing their update keeps the distillation "
                    "signal on the rest while preserving policy entropy. 0.0 = disabled. Try 0.002 "
                    "(0.2%). The loss VALUE is unchanged (kept for logging); only gradient is cut."
                ),
            )
            parser.add_argument(
                "--sdpo-distillation-add-tail",
                action=argparse.BooleanOptionalAction,
                default=True,
                help=(
                    "Add an aggregated tail bucket to the top-k distributions before the KD "
                    "divergence (matches original SDPO distillation_add_tail=True). "
                    "--no-sdpo-distillation-add-tail renormalizes over the top-k instead."
                ),
            )
            parser.add_argument(
                "--sdpo-pure-distill",
                action=argparse.BooleanOptionalAction,
                default=True,
                help=(
                    "Pure distillation: the SDPO group RM returns a task reward of 0 for every "
                    "trace, so the GRPO advantage is 0 and the training target is exactly "
                    "-opd_kl_coef * divergence (the distillation signal only). Trace correctness "
                    "is still computed internally to pick correct-peer prefixes. Use "
                    "--no-sdpo-pure-distill to keep the mixed GRPO(task reward) + distillation target."
                ),
            )
            parser.add_argument(
                "--sdpo-remove-thinking-from-demonstration",
                action=argparse.BooleanOptionalAction,
                default=False,
                help=(
                    "Strip <think>...</think> blocks from the correct peer response before "
                    "inserting it as the teacher prefix. Recommended for thinking models "
                    "(e.g. Qwen3) where the peer response contains a long reasoning chain; "
                    "leaving it in makes the teacher prefix huge and teaches the student to "
                    "echo the reasoning verbatim rather than reason independently. "
                    "Default False (safe for non-thinking models like Qwen2.5)."
                ),
            )
            parser.add_argument(
                "--sdpo-reframe-multiturn-prefix",
                action=argparse.BooleanOptionalAction,
                default=False,
                help=(
                    "For NATIVE multi-turn tool-calling peer traces, reframe the ChatML "
                    "<|im_start|>/<|im_end|> turn boundaries into short NLP markers "
                    "('Round N reasoning and tool call:' / 'Observation:') before splicing "
                    "the trace into the teacher prefix. Keeps <think>/<tool_call>/"
                    "<tool_response> content verbatim -- only the control-token boundaries "
                    "are rewritten. Without this, a multi-turn trace leaks raw "
                    "<|im_end|><|im_start|>role tokens into the teacher's user turn "
                    "(_strip_response_eos only removes the trailing one), teaching the "
                    "student to emit control-token garbage. No-op for single-turn traces. "
                    "Default False."
                ),
            )
            parser.add_argument(
                "--sdpo-tool-grammar",
                type=str,
                default="qwen25",
                choices=["qwen25", "qwen3_coder"],
                help=(
                    "Tool-call GRAMMAR the teacher prefix / skill renders tool calls in, so "
                    "the distilled text byte-matches what the student model emits. "
                    "'qwen25' (Qwen3-4B): JSON object inside <tool_call> tags. "
                    "'qwen3_coder' (Qwen3.5-4B): XML <function=NAME><parameter=P>value tags. "
                    "MUST match --generate-tool-call-parser. Default qwen25."
                ),
            )
            parser.add_argument(
                "--sdpo-code-require-tool",
                action=argparse.BooleanOptionalAction,
                default=True,
                help=(
                    "For code-domain samples (metadata['domain']=='code'): grade the code the "
                    "model actually RAN through code_interpreter (metadata['tool_trace']'s last "
                    "call), NOT a text ```python fence. Makes the tool MANDATORY -- a trace that "
                    "never executes its solution has no gradable candidate and is scored wrong, so "
                    "the only path to a correct grade is running the solution via the tool. This "
                    "counters Qwen3-4B's habit of one-shotting a code fence as its final answer "
                    "(observed: 0/512 code traces used the tool). Set --no-sdpo-code-require-tool "
                    "to grade the response fence instead. Default True."
                ),
            )
            parser.add_argument(
                "--sdpo-dynamic-filter-min-correct",
                type=int,
                default=0,
                help=(
                    "Threshold for the SDPO dynamic-sampling filter "
                    "(miles.rollout.filter_hub.dynamic_sampling_filters.check_sdpo_group_has_prefix): "
                    "drop any rollout group with fewer than this many correct traces, then re-sample "
                    "(DAPO-style oversampling) so every kept group can give EVERY trace a correct-peer "
                    "prefix. Default 0 = keep every group (no-op even if the filter path is set), so "
                    "this stays inert unless you opt in. Set 2 for full prefix coverage (with 1 correct, "
                    "that lone trace self-excludes to an empty peer pool); 1 guarantees a prefix for the "
                    "incorrect traces only. Only takes effect when --dynamic-sampling-filter-path points "
                    "at check_sdpo_group_has_prefix. Raise --over-sampling-batch-size so oversampling can "
                    "meet the target when the threshold is high."
                ),
            )
            # ---- LLM-as-judge grading (examples/SRD/sdpo.py) --------------------
            parser.add_argument(
                "--sdpo-judge",
                action="store_true",
                default=False,
                help=(
                    "Grade trace correctness with an LLM judge over an OpenAI-compatible chat "
                    "API instead of exact letter/math matching. The judge sees the question, the "
                    "reference answer, the model's FULL response, and its extracted <answer> "
                    "content, and returns correct/incorrect. This defeats the MCQ letter-guessing "
                    "reward hack (a bare, un-reasoned answer that happens to match is not credited)."
                ),
            )
            parser.add_argument(
                "--sdpo-eval-judge",
                action="store_true",
                default=False,
                help=(
                    "Enable the LLM judge on the EVAL path only, leaving training grading "
                    "untouched. Use this (not --sdpo-judge) when the eval set contains rows that "
                    "deterministic matching cannot grade -- e.g. AMO-Bench's 11/50 "
                    "answer_type='description' problems -- but training should keep its own "
                    "grader (--sdpo-grader dapo). --sdpo-judge implies this."
                ),
            )
            parser.add_argument(
                "--sdpo-judge-backend",
                type=str,
                default="openai",
                choices=["bedrock", "openai"],
                help=(
                    "Transport for the LLM judge. 'openai' (default) is the "
                    "OpenAI-compatible HTTP path (--sdpo-judge-base-url/-api-key-env). "
                    "'bedrock' calls the AWS Bedrock Converse API with boto3, authenticating "
                    "off the instance IAM role -- no API key and no shared gateway in the path."
                ),
            )
            parser.add_argument(
                "--sdpo-judge-base-url",
                type=str,
                default="https://api.openai.com/v1",
                help=(
                    "OpenAI-compatible base URL for the LLM judge (default: OpenAI API). Only "
                    "used when --sdpo-judge-backend=openai."
                ),
            )
            parser.add_argument(
                "--sdpo-judge-model",
                type=str,
                default="us.openai.gpt-5.6-luna",
                help=(
                    "Judge model id. Under --sdpo-judge-backend=bedrock this is a Bedrock modelId; "
                    "luna needs a cross-region inference profile, so use the 'us.'-prefixed id "
                    "(bare 'openai.gpt-5.6-luna' is not on-demand invocable and is auto-upgraded). "
                    "Under backend=openai it is the id served by --sdpo-judge-base-url."
                ),
            )
            parser.add_argument(
                "--sdpo-judge-region",
                type=str,
                default="us-west-2",
                help="AWS region for --sdpo-judge-backend=bedrock.",
            )
            parser.add_argument(
                "--sdpo-judge-api-key-env",
                type=str,
                default="OPENAI_API_KEY",
                help=(
                    "Env var holding the judge API key (may be unset/EMPTY for a keyless proxy). "
                    "Only used when --sdpo-judge-backend=openai; Bedrock uses the IAM role."
                ),
            )
            parser.add_argument(
                "--sdpo-judge-max-concurrency",
                type=int,
                default=32,
                help="Max concurrent judge requests (bounds load on the gateway).",
            )
            parser.add_argument(
                "--sdpo-judge-max-tokens",
                type=int,
                default=2048,
                help=(
                    "max_completion_tokens for the judge call. gpt-5* are reasoning models that "
                    "spend tokens on internal reasoning before the verdict, so keep this generous."
                ),
            )
            parser.add_argument(
                "--sdpo-answer-tag",
                type=str,
                default="answer",
                help=(
                    "XML tag the model wraps its final answer in (<answer>...</answer>). The judge "
                    "extracts this as the model's answer; an empty/missing tag counts as incorrect."
                ),
            )
            parser.add_argument(
                "--sdpo-search-judge-fallback",
                action="store_true",
                default=False,
                help=(
                    "Search/QA domain double-check: EM (exact-match against golden_answers) grades "
                    "first as usual; ONLY when EM says 'wrong' does this ask the LLM judge (same "
                    "--sdpo-judge-base-url/-model/-api-key-env gateway as --sdpo-judge) for a second "
                    "opinion before finalizing 'incorrect'. Reduces EM's false negatives (a correct "
                    "answer phrased differently than the one golden string it happens to compare "
                    "against) without touching an EM hit or spending judge calls on already-correct "
                    "traces. Independent of --sdpo-judge (which replaces math/mcq grading entirely); "
                    "this only ever narrows search's 'incorrect' set, never widens it beyond EM's own "
                    "'correct' set."
                ),
            )
            parser.add_argument(
                "--sdpo-grader",
                type=str,
                choices=["mcq", "dapo"],
                default="mcq",
                help=(
                    "How the SDPO group RM grades trace correctness (to pick correct-peer "
                    "prefixes). 'mcq' (default): <answer>-letter match / math-verl grading for the "
                    "SciKnowEval MCQ dataset. 'dapo': DAPO math grader (integer boxed answers) for "
                    "the zhuzilin/dapo-math-17k dataset."
                ),
            )
            # ---- Trace condensation / SkillOpt (examples/SRD/sdpo.py) ----------
            parser.add_argument(
                "--sdpo-trace-condense",
                action="store_true",
                default=False,
                help=(
                    "Before splicing the correct-peer solution into the teacher prompt, distill it "
                    "into a short transferable SKILL (<=3 procedural bullets, no answer) via an "
                    "OpenAI-compatible LLM, and use that skill as the teacher prefix instead of the "
                    "full trace (SkillOpt-style, matches lasgroup/SDPO trace_condense). Falls back "
                    "to the full trace on any condensation failure."
                ),
            )
            parser.add_argument(
                "--sdpo-condense-base-url",
                type=str,
                default="https://api.openai.com/v1",
                help="OpenAI-compatible base URL for the skill condenser (default: OpenAI API).",
            )
            parser.add_argument(
                "--sdpo-condense-model",
                type=str,
                default="gpt-5.4-mini",
                help="Model id used to distill traces into skills.",
            )
            parser.add_argument(
                "--sdpo-condense-api-key-env",
                type=str,
                default="OPENAI_API_KEY",
                help="Env var holding the condenser API key.",
            )
            parser.add_argument(
                "--sdpo-condense-max-tokens",
                type=int,
                default=2048,
                help="max_completion_tokens for the skill condenser call (gpt-5* spend tokens on reasoning).",
            )
            parser.add_argument(
                "--sdpo-condense-max-concurrency",
                type=int,
                default=32,
                help="Max concurrent skill-condenser requests (bounds gateway load).",
            )
            # ---- Self-generated skill + skill-SDPO (examples/SRD/sdpo.py) ------
            parser.add_argument(
                "--sdpo-self-skill",
                action="store_true",
                default=False,
                help=(
                    "The CURRENT policy self-generates the skill during rollout (an extra "
                    "generation against the rollout engine with a skill-gen prompt), and that "
                    "on-policy skill replaces the peer trace as the response-SDPO teacher prefix. "
                    "Unlike --sdpo-trace-condense (external LLM), the skill is trainable — see "
                    "--sdpo-skill-kd. Mutually exclusive with --sdpo-trace-condense."
                ),
            )
            parser.add_argument(
                "--sdpo-skill-kd",
                action="store_true",
                default=False,
                help=(
                    "Add a SECOND KD objective on the self-generated skill's own tokens (requires "
                    "--sdpo-self-skill). Same divergence/IS machinery as the response KD, on the "
                    "(skill-gen prompt, skill) pair with a teacher hint (see --sdpo-skill-kd-mode)."
                ),
            )
            parser.add_argument(
                "--sdpo-skill-kd-coef",
                type=float,
                default=1.0,
                help="Weight of the skill KD term (independent of the response --sdpo-kd-coef).",
            )
            parser.add_argument(
                "--sdpo-skill-kd-mode",
                type=str,
                choices=[
                    "self-success",
                    "problem-only",
                    "pitfall-condense",
                    "both",
                    "blind-correct",
                    "both-blind",
                ],
                default="self-success",
                help=(
                    "Which skills get skill-KD and what the teacher hint is. 'self-success': only "
                    "when the sample itself answered correctly; teacher hint = the sample's OWN "
                    "correct trace (student prompt already states the solution is correct, so the "
                    "student/teacher information gap is small -- see 'blind-correct' below). "
                    "'problem-only': any sample, teacher = skill-gen prompt with NO hint "
                    "(regularization toward the EMA teacher's skill, no correct-answer info). "
                    "'pitfall-condense': for FAILED traces (skill-source incorrect|all); teacher = "
                    "the problem-solving prompt with the trace's own generated pitfalls as the "
                    "privileged hint, distilling the group's per-trace pitfalls. 'both' (requires "
                    "skill-source all): correct traces use the self-success solution-skill KD AND "
                    "failed traces use the pitfall-condense KD, so both skill flavors are trained. "
                    "'blind-correct': symmetric counterpart to pitfall-condense for CORRECT traces "
                    "(requires skill-source correct|all) -- student regenerates a knowledge "
                    "prediction from the PROBLEM ONLY (no solution, no attempt), teacher = same "
                    "problem-only prompt + the trace's own correct solution as privileged info, "
                    "giving a genuine (not one-sentence) information gap like pitfall-condense's. "
                    "'both-blind' (requires skill-source all): blind-correct on correct traces AND "
                    "pitfall-condense on failed traces -- the fully-symmetric version of 'both', "
                    "isolating whether self-success's weak KD signal (vs pitfall-condense's strong "
                    "one) explains an observed correct-vs-pitfall skill-KD contribution imbalance."
                ),
            )
            parser.add_argument(
                "--sdpo-blind-correct-info",
                type=str,
                choices=["trace", "group-skills"],
                default="trace",
                help=(
                    "WHAT the blind-correct (KNOWLEDGE-foresight) teacher gets as privileged "
                    "info, under --sdpo-skill-kd-mode blind-correct|both-blind. 'trace' "
                    "(default, original behaviour): the trace's OWN full correct solution. "
                    "'group-skills': every correct trace's HINDSIGHT skill in the group, "
                    "concatenated -- the exact analogue of what pitfall-condense already does "
                    "on the failure side (all failed traces' per-trace pitfalls concatenated). "
                    "Motivation: the foresight student predicts general, transferable knowledge, "
                    "so a teacher conditioned on transferable skills distilled from the whole "
                    "group is a better-matched target than one conditioned on a single "
                    "problem-specific solution trace -- and it is group-shared, so the two KD "
                    "channels become symmetric in aggregation as well as in prompt shape. "
                    "Also cheaper: skills are far shorter than full traces (capped by "
                    "--sdpo-max-prefix-chars). Falls back to 'trace' per sample if no correct "
                    "trace in the group produced a hindsight skill."
                ),
            )
            parser.add_argument(
                "--sdpo-pitfall-summary-backend",
                type=str,
                choices=["self", "external"],
                default="self",
                help=(
                    "Second-stage aggregation of a group's per-trace pitfalls into one shared "
                    "'common failure lessons' list. 'self' (default): the current policy over the "
                    "rollout engine (on-policy, same generator as self-skill). 'external': the "
                    "OpenAI-compatible LLM (--sdpo-condense-* endpoint). Only used when self-skill "
                    "covers incorrect traces (--sdpo-skill-source incorrect|all)."
                ),
            )
            parser.add_argument(
                "--sdpo-skill-max-new-tokens",
                type=int,
                default=512,
                help="max_new_tokens for the self-skill generation call during rollout.",
            )
            parser.add_argument(
                "--sdpo-eval-skill-mode",
                type=str,
                choices=["off", "correct", "pitfall", "all"],
                default="off",
                help=(
                    "EVAL-time skill augmentation: before the real eval rollout, self-generate a "
                    "blind, problem-only skill (same self-predict prompts as training's blind-"
                    "correct/pitfall-condense skill-gen) and splice it into the eval prompt's user "
                    "turn, so eval measures the model answering WITH its own self-predicted skill "
                    "already in context -- not auto-derived from --sdpo-skill-kd-mode, set it "
                    "manually to match whichever skill type(s) that run actually trained. 'off' "
                    "(default): no augmentation. 'correct': self-predict the knowledge/rules skill "
                    "only (e.g. for a run trained with skill-kd-mode self-success/blind-correct). "
                    "'pitfall': self-predict the pitfalls-to-avoid skill only (e.g. for "
                    "pitfall-condense). 'all': self-predict BOTH and concatenate both (e.g. for "
                    "both/both-blind). Requires wiring --custom-generate-function-path "
                    "examples.SRD.sdpo.sdpo_eval_generate; a no-op during training regardless."
                ),
            )
            parser.add_argument(
                "--sdpo-skill-source",
                type=str,
                choices=["correct", "incorrect", "env_feedback", "all"],
                default="correct",
                help=(
                    "Which rollout traces get a self-generated skill (and, if eligible, skill-KD). "
                    "Does NOT change the response-SDPO teacher prefix (always a correct peer). "
                    "'correct': only traces the sample answered correctly. 'incorrect': only wrong "
                    "traces. 'env_feedback': only wrong traces that have a populated "
                    "sample.metadata['tool_trace'] (a rollout that called a tool, e.g. "
                    "examples/SRD's code_interpreter) -- the pitfall skill is grounded in "
                    "that trace's actual tool calls/results (see --sdpo-env-feedback-max-chars), "
                    "the direct analogue of lasgroup/SDPO's environment-feedback reprompt. 'all': "
                    "every trace. The skill is generated from the trace's own response as the "
                    "worked solution."
                ),
            )
            parser.add_argument(
                "--sdpo-env-feedback-max-chars",
                type=int,
                default=2000,
                help=(
                    "Truncation budget (characters, keeping the TAIL) for the rendered tool-"
                    "execution trace spliced into the --sdpo-skill-source env_feedback pitfall "
                    "prompt. Analogous to lasgroup/SDPO's max_reprompt_len/reprompt_truncation."
                ),
            )
            parser.add_argument(
                "--sdpo-max-prefix-chars",
                type=int,
                default=20000,
                help=(
                    "Truncation budget (characters, keeping the TAIL) for a multi-turn peer "
                    "trace's reframed prose (_reframe_messages_to_prose) before it is spliced "
                    "into the teacher prompt as the correct-peer prefix. Without this, one "
                    "pathologically long peer trace (e.g. a code-domain trace with a long "
                    "debugging loop) becomes the teacher prefix for every OTHER sample in its "
                    "group, and a single such sample can't be split across dynamic-batch-size "
                    "microbatches -- it OOMs the vocab-parallel forward alone. 0 = no cap."
                ),
            )
            parser.add_argument(
                "--sdpo-prefer-tool-use-peer",
                action="store_true",
                default=False,
                help=(
                    "When picking a trace's correct-peer teacher prefix, prefer a correct peer "
                    "whose sample.metadata['tool_call_count'] > 0 (falls back to uniform random "
                    "among ALL correct peers if none used a tool). For agentic examples where "
                    "tool use is optional (e.g. examples/SRD), uniform random selection "
                    "creates a feedback loop: no-tool traces are often correct more often (less "
                    "risk of a mid-trace tool error), so the KD teacher prefix drifts toward "
                    "no-tool text over training, which then teaches the student to skip the tool "
                    "even more -- observed as a rising rollout/agentic/zero_tool_call_frac despite "
                    "task reward never penalizing tool use. No effect when no sample in the group "
                    "carries a tool_call_count key (e.g. non-agentic examples)."
                ),
            )
            parser.add_argument(
                "--sdpo-response-prefix",
                type=str,
                choices=["trace", "skill"],
                default="trace",
                help=(
                    "What the RESPONSE-SDPO teacher prefix is: 'trace' (default) = the correct "
                    "peer's full solution (base SDPO); 'skill' = that peer's self-generated skill "
                    "instead (requires --sdpo-self-skill and the peer to have a skill; falls back "
                    "to the full trace when the peer has none). Lets the teacher's privileged info "
                    "be the distilled skill rather than the worked solution."
                ),
            )
            # ---- RLSD: RLVR with Self-Distillation (arXiv:2604.03128) ----------
            # RLSD replaces SDPO's distribution-matching KD loss with a MULTIPLICATIVE
            # reweighting of the GRPO advantage: direction still comes exclusively from
            # the environment reward (sign of A), while the self-teacher's evidence
            # ratio P_T(y_t)/P_S(y_t) only modulates magnitude -- avoiding the KD loss's
            # privileged-information leakage (a wrong trace can never be pulled toward
            # tokens the teacher favors). Reuses SDPO's Megatron self-teacher/peer-
            # prefix plumbing (--sdpo-teacher-backend megatron); needs the SAMPLED-token
            # teacher log-prob, not a top-k distribution, so pair with
            # --sdpo-logprob-mode sampled. See miles/backends/training_utils/
            # loss_hub/rlsd.py.
            parser.add_argument(
                "--sdpo-rlsd",
                action="store_true",
                default=False,
                help=(
                    "Enable RLSD: multiplicatively reweight the GRPO advantage by the "
                    "self-teacher's per-token evidence ratio instead of adding a KD loss "
                    "(--sdpo-kd-loss) or an additive KL penalty (--use-opd). Requires "
                    "--sdpo-teacher-backend megatron and --sdpo-logprob-mode sampled "
                    "(reads rollout_data['teacher_log_probs']); mutually exclusive with "
                    "--sdpo-kd-loss and --use-opd."
                ),
            )
            parser.add_argument(
                "--sdpo-rlsd-clip-eps",
                type=float,
                default=0.2,
                help=(
                    "eps_w: clip the per-token evidence weight w_t to [1-eps_w, 1+eps_w] "
                    "before it reweights the advantage (matches the paper's eps_w=0.2, "
                    "the trust-region analogue of GRPO's importance-ratio clip)."
                ),
            )
            parser.add_argument(
                "--sdpo-rlsd-lambda-init",
                type=float,
                default=0.5,
                help=(
                    "Initial mixing coefficient lambda for A_hat_t = A * ((1-lambda) + "
                    "lambda * clip(w_t, ...)): lambda=0 is plain GRPO (uniform advantage), "
                    "lambda=1 is fully reweighted. Matches the paper's lambda=0.5 start."
                ),
            )
            parser.add_argument(
                "--sdpo-rlsd-lambda-warmup-steps",
                type=int,
                default=50,
                help=(
                    "Linearly decay lambda from --sdpo-rlsd-lambda-init to 0 over this "
                    "many rollouts (matches the paper's 50-step decay), so training "
                    "settles into plain GRPO rather than an abrupt on/off transition. "
                    "0 disables decay (lambda stays at its init value)."
                ),
            )
            # ---- EPO: PMI-credit self-distillation (examples/EPO/epo.py) --------
            # EPO decouples SDPO's teacher-vs-student divergence into a CREDIT weight
            # (|logp_with_privileged_context - logp_without|, i.e. the pointwise
            # mutual information a response token carries about the ground-truth
            # solution) and a DIRECTION that comes from the task outcome reward
            # instead of the teacher's KL direction. Reuses the SAME teacher
            # plumbing as SDPO (--sdpo-ema-teacher / --sdpo-teacher-backend /
            # sdpo_teacher_prompt_tokens) and the SAME GRPO advantage estimator; the
            # only new mechanism is the credit weight fused into the advantages.
            parser.add_argument(
                "--epo-credit-loss",
                action="store_true",
                default=False,
                help=(
                    "Enable EPO: compute per-token PMI credit_t = "
                    "|logp(y_t | x, f, y_<t>) - logp(y_t | x, y_<t>)| under the SDPO "
                    "teacher weights (EMA teacher if --sdpo-ema-teacher, else the live "
                    "self-teacher), and multiply it elementwise into the GRPO "
                    "advantages (reward - baseline) computed by "
                    "--advantage-estimator grpo. Requires "
                    "--custom-rm-path examples.EPO.epo.epo_group_reward (or an "
                    "equivalent reward fn that stashes sdpo_teacher_prompt_tokens) so "
                    "a correct-peer/ground-truth prefix is available for the "
                    "privileged-context forward pass."
                ),
            )
            parser.add_argument(
                "--epo-credit-clip",
                type=float,
                default=5.0,
                help="Clamp credit_t to this max value before normalization/fusion (0 = no clamp).",
            )
            parser.add_argument(
                "--epo-credit-normalize",
                action=argparse.BooleanOptionalAction,
                default=True,
                help=(
                    "Normalize credit_t to a local (per-DP-rank rollout) mean of 1 over "
                    "active response tokens before fusing into advantages, so the "
                    "credit weight only reshapes WHERE the gradient lands within a "
                    "trace, not the overall reward scale. --no-epo-credit-normalize "
                    "uses the raw (clamped) credit value."
                ),
            )
            parser.add_argument(
                "--epo-credit-mode",
                type=str,
                choices=["abs_logp_diff", "topk_divergence"],
                default="abs_logp_diff",
                help=(
                    "How credit_t is computed from the two (with/without privileged-"
                    "context) forwards. 'abs_logp_diff' (default, cheap): "
                    "|logp_plus - logp_minus| on the SAMPLED token only -- the literal "
                    "pointwise-log-likelihood-ratio reading of the PMI definition. "
                    "'topk_divergence': a distribution-level divergence (see "
                    "--epo-credit-divergence) between the with- and without-context "
                    "top-k next-token distributions, reusing SDPO's "
                    "_sdpo_topk_distribution_divergence -- captures how much the WHOLE "
                    "next-token belief shifts, not just the sampled token, at the cost "
                    "of a second top-k forward (compute_topk_logprobs instead of "
                    "compute_log_prob) and an extra --opd-log-prob-top-k top-k gather."
                ),
            )
            parser.add_argument(
                "--epo-credit-divergence",
                type=str,
                choices=["reverse_kl", "forward_kl", "jeffrey", "jsd"],
                default="jsd",
                help=(
                    "Divergence used when --epo-credit-mode topk_divergence: matches "
                    "_sdpo_topk_distribution_divergence's mode strings (any value other "
                    "than reverse_kl/forward_kl/jeffrey falls through to jsd). Only "
                    "read when --epo-credit-mode is topk_divergence."
                ),
            )
            parser.add_argument(
                "--epo-credit-skill",
                action="store_true",
                default=False,
                help=(
                    "Use the peer's SELF-GENERATED SKILL (a condensed solution roadmap, "
                    "same mechanism as SDPO's --sdpo-self-skill / --sdpo-response-prefix "
                    "skill) as the privileged context f, instead of the peer's full raw "
                    "trace. The skill is generated on the fly from the correct peer's "
                    "own response via _self_generate_skill (an extra rollout-engine "
                    "generation call per correct trace), then spliced in exactly like "
                    "the full-trace prefix. A peer with no successfully-generated skill "
                    "falls back to its full trace. Purely a diagnostic/ablation toggle -- "
                    "does not touch SDPO's --sdpo-skill-kd (no second KD objective on "
                    "the skill's own tokens; EPO has no KD-loss term to begin with)."
                ),
            )
            parser.add_argument(
                "--epo-credit-token-diagnostics",
                action="store_true",
                default=False,
                help=(
                    "Split credit_t by a cheap regex-based token category "
                    "(epistemic/strategy/compute/format/other -- see "
                    "miles.utils.token_category.classify_response_tokens) and log the "
                    "per-category mean credit as rollout/epo_credit_{category}_mean. "
                    "Answers 'does epistemic-token credit collapse first?' "
                    "(diagnosis plan experiment 1.1) as a free training-time panel "
                    "instead of a separate offline script. Costs one "
                    "convert_ids_to_tokens call per sample per rollout (cheap vocab "
                    "lookup, not detokenization) -- off by default since it is a "
                    "diagnostic, not needed for training."
                ),
            )
            return parser

        def add_lora_arguments(parser):
            """Add LoRA-related arguments for Megatron backend."""
            parser.add_argument(
                "--lora-rank",
                type=int,
                default=0,
                help="LoRA rank. Set to 0 to disable LoRA (default: 0)",
            )
            parser.add_argument(
                "--lora-alpha",
                type=int,
                default=16,
                help="LoRA alpha for scaling (default: 16)",
            )
            parser.add_argument(
                "--lora-dropout",
                type=float,
                default=0.0,
                help="LoRA dropout rate (default: 0.0)",
            )
            parser.add_argument(
                "--lora-type",
                type=str,
                default="lora",
                choices=["lora", "canonical_lora"],
                help="LoRA variant to use: 'lora' (standard) or 'canonical_lora' (split Q/K/V) (default: lora)",
            )
            parser.add_argument(
                "--target-modules",
                type=str,
                default=None,
                help="Target modules for LoRA. Use 'all-linear' or comma-separated module names "
                "(e.g., 'q_proj,k_proj,v_proj,o_proj' for HF naming or 'linear_qkv,linear_proj' for Megatron naming)",
            )
            parser.add_argument(
                "--exclude-modules",
                type=str,
                default=None,
                help="Modules to exclude from LoRA (comma-separated)",
            )
            parser.add_argument(
                "--lora-adapter-path",
                type=str,
                default=None,
                help="Path to load pre-trained LoRA adapter weights (default: None)",
            )
            parser.add_argument(
                "--lora-sync-from-tensor",
                action="store_true",
                default=False,
                help="Sync LoRA weights via tensor instead of file (more efficient)",
            )
            parser.add_argument(
                "--lora-base-cpu-backup",
                action="store_true",
                default=False,
                help=(
                    "LoRA + colocate: keep SGLang-side CPU mirror of base weights "
                    "and skip per-step base sync. Trades host RAM for faster "
                    "onload/offload. Ignored unless --colocate and LoRA are both on."
                ),
            )
            parser.add_argument(
                "--experts-shared-outer-loras",
                action="store_true",
                default=False,
                help="Enable shared-outer grouped-expert LoRA (gate_up lora_A and "
                "down lora_B shared across experts, expert_dim=1). Matches SGLang "
                "PR #21466's experts_shared_outer_loras=True serving contract.",
            )
            return parser

        def add_router_arguments(parser):
            parser.add_argument(
                "--use-miles-router",
                action="store_true",
                default=False,
                help="Whether to use MilesRouter for text-based routing instead of SGLang token-based routing",
            )
            parser.add_argument(
                "--miles-router-timeout",
                type=float,
                default=None,
                help="Timeout for MilesRouter HTTP requests in seconds.",
            )
            parser.add_argument(
                "--miles-router-max-connections",
                type=int,
                default=None,
                help="Max connections for MilesRouter HTTP client.",
            )
            parser.add_argument(
                "--miles-router-health-check-failure-threshold",
                type=int,
                default=3,
                help="Number of consecutive failures before marking a worker as unhealthy.",
            )
            RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
            return parser

        # wandb
        def add_wandb_arguments(parser):
            # wandb parameters
            parser.add_argument("--use-wandb", action="store_true", default=False)
            parser.add_argument(
                "--wandb-mode",
                type=str,
                default=None,
                choices=["online", "offline", "disabled"],
                help="W&B mode: online (default), offline (local only), or disabled. Overrides WANDB_MODE env var.",
            )
            parser.add_argument(
                "--wandb-dir",
                type=str,
                default=None,
                help="Directory to store wandb logs. Default is ./wandb in current directory.",
            )
            parser.add_argument("--wandb-key", type=str, default=None)
            parser.add_argument("--wandb-host", type=str, default=None)
            parser.add_argument("--wandb-team", type=str, default=None)
            parser.add_argument("--wandb-group", type=str, default=None)
            reset_arg(parser, "--wandb-project", type=str, default=None)
            parser.add_argument(
                "--disable-wandb-random-suffix",
                action="store_false",
                dest="wandb_random_suffix",
                default=True,
                help=(
                    "Whether to add a random suffix to the wandb run name. "
                    "By default, we will add a random 6 length string with characters to the run name."
                ),
            )
            parser.add_argument(
                "--wandb-always-use-train-step",
                action="store_true",
                default=False,
                help=(
                    "Whether to always use train step as the step metric in wandb. "
                    "If set, we will always use the train steps for wandb logging, "
                    "otherwise, will use rollout step for most info other than train/*. "
                ),
            )
            parser.add_argument(
                "--log-multi-turn",
                action="store_true",
                default=False,
                help="Whether to log information for multi-turn rollout.",
            )
            parser.add_argument(
                "--log-passrate",
                action="store_true",
                default=False,
                help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout.",
            )
            parser.add_argument(
                "--log-reward-category",
                type=str,
                default=None,
                help=(
                    "Log statistics of the category of reward, such as why the reward function considers it as failed. "
                    "Specify the key in the reward dict using this argument."
                ),
            )
            parser.add_argument(
                "--log-correct-samples",
                action="store_true",
                default=False,
                help="Explicitly log metrics for correct samples.",
            )
            parser.add_argument("--wandb-run-id", type=str, default=None)
            return parser

        # mlflow
        def add_mlflow_arguments(parser):
            parser.add_argument("--use-mlflow", action="store_true", default=False)
            parser.add_argument(
                "--mlflow-tracking-uri",
                type=str,
                default=None,
                help="MLflow tracking server URI. Defaults to MLFLOW_TRACKING_URI env var, or local mlruns/ directory.",
            )
            parser.add_argument(
                "--mlflow-experiment-name",
                type=str,
                default="miles",
                help="MLflow experiment name.",
            )
            parser.add_argument(
                "--mlflow-run-name",
                type=str,
                default=None,
                help="MLflow run name. Defaults to --wandb-group if not set.",
            )
            parser.add_argument("--mlflow-run-id", type=str, default=None)
            return parser

        # tensorboard
        def add_tensorboard_arguments(parser):
            # tb_project_name, tb_experiment_name
            parser.add_argument("--use-tensorboard", action="store_true", default=False)
            parser.add_argument(
                "--tb-project-name",
                type=str,
                default=None,
                help="Directory to store tensorboard logs. Default is  os.environ.get('TENSORBOARD_DIR') directory.",
            )
            parser.add_argument("--tb-experiment-name", type=str, default=None)

            return parser

        # prometheus
        def add_prometheus_arguments(parser):
            parser.add_argument("--use-prometheus", action="store_true", default=False)
            parser.add_argument(
                "--prometheus-port",
                type=int,
                default=int(os.environ.get("PROMETHEUS_PORT", "9090")),
                help="Port for the Prometheus metrics HTTP server. "
                "Prometheus scrapes /metrics on this port. "
                "Defaults to PROMETHEUS_PORT env var or 9090.",
            )
            parser.add_argument(
                "--prometheus-run-name",
                type=str,
                default=None,
                help="Human-readable run name attached as a 'run_name' label to all "
                "Prometheus metrics. Used to distinguish runs in Grafana. "
                "Defaults to --wandb-group if set.",
            )
            return parser

        # debug
        def add_debug_arguments(parser):
            parser.add_argument(
                "--save-debug-rollout-data",
                type=str,
                default=None,
                help=(
                    "Save the rollout data to this path for debugging. "
                    "The file will be saved to `save_debug_rollout_data.format(rollout_id)`."
                ),
            )
            parser.add_argument(
                "--load-debug-rollout-data",
                type=str,
                default=None,
                help=(
                    "Load the rollout data from this path for debugging. "
                    "The file will be loaded from `load_debug_rollout_data.format(rollout_id)`. "
                    "When this is enabled, miles will not instantiate sglang servers."
                ),
            )
            parser.add_argument(
                "--load-debug-rollout-data-subsample",
                type=float,
                default=None,
                help="Subsample a portion of the debug rollout data for faster debugging.",
            )
            parser.add_argument(
                "--debug-rollout-only",
                action="store_true",
                default=False,
                help=(
                    "Whether to only run the rollout generation without training. "
                    "This is useful for debugging the rollout generation function."
                ),
            )
            parser.add_argument(
                "--debug-train-only",
                action="store_true",
                default=False,
                help=(
                    "Whether to only run the training without sglang servers. "
                    "This is useful for debugging the rollout generation function."
                ),
            )
            parser.add_argument(
                "--save-debug-train-data",
                type=str,
                default=None,
                help=(
                    "Save the train data to this path for debugging. "
                    "The file will be saved to `save_debug_train_data.format(rollout_id)`."
                ),
            )
            parser.add_argument(
                "--dump-details",
                type=str,
                default=None,
                help=("Dump all details of training for post-hoc analysis and visualization."),
            )
            parser.add_argument(
                "--dump-train-data",
                action=argparse.BooleanOptionalAction,
                default=True,
                help=(
                    "Whether --dump-details also dumps per-rank train_data/*.pt (heavy per-token "
                    "training tensors). --no-dump-train-data skips it (keeps rollout_data + sdpo dumps)."
                ),
            )
            parser.add_argument(
                "--dump-policy-loss-debug",
                action=argparse.BooleanOptionalAction,
                default=True,
                help=(
                    "Whether --dump-details also dumps policy_loss_debug/*.pt (per-call loss debug "
                    "tensors). --no-dump-policy-loss-debug skips it."
                ),
            )
            parser.add_argument(
                "--dumper-enable",
                action="store_true",
                default=False,
                help="Enable sglang dumper for all three phases (sglang inference, "
                "megatron forward-only, megatron forward-backward). "
                "Per-phase --dumper-inference/--dumper-fwd-only/--dumper-fwd-bwd can override.",
            )
            parser.add_argument(
                "--dumper-dir",
                type=str,
                default="/tmp/dumper",
                help="Base output directory for sglang dumper. Three subdirs are created: "
                "inference/, fwd_only/, fwd_bwd/.",
            )
            parser.add_argument(
                "--dumper-inference",
                nargs="*",
                default=None,
                help="SGLang inference phase dumper config as key=value pairs. "
                "Keys map to DumperConfig fields (e.g. enable=true filter=whatever).",
            )
            parser.add_argument(
                "--dumper-fwd-only",
                nargs="*",
                default=None,
                help="Megatron forward-only phase dumper config as key=value pairs.",
            )
            parser.add_argument(
                "--dumper-fwd-bwd",
                nargs="*",
                default=None,
                help="Megatron forward-backward phase dumper config as key=value pairs.",
            )
            parser.add_argument(
                "--dumper-source-patcher-config-inference",
                type=str,
                default=None,
                help="Path to YAML config file for source patcher applied in SGLang inference engines.",
            )
            parser.add_argument(
                "--dumper-source-patcher-config-train",
                type=str,
                default=None,
                help="Path to YAML config file for source patcher applied in Megatron training actors.",
            )
            # use together with --record-memory-history and --memory-snapshot-path (defined in Megatron)
            parser.add_argument(
                "--memory-snapshot-dir",
                type=str,
                default=".",
            )
            parser.add_argument(
                "--memory-snapshot-num-steps",
                type=int,
                default=None,
            )
            parser.add_argument(
                "--profile-target",
                type=str,
                choices=["train_overall", "train_actor", "train_log_probs"],
                default=["train_overall"],
                nargs="+",
            )
            parser.add_argument(
                "--memory-recorder",
                type=str,
                choices=["torch", "memray"],
                default="torch",
            )
            parser.add_argument("--check-weight-update-equal", action="store_true")
            parser.add_argument(
                "--check-weight-update-selector",
                type=str,
                default="all",
                choices=["all", "target", "draft"],
                help="Which model the post-update equality check covers: 'all' (target + "
                "draft/MTP), 'target' (target model only; skips the draft, e.g. when MTP "
                "training is off), or 'draft' (draft/MTP worker only).",
            )
            parser.add_argument(
                "--check-weight-update-skip-list",
                type=str,
                nargs="*",
                default=None,
                help="Weight-name substrings to exclude from the post-update equality check; "
                "their mismatches are downgraded to non-fatal info (e.g. MTP/draft layer names "
                "that are absent on the training side).",
            )
            parser.add_argument(
                "--check-weight-update-allow-quant-error",
                action="store_true",
                help="When comparing weights after update, allow quantized tensors to differ "
                "by up to 1 ULP of the quantized dtype per side (compared in dequantized space).",
            )
            parser.add_argument(
                "--env-report",
                type=str,
                default=os.environ.get("MILES_SCRIPT_ENV_REPORT", ""),
                help="JSON string containing environment report from external launcher.",
            )
            return parser

        def add_network_arguments(parser):
            parser.add_argument("--http-proxy", type=str, default=None)
            parser.add_argument("--use-distributed-post", action="store_true", default=False)
            return parser

        def add_reward_model_arguments(parser):
            parser.add_argument(
                "--rm-type",
                type=str,
                default=None,
                help="Type of the reward model",
            )
            parser.add_argument(
                "--reward-key",
                type=str,
                default=None,
                help=(
                    "Some reward model may return a dict instead of a value, "
                    "this is the key to extract the reward value from the dict. "
                ),
            )
            parser.add_argument(
                "--eval-reward-key",
                type=str,
                default=None,
                help="The eval variant for --reward-key",
            )
            parser.add_argument(
                "--group-rm", action="store_true", default=False, help="Whether to do rm on a whole group."
            )
            parser.add_argument(
                "--rm-url",
                type=str,
                default=None,
                help="URL for the reward model service for --rm-type remote_rm, e.g. http://localhost:8000",
            )
            parser.add_argument(
                "--custom-rm-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom reward model function. "
                    "If set, we will use this function to calculate the reward instead of the default one. "
                    "The function should have the signature `def custom_rm(args, sample) -> float`."
                ),
            )
            parser.add_argument(
                "--eval-custom-rm-path",
                type=str,
                default=None,
                help=(
                    "Path to a per-sample reward function used only during eval rollout, with the "
                    "signature `async def eval_rm(args, sample) -> float`. Set this to run eval while "
                    "training uses a group RM (--group-rm): the group RM cannot score eval samples "
                    "(there is no group step in eval), so this per-sample function grades them instead. "
                    "Without it, eval under --group-rm is disallowed."
                ),
            )
            parser.add_argument(
                "--custom-reward-post-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom function that will post process reward, by default it will be the normalization for grpo. "
                ),
            )
            parser.add_argument(
                "--custom-convert-samples-to-train-data-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom function that converts samples to training data. "
                    "If set, this function will replace the default _convert_samples_to_train_data. "
                    "The function should have the signature `def convert_samples_to_train_data(args, samples) -> dict`."
                ),
            )
            return parser

        def add_rollout_buffer_arguments(parser):
            parser.add_argument(
                "--rollout-buffer-url",
                type=str,
                default=None,
                help="URL for the rollout buffer",
            )

            parser.add_argument(
                "--fetch-trajectory-retry-times",
                type=int,
                default=-1,
                help="Number of times to retry fetching trajectory, -1 means unlimited retry",
            )
            parser.add_argument(
                "--min-batch-collection-ratio",
                type=float,
                default=1,
                help="Minimum batch collection ratio",
            )
            parser.add_argument(
                "--rollout-task-type",
                type=str,
                default="math",
            )
            parser.add_argument(
                "--loss-mask-type",
                type=str,
                default="qwen",
                choices=["qwen", "qwen3", "distill_qwen"],
                help="Loss mask type",
            )
            parser.add_argument(
                "--data-pad-size-multiplier",
                type=int,
                default=128,
                help="Multiplier for data padding size in data processing.",
            )
            parser.add_argument(
                "--rollout-sample-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout sample filter function. "
                    "This function determines whether a sample will participate in loss calculation. "
                    "The function is called as `fn(args, data)` where `data` is `list[list[Sample]]` "
                    "(grouped by n_samples_per_prompt), and should return None. "
                    "To exclude a sample from the loss, set `sample.remove_sample = True`. "
                    "Note: This attribute does not determine whether the sample participates in advantage normalization."
                ),
            )
            parser.add_argument(
                "--rollout-all-samples-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout all samples process function that "
                    "can process all samples including filtered ones."
                ),
            )
            parser.add_argument(
                "--disable-rollout-trim-samples",
                action="store_true",
                default=False,
                help="disable trim samples in rollout buffer when converting samples to train data",
            )
            parser.add_argument(
                "--use-dynamic-global-batch-size",
                action="store_true",
                default=False,
                help="enable dynamic global batch size, disable trim samples in rollout buffer when converting samples to train data",
            )
            return parser

        def add_custom_megatron_plugins_arguments(parser):
            """
            Add custom Megatron plugins arguments.
            This is a placeholder for any additional arguments that might be needed.
            """
            # Custom arguments can be added here
            parser.add_argument(
                "--freeze-indexer",
                action="store_true",
                default=False,
            )
            parser.add_argument(
                "--custom-megatron-init-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-log-prob-hook-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-train-step-hook-path",
                type=str,
                default=None,
            )
            return parser

        def add_mtp_training_arguments(parser):
            """Add MTP training specific arguments."""
            reset_arg(parser, "--mtp-num-layers", type=int, default=None)
            reset_arg(parser, "--mtp-loss-scaling-factor", type=float, default=0.2)
            parser.add_argument(
                "--enable-mtp-training",
                action="store_true",
                default=False,
                help="Enable MTP layer parameter updates during training",
            )

            return parser

        def add_prefill_decode_disaggregation_arguments(parser):
            parser.add_argument(
                "--prefill-num-servers",
                type=int,
                default=None,
                help="Number of prefill servers for disaggregation.",
            )
            return parser

        def add_ci_arguments(parser):
            parser.add_argument(
                "--ci-test",
                action="store_true",
            )
            parser.add_argument(
                "--ci-disable-kl-checker",
                action="store_true",
            )
            parser.add_argument(
                "--ci-disable-logprobs-checker",
                action="store_true",
            )
            parser.add_argument(
                "--ci-metric-checker-key",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--ci-metric-checker-threshold",
                type=float,
                default=None,
            )
            parser.add_argument(
                "--ci-save-grad-norm",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--ci-load-grad-norm",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--ci-save-model-hash",
                action="store_true",
            )
            parser.add_argument(
                "--ci-check-model-hash",
                action="store_true",
            )
            return parser

        def add_session_arguments(parser):
            parser.add_argument(
                "--use-session-server",
                action="store_true",
                default=False,
                help="Start a standalone session server for TITO/session support. "
                "Requires --hf-checkpoint and --chat-template-path to also be set.",
            )
            parser.add_argument(
                "--session-server-ip",
                type=str,
                default=None,
                help="IP address of the standalone session server. Defaults to sglang-router-ip.",
            )
            parser.add_argument(
                "--session-server-port",
                type=int,
                default=None,
                help="Port of the standalone session server. Auto-allocated if not set.",
            )
            parser.add_argument(
                "--tito-model",
                type=str,
                default="default",
                choices=[t.value for t in TITOTokenizerType],
                help="TITO tokenizer type for pretokenized prefix reuse. "
                "Controls how token IDs are computed for messages appended after "
                "the pretokenized prefix in multi-turn agentic sessions.",
            )
            parser.add_argument(
                "--tito-allowed-append-roles",
                nargs="+",
                default=["tool"],
                choices=["tool", "user", "system"],
                help="Message roles allowed to be appended after the pretokenized "
                "assistant prefix in TITO sessions (default: tool).",
            )
            return parser

        def add_user_provided_function_arguments(parser):
            try:
                args_partial, _ = parser.parse_known_args()
            except SystemExit:
                return parser
            for path in [
                args_partial.rollout_function_path,
                args_partial.custom_generate_function_path,
            ]:
                try:
                    fn = load_function(path)
                except (ModuleNotFoundError, ValueError):
                    continue
                if fn is not None and callable(getattr(fn, "add_arguments", None)):
                    fn.add_arguments(parser)
            return parser

        # Add custom arguments in front to prevent overwritten some miles arguments.
        if add_custom_arguments is not None:
            parser = add_custom_arguments(parser)

        parser = add_cluster_arguments(parser)
        parser = add_train_arguments(parser)
        parser = add_rollout_arguments(parser)
        parser = add_fault_tolerance_arguments(parser)
        parser = add_data_arguments(parser)
        parser = add_eval_arguments(parser)
        parser = add_algo_arguments(parser)
        parser = add_on_policy_distillation_arguments(parser)
        parser = add_lora_arguments(parser)
        parser = add_wandb_arguments(parser)
        parser = add_mlflow_arguments(parser)
        parser = add_tensorboard_arguments(parser)
        parser = add_prometheus_arguments(parser)
        parser = add_router_arguments(parser)
        parser = add_debug_arguments(parser)
        parser = add_sglang_arguments(parser)
        # required whenever expert projections are LoRA targets, inert otherwise
        # (sglang's own default is False)
        parser.set_defaults(sglang_lora_use_virtual_experts=True)
        parser.add_argument(
            "--no-sglang-lora-use-virtual-experts",
            dest="sglang_lora_use_virtual_experts",
            action="store_false",
            help="Serve MoE-expert LoRA through sglang's fused_moe_lora alignment path instead "
            "of the virtual-experts path.",
        )
        parser = add_session_arguments(parser)
        parser = add_network_arguments(parser)
        parser = add_reward_model_arguments(parser)
        parser = add_rollout_buffer_arguments(parser)
        parser = add_mtp_training_arguments(parser)
        parser = add_prefill_decode_disaggregation_arguments(parser)
        parser = add_ci_arguments(parser)
        parser = add_custom_megatron_plugins_arguments(parser)
        if enable_experimental_rollout_refactor():
            parser = add_user_provided_function_arguments(parser)

        reset_arg(
            parser,
            "--custom-config-path",
            type=str,
            default=None,
            help="Path to the YAML config for custom function arguments.",
        )
        reset_arg(parser, "--padded-vocab-size", type=int, default=None)

        return parser

    return add_miles_arguments


def parse_args(add_custom_arguments=None):
    # Users may call `parse_args` very early, thus we ensure logger is configured here
    configure_logger()

    add_miles_arguments = get_miles_extra_args_provider(add_custom_arguments)

    backend = parse_args_train_backend()
    if backend == "megatron":
        from miles.backends.megatron_utils.arguments import parse_args as megatron_parse_args
        from miles.backends.megatron_utils.arguments import set_default_megatron_args
        from miles.backends.megatron_utils.arguments import validate_args as megatron_validate_args

        args = megatron_parse_args(extra_args_provider=add_miles_arguments)
        args.compress_ratios = None
        if args.hf_checkpoint:
            hf_config = load_hf_config(args.hf_checkpoint)
            args.compress_ratios = getattr(hf_config, "compress_ratios", None)
            hf_validate_args(args, hf_config)

            if is_dsa(hf_config):
                args.indexer_rope_interleave = bool(getattr(hf_config, "indexer_rope_interleave", False))
                logger.info(f"Setting indexer_rope_interleave: {args.indexer_rope_interleave} into args")

        args.rank = 0
        args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
        args = set_default_megatron_args(args)
    else:
        from miles.backends.experimental.fsdp_utils.arguments import load_fsdp_args

        args = load_fsdp_args(extra_args_provider=add_miles_arguments)
        args.rank = 0  # Primary process rank for wandb initialization
        args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node

        assert args.context_parallel_size == 1, "Context parallelism is not supported for FSDP backend."

        if not args.ci_test:
            raise ValueError(
                "The FSDP backend has known issues with SGLang v0.5.10 and is not actively maintained in the current version. "
                "It has been moved to miles.backends.experimental. "
                "Contributions are welcome if you are interested in improving it."
            )

    miles_validate_args(args)

    if backend == "megatron":
        megatron_validate_args(args)

        # always use varlen
        args.variable_seq_lengths = True
        if getattr(args, "moe_token_dispatcher_type", None) == "allgather":
            logger.info(
                "--moe-token-dispatcher-type allgather does not support variable sequence length, "
                "please use alltoall dispatcher instead."
            )
            args.moe_token_dispatcher_type = "alltoall"

        if args.pipeline_model_parallel_size == 1:
            assert args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None, (
                "decoder_first_pipeline_num_layers and decoder_last_pipeline_num_layers should be None when "
                "pipeline_model_parallel_size is 1."
            )

    sglang_validate_args(args)

    return args


def parse_args_train_backend():
    if os.environ.get("MILES_BACKEND") is not None:
        raise Exception("`MILES_BACKEND` is deprecated, please use --train-backend directly.")

    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args_partial, _ = parser.parse_known_args()
    return args_partial.train_backend


def _resolve_eval_datasets(args) -> list[EvalDatasetConfig]:
    """
    Build evaluation dataset configurations from either --eval-config or --eval-prompt-data.
    """
    datasets_config = []
    defaults: dict[str, Any] = {}

    if args.eval_config:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.eval_config)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            raise ValueError("--eval-config must contain a mapping at the root.")

        eval_cfg = cfg_dict.get("eval", cfg_dict)
        if not isinstance(eval_cfg, dict):
            raise ValueError("--eval-config must define an `eval` mapping or be a mapping itself.")

        defaults = dict(eval_cfg.get("defaults") or {})
        datasets_config = ensure_dataset_list(eval_cfg.get("datasets"))
        if not datasets_config:
            raise ValueError("--eval-config does not define any datasets under `eval.datasets`.")
    elif args.eval_prompt_data:
        values = list(args.eval_prompt_data)
        if len(values) == 1:
            logger.info("[legacy] only one eval_prompt_data detected, will assume it is data for aime")
            values = ["aime", values[0]]
        if len(values) % 2 != 0:
            raise ValueError("eval prompt data must be provided as name/path pairs.")
        datasets_config = [{"name": values[i], "path": values[i + 1]} for i in range(0, len(values), 2)]
    else:
        datasets_config = []

    eval_datasets = build_eval_dataset_configs(args, datasets_config, defaults)
    if eval_datasets:
        args.eval_prompt_data = [item for dataset in eval_datasets for item in (dataset.name, dataset.path)]
    else:
        args.eval_prompt_data = None

    return eval_datasets


def miles_validate_args(args):
    args.eval_datasets = _resolve_eval_datasets(args)

    if args.recompute_logprobs_via_prefill:
        assert args.true_on_policy_mode, "--recompute-logprobs-via-prefill requires --true-on-policy-mode"

    # Normalize --tito-allowed-append-roles: lowercase + deduplicate.
    raw_roles = getattr(args, "tito_allowed_append_roles", ["tool"])
    args.tito_allowed_append_roles = sorted(set(r.lower() for r in raw_roles))

    if not args.use_session_server:
        misconfigured = []
        if args.tito_model != TITOTokenizerType.DEFAULT.value:
            misconfigured.append(f"--tito-model={args.tito_model}")
        if args.tito_allowed_append_roles != ["tool"]:
            misconfigured.append(f"--tito-allowed-append-roles={args.tito_allowed_append_roles}")
        if misconfigured:
            raise ValueError(
                f"{', '.join(misconfigured)} require --use-session-server; "
                "these flags only configure the session-server TITO middleware."
            )

    if "user" in args.tito_allowed_append_roles:
        logger.warning(
            "--tito-allowed-append-roles includes 'user'. "
            "Incremental tokenization assumes appended messages do not change how "
            "earlier turns render, which may not hold for user messages on "
            "context-sensitive chat templates (e.g. last_query_index logic, "
            "thinking-token trimming). This can cause input_ids to diverge from "
            "the canonical template output. Use at your own risk."
        )

    if args.debug_disable_optimizer:
        args.no_load_optim = True
        args.no_save_optim = True

    # Normalize the deprecated ``--chat-template-path=autofix`` alias to None
    # up-front so the rest of this block treats it as "no path given".
    if args.chat_template_path == "autofix":
        logger.warning(
            "--chat-template-path=autofix is deprecated; remove the flag and rely "
            "on --tito-model + --tito-allowed-append-roles to auto-resolve. The "
            "alias will be removed in a future release."
        )
        args.chat_template_path = None

    # Auto-resolve a bundled fixed chat-template only when:
    #   1. the caller did NOT pass --chat-template-path (an explicit path always
    #      wins and is never overridden)
    #   2. the caller chose a non-default --tito-model family (DEFAULT means
    #      "use the model's native HF chat template", which is loaded by
    #      load_tokenizer — no override needed here)
    should_auto_resolve = args.chat_template_path is None and args.tito_model != TITOTokenizerType.DEFAULT.value

    if should_auto_resolve:
        tito_model = TITOTokenizerType(args.tito_model)
        from miles.utils.chat_template_utils import resolve_fixed_chat_template

        resolved_path, resolved_kwargs = resolve_fixed_chat_template(
            tito_model,
            allowed_append_roles=args.tito_allowed_append_roles,
        )
        if resolved_path is not None:
            args.chat_template_path = resolved_path
        # Merge inferred kwargs.  User-explicit values win on conflict; only
        # keys the user did not set are auto-filled.
        if resolved_kwargs:
            user_kwargs = args.apply_chat_template_kwargs or {}
            for key, value in resolved_kwargs.items():
                if key in user_kwargs:
                    continue
                user_kwargs[key] = value
                logger.warning(
                    "Auto-set --apply-chat-template-kwargs %s=%r for tito_model=%s "
                    "(allowed_append_roles=%s); pass an explicit value to override.",
                    key,
                    value,
                    tito_model.value,
                    sorted(args.tito_allowed_append_roles),
                )
            args.apply_chat_template_kwargs = user_kwargs

    if args.chat_template_path is not None:
        if not os.path.isfile(args.chat_template_path):
            raise FileNotFoundError(f"--chat-template-path file not found: {args.chat_template_path}")
        args.sglang_chat_template = args.chat_template_path

    if args.kl_coef != 0 or args.use_kl_loss:
        if not os.path.exists(args.ref_load):
            raise FileNotFoundError(f"ref_load {args.ref_load} does not exist, please check the path.")

        if not os.path.exists(os.path.join(args.ref_load, "latest_checkpointed_iteration.txt")):
            logger.info(
                f"ref_load {args.ref_load} does not have latest_checkpointed_iteration.txt, "
                "please make sure it is a valid megatron checkpoint directory."
            )

    # Validate RLSD (--sdpo-rlsd): needs the sampled-token self-teacher log-prob
    # (rollout_data["teacher_log_probs"]), which only the Megatron self-teacher's
    # "sampled" logprob-mode path writes (see actor.py::_compute_sdpo_teacher_log_probs);
    # the "topk"/KD-loss path stashes a top-k distribution target instead
    # (sdpo_teacher_topk_logprobs/ids), which RLSD's advantage-reweighting formula
    # does not consume. Mutually exclusive with --sdpo-kd-loss/--use-opd: all three
    # are alternative ways of turning the same teacher-vs-student divergence into a
    # training signal (additive KD loss / additive KL-in-advantage / multiplicative
    # advantage-reweighting) and combining them double-counts the same divergence.
    if args.sdpo_rlsd:
        if getattr(args, "sdpo_teacher_backend", "sglang") != "megatron":
            raise ValueError("--sdpo-rlsd requires --sdpo-teacher-backend megatron.")
        if getattr(args, "sdpo_logprob_mode", "topk") != "sampled":
            raise ValueError("--sdpo-rlsd requires --sdpo-logprob-mode sampled.")
        if getattr(args, "sdpo_kd_loss", False):
            raise ValueError("--sdpo-rlsd and --sdpo-kd-loss are mutually exclusive (alternative KD mechanisms).")
        if args.use_opd:
            raise ValueError("--sdpo-rlsd and --use-opd are mutually exclusive (alternative KD mechanisms).")

    # --sdpo-eval-skill-mode only takes effect through a custom eval generate
    # function that actually performs the skill-splice; a mode set without
    # wiring one of those would silently no-op every eval. Two known
    # implementations share this contract (self-predict + splice, then
    # dispatch to the real rollout): examples.SRD.sdpo.sdpo_eval_generate
    # (single-turn) and examples.SRD.sdpo_react.
    # sdpo_react_eval_generate_with_skill (multi-turn tool-calling, so
    # agentic domains like webshop/alfworld keep their tool loop instead of
    # silently degrading to one text completion).
    _EVAL_SKILL_GENERATE_FN_PATHS = (
        "examples.SRD.sdpo.sdpo_eval_generate",
        "examples.SRD.sdpo_react.sdpo_react_eval_generate_with_skill",
    )
    if getattr(args, "sdpo_eval_skill_mode", "off") != "off":
        eval_datasets = getattr(args, "eval_datasets", None) or []
        wired = getattr(args, "custom_generate_function_path", None) in _EVAL_SKILL_GENERATE_FN_PATHS or any(
            getattr(d, "custom_generate_function_path", None) in _EVAL_SKILL_GENERATE_FN_PATHS
            for d in eval_datasets
        )
        if not wired:
            raise ValueError(
                "--sdpo-eval-skill-mode is set but --custom-generate-function-path is not wired "
                f"(globally or per eval dataset) to one of {_EVAL_SKILL_GENERATE_FN_PATHS} -- "
                "the mode would silently no-op every eval rollout."
            )

    # Validate on-policy distillation (OPD) arguments
    if args.use_opd:
        if args.opd_type is None:
            raise ValueError("--opd-type must be specified when --use-opd is enabled. Choose 'sglang' or 'megatron'.")
        if args.opd_log_prob_top_k < 0:
            raise ValueError("--opd-log-prob-top-k must be non-negative.")
        if args.opd_log_prob_top_k > 0 and args.opd_type != "sglang":
            raise ValueError("--opd-log-prob-top-k is currently supported only with --opd-type=sglang.")

        if args.opd_type == "megatron":
            if args.opd_teacher_load is None:
                raise ValueError(
                    "--opd-teacher-load is required when --opd-type=megatron. "
                    "Please provide the path to the teacher model checkpoint."
                )
            if not os.path.exists(args.opd_teacher_load):
                raise FileNotFoundError(
                    f"opd_teacher_load {args.opd_teacher_load} does not exist, please check the path."
                )
            if not os.path.exists(os.path.join(args.opd_teacher_load, "latest_checkpointed_iteration.txt")):
                logger.info(
                    f"opd_teacher_load {args.opd_teacher_load} does not have latest_checkpointed_iteration.txt, "
                    "please make sure it is a valid megatron checkpoint directory."
                )

        elif args.opd_type == "sglang":
            if args.opd_teacher_load is not None:
                raise ValueError(
                    "--opd-teacher-load should not be set when --opd-type=sglang. "
                    "In sglang mode, teacher log-probs are obtained from external server during rollout."
                )
    else:
        if args.opd_teacher_load is not None:
            raise ValueError("--opd-teacher-load is set but --use-opd is not enabled. Please add --use-opd flag.")

    # TODO: During loading, we need to set the start_rollout_id here.
    if args.megatron_to_hf_mode == "bridge":
        # Fresh runs pass a not-yet-created `--load` dir; fall back to the reference
        # weights (loaded via the HF bridge) instead of asserting in load_checkpoint.
        # Mirrors the non-bridge branch below.
        if (
            args.load is None
            or not os.path.exists(args.load)
            or not os.path.exists(os.path.join(args.load, "latest_checkpointed_iteration.txt"))
        ):
            args.load = args.ref_load or args.hf_checkpoint
        args.start_rollout_id = 0
    else:
        if (
            args.load is None
            or not os.path.exists(args.load)
            or not os.path.exists(os.path.join(args.load, "latest_checkpointed_iteration.txt"))
        ):
            args.no_load_optim = True
            args.no_load_rng = True
            args.finetune = True
            args.load = args.ref_load
            if args.ref_ckpt_step is not None:
                args.ckpt_step = args.ref_ckpt_step
            args.start_rollout_id = 0

    if args.eval_interval is not None:
        assert args.eval_datasets, "Evaluation datasets must be configured when eval_interval is set."

    if args.save_interval is not None:
        assert args.save is not None, "'--save' is required when save_interval is set."

    # Parse LoRA target modules
    if args.lora_rank > 0:
        assert args.target_modules is not None, "'--target-modules' is required when LoRA is enabled."

        if args.target_modules == "all-linear":
            # MLA projections are HF-config-gated (SGLang sizes LoRA buffers per module name;
            # listing them on a dense model crashes the engine). The DSA indexer stays excluded.
            modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            hf_config = load_hf_config(args.hf_checkpoint)
            if getattr(hf_config, "kv_lora_rank", None):
                modules += ["kv_a_proj_with_mqa", "kv_b_proj"]
                if getattr(hf_config, "q_lora_rank", None):
                    modules += ["q_a_proj", "q_b_proj"]
        elif "," in args.target_modules:
            modules = [m.strip() for m in args.target_modules.split(",")]
        else:
            modules = [args.target_modules]

        if args.exclude_modules:
            exclude_set = (
                set(m.strip() for m in args.exclude_modules.split(","))
                if "," in args.exclude_modules
                else {args.exclude_modules}
            )
            modules = [m for m in modules if m not in exclude_set]

        args.target_modules = modules

        # Training and serving must agree on shared-outer grouped-expert LoRA
        # (expert_dim=1 buffers in SGLang).
        if args.experts_shared_outer_loras:
            args.sglang_experts_shared_outer_loras = True
        assert args.experts_shared_outer_loras == bool(
            args.sglang_experts_shared_outer_loras
        ), "experts_shared_outer_loras and sglang_experts_shared_outer_loras must agree"

        # the two MoE-expert adapter layouts are not checkpoint-compatible; say which one runs
        _expert_leaves = ("linear_fc1", "linear_fc2", "gate_proj", "up_proj", "down_proj")
        if any(leaf in str(tm) for tm in modules for leaf in _expert_leaves):
            logger.warning(
                "MoE-expert LoRA layout: %s (--experts-shared-outer-loras).",
                "shared-outer" if args.experts_shared_outer_loras else "per-expert",
            )

    assert not (args.kl_coef != 0 and args.kl_loss_coef != 0), "Only one of kl_coef and kl_loss_coef can be set"

    if args.advantage_estimator in ["reinforce_plus_plus", "reinforce_plus_plus_baseline"]:
        assert args.normalize_advantages, (
            "The 'reinforce_plus_plus' and 'reinforce_plus_plus_baseline' advantage estimators "
            "require advantage normalization. Please add `--normalize-advantages` to your command."
        )

    if args.use_rollout_logprobs:
        assert not args.use_tis, "use_rollout_logprobs and use_tis cannot be set at the same time."

    if args.get_mismatch_metrics:
        assert (
            args.custom_tis_function_path is not None
        ), "custom_tis_function_path must be set when get_mismatch_metrics is set"

        if args.use_rollout_logprobs:
            logger.info(
                "get_mismatch_metrics is set; For metrics calculation, the log probs will still be recomputed by training engine. One more forward pass will be applied."
            )

    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None, "max_tokens_per_gpu must be set when use_dynamic_batch_size is set"
        if args.log_probs_max_tokens_per_gpu is None:
            args.log_probs_max_tokens_per_gpu = args.max_tokens_per_gpu

    if args.eps_clip_high is None:
        args.eps_clip_high = args.eps_clip

    if args.eval_reward_key is None:
        args.eval_reward_key = args.reward_key

    if args.dump_details is not None:
        args.save_debug_rollout_data = f"{args.dump_details}/rollout_data/{{rollout_id}}.pt"
        if getattr(args, "dump_train_data", True):
            args.save_debug_train_data = f"{args.dump_details}/train_data/{{rollout_id}}_{{rank}}.pt"

    if args.load_debug_rollout_data is not None:
        logger.info(
            f"load_debug_rollout_data {args.load_debug_rollout_data} is set, "
            "will not instantiate sglang servers and will only run the training process."
        )
        args.debug_train_only = True

    args.use_critic = args.advantage_estimator == "ppo"
    if args.critic_num_gpus_per_node is None:
        args.critic_num_gpus_per_node = args.actor_num_gpus_per_node
    if args.critic_num_nodes is None:
        args.critic_num_nodes = args.actor_num_nodes
    if args.critic_load is None:
        args.critic_load = args.load
    if args.critic_lr is None:
        args.critic_lr = args.lr

    if args.offload:
        args.offload_train = True
        args.offload_rollout = True
    del args.offload

    if args.debug_rollout_only:
        if args.colocate and (not args.rollout_num_gpus):
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        else:
            args.actor_num_gpus_per_node = min(8, args.rollout_num_gpus)
            args.actor_num_nodes = args.rollout_num_gpus // args.actor_num_gpus_per_node
        args.colocate = False
        args.offload_train = args.offload_rollout = False
        if args.train_memory_margin_bytes > 0:
            logger.warning("Force train_memory_margin_bytes=0 since debug_rollout_only does not support it")
            args.train_memory_margin_bytes = 0

    assert not (args.debug_rollout_only and args.debug_train_only), (
        "debug_rollout_only and debug_train_only cannot be set at the same time, " "please set only one of them."
    )

    if args.ci_test and not args.debug_rollout_only and not args.debug_train_only:
        args.check_weight_update_equal = True

    # always true on offload for colocate at the moment.
    if args.update_weight_transfer_mode == "p2p":
        assert not args.colocate, (
            "P2P weight transfer mode is not compatible with --colocate. "
            "Please use broadcast mode or disable colocate."
        )
        assert (
            getattr(args, "prefill_num_servers", None) is None
        ), "P2P weight transfer mode has not been tested when PD is enabled."
        assert args.lora_rank <= 0, "LoRA weight sync is not supported for p2p (RDMA) weight transfer."

    if args.update_weight_transfer_mode == "disk-delta":
        assert not args.colocate, (
            "Disk-delta weight transfer mode is not compatible with --colocate. Colocate transfers "
            "weights via CUDA IPC (only a handle crosses processes), so the delta bookkeeping "
            "(snapshot + diff + encode) is pure overhead."
        )
        assert (
            getattr(args, "prefill_num_servers", None) is None
        ), "Disk-delta weight transfer mode has not been tested when PD is enabled."
        assert args.lora_rank <= 0, "LoRA weight sync is not supported for disk-delta weight transfer."
        assert args.update_weight_disk_dir, (
            "--update-weight-transfer-mode=disk-delta requires --update-weight-disk-dir to point at "
            "a filesystem shared between the trainer and the rollout engines."
        )
        assert args.update_weight_local_checkpoint_dir, (
            "--update-weight-transfer-mode=disk-delta requires --update-weight-local-checkpoint-dir "
            "(a rollout-host-local directory, e.g. NVMe)."
        )
        assert os.path.isdir(args.hf_checkpoint), (
            "--update-weight-transfer-mode=disk-delta requires --hf-checkpoint to be a local directory: "
            "the baseline snapshot is seeded from its safetensors bytes."
        )

    if args.colocate:
        if args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True
        # Older sglang builds do not expose --sglang-cuda-graph-backend-prefill
        # (their cuda-graph config is nested, not a flat flag), so the attr may be
        # absent. Guard with hasattr so colocate works on either build.
        if not hasattr(args, "sglang_cuda_graph_backend_prefill"):
            pass
        elif args.sglang_cuda_graph_backend_prefill is None:
            args.sglang_cuda_graph_backend_prefill = "disabled"
            logger.info(
                "Colocate mode: defaulting --sglang-cuda-graph-backend-prefill=disabled to avoid NVLS OOM. "
                "Set --sglang-cuda-graph-backend-prefill explicitly to override."
            )
        elif args.sglang_cuda_graph_backend_prefill != "disabled":
            logger.warning(
                f"Warning: colocate mode with --sglang-cuda-graph-backend-prefill="
                f"{args.sglang_cuda_graph_backend_prefill} may trigger NVLS OOM."
            )
        if args.rollout_num_gpus != args.actor_num_gpus_per_node * args.actor_num_nodes:
            logger.info(
                f"rollout_num_gpus {args.rollout_num_gpus} != actor_num_gpus_per_node {args.actor_num_gpus_per_node} "
                f"* actor_num_nodes {args.actor_num_nodes}, overriding rollout_num_gpus to match actor_num_gpus_per_node * actor_num_nodes."
            )
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
            if args.use_critic:
                args.rollout_num_gpus += args.critic_num_gpus_per_node * args.critic_num_nodes

    if args.offload_train is None:
        args.offload_train = False
    if args.offload_rollout is None:
        args.offload_rollout = False

    if args.offload_train:
        args.disable_grad_buffers_cpu_backup = True
        args.disable_param_buffers_cpu_backup = args.enable_weights_backuper

    if args.eval_function_path is None:
        args.eval_function_path = args.rollout_function_path

    if args.num_steps_per_rollout is not None:
        global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt // args.num_steps_per_rollout
        if args.global_batch_size is not None:
            assert args.global_batch_size == global_batch_size, (
                f"global_batch_size {args.global_batch_size} is not equal to "
                f"rollout_batch_size {args.rollout_batch_size} * n_samples_per_prompt {args.n_samples_per_prompt} "
                f"// num_steps_per_rollout {args.num_steps_per_rollout}"
            )
        args.global_batch_size = global_batch_size

    if args.n_samples_per_prompt == 1:
        args.grpo_std_normalization = False
        logger.info("n_samples_per_prompt is set to 1, grpo_std_normalization will be set to False.")

    if args.over_sampling_batch_size is None:
        args.over_sampling_batch_size = args.rollout_batch_size

    assert args.over_sampling_batch_size >= args.rollout_batch_size, (
        f"over_sampling_batch_size {args.over_sampling_batch_size} should be greater than or equal to "
        f"rollout_batch_size {args.rollout_batch_size}"
    )

    if args.num_epoch is not None:
        if args.num_rollout is not None:
            logger.info("Both num_epoch and num_rollout are set, num_epoch will be ignored.")
        else:
            assert args.rollout_global_dataset, (
                "num_epoch is set, but rollout_global_dataset is not set, "
                "please remove --disable-rollout-global-dataset to use num_epoch"
            )
    else:
        # if num_epoch is not set, we should set num_rollout
        assert args.num_rollout is not None, (
            "num_epoch is not set, but num_rollout is not set, " "please set --num-rollout or --num-epoch"
        )

    if args.enable_mtp_training:
        assert args.mtp_num_layers, "mtp_num_layers must be set when enable_mtp_training is set"

    if args.use_rollout_routing_replay:
        args.use_routing_replay = True

    if args.custom_config_path:
        with open(args.custom_config_path) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(args, k):
                logger.info(f"Warning: Argument {k} is already set to {getattr(args, k)}, will override with {v}.")
            setattr(args, k, v)

    if args.use_rollout_indexer_replay:
        args.use_indexer_replay = True
        assert args.context_parallel_size == 1, "indexer replay does not support context parallelism yet"

    if args.eval_max_context_len is None:
        logger.info(
            f"args.eval_max_context_len is not set. Use args.rollout_max_context_len {args.rollout_max_context_len} as default value."
        )
        args.eval_max_context_len = args.rollout_max_context_len

    if args.rollout_max_context_len is not None:
        if args.rollout_max_prompt_len is None:
            args.rollout_max_prompt_len = args.rollout_max_context_len - 1
            logger.info(
                f"args.rollout_max_prompt_len is not set. Use args.rollout_max_context_len - 1 ({args.rollout_max_context_len} - 1) as default value so that there is at least one generated token to compute loss."
            )
        assert (
            args.rollout_max_prompt_len <= args.rollout_max_context_len - 1
        ), f"args.rollout_max_prompt_len ({args.rollout_max_prompt_len}) must be smaller than args.rollout_max_context_len ({args.rollout_max_context_len}) so that there is at least one generated token to compute loss."

    assert not (
        args.prefill_num_servers is not None and args.rollout_external
    ), "prefill_num_servers cannot be set when rollout_external is set."

    assert not (
        getattr(args, "sglang_config", None) is not None and args.rollout_external
    ), "sglang_config cannot be set when rollout_external is set."

    assert not (
        getattr(args, "sglang_config", None) is not None and getattr(args, "prefill_num_servers", None) is not None
    ), "sglang_config and prefill_num_servers are mutually exclusive. Use server_groups in the YAML config instead."

    if args.qkv_format == "bshd":
        assert args.train_backend == "megatron", "bshd format is only supported for megatron backend."
        assert (
            args.use_dynamic_batch_size is False
        ), "Dynamic batch size is not supported for bshd format. Please specify --micro-batch-size instead."

    _maybe_apply_dumper_overrides(args)


def _maybe_apply_dumper_overrides(args) -> None:
    if not args.dumper_enable:
        return

    if args.use_fault_tolerance:
        logger.info("Dumper mode: disabling --use-fault-tolerance to suppress RolloutHealthMonitor heartbeats")
        args.use_fault_tolerance = False

    logger.info("Dumper mode: all heartbeat mechanisms disabled")
    args.router_disable_health_check = True
    args.rollout_health_check_interval = 1e18

    if args.start_rollout_id is None:
        args.start_rollout_id = 0

    args.num_rollout = (args.start_rollout_id or 0) + 1
    logger.info(
        "Dumper mode: forced rollout range [%d, %d), disabled eval and save",
        args.start_rollout_id,
        args.num_rollout,
    )
    args.eval_interval = None
    args.save = None
    args.save_interval = None
    args.save_retain_interval = None


def hf_validate_args(args, hf_config):
    def equal(x, y):
        return x == y

    errors = []

    # multimodal models have different config structure
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config

    if hasattr(hf_config, "rope_parameters") and isinstance(hf_config.rope_parameters, dict):
        if "rope_theta" in hf_config.rope_parameters:
            hf_config.rope_theta = hf_config.rope_parameters["rope_theta"]
        else:
            # Gemma-4 nests rope_theta per attention type; take the first.
            for _entry in hf_config.rope_parameters.values():
                if isinstance(_entry, dict) and "rope_theta" in _entry:
                    hf_config.rope_theta = _entry["rope_theta"]
                    break

    model_name = (args.model_name or "").lower().replace("-", "").replace("_", "")
    if (hf_config.model_type == "deepseek_v4" or "deepseekv4" in model_name) and args.context_parallel_size > 1:
        assert args.allgather_cp, "zigzag CP is not supported for DeepSeek V4."

    for hf_config_name, megatron_config_name, compare_fn in [
        ("hidden_size", "hidden_size", equal),
        ("num_attention_heads", "num_attention_heads", equal),
        ("num_hidden_layers", "num_layers", equal),
        ("intermediate_size", "ffn_hidden_size", equal),
        ("moe_intermediate_size", "moe_ffn_hidden_size", equal),
        ("tie_word_embeddings", "untie_embeddings_and_output_weights", lambda x, y: not x == y),
        (
            "rms_norm_eps",
            "norm_epsilon" if os.getenv("DEPRECATED_MEGATRON_COMPATIBLE", "0") == "1" else "layernorm_epsilon",
            equal,
        ),
        ("rope_theta", "rotary_base", equal),
    ]:
        # FIXME: Qwen3.5 transfomers has bug.
        if getattr(hf_config, "model_type", "") == "qwen3_5_moe_text" and hf_config_name == "intermediate_size":
            continue
        if getattr(hf_config, "model_type", "") == "deepseek_v4" and hf_config_name == "intermediate_size":
            continue
        if hasattr(hf_config, hf_config_name):
            if not compare_fn(getattr(hf_config, hf_config_name), getattr(args, megatron_config_name)):
                errors.append(
                    f"{hf_config_name} in hf config {getattr(hf_config, hf_config_name)} is not equal to "
                    f"{megatron_config_name} {getattr(args, megatron_config_name)}, please check the config."
                )

    if len(errors) > 0:
        raise AssertionError("hf_validate_args failed: " + "; ".join(errors))
