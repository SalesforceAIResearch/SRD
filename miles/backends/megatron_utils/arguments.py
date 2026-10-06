import logging
import os

from megatron.training.arguments import parse_args, validate_args

try:
    from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
except ImportError:
    # Megatron-LM trees from ~2026-08-27 on dropped megatron/training/tokenizer/
    # (tokenizers moved under megatron.core.tokenizers) and re-homed the vocab
    # padding helper in megatron/training/vocab_utils.py with an explicit
    # signature instead of an `args` blob. Same arithmetic, so shim it rather
    # than pin the container: without this, `import train` dies at
    # ModuleNotFoundError before parse_args on any newer image.
    from megatron.training.vocab_utils import calculate_padded_vocab_size

    def _vocab_size_with_padding(orig_vocab_size, args, logging_enabled=True):
        return calculate_padded_vocab_size(
            orig_vocab_size,
            args.make_vocab_size_divisible_by,
            args.tensor_model_parallel_size,
            logging_enabled=logging_enabled,
        )

__all__ = ["validate_args", "parse_args", "set_default_megatron_args"]

logger = logging.getLogger(__name__)


def set_default_megatron_args(args):
    # Muon currently owns its sharding path, and Megatron's distributed optimizer
    # only supports Adam-family optimizers.
    args.use_distributed_optimizer = args.optimizer is None or args.optimizer.lower() == "adam"
    # TODO: maybe change this after megatron has good fp8 support
    args.bf16 = not args.fp16
    # placeholders
    if args.seq_length is None:
        args.seq_length = 4096
    args.max_position_embeddings = args.seq_length
    # Notice(Jiajun): new megatron has removed this argument and use dp_reshardable instead of fully_shard
    if os.getenv("DEPRECATED_MEGATRON_COMPATIBLE", "0") == "1":
        args.dist_ckpt_save_pre_mcore_014 = True
    # compatible for megatron
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    if not args.tokenizer_model and not args.tokenizer_type:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"

    if not hasattr(args, "miles_dsa_topk_backend"):
        args.miles_dsa_topk_backend = "torch"

    return args
