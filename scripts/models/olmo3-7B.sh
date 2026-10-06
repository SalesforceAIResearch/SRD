MODEL_ARGS=(
   # Custom POST-NORM layer spec (Olmo2/Olmo3): pure post-norm + full-dim QK-norm.
   # Megatron's stock spec can't express pure post-norm (it always fuses an input
   # norm into linear_qkv), so we inject our own. Weight loading uses the matching
   # miles_plugins.mbridge.olmo3 bridge.
   --spec "miles_plugins.models.olmo3" "get_olmo3_spec"
   # Olmo-3-7B-Instruct (allenai/Olmo-3-7B-Instruct), config.json:
   #   model_type=olmo3, hidden=4096, layers=32, heads=32, kv_heads=32 (full MHA),
   #   intermediate=11008, vocab=100278, rms_norm_eps=1e-6, rope_theta=500000,
   #   attention_bias=false, tie_word_embeddings=false, head_dim=128 (4096/32).
   # NOTE: Olmo2/Olmo3 use POST-normalization (norm AFTER attn/MLP) + QK-norm.
   # Megatron-bridge ships `olmoe` (MoE) but NOT dense olmo3, so the HF->torch_dist
   # conversion may fail; if it does, dense Olmo3 isn't yet supported here.
   --swiglu
   --num-layers 32
   --hidden-size 4096
   --ffn-hidden-size 11008
   --num-attention-heads 32
   # full MHA (kv_heads == heads); still pass GQA flag with equal groups for uniformity
   --group-query-attention
   --num-query-groups 32
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 500000
   --vocab-size 100278
   # Don't pad the vocab (default rounds up to 100352, a multiple of 128), because
   # mbridge's conversion path leaves make_vocab_size_divisible_by=None and cannot
   # reconcile a padded Megatron embedding (100352) with the HF weight (100278) —
   # the scatter fails on the size mismatch. 100278 is even so TP=2 still splits.
   --make-vocab-size-divisible-by 1
   --kv-channels 128
   --qk-layernorm
   --untie-embeddings-and-output-weights
)
