# Olmo3 (dense) mbridge plugin for miles — weight name mapping for HF
# `allenai/Olmo-3-7B-Instruct` (model_type="olmo3") <-> Megatron GPTModel.
#
# The matching POST-NORM layer spec (Olmo3TransformerLayer + full-dim QK-norm
# Olmo3SelfAttention) lives in miles_plugins.models.olmo3 and is shared by both
# the args-driven model build (--spec) and this bridge, so the built architecture
# and the loaded weight names always agree.
#
# Olmo3 specifics reflected in the mapping:
#   * QK-norm  -> self_attention.q_layernorm / k_layernorm
#   * post-attn norm  -> post_self_attn_layernorm  (HF post_attention_layernorm)
#   * post-mlp  norm  -> post_mlp_layernorm        (HF post_feedforward_layernorm)
#   * NO fused input / pre-mlp norms (pure post-norm), so there is NO
#     linear_qkv.layer_norm_weight / linear_fc1.layer_norm_weight entry.

from typing import Optional

from mbridge.core import LLMBridge, register_model

from miles_plugins.models.olmo3 import get_olmo3_spec


@register_model("olmo3")
class Olmo3Bridge(LLMBridge):
    """Bridge for dense Olmo3 (allenai/Olmo-3-7B-Instruct)."""

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }
    # Pure post-norm: NO linear_qkv.layer_norm_weight / linear_fc1.layer_norm_weight.
    # NOTE: mbridge routes a weight to _ATTENTION/_MLP/_OTHER by NAME substring
    # (".self_attention." / "mlp" / else), so post_self_attn_layernorm (which has
    # neither substring) must live in _OTHER_MAPPING; post_mlp_layernorm has the
    # "mlp" substring so it belongs in _MLP_MAPPING.
    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": ["model.layers.{layer_number}.self_attn.o_proj.weight"],
        "self_attention.q_layernorm.weight": ["model.layers.{layer_number}.self_attn.q_norm.weight"],
        "self_attention.k_layernorm.weight": ["model.layers.{layer_number}.self_attn.k_norm.weight"],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
    }
    _MLP_MAPPING = {
        # post-feedforward norm first (its key is a unique substring; safe regardless
        # of dict order since 'post_mlp_layernorm.weight' does not appear in others)
        "post_mlp_layernorm.weight": ["model.layers.{layer_number}.post_feedforward_layernorm.weight"],
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
    }
    # post-attention norm routes here (no ".self_attention." / "mlp" substring).
    _OTHER_MAPPING = {
        "post_self_attn_layernorm.weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
    }

    def _adjust_mapping_for_shared_weights(self):
        if getattr(self.hf_config, "tie_word_embeddings", False):
            self._DIRECT_MAPPING["output_layer.weight"] = "model.embed_tokens.weight"

    def _get_hf_shared_weight_keys(self):
        if getattr(self.hf_config, "tie_word_embeddings", False):
            return ["model.embed_tokens.weight"]
        return []

    def _build_config(self):
        return self._build_base_config(
            add_qkv_bias=False,
            qk_layernorm=True,  # full-dim QK-norm installed via the custom spec
        )

    def _get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        assert self.config.normalization == "RMSNorm", "Olmo3 uses RMSNorm"
        return get_olmo3_spec(None, self.config, vp_stage)
