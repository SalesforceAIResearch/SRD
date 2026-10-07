# Olmo3 (dense) custom layer spec for miles' args-driven ("raw") model build.
#
# Olmo3 is PURE POST-NORM (Olmo2/Olmo3 style):
#     x = x + post_self_attn_layernorm(attn(x))
#     x = x + post_mlp_layernorm(mlp(x))
# with NO input/pre-mlp layernorm. Megatron's stock
# get_gpt_layer_with_transformer_engine_spec always fuses an input norm into
# linear_qkv (column_parallel_layer_norm_linear) and a pre-mlp norm into
# linear_fc1, i.e. it can only do pre-norm or sandwich (GLM4/Gemma), never pure
# post-norm. So we build a custom spec with plain (non-norm-fused) TE linears and
# IdentityOp for the pre-norms, plus TENorm post-norms.
#
# Olmo3 also uses QK-norm over the FULL concatenated dim (num_heads*head_dim),
# like OlMoE — not Megatron's per-head qk_layernorm. Olmo3SelfAttention overrides
# the q/k layernorm sizes and applies them to the flattened q/k before the head
# split.
#
# Wired via:  --spec "miles_plugins.models.olmo3" "get_olmo3_spec"
# Weight loading is handled by the matching mbridge plugin miles_plugins.mbridge.olmo3.

import torch
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.utils import make_viewless_tensor

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TENorm,
    TERowParallelLinear,
)

try:
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    SplitAlongDim = None


class Olmo3SelfAttention(SelfAttention):
    """SelfAttention with Olmo3/OlMoE-style QK-norm over the FULL concatenated
    q/k dim (num_heads*head_dim), applied before the head split."""

    def __init__(self, config, submodules, layer_number, attn_mask_type=AttnMaskType.causal, **kwargs):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            **kwargs,
        )
        # Olmo3 QK-norm is a single RMSNorm over the FULL q/k dim (all heads). Under
        # tensor parallelism the qkv projection is sharded across ranks, so a local
        # RMSNorm would normalize over only the on-rank head shard — different from
        # HF's full-dim RMS. Correct handling would need a cross-TP allreduce of the
        # RMS; instead we require TP=1 (a 7B model fits on one H200), matching how
        # the OlMoE bridge is used.
        assert config.tensor_model_parallel_size == 1, (
            "Olmo3 full-dim QK-norm requires --tensor-model-parallel-size 1 "
            f"(got {config.tensor_model_parallel_size}); use DP instead of TP."
        )
        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=self.hidden_size_per_attention_head * self.config.num_attention_heads,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.k_layernorm = build_module(
            submodules.k_layernorm,
            hidden_size=self.hidden_size_per_attention_head * self.config.num_query_groups,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, **kwargs):
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]
        if SplitAlongDim is not None:
            query, key, value = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:
            query, key, value = torch.split(mixed_qkv, split_arg_list, dim=3)

        sq, b = query.size(0), query.size(1)
        # Full-dim QK-norm (matches HF: q_norm(q_proj(x)), k_norm(k_proj(x))).
        query = self.q_layernorm(query.reshape(sq, b, -1))
        key = self.k_layernorm(key.reshape(sq, b, -1))
        query = query.view(sq, b, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        key = key.view(sq, b, self.num_query_groups_per_partition, self.hidden_size_per_attention_head)
        value = value.view(sq, b, self.num_query_groups_per_partition, self.hidden_size_per_attention_head)
        return query, key, value


class Olmo3TransformerLayer(TransformerLayer):
    """Pure post-norm transformer layer (Olmo2/Olmo3)."""

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        attention_bias=None,
        **kwargs,
    ):
        import inspect

        # Forward only the kwargs the installed Megatron SelfAttention.forward
        # actually accepts (the set of rotary/inference/packed args varies by
        # version; extras like padding_mask/context must be dropped).
        if not hasattr(self, "_attn_forward_params"):
            self._attn_forward_params = set(
                inspect.signature(type(self.self_attention).forward).parameters.keys()
            )
        attn_kwargs = {k: v for k, v in kwargs.items() if k in self._attn_forward_params}

        # ---- Self attention (post-norm) ----
        residual = hidden_states
        ln_out = self.input_layernorm(hidden_states)  # IdentityOp
        attn_owb = self.self_attention(
            ln_out,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            **attn_kwargs,
        )
        attn_out = attn_owb[0] if isinstance(attn_owb, tuple) else attn_owb
        attn_bias = attn_owb[1] if isinstance(attn_owb, tuple) and len(attn_owb) > 1 else None
        if attn_bias is not None:
            attn_out = attn_out + attn_bias
        hidden_states = residual + self.post_self_attn_layernorm(attn_out)

        # ---- MLP (post-norm) ----
        residual = hidden_states
        ln_out = self.pre_mlp_layernorm(hidden_states)  # IdentityOp
        mlp_owb = self.mlp(ln_out)
        mlp_out = mlp_owb[0] if isinstance(mlp_owb, tuple) else mlp_owb
        mlp_bias = mlp_owb[1] if isinstance(mlp_owb, tuple) and len(mlp_owb) > 1 else None
        if mlp_bias is not None:
            mlp_out = mlp_out + mlp_bias
        hidden_states = residual + self.post_mlp_layernorm(mlp_out)

        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)
        return output, context


def get_olmo3_spec(args, config, vp_stage=None) -> ModuleSpec:
    """Custom post-norm layer spec for dense Olmo3."""
    return ModuleSpec(
        module=Olmo3TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp,
            self_attention=ModuleSpec(
                module=Olmo3SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,  # plain: NO fused input norm
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            post_self_attn_layernorm=TENorm,
            pre_mlp_layernorm=IdentityOp,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,  # plain: NO fused pre-mlp norm
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            post_mlp_layernorm=TENorm,
        ),
    )
