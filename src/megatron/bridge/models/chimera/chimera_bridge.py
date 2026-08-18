# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from functools import partial

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


@dataclass
class ChimeraModelProvider(GPTModelProvider):
    """GPT provider with Chimera-only, checkpointed inference metadata."""

    # The router correction bias is always part of the checkpoint. This flag only
    # controls whether downstream inference applies that frozen bias at routing time.
    chimera_load_with_bias: bool = True


@MegatronModelBridge.register_bridge(
    source="ChimeraForCausalLM",
    target=GPTModel,
    provider=ChimeraModelProvider,
    model_type="chimera",
)
class ChimeraBridge(MegatronModelBridge):
    """Megatron Bridge for Chimera sparse MoE causal language models."""

    @classmethod
    def megatron_to_hf_config(cls, provider: ChimeraModelProvider) -> dict:
        """Convert Megatron provider config to a Chimera Hugging Face config dictionary."""
        hf_config = super().megatron_to_hf_config(provider)
        if hf_config.get("rope_theta") is not None:
            hf_config["rope_theta"] = float(hf_config["rope_theta"])

        moe_layer_freq = getattr(provider, "moe_layer_freq", None) or []
        first_k_dense_replace = 0
        for is_moe in moe_layer_freq:
            if is_moe:
                break
            first_k_dense_replace += 1

        last_k_dense_replace = 0
        for is_moe in reversed(moe_layer_freq):
            if is_moe:
                break
            last_k_dense_replace += 1

        shared_size = getattr(provider, "moe_shared_expert_intermediate_size", None)
        moe_ffn = getattr(provider, "moe_ffn_hidden_size", None)
        n_shared_experts = 0
        if shared_size and moe_ffn:
            n_shared_experts = max(1, shared_size // moe_ffn)

        hf_config.update(
            {
                "architectures": ["ChimeraForCausalLM"],
                "first_k_dense_replace": first_k_dense_replace,
                "last_k_dense_replace": last_k_dense_replace,
                "moe_intermediate_size": moe_ffn,
                "n_group": getattr(provider, "moe_router_num_groups", 1) or 1,
                "n_routed_experts": getattr(provider, "num_moe_experts", None),
                "n_shared_experts": n_shared_experts,
                "norm_topk_prob": True,
                "original_max_position_embeddings": getattr(provider, "yarn_original_max_position_embeddings", None),
                "pad_token_id": 1,
                "bos_token_id": 0,
                "eos_token_id": 1,
                "qk_layernorm": getattr(provider, "qk_layernorm", False),
                "load_with_bias": getattr(provider, "chimera_load_with_bias", True),
                "router_aux_loss_coef": getattr(provider, "moe_aux_loss_coeff", 0.0),
                "router_bias_update_rate": getattr(provider, "moe_router_bias_update_rate", 0.0),
                "router_load_balancing_type": getattr(
                    provider, "moe_router_load_balancing_type", "quantile_balancing"
                ),
                "router_z_loss_coef": getattr(provider, "moe_z_loss_coeff", 0.001),
                "moe_qb_num_bins": getattr(provider, "moe_qb_num_bins", 1000),
                "moe_qb_ema_decay": getattr(provider, "moe_qb_ema_decay", 0.0),
                "scoring_func": "sigmoid",
                "shared_expert_intermediate_size": shared_size // n_shared_experts if n_shared_experts else 0,
                "topk_group": getattr(provider, "moe_router_group_topk", 1) or 1,
                "topk_method": "noaux_tc",
            }
        )

        if "rope_scaling" in hf_config:
            hf_config["rope_scaling"]["type"] = hf_config["rope_scaling"].pop(
                "rope_type", hf_config["rope_scaling"].get("type", "yarn")
            )

        return hf_config

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> ChimeraModelProvider:
        """Convert a Chimera Hugging Face config to a Megatron GPT model provider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        provider.transformer_layer_spec = partial(get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE)
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.qk_layernorm = getattr(hf_config, "qk_layernorm", False)
        provider.share_embeddings_and_output_weights = False
        provider.hidden_dropout = 0.0
        provider.autocast_dtype = torch.bfloat16
        provider.persist_layer_norm = True
        provider.bias_activation_fusion = True
        provider.bias_dropout_fusion = True
        provider.yarn_beta_fast = getattr(hf_config, "yarn_beta_fast", None) or 32.0
        provider.yarn_beta_slow = getattr(hf_config, "yarn_beta_slow", None) or 1.0
        provider.yarn_correction_range_round_to_int = False

        provider.moe_grouped_gemm = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = getattr(
            hf_config, "router_load_balancing_type", "quantile_balancing"
        )
        provider.moe_aux_loss_coeff = hf_config.router_aux_loss_coef
        provider.moe_z_loss_coeff = getattr(hf_config, "router_z_loss_coef", 0.001)
        provider.moe_qb_num_bins = getattr(hf_config, "moe_qb_num_bins", 1000)
        provider.moe_qb_ema_decay = getattr(hf_config, "moe_qb_ema_decay", 0.0)
        provider.moe_router_pre_softmax = False
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_dtype = "fp32"
        provider.moe_permute_fusion = True
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_bias_update_rate = getattr(hf_config, "router_bias_update_rate", 0.0)
        provider.chimera_load_with_bias = getattr(hf_config, "load_with_bias", True)
        provider.moe_shared_expert_gate = False
        provider.moe_shared_expert_overlap = hf_config.n_shared_experts > 0
        provider.moe_shared_expert_intermediate_size = (
            hf_config.shared_expert_intermediate_size * hf_config.n_shared_experts
            if hf_config.n_shared_experts > 0
            else None
        )

        first_dense = hf_config.first_k_dense_replace
        last_dense = hf_config.last_k_dense_replace
        moe_layers = hf_config.num_hidden_layers - first_dense - last_dense
        if moe_layers < 0:
            raise ValueError(
                "Chimera dense replacement layers cannot exceed num_hidden_layers: "
                f"{first_dense=} {last_dense=} {hf_config.num_hidden_layers=}."
            )
        provider.moe_layer_freq = [0] * first_dense + [1] * moe_layers + [0] * last_dense

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return parameter mappings between Chimera HF and Megatron-Core GPT formats."""
        hf_config = getattr(self, "hf_config", None)
        qk_layernorm = getattr(hf_config, "qk_layernorm", True) if hf_config is not None else True
        has_shared_experts = getattr(hf_config, "n_shared_experts", 0) > 0 if hf_config is not None else False
        mapping_list = []
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Attention
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            # Dense MLP
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # Router
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
        }

        if qk_layernorm:
            param_mappings.update(
                {
                    "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
                    "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
                }
            )
        if has_shared_experts:
            param_mappings["decoder.layers.*.mlp.shared_experts.linear_fc2.weight"] = (
                "model.layers.*.mlp.shared_experts.down_proj.weight"
            )

        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.extend(
            [
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
            ]
        )

        if has_shared_experts:
            mapping_list.append(
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                )
            )

        return MegatronMappingRegistry(*mapping_list)
