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


CHIMERA_CONTEXT_PHASES = {
    "8k": {"max_position_embeddings": 8192, "factor": 1.0},
    "32k": {"max_position_embeddings": 32768, "factor": 4.0},
    "64k": {"max_position_embeddings": 65536, "factor": 8.0},
    "128k": {"max_position_embeddings": 131072, "factor": 16.0},
}
CHIMERA_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS = 8192


def _context_phase(max_position_embeddings: int, factor: float, requested_phase: str | None = None) -> str:
    """Resolve and validate one canonical Chimera YaRN context phase."""
    for phase, geometry in CHIMERA_CONTEXT_PHASES.items():
        if max_position_embeddings == geometry["max_position_embeddings"] and factor == geometry["factor"]:
            if requested_phase is not None and requested_phase != phase:
                raise ValueError(
                    f"Chimera context_phase={requested_phase!r} does not match "
                    f"max_position_embeddings={max_position_embeddings} and factor={factor}."
                )
            return phase
    raise ValueError(
        "Chimera requires one of the canonical YaRN context geometries: "
        f"{CHIMERA_CONTEXT_PHASES}; found max_position_embeddings={max_position_embeddings}, factor={factor}."
    )


@dataclass
class ChimeraModelProvider(GPTModelProvider):
    """GPT provider with Chimera-only, checkpointed inference metadata."""

    # The router correction bias is always part of the checkpoint. This flag only
    # controls whether downstream inference applies that frozen bias at routing time.
    chimera_load_with_bias: bool = True
    chimera_context_phase: str | None = None


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
        if provider.position_embedding_type != "yarn":
            raise ValueError("Chimera supports only position_embedding_type='yarn'.")
        factor = provider.yarn_rotary_scaling_factor
        original_max = provider.yarn_original_max_position_embeddings
        if original_max != CHIMERA_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:
            raise ValueError(f"Chimera requires yarn_original_max_position_embeddings=8192, found {original_max}.")
        phase = _context_phase(provider.seq_length, factor, provider.chimera_context_phase)
        if provider.yarn_correction_range_round_to_int is not False:
            raise ValueError("Chimera requires yarn_correction_range_round_to_int=False.")
        provider_geometry = {
            "rotary_base": provider.rotary_base,
            "rotary_scaling_factor": provider.rotary_scaling_factor,
            "yarn_beta_fast": provider.yarn_beta_fast,
            "yarn_beta_slow": provider.yarn_beta_slow,
            "yarn_mscale": provider.yarn_mscale,
            "yarn_mscale_all_dim": provider.yarn_mscale_all_dim,
            "layernorm_epsilon": provider.layernorm_epsilon,
        }
        expected_geometry = {
            "rotary_base": 10_000_000,
            "rotary_scaling_factor": factor,
            "yarn_beta_fast": 32.0,
            "yarn_beta_slow": 1.0,
            "yarn_mscale": 1.0,
            "yarn_mscale_all_dim": 0.0,
            "layernorm_epsilon": 1e-5,
        }
        if provider_geometry != expected_geometry:
            raise ValueError(
                f"Chimera provider YaRN geometry differs from the canonical values: {provider_geometry}."
            )

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
                "context_phase": phase,
                "position_embedding_type": "yarn",
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
            hf_config["rope_scaling"]["truncate"] = False
            # Transformers 5 exposes both names. Supplying only rope_scaling lets
            # AutoBridge conforming retain stale rope_parameters from an older
            # reference phase (for example factor=1 at 8K while exporting 32K).
            # Emit both representations from the checkpoint-derived geometry.
            rope_parameters = dict(hf_config["rope_scaling"])
            rope_parameters["rope_type"] = rope_parameters.pop("type")
            rope_parameters["rope_theta"] = hf_config["rope_theta"]
            hf_config["rope_parameters"] = rope_parameters

        return hf_config

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> ChimeraModelProvider:
        """Convert a Chimera Hugging Face config to a Megatron GPT model provider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config
        rope_scaling = getattr(hf_config, "rope_scaling", None) or getattr(hf_config, "rope_parameters", None) or {}
        rope_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
        if getattr(hf_config, "position_embedding_type", None) != "yarn" or rope_type != "yarn":
            raise ValueError("Chimera supports only position_embedding_type='yarn' with YaRN rope scaling.")
        factor = rope_scaling.get("factor")
        original_max = rope_scaling.get(
            "original_max_position_embeddings",
            getattr(hf_config, "original_max_position_embeddings", None),
        )
        if original_max != CHIMERA_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:
            raise ValueError(f"Chimera requires original_max_position_embeddings=8192, found {original_max}.")
        if rope_scaling.get("truncate") is not False:
            raise ValueError("Chimera requires rope_scaling.truncate=False.")
        hf_geometry = {
            "rope_theta": hf_config.rope_theta,
            "beta_fast": rope_scaling.get("beta_fast"),
            "beta_slow": rope_scaling.get("beta_slow"),
            "mscale": rope_scaling.get("mscale"),
            "mscale_all_dim": rope_scaling.get("mscale_all_dim"),
            "rms_norm_eps": hf_config.rms_norm_eps,
            "top_level_original": hf_config.original_max_position_embeddings,
        }
        expected_geometry = {
            "rope_theta": 10_000_000.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
            "rms_norm_eps": 1e-5,
            "top_level_original": CHIMERA_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS,
        }
        if hf_geometry != expected_geometry:
            raise ValueError(f"Chimera HF YaRN geometry differs from the canonical values: {hf_geometry}.")
        phase = _context_phase(
            hf_config.max_position_embeddings,
            factor,
            getattr(hf_config, "context_phase", None),
        )

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
        provider.rotary_scaling_factor = factor
        provider.chimera_context_phase = phase

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
