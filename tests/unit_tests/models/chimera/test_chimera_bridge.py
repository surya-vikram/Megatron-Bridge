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

"""Unit tests for the Chimera model bridge."""

from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.chimera import ChimeraBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


@pytest.fixture
def chimera_config_dict() -> dict:
    """Create the locked Chimera 10B configuration."""
    return {
        "architectures": ["ChimeraForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "first_k_dense_replace": 2,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 8192,
        "last_k_dense_replace": 0,
        "max_position_embeddings": 32768,
        "mlp_bias": False,
        "model_type": "chimera",
        "moe_intermediate_size": 1024,
        "n_group": 1,
        "n_routed_experts": 64,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 16,
        "num_experts_per_tok": 4,
        "num_hidden_layers": 25,
        "num_key_value_heads": 2,
        "original_max_position_embeddings": 8192,
        "pad_token_id": 1,
        "rms_norm_eps": 1e-6,
        "rope_scaling": {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192},
        "rope_theta": 10000000.0,
        "routed_scaling_factor": 1.0,
        "router_aux_loss_coef": 0.001,
        "router_bias_update_rate": 0.0001,
        "scoring_func": "sigmoid",
        "shared_expert_intermediate_size": 1024,
        "tie_word_embeddings": False,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "torch_dtype": "bfloat16",
        "vocab_size": 50176,
    }


@pytest.fixture
def mock_chimera_config(chimera_config_dict: dict) -> Mock:
    """Create a mock Chimera Hugging Face config."""
    config = Mock(spec=list(chimera_config_dict.keys()))
    for key, value in chimera_config_dict.items():
        setattr(config, key, value)
    return config


@pytest.fixture
def mock_pretrained_chimera(mock_chimera_config: Mock) -> Mock:
    """Create a mock PreTrainedCausalLM wrapper for Chimera."""
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = mock_chimera_config
    return pretrained


class TestChimeraBridge:
    """Test ChimeraBridge config and mapping behavior."""

    def test_registration(self) -> None:
        """Test that ChimeraBridge is a MegatronModelBridge."""
        assert issubclass(ChimeraBridge, MegatronModelBridge)

    def test_provider_bridge_maps_locked_config(self, mock_pretrained_chimera: Mock) -> None:
        """Test that the locked Chimera config maps to the expected GPT provider fields."""
        provider = ChimeraBridge().provider_bridge(mock_pretrained_chimera)
        hf_config = mock_pretrained_chimera.config

        assert isinstance(provider, GPTModelProvider)
        assert provider.num_layers == hf_config.num_hidden_layers
        assert provider.hidden_size == hf_config.hidden_size
        assert provider.ffn_hidden_size == hf_config.intermediate_size
        assert provider.moe_ffn_hidden_size == hf_config.moe_intermediate_size
        assert provider.num_attention_heads == hf_config.num_attention_heads
        assert provider.num_query_groups == hf_config.num_key_value_heads
        assert provider.kv_channels == hf_config.head_dim
        assert provider.vocab_size == hf_config.vocab_size
        assert provider.seq_length == hf_config.max_position_embeddings
        assert provider.rotary_base == hf_config.rope_theta
        assert provider.layernorm_epsilon == hf_config.rms_norm_eps
        assert provider.share_embeddings_and_output_weights is False

    def test_provider_bridge_maps_yarn_and_moe(self, mock_pretrained_chimera: Mock) -> None:
        """Test Chimera YaRN and MoE-specific provider fields."""
        provider = ChimeraBridge().provider_bridge(mock_pretrained_chimera)
        hf_config = mock_pretrained_chimera.config

        assert provider.position_embedding_type == "yarn"
        assert provider.yarn_rotary_scaling_factor == 4.0
        assert provider.yarn_original_max_position_embeddings == hf_config.original_max_position_embeddings
        assert provider.yarn_beta_fast == 32.0
        assert provider.yarn_beta_slow == 1.0
        assert provider.yarn_correction_range_round_to_int is False
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.add_qkv_bias is False
        assert provider.add_bias_linear is False
        assert provider.num_moe_experts == hf_config.n_routed_experts
        assert provider.moe_router_topk == hf_config.num_experts_per_tok
        assert provider.moe_router_num_groups == hf_config.n_group
        assert provider.moe_router_group_topk == hf_config.topk_group
        assert provider.moe_router_topk_scaling_factor == hf_config.routed_scaling_factor
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_router_pre_softmax is False
        assert provider.moe_router_enable_expert_bias is True
        assert provider.moe_router_bias_update_rate == 0.0001
        assert provider.moe_router_dtype == "fp32"
        assert provider.moe_aux_loss_coeff == hf_config.router_aux_loss_coef
        assert provider.moe_shared_expert_gate is False
        assert provider.moe_shared_expert_intermediate_size == 1024

    def test_provider_bridge_maps_first_and_last_dense_layers(self, mock_pretrained_chimera: Mock) -> None:
        """Test Chimera's first and last dense layer mask."""
        provider = ChimeraBridge().provider_bridge(mock_pretrained_chimera)

        assert provider.moe_layer_freq == [0] * 2 + [1] * 23

    def test_megatron_to_hf_config_preserves_chimera_fields(self, mock_pretrained_chimera: Mock) -> None:
        """Test Chimera-specific HF config fields are reconstructed on export."""
        provider = ChimeraBridge().provider_bridge(mock_pretrained_chimera)
        hf_config = ChimeraBridge.megatron_to_hf_config(provider)

        assert hf_config["architectures"] == ["ChimeraForCausalLM"]
        assert hf_config["model_type"] == "chimera"
        assert hf_config["first_k_dense_replace"] == 2
        assert hf_config["last_k_dense_replace"] == 0
        assert hf_config["n_routed_experts"] == 64
        assert hf_config["num_experts_per_tok"] == 4
        assert hf_config["n_shared_experts"] == 1
        assert hf_config["moe_intermediate_size"] == 1024
        assert hf_config["shared_expert_intermediate_size"] == 1024
        assert hf_config["router_bias_update_rate"] == 0.0001
        assert hf_config["norm_topk_prob"] is True
        assert hf_config["topk_method"] == "noaux_tc"
        assert hf_config["scoring_func"] == "sigmoid"
        assert hf_config["rope_scaling"]["type"] == "yarn"
        assert hf_config["rope_scaling"]["factor"] == 4.0
        assert hf_config["rope_scaling"]["original_max_position_embeddings"] == 8192
        assert hf_config["torch_dtype"] == "bfloat16"

    def test_mapping_registry_contains_core_mappings(self) -> None:
        """Test that the mapping registry contains Chimera's core parameter mappings."""
        registry = ChimeraBridge().mapping_registry()
        mapping_dict = {mapping.megatron_param: mapping.hf_param for mapping in registry.mappings}

        assert mapping_dict["embedding.word_embeddings.weight"] == "model.embed_tokens.weight"
        assert mapping_dict["output_layer.weight"] == "lm_head.weight"
        assert mapping_dict["decoder.final_layernorm.weight"] == "model.norm.weight"
        assert mapping_dict["decoder.layers.*.self_attention.linear_proj.weight"] == (
            "model.layers.*.self_attn.o_proj.weight"
        )
        assert mapping_dict["decoder.layers.*.mlp.router.weight"] == "model.layers.*.mlp.gate.weight"
        assert mapping_dict["decoder.layers.*.mlp.router.expert_bias"] == (
            "model.layers.*.mlp.gate.e_score_correction_bias"
        )
        assert mapping_dict["decoder.layers.*.mlp.shared_experts.linear_fc2.weight"] == (
            "model.layers.*.mlp.shared_experts.down_proj.weight"
        )

    def test_mapping_registry_resolves_qkv_and_expert_patterns(self) -> None:
        """Test reverse and forward lookups for QKV, expert, and shared expert mappings."""
        registry = ChimeraBridge().mapping_registry()

        qkv_mapping = registry.hf_to_megatron_lookup("model.layers.3.self_attn.q_proj.weight")
        assert qkv_mapping.megatron_param == "decoder.layers.3.self_attention.linear_qkv.weight"

        expert_mapping = registry.hf_to_megatron_lookup("model.layers.3.mlp.experts.7.gate_proj.weight")
        assert expert_mapping.megatron_param == "decoder.layers.3.mlp.experts.linear_fc1.weight7"

        dense_mapping = registry.hf_to_megatron_lookup("model.layers.0.mlp.gate_proj.weight")
        assert dense_mapping.megatron_param == "decoder.layers.0.mlp.linear_fc1.weight"

        shared_mapping = registry.hf_to_megatron_lookup("model.layers.3.mlp.shared_experts.gate_proj.weight")
        assert shared_mapping.megatron_param == "decoder.layers.3.mlp.shared_experts.linear_fc1.weight"

    def test_provider_bridge_dtype(self, mock_pretrained_chimera: Mock) -> None:
        """Test dtype mapping from the HF config."""
        provider = ChimeraBridge().provider_bridge(mock_pretrained_chimera)

        assert provider.bf16 is True
        assert provider.fp16 is False
        assert provider.params_dtype == torch.bfloat16
