# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
import logging
import math
from collections.abc import Mapping

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import AutoConfig, Gemma3ForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping
from megatron.bridge.models.conversion.transformers_compat import (
    rope_local_base_freq_from_hf,
    rope_scaling_factor_from_hf,
    rope_theta_from_hf,
)
from megatron.bridge.models.gemma.gemma3_provider import Gemma3ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


logger = logging.getLogger(__name__)

AutoMapping.register_module_type("Gemma3TEDotProductAttention", "replicated")
AutoMapping.register_module_type("TERowParallelLinearLayerNorm", "row")


@MegatronModelBridge.register_bridge(
    source=Gemma3ForCausalLM,
    target=GPTModel,
    provider=Gemma3ModelProvider,
    model_type="gemma3",
)
class Gemma3ModelBridge(MegatronModelBridge):
    """Bridge between Hugging Face Gemma-3 text weights and Megatron-Core."""

    def _infer_text_head_dim(self, hf_pretrained: PreTrainedCausalLM) -> int | None:
        if not hasattr(hf_pretrained, "state"):
            return None

        num_attention_heads = getattr(hf_pretrained.config, "num_attention_heads", None)
        num_key_value_heads = getattr(hf_pretrained.config, "num_key_value_heads", None)
        state = hf_pretrained.state

        q_proj_keys = (
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
        )
        for key in q_proj_keys:
            tensor = state.get(key)
            if tensor is not None and num_attention_heads:
                inferred = tensor.shape[0] // num_attention_heads
                if inferred > 0:
                    return inferred

        k_proj_keys = (
            "language_model.model.layers.0.self_attn.k_proj.weight",
            "model.language_model.layers.0.self_attn.k_proj.weight",
        )
        for key in k_proj_keys:
            tensor = state.get(key)
            if tensor is not None and num_key_value_heads:
                inferred = tensor.shape[0] // num_key_value_heads
                if inferred > 0:
                    return inferred

        q_norm_keys = (
            "language_model.model.layers.0.self_attn.q_norm.weight",
            "model.language_model.layers.0.self_attn.q_norm.weight",
        )
        for key in q_norm_keys:
            tensor = state.get(key)
            if tensor is not None:
                return tensor.shape[0]

        return None

    def _get_hf_state_keys(self, hf_state_dict: Mapping[str, torch.Tensor]) -> set[str]:
        cache = getattr(self, "_hf_state_keys_cache", None)
        if cache is None or getattr(self, "_hf_state_keys_cache_source", None) is not hf_state_dict:
            cache = set(hf_state_dict.keys())
            self._hf_state_keys_cache = cache
            self._hf_state_keys_cache_source = hf_state_dict
        return cache

    def _resolve_import_key(self, key: str, hf_state_dict: Mapping[str, torch.Tensor]) -> str:
        state_keys = self._get_hf_state_keys(hf_state_dict)
        if key in state_keys:
            return key

        alias_cache = getattr(self, "_hf_import_alias_cache", None)
        if alias_cache is None:
            alias_cache = {}
            self._hf_import_alias_cache = alias_cache
        cached = alias_cache.get(key)
        if cached is not None and cached in state_keys:
            return cached

        candidates = [key]
        if key.startswith("model.layers."):
            suffix = key[len("model.layers.") :]
            candidates.extend(
                [
                    f"language_model.model.layers.{suffix}",
                    f"model.language_model.layers.{suffix}",
                ]
            )
        elif key.startswith("language_model.model.layers."):
            suffix = key[len("language_model.model.layers.") :]
            candidates.extend(
                [
                    f"model.language_model.layers.{suffix}",
                    f"model.layers.{suffix}",
                ]
            )
        elif key == "model.embed_tokens.weight":
            candidates.extend(
                [
                    "language_model.model.embed_tokens.weight",
                    "model.language_model.embed_tokens.weight",
                ]
            )
        elif key == "language_model.model.embed_tokens.weight":
            candidates.extend(
                [
                    "model.language_model.embed_tokens.weight",
                    "model.embed_tokens.weight",
                ]
            )
        elif key == "model.norm.weight":
            candidates.extend(
                [
                    "language_model.model.norm.weight",
                    "model.language_model.norm.weight",
                ]
            )
        elif key == "language_model.model.norm.weight":
            candidates.extend(
                [
                    "model.language_model.norm.weight",
                    "model.norm.weight",
                ]
            )

        for candidate in candidates[1:]:
            if candidate in state_keys:
                alias_cache[key] = candidate
                logger.info("Resolved Gemma3 HF key alias %s -> %s", key, candidate)
                return candidate

        raise KeyError(f"Could not resolve Gemma3 HF import key '{key}'")

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(hf_param, dict):
            return {
                key: hf_state_dict[self._resolve_import_key(value, hf_state_dict)] for key, value in hf_param.items()
            }
        return hf_state_dict[self._resolve_import_key(hf_param, hf_state_dict)]

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Gemma3ModelProvider:
        hf_config = hf_pretrained.config

        # Flatten nested text_config attributes to top-level for standard bridge mapping
        if hasattr(hf_config, "text_config"):
            logger.info("Flattening nested Gemma3 text_config for conversion...")
            for key in [
                "num_attention_heads",
                "num_key_value_heads",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "head_dim",
                "sliding_window",
                "vocab_size",
                "max_position_embeddings",
            ]:
                if hasattr(hf_config.text_config, key):
                    setattr(hf_config, key, getattr(hf_config.text_config, key))

        provider = super().provider_bridge(hf_pretrained)
        hf_vl_config = AutoConfig.from_pretrained(hf_pretrained._model_name_or_path, trust_remote_code=True)
        params_dtype = self.dtype_from_hf(hf_vl_config, default=torch.float32)
        provider.fp16 = params_dtype == torch.float16
        provider.bf16 = params_dtype == torch.bfloat16
        provider.params_dtype = params_dtype
        provider.autocast_dtype = params_dtype

        # Robust attribute extraction for multimodal/nested configs
        provider.window_size = getattr(
            hf_config, "sliding_window", getattr(getattr(hf_config, "text_config", {}), "sliding_window", 1024)
        )

        inferred_head_dim = self._infer_text_head_dim(hf_pretrained)
        if inferred_head_dim is not None and provider.kv_channels != inferred_head_dim:
            logger.info(
                "Overriding Gemma3 kv_channels from %s to %s based on HF text weights",
                provider.kv_channels,
                inferred_head_dim,
            )
            provider.kv_channels = inferred_head_dim

        # RoPE handling: Gemma 3 needs a tuple (local, global)
        try:
            local_freq = rope_local_base_freq_from_hf(hf_config)
            global_freq = rope_theta_from_hf(hf_config)
            provider.rotary_base = (local_freq, global_freq)
        except Exception:
            # Fallback to defaults if extraction fails
            if not isinstance(provider.rotary_base, tuple):
                val = float(provider.rotary_base) if provider.rotary_base else 1000000.0
                provider.rotary_base = (10000.0, val)

        if hasattr(hf_config, "query_pre_attn_scalar"):
            provider.softmax_scale = 1.0 / math.sqrt(hf_config.query_pre_attn_scalar)

        provider.rope_scaling_factor = rope_scaling_factor_from_hf(hf_config)
        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider: Gemma3ModelProvider) -> dict:
        hf_config = super().megatron_to_hf_config(provider)
        if isinstance(provider.rotary_base, tuple):
            rope_local_base_freq, rope_theta = provider.rotary_base
            hf_config["rope_local_base_freq"] = rope_local_base_freq
            hf_config["rope_theta"] = rope_theta
        hf_config["sliding_window"] = provider.window_size
        if provider.softmax_scale:
            query_pre_attn_scalar = 1.0 / (provider.softmax_scale**2)
            rounded = round(query_pre_attn_scalar)
            if math.isclose(query_pre_attn_scalar, rounded, rel_tol=0.0, abs_tol=1e-9):
                query_pre_attn_scalar = rounded
            hf_config["query_pre_attn_scalar"] = query_pre_attn_scalar
        if getattr(provider, "rope_scaling_factor", 1.0) != 1.0:
            hf_config["rope_scaling"] = {
                "factor": provider.rope_scaling_factor,
                "type": "linear",
            }
        return hf_config

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping = {
            "embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "language_model.model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "language_model.model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "language_model.model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "language_model.model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": (
                "language_model.model.layers.*.post_attention_layernorm.weight"
            ),
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "language_model.model.layers.*.pre_feedforward_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "language_model.model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight": (
                "language_model.model.layers.*.post_feedforward_layernorm.weight"
            ),
            "decoder.final_layernorm.weight": "language_model.model.norm.weight",
        }
        mapping_list = []
        for megatron_param, hf_param in mapping.items():
            resolved_mapping = AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
            resolved_mapping.allow_hf_name_mismatch = True
            mapping_list.append(resolved_mapping)

        qkv_mapping = QKVMapping(
            megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
            q="language_model.model.layers.*.self_attn.q_proj.weight",
            k="language_model.model.layers.*.self_attn.k_proj.weight",
            v="language_model.model.layers.*.self_attn.v_proj.weight",
        )
        qkv_mapping.allow_hf_name_mismatch = True
        mapping_list.append(qkv_mapping)

        gated_mlp_mapping = GatedMLPMapping(
            megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
            gate="language_model.model.layers.*.mlp.gate_proj.weight",
            up="language_model.model.layers.*.mlp.up_proj.weight",
        )
        gated_mlp_mapping.allow_hf_name_mismatch = True
        mapping_list.append(gated_mlp_mapping)

        return MegatronMappingRegistry(*mapping_list)
