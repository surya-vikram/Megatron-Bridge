# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, field

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from transformers import SiglipVisionConfig

from megatron.bridge.models.gemma.gemma3_provider import Gemma3ModelProvider
from megatron.bridge.models.gemma_vl.modeling_gemma3_vl import Gemma3VLModel, Gemma3VLMultimodalProjectorConfig


@dataclass
class Gemma3VLModelProvider(Gemma3ModelProvider):
    """
    Base model provider for Gemma VL Models.
    """

    # VL models shouldn't scatter embeddings across sequence parallel regions because
    # the vision embeddings are going to be inserted into the language embeddings.
    scatter_embedding_sequence_parallel: bool = False

    # Vision configuration
    vision_config: SiglipVisionConfig = field(default_factory=SiglipVisionConfig)
    return_dict: bool = True
    # We need to provide the HF text config in order to initialize the Gemma3MultiModalProjector
    vision_projector_config: Gemma3VLMultimodalProjectorConfig = field(
        default_factory=Gemma3VLMultimodalProjectorConfig
    )
    mm_tokens_per_image: int = 256

    # Token IDs
    bos_token_id: int = 2
    eos_token_id: int = 1
    vision_start_token_id: int = 255999
    vision_end_token_id: int = 256000
    image_token_id: int = 262144

    # Freeze options
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Gemma3VLModel:
        model = Gemma3VLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        # Apply freeze options if any are enabled
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


@dataclass
class Gemma3VLModelProvider4B(Gemma3VLModelProvider):
    """Gemma3 4B VL config"""

    is_vision_language: bool = True
    num_layers: int = 34
    hidden_size: int = 2560
    num_attention_heads: int = 8
    num_query_groups: int = 4
    kv_channels: int = 256
    ffn_hidden_size: int = 10240
    window_size: int = 1024
    rope_scaling_factor: float = 8.0
    vocab_size: int = 262_208

    def __post_init__(self):
        super().__post_init__()
        # Set projector hidden size to match language model hidden size
        if self.vision_projector_config.hidden_size != self.hidden_size:
            self.vision_projector_config.hidden_size = self.hidden_size


@dataclass
class Gemma3VLModelProvider12B(Gemma3VLModelProvider):
    """Gemma3 12B VL config"""

    is_vision_language: bool = True
    num_layers: int = 48
    hidden_size: int = 3840
    num_attention_heads: int = 16
    num_query_groups: int = 8
    kv_channels: int = 256
    ffn_hidden_size: int = 15360
    window_size: int = 1024
    rope_scaling_factor: float = 8.0
    vocab_size: int = 262_208

    def __post_init__(self):
        super().__post_init__()
        if self.vision_projector_config.hidden_size != self.hidden_size:
            self.vision_projector_config.hidden_size = self.hidden_size


@dataclass
class Gemma3VLModelProvider27B(Gemma3VLModelProvider):
    """Gemma3 27B VL config"""

    is_vision_language: bool = True
    num_layers: int = 62
    hidden_size: int = 5376
    num_attention_heads: int = 32
    num_query_groups: int = 16
    kv_channels: int = 128
    ffn_hidden_size: int = 21504
    window_size: int = 1024
    rope_scaling_factor: float = 8.0
    vocab_size: int = 262_208

    def __post_init__(self):
        super().__post_init__()
        if self.vision_projector_config.hidden_size != self.hidden_size:
            self.vision_projector_config.hidden_size = self.hidden_size
