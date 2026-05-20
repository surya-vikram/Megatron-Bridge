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

import argparse
import logging

import torch
from transformers import AutoConfig

from megatron.bridge import AutoBridge


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for HF to Megatron conversion."""

    parser = argparse.ArgumentParser(description="Import HuggingFace Gemma-3 checkpoint into Megatron format")
    parser.add_argument("--hf-model-path", required=True, help="Path to the source HuggingFace Gemma-3 checkpoint")
    parser.add_argument("--megatron-path", required=True, help="Output Megatron checkpoint directory")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=sorted(DTYPE_MAP.keys()),
        help="Dtype to use while importing weights",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the HuggingFace config/model",
    )
    return parser.parse_args()


def main() -> None:
    """Import a Gemma-3 Hugging Face checkpoint into Megatron format."""

    args = parse_args()
    config = AutoConfig.from_pretrained(args.hf_model_path, trust_remote_code=args.trust_remote_code)
    config.architectures = ["Gemma3ForCausalLM"]

    logging.basicConfig(level=logging.INFO)
    logger.info("Importing Gemma-3 text backbone from %s", args.hf_model_path)
    logger.info("Writing Megatron checkpoint to %s", args.megatron_path)
    logger.info("Forced architecture: %s", config.architectures)

    AutoBridge.import_ckpt(
        args.hf_model_path,
        args.megatron_path,
        config=config,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=DTYPE_MAP[args.torch_dtype],
    )

    logger.info("Successfully converted Gemma-3 checkpoint into %s", args.megatron_path)


if __name__ == "__main__":
    main()
