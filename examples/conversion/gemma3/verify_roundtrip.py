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
import math

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for roundtrip verification."""

    parser = argparse.ArgumentParser(description="Verify Gemma-3 HF roundtrip output parity")
    parser.add_argument("--original-path", required=True, help="Path to the original HuggingFace checkpoint")
    parser.add_argument("--roundtrip-path", required=True, help="Path to the roundtrip HuggingFace checkpoint")
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt used for parity verification")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--device", default="cuda", help="Device for generation")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading tokenizer/config/model",
    )
    return parser.parse_args()


def run_inference(
    model_path: str, name: str, prompt: str, max_new_tokens: int, device: str, trust_remote_code: bool
) -> dict[str, list[float] | list[int] | str]:
    """Run deterministic generation for one checkpoint."""

    logger.info("Running inference for %s (%s)", name, model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    config.architectures = ["Gemma3ForCausalLM"]

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )

    generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    logger.info("[%s] Generated text: %s", name, generated_text)

    return {
        "text": generated_text,
        "tokens": output.sequences[0].tolist(),
        "logits_top1": [torch.max(logits).item() for logits in output.logits],
    }


def main() -> int:
    """Compare original and roundtrip checkpoints for exact generation parity."""

    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    res_orig = run_inference(
        args.original_path,
        "ORIGINAL",
        args.prompt,
        args.max_new_tokens,
        args.device,
        args.trust_remote_code,
    )
    res_round = run_inference(
        args.roundtrip_path,
        "ROUNDTRIP",
        args.prompt,
        args.max_new_tokens,
        args.device,
        args.trust_remote_code,
    )

    text_match = res_orig["text"] == res_round["text"]
    token_match = res_orig["tokens"] == res_round["tokens"]
    logit_diffs = [abs(a - b) for a, b in zip(res_orig["logits_top1"], res_round["logits_top1"])]
    max_diff = max(logit_diffs) if logit_diffs else math.nan

    logger.info("Final verification results")
    logger.info("Text Match: %s", text_match)
    logger.info("Token Match: %s", token_match)
    logger.info("Max Logit Diff: %s", max_diff)

    if text_match and token_match:
        logger.info("Roundtrip success: weights preserved exactly")
        return 0

    logger.error("Roundtrip failure: output mismatch detected")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
