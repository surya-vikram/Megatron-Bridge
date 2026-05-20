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
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path

from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from megatron.bridge import AutoBridge


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Megatron to HF export."""

    parser = argparse.ArgumentParser(description="Export Megatron Gemma-3 checkpoint back to HuggingFace format")
    parser.add_argument("--hf-source-path", required=True, help="Original HuggingFace Gemma-3 checkpoint path")
    parser.add_argument("--megatron-path", required=True, help="Megatron checkpoint directory")
    parser.add_argument(
        "--raw-export-path", required=True, help="Temporary HF export directory for Bridge text weights"
    )
    parser.add_argument("--hf-export-path", required=True, help="Final merged HuggingFace export directory")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the HuggingFace config",
    )
    return parser.parse_args()


def build_shard_map(index_path: Path) -> dict[str, list[str]]:
    """Group HF parameter names by target safetensors shard."""

    weight_map = json.loads(index_path.read_text())["weight_map"]
    shard_map: dict[str, list[str]] = defaultdict(list)
    for key, shard_name in weight_map.items():
        shard_map[shard_name].append(key)
    return dict(shard_map)


def remap_text_key(key: str) -> str:
    """Map raw Bridge export keys into Gemma-3 text checkpoint keys."""

    if key.startswith("language_model.model."):
        return key
    if key.startswith("model.layers."):
        return key.replace("model.layers.", "language_model.model.layers.", 1)
    if key == "model.embed_tokens.weight":
        return "language_model.model.embed_tokens.weight"
    if key == "model.norm.weight":
        return "language_model.model.norm.weight"
    return key


def export_text_backbone(
    hf_source_path: Path, megatron_path: Path, raw_export_path: Path, trust_remote_code: bool
) -> None:
    """Export the Megatron text backbone into a temporary HF directory."""

    if raw_export_path.exists():
        shutil.rmtree(raw_export_path)

    config = AutoConfig.from_pretrained(str(hf_source_path), trust_remote_code=trust_remote_code)
    config.architectures = ["Gemma3ForCausalLM"]
    bridge = AutoBridge.from_hf_config(config)

    logger.info("Exporting Megatron checkpoint to temporary HF dir: %s", raw_export_path)
    bridge.export_ckpt(str(megatron_path), str(raw_export_path))


def merge_full_checkpoint(hf_source_path: Path, raw_export_path: Path, hf_export_path: Path) -> None:
    """Merge exported text weights back into the original multimodal HF checkpoint."""

    if hf_export_path.exists():
        shutil.rmtree(hf_export_path)
    hf_export_path.mkdir(parents=True, exist_ok=True)

    logger.info("Copying base HF checkpoint skeleton to %s", hf_export_path)
    for item in hf_source_path.iterdir():
        if item.name.startswith("model-") and item.suffix == ".safetensors":
            continue
        if item.name == "model.safetensors.index.json":
            continue
        destination = hf_export_path / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)

    base_index_path = hf_source_path / "model.safetensors.index.json"
    base_index = json.loads(base_index_path.read_text())

    remapped_text_tensors: dict[str, object] = {}
    for shard_path in sorted(raw_export_path.glob("model-*.safetensors")):
        for key, value in load_file(shard_path).items():
            remapped_text_tensors[remap_text_key(key)] = value

    if not remapped_text_tensors:
        raise RuntimeError(f"No text tensors were exported into {raw_export_path}")

    logger.info("Writing merged safetensor shards")
    for shard_name, shard_keys in build_shard_map(base_index_path).items():
        merged_shard = load_file(hf_source_path / shard_name)
        updated_tensors = {key: remapped_text_tensors.get(key, merged_shard[key]) for key in shard_keys}
        save_file(updated_tensors, str(hf_export_path / shard_name))

    base_index["metadata"] = {
        **base_index.get("metadata", {}),
        "source_base_checkpoint": str(hf_source_path),
        "source_text_export": str(raw_export_path),
        "merge_strategy": "base_multimodal_plus_roundtrip_text",
    }
    (hf_export_path / "model.safetensors.index.json").write_text(json.dumps(base_index, indent=2) + "\n")

    config_path = hf_export_path / "config.json"
    config = json.loads(config_path.read_text())
    config["torch_dtype"] = "bfloat16"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    """Export a Megatron Gemma-3 checkpoint and restore multimodal weights."""

    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    hf_source_path = Path(args.hf_source_path)
    megatron_path = Path(args.megatron_path)
    raw_export_path = Path(args.raw_export_path)
    hf_export_path = Path(args.hf_export_path)

    export_text_backbone(hf_source_path, megatron_path, raw_export_path, args.trust_remote_code)
    merge_full_checkpoint(hf_source_path, raw_export_path, hf_export_path)
    logger.info("Successfully exported full roundtrip checkpoint to %s", hf_export_path)


if __name__ == "__main__":
    main()
