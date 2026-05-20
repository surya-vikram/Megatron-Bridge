# Gemma3 Conversion Workflow

This directory contains a reproducible Gemma-3 conversion workflow for:
- HuggingFace -> Megatron checkpoint import
- Megatron -> HuggingFace export
- Gemma-3 text-backbone roundtrip verification
- base-model vision tower / multimodal projector reattachment on export

## Files
- `setup_bridge_env.sh` - sync the Bridge Python environment
- `convert_gemma3.py` - import HuggingFace Gemma-3 into Megatron format
- `export_gemma3.py` - export Megatron Gemma-3 back to HuggingFace and merge base multimodal assets
- `verify_roundtrip.py` - compare original HF text generation vs roundtrip HF text generation
- `run_gemma3_roundtrip.sh` - one-command driver for setup, convert, export, and verify

## Why this exists
Gemma-3 requires Bridge-side fixes in `src/megatron/bridge/models/gemma/gemma3_bridge.py`.
This branch contains those fixes and these scripts package the working flow so conversion does not require manual debugging.

## Setup
```bash
cd /root/Megatron-Bridge
bash examples/conversion/gemma3/setup_bridge_env.sh
```

## End-to-end usage
```bash
cd /root/Megatron-Bridge
bash examples/conversion/gemma3/run_gemma3_roundtrip.sh \
  --hf-source-path /root/models/gemma-3-4b-pt-hf \
  --megatron-path /root/models/gemma-3-4b-pt-mcore \
  --raw-export-path /root/models/gemma-3-4b-pt-roundtrip-hf-raw \
  --hf-export-path /root/models/gemma-3-4b-pt-roundtrip-hf
```

## Direct commands
### Import HF -> Megatron
```bash
cd /root/Megatron-Bridge
source .venv/bin/activate
export PYTHONPATH=/root/Megatron-Bridge/src
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
python examples/conversion/gemma3/convert_gemma3.py \
  --hf-model-path /root/models/gemma-3-4b-pt-hf \
  --megatron-path /root/models/gemma-3-4b-pt-mcore
```

### Export Megatron -> HF
```bash
cd /root/Megatron-Bridge
source .venv/bin/activate
export PYTHONPATH=/root/Megatron-Bridge/src
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29502 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
python examples/conversion/gemma3/export_gemma3.py \
  --hf-source-path /root/models/gemma-3-4b-pt-hf \
  --megatron-path /root/models/gemma-3-4b-pt-mcore \
  --raw-export-path /root/models/gemma-3-4b-pt-roundtrip-hf-raw \
  --hf-export-path /root/models/gemma-3-4b-pt-roundtrip-hf
```

### Verify roundtrip
```bash
cd /root/Megatron-Bridge
source .venv/bin/activate
python examples/conversion/gemma3/verify_roundtrip.py \
  --original-path /root/models/gemma-3-4b-pt-hf \
  --roundtrip-path /root/models/gemma-3-4b-pt-roundtrip-hf
```

## Notes
- Export keeps the base-model multimodal files so the output checkpoint remains a complete Gemma-3 package.
- Verification intentionally forces `Gemma3ForCausalLM` loading to test the text backbone deterministically.
- These scripts assume single-process conversion with `WORLD_SIZE=1`.
