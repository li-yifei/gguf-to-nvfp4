# GGUF to NVFP4: Quantizing Qwen3.5-27B from GGUF

A complete pipeline for converting Qwen3.5-27B GGUF models to NVIDIA FP4 (NVFP4) quantized format for efficient deployment with vLLM.

**[Chinese version / 中文版](README.cn.md)**

## Why This Pipeline

Qwen3.5 uses a hybrid Gated-DeltaNet architecture that mixes linear attention (Mamba-style) with full softmax attention in a 3:1 ratio. As of writing, `transformers` does not support loading Qwen3.5 from GGUF format directly. This pipeline manually extracts tensors from GGUF files, applies the necessary transformations, quantizes to NVFP4, and produces a model ready for vLLM serving.

### Pipeline Overview

```
GGUF (bf16 LLM + mmproj vision)
  | step1_convert.py
  v
HuggingFace Safetensors (bf16, sharded)
  | step2_quantize.py
  v
NVFP4 Quantized Model (text-only)
  | step3_stitch_vision.py
  v
Final Model (NVFP4 text + bf16 vision, ready for vLLM)
```

## GGUF Conversion Pitfalls

The Qwen3.5 GGUF format (via llama.cpp) has several non-obvious differences from the HuggingFace safetensors format. Getting any of these wrong produces a model that loads without errors but generates garbage.

### 1. RMSNorm Weight Offset (+1.0)

GGUF stores RMSNorm weights as `1 + learned_parameter`, while HuggingFace stores just the `learned_parameter`.

```python
# Affected: attn_norm, post_attention_norm, output_norm, attn_q_norm, attn_k_norm
# NOT affected: ssm_norm (GroupNorm, different normalization)
if is_rmsnorm_tensor(gguf_name):
    tensor = (tensor.float() - 1.0).to(torch.bfloat16)
```

Without this fix, every LayerNorm/RMSNorm output is shifted, causing gradual degradation -- the first few tokens may look correct but output rapidly becomes incoherent.

### 2. A_log Domain Mismatch

GGUF stores the SSM decay parameter as the materialized value `A = -exp(A_log)`, while HuggingFace expects `A_log` (log-space).

```python
# GGUF ssm_a -> HF A_log
if gguf_name.endswith(".ssm_a"):
    tensor = (-tensor.float()).log().to(torch.bfloat16)
```

### 3. Value Head Permutation (3,16) vs (16,3)

This is the most subtle bug. Qwen3.5's linear attention has 48 value heads organized as 16 KV-groups with 3 heads each. GGUF (llama.cpp) stores these in `(3 heads, 16 groups)` order, while HuggingFace expects `(16 groups, 3 heads)`.

**Affected tensors** (all linear_attn layers, 48 of 64 layers):

| Tensor | Shape | Fix |
|--------|-------|-----|
| `in_proj_a.weight` | [48, D] | Reshape (3,16,D) -> permute (1,0,2) |
| `in_proj_b.weight` | [48, D] | Same as above |
| `dt_bias` | [48] | Reshape (3,16) -> permute (1,0) |
| `A_log` | [48] | Same (after exp fix) |
| `in_proj_qkv.weight` | [10240, D] | V-section only [4096:] |
| `in_proj_z.weight` | [6144, D] | Full reshape (3,16,128,D) -> permute (1,0,2,3) |
| `out_proj.weight` | [D, 6144] | Column permutation |
| `conv1d.weight` | [10240, 1, K] | V-section only [4096:] |

The QKV split for `in_proj_qkv` is: Q=2048 (16 heads x 128 dim) + K=2048 + V=6144 (48 heads x 128 dim) = 10240 total. Only the V section needs permutation.

### 4. Column-Major Shape Reversal

GGUF stores tensor shapes in column-major (Fortran) order. PyTorch uses row-major (C) order.

```python
shape = list(reversed(tensor_info.shape))
```

### 5. Conv1d Weight Reshape

The conv1d weight in GGUF is stored as 2D `[channels, kernel_size]` but PyTorch expects 3D `[channels, 1, kernel_size]`.

```python
if "conv1d.weight" in hf_name and tensor.dim() == 2:
    tensor = tensor.unsqueeze(1).contiguous()
```

## Quick Start

### Prerequisites

```bash
pip install torch transformers>=5.0 safetensors gguf numpy huggingface-hub
pip install llmcompressor datasets  # for quantization step
```

### Step 1: Convert GGUF to HuggingFace Safetensors

```bash
python scripts/step1_convert.py \
    --gguf-llm /path/to/model.bf16.gguf \
    --gguf-vision /path/to/mmproj.gguf \
    --output-dir ./model-bf16-hf \
    --reference-repo huihui-ai/Huihui-Qwen3.5-27B-abliterated
```

The `--reference-repo` provides config.json and tokenizer files. Use a HuggingFace repo of the same model architecture.

### Step 2: NVFP4 Quantization

```bash
python scripts/step2_quantize.py \
    --model-dir ./model-bf16-hf \
    --output-dir ./model-nvfp4
```

This runs oneshot NVFP4 quantization using 512 calibration samples from `neuralmagic/calibration`. The following layers are excluded from quantization:
- `lm_head` -- output projection stays bf16
- `visual.*` -- vision encoder stays bf16
- `*.in_proj_a`, `*.in_proj_b` -- SSM gate parameters stay bf16

### Step 3: Stitch Vision Weights

```bash
python scripts/step3_stitch_vision.py \
    --bf16-dir ./model-bf16-hf \
    --nvfp4-dir ./model-nvfp4
```

This merges the original bf16 vision weights back into the quantized model, remaps weight names from `model.*` to `model.language_model.*`, updates the config for `Qwen3_5ForConditionalGeneration`, and re-shards the output.

## vLLM Deployment

### Config Requirements

The final model config must have:
- `model_type: "qwen3_5"` (top-level)
- `architectures: ["Qwen3_5ForConditionalGeneration"]`
- Nested `text_config` with the text model parameters
- `quantization_config` at the **top level** (not inside `text_config`)
- `dtype: "bfloat16"` at top level
- Weight names using `model.language_model.*` prefix
- Ignore list entries using `model.language_model.layers.*` format

### Docker Compose

```bash
cp deploy/.env.example deploy/.env
# Edit deploy/.env with your paths
docker compose -f deploy/docker-compose.yml up -d
```

See [deploy/](deploy/) for the full configuration.

### Quick Test

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-id",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Architecture Notes

### Qwen3.5 Hybrid Attention

Qwen3.5-27B has 64 layers with a 3:1 linear-to-full attention ratio:
- **Layers 0,1,2, 4,5,6, 8,9,10, ...** (48 layers): Gated DeltaNet linear attention
- **Layers 3, 7, 11, 15, ...** (16 layers): Full softmax attention

Linear attention layers use:
- `in_proj_qkv`: Fused Q/K/V projection [10240, 5120]
- `in_proj_z`: Gate projection [6144, 5120]
- `in_proj_a`, `in_proj_b`: SSM parameters [48, 5120]
- `out_proj`: Output projection [5120, 6144]
- `conv1d`: Causal convolution [10240, 1, 4]
- `A_log`, `dt_bias`: Recurrence parameters [48]
- `norm`: GroupNorm [48]

### Memory Budget (RTX 5090, 32GB VRAM)

| Component | Size |
|-----------|------|
| NVFP4 quantized weights | ~18 GB |
| Vision encoder (bf16) | ~0.9 GB |
| KV cache (fp8, 32K context) | ~8 GB |
| Overhead | ~3 GB |
| **Total** | **~30 GB** |

Use `gpu_memory_utilization=0.90` and `kv_cache_dtype=fp8` for comfortable operation.

## License

The code in this repository is MIT licensed. Model weights are subject to their original licenses.

## Acknowledgments

- [HauhauCS](https://huggingface.co/HauhauCS) for the uncensored Qwen3.5-27B model
- [Kbenkhaled](https://huggingface.co/Kbenkhaled/Qwen3.5-27B-NVFP4) for the NVFP4 quantization recipe
- [Neural Magic / llm-compressor](https://github.com/neuralmagic/llm-compressor) for the quantization framework
- [vLLM](https://github.com/vllm-project/vllm) for serving infrastructure
