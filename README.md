# GGUF to NVFP4: Qwen3.5-27B and Gemma 4 E4B from GGUF

A pipeline for converting GGUF models whose architecture `transformers` does not yet support loading from GGUF directly, into NVIDIA FP4 (NVFP4) quantized HuggingFace safetensors format ready for vLLM serving.

Currently supports:
- **Qwen3.5-27B** (hybrid Gated-DeltaNet, text + vision) -- the original motivation
- **Gemma 4 E4B** (standard GQA + PLE, text + vision + audio) -- see [Gemma 4 E4B Variant](#gemma-4-e4b-variant)

**[Chinese version / 中文版](README.cn.md)**

## Why This Pipeline

Qwen3.5 uses a hybrid Gated-DeltaNet architecture that mixes linear attention (Mamba-style) with full softmax attention in a 3:1 ratio. As of writing, `transformers` does not support loading Qwen3.5 from GGUF format directly. This pipeline manually extracts tensors from GGUF files, applies the necessary transformations, quantizes to NVFP4, and produces a model ready for vLLM serving.

Gemma 4 E4B has a similar problem: `transformers` supports `gemma2`/`gemma3` GGUF loading but not `gemma4`. Many finetunes (notably the HauhauCS uncensored series) are published only as GGUF with no bf16/fp16 safetensors source, so the same GGUF -> HF -> NVFP4 path is needed.

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

## Gemma 4 E4B Variant

A separate entry-point supports **Gemma 4 E4B** multimodal models (text + vision + audio). This targets e.g. HauhauCS's `Gemma-4-E4B-Uncensored-*-Aggressive` series and any other Gemma 4 E4B finetune published only as GGUF.

### Why a Separate Script

Gemma 4 has architectural differences from Qwen3.5 that require a different conversion path:

1. **No hybrid attention.** Standard GQA transformer -- no `(3,16)` value-head permutation, no `A_log` domain mismatch, no conv1d SSM. Half of Qwen3.5's pitfalls disappear.
2. **RMSNorm stores weights as-is.** Unlike Qwen3.5 / Gemma 2 / Gemma 3 where GGUF stores `1 + weight` because HF's RMSNorm forward does `(1 + w) * x`, Gemma 4's `Gemma4RMSNorm.forward` is `normed_output * self.weight.float()` -- **no `+ 1.0`**. Subtracting 1 here silently corrupts every layer norm.
3. **Per-Layer Embedding (PLE).** E4B has per-layer 256-dim embeddings and a global projection from hidden-size to `42 * 256`. New tensor types: `embed_tokens_per_layer`, `per_layer_model_projection`, `per_layer_input_gate`, `per_layer_projection`.
4. **Vision + audio towers use quant-wrapped linears.** HF wraps each vision/audio linear as `self_attn.q_proj.linear.weight` (nested) plus four FakeQuantize scalar bounds (`input_max`, `input_min`, `output_max`, `output_min`) with shape `()`. GGUF has only the bare weight with no scalar equivalent.
5. **Q8_K_P is "Permissive".** HauhauCS's `Q8_K_P` variant keeps attention Q/K/V as F16 and only quantizes MLP + output projections to Q8_0. Better source than pure-Q8 when full bf16 isn't published.
6. **Tied embeddings.** No `lm_head.weight` in the HF state dict; `lm_head` is tied to `embed_tokens` at runtime.

Three of Qwen3.5's five conversion pitfalls (RMSNorm `+1.0`, `A_log` domain, value-head `(3,16)->(16,3)`) do not apply. Two do (column-major shape reversal is handled automatically by `gguf.quants.dequantize`, and conv1d `unsqueeze` is not needed because Gemma 4 has no SSM). In practice Gemma 4 E4B is **simpler** than Qwen3.5 once you know the tensor map.

### Text Tensor Mapping

Per-layer tensors (`blk.<i>` -> `model.language_model.layers.<i>`):

| GGUF suffix | HF suffix |
|-------------|-----------|
| `attn_norm.weight` | `input_layernorm.weight` |
| `attn_q.weight` | `self_attn.q_proj.weight` |
| `attn_k.weight` | `self_attn.k_proj.weight` |
| `attn_v.weight` | `self_attn.v_proj.weight` |
| `attn_output.weight` | `self_attn.o_proj.weight` |
| `attn_q_norm.weight` | `self_attn.q_norm.weight` |
| `attn_k_norm.weight` | `self_attn.k_norm.weight` |
| `post_attention_norm.weight` | `post_attention_layernorm.weight` |
| `ffn_norm.weight` | `pre_feedforward_layernorm.weight` |
| `post_ffw_norm.weight` | `post_feedforward_layernorm.weight` |
| `post_norm.weight` | `post_per_layer_input_norm.weight` |
| `ffn_gate.weight` | `mlp.gate_proj.weight` |
| `ffn_up.weight` | `mlp.up_proj.weight` |
| `ffn_down.weight` | `mlp.down_proj.weight` |
| `inp_gate.weight` | `per_layer_input_gate.weight` |
| `proj.weight` | `per_layer_projection.weight` |
| `layer_output_scale.weight` | `layer_scalar` **(no `.weight` suffix)** |

Globals:

| GGUF | HF |
|------|----|
| `token_embd.weight` | `model.language_model.embed_tokens.weight` |
| `output_norm.weight` | `model.language_model.norm.weight` |
| `per_layer_model_proj.weight` | `model.language_model.per_layer_model_projection.weight` |
| `per_layer_proj_norm.weight` | `model.language_model.per_layer_projection_norm.weight` |
| `per_layer_token_embd.weight` | `model.language_model.embed_tokens_per_layer.weight` |
| `rope_freqs.weight` | *(skip -- runtime-computed by HF)* |

Total: 17 per-layer x 42 layers + 5 globals = **719 text tensors**.

### Vision + Audio: Copy from Reference

Rather than re-implementing the quant-wrapped linear layout and synthesizing FakeQuantize scalar bounds, the Gemma 4 pipeline copies vision, audio, `embed_vision`, and `embed_audio` tensors directly from a reference HF repo (default: `huihui-ai/Huihui-gemma-4-E4B-it-abliterated`). This works because Gemma 4 finetunes typically don't modify vision/audio towers -- they inherit unchanged from `google/gemma-4-e4b-it`. **Verify this is true for any new finetune before using the pipeline on it**, e.g. by dequantizing a few vision tensors from the finetune's GGUF mmproj and diffing against the reference.

### Pipeline

Two stages instead of three -- no stitch, because llmcompressor can quantize the full multimodal `Gemma4ForConditionalGeneration` in one oneshot call with a regex ignore list, and writes a complete checkpoint:

```
GGUF (text Q8_K_P) + Reference HF repo (config/tokenizer + vision/audio tensors)
  | step1_convert_gemma4_e4b.py
  v
HuggingFace Safetensors (bf16, full multimodal)
  | step2_quantize_gemma4_e4b.py   (NVFP4A16, weight-only)
  v
Final Model (NVFP4 text + BF16 vision/audio, ready for vLLM)
```

### Quick Start

```bash
# 1. Download the GGUF source (Q8_K_P is the highest precision HauhauCS publishes for Gemma 4 E4B)
pip install hf-transfer
HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
  HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive \
  Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf \
  --local-dir ./src

# 2. Convert GGUF to HF safetensors (reference repo auto-downloaded)
python scripts/step1_convert_gemma4_e4b.py \
  --gguf-text ./src/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf \
  --reference-repo huihui-ai/Huihui-gemma-4-E4B-it-abliterated \
  --output-dir ./gemma4-e4b-bf16-hf

# 3. NVFP4A16 quantization (weight-only, no calibration data)
python scripts/step2_quantize_gemma4_e4b.py \
  --model-dir ./gemma4-e4b-bf16-hf \
  --output-dir ./gemma4-e4b-nvfp4
```

### Why NVFP4A16 Instead of Full NVFP4 (w4a4)

E4B's architecture (per-layer embeddings, audio encoder, dynamic masking) breaks `fx.symbolic_trace`, which is a prerequisite for the sequential calibration pipeline needed by w4a4. NVFP4A16 quantizes weights from their own min/max statistics with no data flow, so no trace is needed at all. Quality impact is small because the 4-bit weight quantization dominates the final error floor.

### Why Q8_K_P Is an Acceptable Source

The `_P` in `Q8_K_P` stands for **Permissive**: attention Q/K/V projections stay as F16 (no quantization), only MLP and output projections are Q8_0. Since the final target is NVFP4 (4-bit), Q8's ~8-bit error floor is well below the NVFP4 noise floor and does not meaningfully affect quality.

### Known Limitation: No BF16/F16 Source

Many Gemma 4 E4B finetunes only publish GGUF at up to `Q8_K_P` -- no native bf16/fp16 safetensors. This pipeline's `Q8_K_P -> bf16` dequant is the fallback. If a bf16 source becomes available, skip step1 entirely and feed it directly to `step2_quantize_gemma4_e4b.py`.

### Dependencies

The Gemma 4 pipeline needs `transformers >= 5.5` (for `Gemma4ForConditionalGeneration`), `llmcompressor` main branch, and `gguf >= 0.18`. Tested on RTX 5090 with torch `2.11+cu130`.

## License

The code in this repository is MIT licensed. Model weights are subject to their original licenses.

## Acknowledgments

- [HauhauCS](https://huggingface.co/HauhauCS) for the uncensored Qwen3.5-27B and Gemma 4 E4B models
- [Kbenkhaled](https://huggingface.co/Kbenkhaled/Qwen3.5-27B-NVFP4) for the NVFP4 quantization recipe
- [huihui-ai](https://huggingface.co/huihui-ai) for the HF-format Gemma 4 E4B reference used by the Gemma 4 pipeline
- [Neural Magic / llm-compressor](https://github.com/neuralmagic/llm-compressor) for the quantization framework
- [vLLM](https://github.com/vllm-project/vllm) for serving infrastructure
