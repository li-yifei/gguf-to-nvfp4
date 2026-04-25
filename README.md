# GGUF → NVFP4: Convert Qwen3.5, Qwen3.6 Dense/MoE & Gemma 4 E4B for vLLM

> A production pipeline that converts **GGUF** checkpoints of hybrid‑attention, dense, and MoE models into **NVIDIA NVFP4** quantized HuggingFace safetensors — ready to serve with **vLLM** on a single **RTX 5090**. Targets models `transformers` cannot load from GGUF directly (Qwen3.5, Qwen3.6 dense/MoE, Gemma 4).

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/pytorch-2.11%2Bcu130-red.svg">
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-compatible-brightgreen.svg">
  <img alt="NVFP4" src="https://img.shields.io/badge/quantization-NVFP4-76B900.svg">
  <img alt="GPU" src="https://img.shields.io/badge/GPU-RTX%205090-76B900.svg">
</p>

**Languages:** English · [中文](README.cn.md)

---

## TL;DR

`transformers` cannot load Qwen3.5 (hybrid Gated‑DeltaNet), Qwen3.6 dense/MoE (with MTP and vision variants), or Gemma 4 (GQA + PLE) directly from GGUF. This repo provides **model‑specific GGUF → HF → NVFP4** pipelines that extract tensors, fix the known conversion pitfalls, quantize to NVFP4 with `llm-compressor`, and emit a vLLM‑ready multimodal checkpoint.

## Supported Models

| Model | Architecture | Source format | Output (NVFP4) | Target GPU | Stages |
|-------|-------------|---------------|----------------|------------|--------|
| **Qwen3.5‑27B** | Hybrid Gated‑DeltaNet (3:1 linear ∶ full‑attn), text + vision | bf16 GGUF | ~18 GB + 0.9 GB vision | RTX 5090 (32 GB) | 3 (convert → quantize → stitch) |
| **Qwen3.6‑27B Dense** | Hybrid Gated‑DeltaNet (3:1), dense FFN, text + vision + MTP | Q8_K_P GGUF + f16 mmproj | ~17–18 GB | RTX 5090 (32 GB) | 4 (convert → stitch MTP → quantize → pack) |
| **Qwen3.6‑35B‑A3B MoE** | 256 experts (8 routed + 1 shared), hybrid DeltaNet, text + vision + MTP | Q8_K_P GGUF | ~21–22 GB | RTX 5090 (32 GB) | 2 (convert+MTP → quantize) |
| **Gemma 4 E4B** | Standard GQA + Per‑Layer Embedding, text + vision + audio | Q8_K_P GGUF | ~5–6 GB | RTX 5090 / smaller | 2 (convert → NVFP4A16) |

**Quick links:**
[Qwen3.5‑27B](#qwen35-27b--nvfp4) · [Qwen3.6‑27B Dense](#qwen36-27b-dense-hauhaucs--nvfp4) · [Qwen3.6‑35B‑A3B MoE](#qwen36-35b-a3b-moe--nvfp4) · [Gemma 4 E4B](#gemma-4-e4b--nvfp4) · [vLLM deployment](#vllm-deployment) · [GitHub topics](#repository-metadata-for-maintainers)

---

## Why This Pipeline

**Qwen3.5** uses a hybrid Gated‑DeltaNet architecture that interleaves Mamba‑style linear attention with full softmax attention at a 3:1 ratio. As of writing, `transformers` does not support loading Qwen3.5 from GGUF format directly. This repo manually extracts tensors from GGUF, applies the required transformations, quantizes to NVFP4, and produces a model ready for vLLM.

**Qwen3.6‑35B‑A3B** adds Mixture‑of‑Experts on top of the same hybrid attention and carries an additional MTP (Multi‑Token Prediction) head for speculative decoding. Many community finetunes (e.g. HauhauCS uncensored) publish only quantized GGUFs — no native bf16 safetensors.

**Qwen3.6‑27B Dense** keeps the Qwen3.5-style dense FFN and 64-layer hybrid attention, but uses Qwen3.6 tokenizer/config/vision/MTP metadata. The HauhauCS release is GGUF-only, so the reproducible path is GGUF dequantization plus official `Qwen/Qwen3.6-27B` reference tensors for MTP.

**Gemma 4 E4B** has the same GGUF‑loader gap: `transformers` supports `gemma2`/`gemma3` from GGUF but **not `gemma4`**. The HauhauCS uncensored series is published only as GGUF, so the same `GGUF → HF → NVFP4` path applies.

---

## Qwen3.5‑27B → NVFP4

### Pipeline Overview

```
GGUF (bf16 LLM + mmproj vision)
  │  step1_convert.py
  ▼
HuggingFace Safetensors (bf16, sharded)
  │  step2_quantize.py
  ▼
NVFP4 Quantized Model (text‑only)
  │  step3_stitch_vision.py
  ▼
Final Model (NVFP4 text + bf16 vision, vLLM‑ready)
```

### Qwen3.5 GGUF → HF Conversion Pitfalls (RMSNorm, A_log, Value‑Head Permutation)

The Qwen3.5 GGUF format (via `llama.cpp`) has several non‑obvious differences from HuggingFace safetensors. Getting any of these wrong produces a model that loads cleanly but **generates garbage**.

#### 1. RMSNorm Weight Offset (+1.0)

GGUF stores RMSNorm weights as `1 + learned_parameter`; HuggingFace stores just `learned_parameter`.

```python
# Affected: attn_norm, post_attention_norm, output_norm, attn_q_norm, attn_k_norm
# NOT affected: ssm_norm (GroupNorm, different normalization)
if is_rmsnorm_tensor(gguf_name):
    tensor = (tensor.float() - 1.0).to(torch.bfloat16)
```

Without this fix every LayerNorm/RMSNorm output is shifted, causing gradual degradation — the first few tokens may look correct but output rapidly becomes incoherent.

#### 2. A_log Domain Mismatch

GGUF stores the SSM decay parameter as the materialized value `A = -exp(A_log)`; HuggingFace expects `A_log` (log‑space).

```python
# GGUF ssm_a -> HF A_log
if gguf_name.endswith(".ssm_a"):
    tensor = (-tensor.float()).log().to(torch.bfloat16)
```

#### 3. Value‑Head Permutation (3,16) vs (16,3)

The most subtle bug. Qwen3.5's linear attention has 48 value heads organized as 16 KV‑groups with 3 heads each. GGUF (`llama.cpp`) stores these in `(3 heads, 16 groups)` order; HuggingFace expects `(16 groups, 3 heads)`.

**Affected tensors** (all linear_attn layers, 48 of 64 layers):

| Tensor | Shape | Fix |
|--------|-------|-----|
| `in_proj_a.weight` | [48, D] | Reshape (3,16,D) → permute (1,0,2) |
| `in_proj_b.weight` | [48, D] | Same as above |
| `dt_bias` | [48] | Reshape (3,16) → permute (1,0) |
| `A_log` | [48] | Same (after exp fix) |
| `in_proj_qkv.weight` | [10240, D] | V‑section only [4096:] |
| `in_proj_z.weight` | [6144, D] | Full reshape (3,16,128,D) → permute (1,0,2,3) |
| `out_proj.weight` | [D, 6144] | Column permutation |
| `conv1d.weight` | [10240, 1, K] | V‑section only [4096:] |

The QKV split for `in_proj_qkv` is: Q=2048 (16 heads × 128 dim) + K=2048 + V=6144 (48 heads × 128 dim) = 10240 total. Only the V section needs permutation.

#### 4. Column‑Major Shape Reversal

GGUF stores tensor shapes in column‑major (Fortran) order. PyTorch uses row‑major (C) order.

```python
shape = list(reversed(tensor_info.shape))
```

#### 5. Conv1d Weight Reshape

The conv1d weight in GGUF is stored as 2D `[channels, kernel_size]` but PyTorch expects 3D `[channels, 1, kernel_size]`.

```python
if "conv1d.weight" in hf_name and tensor.dim() == 2:
    tensor = tensor.unsqueeze(1).contiguous()
```

### Quick Start

**Prerequisites**

```bash
pip install torch transformers>=5.0 safetensors gguf numpy huggingface-hub
pip install llmcompressor datasets  # for quantization step
```

**Step 1 — GGUF → HF safetensors**

```bash
python scripts/step1_convert.py \
    --gguf-llm /path/to/model.bf16.gguf \
    --gguf-vision /path/to/mmproj.gguf \
    --output-dir ./model-bf16-hf \
    --reference-repo huihui-ai/Huihui-Qwen3.5-27B-abliterated
```

`--reference-repo` provides `config.json` and tokenizer files. Use any HuggingFace repo of the same model architecture.

**Step 2 — NVFP4 Quantization**

```bash
python scripts/step2_quantize.py \
    --model-dir ./model-bf16-hf \
    --output-dir ./model-nvfp4
```

Runs oneshot NVFP4 quantization with 512 calibration samples from `neuralmagic/calibration`. The following layers are excluded:

- `lm_head` — output projection stays bf16
- `visual.*` — vision encoder stays bf16
- `*.in_proj_a`, `*.in_proj_b` — SSM gate parameters stay bf16

**Step 3 — Stitch Vision Weights**

```bash
python scripts/step3_stitch_vision.py \
    --bf16-dir ./model-bf16-hf \
    --nvfp4-dir ./model-nvfp4
```

Merges the original bf16 vision weights back into the quantized model, remaps weight names (`model.*` → `model.language_model.*`), updates `config.json` for `Qwen3_5ForConditionalGeneration`, and re‑shards the output.

### Architecture Notes

**Qwen3.5 Hybrid Attention.** Qwen3.5‑27B has 64 layers with a 3:1 linear‑to‑full attention ratio:

- **Layers 0, 1, 2, 4, 5, 6, 8, 9, 10, …** (48 layers): Gated DeltaNet linear attention
- **Layers 3, 7, 11, 15, …** (16 layers): Full softmax attention

Linear‑attention layers use:

- `in_proj_qkv`: Fused Q/K/V projection `[10240, 5120]`
- `in_proj_z`: Gate projection `[6144, 5120]`
- `in_proj_a`, `in_proj_b`: SSM parameters `[48, 5120]`
- `out_proj`: Output projection `[5120, 6144]`
- `conv1d`: Causal convolution `[10240, 1, 4]`
- `A_log`, `dt_bias`: Recurrence parameters `[48]`
- `norm`: GroupNorm `[48]`

**Memory Budget (RTX 5090, 32 GB VRAM).**

| Component | Size |
|-----------|------|
| NVFP4 quantized weights | ~18 GB |
| Vision encoder (bf16) | ~0.9 GB |
| KV cache (fp8, 32K context) | ~8 GB |
| Overhead | ~3 GB |
| **Total** | **~30 GB** |

Use `gpu_memory_utilization=0.90` and `kv_cache_dtype=fp8` for comfortable operation.

---

## Qwen3.6‑27B Dense (HauhauCS) → NVFP4

This path reproduces the published model
[`lyf/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-NVFP4`](https://huggingface.co/lyf/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-NVFP4)
from the GGUF-only HauhauCS source
[`HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive`](https://huggingface.co/HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive).

### Dense 27B Differences from Qwen3.6 MoE

- Dense FFN: uses `Qwen3_5ForConditionalGeneration`, not `Qwen3_5MoeForConditionalGeneration`.
- 64 transformer layers, with full-attention layers at `3, 7, 11, ...`.
- Linear-attention value heads use the Qwen3.5-style `(3,16)` permutation and `V_SIZE=6144`.
- MTP tensors are not present in the HauhauCS GGUF and must be stitched from official `Qwen/Qwen3.6-27B`.
- Vision remains FP16/BF16; it is copied/packed as an extra shard and is not routed into NVFP4 kernels.

### Pipeline

```
Q8_K_P GGUF + f16 mmproj
  │  step1_convert_qwen36_dense.py
  ▼
HF safetensors (text + vision, BF16/FP16)
  │  stitch_qwen36_mtp.py
  ▼
HF safetensors (text + vision + MTP)
  │  step2_quantize_qwen36_dense.py
  ▼
llm-compressor NVFP4 output
  │  step3_pack_qwen36_dense.py
  ▼
Published full checkpoint (NVFP4 text + BF16 MTP/vision)
```

### Quick Start

```bash
# 1. Download HauhauCS GGUF source
hf download HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive \
  Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf \
  mmproj-Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-f16.gguf \
  --local-dir ./src/qwen36-27b-hauhau

# 2. Download official MTP shards/index from Qwen/Qwen3.6-27B
hf download Qwen/Qwen3.6-27B \
  model.safetensors.index.json \
  model-00013-of-00015.safetensors model-00015-of-00015.safetensors \
  --local-dir ./ref/qwen36-27b-official

# 3. Convert GGUF text + mmproj into HF safetensors
python scripts/step1_convert_qwen36_dense.py \
  --gguf-text ./src/qwen36-27b-hauhau/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf \
  --gguf-vision ./src/qwen36-27b-hauhau/mmproj-Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-f16.gguf \
  --output-dir ./work/qwen36-27b-hauhau-hf \
  --reference-repo Qwen/Qwen3.6-27B

# 4. Add MTP tensors from the official dense model
python scripts/stitch_qwen36_mtp.py \
  --source-dir ./ref/qwen36-27b-official \
  --target-dir ./work/qwen36-27b-hauhau-hf

# 5. Quantize text weights to NVFP4
python scripts/step2_quantize_qwen36_dense.py \
  --model-dir ./work/qwen36-27b-hauhau-hf \
  --output-dir ./work/qwen36-27b-hauhau-nvfp4 \
  --num-samples 256 \
  --dataset-mode ultrachat_nemotron

# 6. Repack visual + MTP tensors into one multimodal extra shard
python scripts/step3_pack_qwen36_dense.py \
  --source-dir ./work/qwen36-27b-hauhau-hf \
  --nvfp4-dir ./work/qwen36-27b-hauhau-nvfp4
```

### Runtime Notes

- On RTX 5090 with `vllm/vllm-openai:cu130-nightly`, `VLLM_NVFP4_GEMM_BACKEND=marlin` is validated and was about 34–36% faster than `flashinfer-cutlass` on the local 1024-token Responses benchmark.
- If Marlin fails with `size_n = 96 is not divisible by tile_n_size = 64`, check that visual tensors and non-64-aligned linear-attention projections were not quantized into NVFP4.
- For 128K agentic serving, use `--language-model-only`, `--kv-cache-dtype fp8`, `--max-num-seqs 1`, and a custom chat template if `/v1/responses` must default to non-thinking output.
- MTP startup works, but measured acceptance was workload-sensitive; do not assume MTP improves throughput without benchmarking your prompt mix.

### Runtime View Builder

The packed checkpoint can be materialized into profile-specific runtime views:

```bash
python scripts/build_qwen36_runtime_view.py \
  --source-dir ./work/qwen36-27b-hauhau-nvfp4 \
  --output-dir ./runtime/qwen36-27b-full \
  --profile full

python scripts/build_qwen36_runtime_view.py \
  --source-dir ./work/qwen36-27b-hauhau-nvfp4 \
  --output-dir ./runtime/qwen36-27b-text \
  --profile text
```

Use `profile=text` for maximum context space, `profile=full` for vision-capable runtime loading, and `profile=no-vision` / `profile=no-mtp` when you want one side of the extra shard only.

---

## Qwen3.6‑35B‑A3B MoE → NVFP4

A separate entry‑point supports **Qwen3.6‑35B‑A3B** (and community finetunes such as the HauhauCS uncensored series). This is a **Mixture‑of‑Experts** model with the same hybrid Gated‑DeltaNet attention as Qwen3.5, but fundamentally different in FFN structure — and it carries an MTP (Multi‑Token Prediction) head for speculative decoding.

### Architecture Differences from Qwen3.5

| | Qwen3.5‑27B | Qwen3.6‑35B‑A3B |
|---|---|---|
| Type | Dense | MoE (256 experts, 8 routed + 1 shared) |
| Layers | 64 | 40 |
| Hidden size | 5120 | 2048 |
| Head dim (full attn) | 128 | 256 |
| V heads (linear attn) | 48 = 3×16 | 32 = 2×16 |
| Full‑attn Q dim | 4096 | 8192 (includes output gate) |
| QKV split (linear) | Q:2048 + K:2048 + V:6144 = 10240 | Q:2048 + K:2048 + V:4096 = 8192 |
| MTP head | — | ✅ (1 layer, 19 tensors, speculative decoding) |
| HF architecture | `Qwen3_5ForConditionalGeneration` | `Qwen3_5MoeForConditionalGeneration` |

### MoE‑Specific Conversion

GGUF stores MoE expert weights as packed 3D tensors. HF expects a fused `gate_up_proj`:

```python
# GGUF: ffn_gate_exps [256, 512, 2048] + ffn_up_exps [256, 512, 2048]
# HF:   experts.gate_up_proj [256, 1024, 2048]   (no .weight suffix!)
fused = torch.cat([gate_exps, up_exps], dim=1)
```

Other MoE tensors:

- `ffn_down_exps` → `experts.down_proj` (no `.weight` suffix)
- `ffn_gate_inp` → `mlp.gate.weight` (router)
- `ffn_gate_shexp` / `ffn_up_shexp` / `ffn_down_shexp` → `shared_expert.{gate,up,down}_proj.weight`
- `ffn_gate_inp_shexp` → `shared_expert_gate.weight` (**needs `unsqueeze(0)`**: GGUF `[2048]` → HF `[1, 2048]`)

### Additional Conversion Pitfalls (Qwen3.6)

Beyond the five Qwen3.5 pitfalls (which all still apply), Qwen3.6 adds:

6. **V‑head permutation is (2,16) not (3,16).** 32 V‑heads = 16 KV‑groups × 2 heads. Same reshape/permute logic, different constants.
7. **Patch embed is 5D.** Vision encoder uses temporal 3D conv. GGUF splits into `v.patch_embd.weight` + `v.patch_embd.weight.1` (two 4D tensors), which must be **stacked** into one 5D tensor `[C, 3, 2, H, W]`.
8. **MTP not in GGUF.** The Multi‑Token Prediction head (19 tensors) must be copied from the base HF model (`Qwen/Qwen3.6-35B-A3B`). Only 2 safetensor shards are needed (`model-00025`, `model-00026`), so selective download is recommended over pulling the full 67 GB reference.
9. **Q8_K_P source.** Unlike Qwen3.5 which ships a bf16 GGUF, HauhauCS publishes only quantized GGUFs. Use `gguf.quants.dequantize()` which handles Q8_K → F32 and auto‑reverses shapes.

### Pipeline

Two stages — no stitch step needed, because the quantization ignore list keeps vision in bf16 and `Qwen3_5MoeForConditionalGeneration` loads the full multimodal model in one shot:

```
Q8_K_P GGUF + mmproj GGUF + Reference HF (config/tokenizer + MTP shards)
  │  step1_convert_qwen36_moe.py        (→ 1045 tensors across 22 shards, ~67 GB)
  ▼                                      (733 text + 333 vision + 19 MTP)
HuggingFace Safetensors (bf16, text + vision + MTP)
  │  step2_quantize_qwen36_moe.py           (conservative: linear_attn + MTP bf16)
  │  step2b_quantize_qwen36_aggressive.py   (aggressive: everything NVFP4)
  ▼
Final NVFP4 Model (~21–22 GB, vLLM‑ready)
```

### Two Quantization Profiles

**Conservative** (`step2_quantize_qwen36_moe.py`) — AEON‑7 / RedHatAI approach. Keeps `linear_attn` (DeltaNet) and MTP in bf16 because `linear_attn` is precision‑sensitive and MTP quality directly affects speculative‑decoding acceptance rates.

```python
ignore=["lm_head", "re:.*visual.*", "re:.*mlp.gate$",
        "re:.*mlp.shared_expert_gate$", "re:.*linear_attn.*", "re:^mtp.*"]
```

**Aggressive** (`step2b_quantize_qwen36_aggressive.py`) — sakamakismile approach. Quantizes everything except `lm_head`, vision, and gates. Smaller footprint buys longer context.

```python
ignore=["lm_head", "re:.*visual.*", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"]
```

| Profile | Size | RTX 5090 text‑only ctx | With vision |
|---------|------|------------------------|-------------|
| Conservative | ~22 GB | ~131K | ~4K |
| Aggressive | ~21 GB | ~131K+ | ~65K |

### Quick Start

```bash
# 1. Download GGUF source
hf download HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive \
  --include "*Q8_K_P*" "*mmproj*" --local-dir ./src

# 2. Download reference config/tokenizer + the 2 MTP shards only
hf download Qwen/Qwen3.6-35B-A3B \
  config.json tokenizer.json tokenizer_config.json chat_template.jinja \
  merges.txt vocab.json generation_config.json preprocessor_config.json \
  video_preprocessor_config.json model.safetensors.index.json \
  model-00025-of-00026.safetensors model-00026-of-00026.safetensors \
  --local-dir ./ref

# 3. Convert GGUF to HF safetensors (text + vision + MTP injection)
python scripts/step1_convert_qwen36_moe.py \
  --gguf-llm ./src/*Q8_K_P*.gguf \
  --gguf-vision ./src/*mmproj*.gguf \
  --output-dir ./qwen36-bf16-hf \
  --reference-repo Qwen/Qwen3.6-35B-A3B

# 4a. Conservative NVFP4 (linear_attn + MTP stay bf16)
python scripts/step2_quantize_qwen36_moe.py \
  --model-dir ./qwen36-bf16-hf \
  --output-dir ./qwen36-nvfp4-conservative

# 4b. OR Aggressive NVFP4 (everything quantized, best for long context)
python scripts/step2b_quantize_qwen36_aggressive.py \
  --model-dir ./qwen36-bf16-hf \
  --output-dir ./qwen36-nvfp4-aggressive
```

### 60 GB RAM Workaround (MoE save‑path patches)

The 67 GB bf16 model exceeds typical 64 GB system RAM. The step2 scripts use `device_map="auto"` with disk offloading, which requires two patches around a `transformers` / `llmcompressor` save bug triggered by MoE + disk offload:

1. **`transformers/integrations/accelerate.py`** (`load_offloaded_parameter`): Wrap `model.get_submodule()` in `try / except AttributeError: continue` to skip non‑matching paths.
2. **`llmcompressor/.../compressed_tensors_utils.py`** (`save_pretrained_wrapper`): Comment out `to_accelerate(model)` and `from_accelerate(model)` to prevent tensor‑name‑prefix triplication.
3. **Post‑save key rename**: The saved safetensors will have a triple `model.language_model.language_model.language_model.` prefix. Fix with:
   ```python
   new_key = key.replace(
       'model.language_model.language_model.language_model.',
       'model.language_model.'
   )
   ```

These patches are not needed on 128 GB+ systems (load without `device_map`).

### vLLM Deployment (Qwen3.6)

```bash
# Text‑only, 100K+ context on RTX 5090
docker run --gpus all -v ./qwen36-nvfp4:/model vllm/vllm-openai:nightly \
  --model /model \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 100000 \
  --max-num-seqs 1 \
  --reasoning-parser qwen3 \
  --language-model-only

# With MTP speculative decoding (when supported)
# --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

AEON‑7 recommends setting `VLLM_TEST_FORCE_FP8_MARLIN=1` if CUTLASS NVFP4 is broken on your SM121 GPU.

---

## Gemma 4 E4B → NVFP4

A separate entry‑point supports **Gemma 4 E4B** multimodal models (**text + vision + audio**). Targets include HauhauCS's `Gemma-4-E4B-Uncensored-*-Aggressive` series and any other Gemma 4 E4B finetune published only as GGUF.

### Why a Separate Script

Gemma 4 has architectural differences from Qwen3.5 that require a different conversion path:

1. **No hybrid attention.** Standard GQA transformer — no `(3,16)` value‑head permutation, no `A_log` domain mismatch, no conv1d SSM. Half of Qwen3.5's pitfalls disappear.
2. **RMSNorm stores weights as‑is.** Unlike Qwen3.5 / Gemma 2 / Gemma 3 — where GGUF stores `1 + weight` because HF's RMSNorm forward is `(1 + w) * x` — Gemma 4's `Gemma4RMSNorm.forward` is `normed_output * self.weight.float()`: **no `+ 1.0`**. Subtracting 1 here silently corrupts every layer norm.
3. **Per‑Layer Embedding (PLE).** E4B has per‑layer 256‑dim embeddings and a global projection from hidden‑size to `42 * 256`. New tensor types: `embed_tokens_per_layer`, `per_layer_model_projection`, `per_layer_input_gate`, `per_layer_projection`.
4. **Vision + audio towers use quant‑wrapped linears.** HF wraps each vision/audio linear as `self_attn.q_proj.linear.weight` (nested) plus four FakeQuantize scalar bounds (`input_max`, `input_min`, `output_max`, `output_min`) with shape `()`. GGUF has only the bare weight with no scalar equivalent.
5. **Q8_K_P is "Permissive".** HauhauCS's `Q8_K_P` variant keeps attention Q/K/V as F16 and only quantizes MLP + output projections to Q8_0. A better source than pure‑Q8 when full bf16 isn't published.
6. **Tied embeddings.** No `lm_head.weight` in the HF state dict; `lm_head` is tied to `embed_tokens` at runtime.

Three of Qwen3.5's five conversion pitfalls (RMSNorm `+1.0`, `A_log` domain, value‑head `(3,16) → (16,3)`) do not apply. Two do (column‑major shape reversal is handled automatically by `gguf.quants.dequantize`, and conv1d `unsqueeze` is not needed because Gemma 4 has no SSM). In practice **Gemma 4 E4B is simpler than Qwen3.5** once you know the tensor map.

### Text Tensor Mapping

Per‑layer tensors (`blk.<i>` → `model.language_model.layers.<i>`):

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
| `rope_freqs.weight` | *(skip — runtime‑computed by HF)* |

Total: 17 per‑layer × 42 layers + 5 globals = **719 text tensors**.

### Vision + Audio: Copy from Reference

Rather than re‑implementing the quant‑wrapped linear layout and synthesizing FakeQuantize scalar bounds, the Gemma 4 pipeline copies vision, audio, `embed_vision`, and `embed_audio` tensors directly from a reference HF repo (default: `huihui-ai/Huihui-gemma-4-E4B-it-abliterated`). This works because Gemma 4 finetunes typically do not modify vision/audio towers — they inherit unchanged from `google/gemma-4-e4b-it`.

> **Verify this assumption for any new finetune** before using the pipeline on it — e.g. dequantize a few vision tensors from the finetune's GGUF `mmproj` and diff against the reference.

### Pipeline

Two stages instead of three — no stitch, because `llmcompressor` can quantize the full multimodal `Gemma4ForConditionalGeneration` in one `oneshot` call with a regex ignore list, and writes a complete checkpoint:

```
GGUF (text Q8_K_P) + Reference HF repo (config/tokenizer + vision/audio tensors)
  │  step1_convert_gemma4_e4b.py
  ▼
HuggingFace Safetensors (bf16, full multimodal)
  │  step2_quantize_gemma4_e4b.py   (NVFP4A16, weight‑only)
  ▼
Final Model (NVFP4 text + BF16 vision/audio, vLLM‑ready)
```

### Quick Start

```bash
# 1. Download the GGUF source (Q8_K_P is the highest precision HauhauCS publishes for Gemma 4 E4B)
pip install hf-transfer
HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
  HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive \
  Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf \
  --local-dir ./src

# 2. Convert GGUF to HF safetensors (reference repo auto‑downloaded)
python scripts/step1_convert_gemma4_e4b.py \
  --gguf-text ./src/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf \
  --reference-repo huihui-ai/Huihui-gemma-4-E4B-it-abliterated \
  --output-dir ./gemma4-e4b-bf16-hf

# 3. NVFP4A16 quantization (weight‑only, no calibration data)
python scripts/step2_quantize_gemma4_e4b.py \
  --model-dir ./gemma4-e4b-bf16-hf \
  --output-dir ./gemma4-e4b-nvfp4
```

### Why NVFP4A16 Instead of Full NVFP4 (w4a4)

E4B's architecture (per‑layer embeddings, audio encoder, dynamic masking) breaks `fx.symbolic_trace`, which is a prerequisite for the sequential calibration pipeline needed by w4a4. NVFP4A16 quantizes weights from their own min/max statistics with no data flow, so no trace is needed at all. Quality impact is small because the 4‑bit weight quantization dominates the final error floor.

### Why Q8_K_P Is an Acceptable Source

The `_P` in `Q8_K_P` stands for **Permissive**: attention Q/K/V projections stay as F16 (no quantization), only MLP and output projections are Q8_0. Since the final target is NVFP4 (4‑bit), Q8's ~8‑bit error floor is well below the NVFP4 noise floor and does not meaningfully affect quality.

### Known Limitation: No BF16/F16 Source

Many Gemma 4 E4B finetunes only publish GGUF at up to `Q8_K_P` — no native bf16/fp16 safetensors. This pipeline's `Q8_K_P → bf16` dequant is the fallback. If a bf16 source becomes available, skip step1 entirely and feed it directly to `step2_quantize_gemma4_e4b.py`.

### Dependencies

The Gemma 4 pipeline needs `transformers >= 5.5` (for `Gemma4ForConditionalGeneration`), `llmcompressor` main branch, and `gguf >= 0.18`. Tested on RTX 5090 with `torch 2.11+cu130`.

---

## vLLM Deployment

### Config Requirements (Qwen3.5)

The final model config must have:

- `model_type: "qwen3_5"` (top‑level)
- `architectures: ["Qwen3_5ForConditionalGeneration"]`
- Nested `text_config` with the text model parameters
- `quantization_config` at the **top level** (not inside `text_config`)
- `dtype: "bfloat16"` at top level
- Weight names using `model.language_model.*` prefix
- Ignore‑list entries using `model.language_model.layers.*` format

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

---

## Repository Metadata (for Maintainers)

Suggested GitHub **About / Description**:

> GGUF → NVFP4 conversion pipeline for Qwen3.5, Qwen3.6‑A3B MoE, and Gemma 4 E4B — produces vLLM‑ready multimodal checkpoints on a single RTX 5090.

Suggested **Topics** (copy‑paste into repo settings):

```
nvfp4  gguf  quantization  vllm  llm-compressor  qwen  qwen3  qwen3-moe
gemma  gemma-4  mixture-of-experts  moe  gated-deltanet  hybrid-attention
multimodal  huggingface  safetensors  rtx-5090  nvidia-fp4  speculative-decoding
```

---

## License

The code in this repository is MIT licensed. Model weights are subject to their original licenses.

## Acknowledgments

- [HauhauCS](https://huggingface.co/HauhauCS) — uncensored Qwen3.5‑27B, Qwen3.6‑35B‑A3B, and Gemma 4 E4B GGUF models
- [sakamakismile](https://huggingface.co/sakamakismile) — Qwen3.6 NVFP4 aggressive quantization recipe reference
- [AEON‑7](https://huggingface.co/AEON-7) — Qwen3.6 NVFP4 conservative quantization insights
- [Kbenkhaled](https://huggingface.co/Kbenkhaled/Qwen3.5-27B-NVFP4) — original NVFP4 quantization recipe
- [huihui‑ai](https://huggingface.co/huihui-ai) — HF‑format Gemma 4 E4B reference used by the Gemma 4 pipeline
- [Neural Magic / llm‑compressor](https://github.com/neuralmagic/llm-compressor) — quantization framework
- [vLLM](https://github.com/vllm-project/vllm) — serving infrastructure

<!--
SEO keywords (for search engines):
Qwen3.5 NVFP4, Qwen3.5-27B NVFP4, Qwen3.6 NVFP4, Qwen3.6-35B-A3B NVFP4,
Qwen3 MoE NVFP4, Gemma 4 E4B NVFP4, GGUF to NVFP4, GGUF to safetensors,
GGUF to HuggingFace, NVFP4 quantization pipeline, Gated DeltaNet conversion,
hybrid attention quantization, vLLM FP4, vLLM NVFP4, RTX 5090 LLM,
llm-compressor NVFP4, Mixture of Experts quantization, MTP speculative decoding,
Per-Layer Embedding Gemma 4, HauhauCS uncensored quantization.
-->
