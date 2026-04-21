# GGUF → NVFP4：Qwen3.5 / Qwen3.6‑MoE (A3B) / Gemma 4 E4B 量化部署（vLLM 可用）

> 一套生产可用的 **GGUF → NVIDIA NVFP4** 量化转换管线，把 `transformers` 无法从 GGUF 直接加载的混合注意力 / MoE 模型（Qwen3.5、Qwen3.6‑MoE、Gemma 4）转成 HuggingFace safetensors，单卡 **RTX 5090** 即可用 **vLLM** 部署。

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/pytorch-2.11%2Bcu130-red.svg">
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-compatible-brightgreen.svg">
  <img alt="NVFP4" src="https://img.shields.io/badge/quantization-NVFP4-76B900.svg">
  <img alt="GPU" src="https://img.shields.io/badge/GPU-RTX%205090-76B900.svg">
</p>

**语言：** 中文 · [English](README.md)

---

## TL;DR

`transformers` 目前**不支持**直接从 GGUF 加载 Qwen3.5（混合 Gated‑DeltaNet）、Qwen3.6‑MoE（A3B，带 MTP 头）和 Gemma 4（GQA + PLE）。本仓库提供**按模型定制**的 GGUF → HF → NVFP4 管线：从 GGUF 提取张量、修复已知转换坑、用 `llm-compressor` 量化到 NVFP4，产出可直接喂给 vLLM 的多模态 checkpoint。

## 支持的模型

| 模型 | 架构 | 源格式 | NVFP4 产出 | 目标 GPU | 阶段数 |
|------|------|--------|-----------|---------|--------|
| **Qwen3.5‑27B** | 混合 Gated‑DeltaNet（线性∶全注意力 = 3∶1），文本 + 视觉 | bf16 GGUF | ~18 GB + 0.9 GB 视觉 | RTX 5090 (32GB) | 3（convert → quantize → stitch） |
| **Qwen3.6‑35B‑A3B MoE** | 256 专家（8 路由 + 1 共享），混合 DeltaNet，文本 + 视觉 + MTP | Q8_K_P GGUF | ~21–22 GB | RTX 5090 (32GB) | 2（convert+MTP → quantize） |
| **Gemma 4 E4B** | 标准 GQA + 每层 embedding (PLE)，文本 + 视觉 + 音频 | Q8_K_P GGUF | ~5–6 GB | RTX 5090 / 更低 | 2（convert → NVFP4A16） |

**快速跳转：**
[Qwen3.5‑27B](#qwen35-27b--nvfp4) · [Qwen3.6‑35B‑A3B MoE](#qwen36-35b-a3b-moe--nvfp4) · [Gemma 4 E4B](#gemma-4-e4b--nvfp4) · [vLLM 部署](#vllm-部署) · [GitHub topics](#仓库元信息供维护者使用)

---

## 为什么需要这套流程

**Qwen3.5** 采用混合 Gated‑DeltaNet 架构，以 3∶1 的比例交替使用 Mamba 风格线性注意力和全 softmax 注意力。目前 `transformers` 不支持直接从 GGUF 加载 Qwen3.5。本仓库从 GGUF 手动提取张量，完成必要的格式转换，量化为 NVFP4，最终产出 vLLM 可部署模型。

**Qwen3.6‑35B‑A3B** 在同样的混合注意力之上引入 Mixture‑of‑Experts，并多了一颗 MTP（Multi‑Token Prediction，多 token 预测）头用于投机解码。社区微调（如 HauhauCS uncensored）普遍只发布量化 GGUF，没有 bf16 原生 safetensors。

**Gemma 4 E4B** 有同样的加载缺口：`transformers` 支持 `gemma2` / `gemma3` 从 GGUF 加载，但**不支持** `gemma4`。HauhauCS 的 uncensored 系列只发布 GGUF，所以同样需要 `GGUF → HF → NVFP4` 路径。

---

## Qwen3.5‑27B → NVFP4

### 流程概览

```
GGUF (bf16 LLM + mmproj 视觉编码器)
  │  step1_convert.py
  ▼
HuggingFace Safetensors (bf16, 分片)
  │  step2_quantize.py
  ▼
NVFP4 量化模型 (仅文本)
  │  step3_stitch_vision.py
  ▼
最终模型 (NVFP4 文本 + bf16 视觉，vLLM 可用)
```

### Qwen3.5 GGUF → HF 转换坑（RMSNorm、A_log、Value Head 重排）

Qwen3.5 的 GGUF 格式（来自 `llama.cpp`）和 HuggingFace safetensors 格式之间有几个不明显的差异。搞错任何一个，模型都能正常加载，但**输出全是乱码**。

#### 1. RMSNorm 权重偏移（+1.0）

GGUF 存储 RMSNorm 权重为 `1 + 学习参数`，而 HuggingFace 只存 `学习参数` 本身。

```python
# 受影响：attn_norm, post_attention_norm, output_norm, attn_q_norm, attn_k_norm
# 不受影响：ssm_norm（GroupNorm，不同的归一化方式）
if is_rmsnorm_tensor(gguf_name):
    tensor = (tensor.float() - 1.0).to(torch.bfloat16)
```

不修复的话，每层 LayerNorm / RMSNorm 的输出都会有偏移，前几个 token 可能看起来正常，输出会迅速退化为乱码。

#### 2. A_log 域不匹配

GGUF 存储 SSM 衰减参数为实际值 `A = -exp(A_log)`，而 HuggingFace 期望的是对数空间的 `A_log`。

```python
# GGUF ssm_a -> HF A_log
if gguf_name.endswith(".ssm_a"):
    tensor = (-tensor.float()).log().to(torch.bfloat16)
```

#### 3. Value Head 排列顺序 (3,16) vs (16,3)

最隐蔽的一个 bug。Qwen3.5 的线性注意力有 48 个 value head，组织为 16 个 KV 组、每组 3 个 head。GGUF（`llama.cpp`）按 `(3 个 head, 16 个组)` 顺序存储，而 HuggingFace 期望 `(16 个组, 3 个 head)` 顺序。

**受影响的张量**（所有 linear_attn 层，64 层中的 48 层）：

| 张量 | 形状 | 修复方式 |
|------|------|----------|
| `in_proj_a.weight` | [48, D] | reshape(3,16,D) → permute(1,0,2) |
| `in_proj_b.weight` | [48, D] | 同上 |
| `dt_bias` | [48] | reshape(3,16) → permute(1,0) |
| `A_log` | [48] | 同上（在 exp 修复之后） |
| `in_proj_qkv.weight` | [10240, D] | 仅 V 部分 [4096:] |
| `in_proj_z.weight` | [6144, D] | 整体 reshape(3,16,128,D) → permute(1,0,2,3) |
| `out_proj.weight` | [D, 6144] | 列排列 |
| `conv1d.weight` | [10240, 1, K] | 仅 V 部分 [4096:] |

`in_proj_qkv` 的 QKV 分割：Q=2048 (16 头 × 128 维) + K=2048 + V=6144 (48 头 × 128 维) = 共 10240。只有 V 部分需要重排。

#### 4. 列优先形状反转

GGUF 按列优先（Fortran）顺序存储张量形状，PyTorch 使用行优先（C）顺序。

```python
shape = list(reversed(tensor_info.shape))
```

#### 5. Conv1d 权重维度调整

GGUF 中的 conv1d 权重是 2D 的 `[channels, kernel_size]`，PyTorch 期望 3D 的 `[channels, 1, kernel_size]`。

```python
if "conv1d.weight" in hf_name and tensor.dim() == 2:
    tensor = tensor.unsqueeze(1).contiguous()
```

### 快速上手

**前置依赖**

```bash
pip install torch transformers>=5.0 safetensors gguf numpy huggingface-hub
pip install llmcompressor datasets  # 量化步骤需要
```

**第一步 — GGUF 转 HuggingFace Safetensors**

```bash
python scripts/step1_convert.py \
    --gguf-llm /path/to/model.bf16.gguf \
    --gguf-vision /path/to/mmproj.gguf \
    --output-dir ./model-bf16-hf \
    --reference-repo huihui-ai/Huihui-Qwen3.5-27B-abliterated
```

`--reference-repo` 提供 `config.json` 和 tokenizer 文件，使用同架构的任意 HuggingFace 仓库即可。

**第二步 — NVFP4 量化**

```bash
python scripts/step2_quantize.py \
    --model-dir ./model-bf16-hf \
    --output-dir ./model-nvfp4
```

使用 `neuralmagic/calibration` 数据集的 512 个样本进行 oneshot NVFP4 量化。以下层不参与量化：

- `lm_head` — 输出投影保持 bf16
- `visual.*` — 视觉编码器保持 bf16
- `*.in_proj_a`, `*.in_proj_b` — SSM 门控参数保持 bf16

**第三步 — 缝合视觉权重**

```bash
python scripts/step3_stitch_vision.py \
    --bf16-dir ./model-bf16-hf \
    --nvfp4-dir ./model-nvfp4
```

将原始 bf16 视觉权重合并回量化模型，重新映射权重名（`model.*` → `model.language_model.*`），更新 config 为 `Qwen3_5ForConditionalGeneration`，并重新分片输出。

### 架构说明

**Qwen3.5 混合注意力。** Qwen3.5‑27B 共 64 层，线性注意力 ∶ 全注意力 = 3∶1：

- **第 0, 1, 2, 4, 5, 6, 8, 9, 10, … 层**（48 层）：Gated DeltaNet 线性注意力
- **第 3, 7, 11, 15, … 层**（16 层）：全 softmax 注意力

线性注意力层的结构：

- `in_proj_qkv`：Q/K/V 融合投影 `[10240, 5120]`
- `in_proj_z`：门控投影 `[6144, 5120]`
- `in_proj_a`, `in_proj_b`：SSM 参数 `[48, 5120]`
- `out_proj`：输出投影 `[5120, 6144]`
- `conv1d`：因果卷积 `[10240, 1, 4]`
- `A_log`, `dt_bias`：循环参数 `[48]`
- `norm`：GroupNorm `[48]`

**显存预算（RTX 5090, 32 GB 显存）**

| 组件 | 大小 |
|------|------|
| NVFP4 量化权重 | ~18 GB |
| 视觉编码器 (bf16) | ~0.9 GB |
| KV 缓存 (fp8, 32K 上下文) | ~8 GB |
| 其他开销 | ~3 GB |
| **总计** | **~30 GB** |

建议使用 `gpu_memory_utilization=0.90` 配合 `kv_cache_dtype=fp8`。

---

## Qwen3.6‑35B‑A3B MoE → NVFP4

本仓库另一个入口支持 **Qwen3.6‑35B‑A3B**（以及 HauhauCS uncensored 等社区微调）。这是**混合专家（MoE）** 模型，使用和 Qwen3.5 相同的混合 Gated‑DeltaNet 注意力，但 FFN 结构完全不同 —— 并且多了一颗 MTP（Multi‑Token Prediction，多 token 预测）头用于投机解码。

### 与 Qwen3.5 的架构差异

| | Qwen3.5‑27B | Qwen3.6‑35B‑A3B |
|---|---|---|
| 类型 | 稠密 | MoE（256 专家，8 路由 + 1 共享） |
| 层数 | 64 | 40 |
| Hidden size | 5120 | 2048 |
| Head dim（全注意力） | 128 | 256 |
| V heads（线性注意力） | 48 = 3×16 | 32 = 2×16 |
| 全注意力 Q 维度 | 4096 | 8192（含输出门） |
| QKV split（线性注意力） | Q:2048 + K:2048 + V:6144 = 10240 | Q:2048 + K:2048 + V:4096 = 8192 |
| MTP 头 | — | ✅（1 层，19 张量，用于投机解码） |
| HF 架构 | `Qwen3_5ForConditionalGeneration` | `Qwen3_5MoeForConditionalGeneration` |

### MoE 专属转换

GGUF 把 MoE 专家权重存为打包的 3D 张量，HF 期望的是融合后的 `gate_up_proj`：

```python
# GGUF: ffn_gate_exps [256, 512, 2048] + ffn_up_exps [256, 512, 2048]
# HF:   experts.gate_up_proj [256, 1024, 2048]   （注意：没有 .weight 后缀！）
fused = torch.cat([gate_exps, up_exps], dim=1)
```

其他 MoE 张量：

- `ffn_down_exps` → `experts.down_proj`（无 `.weight` 后缀）
- `ffn_gate_inp` → `mlp.gate.weight`（路由器）
- `ffn_gate_shexp` / `ffn_up_shexp` / `ffn_down_shexp` → `shared_expert.{gate,up,down}_proj.weight`
- `ffn_gate_inp_shexp` → `shared_expert_gate.weight`（**需要 `unsqueeze(0)`**：GGUF `[2048]` → HF `[1, 2048]`）

### Qwen3.6 额外的转换坑

Qwen3.5 的 5 个坑在 Qwen3.6 上**全部仍然存在**，此外还多出以下几个：

6. **V‑head 重排是 (2,16) 而不是 (3,16)**。32 个 V‑head = 16 KV 组 × 2 head。逻辑相同、常量不同。
7. **Patch embed 是 5D**。视觉编码器使用时间维 3D 卷积。GGUF 拆分为 `v.patch_embd.weight` + `v.patch_embd.weight.1`（两个 4D 张量），必须**堆叠**成一个 5D 张量 `[C, 3, 2, H, W]`。
8. **MTP 不在 GGUF 里**。MTP（Multi‑Token Prediction）头的 19 个张量必须从官方 HF 模型（`Qwen/Qwen3.6-35B-A3B`）复制。只需下载 2 个 safetensor 分片（`model-00025`、`model-00026`），不必拉完整 67 GB 参考模型。
9. **源是 Q8_K_P**。和 Qwen3.5 有 bf16 GGUF 不同，HauhauCS 只发布量化 GGUF。使用 `gguf.quants.dequantize()`，它会自动处理 Q8_K → F32 的解量化和 shape 反转。

### 流程

两阶段，不需要 stitch —— 因为 `Qwen3_5MoeForConditionalGeneration` 可以一次性加载完整多模态模型，同时量化 ignore 列表会让视觉保持 bf16：

```
Q8_K_P GGUF + mmproj GGUF + 参考 HF（config/tokenizer + MTP 分片）
  │  step1_convert_qwen36_moe.py        （→ 1045 个张量，22 个分片，约 67 GB）
  ▼                                      （733 文本 + 333 视觉 + 19 MTP）
HuggingFace Safetensors (bf16, 文本 + 视觉 + MTP)
  │  step2_quantize_qwen36_moe.py           （保守：linear_attn + MTP 保 bf16）
  │  step2b_quantize_qwen36_aggressive.py   （激进：除白名单外全量化）
  ▼
最终 NVFP4 模型（~21–22 GB，vLLM 可用）
```

### 两套量化配置

**保守（Conservative）** —— `step2_quantize_qwen36_moe.py`，对标 AEON‑7 / RedHatAI。保留 `linear_attn`（DeltaNet）和 MTP 为 bf16，因为 `linear_attn` 对精度敏感，MTP 质量直接影响投机解码的接受率。

```python
ignore=["lm_head", "re:.*visual.*", "re:.*mlp.gate$",
        "re:.*mlp.shared_expert_gate$", "re:.*linear_attn.*", "re:^mtp.*"]
```

**激进（Aggressive）** —— `step2b_quantize_qwen36_aggressive.py`，对标 sakamakismile。除 `lm_head`、视觉和门控外全部量化，体积更小，换更长上下文。

```python
ignore=["lm_head", "re:.*visual.*", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"]
```

| 配置 | 体积 | RTX 5090 纯文本上下文 | 带视觉 |
|------|------|----------------------|--------|
| 保守 | ~22 GB | ~131K | ~4K |
| 激进 | ~21 GB | ~131K+ | ~65K |

### 快速上手

```bash
# 1. 下载 GGUF 源
hf download HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive \
  --include "*Q8_K_P*" "*mmproj*" --local-dir ./src

# 2. 下载参考 config/tokenizer + 仅 2 个 MTP 分片
hf download Qwen/Qwen3.6-35B-A3B \
  config.json tokenizer.json tokenizer_config.json chat_template.jinja \
  merges.txt vocab.json generation_config.json preprocessor_config.json \
  video_preprocessor_config.json model.safetensors.index.json \
  model-00025-of-00026.safetensors model-00026-of-00026.safetensors \
  --local-dir ./ref

# 3. GGUF 转 HF safetensors（文本 + 视觉 + MTP 注入）
python scripts/step1_convert_qwen36_moe.py \
  --gguf-llm ./src/*Q8_K_P*.gguf \
  --gguf-vision ./src/*mmproj*.gguf \
  --output-dir ./qwen36-bf16-hf \
  --reference-repo Qwen/Qwen3.6-35B-A3B

# 4a. 保守 NVFP4（linear_attn + MTP 保 bf16）
python scripts/step2_quantize_qwen36_moe.py \
  --model-dir ./qwen36-bf16-hf \
  --output-dir ./qwen36-nvfp4-conservative

# 4b. 或者激进 NVFP4（全量化，长上下文首选）
python scripts/step2b_quantize_qwen36_aggressive.py \
  --model-dir ./qwen36-bf16-hf \
  --output-dir ./qwen36-nvfp4-aggressive
```

### 60 GB 内存的绕行方案（MoE 保存路径补丁）

67 GB 的 bf16 模型超过了典型 64 GB 系统内存。step2 脚本使用 `device_map="auto"` 配合磁盘 offload，这会触发 `transformers` / `llmcompressor` 在 MoE + 磁盘 offload 下的一个保存 bug，需要两处补丁：

1. **`transformers/integrations/accelerate.py`**（`load_offloaded_parameter`）：把 `model.get_submodule()` 包在 `try / except AttributeError: continue` 里，跳过不匹配的路径。
2. **`llmcompressor/.../compressed_tensors_utils.py`**（`save_pretrained_wrapper`）：注释掉 `to_accelerate(model)` 和 `from_accelerate(model)`，防止张量名前缀被三重叠加。
3. **保存后 key 重命名**：保存出的 safetensors 会有三重 `model.language_model.language_model.language_model.` 前缀，修复：
   ```python
   new_key = key.replace(
       'model.language_model.language_model.language_model.',
       'model.language_model.'
   )
   ```

如果系统内存 128 GB+，这些补丁不需要（直接不加 `device_map` 加载即可）。

### vLLM 部署（Qwen3.6）

```bash
# 纯文本，RTX 5090 上 100K+ 上下文
docker run --gpus all -v ./qwen36-nvfp4:/model vllm/vllm-openai:nightly \
  --model /model \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 100000 \
  --max-num-seqs 1 \
  --reasoning-parser qwen3 \
  --language-model-only

# 配合 MTP 投机解码（受支持时）
# --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

AEON‑7 建议：如果你的 SM121 GPU 上 CUTLASS NVFP4 内核有问题，设置 `VLLM_TEST_FORCE_FP8_MARLIN=1`。

---

## Gemma 4 E4B → NVFP4

本仓库另一个入口支持 **Gemma 4 E4B** 多模态模型（**文本 + 视觉 + 音频**），覆盖 HauhauCS 的 `Gemma-4-E4B-Uncensored-*-Aggressive` 系列以及其他只发布 GGUF 的 Gemma 4 E4B 微调。

### 为什么需要一套单独的脚本

Gemma 4 和 Qwen3.5 在架构上有几点差异，需要不同的转换路径：

1. **没有混合注意力**。标准 GQA Transformer —— 没有 `(3,16)` value head 重排、没有 `A_log` 域错位、没有 conv1d SSM。Qwen3.5 五个坑里一半直接消失。
2. **RMSNorm 权重按原值存储**。不同于 Qwen3.5 / Gemma 2 / Gemma 3（这些架构下 GGUF 存 `1 + weight`，因为 HF 的 RMSNorm forward 是 `(1 + w) * x`），Gemma 4 的 `Gemma4RMSNorm.forward` 是 `normed_output * self.weight.float()` —— **没有 `+ 1.0`**。这里减 1 会**悄悄**毁掉每一层归一化。
3. **每层 Embedding (PLE)**。E4B 有每层 256 维的 embedding，以及从 hidden_size 到 `42 * 256` 的全局投影。新增张量类型：`embed_tokens_per_layer`、`per_layer_model_projection`、`per_layer_input_gate`、`per_layer_projection`。
4. **视觉和音频塔用量化包裹的线性层**。HF 把每个视觉/音频 linear 包成 `self_attn.q_proj.linear.weight`（嵌套），外加 4 个 FakeQuantize 标量边界（`input_max`、`input_min`、`output_max`、`output_min`，shape 全是 `()`）。GGUF 只有裸权重，没有标量对应项。
5. **Q8_K_P 是 "Permissive"（宽容型）**。HauhauCS 发布的 `Q8_K_P` 变体把 attention Q/K/V 保留为 F16，只把 MLP 和输出投影量化到 Q8_0。在没有完整 bf16 时，这是比纯 Q8 更好的源。
6. **Tied embeddings**。HF state dict 里**没有** `lm_head.weight`，运行时 `lm_head` 绑定到 `embed_tokens`。

Qwen3.5 的 5 个坑里有 3 个（RMSNorm `+1.0`、`A_log` 域、value head `(3,16) → (16,3)` 重排）在 Gemma 4 上**完全不适用**；另外 2 个也大大简化（列优先 shape 反转被 `gguf.quants.dequantize` 自动处理，conv1d unsqueeze 不需要因为 Gemma 4 没有 SSM）。实际操作下 **Gemma 4 E4B 比 Qwen3.5 更简单**，前提是你拿到对的 tensor 映射表。

### 文本张量映射

每层张量（`blk.<i>` → `model.language_model.layers.<i>`）：

| GGUF 后缀 | HF 后缀 |
|-----------|---------|
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
| `layer_output_scale.weight` | `layer_scalar` **（注意：HF 没有 `.weight` 后缀）** |

全局张量：

| GGUF | HF |
|------|----|
| `token_embd.weight` | `model.language_model.embed_tokens.weight` |
| `output_norm.weight` | `model.language_model.norm.weight` |
| `per_layer_model_proj.weight` | `model.language_model.per_layer_model_projection.weight` |
| `per_layer_proj_norm.weight` | `model.language_model.per_layer_projection_norm.weight` |
| `per_layer_token_embd.weight` | `model.language_model.embed_tokens_per_layer.weight` |
| `rope_freqs.weight` | *（跳过 —— HF 运行时计算）* |

总计：17 个每层张量 × 42 层 + 5 个全局张量 = **719 个文本张量**。

### 视觉 + 音频：从参考仓库直接复制

与其重新实现量化包裹的 linear 嵌套结构和手造 FakeQuantize 标量边界，本 Gemma 4 流程直接从一个参考 HF 仓库（默认 `huihui-ai/Huihui-gemma-4-E4B-it-abliterated`）复制 vision、audio、`embed_vision`、`embed_audio` 张量。前提是 Gemma 4 微调通常不会修改视觉/音频塔，它们都是从 `google/gemma-4-e4b-it` 原样继承过来的。

> **对任何新微调使用本流程前，先验证这一点** —— 比如从微调的 mmproj GGUF dequant 几个视觉张量，和参考仓库对比。

### 流程

只有两个阶段，没有 stitch —— 因为 `llmcompressor` 可以用一条 regex ignore 列表在一次 `oneshot` 调用里量化完整的 `Gemma4ForConditionalGeneration` 多模态模型，并一次性写出完整的 checkpoint：

```
GGUF (text Q8_K_P) + 参考 HF 仓库（config/tokenizer + 视觉/音频张量）
  │  step1_convert_gemma4_e4b.py
  ▼
HuggingFace Safetensors (bf16, 完整多模态)
  │  step2_quantize_gemma4_e4b.py   (NVFP4A16, weight‑only)
  ▼
最终模型 (NVFP4 文本 + BF16 视觉/音频，vLLM 可用)
```

### 快速上手

```bash
# 1. 下载 GGUF 源（Q8_K_P 是 HauhauCS 为 Gemma 4 E4B 发布的最高精度）
pip install hf-transfer
HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
  HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive \
  Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf \
  --local-dir ./src

# 2. GGUF 转 HF safetensors（参考仓库自动下载）
python scripts/step1_convert_gemma4_e4b.py \
  --gguf-text ./src/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf \
  --reference-repo huihui-ai/Huihui-gemma-4-E4B-it-abliterated \
  --output-dir ./gemma4-e4b-bf16-hf

# 3. NVFP4A16 量化（weight‑only，无需校准数据）
python scripts/step2_quantize_gemma4_e4b.py \
  --model-dir ./gemma4-e4b-bf16-hf \
  --output-dir ./gemma4-e4b-nvfp4
```

### 为什么用 NVFP4A16 而不是完整 NVFP4（w4a4）

E4B 的架构（每层 embedding、音频编码器、动态 masking）会让 `fx.symbolic_trace` 崩溃，而 w4a4 需要 trace 来做顺序校准。NVFP4A16 直接从权重自身的 min/max 做量化，没有数据通路、没有 trace、不需要校准数据。由于最终 4‑bit 量化是整体误差的主导项，质量影响很小。

### 为什么 Q8_K_P 是可接受的源

`Q8_K_P` 里的 `_P` 是 **Permissive（宽容）**：attention Q/K/V 投影保持 F16（不量化），只有 MLP 和输出投影是 Q8_0。由于最终目标是 NVFP4（4‑bit），Q8 的 ~8‑bit 误差下限**远低于** NVFP4 的噪声下限，对最终质量没有显著影响。

### 已知限制：没有 BF16/F16 源

许多 Gemma 4 E4B 微调只发布到 `Q8_K_P` 为止的 GGUF，没有 bf16/fp16 原生 safetensors。本流程的 `Q8_K_P → bf16` dequant 是后备方案。如果有 bf16 源可用，完全可以跳过 step1，直接把 bf16 safetensors 喂给 `step2_quantize_gemma4_e4b.py`。

### 依赖要求

Gemma 4 流程需要 `transformers >= 5.5`（识别 `Gemma4ForConditionalGeneration`）、`llmcompressor` main 分支、`gguf >= 0.18`。在 RTX 5090 上用 `torch 2.11+cu130` 验证通过。

---

## vLLM 部署

### 配置要求（Qwen3.5）

最终模型的 config 必须包含：

- 顶层 `model_type: "qwen3_5"`
- `architectures: ["Qwen3_5ForConditionalGeneration"]`
- 嵌套的 `text_config` 包含文本模型参数
- `quantization_config` 放在**顶层**（不是 `text_config` 内部）
- 顶层 `dtype: "bfloat16"`
- 权重名使用 `model.language_model.*` 前缀
- ignore 列表中的条目使用 `model.language_model.layers.*` 格式

### Docker Compose 部署

```bash
cp deploy/.env.example deploy/.env
# 编辑 deploy/.env 填入你的路径
docker compose -f deploy/docker-compose.yml up -d
```

详见 [deploy/](deploy/) 目录。

### 快速测试

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

## 仓库元信息（供维护者使用）

建议的 GitHub **About / Description**：

> GGUF → NVFP4 量化管线，支持 Qwen3.5、Qwen3.6‑A3B MoE 和 Gemma 4 E4B，产出单卡 RTX 5090 可部署的 vLLM‑ready 多模态 checkpoint。

建议的 **Topics**（直接粘贴到 repo 设置）：

```
nvfp4  gguf  quantization  vllm  llm-compressor  qwen  qwen3  qwen3-moe
gemma  gemma-4  mixture-of-experts  moe  gated-deltanet  hybrid-attention
multimodal  huggingface  safetensors  rtx-5090  nvidia-fp4  speculative-decoding
```

---

## 许可证

本仓库代码采用 MIT 许可证。模型权重受其原始许可证约束。

## 致谢

- [HauhauCS](https://huggingface.co/HauhauCS) —— 提供 uncensored Qwen3.5‑27B、Qwen3.6‑35B‑A3B 和 Gemma 4 E4B 模型
- [sakamakismile](https://huggingface.co/sakamakismile) —— Qwen3.6 NVFP4 激进量化方案参考
- [AEON‑7](https://huggingface.co/AEON-7) —— Qwen3.6 NVFP4 保守量化思路
- [Kbenkhaled](https://huggingface.co/Kbenkhaled/Qwen3.5-27B-NVFP4) —— 原始 NVFP4 量化方案
- [huihui‑ai](https://huggingface.co/huihui-ai) —— 提供 Gemma 4 流程使用的 HF 格式 Gemma 4 E4B 参考仓库
- [Neural Magic / llm‑compressor](https://github.com/neuralmagic/llm-compressor) —— 量化框架
- [vLLM](https://github.com/vllm-project/vllm) —— 推理服务框架

<!--
SEO 关键词（供搜索引擎识别）：
Qwen3.5 NVFP4 量化, Qwen3.5-27B NVFP4, Qwen3.6 NVFP4, Qwen3.6-35B-A3B NVFP4,
Qwen3 MoE 量化, Gemma 4 E4B NVFP4, GGUF 转 NVFP4, GGUF 转 safetensors,
GGUF 转 HuggingFace, NVFP4 量化管线, Gated DeltaNet 转换, 混合注意力量化,
vLLM FP4, vLLM NVFP4, RTX 5090 大模型, llm-compressor NVFP4, MoE 量化,
MTP 投机解码, Per-Layer Embedding Gemma 4, HauhauCS uncensored 量化.
-->
