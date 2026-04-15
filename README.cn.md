# GGUF 转 NVFP4：Qwen3.5-27B 和 Gemma 4 E4B GGUF 模型量化

一套针对 `transformers` 尚不支持 GGUF 直接加载的模型架构，把 GGUF 模型转换为 NVIDIA FP4 (NVFP4) 量化 HuggingFace safetensors 格式，可直接用 vLLM 部署。

当前支持：
- **Qwen3.5-27B**（混合 Gated-DeltaNet，文本 + 视觉）—— 最初的动机
- **Gemma 4 E4B**（标准 GQA + PLE，文本 + 视觉 + 音频）—— 见 [Gemma 4 E4B 变体](#gemma-4-e4b-变体)

**[English version](README.md)**

## 为什么需要这套流程

Qwen3.5 采用了混合 Gated-DeltaNet 架构，以 3:1 的比例交替使用线性注意力（Mamba 风格）和全 softmax 注意力。目前 `transformers` 不支持直接从 GGUF 格式加载 Qwen3.5。本项目手动从 GGUF 文件中提取张量，进行必要的格式转换，量化为 NVFP4，生成可直接用 vLLM 部署的模型。

Gemma 4 E4B 有类似的问题：`transformers` 支持 `gemma2`/`gemma3` 的 GGUF 加载，但**不支持** `gemma4`。许多微调模型（特别是 HauhauCS 的 uncensored 系列）只发布 GGUF，没有 bf16/fp16 safetensors 源，因此需要同样的 GGUF → HF → NVFP4 路径。

### 流程概览

```
GGUF (bf16 LLM + mmproj 视觉编码器)
  | step1_convert.py
  v
HuggingFace Safetensors (bf16, 分片)
  | step2_quantize.py
  v
NVFP4 量化模型 (仅文本部分)
  | step3_stitch_vision.py
  v
最终模型 (NVFP4 文本 + bf16 视觉, 可直接部署)
```

## GGUF 转换中的坑

Qwen3.5 的 GGUF 格式（来自 llama.cpp）和 HuggingFace safetensors 格式之间有几个不明显的差异。搞错任何一个都会导致模型能正常加载，但输出全是乱码。

### 1. RMSNorm 权重偏移 (+1.0)

GGUF 存储 RMSNorm 权重为 `1 + 学习参数`，而 HuggingFace 只存储`学习参数`本身。

```python
# 受影响：attn_norm, post_attention_norm, output_norm, attn_q_norm, attn_k_norm
# 不受影响：ssm_norm（GroupNorm，不同的归一化方式）
if is_rmsnorm_tensor(gguf_name):
    tensor = (tensor.float() - 1.0).to(torch.bfloat16)
```

不修复的话，每层 LayerNorm/RMSNorm 的输出都会有偏移，前几个 token 可能看起来正常，但输出会迅速退化为乱码。

### 2. A_log 域不匹配

GGUF 存储 SSM 衰减参数为实际值 `A = -exp(A_log)`，而 HuggingFace 期望的是对数空间的 `A_log`。

```python
# GGUF ssm_a -> HF A_log
if gguf_name.endswith(".ssm_a"):
    tensor = (-tensor.float()).log().to(torch.bfloat16)
```

### 3. Value Head 排列顺序 (3,16) vs (16,3)

这是最隐蔽的 bug。Qwen3.5 的线性注意力有 48 个 value head，组织为 16 个 KV 组、每组 3 个 head。GGUF（llama.cpp）按 `(3 个 head, 16 个组)` 顺序存储，而 HuggingFace 期望 `(16 个组, 3 个 head)` 顺序。

**受影响的张量**（所有 linear_attn 层，64 层中的 48 层）：

| 张量 | 形状 | 修复方式 |
|------|------|----------|
| `in_proj_a.weight` | [48, D] | reshape(3,16,D) -> permute(1,0,2) |
| `in_proj_b.weight` | [48, D] | 同上 |
| `dt_bias` | [48] | reshape(3,16) -> permute(1,0) |
| `A_log` | [48] | 同上（在 exp 修复之后） |
| `in_proj_qkv.weight` | [10240, D] | 仅 V 部分 [4096:] |
| `in_proj_z.weight` | [6144, D] | 整体 reshape(3,16,128,D) -> permute(1,0,2,3) |
| `out_proj.weight` | [D, 6144] | 列排列 |
| `conv1d.weight` | [10240, 1, K] | 仅 V 部分 [4096:] |

`in_proj_qkv` 的 QKV 分割：Q=2048 (16 头 x 128 维) + K=2048 + V=6144 (48 头 x 128 维) = 共 10240。只有 V 部分需要重排。

### 4. 列优先形状反转

GGUF 按列优先（Fortran）顺序存储张量形状，PyTorch 使用行优先（C）顺序。

```python
shape = list(reversed(tensor_info.shape))
```

### 5. Conv1d 权重维度调整

GGUF 中的 conv1d 权重是 2D 的 `[channels, kernel_size]`，PyTorch 期望 3D 的 `[channels, 1, kernel_size]`。

```python
if "conv1d.weight" in hf_name and tensor.dim() == 2:
    tensor = tensor.unsqueeze(1).contiguous()
```

## 快速上手

### 前置依赖

```bash
pip install torch transformers>=5.0 safetensors gguf numpy huggingface-hub
pip install llmcompressor datasets  # 量化步骤需要
```

### 第一步：GGUF 转 HuggingFace Safetensors

```bash
python scripts/step1_convert.py \
    --gguf-llm /path/to/model.bf16.gguf \
    --gguf-vision /path/to/mmproj.gguf \
    --output-dir ./model-bf16-hf \
    --reference-repo huihui-ai/Huihui-Qwen3.5-27B-abliterated
```

`--reference-repo` 提供 config.json 和 tokenizer 文件，使用同架构的 HuggingFace 仓库即可。

### 第二步：NVFP4 量化

```bash
python scripts/step2_quantize.py \
    --model-dir ./model-bf16-hf \
    --output-dir ./model-nvfp4
```

使用 `neuralmagic/calibration` 数据集的 512 个样本进行 oneshot NVFP4 量化。以下层不参与量化：
- `lm_head` -- 输出投影保持 bf16
- `visual.*` -- 视觉编码器保持 bf16
- `*.in_proj_a`, `*.in_proj_b` -- SSM 门控参数保持 bf16

### 第三步：缝合视觉权重

```bash
python scripts/step3_stitch_vision.py \
    --bf16-dir ./model-bf16-hf \
    --nvfp4-dir ./model-nvfp4
```

将原始 bf16 视觉权重合并回量化模型，重新映射权重名称（`model.*` -> `model.language_model.*`），更新 config 为 `Qwen3_5ForConditionalGeneration`，并重新分片输出。

## vLLM 部署

### 配置要求

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

## 架构说明

### Qwen3.5 混合注意力

Qwen3.5-27B 有 64 层，线性注意力与全注意力的比例为 3:1：
- **第 0,1,2, 4,5,6, 8,9,10, ... 层**（48 层）：Gated DeltaNet 线性注意力
- **第 3, 7, 11, 15, ... 层**（16 层）：全 softmax 注意力

### 显存预算（RTX 5090, 32GB 显存）

| 组件 | 大小 |
|------|------|
| NVFP4 量化权重 | ~18 GB |
| 视觉编码器 (bf16) | ~0.9 GB |
| KV 缓存 (fp8, 32K 上下文) | ~8 GB |
| 其他开销 | ~3 GB |
| **总计** | **~30 GB** |

建议使用 `gpu_memory_utilization=0.90` 配合 `kv_cache_dtype=fp8`。

## Gemma 4 E4B 变体

本仓库另一个入口支持 **Gemma 4 E4B** 多模态模型（文本 + 视觉 + 音频），覆盖 HauhauCS 的 `Gemma-4-E4B-Uncensored-*-Aggressive` 系列以及其他只发布 GGUF 的 Gemma 4 E4B 微调。

### 为什么需要一套单独的脚本

Gemma 4 和 Qwen3.5 在以下几点有架构差异，需要不同的转换路径：

1. **没有混合注意力。** 标准 GQA Transformer —— 没有 `(3,16)` value head 重排、没有 `A_log` 域错位、没有 conv1d SSM。Qwen3.5 五个坑里一半直接消失。
2. **RMSNorm 权重按原值存储。** 不同于 Qwen3.5 / Gemma 2 / Gemma 3（这些架构下 GGUF 存 `1 + weight`，因为 HF 的 RMSNorm forward 是 `(1 + w) * x`），Gemma 4 的 `Gemma4RMSNorm.forward` 是 `normed_output * self.weight.float()` —— **没有 `+ 1.0`**。这里减 1 会**悄悄**毁掉每一层归一化。
3. **Per-Layer Embedding (PLE)**。E4B 有每层 256 维的 embedding，以及从 hidden_size 到 `42 * 256` 的全局投影。新增张量类型：`embed_tokens_per_layer`、`per_layer_model_projection`、`per_layer_input_gate`、`per_layer_projection`。
4. **视觉和音频塔用量化包裹的线性层。** HF 把每个视觉/音频 linear 包成 `self_attn.q_proj.linear.weight`（嵌套），外加 4 个 FakeQuantize 标量边界（`input_max`、`input_min`、`output_max`、`output_min`，shape 全是 `()`）。GGUF 只有裸权重，没有标量对应项。
5. **Q8_K_P 是 "Permissive"**。HauhauCS 发布的 `Q8_K_P` 变体把 attention Q/K/V 保留为 F16，只把 MLP 和输出投影量化到 Q8_0。在没有完整 bf16 时，这是比纯 Q8 更好的源。
6. **Tied embeddings**。HF state dict 里**没有** `lm_head.weight`，运行时 `lm_head` 绑定到 `embed_tokens`。

Qwen3.5 的 5 个坑里有 3 个（RMSNorm `+1.0`、`A_log` 域、value head `(3,16)→(16,3)` 重排）在 Gemma 4 上**完全不适用**；另外 2 个也大大简化（列优先 shape 反转被 `gguf.quants.dequantize` 自动处理，conv1d unsqueeze 不需要因为 Gemma 4 没有 SSM）。实际操作下 Gemma 4 E4B 比 Qwen3.5 **更简单**，前提是你拿到对的 tensor 映射表。

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
| `rope_freqs.weight` | *(跳过 —— HF 运行时计算)* |

总计：17 个每层张量 × 42 层 + 5 个全局张量 = **719 个文本张量**。

### 视觉 + 音频：从参考仓库直接复制

与其重新实现量化包裹的 linear 嵌套结构和手造 FakeQuantize 标量边界，本 Gemma 4 流程直接从一个参考 HF 仓库（默认 `huihui-ai/Huihui-gemma-4-E4B-it-abliterated`）复制 vision、audio、`embed_vision`、`embed_audio` 张量。这样做的前提是：Gemma 4 微调通常不会修改视觉/音频塔，它们都是从 `google/gemma-4-e4b-it` 原样继承过来的。**对任何新微调使用本流程前，先验证这一点**，比如从微调的 mmproj GGUF dequant 几个视觉张量，和参考仓库对比。

### 流程

只有两个阶段，没有 stitch —— 因为 llmcompressor 可以用一条 regex ignore 列表在一次 `oneshot` 调用里量化完整的 `Gemma4ForConditionalGeneration` 多模态模型，并一次性写出完整的 checkpoint：

```
GGUF (text Q8_K_P) + 参考 HF 仓库（config/tokenizer + 视觉/音频张量）
  | step1_convert_gemma4_e4b.py
  v
HuggingFace Safetensors (bf16, 完整多模态)
  | step2_quantize_gemma4_e4b.py   (NVFP4A16, weight-only)
  v
最终模型 (NVFP4 文本 + BF16 视觉/音频, 可直接部署 vLLM)
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

# 3. NVFP4A16 量化（weight-only，无需校准数据）
python scripts/step2_quantize_gemma4_e4b.py \
  --model-dir ./gemma4-e4b-bf16-hf \
  --output-dir ./gemma4-e4b-nvfp4
```

### 为什么用 NVFP4A16 而不是完整 NVFP4 (w4a4)

E4B 的架构（每层 embedding、音频编码器、动态 masking）会让 `fx.symbolic_trace` 崩溃，而 w4a4 需要 trace 来做顺序校准。NVFP4A16 直接从权重自身的 min/max 做量化，没有数据通路、没有 trace、不需要校准数据。由于最终 4-bit 量化是整体误差的主导项，质量影响很小。

### 为什么 Q8_K_P 是可接受的源

`Q8_K_P` 里的 `_P` 是 **Permissive**：attention Q/K/V 投影保持 F16（不量化），只有 MLP 和输出投影是 Q8_0。由于最终目标是 NVFP4 (4-bit)，Q8 的 ~8-bit 误差下限**远低于** NVFP4 的噪声下限，对最终质量没有显著影响。

### 已知限制：没有 BF16/F16 源

许多 Gemma 4 E4B 微调只发布到 `Q8_K_P` 为止的 GGUF，没有 bf16/fp16 原生 safetensors。本流程的 `Q8_K_P → bf16` dequant 是后备方案。如果有 bf16 源可用，完全可以跳过 step1，直接把 bf16 safetensors 喂给 `step2_quantize_gemma4_e4b.py`。

### 依赖要求

Gemma 4 流程需要 `transformers >= 5.5`（识别 `Gemma4ForConditionalGeneration`）、`llmcompressor` main 分支、`gguf >= 0.18`。在 RTX 5090 上用 torch `2.11+cu130` 验证通过。

## 许可证

本仓库代码采用 MIT 许可证。模型权重受其原始许可证约束。

## 致谢

- [HauhauCS](https://huggingface.co/HauhauCS) 提供 uncensored Qwen3.5-27B 和 Gemma 4 E4B 模型
- [Kbenkhaled](https://huggingface.co/Kbenkhaled/Qwen3.5-27B-NVFP4) 提供 NVFP4 量化方案
- [huihui-ai](https://huggingface.co/huihui-ai) 提供 Gemma 4 流程使用的 HF 格式 Gemma 4 E4B 参考仓库
- [Neural Magic / llm-compressor](https://github.com/neuralmagic/llm-compressor) 量化框架
- [vLLM](https://github.com/vllm-project/vllm) 推理服务框架
