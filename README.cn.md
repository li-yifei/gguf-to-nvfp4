# GGUF 转 NVFP4：Qwen3.5-27B GGUF 模型量化指南

将 Qwen3.5-27B 的 GGUF 模型转换为 NVIDIA FP4 (NVFP4) 量化格式，用于 vLLM 高效部署的完整流程。

**[English version](README.md)**

## 为什么需要这套流程

Qwen3.5 采用了混合 Gated-DeltaNet 架构，以 3:1 的比例交替使用线性注意力（Mamba 风格）和全 softmax 注意力。目前 `transformers` 不支持直接从 GGUF 格式加载 Qwen3.5。本项目手动从 GGUF 文件中提取张量，进行必要的格式转换，量化为 NVFP4，生成可直接用 vLLM 部署的模型。

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

## 许可证

本仓库代码采用 MIT 许可证。模型权重受其原始许可证约束。

## 致谢

- [HauhauCS](https://huggingface.co/HauhauCS) 提供 uncensored Qwen3.5-27B 模型
- [Kbenkhaled](https://huggingface.co/Kbenkhaled/Qwen3.5-27B-NVFP4) 提供 NVFP4 量化方案
- [Neural Magic / llm-compressor](https://github.com/neuralmagic/llm-compressor) 量化框架
- [vLLM](https://github.com/vllm-project/vllm) 推理服务框架
