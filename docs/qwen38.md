# Qwen3.8-27B: native BF16 → ModelOpt NVFP4 → compressed-tensors

Reviewed against local files on **2026-10-12**. This supplements the older GGUF
pipelines; it is a native HuggingFace BF16 route, not a tested Qwen3.8 GGUF converter.

## 中文说明

近期已实际使用的流程是：原生 BF16 多模态 checkpoint → ModelOpt NVFP4 W4A4
→ compressed-tensors `nvfp4-pack-quantized` → 补齐同源 BF16 MTP → 静态检查
→ vLLM 文本/MTP smoke test。语言模型 Linear 量化；视觉、MTP、embedding、
`lm_head` 和 Gated DeltaNet `conv1d` 保留非 NVFP4 状态。

本次核对发现，本机 Huihui 目录仅剩元数据，索引引用的权重分片缺失；历史量化和
vLLM Docker 镜像也不在本地。下面提供整理后的脚本和历史证据，不代表今天重新
完成了量化或推理。需要先恢复精确来源的权重，并准备兼容运行环境。

Huihui 历史记录证明的是 **4096 context 下的文本 + MTP n=3**。
333 个视觉张量的静态存在，不等于图片/视频推理通过；其他变体的多模态测试不能
转移到 Huihui。也不能把另一个 GGUF/Strata 后端的 200k+ 结果用于本 NVFP4 模型。

## Historical source and validation

| Item | Recorded value |
|---|---|
| Source | `huihui-ai/Huihui-Qwen3.8-27B-abliterated` |
| Source revision | `d42ca8978c5a66e92c3446d46e8adfe03ef692ff` |
| Derived artifact ID | `lyf/Qwen3.8-27B-Huihui-Abliterated-NVFP4-MTP-VL` |
| Calibration | CNN/DailyMail 3.0.0; 20 samples, maximum sample length 8192 |
| Format | compressed-tensors `nvfp4-pack-quantized`, W4A4 |
| Tensors | 2687 total; 333 BF16 vision; 15 BF16 MTP; 3 shards |
| GPU | RTX 5090 32 GB; historical power cap 450 W |
| Runtime image tag | `vllm/vllm-openai:qwen38-x86_64-cu130` |
| Profile | context 4096, FP8 KV, one sequence, MTP n=3 |
| Historical checks | health, models, 1024-output-token text, native MTP |
| MTP measurement | 572/1362 accepted drafts (42.0%); mean acceptance length 2.33 |
| Per-position acceptance | 0.602 / 0.429 / 0.295 |
| Client elapsed | 10.73 seconds for that 1024-token request |

Evidence copied from the local release metadata, without changing its historical
claims: [build manifest](evidence/qwen38-huihui/BUILD_MANIFEST.json),
[validation report](evidence/qwen38-huihui/VALIDATION_REPORT.json).
These are archived reports, not independent reruns or general quality benchmarks.
The runtime image digest and complete package lock were not captured in these
Huihui reports; a mutable image tag alone does not reproduce that environment.
The calibration settings are sample limits, not proof of exactly 163,840 input tokens.

## Reproduce in an isolated environment

Use a Qwen3.8-capable Transformers/ModelOpt stack and a compressed-tensors build
exporting `ModelOptNvfp4Converter` and `convert_checkpoint`.
[requirements-qwen38.txt](../requirements-qwen38.txt) lists this route's separate
dependencies; it is not a tested version lock. The older llm-compressor requirements
remain applicable to the older pipelines. GPU quantization requires substantial
CPU RAM and offload disk; the historical Huihui budgets were 4 GiB GPU / 52 GiB CPU.

Run from the repository root. Choose new, empty output directories. The scripts
refuse to replace populated quantization/conversion outputs.

```bash
export SOURCE=/absolute/path/to/huihui-bf16
export RAW=/absolute/path/to/new-modelopt-output
export FINAL=/absolute/path/to/new-compressed-tensors-output
export OFFLOAD=/absolute/path/to/offload

hf download huihui-ai/Huihui-Qwen3.8-27B-abliterated \
  --revision d42ca8978c5a66e92c3446d46e8adfe03ef692ff --local-dir "$SOURCE"

python scripts/quantize_qwen38_modelopt.py \
  --model-id "$SOURCE" --output-dir "$RAW" --offload "$OFFLOAD" \
  --gpu-budget 4GiB --cpu-budget 52GiB \
  --calib-size 20 --calib-seq 8192 --dataset cnn_dailymail

# Model loading/export may omit native MTP; always restore from this SAME source.
python scripts/restore_qwen38_mtp.py "$SOURCE" "$RAW"
python scripts/convert_qwen38_modelopt.py "$RAW" "$FINAL" --workers 2
python scripts/restore_qwen38_mtp.py "$SOURCE" "$FINAL"
python scripts/validate_qwen38_checkpoint.py "$FINAL" --source "$SOURCE"
```

The converter detects quantized modules by `.input_scale`, then excludes ordinary
unquantized state using exact module names. This avoids treating norms/bias-only
modules as quantized Linear layers. It also excludes vision, MTP, `lm_head`, token
embeddings and `conv1d`. Scale/payload conversion is delegated to compressed-tensors,
not performed by renaming tensors manually. Processor/tokenizer assets present in
the raw export are copied when missing; confirm image/video processor metadata from
the BF16 source before attempting multimodal serving.

The MTP restorer checks existing tensors byte-for-byte, fills missing tensors only,
and rejects mismatches. Never use an official checkpoint or another behavioral
variant as a substitute for the quantized source's MTP head.

The standalone validator uses only Python's standard library and reads safetensors
headers without loading the language model. It checks shard presence, index mapping,
byte bounds, expected MTP/vision counts and BF16 types. With `--source`, it also
compares all MTP/vision tensor bytes to the original source. This does not establish
quantization quality, runtime compatibility, or source revision authenticity.

## Runtime smoke profile

Prepare a compatible image explicitly; the historical custom tag may not be publicly
available. No scripts build images, download weights, stop containers or start a
service automatically as part of validation.

```bash
MODEL_DIR="$FINAL" VLLM_IMAGE=your-qwen38-capable-image PORT=8030 \
  bash deploy/serve_qwen38_smoke.sh

curl --fail http://127.0.0.1:8030/health
curl --fail http://127.0.0.1:8030/v1/models
curl --fail http://127.0.0.1:8030/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-nvfp4","messages":[{"role":"user","content":"用中文解释 17+25 的结果。"}],"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```

Check server logs/metrics for native MTP draft acceptance. HTTP success alone does
not prove MTP is active. Test actual image/video requests separately. Any larger
context, production workload, tool calling or client-specific Responses/compaction
endpoint needs its own test; this profile makes no 200k-context claim.

## Current checkout verification

On 2026-10-12 the local release index still described 2687 tensors, but its referenced
`.safetensors` files were absent. The validator correctly fails on the missing first
shard. Full source-byte validation, GPU quantization and serving were not rerun.
The scripts are adaptations of local August workflows, not a newly GPU-validated release.

Repository checks for the new helpers:

```bash
python -m unittest discover -s tests -v
```

Requires PyTorch and safetensors. Five synthetic CPU tests cover a valid checkpoint,
a missing shard, an incorrect index, a wrong-source MTP tensor, and partial MTP
restoration with an idempotent second run. These tests do not exercise GPU quantization.
