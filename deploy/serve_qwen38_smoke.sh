#!/usr/bin/env bash
# Historical 4096-token smoke profile; requires a Qwen3.8-capable vLLM image.
set -euo pipefail
: "${MODEL_DIR:?Set MODEL_DIR to the converted checkpoint}"
: "${VLLM_IMAGE:?Set VLLM_IMAGE to an available Qwen3.8-capable image}"
exec docker run --rm --gpus all --ipc=host --network=host \
  -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$(realpath "$MODEL_DIR"):/model:ro" \
  "$VLLM_IMAGE" /model \
  --served-model-name qwen38-nvfp4 \
  --host 127.0.0.1 --port "${PORT:-8030}" \
  --max-model-len 4096 --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 --max-num-seqs 1 \
  --max-num-batched-tokens 1024 --enable-prefix-caching \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
