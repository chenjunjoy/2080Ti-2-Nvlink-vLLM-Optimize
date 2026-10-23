#!/usr/bin/env bash
# ============================================================
# Qwen3.8-27B-AWQ-INT4 + MTP3 路线 —— 用于与 NVFP4 版做同口径对照
# 模型: cyankiwi/Qwen3.8-27B-AWQ-INT4 (compressed-tensors int4 pack-quantized)
# 投机: MTP3 (模型自带 mtp 头)
# 192K / FP16 KV(8.5GB 硬上限) / max-num-seqs=64 / 纯文本
# 端口 8003
# ============================================================
set -euo pipefail

PROJECT=/home/jim/vLLM-2080Ti-Definitive
MODEL=/srv/share/models/cyankiwi-Qwen3.8-27B-AWQ-INT4
SERVED_NAME=Qwen3.8-27B-AWQ-INT4
PORT=8003

cd "$PROJECT"

export PATH="$PROJECT/.venv/bin:/usr/local/cuda-13.0/bin:/home/jim/.local/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH=/usr/local/cuda-13.0
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc
export HF_ENDPOINT=https://hf-mirror.com
export TORCH_CUDA_ARCH_LIST=7.5
export FLASHINFER_ENABLE_AOT=1
export FLASHINFER_WORKSPACE_BASE="$PROJECT"
export FLASHQLA_ROOT="$PROJECT/.deps/FlashQLA-SM70-SM75"
export TORCH_EXTENSIONS_DIR="$PROJECT/.deps/FlashQLA-SM70-SM75/.torch_extensions_vllm_flashqla_legacy"

# ---- 博主的 fork 专用环境变量 ----
export VLLM_SM75_SPEC_SYNC_MODE=auto

# Keep the vllm-turing plugin scoped to the uncensored 27B route; this AWQ
# route uses the fork's own kernels (2026-10-23).
export VLLM_TURING_DISABLE=1
export VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$PROJECT/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --dtype half \
  --tensor-parallel-size 2 \
  --disable-custom-all-reduce \
  --quantization compressed-tensors \
  --max-model-len 196608 \
  --kv-cache-memory-bytes 8500000000 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 64 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --generation-config vllm \
  --language-model-only \
  --skip-mm-profiling \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --per-request-spec-decode-metrics detailed \
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'
