#!/usr/bin/env bash
# ============================================================
# vLLM 运行脚本(供 systemd 服务 vllm.service 调用)
# 模型: Qwen3.8-27B-NVFP4 + DFlash2 投机解码, 256K 上下文, 工具调用
# 不要手动执行本脚本(前台阻塞); 用 start-vllm.sh / systemctl 管理
# ============================================================
set -euo pipefail

PROJECT=/home/jim/vLLM-2080Ti-Definitive
MODEL=/home/jim/models/Qwen3.8-27B-NVFP4
DRAFT=/home/jim/models/Qwen3.8-27B-DFlash2

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
export VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1
export VLLM_DISABLE_TILELANG=1
export VLLM_ENFORCE_STRICT_TOOL_CALLING=1
export VLLM_KV_CACHE_LAYOUT=BLHNC
export VLLM_SM75_SPEC_SYNC_MODE=safe
export VLLM_USE_V2_MODEL_RUNNER=1

# Keep the vllm-turing plugin scoped to the uncensored 27B route; this route is
# validated on the fork's own kernels (2026-10-23).
export VLLM_TURING_DISABLE=1

exec "$PROJECT/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model "$MODEL" \
  --served-model-name Qwen3.8-27B-NVFP4 \
  --dtype half \
  --tensor-parallel-size 2 \
  --generation-config auto \
  --gpu-memory-utilization 0.96 \
  --max-model-len 262144 \
  --enable-chunked-prefill \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --pipeline-parallel-size 1 \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --disable-log-stats \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt '{"image":64,"video":0,"audio":0}' \
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}' \
  --per-request-spec-decode-metrics detailed \
  --speculative-config "{\"method\":\"dflash\",\"num_speculative_tokens\":7,\"model\":\"$DRAFT\",\"attention_backend\":\"TRITON_ATTN\",\"kv_cache_dtype\":\"float16\",\"draft_sample_method\":\"greedy\"}" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8],"max_cudagraph_capture_size":8}'
