#!/usr/bin/env bash
# ============================================================
# vLLM 运行脚本(供 systemd 服务 vllm-uncensored.service 调用)
# 模型: Qwen3.8-27B-abliterated-NVFP4(无审查/去拒答)
# 投机解码: DFlash2/7(草稿 incoai/Qwen3.8-27B-DFlash2)
#   实测(本机无审查模型, 作者口径): 4K/128 解码 199.57 tok/s, 32K/512 解码 190.63 tok/s
#   —— 约为 MTP/4(114.46 / 108.54)的 1.74 倍
# 256K 上下文, text+image, 工具调用, 端口 8002
# 命令由 launcher.sh DRY_RUN 生成,保证与项目路由一致。
# 不要手动执行(前台阻塞); 用 start/stop-uncensored.sh 或 systemctl 管理
# ============================================================
set -euo pipefail

PROJECT=/home/jim/vLLM-2080Ti-Definitive
MODEL=/srv/share/models/Qwen3.8-27B-abliterated-NVFP4
DRAFT=/srv/share/models/Qwen3.8-27B-DFlash2
SERVED_NAME=Qwen3.8-27B-Uncensored
PORT=8002

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

exec "$PROJECT/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
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
