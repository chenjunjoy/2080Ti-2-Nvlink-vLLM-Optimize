#!/usr/bin/env bash
# ============================================================
# 在原模型 / 无审查模型之间切换(两者显存互斥,只能跑一个)
# 用法:
#   ./switch-model.sh original      # 切到 Qwen3.8-27B-NVFP4 (8000)
#   ./switch-model.sh uncensored    # 切到 Qwen3.8-27B-Uncensored (8002)
#   ./switch-model.sh status        # 查看当前状态
# ============================================================
set -euo pipefail

wait_health() {  # $1=port  $2=seconds
  local port=$1 secs=$2
  for i in $(seq "$secs"); do
    if curl -sf -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

status() {
  echo "=== 当前状态 ==="
  for s in vllm vllm-uncensored; do
    printf "  %-18s active=%-8s enabled=%s\n" "$s" "$(systemctl is-active $s 2>&1)" "$(systemctl is-enabled $s 2>&1)"
  done
  for p in 8000 8002; do
    printf "  :%s/health -> %s\n" "$p" "$(curl -s -m3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$p/health 2>/dev/null)"
  done
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  GPU /'
}

case "${1:-}" in
  original)
    echo "==> 切换到原模型 (Qwen3.8-27B-NVFP4, 8000)"
    sudo systemctl stop vllm-uncensored 2>/dev/null || true
    sudo systemctl start vllm
    wait_health 8000 300 && echo "    就绪 ✓ (http://192.168.1.13:8000/v1)" || echo "    等待超时,查看: journalctl -u vllm -n 80"
    ;;
  uncensored)
    echo "==> 切换到无审查模型 (Qwen3.8-27B-Uncensored, 8002)"
    sudo systemctl stop vllm 2>/dev/null || true
    sudo systemctl start vllm-uncensored
    wait_health 8002 300 && echo "    就绪 ✓ (http://192.168.1.13:8002/v1)" || echo "    等待超时,查看: journalctl -u vllm-uncensored -n 80"
    ;;
  status|"")
    status
    ;;
  *)
    echo "用法: $0 {original|uncensored|status}"; exit 1
    ;;
esac
