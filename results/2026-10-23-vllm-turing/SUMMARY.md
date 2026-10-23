# vLLM 无审查版 × vllm-turing 优化（2026-10-23）

模型：`Qwen3.8-27B-abliterated-NVFP4` + DFlash2/7，TP2，2×2080Ti 22G，FP8 KV，262144 context。
插件：`/srv/share/vllm-turing`（editable 安装进 vLLM-2080Ti-Definitive 的 venv）。
服务：端口 8002，`/home/jim/scripts/vllm-uncensored-run.sh`。

## 配置演进

| 版本 | 关键变化 |
|---|---|
| baseline | 原脚本：batch 2048、gpu-util 0.96、无插件 |
| plugin | vllm-turing 插件，其余不变 |
| P1 | plugin + `--mamba-ssm-cache-dtype float16 --block-size 880` |
| P2 | P1 + batch 7040（出现 OOM 警告，不采用） |
| **P4 / final** | **P1 + batch 4096 + gpu-util 0.95**（无 OOM） |

## 结果

| 指标 | baseline | plugin | P1 | **P4/final** | final vs baseline |
|---|---:|---:|---:|---:|---:|
| decode 256 tok | 93.96 | 96.76 | 100.02 | **99.57** | **+6.0%** |
| 6K prefill | 5.05 s | 4.97 s | 5.01 s | **4.96 s** | **+1.8%** |
| 23.8K prefill | 1095.7 | 1156.6 | 1156.9 | **1173.5** | **+7.1%** |
| 89.2K prefill | 864.4 | 1031.0 | 1034.8 | **1057.5** | **+22.3%** |
| 89K decode | 81.0 | 84.3 | 87.4 | **87.7** | **+8.2%**（P2 曾 94.2，波动大） |
| KV 容量 | 348,979 | 419,430 | 461,830 | **443,357** | **+27.0%** |

单位：prefill/decode = tok/s。

## 插件贡献

- `embed8` / `lmhead` / `fiws`：KV 349k → 419k，decode +3%。
- `fiattn8` / `fdec` / `gdn` / `customar` 等：89K prefill 864 → 1031（+19%）。
- `--mamba-ssm-cache-dtype float16 --block-size 880`：KV 419k → 462k，decode 再 +3%。
- `batch 7040`：长上下文 decode 略快，但长 prompt 触发 PyTorch OOM 警告（仍能继续）；改为 4096 + util 0.95 后无 OOM，保留大部分收益。

## 质量/稳定性

- 短答 3/3：巴黎、5、疑是地上霜。
- 针召回：22K 和 89K 各 1 次，均正确（`7391-KX` / `8642-QZ`）。
- P4/final 长请求 OOM 计数：0；空闲显存约 2.0 GiB/卡。
- 注意：plugin 的 `vitlowmem` 依赖 `vllm.vllm_flash_attn.layers`，本机 vLLM 0.2.2 没有该模块，已用 `VLLM_TURING_VIT_LOWMEM=0` 关闭视觉低内存补丁。

## 未完成

- 仓库主线的 **W8A8 SmoothQuant** 路线未做：需要 BF16 原始权重 + 校准（约 55 GB 下载/转换），本机只有 NVFP4/INT4 权重；因此没有复现 README 的 110 tok/s / 3550 prefill。
- `--enable-mamba-fine-grained-prefix-cache --prefix-match-unit 80` 未验证（当前 prefix cache 已能把 22K 重复提示的 TTFT 从 19.1 s 降到 1.14 s）。
- 200K+ 单请求、多并发、工具调用未压测。

## 回滚

```bash
# 停服务（手工进程）
python3 /tmp/opencode/stop-vllm.py 8002   # 或 systemctl stop vllm-uncensored

# 恢复脚本
cp /srv/share/vLLM-optimize/backups/vllm-uncensored-run.sh.pre-turing-20261023-180714 \
   /home/jim/scripts/vllm-uncensored-run.sh

# 卸载插件
uv pip uninstall --python /home/jim/vLLM-2080Ti-Definitive/.venv/bin/python vllm-turing
```
