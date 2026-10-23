# vLLM @ 2×RTX 2080 Ti + NVLink —— 优化复盘

> 硬件：2 × RTX 2080 Ti 22GB（NVLink NV2，2 × 25.781 GB/s），76 GB 内存
> 软件：**vLLM 2080 Ti Definitive Edition**（`github.com/weicj/vLLM-2080Ti-Definitive`）
> 版本：`v0.2.2-post3`（上游基线 `v0.29.1rc0-33-gb23433088b`），CUDA 13.0，torch 2.13.0+cu130，SM75
> 服务：`vllm.service`（原模型 :8000）、`vllm-uncensored.service`（无审查 :8002），互斥、均不开机自启
> 整理日期：2026-10-19（2026-10-22 追加 §9：博主同款 AWQ-INT4 复刻 + text-only A/B）
> 依据：`/home/jim/vLLM-2080Ti-Definitive`、`/home/jim/scripts/`、systemd 单元、`build-logs/`、`run-logs/`、`/home/jim/launch*.log`、`/srv/share/strata-ops/BENCHMARK-vs-vLLM.zh-CN.md`

---

## 0. TL;DR

1. **上游 fork 提供 SM75 适配能力**（Marlin、FlashInfer/FlashQLA、TurboQuant/INT8 KV、MTP/DFlash2、CUDA Graph）；**本机的工作是把它在 2×2080 Ti + NVLink 上跑通并调优**。
2. 本机关键决策：
   - **DFlash2/7 投机解码** 取代模型自带 MTP/4：本地实测 decode **93.7 vs 69.9 tok/s（+34%）**，prefill 基本不变。
   - **FP8 KV + 256K 上下文**：先在启动期解决 KV 显存不足（4.96 GiB 需求 vs 4.77 可用），最终以“关桌面 + FP8 KV + 0.96 显存占比”跑通 262144。
   - **FlashQLA legacy 预编译 + FlashInfer AOT + 禁用 TileLang**：避免运行期 JIT、适配 SM75。
   - **NVLink 保留**：对 TP=2 的 **6K 提示读取 +35.6%**（4.66 s vs 6.32 s），decode 仅 +1.5%。
3. 与 Strata 的横向结论：**vLLM 27B 长提示读取最快（4.66 s）、256K 上下文、生成 93.7 tok/s（无审查 + DFlash2）**；Strata IQ2_XS 是 125B、生成 84.3 tok/s、6K 读取 6.61 s。
4. 仍需注意：SM75 **无原生 FP8/FP4**（走 Marlin weight-only）；NVFP4 + FP16KV 有**明显质量塌陷**，推荐 FP8 KV；`max_num_batched_tokens=2048` 与投机解码搭配被官方警告“可能次优”。

---

## 1. 先分清：上游 fork 与本机改动

| 类别 | 内容 | 归属 |
|---|---|---|
| SM75 源码适配 | Turing/SM75 kernel 选择、Marlin、FlashInfer/FlashQLA 路由、TurboQuant/INT8 KV、CUDA Graph、hybrid Mamba 前缀缓存 | **上游 fork（weicj）** |
| 启动 profile | `profiles/2x2080Ti/...` 的 .env 路线预设 | **上游 fork** |
| 本机构建 | CUDA 13 + torch 2.13 + `TORCH_CUDA_ARCH_LIST=7.5`、FlashQLA SM70/SM75 legacy 扩展、FlashInfer AOT、glibc 补丁 | **本机** |
| 本机运行参数 | TP=2、FP8 KV、256K、DFlash2/7、cudagraph size 8、chunked prefill、prefix cache、mamba align | **本机** |
| 本机服务化 | 两个 systemd 服务、互斥、切换脚本、桌面 GPU 释放 | **本机** |
| 本机实测 | DFlash2 vs MTP、NVLink 开/关、vs Strata | **本机** |

> 本复盘只把“本机实际做过并有证据”的内容算作优化；上游 fork 的能力单独标注。

---

## 2. 本机优化清单

### 2.1 构建与工具链

- **目标架构固定 SM75**：`TORCH_CUDA_ARCH_LIST=7.5`，避免为其它架构编译/运行期 JIT。
- **FlashQLA SM70/SM75 legacy 扩展预编译**：
  - `FLASHQLA_ROOT=$PROJECT/.deps/FlashQLA-SM70-SM75`
  - `TORCH_EXTENSIONS_DIR=.../.torch_extensions_vllm_flashqla_legacy`
  - 产物符号：`gdn_forward`、`gdn_forward_varlen`
  - 作用：GDN/Mamba 预填走 `flashqla_legacy`，避免请求期首次 JIT 卡顿。
- **FlashInfer AOT**：`FLASHINFER_ENABLE_AOT=1`。
- **CUDA 13 + 新 glibc 编译修复**：`toolchain-patches/cuda-13.0-glibc-2.41-rsqrt.patch`
  - 把 `__GNUC_PREREQ(2,41)` 修正为 `__GLIBC_PREREQ(2,41)`；
  - 在 glibc ≥ 2.42 时为 `rsqrt/rsqrtf` 增加 `noexcept(true)`；
  - 否则 CUDA 头文件在现代 glibc 上编译失败。
- 依赖镜像：torch/triton 走 SJTU 镜像，模型走 `HF_ENDPOINT=https://hf-mirror.com`。

### 2.2 运行期环境变量（两个服务一致）

```bash
TORCH_CUDA_ARCH_LIST=7.5
FLASHINFER_ENABLE_AOT=1
FLASHQLA_ROOT=$PROJECT/.deps/FlashQLA-SM70-SM75
TORCH_EXTENSIONS_DIR=$PROJECT/.deps/FlashQLA-SM70-SM75/.torch_extensions_vllm_flashqla_legacy
VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1   # 允许 hybrid Mamba + 投机解码走 FULL CUDA Graph
VLLM_DISABLE_TILELANG=1                  # SM75 上禁用 TileLang 路径
VLLM_ENFORCE_STRICT_TOOL_CALLING=1
VLLM_KV_CACHE_LAYOUT=BLHNC               # hybrid Mamba 的 KV 布局
VLLM_SM75_SPEC_SYNC_MODE=safe            # SM75 投机解码同步策略
VLLM_USE_V2_MODEL_RUNNER=1
```

### 2.3 服务参数（关键项）

```text
--dtype half                      # 2080 Ti 无原生 FP8/FP4 → Marlin weight-only
--tensor-parallel-size 2          # 双卡 TP
--gpu-memory-utilization 0.96
--max-model-len 262144            # 256K
--kv-cache-dtype fp8              # 256K KV 装得下
--mamba-cache-mode align          # hybrid Mamba/GDN 对齐
--enable-chunked-prefill
--max-num-seqs 1
--max-num-batched-tokens 2048
--enable-prefix-caching
--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'
--speculative-config '{"method":"dflash","num_speculative_tokens":7,
   "model":<DFlash2>,"attention_backend":"TRITON_ATTN",
   "kv_cache_dtype":"float16","draft_sample_method":"greedy"}'
--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE",
   "cudagraph_capture_sizes":[8],"max_cudagraph_capture_size":8}'
--reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice
--limit-mm-per-prompt '{"image":64,"video":0,"audio":0}'
```

### 2.4 显存与启动问题（本机踩坑）

| 现象 | 日志证据 | 处理 |
|---|---|---|
| `Free memory cuda:0 (20.49/21.49 GiB) < 0.96 (20.63 GiB)` | `launch.log` | 用 `display-off.sh` 关掉桌面/GDM，释放约 1 GiB |
| 262144 需要 **4.96 GiB** KV，但只有 **4.77 GiB** | `launch2.log` | 先用 248832 做编译预热；关桌面 + FP8 KV 后 262144 可跑 |
| 投机解码 + `max_num_batched_tokens=2048` 警告“可能次优” | `run-logs/...prewarm...log` | 待调（见 §6） |
| draft 不支持外部多模态 embedding | 同上 | 图像理解由 target 做，draft 用纯文本 |
| `vllm._deepselect_C` 缺失 | 同上 | 回退路径，非致命 |

### 2.5 服务化与互斥

- `vllm.service`（原模型，`Qwen3.8-27B-NVFP4`，:8000）
- `vllm-uncensored.service`（`Qwen3.8-27B-abliterated-NVFP4`，:8002）
- `Conflicts=` 互斥；两者都与 Strata 互斥（双卡显存只能跑一个）。
- `switch-model.sh {original|uncensored|status}` 一键切换 + 健康检查。

---

## 3. DFlash2 vs 自带 MTP（本机实测）

同一模型（无审查 abliterated）、同一方法，只换草稿头：

| 草稿头 | 每次推测 token | decode | 6K 提示 | 每窗口接受 |
|---|---:|---:|---:|---:|
| **DFlash2（外部草稿）** | 7 | **93.2–93.7 tok/s** | 5.08–5.11 s | ~3.46 token |
| 模型自带 MTP | 4 | 69.9 tok/s | 5.17 s | ~2.98 token |

- **DFlash2 快约 34%**，prefill 基本不变；原因是接受率更高（3.46 vs 2.98）。
- 决定：**长期保持 DFlash2**。
- 注意：fork 作者口径为 4K/128 **199.57**、32K/512 **190.63** tok/s（约为 MTP/4 的 1.74×），那是**高接受率合成输入**；本机真实提示词下是 93.7 tok/s，两者不矛盾，但不能混用。

---

## 4. NVLink 到底有没有用（vLLM TP=2）

| 指标 | NVLink 开 | NVLink 关 | 收益 |
|---|---:|---:|---:|
| decode | 75.4 tok/s | 74.3 tok/s | **+1.5%** |
| 6K 提示读取 | **4.66 s** | 6.32 s | **+35.6%** |

- 关闭方法：`VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 NCCL_P2P_DISABLE=1`。
- 结论：**NVLink 主要提速 prefill（长提示的 AllReduce/通信），decode 受显存带宽限制，收益很小。**
- 因此保留 NVLink；但不要期待它显著提升生成速度。

---

## 5. 与 Strata / 其它服务横向对比（本机实测）

| 服务 · 模型 | decode | TTFT | 6K 提示读取 |
|---|---:|---:|---:|
| Strata · IQ2_XS（125B） | 84.3 tok/s | 78 ms | 6.61 s |
| Strata · Orca IQ3_XXS（125B） | 69.6 tok/s | 86 ms | 6.72 s |
| vLLM · 27B NVFP4（NVLink 开） | 75.4 tok/s | 92 ms | **4.66 s** |
| vLLM · 27B 无审查 + DFlash2 | **93.7 tok/s** | 133 ms | 5.08 s |
| vLLM · 27B 无审查 + MTP | 69.9 tok/s | 141 ms | 5.17 s |

- **读长提示 / 256K 上下文** → vLLM 27B。
- **生成最快（单请求）** → vLLM 27B 无审查 + DFlash2（93.7）。
- **模型规模 / 中文** → Strata IQ2_XS（125B，84.3）。
- 方法学：vLLM 会把约 2.8 个 token 合并进一个 SSE chunk，按 chunk 计数会低估 decode；本表已用服务端 `completion_tokens` 校正。

---

## 6. 已知问题与后续可优化项

1. **`max-num-batched-tokens=2048` 与投机解码**
   - 官方警告：`max_num_scheduled_tokens is set to 2048 ... may lead to suboptimal performance`。
   - 建议：提高到 4096/8192（或减少 `num_speculative_tokens`），实测 decode 是否改善。
2. **FP8/FP4 非原生**
   - SM75 走 Marlin weight-only；计算密集负载可能退化。
   - **NVFP4 + FP16KV 有显著质量塌陷**，务必用 **FP8 KV**。
3. **上下文与显存的取舍**
   - 262144 + FP8 KV 已跑通，但启动余量很小；若再开并发/视觉，考虑 248832 或 `--vram` 类余量参数。
4. **单并发**
   - `--max-num-seqs 1` 是为单用户低延迟调的；多用户需重新调 `max-num-seqs` / `max-num-batched-tokens` / KV。
5. **草稿头选择**
   - 可本地再试 DFlash2 的 K=5/9、DFlash(非2)、以及 FP8 目标模型（w8a16）对比质量/速度。
6. **TileLang/DeepSelect 缺失**
   - `VLLM_DISABLE_TILELANG=1`、`vllm._deepselect_C` 缺失属于已知回退；如上游提供 SM75 构建可再评估。

---

## 7. 复现

```bash
# 构建（首次）
cd /home/jim/vLLM-2080Ti-Definitive
./build.sh

# 启动原模型
sudo systemctl start vllm          # :8000, 无需 key
# 启动无审查模型
sudo systemctl start vllm-uncensored  # :8002
./scripts/switch-model.sh status

# 基准（strata-ops 通用流式脚本）
python3 /srv/share/strata-ops/llm-bench.py <标签> http://127.0.0.1:8002 Qwen3.8-27B-Uncensored - 3 \
  '{"chat_template_kwargs":{"enable_thinking":false}}'
```

证据文件：
- 运行脚本：`/home/jim/scripts/vllm-run.sh`、`vllm-uncensored-run.sh`、`switch-model.sh`
- 单元：`/etc/systemd/system/vllm.service`、`vllm-uncensored.service`
- 构建：`build-logs/build-20261010-091133.log`、`toolchain-patches/cuda-13.0-glibc-2.41-rsqrt.patch`
- 运行：`run-logs/vllm-*.log`、`/home/jim/launch*.log`
- 基准：`/srv/share/strata-ops/BENCHMARK-vs-vLLM.zh-CN.md`、`logs/bench*`

---

## 8. 局限

- 单机、单用户、单请求为主；并发/批处理未系统覆盖。
- 本机真实提示词下的 decode（93.7）明显低于 fork 作者的合成输入口径（199–220），不能直接对比。
- DFlash2 vs MTP、NVLink 开/关均为单批次测量，存在 ±5–10% 波动。
- 质量评估只做了“可用性/图文理解”抽查，未做系统基准。
- 上游 fork 的 SM75 实现细节不在本复盘范围内，本文只记录本机部署与调优。

---

## 9. 2026-10-22 追加：博主同款 AWQ-INT4 复刻 + text-only A/B

### 9.1 背景

拿到一位博主（小红书「🇨🇳武文」）的公开配置：`cyankiwi/Qwen3.8-27B-AWQ-INT4` + MTP3 + 192K + 固定 8.5 GB FP16 KV + `max-num-seqs=64` + 纯文本。目标是**在本机复刻同一配置**，与博主公布数据做同口径对比。

### 9.2 复刻清单

| 项 | 博主 | 本机复刻 | 一致性 |
|---|---|---|---|
| 权重 | `cyankiwi/Qwen3.8-27B-AWQ-INT4`（21.04 GB） | 同（`/srv/share/models/`，5 分片字节校验一致） | ✅ |
| 量化 | compressed-tensors int4 pack-quantized / group 32 / mse | 同 | ✅ |
| 投机 | `{"method":"mtp","num_speculative_tokens":3}` | 同 | ✅ |
| 上下文 | 196608 | 196608 | ✅ |
| KV | `--kv-cache-memory-bytes 8500000000`（FP16） | 同 | ✅ |
| 并发/批 | `max-num-seqs=64` / `max-num-batched-tokens=1024` | 同 | ✅ |
| 工具 | `qwen3_coder` | 同 | ✅ |
| 形态 | `--language-model-only --skip-mm-profiling` | 同 | ✅ |
| 生成 | `--generation-config vllm` | 同 | ✅ |
| 环境 | `SM75_SPEC_SYNC_MODE=auto`、`ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=0`、`MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`、`expandable_segments:True` | 同 | ✅ |
| **实测 KV 容量** | **229,752 tokens** | **229,752 tokens** | ✅ 逐位一致 |
| **实测 MTP 接受率** | **~73%** | **73%（mean_accept_len 3.182 / accept_rate 0.7273）** | ✅ 一致 |

> 服务：`vllm-awq-int4.service`（端口 8003，与其它 27B 服务 `Conflicts=` 互斥）。

### 9.3 复刻中遇到的两个环境级阻塞（重要）

1. **custom all-reduce 在 CUDA 13.0 下崩溃**
   ```
   Failed: Cuda error .../csrc/custom_all_reduce.cuh:170 'invalid argument'
   → Worker died → EngineCore initialization failed
   ```
   - 试过限制 CUDA Graph 档位（[1..64]）无效；只要启用 custom all-reduce 就崩。
   - 结论：与博主 CUDA 12.8 的工具链差异导致，**只能 `--disable-custom-all-reduce`（NCCL 回退）**。
2. **CUDA Graph 被强制降级**
   ```
   CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for
   attention backend FlashInferBackend ... setting cudagraph_mode=PIECEWISE
   ```
   - 博主表里是 FULL_AND_PIECEWISE（51 档全图）；本机只能跑 PIECEWISE。
   - 这是 decode 每步开销变大的主要嫌疑。

### 9.4 同口径结果（`/srv/share/strata-ops/llm-bench.py`，真实提示词，3 次中位）

| 服务 · 模型 | decode | TTFT | 6K 提示读取 |
|---|---:|---:|---:|
| **vLLM · 无审查 + DFlash2（带视觉，当前长期）** | **93.74** | 133 ms | 5.08 s |
| vLLM · 无审查 + text-only + DFlash2（本次 A/B） | **94.66** | 130 ms | 5.06 s |
| vLLM · 无审查 + 自带 MTP | 69.85 | 141 ms | 5.17 s |
| vLLM · 基座 Qwen3.8-27B-NVFP4（:8000） | 75.39 | 92 ms | 4.66 s |
| **vLLM · AWQ-INT4 + MTP3（博主复刻）** | **37.17** | 85 ms | **4.17 s** |
| Strata · IQ2_XS（125B） | 84.32 | 78 ms | 6.61 s |
| Strata · Orca IQ3_XXS（125B） | 69.56 | 86 ms | 6.72 s |
| *博主公布（AWQ+MTP3）* | *~84* | *74 ms* | *—* |

### 9.5 与博主的差异归因

| 指标 | 博主 | 本机复刻 | 说明 |
|---|---:|---:|---|
| decode | ~84 | **37.17** | ❌ 只有 44% |
| MTP 接受率 | 73% | **73%** | ✅ 配置正确 |
| KV 容量 | 229,752 | **229,752** | ✅ 配置正确 |
| 6K prefill | — | **4.17 s** | 本机最快 |

**结论：配置复刻是准确的（接受率、KV 逐位一致），但博主公布的 decode 在本机复现不了。** 差异来自环境而非配置：CUDA 13.0 + NCCL 回退 + CUDA Graph 降级（PIECEWISE）+ 框架 v0.2.2 vs v0.1.17。

### 9.6 text-only A/B（NVFP4 + DFlash2）

| 版本 | decode | TTFT | 6K prefill | KV 容量 | DFlash2 接受率 |
|---|---:|---:|---:|---:|---:|
| A：带视觉 | 93.74 | 133 ms | 5.08 s | 348,979 | 48.6% |
| B：text-only（`--language-model-only --skip-mm-profiling`） | 94.66 | 130 ms | 5.06 s | **432,537** | 49.6% |
| 变化 | +1% | −2% | **−0.4%（无实质提升）** | **+24%** | ≈ |

- 日志确认视觉塔关闭：`All limits of multimodal modalities ... set to 0, running in text-only mode.`
- **text-only 并不提升速度**，只把 KV 容量提高约 24%；在 `max-num-seqs=1` 下这部分 KV 用不到，且会失去图像输入。
- 因此 **保留带视觉版本**（A），text-only 不作为默认。

### 9.7 结论与后续

1. **保持 NVFP4 + DFlash2（带视觉）** 作为本机最优：decode 93.74，256K，支持图像。
2. **不要换 AWQ-INT4**：本机 decode 只有 37.17（约为 NVFP4 版的 40%），只胜在 prefill（4.17 s）。
3. AWQ 复刻版 6K prefill 快于 NVFP4（4.17 vs 5.06 s），**原因不是 text-only**（已证伪），剩余嫌疑是 `max-num-batched-tokens=1024` 或 AWQ-INT4 的解量化路径。
4. 待验证的优化：把 target attention backend 从 FlashInfer 换成 `TRITON_ATTN`，看能否恢复 FULL_AND_PIECEWISE，从而提升 decode。
5. 若未来升级到 CUDA 12.8 工具链，可重新评估 custom all-reduce 与全图捕获。

> 证据：`/srv/share/strata-ops/logs/bench-awq-int4.txt`、`bench-textonly-nvfp4.txt`、`bench2-vllm-uncensored.txt`；服务脚本 `/home/jim/scripts/vllm-awq-int4-run.sh`、`vllm-uncensored-run.sh`；单元 `/etc/systemd/system/vllm-awq-int4.service`。


---

## 10. 2026-10-23 追加：vllm-turing 插件优化无审查版

> 参考：`https://github.com/cookjohn/vllm-turing`（克隆在 `/srv/share/vllm-turing`）。
> 对象：`Qwen3.8-27B-abliterated-NVFP4` + DFlash2/7，TP2，FP8 KV，262144 context，端口 8002。
> 约束：不换模型、不下载新权重；只在本机已有 vLLM 0.2.2 fork 上叠加兼容的插件与启动参数。

### 10.1 做了什么

1. 克隆并 editable 安装插件到现有 venv：
   ```bash
   git clone --depth 1 https://github.com/cookjohn/vllm-turing.git /srv/share/vllm-turing
   uv pip install --python /home/jim/vLLM-2080Ti-Definitive/.venv/bin/python -e /srv/share/vllm-turing
   ```
2. 本机 vLLM 缺少 `vllm.vllm_flash_attn.layers`，插件的 `vitlowmem` 会在此报错；在无审查启动脚本中设 `VLLM_TURING_VIT_LOWMEM=0`。
3. 采用插件与仓库建议的兼容项：
   - `--mamba-ssm-cache-dtype float16`
   - `--block-size 880`
   - `--max-num-batched-tokens 4096`（仓库原文 7040；本机长 prompt 会 OOM，降到 4096）
   - `--gpu-memory-utilization 0.95`（原 0.96，留出长 prompt 激活余量）
4. 插件只对无审查版启用；基座版与 AWQ 版脚本加 `VLLM_TURING_DISABLE=1`，避免影响已验证路线。

### 10.2 结果（同机、同模型、单请求）

| 指标 | baseline | 插件 | 插件+P1 | **最终 P4** | 最终 vs baseline |
|---|---:|---:|---:|---:|---:|
| decode 256 tok | 93.96 | 96.76 | 100.02 | **99.57** | **+6.0%** |
| 6K prefill | 5.05 s | 4.97 s | 5.01 s | **4.96 s** | **+1.8%** |
| 23.8K prefill | 1095.7 | 1156.6 | 1156.9 | **1173.5** | **+7.1%** |
| 89.2K prefill | 864.4 | 1031.0 | 1034.8 | **1057.5** | **+22.3%** |
| 89K decode | 81.0 | 84.3 | 87.4 | **87.7** | **+8.2%**（P2 曾 94.2，波动大） |
| KV 容量 | 348,979 | 419,430 | 461,830 | **443,357** | **+27.0%** |

- 插件主要收益：int8 embedding/LM head/共享 FlashInfer workspace 把 KV 从 349k 提到 419k；`fiattn8`/`fdec`/`gdn`/`customar` 让 89K prefill +19%。
- `SSM float16 + block 880` 再把 KV 提到 462k、decode +3%。
- `batch 7040` 长上下文 decode 略快，但长 prompt 触发 PyTorch OOM 警告（请求仍完成）；最终用 4096 + util 0.95，无 OOM。

### 10.3 验证

- 短答 3/3：巴黎、5、疑是地上霜。
- 针召回：22K 与 89K 各 1 次，均命中（`7391-KX` / `8642-QZ`）。
- 最终配置长请求 OOM = 0；空闲显存约 2.0 GiB/卡；KV 443,357 tokens（1.69× 262K）。

### 10.4 未完成与注意

- **W8A8 SmoothQuant 路线未做**：需要 BF16 原始权重 + 校准（约 55 GB 下载/转换），本机只有 NVFP4/INT4 权重；因此没有复现仓库 README 的 ~110 tok/s / 3550 prefill。
- 插件是按 vLLM 0.31 写的，本机是 0.2.2 fork；部分组件不兼容（`vitlowmem` 已关闭，`nosync` 因缺 FlashInfer patch 自动 off）。升级 vLLM 后需重新核对。
- `--enable-mamba-fine-grained-prefix-cache --prefix-match-unit 80` 未验证；当前 prefix cache 已能把 22K 重复提示的 TTFT 从 19.1 s 降到 1.14 s。
- 回滚脚本与卸载命令见 [`results/2026-10-23-vllm-turing/SUMMARY.md`](results/2026-10-23-vllm-turing/SUMMARY.md)。

> 证据：`results/2026-10-23-vllm-turing/`；脚本 `/home/jim/scripts/vllm-uncensored-run.sh`；备份 `backups/*.pre-turing-20261023-180714`。
