# Strata 接入指南（引擎组件 `ENGINE=strata`）

> 状态：结构适配完成（`engines/strata.sh` + `hosts/engine/strata.env` + `probe/engines.py`）。
> 上游事实与 ⭐ 真机核对项见 [`design/strata-engine/事实核对.md`](design/strata-engine/事实核对.md)
> 与 [`design/strata-engine/施工图.md`](design/strata-engine/施工图.md) §4；本文只讲**怎么用**。
> 契约背景：[`design/component-interface/设计.md`](design/component-interface/设计.md)。

---

## 1. 两种探测姿势的选型

| 姿势 | 适用 | 入口 |
|---|---|---|
| **外部只读探测**（主推） | Strata 常态：消费级 PC / Windows / 已在跑的服务（`strata launch --access lan`） | [`hosts/deploy/external-probe.sh`](../hosts/deploy/external-probe.sh) |
| **host 拉起** | Linux NVIDIA/AMD 机器、`strata` 命令可用、愿意让本工具管进程生命周期 | `hosts/vllm-host.sh up engine/strata` |

> docker 模式**不支持** Strata（上游无官方镜像承诺，`RUNTIME=docker` 会被启动链显式拒绝）。

---

## 2. 外部只读流程（主推）

### ① Strata 侧：起服务并拿到地址与 key

```bash
# 交互式向导（桌面 App / Terminal 均可）
strata launch --access lan        # 生成 API key；默认端口 8080
strata status                     # 确认已加载模型（/v1/models 只列已加载模型）
```

### ② 本工具侧：自检 + 全链路探测

```bash
# 环境自检（PROBE_TIMEOUT 建议放宽，见 §5 压测语义）
VLLM_BASE_URL=http://<pc>:8080 VLLM_API_KEY=<key> \
PROBE_OUTDIR=/tmp/strata-check PROBE_TIMEOUT=1800 \
  python3 -m probe env

# 全链路（env → metrics → http → suite → report）
ENGINE=strata PORT=8080 \
VLLM_BASE_URL=http://<pc>:8080 VLLM_API_KEY=<key> PROBE_TIMEOUT=1800 \
  bash hosts/deploy/external-probe.sh <strata 的 serve.log 或任意存在文件> engine/strata
```

固定参数建议：

| 参数 | 建议值 | 原因 |
|---|---|---|
| `ENGINE` | `strata` | 探针按能力位协商：缺 `/v1/completions`、缺逐 step 日志等都会显式降级/拒绝 |
| `PROBE_ENDPOINT` | `chat`（默认） | Strata **未提供** `/v1/completions`；设 completions 会被直接拒绝 |
| `PROBE_TIMEOUT` | `≥1800` | 长 prompt 首次全量读取约 1 分钟/30K tokens（前缀缓存后秒级） |
| `PROBE_MAX_TOKENS` | `≥256` | **思考计入 `max_tokens`** 且默认 high 档：`128` 可能全被思考吃掉 |
| `PROBE_CONCURRENCY` | `1` 起步 | Strata **默认单请求串行**；`--parallel N` 开多槽后再加并发 |
| `SERVED_NAME` | `strata` | 元数据端点对未知 model 名返回 404；统一用已加载模型名最稳 |

---

## 3. host 拉起流程

```bash
# 预检（strata 未装时 engine 预检按 warn 处置，不阻塞）
hosts/vllm-host.sh check engine/strata

# 先看将执行的命令（custom 模式：无 vLLM flag，只有入口 + EXTRA_ENGINE_ARGS）
hosts/vllm-host.sh dry-run engine/strata

# 启动（ENGINE_ENTRYPOINT="strata launch --yes --no-tui"）
hosts/vllm-host.sh up engine/strata
# 或显式指定厂商做遥测（SMI 面继续按 VENDOR 走）
hosts/vllm-host.sh up vendor/nvidia engine/strata

# 探测 / 报告 / 停止
hosts/vllm-host.sh probe  engine/strata
hosts/vllm-host.sh report engine/strata
hosts/vllm-host.sh down   engine/strata
```

要点：

* **模型/端口/上下文固化**：优先写进 Strata 的每模型配置 `strata-<model>.json`
  （`args` 数组 = 引擎 CLI 参数，`env` 块 = `STRATA_*` 环境变量）；端口也可用
  `HOST_SERVE_ENTRYPOINT` 覆盖入口传入（⭐ `strata launch --port` 支持待核）。
  ```bash
  HOST_SERVE_ENTRYPOINT="strata launch --model q2_0 --yes --no-tui --port 8100" \
    hosts/vllm-host.sh up engine/strata
  ```
* **追加引擎参数/环境**（模板是追加型键，显式导出会丢弃模板追加并打 ⚠️）：
  ```bash
  EXTRA_ENGINE_ARGS="--spec 4 --mtp --context 32768" \
  EXTRA_ENV="STRATA_IQ_MT_MIN=1" \
    hosts/vllm-host.sh up engine/strata
  ```
* **停止**：`hosts/vllm-host.sh down engine/strata` 按进程组 SIGTERM（serve.pid）；
  进程正则兜底用 `HOST_ENGINE_PATTERN`（模板已给 `strata|server\.py`，⭐ 真机核对后修正），
  另可用 `strata launch --stop` 兜底。
* `strata launch --plan` 可做启动前预演；`strata doctor` 排障（见 §6）。

---

## 4. 指标抓取说明

* Strata 的 `GET /metrics` **默认返回 JSON**（监控页数据）；带 `?format=prometheus`
  或 `Accept: text/plain` 才返回 Prometheus 文本。**探针已自动带后缀**
  （profile 字段 `ENGINE_METRICS_FETCH_SUFFIX="?format=prometheus"`），就绪轮询同样生效。
* 指标构成（详见[事实核对 §6](design/strata-engine/事实核对.md)）：
  * **vLLM 同名子集**：running/waiting、token 与 request 计数器、prefix_cache、
    spec_decode 计数器、KV 使用率、TTFT/ITL/e2e 直方图 —— 解析层直接复用；
  * **缺失的 vLLM 指标**：5 个生命周期直方图（queue/prefill/decode/inference/per-token）、
    `vllm:cache_config_info`、kv_block_*、preemptions —— 报告里按 `available=False` 行展示，
    属**预期缺项**，不是故障；
  * **`strata:` 专有指标**：`strata:live_*`、`strata:last_hit_rate`（专家缓存命中率）、
    `strata:gpu_*`/`strata:ram_*` 等 —— F3 的可选采集面（`--engine-extras`）。
* Prometheus/Grafana 直接抓：抓取配置加参数 `?format=prometheus`，或请求头
  `Accept: text/plain`。
* 就绪判定：`vllm:cache_config_info` 在 Strata 上不存在，就绪轮询已换
  `^vllm:num_requests_(running|waiting)`（⭐ 真机核对实际指标名）。

---

## 5. 压测语义差异（解释数字前必读）

| 差异 | 对探测数字的影响 | 建议 |
|---|---|---|
| 思考计入 `max_tokens`（默认 high 档） | 回答 token 被压缩，TPOT/e2e 口径与 vLLM 不同 | `PROBE_MAX_TOKENS≥256`；对照实验固定思考等级 |
| 默认单请求串行 | `CONCURRENCY>1` 的吞吐数字不反映并行能力 | 开 `--parallel N` 后再测并发；报告注记会提示 |
| 首 read 慢（~1 分钟/30K tokens） | suite 首场景 TTFT 偏大 | 套件自带 warmup 场景；冷数据先预热一次 |
| `reasoning_content` 流式增量 | TTFT 按 reasoning 首 token 计（与 vLLM 口径一致） | 无需改参数；报告会标注口径 |
| A/B 可复现组合 | 默认自适应行为引入抖动 | `EXTRA_ENV="STRATA_IQ_MT_MIN=1"` + `EXTRA_ENGINE_ARGS="--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0"`（decode 变慢，仅对比实验用） |

---

## 6. 排障

| 症状 | 工具/做法 |
|---|---|
| 起不来/行为异常 | `strata doctor`；`strata launch --plan` 预演 |
| 实时吞吐/状态 | `strata monitor --url http://127.0.0.1:8080`；`GET /status`、`GET /slots` |
| 元数据 | `GET /props`（chat template 能力、上下文上限、引擎版本 when available） |
| 端口占用 | 通常是已有实例（含桌面 App）还在跑；换 PORT 或先停旧实例 |
| 内存不足 | low-RAM 模式自动启用（更慢）；换更小量化包（见事实核对 §2/§3） |
| draft head 不 fit | `--spec N` 降 N 或换 `--draft-vocab` |
| 首读特别慢 | 首次全量读专家属正常（~1 分钟/30K tokens）；前缀缓存后续秒级 |
| 探针 completions 报错 | 预期行为：Strata 无 `/v1/completions`，用 `PROBE_ENDPOINT=chat` |
| hostcheck 警告 engine 版本 | `strata` 不在 PATH（App 安装常见）：warn 不阻塞；确认服务可达即可 |

---

## 7. 真机验证清单（逐项预期，硬件就绪后回填施工图 §4）

1. **http**：TTFT/ITL 出数；`reasoning_content` 增量被计入首 token；无 `chunk.metrics`
   （`server_side.available=False` 属预期）。
2. **metrics**：vLLM 同名子集出数；5 个生命周期直方图与 `cache_config_info` 行
   `available=False`；`?format=prometheus` 抓取命中（不带后缀时探针报错为预期防护）。
3. **suite**：8 场景可跑完（`PROBE_MAX_TOKENS≥256`）；schema 缺项为 null（compare 兼容）。
4. **report**：概览出现「引擎」行（strata）；§3 逐 step / §5 per-request / §7 kernel 出现
   能力降级注记；diagnose 不给 vLLM flag 建议（改给引擎注记）。
5. **compare**：vLLM 基线 vs strata 可并排，缺项 null 不报错。
6. ⭐ 待核对项（回填 `design/strata-engine/施工图.md` §4）：`--port` flag、进程正则、
   版本命令、就绪指标名、fetch suffix 必要性、spec 计数器归因支撑、`--parallel` 行为、
   App 安装时 `strata --version` 的表现。
